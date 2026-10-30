"""Huada port of 50M_Model/largeScale_fineTune.py.

The backbone, representation layer, masked-mean pooling, heads, optimizer
choices, and epoch loop follow upstream. Adaptations are limited to the shared
Huada Lance contract, explicit subject manifests, regression, richer metrics,
checkpoint auditing, smoke mode, evaluation mode, and sex-stratified subject CV4.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import lance
import numpy as np
import pyarrow
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from configs import Config
from dataset import collate_fn
from lance_dataset import (
    LanceEEGPretrainDataset,
    SEX_LABEL_MAPPING,
    SUPPORTED_TASKS,
    load_storage_options,
    load_subject_manifest,
)
from model import EEGPretrainModel
from prefetch_loader import PrefetchLoader
from sex_cv4 import (
    PROTOCOL as SEX_CV4_PROTOCOL,
    build_sex_stratified_subject_cv4,
    binary_classification_metrics,
    write_oof_artifacts,
    write_split_artifacts,
)
from utils import set_seed


UPSTREAM_URL = "https://github.com/ionaaaa/50M_Model"
UPSTREAM_BRANCH = "main"
UPSTREAM_COMMIT = "0f102b10ac885df759f1d7052a36669c7412c5e6"


def get_rank():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return int(os.environ.get("RANK", "0"))


def is_main_process():
    return get_rank() == 0


def safe_name(name):
    name = str(name)
    for char in [" ", "/", "\\", ":", "|", "(", ")", "[", "]"]:
        name = name.replace(char, "_")
    while "__" in name:
        name = name.replace("__", "_")
    return name.strip("_")


def get_git_state():
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        branch = subprocess.check_output(
            ["git", "branch", "--show-current"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
        ).strip())
        return {"commit": commit, "branch": branch, "dirty": dirty}
    except Exception:
        return {"commit": None, "branch": None, "dirty": None}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_snapshot(path):
    path = Path(path)
    if not path.exists() or not path.is_dir():
        return None
    files = []
    for item in sorted(path.rglob("*")):
        if item.is_file():
            stat = item.stat()
            files.append({
                "path": item.relative_to(path).as_posix(),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            })
    payload = json.dumps(files, sort_keys=True).encode("utf-8")
    return {"manifest_sha256": hashlib.sha256(payload).hexdigest(), "files": files}


def json_dump(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-lance", required=True)
    parser.add_argument("--task", required=True, choices=sorted(SUPPORTED_TASKS))
    parser.add_argument("--subject-split-manifest")
    parser.add_argument("--storage-options")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--downstream-checkpoint", type=Path)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--mode", choices=["linear_probe", "finetune"], default="linear_probe")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--protocol",
        choices=["legacy_single_split", SEX_CV4_PROTOCOL],
        default="legacy_single_split",
    )
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--pooling", choices=["mean", "flatten"], default=None)
    parser.add_argument("--head-lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-max-samples", type=int, default=4)
    parser.add_argument("--smoke-max-samples-per-subject", type=int, default=1)
    parser.add_argument("--allow-random-init-smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def apply_cli_overrides(cfg, args):
    cfg.data.lance_path = args.input_lance
    cfg.data.storage_options_path = args.storage_options
    cfg.finetune.task = args.task
    cfg.finetune.subject_split_manifest = args.subject_split_manifest or ""
    cfg.finetune.output_root = str(args.output_dir)
    cfg.finetune.pretrained_ckpt_path = "" if args.checkpoint is None else str(args.checkpoint)
    cfg.finetune.downstream_checkpoint = "" if args.downstream_checkpoint is None else str(args.downstream_checkpoint)
    cfg.finetune.mode = args.mode
    cfg.finetune.device = args.device
    cfg.finetune.num_epochs = args.epochs
    cfg.finetune.batch_size = args.batch_size
    cfg.finetune.num_workers = args.num_workers
    cfg.finetune.seed = args.seed
    cfg.finetune.split_seed = args.split_seed
    cfg.finetune.protocol = args.protocol
    cfg.finetune.folds = args.folds
    if args.pooling is not None:
        cfg.finetune.pooling = args.pooling
    if args.head_lr is not None:
        cfg.finetune.head_lr = args.head_lr
    if args.weight_decay is not None:
        cfg.finetune.weight_decay = args.weight_decay
    cfg.finetune.smoke = args.smoke
    cfg.finetune.smoke_max_samples = args.smoke_max_samples
    cfg.finetune.smoke_max_samples_per_subject = args.smoke_max_samples_per_subject
    cfg.finetune.allow_random_init_smoke = args.allow_random_init_smoke
    cfg.finetune.evaluate_only = args.evaluate
    cfg.finetune.num_classes = 2 if args.task == "sex_classification" else 1
    cfg.finetune.metric_for_best = "val_acc" if args.task == "sex_classification" else "val_loss"
    validate_resolved_hyperparameters(cfg)


def validate_resolved_hyperparameters(cfg):
    if cfg.finetune.pooling not in {"mean", "flatten"}:
        raise ValueError(f"Unsupported pooling: {cfg.finetune.pooling!r}")
    if cfg.finetune.head_lr <= 0:
        raise ValueError("head_lr must be > 0")
    if cfg.finetune.weight_decay < 0:
        raise ValueError("weight_decay must be >= 0")


def setup_experiment_dir(cfg, overwrite=False):
    run_dir = Path(cfg.finetune.output_root)
    if run_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing output directory: {run_dir}")
        if any(run_dir.iterdir()):
            raise FileExistsError("--overwrite does not delete non-empty directories; choose a new output")
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run_dir / "checkpoints"
    metrics_dir = run_dir / "metrics"
    checkpoint_dir.mkdir()
    metrics_dir.mkdir()
    cfg.finetune.run_dir = str(run_dir.resolve())
    cfg.finetune.checkpoint_dir = str(checkpoint_dir.resolve())
    cfg.finetune.metrics_dir = str(metrics_dir.resolve())
    cfg.finetune.save_path = str((checkpoint_dir / "downstream_finetune.pt").resolve())
    cfg.finetune.scaling_result_csv = str((metrics_dir / "summary.csv").resolve())
    return run_dir


class BatchedSubset(torch.utils.data.Dataset):
    """Upstream subset wrapper that preserves batched Lance reads."""

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = [int(index) for index in indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[int(idx)]]

    def __getitems__(self, indices):
        mapped = [self.indices[int(index)] for index in indices]
        return self.dataset.__getitems__(mapped)


class EEGDownstreamClassifier(nn.Module):
    """Upstream downstream model; ``num_classes=1`` is the regression adapter."""

    def __init__(
        self,
        backbone,
        d_model,
        num_classes,
        probe_layer_idx=None,
        freeze_backbone=False,
        use_mlp_head=False,
        mlp_hidden_dim=512,
        head_dropout=0.0,
        pooling="mean",
        num_tokens=None,
    ):
        super().__init__()
        self.backbone = backbone
        self.probe_layer_idx = probe_layer_idx
        self.freeze_backbone = freeze_backbone
        self.pooling = str(pooling)
        self.d_model = int(d_model)
        self.num_tokens = None if num_tokens is None else int(num_tokens)
        for parameter in self.backbone.parameters():
            parameter.requires_grad = not freeze_backbone
        if self.pooling == "mean":
            head_input_dim = self.d_model
        elif self.pooling == "flatten":
            if self.num_tokens is None or self.num_tokens <= 0:
                raise ValueError("flatten pooling requires a positive num_tokens")
            head_input_dim = self.num_tokens * self.d_model
        else:
            raise ValueError(f"Unsupported pooling: {self.pooling!r}")
        self.head_input_dim = int(head_input_dim)
        if use_mlp_head:
            self.head = nn.Sequential(
                nn.Linear(head_input_dim, mlp_hidden_dim), nn.GELU(),
                nn.Dropout(head_dropout), nn.Linear(mlp_hidden_dim, num_classes),
            )
        else:
            self.head = nn.Linear(head_input_dim, num_classes)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def pool_valid_tokens(self, token_embeddings, token_valid_mask):
        valid_mask = token_valid_mask.to(
            device=token_embeddings.device,
            dtype=token_embeddings.dtype,
        ).unsqueeze(-1)
        summed = (token_embeddings * valid_mask).sum(dim=1)
        denom = valid_mask.sum(dim=1).clamp_min(1.0)
        return summed / denom

    def flatten_valid_tokens(self, token_embeddings, token_valid_mask):
        valid_mask = token_valid_mask.to(
            device=token_embeddings.device,
            dtype=token_embeddings.dtype,
        ).unsqueeze(-1)
        return (token_embeddings * valid_mask).flatten(start_dim=1)

    def extract_features(self, token_embeddings, token_valid_mask):
        if self.pooling == "mean":
            return self.pool_valid_tokens(token_embeddings, token_valid_mask)
        if self.pooling == "flatten":
            return self.flatten_valid_tokens(token_embeddings, token_valid_mask)
        raise ValueError(f"Unsupported pooling: {self.pooling!r}")

    def forward(self, token_inputs, token_channel_indices, token_time_indices, token_valid_mask):
        kwargs = {
            "token_inputs": token_inputs,
            "token_channel_indices": token_channel_indices,
            "token_time_indices": token_time_indices,
            "return_layer_idx": self.probe_layer_idx,
        }
        if self.freeze_backbone:
            with torch.no_grad():
                token_embeddings = self.backbone.extract_token_embeddings(**kwargs)
        else:
            token_embeddings = self.backbone.extract_token_embeddings(**kwargs)
        features = self.extract_features(token_embeddings, token_valid_mask)
        return self.head(features)


def compute_shape_params(cfg):
    patch_len = int(round(cfg.data.patch_seconds * cfg.data.target_sfreq))
    patch_stride = int(round(cfg.data.patch_stride_seconds * cfg.data.target_sfreq))
    total_points = int(round(cfg.data.clip_seconds * cfg.data.target_sfreq))
    num_time_patches = int(np.floor((total_points - patch_len) / patch_stride) + 1)
    return {
        "patch_len": patch_len,
        "patch_stride": patch_stride,
        "total_points": total_points,
        "num_time_patches": num_time_patches,
        "num_tokens": cfg.data.n_channels * num_time_patches,
        "frames_per_patch": int(round(cfg.data.patch_seconds / cfg.data.target_frame_seconds)),
        "n_bands": len(cfg.data.band_defs),
    }


def resolved_head_dimensions(cfg, shape_params):
    if cfg.finetune.pooling == "mean":
        input_dim = int(cfg.model.d_model)
    elif cfg.finetune.pooling == "flatten":
        input_dim = int(shape_params["num_tokens"]) * int(cfg.model.d_model)
    else:
        raise ValueError(f"Unsupported pooling: {cfg.finetune.pooling!r}")
    if cfg.finetune.use_mlp_head:
        parameter_count = (
            input_dim * cfg.finetune.mlp_hidden_dim + cfg.finetune.mlp_hidden_dim
            + cfg.finetune.mlp_hidden_dim * cfg.finetune.num_classes
            + cfg.finetune.num_classes
        )
    else:
        parameter_count = input_dim * cfg.finetune.num_classes + cfg.finetune.num_classes
    return input_dim, int(parameter_count)


def move_batch_to_device(batch, device):
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) and value.device.type == "cpu" else value
        for key, value in batch.items()
    }


def get_labels_from_batch(batch, task="sex_classification"):
    labels = batch["label"]
    if task == "sex_classification":
        return labels.long().view(-1)
    return labels.float().view(-1)


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def compute_probe_layer_idx(depth, probe_layer_ratio):
    return max(0, min(int(round(depth * probe_layer_ratio)) - 1, depth - 1))


def load_downstream_dataset(cfg, split=None, max_samples=None):
    return LanceEEGPretrainDataset(
        lance_path=cfg.data.lance_path,
        storage_options=load_storage_options(cfg.data.storage_options_path),
        task=cfg.finetune.task,
        subject_split_manifest=cfg.finetune.subject_split_manifest or None,
        split=split,
        max_samples=max_samples,
    )


def validate_subject_split_manifest(path, task):
    rows = load_subject_manifest(path)
    split_subjects = {"train": set(), "validation": set(), "test": set()}
    excluded = []
    for subject_id, row in rows.items():
        if row.get("task") and row["task"] != task:
            raise ValueError(f"Manifest task mismatch for {subject_id}")
        reason = str(row.get("exclusion_reason", "")).strip()
        if reason:
            excluded.append({"subject_id": subject_id, "reason": reason})
            continue
        split = str(row.get("split", "")).strip()
        if split not in split_subjects:
            raise ValueError(f"Invalid split {split!r} for {subject_id}")
        split_subjects[split].add(subject_id)
    names = list(split_subjects)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            overlap = split_subjects[left].intersection(split_subjects[right])
            if overlap:
                raise ValueError(f"Subject leakage between {left}/{right}: {sorted(overlap)}")
    if any(not subjects for subjects in split_subjects.values()):
        raise ValueError(f"Full run requires non-empty train/validation/test subjects: {split_subjects}")
    return {key: sorted(value) for key, value in split_subjects.items()}, excluded


def build_train_val_test_split(dataset, cfg):
    """Replacement for upstream window slicing: select by explicit subject split."""
    del dataset
    if not cfg.finetune.subject_split_manifest:
        raise ValueError("Full runs require --subject-split-manifest")
    subject_lists, excluded = validate_subject_split_manifest(
        cfg.finetune.subject_split_manifest, cfg.finetune.task
    )
    train_set = load_downstream_dataset(cfg, split="train")
    val_set = load_downstream_dataset(cfg, split="validation")
    test_set = load_downstream_dataset(cfg, split="test")
    if min(len(train_set), len(val_set), len(test_set)) < 1:
        raise ValueError("Subject manifest produced an empty window split")
    return train_set, val_set, test_set, subject_lists, excluded


def _loader(dataset, cfg, device, shuffle):
    base = DataLoader(
        dataset,
        batch_size=min(cfg.finetune.batch_size, max(len(dataset), 1)),
        shuffle=shuffle,
        num_workers=cfg.finetune.num_workers,
        pin_memory=str(device).startswith("cuda"),
        collate_fn=collate_fn,
        drop_last=False,
    )
    if cfg.finetune.use_prefetch_loader:
        return PrefetchLoader(base, device=device, prefetch=cfg.finetune.prefetch)
    return base


def build_loaders(train_set, val_set, test_set, cfg, device):
    return (
        _loader(train_set, cfg, device, True),
        _loader(val_set, cfg, device, False),
        _loader(test_set, cfg, device, False),
    )


def strip_module_prefix(state_dict):
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {key.replace("module.", "", 1): value for key, value in state_dict.items()}


def _checkpoint_state_dict(checkpoint):
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def _audit_backbone_state_dict(backbone, state_dict):
    target = backbone.state_dict()
    state_dict = strip_module_prefix(state_dict)
    matched = {}
    unexpected = []
    shape_mismatch = []
    for key, value in state_dict.items():
        if key not in target:
            unexpected.append(key)
        elif tuple(value.shape) != tuple(target[key].shape):
            shape_mismatch.append({
                "key": key,
                "checkpoint_shape": list(value.shape),
                "model_shape": list(target[key].shape),
            })
        else:
            matched[key] = value
    missing = sorted(set(target).difference(matched))
    matched_parameter_count = sum(int(target[key].numel()) for key in matched)
    total_parameter_count = sum(int(value.numel()) for value in target.values())
    fraction = matched_parameter_count / max(total_parameter_count, 1)
    return matched, {
        "matched_key_count": len(matched),
        "missing_keys": missing,
        "unexpected_keys": sorted(unexpected),
        "shape_mismatch_keys": shape_mismatch,
        "matched_parameter_count": matched_parameter_count,
        "total_parameter_count": total_parameter_count,
        "matched_parameter_fraction": float(fraction),
    }


def build_backbone(cfg, shape_params, device, load_pretrained=True, minimum_match_fraction=0.99):
    backbone = EEGPretrainModel(
        input_dim=shape_params["patch_len"], d_model=cfg.model.d_model,
        n_heads=cfg.model.n_heads, depth=cfg.model.depth,
        mlp_ratio=cfg.model.mlp_ratio, dropout=cfg.model.dropout,
        n_channels=cfg.data.n_channels,
        n_time_patches=shape_params["num_time_patches"],
        n_bands=shape_params["n_bands"],
        frames_per_patch=shape_params["frames_per_patch"],
    ).to(device)
    report = {
        "checkpoint_path": None, "checkpoint_sha256": None, "strict": False,
        "matched_key_count": 0, "missing_keys": [], "unexpected_keys": [],
        "shape_mismatch_keys": [], "matched_parameter_fraction": 0.0,
        "random_init": True,
    }
    if not load_pretrained:
        return backbone, report
    ckpt_path = cfg.finetune.pretrained_ckpt_path
    if not ckpt_path:
        if cfg.finetune.smoke and cfg.finetune.allow_random_init_smoke:
            return backbone, report
        raise ValueError("A pretrained --checkpoint is required (random init is smoke-only)")
    if not Path(ckpt_path).is_file():
        raise FileNotFoundError(ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    matched, audit = _audit_backbone_state_dict(backbone, _checkpoint_state_dict(checkpoint))
    report = {
        "checkpoint_path": str(Path(ckpt_path).resolve()),
        "checkpoint_sha256": sha256_file(ckpt_path),
        "strict": False,
        **audit,
        "random_init": False,
    }
    if report["matched_parameter_fraction"] < float(minimum_match_fraction):
        raise RuntimeError(
            "Pretrained backbone checkpoint matched only "
            f"{report['matched_parameter_fraction']:.6%}; required >= {minimum_match_fraction:.2%}. "
            f"Audit: {report}"
        )
    incompatible = backbone.load_state_dict(matched, strict=False)
    if sorted(incompatible.missing_keys) != sorted(report["missing_keys"]):
        raise RuntimeError("Internal checkpoint audit/load mismatch")
    return backbone, report


def build_downstream_model(cfg, backbone, device, shape_params):
    freeze_backbone = cfg.finetune.mode == "linear_probe"
    probe_layer_idx = compute_probe_layer_idx(cfg.model.depth, cfg.finetune.probe_layer_ratio)
    model = EEGDownstreamClassifier(
        backbone=backbone,
        d_model=cfg.model.d_model,
        num_classes=cfg.finetune.num_classes,
        probe_layer_idx=probe_layer_idx,
        freeze_backbone=freeze_backbone,
        use_mlp_head=cfg.finetune.use_mlp_head,
        mlp_hidden_dim=cfg.finetune.mlp_hidden_dim,
        head_dropout=cfg.finetune.head_dropout,
        pooling=cfg.finetune.pooling,
        num_tokens=shape_params["num_tokens"],
    ).to(device)
    return model, probe_layer_idx


def head_parameter_count(model):
    return sum(int(parameter.numel()) for parameter in model.head.parameters())


def model_training_metadata(cfg, model, shape_params, optimizer_report=None):
    return {
        "pooling": model.pooling,
        "head_lr": float(cfg.finetune.head_lr),
        "weight_decay": float(cfg.finetune.weight_decay),
        "optimizer_name": None if optimizer_report is None else optimizer_report["optimizer_name"],
        "optimizer_param_groups": [] if optimizer_report is None else optimizer_report["param_groups"],
        "head_input_dim": int(model.head_input_dim),
        "head_parameter_count": head_parameter_count(model),
        "num_tokens": int(shape_params["num_tokens"]),
        "d_model": int(cfg.model.d_model),
    }


def _checkpoint_head_input_dim(state_dict):
    candidates = [
        value for key, value in state_dict.items()
        if key.endswith("head.weight") or key.endswith("head.0.weight")
    ]
    if len(candidates) != 1 or candidates[0].ndim != 2:
        raise RuntimeError("Cannot uniquely determine downstream head input dimension")
    return int(candidates[0].shape[1])


def _infer_checkpoint_pooling(model, head_input_dim):
    candidates = []
    if head_input_dim == model.d_model:
        candidates.append("mean")
    if model.num_tokens is not None and head_input_dim == model.num_tokens * model.d_model:
        candidates.append("flatten")
    if len(candidates) != 1:
        raise RuntimeError(
            "Cannot uniquely infer checkpoint pooling from head input dimension "
            f"{head_input_dim}; expected mean={model.d_model}, "
            f"flatten={model.num_tokens}*{model.d_model}"
        )
    return candidates[0]


def load_downstream_checkpoint_strict(model, path, device):
    """Load a downstream checkpoint with explicit/inferred pooling compatibility checks."""
    payload = torch.load(path, map_location=device, weights_only=False)
    if "model_state_dict" not in payload:
        raise RuntimeError("Downstream checkpoint has no model_state_dict")
    state_dict = strip_module_prefix(payload["model_state_dict"])
    actual_head_input_dim = _checkpoint_head_input_dim(state_dict)
    checkpoint_pooling = payload.get("pooling")
    pooling_inferred = checkpoint_pooling is None
    if pooling_inferred:
        checkpoint_pooling = _infer_checkpoint_pooling(model, actual_head_input_dim)
        print(
            "Legacy downstream checkpoint has no pooling metadata; inferred "
            f"pooling={checkpoint_pooling!r} from head_input_dim={actual_head_input_dim}."
        )
    elif checkpoint_pooling not in {"mean", "flatten"}:
        raise RuntimeError(f"Invalid checkpoint pooling metadata: {checkpoint_pooling!r}")
    metadata_head_input_dim = payload.get("head_input_dim")
    if metadata_head_input_dim is not None and int(metadata_head_input_dim) != actual_head_input_dim:
        raise RuntimeError(
            "Checkpoint head_input_dim metadata does not match head weight shape: "
            f"{metadata_head_input_dim} != {actual_head_input_dim}"
        )
    if checkpoint_pooling != model.pooling:
        raise RuntimeError(
            f"Checkpoint pooling {checkpoint_pooling!r} does not match requested pooling {model.pooling!r}"
        )
    if actual_head_input_dim != model.head_input_dim:
        raise RuntimeError(
            f"Checkpoint head_input_dim {actual_head_input_dim} does not match model {model.head_input_dim}"
        )
    target = model.state_dict()
    mismatches = []
    for key in sorted(set(state_dict).intersection(target)):
        if tuple(state_dict[key].shape) != tuple(target[key].shape):
            mismatches.append((key, tuple(state_dict[key].shape), tuple(target[key].shape)))
    if mismatches:
        detail = ", ".join(f"{key}: {old} != {new}" for key, old, new in mismatches)
        raise RuntimeError("Checkpoint shape mismatch: " + detail)
    model.load_state_dict(state_dict, strict=True)
    payload["_load_report"] = {
        "pooling": checkpoint_pooling,
        "pooling_inferred": pooling_inferred,
        "head_input_dim": actual_head_input_dim,
    }
    return payload


def make_optimizer(cfg, model):
    if cfg.finetune.mode == "finetune":
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": model.backbone.parameters(), "name": "backbone",
                    "lr": cfg.finetune.backbone_lr, "weight_decay": cfg.finetune.weight_decay,
                },
                {
                    "params": model.head.parameters(), "name": "head",
                    "lr": cfg.finetune.head_lr, "weight_decay": cfg.finetune.weight_decay,
                },
            ],
            weight_decay=cfg.finetune.weight_decay,
        )
    else:
        optimizer = torch.optim.SGD(
            [{
                "params": model.head.parameters(), "name": "head",
                "lr": cfg.finetune.head_lr, "weight_decay": cfg.finetune.weight_decay,
            }],
            lr=cfg.finetune.head_lr,
            weight_decay=cfg.finetune.weight_decay,
        )
    validate_optimizer_configuration(cfg, optimizer)
    return optimizer


def optimizer_configuration(optimizer):
    return {
        "optimizer_name": optimizer.__class__.__name__,
        "param_groups": [
            {
                "param_group_name": str(group.get("name", f"group_{index}")),
                "learning_rate": float(group["lr"]),
                "weight_decay": float(group["weight_decay"]),
                "parameter_count": sum(int(parameter.numel()) for parameter in group["params"]),
            }
            for index, group in enumerate(optimizer.param_groups)
        ],
    }


def validate_optimizer_configuration(cfg, optimizer):
    report = optimizer_configuration(optimizer)
    groups = {group["param_group_name"]: group for group in report["param_groups"]}
    if "head" not in groups:
        raise RuntimeError("Optimizer has no named head parameter group")
    head = groups["head"]
    if not math.isclose(head["learning_rate"], float(cfg.finetune.head_lr)):
        raise RuntimeError("Optimizer head learning rate does not match resolved head_lr")
    if not math.isclose(head["weight_decay"], float(cfg.finetune.weight_decay)):
        raise RuntimeError("Optimizer head weight decay does not match resolved weight_decay")
    if cfg.finetune.mode == "finetune":
        if "backbone" not in groups:
            raise RuntimeError("Finetune optimizer has no named backbone parameter group")
        if not math.isclose(groups["backbone"]["learning_rate"], float(cfg.finetune.backbone_lr)):
            raise RuntimeError("Optimizer backbone learning rate does not match backbone_lr")
    return report


def make_criterion(task):
    return nn.CrossEntropyLoss() if task == "sex_classification" else nn.MSELoss()


def compute_metrics(task, targets, predictions, average_loss, smoke=False):
    targets = np.asarray(targets)
    predictions = np.asarray(predictions)
    result = {"loss": float(average_loss), "num_samples": int(targets.size)}
    if task == "sex_classification":
        targets = targets.astype(np.int64)
        predictions = predictions.astype(np.int64)
        confusion = np.zeros((2, 2), dtype=np.int64)
        for truth, prediction in zip(targets, predictions):
            confusion[truth, prediction] += 1
        result["accuracy"] = float((targets == predictions).mean()) if targets.size else None
        observed = sorted(set(targets.tolist()))
        if len(observed) < 2:
            balanced = macro_f1 = None
        else:
            recalls = []
            f1s = []
            for label in range(2):
                tp = confusion[label, label]
                fn = confusion[label].sum() - tp
                fp = confusion[:, label].sum() - tp
                recalls.append(tp / (tp + fn) if tp + fn else 0.0)
                precision = tp / (tp + fp) if tp + fp else 0.0
                recall = recalls[-1]
                f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
            balanced = float(np.mean(recalls))
            macro_f1 = float(np.mean(f1s))
        result.update({
            "balanced_accuracy": balanced,
            "macro_f1": macro_f1,
            "confusion_matrix": confusion.tolist(),
            "metrics_not_statistically_meaningful": bool(smoke or len(observed) < 2),
        })
    else:
        targets = targets.astype(np.float64)
        predictions = predictions.astype(np.float64)
        errors = predictions - targets
        mae = float(np.mean(np.abs(errors))) if targets.size else None
        rmse = float(np.sqrt(np.mean(errors ** 2))) if targets.size else None
        denominator = float(np.sum((targets - targets.mean()) ** 2)) if targets.size else 0.0
        r2 = None if targets.size < 2 or denominator == 0.0 else float(1.0 - np.sum(errors ** 2) / denominator)
        result.update({
            "mae": mae, "rmse": rmse, "r2": r2,
            "r2_reliable": bool(targets.size >= 2 and denominator > 0.0 and not smoke),
            "metrics_not_statistically_meaningful": bool(smoke or targets.size < 2),
        })
    return result


def _forward_outputs(model, batch, task):
    logits = model(
        token_inputs=batch["token_inputs"],
        token_channel_indices=batch["token_channel_indices"],
        token_time_indices=batch["token_time_indices"],
        token_valid_mask=batch["token_valid_mask"],
    )
    return logits if task == "sex_classification" else logits.view(-1)


def run_epoch(model, loader, criterion, device, task, optimizer=None, smoke=False):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_count = 0
    targets_out = []
    predictions_out = []
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        labels = get_labels_from_batch(batch, task)
        with torch.set_grad_enabled(is_train):
            outputs = _forward_outputs(model, batch, task)
            loss = criterion(outputs, labels)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        batch_size = int(labels.numel())
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size
        targets_out.extend(labels.detach().cpu().tolist())
        if task == "sex_classification":
            predictions_out.extend(outputs.argmax(dim=-1).detach().cpu().tolist())
        else:
            predictions_out.extend(outputs.detach().cpu().tolist())
        if smoke:
            break
    return compute_metrics(
        task, targets_out, predictions_out, total_loss / max(total_count, 1), smoke=smoke
    )


def train_one_epoch(model, loader, optimizer, criterion, device, task="sex_classification"):
    return run_epoch(model, loader, criterion, device, task, optimizer=optimizer)


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device, task="sex_classification", smoke=False):
    return run_epoch(model, loader, criterion, device, task, optimizer=None, smoke=smoke)


@torch.no_grad()
def evaluate_sex_test_once(model, loader, criterion, device, fold, checkpoint_path):
    """Evaluate a selected fold checkpoint once and retain only test-window OOF rows."""
    model.eval()
    total_loss = 0.0
    total_count = 0
    rows = []
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        labels = get_labels_from_batch(batch, "sex_classification")
        logits = _forward_outputs(model, batch, "sex_classification")
        probabilities = torch.softmax(logits, dim=-1)
        loss = criterion(logits, labels)
        predictions = probabilities.argmax(dim=-1)
        count = int(labels.numel())
        total_loss += float(loss.item()) * count
        total_count += count
        for index in range(count):
            label = int(labels[index].item())
            prediction = int(predictions[index].item())
            rows.append({
                "fold": int(fold),
                "sample_id": str(batch["sample_id"][index]),
                "subject_id": str(batch["subject_id"][index]),
                "sex": "female" if label == 0 else "male",
                "true_label": label,
                "prob_female": float(probabilities[index, 0].item()),
                "prob_male": float(probabilities[index, 1].item()),
                "predicted_label": prediction,
                "predicted_sex": "female" if prediction == 0 else "male",
                "num_valid_tokens": int(batch["token_valid_mask"][index].sum().item()),
                "checkpoint_path": str(Path(checkpoint_path).resolve()),
            })
    if not rows:
        raise RuntimeError(f"Fold {fold} test loader produced no rows")
    metrics = binary_classification_metrics(
        [row["true_label"] for row in rows],
        [row["predicted_label"] for row in rows],
    )
    metrics.update({"loss": total_loss / max(total_count, 1), "num_samples": total_count})
    return metrics, rows


def save_downstream_checkpoint(
    path, model, cfg, shape_params, probe_layer_idx, epoch, metrics,
    checkpoint_report, optimizer_report,
):
    training_metadata = model_training_metadata(cfg, model, shape_params, optimizer_report)
    payload = {
        "model_state_dict": model.state_dict(),
        "head_state_dict": model.head.state_dict(),
        "backbone_state_dict": model.backbone.state_dict(),
        "pretrained_ckpt_path": cfg.finetune.pretrained_ckpt_path or None,
        "pretrained_checkpoint_report": checkpoint_report,
        "probe_layer_idx": probe_layer_idx,
        "probe_layer_ratio": cfg.finetune.probe_layer_ratio,
        "mode": cfg.finetune.mode,
        "task": cfg.finetune.task,
        "label_mapping": SEX_LABEL_MAPPING if cfg.finetune.task == "sex_classification" else None,
        "config": asdict(cfg),
        "shape_params": shape_params,
        "epoch": epoch,
        "metrics": metrics,
        "pooling": training_metadata["pooling"],
        "head_input_dim": training_metadata["head_input_dim"],
        "head_parameter_count": training_metadata["head_parameter_count"],
        "head_lr": training_metadata["head_lr"],
        "weight_decay": training_metadata["weight_decay"],
        "optimizer_name": training_metadata["optimizer_name"],
        "optimizer_param_groups": training_metadata["optimizer_param_groups"],
        "upstream": {"url": UPSTREAM_URL, "branch": UPSTREAM_BRANCH, "commit": UPSTREAM_COMMIT},
    }
    temporary = str(path) + ".tmp"
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return payload


def _best_value(task, metrics):
    if task == "sex_classification":
        value = metrics.get("accuracy")
        return -float("inf") if value is None else float(value)
    return -float(metrics["loss"])


def run_one_checkpoint(
    cfg,
    shape_params,
    train_loader,
    val_loader,
    test_loader,
    device,
    checkpoint_report=None,
    test_prediction_context=None,
):
    backbone, loaded_report = build_backbone(cfg, shape_params, device, load_pretrained=not cfg.finetune.evaluate_only)
    checkpoint_report = checkpoint_report or loaded_report
    model, probe_layer_idx = build_downstream_model(cfg, backbone, device, shape_params)
    criterion = make_criterion(cfg.finetune.task)

    if cfg.finetune.evaluate_only:
        path = cfg.finetune.downstream_checkpoint
        if not path or not Path(path).is_file():
            raise FileNotFoundError(path or "--downstream-checkpoint is required for --evaluate")
        load_downstream_checkpoint_strict(model, path, device)
        test_metrics = validate_one_epoch(model, test_loader, criterion, device, cfg.finetune.task)
        return model, {"test": test_metrics}, probe_layer_idx, checkpoint_report, []

    optimizer = make_optimizer(cfg, model)
    optimizer_report = optimizer_configuration(optimizer)
    training_metadata = model_training_metadata(cfg, model, shape_params, optimizer_report)
    print(json.dumps({"resolved_downstream": training_metadata}, ensure_ascii=False))
    best_score = -float("inf")
    best_payload = None
    epoch_rows = []
    for epoch in range(cfg.finetune.num_epochs):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, cfg.finetune.task)
        val_metrics = validate_one_epoch(model, val_loader, criterion, device, cfg.finetune.task)
        epoch_metrics = {"epoch": epoch + 1, "train": train_metrics, "validation": val_metrics}
        epoch_rows.append(epoch_metrics)
        score = _best_value(cfg.finetune.task, val_metrics)
        if score > best_score:
            best_score = score
            best_payload = save_downstream_checkpoint(
                cfg.finetune.save_path, model, cfg, shape_params, probe_layer_idx,
                epoch + 1, epoch_metrics, checkpoint_report, optimizer_report,
            )
    if best_payload is None:
        raise RuntimeError("No best checkpoint was produced")
    load_downstream_checkpoint_strict(model, cfg.finetune.save_path, device)
    if test_prediction_context is not None:
        test_metrics, test_predictions = evaluate_sex_test_once(
            model,
            test_loader,
            criterion,
            device,
            test_prediction_context["fold"],
            cfg.finetune.save_path,
        )
    else:
        test_metrics = validate_one_epoch(model, test_loader, criterion, device, cfg.finetune.task)
        test_predictions = []
    best_payload["metrics"]["test"] = test_metrics
    temporary = cfg.finetune.save_path + ".tmp"
    torch.save(best_payload, temporary)
    os.replace(temporary, cfg.finetune.save_path)
    return model, {
        "epochs": epoch_rows, "test": test_metrics,
        "training_metadata": training_metadata,
    }, probe_layer_idx, checkpoint_report, test_predictions


def run_smoke(cfg, shape_params, device):
    dataset = load_downstream_dataset(cfg, max_samples=cfg.finetune.smoke_max_samples)
    if not len(dataset):
        raise ValueError("No eligible smoke samples")
    loader = _loader(dataset, cfg, device, shuffle=False)
    backbone, checkpoint_report = build_backbone(cfg, shape_params, device)
    model, probe_layer_idx = build_downstream_model(cfg, backbone, device, shape_params)
    criterion = make_criterion(cfg.finetune.task)
    optimizer = make_optimizer(cfg, model)
    optimizer_report = optimizer_configuration(optimizer)
    training_metadata = model_training_metadata(cfg, model, shape_params, optimizer_report)
    print(json.dumps({"resolved_downstream": training_metadata}, ensure_ascii=False))
    metrics = run_epoch(
        model, loader, criterion, device, cfg.finetune.task, optimizer=optimizer, smoke=True
    )
    subjects = sorted({row["subject_id"] for row in dataset.get_subject_label_rows()})
    checkpoint_metrics = {"epoch": 1, "train": metrics, "smoke": True}
    save_downstream_checkpoint(
        cfg.finetune.save_path, model, cfg, shape_params, probe_layer_idx,
        1, checkpoint_metrics, checkpoint_report, optimizer_report,
    )
    return {
        "metrics": checkpoint_metrics,
        "sample_count": len(dataset),
        "subjects": subjects,
        "split_subjects": {"train": subjects, "validation": [], "test": []},
        "excluded": ([{"subject_id": "SUB5", "reason": "age_missing"}]
                      if cfg.finetune.task == "age_regression" else []),
        "training_metadata": training_metadata,
    }, checkpoint_report


def _limit_rows_per_subject(rows, maximum):
    if maximum is None:
        return list(rows)
    counts = {}
    selected = []
    for row in rows:
        subject = row["subject_id"]
        if counts.get(subject, 0) >= int(maximum):
            continue
        counts[subject] = counts.get(subject, 0) + 1
        selected.append(row)
    return selected


def run_sex_cv4(cfg, shape_params, device, run_dir):
    """Run four isolated folds and create exact test-only OOF artifacts."""
    if cfg.finetune.task != "sex_classification":
        raise ValueError(f"{SEX_CV4_PROTOCOL} is only valid for sex_classification")
    if cfg.finetune.folds != 4:
        raise ValueError(f"{SEX_CV4_PROTOCOL} requires --folds 4")
    if cfg.finetune.evaluate_only:
        raise ValueError("CV training selects one validation checkpoint per fold; --evaluate is unsupported")
    if not cfg.finetune.pretrained_ckpt_path:
        raise ValueError("CV requires the real pretrained 50M --checkpoint; random initialization is forbidden")

    dataset = load_downstream_dataset(cfg)
    metadata_rows = dataset.get_subject_label_rows()
    subject_labels = {}
    for row in metadata_rows:
        subject = row["subject_id"]
        label = int(row["label"])
        if subject in subject_labels and subject_labels[subject] != label:
            raise ValueError(f"Inconsistent sex labels for subject {subject}")
        subject_labels[subject] = label
    split_payload = build_sex_stratified_subject_cv4(subject_labels, cfg.finetune.split_seed)
    write_split_artifacts(run_dir, split_payload)

    selected_rows = _limit_rows_per_subject(
        metadata_rows,
        cfg.finetune.smoke_max_samples_per_subject if cfg.finetune.smoke else None,
    )
    eligible_samples = {
        row["sample_id"]: {"subject_id": row["subject_id"], "label": int(row["label"])}
        for row in selected_rows
    }
    if len(eligible_samples) != len(selected_rows):
        raise ValueError("sample_id must be unique in the eligible CV input")

    oof_rows = []
    per_fold_rows = []
    checkpoint_reports = {}
    subject_test_folds = {}
    fold_results = []
    head_input_dim, expected_head_parameter_count = resolved_head_dimensions(cfg, shape_params)

    for fold_info in split_payload["folds"]:
        fold = int(fold_info["fold"])
        fold_seed = int(cfg.finetune.seed) + fold
        set_seed(fold_seed)
        cfg_fold = copy.deepcopy(cfg)
        cfg_fold.finetune.seed = fold_seed
        fold_dir = Path(run_dir) / f"fold_{fold}"
        checkpoint_dir = fold_dir / "checkpoints"
        metrics_dir = fold_dir / "metrics"
        checkpoint_dir.mkdir(parents=True, exist_ok=False)
        metrics_dir.mkdir(parents=True, exist_ok=False)
        cfg_fold.finetune.run_dir = str(fold_dir.resolve())
        cfg_fold.finetune.checkpoint_dir = str(checkpoint_dir.resolve())
        cfg_fold.finetune.metrics_dir = str(metrics_dir.resolve())
        cfg_fold.finetune.save_path = str((checkpoint_dir / "downstream_finetune.pt").resolve())

        split_subjects = {
            "train": set(fold_info["train_subjects"]),
            "validation": set(fold_info["validation_subjects"]),
            "test": set(fold_info["test_subjects"]),
        }
        indices_by_split = {
            split_name: [row["index"] for row in selected_rows if row["subject_id"] in subjects]
            for split_name, subjects in split_subjects.items()
        }
        if any(not values for values in indices_by_split.values()):
            raise ValueError(f"Fold {fold} has an empty window split")
        loaders = build_loaders(
            BatchedSubset(dataset, indices_by_split["train"]),
            BatchedSubset(dataset, indices_by_split["validation"]),
            BatchedSubset(dataset, indices_by_split["test"]),
            cfg_fold,
            device,
        )
        model, metrics, _, checkpoint_report, predictions = run_one_checkpoint(
            cfg_fold,
            shape_params,
            *loaders,
            device,
            test_prediction_context={"fold": fold},
        )
        del model
        downstream_sha256 = sha256_file(cfg_fold.finetune.save_path)
        checkpoint_reports[str(fold)] = checkpoint_report
        checkpoint_reports[str(fold)]["fold_seed"] = fold_seed
        checkpoint_reports[str(fold)]["selected_downstream_checkpoint_path"] = cfg_fold.finetune.save_path
        checkpoint_reports[str(fold)]["selected_downstream_checkpoint_sha256"] = downstream_sha256
        oof_rows.extend(predictions)
        for subject in fold_info["test_subjects"]:
            subject_test_folds[subject] = fold
        test_metrics = metrics["test"]
        training_metadata = metrics["training_metadata"]
        if training_metadata["head_input_dim"] != head_input_dim:
            raise RuntimeError("Fold head_input_dim does not match resolved pooling")
        if training_metadata["head_parameter_count"] != expected_head_parameter_count:
            raise RuntimeError("Fold head parameter count does not match resolved configuration")
        per_fold_rows.append({
            "fold": fold,
            "fold_seed": fold_seed,
            "num_windows": len(predictions),
            "num_subjects": 2,
            "loss": test_metrics["loss"],
            "accuracy": test_metrics["accuracy"],
            "balanced_accuracy": test_metrics["balanced_accuracy"],
            "macro_f1": test_metrics["macro_f1"],
            "confusion_matrix": json.dumps(test_metrics["confusion_matrix"]),
            "checkpoint_path": cfg_fold.finetune.save_path,
            "checkpoint_sha256": downstream_sha256,
            "matched_key_count": checkpoint_report["matched_key_count"],
            "matched_parameter_fraction": checkpoint_report["matched_parameter_fraction"],
            "pooling": training_metadata["pooling"],
            "head_lr": training_metadata["head_lr"],
            "weight_decay": training_metadata["weight_decay"],
            "optimizer_name": training_metadata["optimizer_name"],
            "head_input_dim": training_metadata["head_input_dim"],
            "head_parameter_count": training_metadata["head_parameter_count"],
        })
        fold_result = {
            "fold": fold,
            "fold_seed": fold_seed,
            "split_subjects": {key: sorted(value) for key, value in split_subjects.items()},
            "metrics": metrics,
            "checkpoint_report": checkpoint_report,
            "training_metadata": training_metadata,
        }
        json_dump(metrics_dir / "fold_result.json", fold_result)
        fold_results.append(fold_result)

    if set(subject_test_folds) != set(subject_labels):
        raise RuntimeError("Not every subject received exactly one test fold")
    summary = write_oof_artifacts(
        run_dir,
        split_payload,
        oof_rows,
        per_fold_rows,
        eligible_samples,
        checkpoint_reports,
        head_input_dim,
        expected_head_parameter_count,
        {
            "pooling": cfg.finetune.pooling,
            "head_lr": float(cfg.finetune.head_lr),
            "weight_decay": float(cfg.finetune.weight_decay),
            "optimizer_name": "SGD" if cfg.finetune.mode == "linear_probe" else "AdamW",
            "num_tokens": int(shape_params["num_tokens"]),
            "d_model": int(cfg.model.d_model),
        },
    )
    summary["smoke"] = bool(cfg.finetune.smoke)
    summary["fold_seeds"] = [int(cfg.finetune.seed) + fold for fold in range(4)]
    json_dump(Path(run_dir) / "sex_cv4_summary.json", summary)
    return {
        "metrics": {
            "window": summary["window_metrics"],
            "subject": summary["subject_metrics"],
            "per_fold": per_fold_rows,
        },
        "sample_count": len(selected_rows),
        "split_subjects": split_payload["folds"],
        "excluded": [],
        "fold_results": fold_results,
        "training_metadata": fold_results[0]["training_metadata"],
    }, checkpoint_reports


def write_outputs(run_dir, cfg, args, result, checkpoint_report, schema, source_before, source_after):
    shape_params = compute_shape_params(cfg)
    resolved_input_dim, resolved_parameter_count = resolved_head_dimensions(cfg, shape_params)
    training_metadata = result.get("training_metadata") or {
        "pooling": cfg.finetune.pooling,
        "head_lr": float(cfg.finetune.head_lr),
        "weight_decay": float(cfg.finetune.weight_decay),
        "optimizer_name": None,
        "optimizer_param_groups": [],
        "head_input_dim": resolved_input_dim,
        "head_parameter_count": resolved_parameter_count,
        "num_tokens": shape_params["num_tokens"],
        "d_model": cfg.model.d_model,
    }
    config_payload = {
        "config": asdict(cfg),
        "cli_args": vars(args),
        "actual_command": subprocess.list2cmdline([sys.executable, *sys.argv]),
    }
    config_payload["cli_args"]["output_dir"] = str(config_payload["cli_args"]["output_dir"])
    for key in ["checkpoint", "downstream_checkpoint"]:
        if config_payload["cli_args"].get(key) is not None:
            config_payload["cli_args"][key] = str(config_payload["cli_args"][key])
    json_dump(run_dir / "experiment_config.json", config_payload)
    json_dump(run_dir / "config.json", config_payload)
    json_dump(run_dir / "metrics" / "metrics.json", result)
    provenance = {
        "created_at": datetime.now().astimezone().isoformat(),
        "project_git": get_git_state(),
        "upstream_50m": {"url": UPSTREAM_URL, "branch": UPSTREAM_BRANCH, "commit": UPSTREAM_COMMIT},
        "input_lance": cfg.data.lance_path,
        "input_lance_schema": schema,
        "input_snapshot_before": source_before,
        "input_snapshot_after": source_after,
        "input_unchanged": source_before == source_after if source_before is not None else None,
        "task": cfg.finetune.task,
        "label_mapping": SEX_LABEL_MAPPING if cfg.finetune.task == "sex_classification" else None,
        "split_seed": cfg.finetune.split_seed,
        "split_subjects": result.get("split_subjects"),
        "excluded": result.get("excluded", []),
        "checkpoint": checkpoint_report,
        "model_config": asdict(cfg.model),
        "smoke": cfg.finetune.smoke,
        "actual_command": config_payload["actual_command"],
        "versions": {
            "python": platform.python_version(), "torch": torch.__version__,
            "lance": lance.__version__, "pyarrow": pyarrow.__version__,
        },
        "protocol": cfg.finetune.protocol,
        "pooling": training_metadata["pooling"],
        "head_lr": training_metadata["head_lr"],
        "weight_decay": training_metadata["weight_decay"],
        "optimizer_name": training_metadata["optimizer_name"],
        "optimizer_param_groups": training_metadata["optimizer_param_groups"],
        "head_input_dim": training_metadata["head_input_dim"],
        "head_parameter_count": training_metadata["head_parameter_count"],
        "num_tokens": training_metadata["num_tokens"],
        "d_model": training_metadata["d_model"],
        "mask_application": (
            "masked mean excludes invalid tokens from numerator and denominator"
            if cfg.finetune.pooling == "mean"
            else "masked flatten zeros invalid token embeddings before preserving token order"
        ),
        "scheduler": None,
    }
    json_dump(run_dir / "run_provenance.json", provenance)
    json_dump(run_dir / "processing_provenance.json", provenance)
    json_dump(run_dir / "summary.json", {
        "task": cfg.finetune.task,
        "protocol": cfg.finetune.protocol,
        "smoke": bool(cfg.finetune.smoke),
        **training_metadata,
        "metrics": result.get("metrics"),
    })
    return provenance


def main(argv=None):
    args = parse_args(argv)
    if args.allow_random_init_smoke and not args.smoke:
        raise ValueError("--allow-random-init-smoke is only valid with --smoke")
    if args.smoke and args.evaluate:
        raise ValueError("Smoke mode performs a mechanical train step; do not combine with --evaluate")
    if args.protocol == SEX_CV4_PROTOCOL and args.task != "sex_classification":
        raise ValueError(f"{SEX_CV4_PROTOCOL} is only valid for sex_classification")
    if args.protocol == SEX_CV4_PROTOCOL and args.folds != 4:
        raise ValueError(f"{SEX_CV4_PROTOCOL} requires --folds 4")
    if args.protocol == SEX_CV4_PROTOCOL and args.allow_random_init_smoke:
        raise ValueError("CV folds must all load the same real pretrained checkpoint")
    if not args.smoke and args.protocol == "legacy_single_split" and not args.subject_split_manifest:
        raise ValueError("Full train/evaluation requires --subject-split-manifest")
    if min(args.epochs, args.batch_size, args.smoke_max_samples, args.smoke_max_samples_per_subject) <= 0:
        raise ValueError("epochs, batch-size, and smoke sample limits must be positive")

    cfg = Config()
    apply_cli_overrides(cfg, args)
    run_dir = setup_experiment_dir(cfg, overwrite=args.overwrite)
    set_seed(cfg.finetune.seed)
    device = torch.device(cfg.finetune.device)
    shape_params = compute_shape_params(cfg)
    storage_options = load_storage_options(cfg.data.storage_options_path)
    ds = lance.dataset(cfg.data.lance_path, **({} if storage_options is None else {"storage_options": storage_options}))
    schema = str(ds.schema)
    source_before = source_snapshot(cfg.data.lance_path)

    if cfg.finetune.protocol == SEX_CV4_PROTOCOL:
        result, checkpoint_report = run_sex_cv4(cfg, shape_params, device, run_dir)
    elif cfg.finetune.smoke:
        result, checkpoint_report = run_smoke(cfg, shape_params, device)
    else:
        subject_lists, excluded = validate_subject_split_manifest(
            cfg.finetune.subject_split_manifest, cfg.finetune.task
        )
        train_set, val_set, test_set, subject_lists, excluded = build_train_val_test_split(None, cfg)
        loaders = build_loaders(train_set, val_set, test_set, cfg, device)
        _, metrics, _, checkpoint_report, _ = run_one_checkpoint(
            cfg, shape_params, *loaders, device
        )
        result = {
            "metrics": metrics,
            "training_metadata": metrics.get("training_metadata"),
            "sample_count": {
                "train": len(train_set), "validation": len(val_set), "test": len(test_set)
            },
            "split_subjects": subject_lists,
            "excluded": excluded,
        }

    source_after = source_snapshot(cfg.data.lance_path)
    provenance = write_outputs(
        run_dir, cfg, args, result, checkpoint_report, schema, source_before, source_after
    )
    print(json.dumps({
        "output_dir": str(run_dir.resolve()),
        "result": result,
        "checkpoint": cfg.finetune.save_path,
        "input_unchanged": provenance["input_unchanged"],
    }, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
