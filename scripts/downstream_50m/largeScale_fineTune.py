"""Huada port of 50M_Model/largeScale_fineTune.py.

The backbone, representation layer, masked-flatten pooling, heads, optimizer
choices, and epoch loop follow upstream. Adaptations are limited to the shared
Huada Lance contract, explicit subject manifests, regression, richer metrics,
checkpoint auditing, smoke mode, and evaluation mode.
"""

from __future__ import annotations

import argparse
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
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-max-samples", type=int, default=4)
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
    cfg.finetune.smoke = args.smoke
    cfg.finetune.smoke_max_samples = args.smoke_max_samples
    cfg.finetune.allow_random_init_smoke = args.allow_random_init_smoke
    cfg.finetune.evaluate_only = args.evaluate
    cfg.finetune.num_classes = 2 if args.task == "sex_classification" else 1
    cfg.finetune.metric_for_best = "val_acc" if args.task == "sex_classification" else "val_loss"


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
        num_tokens=None,
    ):
        super().__init__()
        self.backbone = backbone
        self.probe_layer_idx = probe_layer_idx
        self.freeze_backbone = freeze_backbone
        for parameter in self.backbone.parameters():
            parameter.requires_grad = not freeze_backbone
        head_input_dim = d_model if num_tokens is None else num_tokens * d_model
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

    def flatten_valid_tokens(self, token_embeddings, token_valid_mask):
        valid_mask = token_valid_mask.unsqueeze(-1).float()
        return (token_embeddings * valid_mask).flatten(start_dim=1)

    def pool_valid_tokens(self, token_embeddings, token_valid_mask):
        valid_mask = token_valid_mask.unsqueeze(-1).float()
        pooled = (token_embeddings * valid_mask).sum(dim=1)
        return pooled / valid_mask.sum(dim=1).clamp_min(1.0)

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
        flattened = self.flatten_valid_tokens(token_embeddings, token_valid_mask)
        return self.head(flattened)


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


def build_backbone(cfg, shape_params, device, load_pretrained=True):
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
        "path": None, "sha256": None, "strict": False,
        "missing_keys": [], "unexpected_keys": [], "random_init": True,
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
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    state_dict = strip_module_prefix(state_dict)
    missing_keys, unexpected_keys = backbone.load_state_dict(state_dict, strict=False)
    report = {
        "path": str(Path(ckpt_path).resolve()),
        "sha256": sha256_file(ckpt_path),
        "strict": False,
        "missing_keys": list(missing_keys),
        "unexpected_keys": list(unexpected_keys),
        "random_init": False,
    }
    if len(missing_keys) + len(unexpected_keys) > 8:
        raise RuntimeError(f"Checkpoint mismatch is too large: {report}")
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
        num_tokens=shape_params["num_tokens"],
    ).to(device)
    return model, probe_layer_idx


def make_optimizer(cfg, model):
    if cfg.finetune.mode == "finetune":
        return torch.optim.AdamW(
            [
                {"params": model.backbone.parameters(), "lr": cfg.finetune.backbone_lr},
                {"params": model.head.parameters(), "lr": cfg.finetune.head_lr},
            ],
            weight_decay=cfg.finetune.weight_decay,
        )
    return torch.optim.SGD(model.head.parameters(), lr=cfg.finetune.head_lr)


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


def save_downstream_checkpoint(path, model, cfg, shape_params, probe_layer_idx, epoch, metrics, checkpoint_report):
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


def run_one_checkpoint(cfg, shape_params, train_loader, val_loader, test_loader, device, checkpoint_report=None):
    backbone, loaded_report = build_backbone(cfg, shape_params, device, load_pretrained=not cfg.finetune.evaluate_only)
    checkpoint_report = checkpoint_report or loaded_report
    model, probe_layer_idx = build_downstream_model(cfg, backbone, device, shape_params)
    criterion = make_criterion(cfg.finetune.task)

    if cfg.finetune.evaluate_only:
        path = cfg.finetune.downstream_checkpoint
        if not path or not Path(path).is_file():
            raise FileNotFoundError(path or "--downstream-checkpoint is required for --evaluate")
        payload = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        test_metrics = validate_one_epoch(model, test_loader, criterion, device, cfg.finetune.task)
        return model, {"test": test_metrics}, probe_layer_idx, checkpoint_report

    optimizer = make_optimizer(cfg, model)
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
                epoch + 1, epoch_metrics, checkpoint_report,
            )
    if best_payload is None:
        raise RuntimeError("No best checkpoint was produced")
    model.load_state_dict(best_payload["model_state_dict"], strict=True)
    test_metrics = validate_one_epoch(model, test_loader, criterion, device, cfg.finetune.task)
    best_payload["metrics"]["test"] = test_metrics
    temporary = cfg.finetune.save_path + ".tmp"
    torch.save(best_payload, temporary)
    os.replace(temporary, cfg.finetune.save_path)
    return model, {"epochs": epoch_rows, "test": test_metrics}, probe_layer_idx, checkpoint_report


def run_smoke(cfg, shape_params, device):
    dataset = load_downstream_dataset(cfg, max_samples=cfg.finetune.smoke_max_samples)
    if not len(dataset):
        raise ValueError("No eligible smoke samples")
    loader = _loader(dataset, cfg, device, shuffle=False)
    backbone, checkpoint_report = build_backbone(cfg, shape_params, device)
    model, probe_layer_idx = build_downstream_model(cfg, backbone, device, shape_params)
    criterion = make_criterion(cfg.finetune.task)
    optimizer = make_optimizer(cfg, model)
    metrics = run_epoch(
        model, loader, criterion, device, cfg.finetune.task, optimizer=optimizer, smoke=True
    )
    subjects = sorted({row["subject_id"] for row in dataset.get_subject_label_rows()})
    checkpoint_metrics = {"epoch": 1, "train": metrics, "smoke": True}
    save_downstream_checkpoint(
        cfg.finetune.save_path, model, cfg, shape_params, probe_layer_idx,
        1, checkpoint_metrics, checkpoint_report,
    )
    return {
        "metrics": checkpoint_metrics,
        "sample_count": len(dataset),
        "subjects": subjects,
        "split_subjects": {"train": subjects, "validation": [], "test": []},
        "excluded": ([{"subject_id": "SUB5", "reason": "age_missing"}]
                     if cfg.finetune.task == "age_regression" else []),
    }, checkpoint_report


def write_outputs(run_dir, cfg, args, result, checkpoint_report, schema, source_before, source_after):
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
        "mask_application": "token embeddings multiplied by token_valid_mask before upstream flatten pooling",
        "scheduler": None,
    }
    json_dump(run_dir / "run_provenance.json", provenance)
    return provenance


def main(argv=None):
    args = parse_args(argv)
    if args.allow_random_init_smoke and not args.smoke:
        raise ValueError("--allow-random-init-smoke is only valid with --smoke")
    if args.smoke and args.evaluate:
        raise ValueError("Smoke mode performs a mechanical train step; do not combine with --evaluate")
    if not args.smoke and not args.subject_split_manifest:
        raise ValueError("Full train/evaluation requires --subject-split-manifest")
    if args.epochs <= 0 or args.batch_size <= 0 or args.smoke_max_samples <= 0:
        raise ValueError("epochs, batch-size, and smoke-max-samples must be positive")

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

    if cfg.finetune.smoke:
        result, checkpoint_report = run_smoke(cfg, shape_params, device)
    else:
        subject_lists, excluded = validate_subject_split_manifest(
            cfg.finetune.subject_split_manifest, cfg.finetune.task
        )
        train_set, val_set, test_set, subject_lists, excluded = build_train_val_test_split(None, cfg)
        loaders = build_loaders(train_set, val_set, test_set, cfg, device)
        _, metrics, _, checkpoint_report = run_one_checkpoint(
            cfg, shape_params, *loaders, device
        )
        result = {
            "metrics": metrics,
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
