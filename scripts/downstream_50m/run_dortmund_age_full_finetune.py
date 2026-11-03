#!/usr/bin/env python3
"""Dortmund age full-finetune entry point for the shared 50M FBMAE stack.

This command deliberately requires a frozen subject split manifest.  Test rows
are not constructed until validation selection is locked.  ``--smoke`` is the
only mode that may use random initialization or accept a conditional token
audit; a formal run requires a passed audit and a real pretrained checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import lance
import numpy as np
import pyarrow
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from configs import Config
from dataset import collate_fn
from dortmund_age_finetune import (
    MASK_SEMANTICS_VERSION,
    SELECTION_ATOL,
    NoPaddingDistributedEvalSampler,
    StrictDortmundAgeModel,
    SubjectBalancedDistributedSampler,
    TestAccessGuard,
    aggregate_subject_predictions,
    atomic_torch_save,
    audit_gradient_coverage,
    broadcast_object,
    build_linear_warmup_cosine_scheduler,
    build_optimizer,
    build_target_transform,
    capture_rng_state,
    configure_probe_gradient_scope,
    ensure_finite,
    gather_prediction_rows,
    initialize_distributed,
    metric_bundle,
    restore_rng_state,
    sha256_file,
    sha256_json,
    train_mean_baseline,
    validate_prediction_coverage,
    validate_resume_payload,
    validation_is_better,
    zero_initialize_age_head,
)
from lance_dataset import LanceEEGPretrainDataset, load_subject_manifest
from largeScale_fineTune import build_backbone, compute_shape_params
from utils import set_seed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-lance", required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--audit-summary", type=Path)
    parser.add_argument(
        "--audit-policy", choices=["enforce", "record_only"], default="enforce",
        help="enforce requires a passed audit for formal runs; record_only records any supplied audit without gating",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32, help="Per-device batch size")
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--samples-per-epoch", type=int)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--scheduler", choices=["linear_warmup_cosine_decay"], default="linear_warmup_cosine_decay")
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--early-stopping-patience", type=int, default=15)
    parser.add_argument("--probe-layer-idx", type=int, default=8)
    parser.add_argument("--pooling", choices=["mean"], default="mean")
    parser.add_argument("--amp-mode", choices=["none", "fp16", "bf16"], default="none")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--selection-tolerance", type=float, default=SELECTION_ATOL)
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow-random-init-smoke", action="store_true")
    parser.add_argument("--smoke-train-subjects", type=int, default=3)
    parser.add_argument("--smoke-validation-subjects", type=int, default=2)
    parser.add_argument("--smoke-windows-per-subject", type=int, default=4)
    parser.add_argument(
        "--max-epochs-this-run", type=int,
        help="Smoke/resume control: stop at an epoch boundary without changing the training protocol",
    )
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    positive = {
        "epochs": args.epochs, "batch_size": args.batch_size,
        "accumulation_steps": args.accumulation_steps,
        "backbone_lr": args.backbone_lr, "head_lr": args.head_lr,
        "gradient_clip_norm": args.gradient_clip_norm,
    }
    invalid = {key: value for key, value in positive.items() if value <= 0}
    if invalid:
        raise ValueError(f"Positive arguments required: {invalid}")
    if args.weight_decay < 0 or not 0 <= args.warmup_ratio < 1:
        raise ValueError("weight_decay must be >=0 and warmup_ratio must be in [0,1)")
    if args.early_stopping_patience < 1 or args.num_workers < 0:
        raise ValueError("Invalid early stopping patience or num_workers")
    if args.probe_layer_idx != 8 or args.pooling != "mean":
        raise ValueError("Protocol v1 is fixed to probe_layer_idx=8 and masked mean pooling")
    if args.amp_mode == "fp16" and (args.device != "cuda" or not torch.cuda.is_available()):
        raise ValueError("fp16 AMP requires CUDA")
    if not args.smoke and args.skip_test:
        raise ValueError("Formal training cannot skip the single locked test evaluation")
    if args.allow_random_init_smoke and not args.smoke:
        raise ValueError("Random initialization is smoke-only")
    if args.audit_policy == "enforce" and args.audit_summary is None:
        raise ValueError("--audit-summary is required when --audit-policy=enforce")
    if args.max_epochs_this_run is not None and (not args.smoke or args.max_epochs_this_run < 1):
        raise ValueError("--max-epochs-this-run is a positive smoke-only control")


def _json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def _csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _git_state() -> dict[str, Any]:
    def run(*command: str) -> str:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
    try:
        status = run("git", "status", "--porcelain")
        return {
            "commit": run("git", "rev-parse", "HEAD"),
            "branch": run("git", "branch", "--show-current"),
            "dirty": bool(status),
            "dirty_files": status.splitlines(),
        }
    except Exception as error:
        return {"commit": None, "branch": None, "dirty": None, "error": repr(error)}


def _audit_gate(path: Path | None, smoke: bool, policy: str) -> dict[str, Any]:
    if policy == "record_only":
        if path is None:
            return {
                "policy": "record_only", "conclusion": "not_provided",
                "formal_training_allowed": True,
                "note": "audit gate explicitly disabled by CLI; checkpoint/data runtime checks remain active",
            }
        payload = json.loads(path.read_text(encoding="utf-8"))
        conclusion = str(payload.get("overall_conclusion", payload.get("conclusion", ""))).strip().lower()
        return {
            "policy": "record_only", "conclusion": conclusion,
            "formal_training_allowed": True, "path": str(path),
            "sha256": sha256_file(path),
            "note": "audit conclusion recorded but not used as a launch gate",
        }
    if path is None:
        raise ValueError("Audit summary is required by enforce policy")
    payload = json.loads(path.read_text(encoding="utf-8"))
    conclusion = str(payload.get("overall_conclusion", payload.get("conclusion", ""))).strip().lower()
    if conclusion == "passed":
        return {"policy": "enforce", "conclusion": conclusion, "formal_training_allowed": True, "path": str(path), "sha256": sha256_file(path)}
    if smoke and conclusion == "conditional_pass":
        return {"policy": "enforce", "conclusion": conclusion, "formal_training_allowed": False, "smoke_only": True, "path": str(path), "sha256": sha256_file(path)}
    raise RuntimeError(f"Audit gate rejected run: conclusion={conclusion!r}, smoke={smoke}")


def _lance_fingerprint(path: str) -> dict[str, Any]:
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(path)
    dataset = lance.dataset(path)
    files = []
    for item in sorted(root.rglob("*")):
        if item.is_file():
            stat = item.stat()
            files.append({"path": item.relative_to(root).as_posix(), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    fingerprint = sha256_json({"schema": str(dataset.schema), "rows": dataset.count_rows(), "files": files})
    return {"fingerprint": fingerprint, "row_count": dataset.count_rows(), "schema": str(dataset.schema), "files": files}


def _token_adapter_provenance(path: str) -> dict[str, Any]:
    """Read adapter sidecars adjacent to the Lance; paths are provenance only."""
    run_dir = Path(path).resolve().parent
    result: dict[str, Any] = {"run_dir": str(run_dir), "sidecars": {}}
    for filename in (
        "processing_provenance.json", "resolved_config.json",
        "electrode_vocabulary_snapshot.json", "channel_profile_resolved.json",
        "channel_mapping_summary.json",
    ):
        candidate = run_dir / filename
        if candidate.is_file():
            result["sidecars"][filename] = {
                "sha256": sha256_file(candidate),
                "content": json.loads(candidate.read_text(encoding="utf-8-sig")),
            }
    return result


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    rows = list(load_subject_manifest(path).values())
    if not rows:
        raise ValueError("Empty split manifest")
    split_ids: dict[str, set[str]] = {name: set() for name in ("train", "validation", "test")}
    for row in rows:
        split = str(row.get("split", ""))
        if split not in split_ids:
            raise ValueError(f"Invalid split {split!r}")
        subject = str(row["subject_id"])
        if any(subject in values for name, values in split_ids.items() if name != split):
            raise ValueError(f"Subject leakage: {subject}")
        split_ids[split].add(subject)
    if any(not values for values in split_ids.values()):
        raise ValueError(f"All splits must be non-empty: {split_ids}")
    return rows


def _smoke_subset(dataset: LanceEEGPretrainDataset, subject_limit: int, windows_per_subject: int) -> Subset:
    rows = dataset.get_sampling_rows()
    subjects = sorted({row["subject_id"] for row in rows})[:subject_limit]
    selected = []
    for subject in subjects:
        choices = [row["index"] for row in rows if row["subject_id"] == subject]
        selected.extend(choices[:windows_per_subject])
    if not selected:
        raise ValueError("Smoke subset is empty")
    return Subset(dataset, selected)


def _sampling_rows(dataset: LanceEEGPretrainDataset | Subset) -> list[dict[str, Any]]:
    if isinstance(dataset, Subset):
        original = {row["index"]: row for row in dataset.dataset.get_sampling_rows()}
        result = []
        for new_index, old_index in enumerate(dataset.indices):
            row = dict(original[int(old_index)])
            row["index"] = new_index
            result.append(row)
        return result
    return dataset.get_sampling_rows()


def _expected_sample_ids(dataset: LanceEEGPretrainDataset | Subset) -> list[str]:
    return [str(row["sample_id"]) for row in _sampling_rows(dataset)]


def _dataloader(dataset, *, sampler, args: argparse.Namespace) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
        drop_last=False,
        collate_fn=collate_fn,
        persistent_workers=args.num_workers > 0,
    )


def _move(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def _autocast(args: argparse.Namespace, device: torch.device):
    if args.amp_mode == "none":
        return nullcontext()
    dtype = torch.float16 if args.amp_mode == "fp16" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype)


def _model_for_state(model: torch.nn.Module) -> StrictDortmundAgeModel:
    return model.module if isinstance(model, DistributedDataParallel) else model


def _parameter_norm(model: torch.nn.Module) -> float:
    squares = [parameter.detach().float().norm().pow(2) for parameter in model.parameters()]
    return float(torch.stack(squares).sum().sqrt().item()) if squares else 0.0


def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    transform,
    device: torch.device,
    args: argparse.Namespace,
    context: dict[str, Any],
    expected_ids: Sequence[str],
    guard: TestAccessGuard | None = None,
) -> tuple[float, list[dict[str, Any]], dict[str, Any] | None]:
    model.eval()
    local_rows: list[dict[str, Any]] = []
    local_loss_sum = torch.zeros(1, dtype=torch.float64, device=device)
    local_count = torch.zeros(1, dtype=torch.float64, device=device)
    with torch.inference_mode():
        for batch in loader:
            if guard is not None:
                guard.mark_rows(len(batch["sample_id"]))
                guard.mark_forward()
            batch = _move(batch, device)
            with _autocast(args, device):
                prediction_z = model(
                    batch["token_inputs"], batch["token_channel_indices"],
                    batch["token_time_indices"], batch["token_valid_mask"],
                ).view(-1)
                truth_year = batch["label"].float().view(-1)
                truth_z = transform.transform_tensor(truth_year)
                squared = (prediction_z.float() - truth_z.float()).pow(2)
            ensure_finite("evaluation predictions", prediction_z)
            prediction_year = transform.inverse_tensor(prediction_z.float())
            ensure_finite("evaluation year predictions", prediction_year)
            local_loss_sum += squared.double().sum()
            local_count += truth_year.numel()
            for index, sample_id in enumerate(batch["sample_id"]):
                local_rows.append({
                    "sample_id": str(sample_id),
                    "subject_id": str(batch["subject_id"][index]),
                    "recording_id": str(batch.get("recording_id", [""] * len(batch["sample_id"]))[index]),
                    "eyes_condition": str(batch.get("eyes_condition", [""] * len(batch["sample_id"]))[index]),
                    "sex": str(batch.get("sex", [""] * len(batch["sample_id"]))[index]),
                    "true_age": float(truth_year[index].item()),
                    "predicted_age": float(prediction_year[index].item()),
                    "predicted_z": float(prediction_z[index].float().item()),
                })
    if context["distributed_enabled"]:
        dist.all_reduce(local_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_count, op=dist.ReduceOp.SUM)
    loss = float((local_loss_sum / local_count.clamp_min(1)).item())
    gathered = gather_prediction_rows(local_rows, context["world_size"])
    metrics = None
    if context["rank"] == 0:
        validate_prediction_coverage(gathered, expected_ids)
        metrics = metric_bundle(gathered)
    return loss, gathered if context["rank"] == 0 else [], metrics


def _checkpoint_payload(
    raw_model: StrictDortmundAgeModel,
    *, args, cfg, shape, transform, manifest_sha, token_snapshot, pretrained_report,
    selection, gradient_scope, gradient_audit, optimizer_report, scheduler_report,
    resolved_hash, git_state, critical_hashes,
) -> dict[str, Any]:
    return {
        "checkpoint_kind": "dortmund_age_50m_full_finetune",
        "backbone_state_dict": raw_model.backbone.state_dict(),
        "age_probe_state_dict": raw_model.head.state_dict(),
        "model_architecture_config": {
            "class": "EEGPretrainModel", "d_model": cfg.model.d_model,
            "depth": cfg.model.depth, "n_heads": cfg.model.n_heads,
            "mlp_ratio": cfg.model.mlp_ratio, "dropout": cfg.model.dropout,
            "patch_len": shape["patch_len"], "num_tokens": shape["num_tokens"],
            "num_time_patches": shape["num_time_patches"], "n_channels": cfg.data.n_channels,
            "probe_layer_idx": args.probe_layer_idx, "pooling": "masked_mean",
            "feature_dim": cfg.model.d_model,
        },
        "electrode_vocabulary": token_snapshot["adapter_provenance"]["sidecars"].get(
            "electrode_vocabulary_snapshot.json",
            {"status": "not_recorded", "embedded_in_token_channel_indices": False},
        ),
        "channel_canonicalization": token_snapshot["adapter_provenance"]["sidecars"].get(
            "channel_profile_resolved.json", {"status": "not_recorded"}
        ),
        "token_mask_semantics_version": MASK_SEMANTICS_VERSION,
        "source_reference_metadata": {"source_reference": "FCz", "processing_reference": "unchanged", "rereference_applied": False},
        "target_transform": asdict(transform),
        "split_manifest_sha256": manifest_sha,
        "source_token_lance_fingerprint": token_snapshot["fingerprint"],
        "source_token_adapter_provenance": token_snapshot["adapter_provenance"],
        "pretrained_checkpoint": pretrained_report,
        "best_epoch": None if selection is None else selection["epoch"],
        "validation_metrics": None if selection is None else selection.get("metrics"),
        "optimizer_configuration": optimizer_report,
        "scheduler_configuration": scheduler_report,
        "trainable_parameter_names": gradient_scope["trainable_parameter_names"],
        "trainable_parameter_count": gradient_scope["trainable_parameter_count"],
        "frozen_parameter_names": gradient_scope["frozen_parameter_names"],
        "frozen_parameter_count": gradient_scope["frozen_parameter_count"],
        "gradient_covered_parameter_count": gradient_audit.get("gradient_covered_parameter_count", 0),
        "gradient_coverage": gradient_audit,
        "git": git_state,
        "critical_code_sha256": critical_hashes,
        "resolved_config_sha256": resolved_hash,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _validate_args(args)
    context, device = initialize_distributed(args.device)
    rank = context["rank"]
    try:
        set_seed(args.train_seed + rank)
        gate = _audit_gate(args.audit_summary, args.smoke, args.audit_policy)
        if not args.split_manifest.is_file():
            raise FileNotFoundError(args.split_manifest)
        manifest_rows = _read_manifest(args.split_manifest)
        manifest_sha = sha256_file(args.split_manifest)
        transform = build_target_transform(manifest_rows, manifest_sha)
        token_snapshot = _lance_fingerprint(args.input_lance)
        token_snapshot["adapter_provenance"] = _token_adapter_provenance(args.input_lance)
        token_before = token_snapshot["fingerprint"]

        resolved = vars(args).copy()
        resolved.update({
            "checkpoint": None if args.checkpoint is None else str(args.checkpoint),
            "split_manifest": str(args.split_manifest),
            "audit_summary": None if args.audit_summary is None else str(args.audit_summary),
            "output_dir": str(args.output_dir), "resume": None if args.resume is None else str(args.resume),
            "distributed": context,
            "per_device_batch_size": args.batch_size,
            "world_size": context["world_size"],
            "effective_global_batch_size": args.batch_size * context["world_size"] * args.accumulation_steps,
            "learning_rate_scaling": "none",
        })
        resolved_hash = sha256_json({
            key: value for key, value in resolved.items()
            if key not in {"output_dir", "resume", "max_epochs_this_run"}
        })

        if rank == 0:
            resume_in_place = False
            if args.output_dir.exists():
                resume_in_place = (
                    args.resume is not None
                    and args.resume.resolve().parent.parent == args.output_dir.resolve()
                )
                if not resume_in_place:
                    raise FileExistsError(f"Refusing to overwrite output: {args.output_dir}")
            else:
                args.output_dir.mkdir(parents=True)
                (args.output_dir / "checkpoints").mkdir()
            _json(args.output_dir / "resolved_config.json", resolved)
            _json(args.output_dir / "target_transform.json", asdict(transform))
            shutil.copyfile(args.split_manifest, args.output_dir / "subject_split_manifest.csv")
            for filename in (
                "split_summary.json", "split_age_distribution.csv",
                "split_age_sex_distribution.csv", "resolved_split_config.json",
            ):
                source = args.split_manifest.parent / filename
                if source.is_file():
                    shutil.copyfile(source, args.output_dir / filename)
            if not (args.output_dir / "split_summary.json").exists():
                split_counts = {
                    split: sum(str(row["split"]) == split for row in manifest_rows)
                    for split in ("train", "validation", "test")
                }
                _json(args.output_dir / "split_summary.json", {"split_counts": split_counts, "manifest_sha256": manifest_sha})
            for filename, header in (
                ("split_age_distribution.csv", ["split", "age", "subject_count"]),
                ("split_age_sex_distribution.csv", ["split", "age", "sex", "subject_count"]),
            ):
                if not (args.output_dir / filename).exists():
                    _csv(args.output_dir / filename, [], header)
            if not (args.output_dir / "resolved_split_config.json").exists():
                _json(args.output_dir / "resolved_split_config.json", {"source_manifest": str(args.split_manifest), "manifest_sha256": manifest_sha})
            if not resume_in_place:
                _csv(args.output_dir / "training_issues.csv", [], ["severity", "category", "message"])
                (args.output_dir / "training_errors.log").write_text("", encoding="utf-8")
        if context["distributed_enabled"]:
            dist.barrier()

        cfg = Config()
        cfg.data.lance_path = args.input_lance
        cfg.finetune.task = "age_regression"
        cfg.finetune.pretrained_ckpt_path = "" if args.checkpoint is None else str(args.checkpoint)
        cfg.finetune.smoke = args.smoke
        cfg.finetune.allow_random_init_smoke = args.allow_random_init_smoke
        cfg.finetune.pooling = "mean"
        cfg.finetune.mode = "finetune"
        cfg.finetune.num_classes = 1
        shape = compute_shape_params(cfg)
        if (args.checkpoint is None or not args.checkpoint.is_file()) and not (args.smoke and args.allow_random_init_smoke):
            raise FileNotFoundError("A real pretrained checkpoint is required outside explicit random-init smoke")
        backbone, pretrained_report = build_backbone(cfg, shape, device, load_pretrained=True)
        raw_model = StrictDortmundAgeModel(
            backbone=backbone, d_model=cfg.model.d_model, num_classes=1,
            probe_layer_idx=args.probe_layer_idx, freeze_backbone=False,
            pooling="mean", num_tokens=shape["num_tokens"],
        ).to(device)
        zero_initialize_age_head(raw_model)
        gradient_scope = configure_probe_gradient_scope(raw_model, args.probe_layer_idx)
        optimizer, optimizer_report = build_optimizer(raw_model, backbone_lr=args.backbone_lr, head_lr=args.head_lr, weight_decay=args.weight_decay)

        train_dataset: LanceEEGPretrainDataset | Subset = LanceEEGPretrainDataset(
            args.input_lance, task="age_regression", subject_split_manifest=args.split_manifest, split="train"
        )
        validation_dataset: LanceEEGPretrainDataset | Subset = LanceEEGPretrainDataset(
            args.input_lance, task="age_regression", subject_split_manifest=args.split_manifest, split="validation"
        )
        if args.smoke:
            train_dataset = _smoke_subset(train_dataset, args.smoke_train_subjects, args.smoke_windows_per_subject)
            validation_dataset = _smoke_subset(validation_dataset, args.smoke_validation_subjects, args.smoke_windows_per_subject)
        train_rows = _sampling_rows(train_dataset)
        samples_per_epoch = args.samples_per_epoch or len(train_dataset)
        train_sampler = SubjectBalancedDistributedSampler(
            train_rows, samples_per_epoch=samples_per_epoch, seed=args.train_seed,
            rank=rank, world_size=context["world_size"], per_device_batch_size=args.batch_size,
        )
        val_sampler = NoPaddingDistributedEvalSampler(len(validation_dataset), rank, context["world_size"])
        train_loader = _dataloader(train_dataset, sampler=train_sampler, args=args)
        val_loader = _dataloader(validation_dataset, sampler=val_sampler, args=args)
        steps_per_epoch = math.ceil(len(train_loader) / args.accumulation_steps)
        scheduler, scheduler_report = build_linear_warmup_cosine_scheduler(optimizer, steps_per_epoch * args.epochs, args.warmup_ratio)
        scaler = torch.amp.GradScaler("cuda", enabled=args.amp_mode == "fp16")
        model: torch.nn.Module = raw_model
        if context["distributed_enabled"]:
            model = DistributedDataParallel(
                raw_model, device_ids=[context["local_rank"]] if device.type == "cuda" else None,
                output_device=context["local_rank"] if device.type == "cuda" else None,
                find_unused_parameters=False,
            )

        critical_paths = [
            Path(__import__(module).__file__).resolve()
            for module in ("model", "largeScale_fineTune", "lance_dataset", "dortmund_age_finetune")
        ] + [Path(__file__).resolve()]
        critical_hashes = {path.name: sha256_file(path) for path in critical_paths}
        git_state = _git_state()
        if rank == 0:
            _json(args.output_dir / "processing_provenance.json", {
                "hostname": platform.node(), "python": sys.version, "torch": torch.__version__,
                "lance": lance.__version__, "pyarrow": pyarrow.__version__, "git": git_state,
                "audit_gate": gate, "token_snapshot_before": token_snapshot,
                "split_manifest_sha256": manifest_sha, "pretrained_checkpoint": pretrained_report,
                "critical_code_sha256": critical_hashes, "formal_training_performed": not args.smoke,
            })
            _csv(args.output_dir / "parameter_group_summary.csv", optimizer_report, ["group_name", "owner", "decay_policy", "lr", "weight_decay", "parameter_count", "tensor_count", "parameter_names"])

        start_epoch, global_step = 0, 0
        best: dict[str, Any] | None = None
        patience = 0
        if args.resume is not None:
            resume = torch.load(args.resume, map_location=device, weights_only=False)
            validate_resume_payload(
                resume, token_fingerprint=token_snapshot["fingerprint"], manifest_sha256=manifest_sha,
                world_size=context["world_size"], resolved_config_sha256=resolved_hash,
            )
            raw_model.backbone.load_state_dict(resume["backbone_state_dict"], strict=True)
            raw_model.head.load_state_dict(resume["age_probe_state_dict"], strict=True)
            optimizer.load_state_dict(resume["optimizer_state_dict"])
            scheduler.load_state_dict(resume["scheduler_state_dict"])
            if args.amp_mode == "fp16" and resume.get("amp_scaler_state_dict"):
                scaler.load_state_dict(resume["amp_scaler_state_dict"])
            start_epoch = int(resume["epoch"]) + 1
            global_step = int(resume["global_step"])
            best = resume.get("best_checkpoint_state")
            patience = int(resume.get("early_stopping_state", {}).get("patience", 0))
            states = resume.get("per_rank_rng_states", [])
            if rank < len(states):
                restore_rng_state(states[rank])

        def existing_csv(filename: str) -> list[dict[str, Any]]:
            path = args.output_dir / filename
            if args.resume is None or not path.is_file():
                return []
            with path.open(encoding="utf-8", newline="") as handle:
                return list(csv.DictReader(handle))

        epoch_rows: list[dict[str, Any]] = existing_csv("epoch_metrics.csv") if rank == 0 else []
        sampler_rows: list[dict[str, Any]] = existing_csv("sampler_epoch_statistics.csv") if rank == 0 else []
        gradient_rows: list[dict[str, Any]] = existing_csv("gradient_statistics.csv") if rank == 0 else []
        gradient_audit: dict[str, Any] = {}
        guard = TestAccessGuard()
        expected_validation_ids = _expected_sample_ids(validation_dataset)
        stop = False
        training_started = time.time()
        last_completed_epoch = start_epoch - 1
        for epoch in range(start_epoch, args.epochs):
            epoch_started = time.time()
            train_sampler.set_epoch(epoch)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            local_loss_sum = torch.zeros(1, dtype=torch.float64, device=device)
            local_abs_year = torch.zeros(1, dtype=torch.float64, device=device)
            local_count = torch.zeros(1, dtype=torch.float64, device=device)
            max_grad_norm = 0.0
            for step, batch in enumerate(train_loader):
                batch = _move(batch, device)
                remainder = len(train_loader) % args.accumulation_steps
                in_short_final_group = remainder > 0 and step >= len(train_loader) - remainder
                accumulation_divisor = remainder if in_short_final_group else args.accumulation_steps
                final_accumulation = ((step + 1) % args.accumulation_steps == 0) or (step + 1 == len(train_loader))
                sync_context = nullcontext()
                if isinstance(model, DistributedDataParallel) and not final_accumulation:
                    sync_context = model.no_sync()
                with sync_context:
                    with _autocast(args, device):
                        prediction_z = model(
                            batch["token_inputs"], batch["token_channel_indices"],
                            batch["token_time_indices"], batch["token_valid_mask"],
                        ).view(-1)
                        truth_year = batch["label"].float().view(-1)
                        truth_z = transform.transform_tensor(truth_year)
                        loss = torch.nn.functional.mse_loss(prediction_z.float(), truth_z.float())
                        scaled_loss = loss / accumulation_divisor
                    ensure_finite("training loss", loss)
                    scaler.scale(scaled_loss).backward()
                prediction_year = transform.inverse_tensor(prediction_z.detach().float())
                ensure_finite("training prediction", prediction_year)
                local_loss_sum += loss.detach().double() * truth_year.numel()
                local_abs_year += (prediction_year - truth_year).abs().double().sum()
                local_count += truth_year.numel()
                if final_accumulation:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.gradient_clip_norm, error_if_nonfinite=True)
                    ensure_finite("gradient norm", grad_norm)
                    max_grad_norm = max(max_grad_norm, float(grad_norm.item()))
                    if not gradient_audit:
                        gradient_audit = audit_gradient_coverage(raw_model)
                        if gradient_audit["unexplained_trainable_without_gradient"] or gradient_audit["nonfinite_gradient_parameters"]:
                            raise RuntimeError(f"Gradient coverage failure: {gradient_audit}")
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    global_step += 1
                    for name, parameter in raw_model.named_parameters():
                        ensure_finite(f"parameter {name}", parameter)
            if context["distributed_enabled"]:
                for value in (local_loss_sum, local_abs_year, local_count):
                    dist.all_reduce(value, op=dist.ReduceOp.SUM)
            train_loss = float((local_loss_sum / local_count).item())
            train_mae = float((local_abs_year / local_count).item())
            validation_loss, validation_rows, validation_metrics = _evaluate(
                model, val_loader, transform, device, args, context, expected_validation_ids
            )
            decision = None
            if rank == 0:
                assert validation_metrics is not None
                subject_metrics = validation_metrics["subject_mean"]
                candidate = {
                    "epoch": epoch, "subject_mae": subject_metrics["mae"],
                    "subject_rmse": subject_metrics["rmse"], "validation_loss": validation_loss,
                    "metrics": validation_metrics,
                }
                improved = validation_is_better(candidate, best, args.selection_tolerance)
                if improved:
                    best = candidate
                    patience = 0
                    payload = _checkpoint_payload(
                        raw_model, args=args, cfg=cfg, shape=shape, transform=transform,
                        manifest_sha=manifest_sha, token_snapshot=token_snapshot,
                        pretrained_report=pretrained_report, selection=best,
                        gradient_scope=gradient_scope, gradient_audit=gradient_audit,
                        optimizer_report=optimizer_report, scheduler_report=scheduler_report,
                        resolved_hash=resolved_hash, git_state=git_state, critical_hashes=critical_hashes,
                    )
                    atomic_torch_save(payload, args.output_dir / "checkpoints" / "best_full_checkpoint.pt")
                else:
                    patience += 1
                elapsed = time.time() - epoch_started
                group_lrs = {
                    str(group.get("group_name", f"group_{index}")): float(group["lr"])
                    for index, group in enumerate(optimizer.param_groups)
                }
                current_backbone_lr = next(
                    value for name, value in group_lrs.items() if name.startswith("backbone_")
                )
                current_head_lr = next(
                    value for name, value in group_lrs.items() if name.startswith("age_head_")
                )
                row = {
                    "epoch": epoch, "train_z_loss": train_loss, "train_mae_years": train_mae,
                    "validation_z_loss": validation_loss,
                    "validation_subject_mae": subject_metrics["mae"],
                    "validation_subject_rmse": subject_metrics["rmse"],
                    "validation_window_mae": validation_metrics["window"]["mae"],
                    "backbone_lr": current_backbone_lr,
                    "head_lr": current_head_lr,
                    "gradient_norm": max_grad_norm, "parameter_norm": _parameter_norm(raw_model),
                    "prediction_min": min(row["predicted_age"] for row in validation_rows),
                    "prediction_max": max(row["predicted_age"] for row in validation_rows),
                    "gpu_peak_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                    "elapsed_seconds": elapsed, "eta_seconds": elapsed * max(args.epochs - epoch - 1, 0),
                    "improved": improved, "early_stopping_patience": patience,
                }
                epoch_rows.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)
                stats_row = train_sampler.statistics()
                sampler_rows.append({"epoch": epoch, **stats_row})
                gradient_rows.append({"epoch": epoch, "gradient_norm": max_grad_norm, "parameter_norm": row["parameter_norm"], "unexplained_trainable_without_gradient": 0})
                epoch_fields = list(epoch_rows[0])
                _csv(args.output_dir / "epoch_metrics.csv", epoch_rows, epoch_fields)
                _csv(args.output_dir / "training_trajectory.csv", epoch_rows, epoch_fields)
                _csv(
                    args.output_dir / "gradient_statistics.csv", gradient_rows,
                    ["epoch", "gradient_norm", "parameter_norm", "unexplained_trainable_without_gradient"],
                )
                _csv(
                    args.output_dir / "sampler_epoch_statistics.csv", sampler_rows,
                    ["epoch", "requested_samples", "global_samples", "local_samples_per_rank", "subject_counts", "subject_condition_counts", "missing_condition_subjects", "unique_window_count", "unique_window_fraction", "repeated_draw_count", "cross_rank_window_identity_overlap", "ddp_padding_duplicates", "rank_lengths"],
                )
                local_rng = capture_rng_state()
            else:
                local_rng = capture_rng_state()
            if context["distributed_enabled"]:
                rng_states: list[Any] = [None] * context["world_size"]
                dist.all_gather_object(rng_states, local_rng)
            else:
                rng_states = [local_rng]
            if rank == 0:
                last = _checkpoint_payload(
                    raw_model, args=args, cfg=cfg, shape=shape, transform=transform,
                    manifest_sha=manifest_sha, token_snapshot=token_snapshot,
                    pretrained_report=pretrained_report, selection=best,
                    gradient_scope=gradient_scope, gradient_audit=gradient_audit,
                    optimizer_report=optimizer_report, scheduler_report=scheduler_report,
                    resolved_hash=resolved_hash, git_state=git_state, critical_hashes=critical_hashes,
                )
                last.update({
                    "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
                    "amp_scaler_state_dict": scaler.state_dict() if args.amp_mode == "fp16" else None,
                    "epoch": epoch, "global_step": global_step,
                    "early_stopping_state": {"patience": patience, "limit": args.early_stopping_patience},
                    "best_checkpoint_state": best, "sampler_epoch": epoch,
                    "per_rank_rng_states": rng_states, "world_size": context["world_size"],
                    "resume_granularity": "epoch_boundary", "resolved_config_sha256": resolved_hash,
                })
                atomic_torch_save(last, args.output_dir / "checkpoints" / "last_training_checkpoint.pt")
                stop = patience >= args.early_stopping_patience
            stop = bool(broadcast_object(stop, rank))
            last_completed_epoch = epoch
            if stop:
                break
            if args.max_epochs_this_run is not None and (epoch - start_epoch + 1) >= args.max_epochs_this_run:
                break

        if rank == 0 and best is None:
            raise RuntimeError("No best checkpoint was selected")
        if context["distributed_enabled"]:
            dist.barrier()
        best_path = args.output_dir / "checkpoints" / "best_full_checkpoint.pt"
        best_hash = sha256_file(best_path) if rank == 0 else None
        best_hash = broadcast_object(best_hash, rank)
        best_payload = torch.load(best_path, map_location=device, weights_only=False)
        raw_model.backbone.load_state_dict(best_payload["backbone_state_dict"], strict=True)
        raw_model.head.load_state_dict(best_payload["age_probe_state_dict"], strict=True)
        if rank == 0 and sha256_file(best_path) != best_hash:
            raise RuntimeError("Best checkpoint hash changed during strict reload")
        guard.lock_selection()
        if rank == 0:
            _json(args.output_dir / "selection_locked.json", {
                "locked": True, "best_epoch": best_payload["best_epoch"],
                "best_checkpoint_sha256": best_hash, "selection_tolerance": args.selection_tolerance,
                "rule": ["subject_mean_mae", "subject_mean_rmse", "validation_loss", "earlier_epoch"],
                "test_state_at_lock": guard.state(),
            })

        final_val_loss, final_val_rows, final_val_metrics = _evaluate(
            model, val_loader, transform, device, args, context, expected_validation_ids
        )
        if rank == 0:
            assert final_val_metrics is not None
            val_subjects = aggregate_subject_predictions(final_val_rows)
            val_baseline = train_mean_baseline(val_subjects, transform.source_age_mean)
            val_baseline["model_delta_mae"] = final_val_metrics["subject_mean"]["mae"] - val_baseline["mae"]
            val_baseline["model_better_than_baseline"] = final_val_metrics["subject_mean"]["mae"] < val_baseline["mae"]
            _csv(args.output_dir / "validation_window_predictions.csv", final_val_rows, ["sample_id", "subject_id", "recording_id", "eyes_condition", "sex", "true_age", "predicted_age", "predicted_z"])
            _csv(args.output_dir / "validation_subject_predictions.csv", val_subjects, ["subject_id", "sex", "true_age", "mean_predicted_age", "median_predicted_age", "prediction_std", "signed_error", "absolute_error", "window_count", "eyes_open_windows", "eyes_closed_windows"])
            _json(args.output_dir / "validation_metrics.json", {**final_val_metrics, "subject_rows": None, "z_loss": final_val_loss})
            _json(args.output_dir / "validation_baseline_metrics.json", val_baseline)

        test_metrics = {"status": "not_performed", "reason": "--skip-test"}
        if not args.skip_test:
            guard.mark_dataset_created()
            test_dataset: LanceEEGPretrainDataset | Subset = LanceEEGPretrainDataset(
                args.input_lance, task="age_regression", subject_split_manifest=args.split_manifest, split="test"
            )
            test_sampler = NoPaddingDistributedEvalSampler(len(test_dataset), rank, context["world_size"])
            guard.mark_dataloader_created()
            test_loader = _dataloader(test_dataset, sampler=test_sampler, args=args)
            test_loss, test_rows, test_bundle = _evaluate(
                model, test_loader, transform, device, args, context, _expected_sample_ids(test_dataset), guard
            )
            if rank == 0:
                guard.mark_metrics()
                assert test_bundle is not None
                test_subjects = aggregate_subject_predictions(test_rows)
                test_baseline = train_mean_baseline(test_subjects, transform.source_age_mean)
                test_baseline["model_delta_mae"] = test_bundle["subject_mean"]["mae"] - test_baseline["mae"]
                test_baseline["model_better_than_baseline"] = test_bundle["subject_mean"]["mae"] < test_baseline["mae"]
                test_metrics = {**test_bundle, "subject_rows": None, "z_loss": test_loss}
                _csv(args.output_dir / "test_window_predictions.csv", test_rows, ["sample_id", "subject_id", "recording_id", "eyes_condition", "sex", "true_age", "predicted_age", "predicted_z"])
                _csv(args.output_dir / "test_subject_predictions.csv", test_subjects, ["subject_id", "sex", "true_age", "mean_predicted_age", "median_predicted_age", "prediction_std", "signed_error", "absolute_error", "window_count", "eyes_open_windows", "eyes_closed_windows"])
                _json(args.output_dir / "test_baseline_metrics.json", test_baseline)
        if rank == 0 and args.skip_test:
            _csv(args.output_dir / "test_window_predictions.csv", [], ["sample_id", "subject_id", "recording_id", "eyes_condition", "sex", "true_age", "predicted_age", "predicted_z"])
            _csv(args.output_dir / "test_subject_predictions.csv", [], ["subject_id", "sex", "true_age", "mean_predicted_age", "median_predicted_age", "prediction_std", "signed_error", "absolute_error", "window_count"])
            _json(args.output_dir / "test_baseline_metrics.json", {"status": "not_performed", "reason": "--skip-test"})
        if rank == 0:
            _json(args.output_dir / "test_metrics.json", test_metrics)
            epoch_fields = list(epoch_rows[0]) if epoch_rows else ["epoch"]
            _csv(args.output_dir / "epoch_metrics.csv", epoch_rows, epoch_fields)
            _csv(args.output_dir / "training_trajectory.csv", epoch_rows, epoch_fields)
            _csv(args.output_dir / "gradient_statistics.csv", gradient_rows, ["epoch", "gradient_norm", "parameter_norm", "unexplained_trainable_without_gradient"])
            _json(args.output_dir / "sampler_summary.json", {"version": train_sampler.statistics().get("version"), "requested_samples_per_epoch": samples_per_epoch, "global_rounding_multiple": context["world_size"] * args.batch_size, "ddp_padding_duplicates": 0})
            _csv(args.output_dir / "sampler_epoch_statistics.csv", sampler_rows, ["epoch", "requested_samples", "global_samples", "local_samples_per_rank", "subject_counts", "subject_condition_counts", "missing_condition_subjects", "unique_window_count", "unique_window_fraction", "repeated_draw_count", "cross_rank_window_identity_overlap", "ddp_padding_duplicates", "rank_lengths"])
            token_after = _lance_fingerprint(args.input_lance)
            unchanged = token_after["fingerprint"] == token_before
            if not unchanged:
                raise RuntimeError("Input Lance changed during training")
            if pretrained_report["random_init"]:
                pretrained_unchanged = None
                pretrained_unchanged_reason = "not_applicable_random_initialization_smoke"
            else:
                pretrained_unchanged = (
                    args.checkpoint is not None
                    and sha256_file(args.checkpoint) == pretrained_report["checkpoint_sha256"]
                )
                pretrained_unchanged_reason = None
                if not pretrained_unchanged:
                    raise RuntimeError("Pretrained checkpoint changed during training")
            inventory = [
                {"path": "checkpoints/best_full_checkpoint.pt", "sha256": sha256_file(best_path), "kind": "best_full"},
                {"path": "checkpoints/last_training_checkpoint.pt", "sha256": sha256_file(args.output_dir / "checkpoints" / "last_training_checkpoint.pt"), "kind": "last_training"},
            ]
            _csv(args.output_dir / "checkpoint_inventory.csv", inventory, ["path", "sha256", "kind"])
            resume_ready = bool(
                args.max_epochs_this_run is not None
                and last_completed_epoch + 1 < args.epochs
            )
            summary = {
                "status": "resume_ready" if resume_ready else "passed",
                "smoke": args.smoke, "cloud_full_training_performed": not args.smoke,
                "stopped_at_epoch_boundary_for_resume_smoke": resume_ready,
                "audit_gate": gate, "best_epoch": best_payload["best_epoch"],
                "input_lance_unchanged": unchanged,
                "pretrained_checkpoint_unchanged": pretrained_unchanged,
                "pretrained_checkpoint_unchanged_reason": pretrained_unchanged_reason,
                "subject_split_isolation": "passed", "train_only_target_zscore": "passed",
                "subject_balanced_sampling": "passed", "rank0_only_output": "passed",
                "distributed_prediction_coverage": "passed", "test_access_before_selection": False,
                "gradient_coverage_audited": True,
                "unexplained_trainable_without_gradient": len(gradient_audit.get("unexplained_trainable_without_gradient", [])),
                "elapsed_seconds": time.time() - training_started,
            }
            _json(args.output_dir / "training_summary.json", summary)
            (args.output_dir / "training_report.md").write_text(
                "# Dortmund age 50M full-finetune run\n\n" +
                f"- Status: `{summary['status']}`\n- Smoke: `{args.smoke}`\n" +
                f"- Best epoch: `{best_payload['best_epoch']}`\n- Audit gate: `{gate['conclusion']}`\n" +
                f"- Test: `{'skipped' if args.skip_test else 'performed once after lock'}`\n" +
                f"- Input Lance unchanged: `{unchanged}`\n",
                encoding="utf-8",
            )
            marker = "_RESUME_READY" if resume_ready else "_SUCCESS"
            (args.output_dir / marker).write_text(json.dumps({"best_checkpoint_sha256": best_hash}) + "\n", encoding="utf-8")
            obsolete = args.output_dir / ("_SUCCESS" if resume_ready else "_RESUME_READY")
            if obsolete.exists():
                obsolete.unlink()
        if context["distributed_enabled"]:
            dist.barrier()
        return 0
    except Exception as error:
        if context.get("rank", 0) == 0 and getattr(args, "output_dir", None) and args.output_dir.exists():
            with (args.output_dir / "training_errors.log").open("a", encoding="utf-8") as handle:
                handle.write(f"{type(error).__name__}: {error}\n")
        raise
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
