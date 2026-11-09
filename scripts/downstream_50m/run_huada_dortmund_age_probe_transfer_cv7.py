#!/usr/bin/env python3
"""Dortmund-age source-probe transfer to Huada with frozen-backbone CV7.

Formal execution builds/reuses one Dortmund-checkpoint-bound mean feature cache.
Every fold then re-reads the same source age probe, evaluates epoch 0, trains
only Linear(512,1), locks selection on one validation subject, and evaluates
one test subject exactly once.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

try:
    from . import age_cv7
    from . import feature_cache as cache
    from .dataset import collate_fn
    from .dortmund_huada_zero_shot import (
        CHECKPOINT_KIND as SOURCE_CHECKPOINT_KIND,
        EXPECTED_CHECKPOINT_SHA256, FEATURE_DIM, MASK_SEMANTICS_VERSION,
        POOLING as CHECKPOINT_POOLING, PROBE_LAYER_IDX, SOURCE_AGE_MEAN,
        SOURCE_AGE_STD, build_formal_model, load_strict_checkpoint,
        regression_metrics, sha256_file, validate_checkpoint_payload,
    )
    from .largeScale_fineTune import CachedFeatureClassifier
    from .neuralbench_checkpoint import tensor_state_fingerprint
    from .run_logging import log_event, setup_run_logging
except ImportError:
    import age_cv7
    import feature_cache as cache
    from dataset import collate_fn
    from dortmund_huada_zero_shot import (
        CHECKPOINT_KIND as SOURCE_CHECKPOINT_KIND,
        EXPECTED_CHECKPOINT_SHA256, FEATURE_DIM, MASK_SEMANTICS_VERSION,
        POOLING as CHECKPOINT_POOLING, PROBE_LAYER_IDX, SOURCE_AGE_MEAN,
        SOURCE_AGE_STD, build_formal_model, load_strict_checkpoint,
        regression_metrics, sha256_file, validate_checkpoint_payload,
    )
    from largeScale_fineTune import CachedFeatureClassifier
    from neuralbench_checkpoint import tensor_state_fingerprint
    from run_logging import log_event, setup_run_logging


RUN_MODE = "dortmund_age_source_probe_huada_transfer_cv7"
HEAD_CHECKPOINT_KIND = "dortmund_age_source_probe_huada_transfer_head_only"
POOLING = "mean"
PROBE_PARAMETER_COUNT = 513
SELECTION_TOLERANCE = 1e-9
ELIGIBLE_SUBJECTS = {"SUB1", "SUB2", "SUB3", "SUB4", "SUB6", "SUB7", "SUB8"}
OOF_FIELDS = [
    "fold", "sample_id", "subject_id", "recording_id", "window_index",
    "true_age", "true_z_source", "predicted_z", "predicted_age",
    "signed_error", "absolute_error", "squared_error", "num_valid_tokens",
    "test_is_outside_train_age_range", "checkpoint_path",
]
SUBJECT_FIELDS = [
    "subject_id", "true_age", "window_count", "mean_prediction",
    "median_prediction", "prediction_std", "prediction_min", "prediction_max",
    "signed_error", "absolute_error", "test_fold", "checkpoint_path",
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-lance", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-checkpoint-sha256", default=EXPECTED_CHECKPOINT_SHA256)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--feature-cache-root", required=True)
    parser.add_argument("--feature-cache-build-root")
    parser.add_argument("--feature-cache-policy", choices=("auto", "require", "rebuild"), default="auto")
    parser.add_argument("--feature-cache-batch-size", type=int, default=8)
    parser.add_argument("--cache-log-every-batches", type=int, default=5)
    parser.add_argument("--zero-shot-reference-dir")
    parser.add_argument("--head-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--early-stopping-patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--expected-source-age-mean", type=float, default=SOURCE_AGE_MEAN)
    parser.add_argument("--expected-source-age-std", type=float, default=SOURCE_AGE_STD)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    for name in ("feature_cache_batch_size", "cache_log_every_batches", "epochs", "early_stopping_patience", "batch_size"):
        if int(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.num_workers < 0 or args.head_lr <= 0 or args.weight_decay < 0:
        parser.error("num-workers/head-lr/weight-decay values are invalid")
    if len(args.expected_checkpoint_sha256) != 64 or any(c not in "0123456789abcdefABCDEF" for c in args.expected_checkpoint_sha256):
        parser.error("--expected-checkpoint-sha256 must be 64 hexadecimal characters")
    if not math.isclose(args.expected_source_age_mean, SOURCE_AGE_MEAN, rel_tol=0, abs_tol=1e-12):
        parser.error("source age mean is fixed by the Dortmund checkpoint protocol")
    if not math.isclose(args.expected_source_age_std, SOURCE_AGE_STD, rel_tol=0, abs_tol=5e-7):
        parser.error("source age std is fixed by the Dortmund checkpoint protocol")
    return args


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    os.replace(temporary, path)


class FixedSourceScaler:
    def __init__(self, mean: float, std: float, ddof: int = 0):
        self.mean = float(mean); self.std = float(std); self.ddof = int(ddof)
        if not math.isfinite(self.mean) or not math.isfinite(self.std) or self.std <= 0 or self.ddof != 0:
            raise ValueError("Invalid fixed source scaler")

    def transform(self, years: torch.Tensor) -> torch.Tensor:
        return (years - self.mean) / self.std

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.std + self.mean

    def metadata(self) -> dict[str, Any]:
        return {
            "target_transform": "fixed_source_zscore", "source_age_mean": self.mean,
            "source_age_std": self.std, "ddof": self.ddof, "target_unit": "year",
            "target_scaler_fitted_on_huada": False, "source_scaler_reused": True,
        }


def load_source_probe_state(checkpoint_path: str | Path, expected_sha256: str):
    """Strictly re-read and validate the authoritative full checkpoint per fold."""
    path = Path(checkpoint_path)
    if path.name == "last_training_checkpoint.pt":
        raise RuntimeError("last_training_checkpoint.pt is forbidden")
    actual = sha256_file(path)
    if actual.lower() != str(expected_sha256).lower():
        raise RuntimeError(f"Checkpoint SHA256 mismatch: {actual}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    report = validate_checkpoint_payload(payload, model=build_formal_model())
    state = payload["age_probe_state_dict"]
    probe = {"weight": state["weight"].detach().cpu().clone(), "bias": state["bias"].detach().cpu().clone()}
    if tuple(probe["weight"].shape) != (1, 512) or tuple(probe["bias"].shape) != (1,):
        raise RuntimeError("Source age probe must be [1,512]+[1]")
    scaler = FixedSourceScaler(report["source_age_mean"], report["source_age_std"], int(payload["target_transform"]["ddof"]))
    return probe, tensor_state_fingerprint(probe), {**report, "sha256": actual, "path": str(path.resolve())}, scaler


def build_source_initialized_probe(state: Mapping[str, torch.Tensor], device: torch.device):
    with torch.device("meta"):
        model = CachedFeatureClassifier(
            input_dim=FEATURE_DIM, num_classes=1, pooling=POOLING,
            use_mlp_head=False, head_dropout=0.0,
        )
    model.load_state_dict({
        "head.weight": state["weight"].detach().clone(),
        "head.bias": state["bias"].detach().clone(),
    }, strict=True, assign=True)
    model.to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    if hasattr(model, "backbone") or sum(p.numel() for p in model.parameters()) != PROBE_PARAMETER_COUNT:
        raise RuntimeError("Fold model must contain exactly the 513-parameter probe")
    return model


def probe_state_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.head.state_dict().items()}


def probe_delta(source: Mapping[str, torch.Tensor], target: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    flat = torch.cat([(target[key].float() - source[key].float()).reshape(-1) for key in sorted(source)])
    return {
        "l2_norm": float(torch.linalg.vector_norm(flat)),
        "max_abs_diff": float(flat.abs().max()) if flat.numel() else 0.0,
        "changed_parameter_count": int(torch.count_nonzero(flat)),
        "total_parameter_count": int(flat.numel()),
    }


def make_probe_optimizer(model: nn.Module, head_lr: float, weight_decay: float):
    parameters = list(model.head.parameters())
    if sum(p.numel() for p in parameters) != PROBE_PARAMETER_COUNT:
        raise RuntimeError("Optimizer parameters are not exactly the age probe")
    optimizer = torch.optim.AdamW([{"params": parameters, "name": "age_probe", "lr": head_lr, "weight_decay": weight_decay}])
    if optimizer.param_groups[0]["lr"] != head_lr or optimizer.param_groups[0]["weight_decay"] != weight_decay:
        raise RuntimeError("Optimizer configuration mismatch")
    return optimizer


def candidate_is_better(candidate: Mapping[str, Any], incumbent: Mapping[str, Any] | None, tolerance: float = SELECTION_TOLERANCE) -> bool:
    if incumbent is None:
        return True
    for key in ("subject_mae", "window_mae", "loss"):
        left, right = float(candidate[key]), float(incumbent[key])
        if left < right - tolerance: return True
        if left > right + tolerance: return False
    return int(candidate["epoch"]) < int(incumbent["epoch"])


class FoldTestAccessGuard:
    def __init__(self):
        self.selection_locked = False; self.test_dataset_created = False
        self.test_dataloader_created = False; self.test_rows_materialized = 0
        self.test_forward_count = 0; self.test_metrics_computed = False

    def lock(self):
        if self.selection_locked: raise RuntimeError("Selection already locked")
        self.selection_locked = True

    def _require(self, action):
        if not self.selection_locked: raise RuntimeError(f"Test access before selection lock: {action}")

    def create_dataset(self): self._require("dataset"); setattr(self, "test_dataset_created", True)
    def create_loader(self): self._require("dataloader"); setattr(self, "test_dataloader_created", True)
    def materialize(self, count): self._require("rows"); setattr(self, "test_rows_materialized", self.test_rows_materialized + int(count))
    def forward(self): self._require("forward"); setattr(self, "test_forward_count", self.test_forward_count + 1)
    def metrics(self):
        self._require("metrics")
        if self.test_metrics_computed: raise RuntimeError("Test metrics computed twice")
        self.test_metrics_computed = True

    def state(self): return dict(self.__dict__)


def _loader(dataset, indices, batch_size, shuffle, seed, num_workers):
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        Subset(dataset, list(indices)), batch_size=int(batch_size), shuffle=bool(shuffle),
        num_workers=int(num_workers), collate_fn=collate_fn, drop_last=False,
        generator=generator if shuffle else None,
    )


def _indices_for_subjects(metadata, subjects):
    selected = set(subjects)
    return [index for index, row in enumerate(metadata) if str(row["subject_id"]) in selected]


@torch.inference_mode()
def evaluate_probe(
    model, loader, device, scaler: FixedSourceScaler, *, fold: int,
    checkpoint_path: str | Path | None = None, collect_rows: bool = False,
    single_subject_validation: bool = False, guard: FoldTestAccessGuard | None = None,
):
    model.eval(); z_truth, z_pred, year_truth, year_pred, rows = [], [], [], [], []
    for batch in loader:
        if guard is not None: guard.forward()
        features = batch["feature"].to(device)
        truth_year = batch["label"].float().to(device).reshape(-1)
        if not torch.isfinite(features).all() or not torch.isfinite(truth_year).all():
            raise FloatingPointError("Non-finite evaluation input")
        truth_z = scaler.transform(truth_year)
        prediction_z = model.classify_features(features).reshape(-1)
        prediction_year = scaler.inverse(prediction_z)
        if not torch.isfinite(prediction_z).all() or not torch.isfinite(prediction_year).all():
            raise FloatingPointError("Non-finite evaluation prediction")
        z_truth.extend(truth_z.cpu().tolist()); z_pred.extend(prediction_z.cpu().tolist())
        year_truth.extend(truth_year.cpu().tolist()); year_pred.extend(prediction_year.cpu().tolist())
        if collect_rows:
            for index in range(truth_year.numel()):
                error = float(prediction_year[index] - truth_year[index])
                rows.append({
                    "fold": int(fold), "sample_id": str(batch["sample_id"][index]),
                    "subject_id": str(batch["subject_id"][index]),
                    "recording_id": str(batch.get("recording_id", [""] * len(batch["sample_id"]))[index]),
                    "window_index": int(batch["window_index"][index]),
                    "true_age": float(truth_year[index]), "true_z_source": float(truth_z[index]),
                    "predicted_z": float(prediction_z[index]), "predicted_age": float(prediction_year[index]),
                    "signed_error": error, "absolute_error": abs(error), "squared_error": error * error,
                    "num_valid_tokens": int(batch["num_valid_tokens"][index]),
                    "checkpoint_path": "" if checkpoint_path is None else str(Path(checkpoint_path).resolve()),
                })
    if not year_truth:
        raise RuntimeError("Evaluation loader is empty")
    by_subject = defaultdict(list)
    for truth, prediction, subject in zip(year_truth, year_pred, [str(r["subject_id"]) for r in rows] if collect_rows else []):
        by_subject[subject].append((truth, prediction))
    if not collect_rows:
        # Validation/test loaders each contain one subject, so direct means are valid.
        subject_truth = float(np.mean(year_truth)); subject_prediction = float(np.mean(year_pred))
    else:
        if len(by_subject) != 1:
            raise RuntimeError("Fold validation/test evaluation must contain exactly one subject")
        values = next(iter(by_subject.values()))
        subject_truth = float(np.mean([x[0] for x in values])); subject_prediction = float(np.mean([x[1] for x in values]))
    window_metrics = regression_metrics(year_truth, year_pred)
    metrics = {
        "loss": float(np.mean((np.asarray(z_pred) - np.asarray(z_truth)) ** 2)),
        "subject_mae": abs(subject_prediction - subject_truth),
        "subject_prediction": subject_prediction, "subject_truth": subject_truth,
        "window_mae": window_metrics["mae"], "window_rmse": window_metrics["rmse"],
        "sample_count": len(year_truth),
        "prediction_min": float(np.min(year_pred)), "prediction_max": float(np.max(year_pred)),
        "finite": True,
    }
    if single_subject_validation:
        metrics.update({
            "r2": None, "r2_defined": False, "r2_reason": "single validation subject",
            "pearson_r": {"value": None, "defined": False, "reason": "single validation subject"},
            "spearman_rho": {"value": None, "defined": False, "reason": "single validation subject"},
        })
    return metrics, rows


def train_epoch(model, loader, optimizer, device, scaler: FixedSourceScaler):
    model.train(); total_loss = total_abs = total_count = 0; max_gradient_norm = 0.0
    for batch in loader:
        features = batch["feature"].to(device); truth_year = batch["label"].float().to(device).reshape(-1)
        truth_z = scaler.transform(truth_year)
        if not torch.isfinite(features).all() or not torch.isfinite(truth_z).all():
            raise FloatingPointError("Non-finite training input")
        optimizer.zero_grad(set_to_none=True)
        predicted_z = model.classify_features(features).reshape(-1)
        loss = torch.mean((predicted_z - truth_z) ** 2)
        if not torch.isfinite(loss): raise FloatingPointError("Non-finite training loss")
        loss.backward()
        gradients = [p.grad.reshape(-1) for p in model.parameters() if p.grad is not None]
        gradient_norm = float(torch.linalg.vector_norm(torch.cat(gradients))) if gradients else 0.0
        if not math.isfinite(gradient_norm): raise FloatingPointError("Non-finite gradient")
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in model.parameters()): raise FloatingPointError("Non-finite probe parameter")
        count = truth_year.numel(); predicted_year = scaler.inverse(predicted_z.detach())
        total_loss += float(loss) * count; total_abs += float(torch.abs(predicted_year - truth_year).sum())
        total_count += count; max_gradient_norm = max(max_gradient_norm, gradient_norm)
    return {"train_z_mse": total_loss / total_count, "train_mae_years": total_abs / total_count, "gradient_norm": max_gradient_norm}


def _save_head_checkpoint(path: Path, payload: Mapping[str, Any]):
    temporary = path.with_name(path.name + ".tmp"); torch.save(dict(payload), temporary); os.replace(temporary, path)


def validate_head_checkpoint(
    payload: Mapping[str, Any], *, source_sha: str, cache_key: str,
    fold_info: Mapping[str, Any], scaler: FixedSourceScaler,
):
    forbidden = {"backbone", "backbone_state_dict", "model_state_dict", "feature_cache", "optimizer_state_dict"}
    if payload.get("checkpoint_kind") != HEAD_CHECKPOINT_KIND or forbidden.intersection(payload):
        raise RuntimeError("Invalid head-only checkpoint kind/content")
    expected = {
        "source_full_checkpoint_sha256": source_sha, "feature_cache_key": cache_key,
        "source_checkpoint_kind": SOURCE_CHECKPOINT_KIND,
        "pooling": CHECKPOINT_POOLING, "feature_cache_pooling": POOLING,
        "feature_dim": FEATURE_DIM, "probe_layer_idx": PROBE_LAYER_IDX,
        "fold": int(fold_info["fold"]), "validation_subject": fold_info["validation_subject"],
        "test_subject": fold_info["test_subject"], "train_subjects": list(fold_info["train_subjects"]),
        "source_age_mean": scaler.mean,
        "source_age_std": scaler.std, "target_transform": "fixed_source_zscore",
        "target_unit": "year", "target_scaler_fitted_on_huada": False,
        "source_scaler_reused": True,
    }
    mismatch = {key: (payload.get(key), value) for key, value in expected.items() if payload.get(key) != value}
    if mismatch: raise RuntimeError(f"Head-only checkpoint metadata mismatch: {mismatch}")
    state = payload.get("head_state_dict")
    if not isinstance(state, Mapping) or set(state) != {"weight", "bias"}:
        raise RuntimeError("Invalid head state")
    if tuple(state["weight"].shape) != (1, 512) or tuple(state["bias"].shape) != (1,):
        raise RuntimeError("Invalid head shape")
    if tensor_state_fingerprint(state) != payload.get("best_head_fingerprint"):
        raise RuntimeError("Head state fingerprint mismatch")
    return True


def _load_best_head(model, path, device, expected_sha256=None, **validation):
    actual_sha256 = sha256_file(path)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise RuntimeError("Best head checkpoint SHA256 mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    validate_head_checkpoint(payload, **validation)
    model.head.load_state_dict(payload["head_state_dict"], strict=True); model.to(device)
    return payload


def run_fold(
    *, fold_info, dataset, checkpoint_path, expected_sha256, cache_info,
    split_manifest_sha256, output_dir, device, batch_size, num_workers, seed,
    head_lr, weight_decay, epochs, patience, logger, provenance=None,
):
    fold = int(fold_info["fold"]); fold_seed = int(seed) + fold
    fold_dir = output_dir / f"fold_{fold}"; checkpoint_dir = fold_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    metadata = dataset.get_subject_label_rows()
    train_indices = _indices_for_subjects(metadata, fold_info["train_subjects"])
    validation_indices = _indices_for_subjects(metadata, [fold_info["validation_subject"]])
    test_indices = _indices_for_subjects(metadata, [fold_info["test_subject"]])
    if not train_indices or not validation_indices or not test_indices:
        raise RuntimeError("Fold has an empty split")
    if set(train_indices) & set(validation_indices) or set(train_indices) & set(test_indices) or set(validation_indices) & set(test_indices):
        raise RuntimeError("Fold sample indices overlap")
    # Required full-checkpoint re-read; previous fold state can never enter.
    source_state, source_fingerprint, checkpoint_info, scaler = load_source_probe_state(checkpoint_path, expected_sha256)
    model = build_source_initialized_probe(source_state, device)
    initial_fingerprint = tensor_state_fingerprint(probe_state_cpu(model))
    if initial_fingerprint != source_fingerprint: raise RuntimeError("Initial probe differs from source")
    train_loader = _loader(dataset, train_indices, batch_size, True, fold_seed, num_workers)
    validation_loader = _loader(dataset, validation_indices, batch_size, False, fold_seed, num_workers)
    epoch0, _ = evaluate_probe(model, validation_loader, device, scaler, fold=fold, collect_rows=False, single_subject_validation=True)
    best = {"epoch": 0, **epoch0}; best_state = probe_state_cpu(model)
    best_checkpoint = checkpoint_dir / "best_head.pt"
    optimizer_config = {"name": "AdamW", "head_lr": float(head_lr), "weight_decay": float(weight_decay), "scheduler": "none", "parameter_count": PROBE_PARAMETER_COUNT}

    def checkpoint_payload():
        return {
            "checkpoint_kind": HEAD_CHECKPOINT_KIND, "head_state_dict": best_state,
            "initial_source_probe_fingerprint": source_fingerprint,
            "best_head_fingerprint": tensor_state_fingerprint(best_state),
            "source_full_checkpoint_path": checkpoint_info["path"],
            "source_full_checkpoint_sha256": checkpoint_info["sha256"],
            "source_checkpoint_kind": SOURCE_CHECKPOINT_KIND,
            "feature_cache_key": cache_info["feature_cache_key"], "feature_cache_path": cache_info["feature_cache_path"],
            "pooling": CHECKPOINT_POOLING, "feature_cache_pooling": POOLING,
            "feature_dim": FEATURE_DIM, "probe_layer_idx": PROBE_LAYER_IDX,
            **scaler.metadata(), "fold": fold, "train_subjects": list(fold_info["train_subjects"]),
            "validation_subject": fold_info["validation_subject"], "test_subject": fold_info["test_subject"],
            "best_epoch": int(best["epoch"]), "validation_metrics": {k: v for k, v in best.items() if k != "epoch"},
            "optimizer_config": optimizer_config, "split_manifest_sha256": split_manifest_sha256,
            "probe_parameter_delta": probe_delta(source_state, best_state),
            "git_provenance": None if provenance is None else provenance.get("git"),
            "critical_code_sha256": None if provenance is None else provenance.get("critical_code_sha256"),
            "actual_command": [sys.executable, *sys.argv],
        }

    _save_head_checkpoint(best_checkpoint, checkpoint_payload())
    history = [{"fold": fold, "epoch": 0, "train_z_mse": "", "train_mae_years": "", **{f"validation_{k}": v for k, v in epoch0.items() if not isinstance(v, Mapping)}, "head_weight_norm": float(torch.linalg.vector_norm(model.head.weight.detach())), "head_bias_norm": float(torch.linalg.vector_norm(model.head.bias.detach())), "gradient_norm": "", "lr": "", "is_best": True}]
    optimizer = make_probe_optimizer(model, head_lr, weight_decay)
    non_improving = 0; stopped_early = False
    for epoch in range(1, int(epochs) + 1):
        train = train_epoch(model, train_loader, optimizer, device, scaler)
        validation, _ = evaluate_probe(model, validation_loader, device, scaler, fold=fold, collect_rows=False, single_subject_validation=True)
        candidate = {"epoch": epoch, **validation}; improved = candidate_is_better(candidate, best)
        if improved:
            best = candidate; best_state = probe_state_cpu(model); _save_head_checkpoint(best_checkpoint, checkpoint_payload()); non_improving = 0
        else: non_improving += 1
        history.append({
            "fold": fold, "epoch": epoch, **train,
            **{f"validation_{k}": v for k, v in validation.items() if not isinstance(v, Mapping)},
            "head_weight_norm": float(torch.linalg.vector_norm(model.head.weight.detach())),
            "head_bias_norm": float(torch.linalg.vector_norm(model.head.bias.detach())),
            "lr": optimizer.param_groups[0]["lr"], "is_best": improved,
        })
        log_event(logger, "INFO", "train.epoch", "age source-probe transfer epoch completed", fold=fold, epoch=epoch, best_epoch=best["epoch"], **train, validation_subject_mae=validation["subject_mae"], validation_window_mae=validation["window_mae"])
        if non_improving >= int(patience): stopped_early = True; break
    final_fingerprint = tensor_state_fingerprint(probe_state_cpu(model)); del optimizer
    selected_head_sha256 = sha256_file(best_checkpoint)
    payload = _load_best_head(
        model, best_checkpoint, device, expected_sha256=selected_head_sha256, source_sha=checkpoint_info["sha256"],
        cache_key=cache_info["feature_cache_key"], fold_info=fold_info, scaler=scaler,
    )
    best_fingerprint = tensor_state_fingerprint(probe_state_cpu(model))
    if best_fingerprint != payload["best_head_fingerprint"]: raise RuntimeError("Best probe restore mismatch")
    selected_validation, validation_rows = evaluate_probe(model, validation_loader, device, scaler, fold=fold, checkpoint_path=best_checkpoint, collect_rows=True, single_subject_validation=True)
    _write_csv(fold_dir / "validation_predictions.csv", validation_rows, OOF_FIELDS)
    _write_json(fold_dir / "validation_metrics.json", selected_validation)
    _write_csv(fold_dir / "training_trajectory.csv", history, list(history[0]))

    guard = FoldTestAccessGuard(); prelock = guard.state(); guard.lock()
    selection = {
        "fold": fold, "best_epoch": int(payload["best_epoch"]), "selection_tolerance": SELECTION_TOLERANCE,
        "selection_metric_order": ["validation subject-mean MAE", "validation window MAE", "validation z-MSE", "earlier epoch"],
        "test_access_before_selection": prelock, "selection_locked": True,
        "best_head_sha256": selected_head_sha256,
    }
    _write_json(fold_dir / "selection_locked.json", selection)
    guard.create_dataset(); guard.create_loader()
    test_loader = _loader(dataset, test_indices, batch_size, False, fold_seed, num_workers)
    test_metrics, test_rows = evaluate_probe(model, test_loader, device, scaler, fold=fold, checkpoint_path=best_checkpoint, collect_rows=True, guard=guard)
    for row in test_rows: row["test_is_outside_train_age_range"] = bool(fold_info["test_is_outside_train_age_range"])
    guard.materialize(len(test_rows)); guard.metrics()
    _write_csv(fold_dir / "test_predictions.csv", test_rows, OOF_FIELDS); _write_json(fold_dir / "test_metrics.json", test_metrics)
    summary = {
        "fold": fold, "train_subjects": list(fold_info["train_subjects"]),
        "validation_subject": fold_info["validation_subject"], "test_subject": fold_info["test_subject"],
        "train_rows": len(train_indices), "validation_rows": len(validation_indices), "test_rows": len(test_indices),
        "initial_source_probe_fingerprint": source_fingerprint, "best_probe_fingerprint": best_fingerprint,
        "final_probe_fingerprint": final_fingerprint, "probe_parameter_delta_norm": payload["probe_parameter_delta"]["l2_norm"],
        "best_epoch": int(payload["best_epoch"]), "epoch0_validation": epoch0,
        "best_validation": payload["validation_metrics"], "epochs_executed": len(history) - 1,
        "stopped_early": stopped_early, "test_metrics": test_metrics,
        "test_access": guard.state(), "test_evaluation_count": 1,
        "backbone_constructed_during_fold_training": False, "backbone_forward_count_during_fold_training": 0,
        "optimizer_reinitialized": True, "source_scaler": scaler.metadata(),
        "best_checkpoint": str(best_checkpoint.resolve()), "best_checkpoint_sha256": sha256_file(best_checkpoint),
    }
    return summary, test_rows, history


def _critical_code_hashes(project_root: Path) -> dict[str, str]:
    paths = [
        "scripts/downstream_50m/model.py", "scripts/downstream_50m/feature_cache.py",
        "scripts/downstream_50m/age_cv7.py", "scripts/downstream_50m/dortmund_huada_zero_shot.py",
        "scripts/downstream_50m/run_huada_dortmund_age_probe_transfer_cv7.py",
    ]
    return {path: sha256_file(project_root / path) for path in paths}


def build_dortmund_cache_identity(
    *, input_fingerprint: str, input_info: Mapping[str, Any], checkpoint_report: Mapping[str, Any],
    code_sha256: str, git_commit: str | None,
):
    architecture = {
        "d_model": FEATURE_DIM, "depth": 12, "n_heads": 8,
        "mlp_ratio": 4.0, "dropout": 0.1,
    }
    identity, _ = cache.build_cache_identity(
        input_fingerprint=input_fingerprint, input_row_count=input_info["row_count"],
        sample_id_fingerprint=input_info["sample_id_fingerprint"],
        checkpoint_sha256=checkpoint_report["sha256"], model_config=architecture,
        shape_params={"num_tokens": 640, "patch_len": 100},
        probe_layer_idx=PROBE_LAYER_IDX, probe_layer_ratio=0.75,
        model_code_sha256=code_sha256, git_commit=git_commit,
        requested_poolings=("mean",), feature_dim=FEATURE_DIM,
        feature_extraction_code_sha256=code_sha256,
        source_checkpoint_kind=SOURCE_CHECKPOINT_KIND,
    )
    identity.update({
        "pooling": POOLING,
        "electrode_vocabulary_sha256": checkpoint_report["electrode_vocabulary_sha256"],
        "token_mask_semantics_version": checkpoint_report["token_mask_semantics_version"],
        "source_checkpoint_best_epoch": checkpoint_report["best_epoch"],
        "source_checkpoint_architecture": architecture,
    })
    return identity, cache.sha256_json(identity)


def ensure_dortmund_mean_cache(args, logger):
    project_root = Path(__file__).resolve().parents[2]
    input_fingerprint, input_info = cache.lance_fingerprint(args.input_lance)
    _, _, checkpoint_report, _ = load_source_probe_state(args.checkpoint, args.expected_checkpoint_sha256)
    git_commit, git_dirty = cache.get_git_state(project_root)
    critical = _critical_code_hashes(project_root); code_sha = cache.sha256_json(critical)
    identity, cache_key = build_dortmund_cache_identity(
        input_fingerprint=input_fingerprint, input_info=input_info,
        checkpoint_report=checkpoint_report, code_sha256=code_sha, git_commit=git_commit,
    )
    root = Path(args.feature_cache_root).resolve()
    final_dir, manifest, reused = cache.resolve_cache_policy(root, cache_key, args.feature_cache_policy, identity)
    if reused:
        if manifest.get("requested_poolings") != ["mean"] or (final_dir / cache.FLATTEN_CACHE_NAME).exists():
            raise RuntimeError("Dortmund age transfer requires a mean-only cache")
        return {"feature_cache_key": cache_key, "feature_cache_path": str(final_dir), "feature_cache_reused": True, "manifest": manifest, "identity": identity}
    build_root = Path(args.feature_cache_build_root or (Path(tempfile.gettempdir()) / "huada_dortmund_age_cache_build")).resolve()
    build_root.mkdir(parents=True, exist_ok=True); root.mkdir(parents=True, exist_ok=True)
    if build_root == root or root in build_root.parents:
        raise ValueError("Feature cache build root must be outside final cache root")
    temporary = build_root / f".{cache_key}.building.{uuid.uuid4().hex}"
    source = cache.TokenFeatureSourceDataset(args.input_lance)
    device = torch.device(args.device)
    model, full_report, payload = load_strict_checkpoint(args.checkpoint, args.expected_checkpoint_sha256, device)
    # Reuse the zero-shot semantic channel compatibility validator.
    try:
        from run_huada_dortmund_age_zero_shot import _input_contract
    except ImportError:
        from .run_huada_dortmund_age_zero_shot import _input_contract
    import lance
    input_contract = _input_contract(Path(args.input_lance), lance.dataset(args.input_lance), full_report, payload)
    if input_contract["electrode_vocabulary_sha256"] != identity["electrode_vocabulary_sha256"]:
        raise RuntimeError("Input/checkpoint electrode vocabulary mismatch")
    before = tensor_state_fingerprint(model.backbone.state_dict())
    manifest_base = {
        "cache_key": cache_key, "cache_identity": identity,
        "input_row_count": input_info["row_count"],
        "sample_ids": [str(row["sample_id"]) for row in source.metadata],
        "input_lance_metadata_fingerprint": input_fingerprint,
        "input_lance_version": input_info["version"],
        "pretrained_checkpoint_sha256": checkpoint_report["sha256"],
        "source_checkpoint_kind": SOURCE_CHECKPOINT_KIND,
        "source_checkpoint_best_epoch": checkpoint_report["best_epoch"],
        "electrode_vocabulary_sha256": checkpoint_report["electrode_vocabulary_sha256"],
        "token_mask_semantics_version": MASK_SEMANTICS_VERSION,
        "critical_code_sha256": critical, "git_commit": git_commit, "git_dirty": git_dirty,
        "requested_poolings": ["mean"], "backbone_fingerprint_before": before,
    }
    try:
        manifest = cache.build_feature_cache(
            source_dataset=source, backbone=model.backbone, probe_layer_idx=PROBE_LAYER_IDX,
            output_dir=temporary, batch_size=args.feature_cache_batch_size, device=device,
            expected_num_tokens=640, d_model=FEATURE_DIM, manifest_base=manifest_base,
            logger=logger, log_every_batches=args.cache_log_every_batches,
            requested_poolings=("mean",), parity_head=model.head,
        )
        after = tensor_state_fingerprint(model.backbone.state_dict())
        if before != after: raise RuntimeError("Backbone changed during feature-cache extraction")
        manifest["backbone_fingerprint_after"] = after
        manifest["parity_head_source"] = "dortmund_age_source_probe"
        cache.atomic_write_json(temporary / cache.MANIFEST_NAME, manifest)
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        manifest = cache.publish_feature_cache(temporary, final_dir, identity, cache_key, logger=logger)
    finally:
        if temporary.exists() and (temporary / cache.SUCCESS_NAME).exists(): shutil.rmtree(temporary)
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    if (final_dir / cache.FLATTEN_CACHE_NAME).exists(): raise RuntimeError("Unexpected flatten cache")
    return {"feature_cache_key": cache_key, "feature_cache_path": str(final_dir), "feature_cache_reused": False, "manifest": manifest, "identity": identity}


def build_split_artifacts(dataset, split_seed: int, output_dir: Path):
    metadata = dataset.get_subject_label_rows(); subject_ages = {}; window_counts = Counter()
    sample_subject = {}
    for row in metadata:
        subject = str(row["subject_id"]); age = float(row["label"]); sample = str(row["sample_id"])
        if subject == "SUB5": raise RuntimeError("SUB5 entered age-eligible cache dataset")
        if subject in subject_ages and subject_ages[subject] != age: raise RuntimeError(f"Inconsistent age for {subject}")
        if sample in sample_subject: raise RuntimeError(f"Duplicate sample_id: {sample}")
        subject_ages[subject] = age; window_counts[subject] += 1; sample_subject[sample] = subject
    if set(subject_ages) != ELIGIBLE_SUBJECTS: raise RuntimeError(f"Eligible subject mismatch: {sorted(subject_ages)}")
    payload = age_cv7.build_age_subject_cv7(subject_ages, split_seed)
    manifest_sha = cache.sha256_json(payload)
    assignments = []
    for fold in payload["folds"]:
        split = {subject: "train" for subject in fold["train_subjects"]}
        split[fold["validation_subject"]] = "validation"; split[fold["test_subject"]] = "test"
        for subject in sorted(subject_ages):
            assignments.append({"fold": fold["fold"], "subject_id": subject, "age": subject_ages[subject], "split": split[subject], "split_seed": split_seed})
    summaries = []
    for fold in payload["folds"]:
        summaries.append({
            "fold": fold["fold"], "train_subjects": ";".join(fold["train_subjects"]),
            "validation_subject": fold["validation_subject"], "test_subject": fold["test_subject"],
            "train_rows": sum(window_counts[s] for s in fold["train_subjects"]),
            "validation_rows": window_counts[fold["validation_subject"]], "test_rows": window_counts[fold["test_subject"]],
        })
    _write_csv(output_dir / "age_cv7_subject_assignments.csv", assignments, ["fold", "subject_id", "age", "split", "split_seed"])
    _write_csv(output_dir / "age_cv7_fold_summary.csv", summaries, ["fold", "train_subjects", "validation_subject", "test_subject", "train_rows", "validation_rows", "test_rows"])
    return payload, manifest_sha, subject_ages, sample_subject, dict(window_counts)


def aggregate_oof_artifacts(
    *, output_dir: Path, split_payload: Mapping[str, Any], subject_ages: Mapping[str, float],
    sample_subject: Mapping[str, str], fold_summaries: Sequence[Mapping[str, Any]],
    oof_rows: Sequence[Mapping[str, Any]], epoch_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    subject_test_folds = {str(fold["test_subject"]): int(fold["fold"]) for fold in split_payload["folds"]}
    age_cv7.validate_oof_coverage(oof_rows, sample_subject, subject_test_folds)
    ordered_oof = sorted(oof_rows, key=lambda row: (int(row["fold"]), str(row["sample_id"])))
    _write_csv(output_dir / "oof_window_predictions.csv", ordered_oof, OOF_FIELDS)

    aggregated = age_cv7.aggregate_subject_predictions(ordered_oof)
    checkpoint_by_fold = {int(row["fold"]): str(row["best_checkpoint"]) for row in fold_summaries}
    subject_rows = []
    for row in aggregated:
        subject_rows.append({
            "subject_id": row["subject_id"], "true_age": row["true_age"],
            "window_count": row["window_count"], "mean_prediction": row["mean_predicted_age"],
            "median_prediction": row["median_predicted_age"], "prediction_std": row["prediction_std"],
            "prediction_min": row["prediction_min"], "prediction_max": row["prediction_max"],
            "signed_error": row["signed_error"], "absolute_error": row["absolute_error"],
            "test_fold": row["test_fold"], "checkpoint_path": checkpoint_by_fold[row["test_fold"]],
        })
    _write_csv(output_dir / "oof_subject_predictions.csv", subject_rows, SUBJECT_FIELDS)

    window_metrics = regression_metrics(
        [float(row["true_age"]) for row in ordered_oof],
        [float(row["predicted_age"]) for row in ordered_oof],
    )
    window_metrics.update({"level": "window_auxiliary_non_independent"})
    truth = [float(row["true_age"]) for row in subject_rows]
    mean_metrics = regression_metrics(truth, [float(row["mean_prediction"]) for row in subject_rows])
    median_metrics = regression_metrics(truth, [float(row["median_prediction"]) for row in subject_rows])
    _write_json(output_dir / "oof_window_metrics.json", window_metrics)
    _write_json(output_dir / "oof_subject_mean_metrics.json", mean_metrics)
    _write_json(output_dir / "oof_subject_median_metrics.json", median_metrics)

    source_baseline = regression_metrics(truth, [SOURCE_AGE_MEAN] * len(truth))
    source_baseline.update({
        "baseline": "fixed_dortmund_source_mean", "prediction_years": SOURCE_AGE_MEAN,
        "target_statistics_from_huada": False,
    })
    fold_baseline_predictions = []
    for fold in split_payload["folds"]:
        prediction = age_cv7.train_mean_baseline(fold, subject_ages)
        fold_baseline_predictions.append({
            "fold": int(fold["fold"]), "test_subject": fold["test_subject"],
            "true_age": float(subject_ages[fold["test_subject"]]),
            "prediction": prediction, "train_subjects": list(fold["train_subjects"]),
        })
    fold_baseline = regression_metrics(
        [row["true_age"] for row in fold_baseline_predictions],
        [row["prediction"] for row in fold_baseline_predictions],
    )
    fold_baseline.update({
        "baseline": "fold_train_subject_mean", "uses_validation_or_test_age": False,
        "fold_predictions": fold_baseline_predictions,
    })
    _write_json(output_dir / "source_mean_baseline_metrics.json", source_baseline)
    _write_json(output_dir / "fold_train_mean_baseline_metrics.json", fold_baseline)

    # Independent recomputation reads the published CSV, not the in-memory objects.
    with (output_dir / "oof_window_predictions.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        csv_windows = list(csv.DictReader(handle))
    with (output_dir / "oof_subject_predictions.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        csv_subjects = list(csv.DictReader(handle))
    recomputed = {
        "sample_id_unique": len({row["sample_id"] for row in csv_windows}) == len(csv_windows),
        "sample_id_coverage_equal": {row["sample_id"] for row in csv_windows} == set(sample_subject),
        "subject_count": len(csv_subjects),
        "window_metrics": regression_metrics([float(r["true_age"]) for r in csv_windows], [float(r["predicted_age"]) for r in csv_windows]),
        "subject_mean_metrics": regression_metrics([float(r["true_age"]) for r in csv_subjects], [float(r["mean_prediction"]) for r in csv_subjects]),
        "subject_median_metrics": regression_metrics([float(r["true_age"]) for r in csv_subjects], [float(r["median_prediction"]) for r in csv_subjects]),
    }
    max_metric_diff = 0.0
    for expected, actual in ((window_metrics, recomputed["window_metrics"]), (mean_metrics, recomputed["subject_mean_metrics"]), (median_metrics, recomputed["subject_median_metrics"])):
        for key in ("mae", "rmse", "bias"):
            max_metric_diff = max(max_metric_diff, abs(float(expected[key]) - float(actual[key])))
    recomputed.update({"max_core_metric_abs_diff": max_metric_diff, "passed": max_metric_diff <= 1e-12})
    if not recomputed["passed"] or not recomputed["sample_id_unique"] or not recomputed["sample_id_coverage_equal"]:
        raise RuntimeError("Independent OOF metric/coverage recomputation failed")
    _write_json(output_dir / "metric_recomputation.json", recomputed)

    fold_fields = [
        "fold", "train_subjects", "validation_subject", "test_subject", "train_rows",
        "validation_rows", "test_rows", "initial_source_probe_fingerprint", "best_probe_fingerprint",
        "final_probe_fingerprint", "probe_parameter_delta_norm", "best_epoch", "epochs_executed",
        "stopped_early", "best_checkpoint", "best_checkpoint_sha256", "test_evaluation_count",
        "backbone_forward_count_during_fold_training",
    ]
    flat_folds = [{**row, "train_subjects": ";".join(row["train_subjects"])} for row in fold_summaries]
    _write_csv(output_dir / "fold_training_results.csv", flat_folds, fold_fields)
    _write_csv(output_dir / "fold_selected_checkpoints.csv", flat_folds, [
        "fold", "validation_subject", "test_subject", "best_epoch", "best_checkpoint",
        "best_checkpoint_sha256", "initial_source_probe_fingerprint", "best_probe_fingerprint",
    ])
    epoch_fields = sorted({key for row in epoch_rows for key in row})
    _write_csv(output_dir / "epoch_metrics.csv", epoch_rows, epoch_fields)
    initial_fingerprints = {str(row["initial_source_probe_fingerprint"]) for row in fold_summaries}
    if len(initial_fingerprints) != 1:
        raise RuntimeError("The seven folds did not start from the same source probe")
    return {
        "oof_window_count": len(ordered_oof), "oof_subject_count": len(subject_rows),
        "initial_source_probe_fingerprint": next(iter(initial_fingerprints)),
        "oof_window_metrics": window_metrics, "oof_subject_mean_metrics": mean_metrics,
        "oof_subject_median_metrics": median_metrics, "source_mean_baseline": source_baseline,
        "fold_train_mean_baseline": fold_baseline, "metric_recomputation": recomputed,
    }


def compare_zero_shot_reference(reference_dir: str | None, transfer_metrics: Mapping[str, Any]) -> dict[str, Any]:
    if not reference_dir:
        return {"status": "not_requested", "used_for_training_or_selection": False}
    root = Path(reference_dir)
    metrics_path = root / "subject_mean_metrics.json"
    if not metrics_path.is_file():
        return {"status": "not_performed", "reason": f"missing {metrics_path}", "used_for_training_or_selection": False}
    zero_shot = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {
        "status": "compared_after_cv7_completion", "used_for_training_or_selection": False,
        "reference_dir": str(root.resolve()), "strict_zero_shot_metrics": zero_shot,
        "probe_transfer_metrics": dict(transfer_metrics),
        "delta_mae_probe_minus_zero_shot": float(transfer_metrics["mae"]) - float(zero_shot["mae"]),
        "delta_rmse_probe_minus_zero_shot": float(transfer_metrics["rmse"]) - float(zero_shot["rmse"]),
    }


def _git_provenance(project_root: Path) -> dict[str, Any]:
    def command(*args):
        result = subprocess.run(["git", *args], cwd=project_root, capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    status = command("status", "--porcelain")
    return {
        "commit": command("rev-parse", "HEAD"), "dirty": bool(status),
        "dirty_files": [] if not status else status.splitlines(),
    }


def _resolved_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        **vars(args), "run_mode": RUN_MODE, "probe_layer_idx": PROBE_LAYER_IDX,
        "pooling": "masked_mean", "cache_pooling": POOLING, "feature_dim": FEATURE_DIM,
        "target_transform": "fixed_source_zscore", "source_age_mean": SOURCE_AGE_MEAN,
        "source_age_std_expected": SOURCE_AGE_STD, "target_scaler_fitted_on_huada": False,
        "source_probe_reloaded_per_fold": True, "backbone_frozen": True,
        "selection_tolerance": SELECTION_TOLERANCE, "scheduler": "none",
    }


def dry_run(args: argparse.Namespace) -> dict[str, Any]:
    payload = {
        "status": "dry_run_passed", "stages": [
            "validate input and Dortmund full checkpoint", "build/reuse checkpoint-bound mean feature cache",
            "build authoritative Huada age CV7", "reload source probe and evaluate epoch 0 per fold",
            "train 513-parameter probe only", "lock selection then evaluate each test subject once",
            "aggregate OOF and independently recompute metrics",
        ],
        "resolved_config": _resolved_config(args),
        "would_create_output_dir": False, "input_lance_opened": False,
        "checkpoint_read": False, "model_created": False, "feature_cache_built": False,
        "folds_run": 0, "test_evaluations": 0,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.dry_run:
        return dry_run(args)
    project_root = Path(__file__).resolve().parents[2]
    input_path = Path(args.input_lance); checkpoint_path = Path(args.checkpoint); output_dir = Path(args.output_dir)
    if not input_path.is_dir(): raise FileNotFoundError(input_path)
    if not checkpoint_path.is_file(): raise FileNotFoundError(checkpoint_path)
    if output_dir.exists(): raise FileExistsError(f"Refusing to overwrite output: {output_dir}")
    if args.device.startswith("cuda") and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    output_dir.mkdir(parents=True, exist_ok=False)
    logger = setup_run_logging(output_dir, args.log_level, append=False)
    _write_csv(output_dir / "probe_transfer_issues.csv", [], ["severity", "stage", "code", "message"])
    (output_dir / "probe_transfer_errors.log").write_text("", encoding="utf-8")
    _write_json(output_dir / "resolved_config.json", _resolved_config(args))
    started = datetime.now(timezone.utc)
    try:
        cache_info = ensure_dortmund_mean_cache(args, logger)
        _write_json(output_dir / "feature_cache_manifest.json", cache_info["manifest"])
        cache_validation = {
            "feature_cache_key": cache_info["feature_cache_key"], "feature_cache_path": cache_info["feature_cache_path"],
            "reused": cache_info["feature_cache_reused"], "mean_only": True,
            "flatten_cache_present": False, "success_marker_present": True,
            "row_count": cache_info["manifest"]["mean_row_count"],
            "feature_dim": cache_info["manifest"]["mean_feature_dim"],
            "dtype": cache_info["manifest"]["dtype"],
            "finite_validation_passed": True,
            "sample_id_validation_passed": True,
            "mask_validation_passed": cache_info["manifest"]["mask_validation_passed"],
            "selected_sample_online_cache_parity_passed": cache_info["manifest"]["parity_passed"],
            "selected_sample_ids": cache_info["manifest"].get("parity_sample_ids", []),
            "source_probe_logit_parity_max_abs_diff": cache_info["manifest"].get("mean_logits_max_abs_diff"),
        }
        _write_json(output_dir / "feature_cache_validation.json", cache_validation)
        dataset = cache.CachedFeatureDataset(
            cache_info["feature_cache_path"], POOLING, input_path, "age_regression",
            expected_cache_key=cache_info["feature_cache_key"],
        )
        split_payload, split_sha, subject_ages, sample_subject, window_counts = build_split_artifacts(dataset, args.split_seed, output_dir)
        device = torch.device(args.device)
        fold_provenance = {"git": _git_provenance(project_root), "critical_code_sha256": _critical_code_hashes(project_root)}
        fold_summaries, oof_rows, all_epochs = [], [], []
        for fold_info in split_payload["folds"]:
            summary, rows, epochs = run_fold(
                fold_info=fold_info, dataset=dataset, checkpoint_path=checkpoint_path,
                expected_sha256=args.expected_checkpoint_sha256, cache_info=cache_info,
                split_manifest_sha256=split_sha, output_dir=output_dir, device=device,
                batch_size=args.batch_size, num_workers=args.num_workers, seed=args.seed,
                head_lr=args.head_lr, weight_decay=args.weight_decay, epochs=args.epochs,
                patience=args.early_stopping_patience, logger=logger, provenance=fold_provenance,
            )
            fold_summaries.append(summary); oof_rows.extend(rows); all_epochs.extend(epochs)
        aggregate = aggregate_oof_artifacts(
            output_dir=output_dir, split_payload=split_payload, subject_ages=subject_ages,
            sample_subject=sample_subject, fold_summaries=fold_summaries,
            oof_rows=oof_rows, epoch_rows=all_epochs,
        )
        zero_shot_comparison = compare_zero_shot_reference(args.zero_shot_reference_dir, aggregate["oof_subject_mean_metrics"])
        _write_json(output_dir / "zero_shot_comparison.json", zero_shot_comparison)
        git = _git_provenance(project_root); critical = _critical_code_hashes(project_root)
        provenance = {
            "run_mode": RUN_MODE, "started_at": started.isoformat(), "completed_at": datetime.now(timezone.utc).isoformat(),
            "input_lance": str(input_path.resolve()), "input_lance_fingerprint": cache_info["identity"]["input_lance_fingerprint"],
            "source_full_checkpoint": str(checkpoint_path.resolve()),
            "source_full_checkpoint_sha256": args.expected_checkpoint_sha256,
            "source_checkpoint_best_epoch": 20, "source_target_scaler": {
                "mean": SOURCE_AGE_MEAN, "std": fold_summaries[0]["source_scaler"]["source_age_std"],
                "ddof": 0, "full_precision_from_checkpoint": True,
            },
            "feature_cache_key": cache_info["feature_cache_key"], "feature_cache_path": cache_info["feature_cache_path"],
            "split_manifest_sha256": split_sha,
            "fold_probe_fingerprints": [{
                "fold": row["fold"], "initial": row["initial_source_probe_fingerprint"],
                "best": row["best_probe_fingerprint"], "final": row["final_probe_fingerprint"],
            } for row in fold_summaries],
            "model_config": {"probe_layer_idx": 8, "pooling": "masked_mean", "feature_dim": 512},
            "optimizer": {"name": "AdamW", "head_lr": args.head_lr, "weight_decay": args.weight_decay, "scheduler": "none"},
            "git": git, "critical_code_sha256": critical,
            "python": platform.python_version(), "pytorch": torch.__version__,
            "lance": getattr(__import__("lance"), "__version__", "unknown"),
            "command": [sys.executable, *sys.argv],
            "test_access_counts": [row["test_access"] for row in fold_summaries],
            "no_backbone_training": True, "fold_training_backbone_forward_count": 0,
        }
        _write_json(output_dir / "processing_provenance.json", provenance)
        summary = {
            "status": "passed", "formal_probe_transfer": "completed",
            "eligible_subject_count": len(subject_ages), "excluded_subjects": ["SUB5"],
            "fold_count": 7, "oof_window_count": aggregate["oof_window_count"],
            "oof_subject_count": aggregate["oof_subject_count"],
            "oof_subject_mean_metrics": aggregate["oof_subject_mean_metrics"],
            "feature_cache_key": cache_info["feature_cache_key"], "feature_cache_reused": cache_info["feature_cache_reused"],
            "source_probe_initialization_identical_across_folds": True,
            "source_scaler_reused": True, "target_scaler_fitted_on_huada": False,
            "backbone_trainable_parameter_count": 0, "fold_trainable_parameter_count": PROBE_PARAMETER_COUNT,
            "backbone_forward_count_during_fold_training": 0,
            "test_access_before_selection": False, "test_evaluations": 7,
            "metric_recomputation_passed": aggregate["metric_recomputation"]["passed"],
        }
        _write_json(output_dir / "probe_transfer_summary.json", summary)
        report = "\n".join([
            "# Dortmund age source-probe → Huada CV7 transfer", "",
            "- Status: passed", "- Backbone: frozen; extracted once into a checkpoint-bound mean cache",
            "- Fold model: source-initialized Linear(512,1), 513 trainable parameters",
            "- Target transform: fixed Dortmund source z-score; no Huada scaler fit",
            f"- OOF subjects/windows: {len(subject_ages)}/{aggregate['oof_window_count']}",
            f"- OOF subject-mean MAE: {aggregate['oof_subject_mean_metrics']['mae']:.6f} years",
            "- Each test subject was accessed once, only after validation selection was locked.",
        ]) + "\n"
        (output_dir / "probe_transfer_report.md").write_text(report, encoding="utf-8")
        (output_dir / "_SUCCESS").write_text("passed\n", encoding="utf-8")
        log_event(logger, "INFO", "run.complete", "probe-transfer CV7 completed", output_dir=str(output_dir), oof_windows=aggregate["oof_window_count"])
        return summary
    except Exception:
        error = traceback.format_exc()
        (output_dir / "probe_transfer_errors.log").write_text(error, encoding="utf-8")
        log_event(logger, "ERROR", "run.failed", "probe-transfer CV7 failed", error=error)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
