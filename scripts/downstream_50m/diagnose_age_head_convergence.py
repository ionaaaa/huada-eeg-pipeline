#!/usr/bin/env python3
"""Diagnose convergence of the cached-feature linear age head without test access.

This is deliberately separate from the formal age CV7 sweep.  It never builds
features, constructs a backbone, selects a candidate, evaluates test data, or
produces an OOF/formal performance result.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import logging
import math
import os
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import lance
import numpy as np
import pyarrow as pa
import torch
from torch.utils.data import DataLoader, Dataset

try:  # package imports in tests
    from .age_cv7 import ALGORITHM_VERSION, PROTOCOL, build_age_subject_cv7
    from .feature_cache import (
        CACHE_DTYPE,
        MANIFEST_NAME,
        MEAN_CACHE_NAME,
        PROHIBITED_CACHE_COLUMNS,
        SUCCESS_NAME,
        sha256_json,
        validate_cache_manifest,
    )
    from .largeScale_fineTune import CachedFeatureClassifier, make_criterion
    from .run_logging import log_event, log_exception, setup_run_logging
except ImportError:  # direct script execution
    from age_cv7 import ALGORITHM_VERSION, PROTOCOL, build_age_subject_cv7
    from feature_cache import (
        CACHE_DTYPE,
        MANIFEST_NAME,
        MEAN_CACHE_NAME,
        PROHIBITED_CACHE_COLUMNS,
        SUCCESS_NAME,
        sha256_json,
        validate_cache_manifest,
    )
    from largeScale_fineTune import CachedFeatureClassifier, make_criterion
    from run_logging import log_event, log_exception, setup_run_logging


TASK = "age_regression"
FEATURE_DIM = 512
POOLING = "mean"
TARGET_TRANSFORM = "raw_years"
CHECKPOINT_KIND = "age_head_convergence_diagnostic_only"
EXPLOSION_DIAGNOSTIC_THRESHOLD = 1e6
EXPECTED_SUBJECTS = {"SUB1", "SUB2", "SUB3", "SUB4", "SUB5", "SUB6", "SUB7", "SUB8"}
ISSUE_FIELDS = [
    "severity", "stage", "experiment", "epoch", "global_step", "code",
    "message", "value", "threshold",
]
STEP_FIELDS = [
    "experiment", "epoch", "global_step", "batch_index", "batch_size", "lr",
    "loss_mse", "mae_years", "rmse_years", "target_min", "target_max",
    "target_mean", "target_std", "prediction_min", "prediction_max",
    "prediction_mean", "prediction_std", "feature_l2_norm_min",
    "feature_l2_norm_max", "feature_l2_norm_mean", "feature_l2_norm_std",
    "head_weight_l2_norm", "head_weight_abs_max", "head_bias",
    "head_parameter_delta_from_init", "gradient_total_l2_norm",
    "weight_gradient_l2_norm", "bias_gradient_abs", "loss_finite",
    "prediction_finite", "parameter_finite", "gradient_finite",
    "optimizer_step_applied", "gradient_norm_before_clip",
    "gradient_norm_after_clip", "gradient_clip_threshold", "elapsed_seconds",
]
EPOCH_FIELDS = [
    "experiment", "epoch", "optimizer_steps", "train_mse", "train_mae_years",
    "train_rmse_years", "validation_mse", "validation_mae_years",
    "validation_rmse_years", "prediction_min", "prediction_max",
    "prediction_mean", "prediction_std", "head_weight_l2_norm", "head_bias",
    "head_parameter_delta_from_init", "gradient_norm_mean", "gradient_norm_max",
    "learning_rate", "epoch_elapsed_seconds", "all_finite",
]


class DiagnosticError(RuntimeError):
    """Base diagnostic error with a stable status and issue payload."""

    status = "failed_runtime"

    def __init__(self, message: str, *, issue: dict[str, Any] | None = None):
        super().__init__(message)
        self.issue = issue or {}


class DataValidationError(DiagnosticError):
    status = "failed_data_validation"


class NonFiniteError(DiagnosticError):
    status = "failed_non_finite"


@dataclass(frozen=True)
class FeatureRecord:
    sample_id: str
    subject_id: str
    recording_id: str
    window_index: int
    age: float
    feature: np.ndarray
    num_valid_tokens: int = 0


class FeatureRecordDataset(Dataset):
    def __init__(self, records: Sequence[FeatureRecord], forbidden_subjects: Iterable[str] = ()):
        self.records = list(records)
        self.forbidden_subjects = {str(value) for value in forbidden_subjects}
        if not self.records:
            raise DataValidationError("FeatureRecordDataset cannot be empty")
        self._assert_safe()

    def _assert_safe(self) -> None:
        forbidden = sorted({row.subject_id for row in self.records} & self.forbidden_subjects)
        if forbidden:
            raise DataValidationError(f"Forbidden test subject materialized: {forbidden}")
        ids = [row.sample_id for row in self.records]
        if len(ids) != len(set(ids)):
            raise DataValidationError("Duplicate sample_id in diagnostic dataset")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.records[int(index)]
        if row.subject_id in self.forbidden_subjects:
            raise AssertionError(f"Test sentinel accessed: {row.sample_id}")
        return {
            "sample_id": row.sample_id,
            "subject_id": row.subject_id,
            "recording_id": row.recording_id,
            "window_index": row.window_index,
            "feature": torch.from_numpy(np.asarray(row.feature, dtype=np.float32)),
            "label": torch.tensor(float(row.age), dtype=torch.float32),
        }


def collate_records(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sample_id": [row["sample_id"] for row in batch],
        "subject_id": [row["subject_id"] for row in batch],
        "recording_id": [row["recording_id"] for row in batch],
        "window_index": torch.tensor([row["window_index"] for row in batch], dtype=torch.int64),
        "feature": torch.stack([row["feature"] for row in batch]),
        "label": torch.stack([row["label"] for row in batch]),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-lance", required=True)
    parser.add_argument("--feature-cache-lance", required=True)
    parser.add_argument("--feature-cache-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--mini-samples-per-subject", type=int, default=4)
    parser.add_argument("--mini-max-epochs", type=int, default=1000)
    parser.add_argument("--full-max-epochs", type=int, default=300)
    parser.add_argument("--mini-convergence-mae", type=float, default=0.10)
    parser.add_argument("--full-train-good-mae", type=float, default=1.0)
    parser.add_argument("--plateau-window", type=int, default=10)
    parser.add_argument("--plateau-slope-tolerance", type=float, default=1e-4)
    parser.add_argument("--step-log-interval", type=int, default=1)
    parser.add_argument("--gradient-clip-norm", type=float, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    positive = {
        "batch_size": args.batch_size,
        "mini_samples_per_subject": args.mini_samples_per_subject,
        "mini_max_epochs": args.mini_max_epochs,
        "full_max_epochs": args.full_max_epochs,
        "step_log_interval": args.step_log_interval,
        "plateau_window": args.plateau_window,
    }
    for name, value in positive.items():
        if value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 0 <= args.fold < 7:
        parser.error("--fold must be in [0, 6]")
    if args.head_lr <= 0 or args.weight_decay < 0:
        parser.error("head LR must be positive and weight decay cannot be negative")
    if args.gradient_clip_norm is not None and args.gradient_clip_norm <= 0:
        parser.error("--gradient-clip-norm must be positive when enabled")
    return args


def _subject_number(subject_id: str) -> int:
    text = str(subject_id).upper()
    if not text.startswith("SUB") or not text[3:].isdigit():
        raise DataValidationError(f"Invalid subject ID: {subject_id!r}")
    return int(text[3:])


def resolve_canonical_fold(subject_ids: Iterable[str], split_seed: int, fold: int) -> dict[str, Any]:
    """Reuse the canonical splitter without supplying or reading test labels."""

    eligible = sorted({str(value) for value in subject_ids}, key=_subject_number)
    payload = build_age_subject_cv7({subject: 0.0 for subject in eligible}, int(split_seed))
    selected = dict(payload["folds"][int(fold)])
    return {
        "protocol": payload["protocol"],
        "algorithm_version": payload["algorithm_version"],
        "split_seed": int(split_seed),
        "permutation": payload["permutation"],
        "fold": int(fold),
        "train_subjects": list(selected["train_subjects"]),
        "validation_subject": selected["validation_subject"],
        "validation_subjects": list(selected["validation_subjects"]),
        "test_subject": selected["test_subject"],
        "test_subjects": list(selected["test_subjects"]),
        "labels_used_to_resolve_assignment": False,
    }


def _lance_filter(subjects: Iterable[str]) -> str:
    values = sorted({str(value).replace("'", "''") for value in subjects}, key=_subject_number)
    if not values:
        raise DataValidationError("Cannot query an empty subject set")
    return "subject_id IN (" + ",".join(f"'{value}'" for value in values) + ")"


class LanceDiagnosticSource:
    """Selective source/cache reader that cannot query a fold's test subject."""

    def __init__(self, input_lance: str | Path, cache_lance: str | Path, manifest_path: str | Path):
        self.input_lance = Path(input_lance)
        self.cache_lance = Path(cache_lance)
        self.manifest_path = Path(manifest_path)
        if not self.input_lance.exists():
            raise DataValidationError(f"Input Lance does not exist: {self.input_lance}")
        if not self.cache_lance.exists():
            raise DataValidationError(f"Mean feature cache does not exist: {self.cache_lance}")
        if not self.manifest_path.is_file():
            raise DataValidationError(f"Cache manifest does not exist: {self.manifest_path}")
        cache_dir = self.cache_lance.parent
        if self.manifest_path.resolve() != (cache_dir / MANIFEST_NAME).resolve():
            raise DataValidationError("Explicit cache manifest must be cache_manifest.json beside cache")
        if self.cache_lance.resolve() != (cache_dir / MEAN_CACHE_NAME).resolve():
            raise DataValidationError("Diagnostic accepts only the canonical mean_features.lance")
        try:
            self.manifest = validate_cache_manifest(cache_dir)
        except Exception as exc:
            raise DataValidationError(str(exc)) from exc
        requested = tuple(self.manifest.get("requested_poolings", ()))
        if POOLING not in requested:
            raise DataValidationError("Cache manifest does not include mean pooling")
        if int(self.manifest.get("mean_feature_dim", -1)) != FEATURE_DIM:
            raise DataValidationError("Mean cache feature dimension must be 512")
        if str(self.manifest.get("dtype")) != CACHE_DTYPE:
            raise DataValidationError("Mean cache dtype must be float32")
        identity = self.manifest.get("cache_identity")
        if identity is not None and sha256_json(identity) != self.manifest.get("cache_key"):
            raise DataValidationError("Feature cache manifest identity hash mismatch")
        manifest_ids = self.manifest.get("sample_ids")
        if not isinstance(manifest_ids, list):
            raise DataValidationError("Feature cache manifest has no sample_ids integrity list")
        if (
            len(manifest_ids) != int(self.manifest.get("input_row_count", -1))
            or len(manifest_ids) != len(set(str(value) for value in manifest_ids))
        ):
            raise DataValidationError("Feature cache manifest sample_ids are duplicated or incomplete")
        self.source = lance.dataset(str(self.input_lance))
        self.cache = lance.dataset(str(self.cache_lance))
        required_source = {"sample_id", "subject_id", "recording_id", "window_index", "age"}
        if not required_source.issubset(self.source.schema.names):
            raise DataValidationError(
                f"Input Lance missing diagnostic columns: {sorted(required_source-set(self.source.schema.names))}"
            )
        required_cache = {"sample_id", "subject_id", "recording_id", "feature", "num_valid_tokens"}
        if not required_cache.issubset(self.cache.schema.names):
            raise DataValidationError("Mean cache schema is missing required fields")
        if set(self.cache.schema.names) & PROHIBITED_CACHE_COLUMNS:
            raise DataValidationError("Feature cache contains task/split/prediction columns")
        feature_type = self.cache.schema.field("feature").type
        if (
            not pa.types.is_fixed_size_list(feature_type)
            or feature_type.list_size != FEATURE_DIM
            or feature_type.value_type != pa.float32()
        ):
            raise DataValidationError("Mean cache feature schema must be fixed_size_list<float32>[512]")
        if int(self.source.count_rows()) != int(self.manifest.get("input_row_count", -1)):
            raise DataValidationError("Source row count differs from cache manifest")
        if int(self.cache.count_rows()) != int(self.manifest.get("mean_row_count", -1)):
            raise DataValidationError("Mean cache row count differs from manifest")

    def discover_subject_ids(self) -> list[str]:
        rows = self.source.to_table(columns=["subject_id"]).to_pylist()
        subjects = sorted({str(row["subject_id"]) for row in rows}, key=_subject_number)
        if set(subjects) != EXPECTED_SUBJECTS:
            raise DataValidationError(f"Unexpected Huada subject set: {subjects}")
        sub5 = self.source.to_table(
            columns=["subject_id", "age"], filter="subject_id = 'SUB5'"
        ).to_pylist()
        if not sub5 or any(row["age"] is not None for row in sub5):
            raise DataValidationError("SUB5 must have age=null")
        return [subject for subject in subjects if subject != "SUB5"]

    def materialize_subjects(
        self, subjects: Iterable[str], *, forbidden_subjects: Iterable[str]
    ) -> list[FeatureRecord]:
        allowed = {str(value) for value in subjects}
        forbidden = {str(value) for value in forbidden_subjects}
        overlap = allowed & forbidden
        if overlap:
            raise AssertionError(f"Test subject requested from cache: {sorted(overlap)}")
        predicate = _lance_filter(allowed)
        source_rows = self.source.to_table(
            columns=["sample_id", "subject_id", "recording_id", "window_index", "age"],
            filter=predicate,
        ).to_pylist()
        cache_rows = self.cache.to_table(
            columns=["sample_id", "subject_id", "recording_id", "feature", "num_valid_tokens"],
            filter=predicate,
        ).to_pylist()
        return join_selected_rows(source_rows, cache_rows, allowed, forbidden)


def join_selected_rows(
    source_rows: Sequence[dict[str, Any]],
    cache_rows: Sequence[dict[str, Any]],
    allowed_subjects: Iterable[str],
    forbidden_subjects: Iterable[str],
) -> list[FeatureRecord]:
    allowed = {str(value) for value in allowed_subjects}
    forbidden = {str(value) for value in forbidden_subjects}
    for row in [*source_rows, *cache_rows]:
        subject = str(row["subject_id"])
        if subject in forbidden or subject not in allowed:
            raise DataValidationError(f"Forbidden/unrequested row materialized: {subject}")
    source_ids = [str(row["sample_id"]) for row in source_rows]
    cache_ids = [str(row["sample_id"]) for row in cache_rows]
    if len(source_ids) != len(set(source_ids)) or len(cache_ids) != len(set(cache_ids)):
        raise DataValidationError("Duplicate sample_id in selected source/cache rows")
    if set(source_ids) != set(cache_ids):
        missing = sorted(set(source_ids) - set(cache_ids))
        extra = sorted(set(cache_ids) - set(source_ids))
        raise DataValidationError(f"Selected cache/source sample mismatch; missing={missing}, extra={extra}")
    cache_by_id = {str(row["sample_id"]): row for row in cache_rows}
    records: list[FeatureRecord] = []
    ages: dict[str, float] = {}
    for source_row in source_rows:
        sample_id = str(source_row["sample_id"])
        cached = cache_by_id[sample_id]
        subject = str(source_row["subject_id"])
        if str(cached["subject_id"]) != subject:
            raise DataValidationError(f"Cache/source subject mismatch for {sample_id}")
        if source_row["age"] is None:
            raise DataValidationError(f"Missing age in train/validation row: {sample_id}")
        age = float(source_row["age"])
        if not math.isfinite(age):
            raise DataValidationError(f"Non-finite age for {sample_id}")
        previous = ages.setdefault(subject, age)
        if previous != age:
            raise DataValidationError(f"Inconsistent age within {subject}")
        feature = np.asarray(cached["feature"], dtype=np.float32).reshape(-1)
        if feature.shape != (FEATURE_DIM,):
            raise DataValidationError(f"Feature dimension is not 512 for {sample_id}")
        if not np.isfinite(feature).all():
            raise NonFiniteError(f"Feature contains NaN/Inf for {sample_id}")
        records.append(FeatureRecord(
            sample_id=sample_id,
            subject_id=subject,
            recording_id=str(source_row.get("recording_id") or cached.get("recording_id") or ""),
            window_index=int(source_row["window_index"]),
            age=age,
            feature=feature.copy(),
            num_valid_tokens=int(cached["num_valid_tokens"]),
        ))
    if not records:
        raise DataValidationError("No train/validation cached records were selected")
    return records


def load_allowed_records(source: Any, fold_info: dict[str, Any]) -> list[FeatureRecord]:
    allowed = list(fold_info["train_subjects"]) + list(fold_info["validation_subjects"])
    test_subjects = set(fold_info["test_subjects"])
    records = source.materialize_subjects(allowed, forbidden_subjects=test_subjects)
    if any(row.subject_id in test_subjects for row in records):
        raise AssertionError("Test row crossed the diagnostic access boundary")
    return records


def stratified_mini_sample(
    train_records: Sequence[FeatureRecord], samples_per_subject: int, seed: int
) -> list[FeatureRecord]:
    grouped: dict[str, list[FeatureRecord]] = defaultdict(list)
    for row in train_records:
        grouped[row.subject_id].append(row)
    selected: list[FeatureRecord] = []
    for subject_index, subject in enumerate(sorted(grouped, key=_subject_number)):
        rows = sorted(grouped[subject], key=lambda row: row.sample_id)
        if len(rows) < int(samples_per_subject):
            raise DataValidationError(
                f"Subject {subject} has only {len(rows)} rows; need {samples_per_subject}"
            )
        rng = np.random.default_rng(int(seed) + 1009 * (subject_index + 1))
        indices = sorted(rng.choice(len(rows), size=int(samples_per_subject), replace=False).tolist())
        selected.extend(rows[index] for index in indices)
    if len(grouped) < 2 or len({row.age for row in selected}) < 2:
        raise DataValidationError("Mini-overfit sample must span subjects and at least two ages")
    return selected


def _array_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(values)), "p01": float(np.quantile(values, 0.01)),
        "p05": float(np.quantile(values, 0.05)), "median": float(np.median(values)),
        "mean": float(np.mean(values)), "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)), "max": float(np.max(values)),
    }


def feature_diagnostics(records: Sequence[FeatureRecord]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not records:
        raise DataValidationError("Cannot diagnose empty features")
    features = np.stack([np.asarray(row.feature) for row in records])
    if features.dtype != np.float32 or features.ndim != 2 or features.shape[1] != FEATURE_DIM:
        raise DataValidationError("Diagnostic features must be float32 [N,512]")
    finite = np.isfinite(features)
    if not finite.all():
        raise NonFiniteError("Feature diagnostics found NaN/Inf")
    ages = np.asarray([row.age for row in records], dtype=np.float64)
    norms = np.linalg.norm(features.astype(np.float64), axis=1)
    means = features.mean(axis=0, dtype=np.float64)
    stds = features.std(axis=0, dtype=np.float64)
    hashes = [hashlib.sha256(row.tobytes()).hexdigest() for row in features]
    duplicate_features = sum(count - 1 for count in Counter(hashes).values() if count > 1)
    sample_ids = [row.sample_id for row in records]
    singular = np.linalg.svd(features.astype(np.float64), compute_uv=False)
    positive = singular[singular > np.finfo(np.float64).eps]
    probabilities = positive / positive.sum() if positive.size and positive.sum() else np.array([])
    effective_rank = float(np.exp(-np.sum(probabilities * np.log(probabilities)))) if probabilities.size else 0.0
    condition = float(positive[0] / positive[-1]) if positive.size else None
    summary = {
        "row_count": len(records),
        "subject_count": len({row.subject_id for row in records}),
        "age_min": float(ages.min()), "age_max": float(ages.max()),
        "age_mean": float(ages.mean()), "age_std": float(ages.std()),
        "feature_dim": int(features.shape[1]), "dtype": str(features.dtype),
        "finite_fraction": float(finite.mean()),
        "zero_variance_dimension_count": int(np.sum(stds == 0.0)),
        "feature_l2_norm": _array_summary(norms),
        "per_dimension_mean_summary": _array_summary(means),
        "per_dimension_std_summary": _array_summary(stds),
        "global_feature_abs_max": float(np.abs(features).max()),
        "duplicate_sample_id_count": len(sample_ids) - len(set(sample_ids)),
        "duplicate_feature_row_count": int(duplicate_features),
        "matrix_rank": int(np.linalg.matrix_rank(features.astype(np.float64))),
        "effective_rank": effective_rank,
        "condition_number_diagnostic": condition,
        "singular_value_summary": _array_summary(singular),
        "training_transform_applied": False,
    }
    dimension_rows = [
        {"dimension": index, "mean": float(means[index]), "std": float(stds[index]),
         "min": float(features[:, index].min()), "max": float(features[:, index].max())}
        for index in range(FEATURE_DIM)
    ]
    return summary, dimension_rows


def write_feature_diagnostics(
    output_dir: Path,
    mini_records: Sequence[FeatureRecord],
    train_records: Sequence[FeatureRecord],
    validation_records: Sequence[FeatureRecord],
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    dimension_sets: dict[str, list[dict[str, Any]]] = {}
    for name, rows in (
        ("mini_overfit", mini_records), ("full_train", train_records),
        ("validation", validation_records),
    ):
        summaries[name], dimension_sets[name] = feature_diagnostics(rows)
    _write_json(output_dir / "feature_diagnostics.json", summaries)
    with (output_dir / "feature_dimension_statistics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = ["split", "dimension", "mean", "std", "min", "max"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, rows in dimension_sets.items():
            writer.writerows({"split": name, **row} for row in rows)
    return summaries


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


class StreamingCsvWriter:
    def __init__(self, path: Path, fields: Sequence[str]):
        self.handle = path.open("w", encoding="utf-8", newline="")
        self.writer = csv.DictWriter(self.handle, fieldnames=list(fields))
        self.writer.writeheader()
        self.count = 0

    def write(self, row: dict[str, Any]) -> None:
        self.writer.writerow({key: row.get(key) for key in self.writer.fieldnames})
        self.handle.flush()
        self.count += 1

    def close(self) -> None:
        self.handle.close()

    def __enter__(self) -> "StreamingCsvWriter":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def _tensor_finite(tensor: torch.Tensor | None) -> bool:
    return tensor is not None and bool(torch.isfinite(tensor).all().item())


def _head_vector(model: CachedFeatureClassifier) -> torch.Tensor:
    return torch.cat([parameter.detach().reshape(-1).cpu() for parameter in model.head.parameters()])


def _head_fingerprint(model: CachedFeatureClassifier) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(model.head.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def create_age_head(seed: int, device: torch.device) -> CachedFeatureClassifier:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        model = CachedFeatureClassifier(
            input_dim=FEATURE_DIM, num_classes=1, pooling=POOLING,
            use_mlp_head=False, head_dropout=0.0,
        )
    model = model.to(device)
    if sum(parameter.numel() for parameter in model.parameters()) != 513:
        raise RuntimeError("Diagnostic age head must have exactly 513 parameters")
    return model


def _stats(values: torch.Tensor) -> dict[str, float]:
    array = values.detach().float().reshape(-1)
    return {
        "min": float(array.min().item()), "max": float(array.max().item()),
        "mean": float(array.mean().item()),
        "std": float(array.std(unbiased=False).item()),
    }


def _parameter_stats(
    model: CachedFeatureClassifier, initial: torch.Tensor
) -> dict[str, Any]:
    weight = model.head.weight.detach()
    bias = model.head.bias.detach()
    current = _head_vector(model)
    finite = all(torch.isfinite(parameter).all().item() for parameter in model.head.parameters())
    return {
        "head_weight_l2_norm": float(torch.linalg.vector_norm(weight).item()),
        "head_weight_abs_max": float(weight.abs().max().item()),
        "head_bias": float(bias.reshape(-1)[0].item()),
        "head_parameter_delta_from_init": float(torch.linalg.vector_norm(current - initial).item()),
        "parameter_finite": bool(finite),
    }


def _gradient_stats(model: CachedFeatureClassifier) -> dict[str, Any]:
    weight_grad = model.head.weight.grad
    bias_grad = model.head.bias.grad
    gradients = [gradient for gradient in (weight_grad, bias_grad) if gradient is not None]
    finite = bool(gradients) and all(torch.isfinite(gradient).all().item() for gradient in gradients)
    total = math.sqrt(sum(float(torch.sum(gradient.detach().float() ** 2).item()) for gradient in gradients))
    return {
        "gradient_total_l2_norm": float(total),
        "weight_gradient_l2_norm": (
            None if weight_grad is None else float(torch.linalg.vector_norm(weight_grad).item())
        ),
        "bias_gradient_abs": None if bias_grad is None else float(bias_grad.abs().max().item()),
        "gradient_finite": finite,
    }


def perform_training_step(
    *, model: CachedFeatureClassifier, optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module, batch: dict[str, Any], device: torch.device,
    experiment: str, epoch: int, global_step: int, batch_index: int,
    initial_parameters: torch.Tensor, gradient_clip_norm: float | None,
    elapsed_seconds: float,
) -> dict[str, Any]:
    forbidden = batch.get("forbidden_subjects", set())
    if set(batch["subject_id"]) & set(forbidden):
        raise AssertionError("Test subject appeared in a training batch")
    features = batch["feature"].to(device)
    targets = batch["label"].to(device).view(-1)
    feature_finite = bool(torch.isfinite(features).all().item())
    if not feature_finite:
        raise NonFiniteError("Training batch feature contains NaN/Inf")
    optimizer.zero_grad(set_to_none=True)
    predictions = model.classify_features(features).view(-1)
    prediction_finite = bool(torch.isfinite(predictions).all().item())
    if not prediction_finite:
        raise NonFiniteError("Prediction became NaN/Inf")
    loss = criterion(predictions, targets)
    loss_finite = bool(torch.isfinite(loss).item())
    if not loss_finite:
        raise NonFiniteError("Loss became NaN/Inf")
    loss.backward()
    gradients = _gradient_stats(model)  # deliberately after backward and before step
    if not gradients["gradient_finite"]:
        raise NonFiniteError("Gradient became NaN/Inf")
    raw_gradient_norm = float(gradients["gradient_total_l2_norm"])
    before_clip = after_clip = None
    if gradient_clip_norm is not None:
        before_clip = raw_gradient_norm
        torch.nn.utils.clip_grad_norm_(model.head.parameters(), float(gradient_clip_norm))
        clipped = _gradient_stats(model)
        if not clipped["gradient_finite"]:
            raise NonFiniteError("Gradient became non-finite after clipping")
        after_clip = float(clipped["gradient_total_l2_norm"])
    optimizer.step()
    parameters = _parameter_stats(model, initial_parameters)
    if not parameters["parameter_finite"]:
        raise NonFiniteError("Head parameters became NaN/Inf after optimizer step")
    target_stats = _stats(targets)
    prediction_stats = _stats(predictions)
    feature_norm_stats = _stats(torch.linalg.vector_norm(features.float(), dim=1))
    errors = predictions.detach() - targets
    return {
        "experiment": experiment, "epoch": epoch, "global_step": global_step,
        "batch_index": batch_index, "batch_size": int(targets.numel()),
        "lr": float(optimizer.param_groups[0]["lr"]),
        "loss_mse": float(loss.item()), "mae_years": float(errors.abs().mean().item()),
        "rmse_years": float(torch.sqrt(torch.mean(errors ** 2)).item()),
        "target_min": target_stats["min"], "target_max": target_stats["max"],
        "target_mean": target_stats["mean"], "target_std": target_stats["std"],
        "prediction_min": prediction_stats["min"], "prediction_max": prediction_stats["max"],
        "prediction_mean": prediction_stats["mean"], "prediction_std": prediction_stats["std"],
        "feature_l2_norm_min": feature_norm_stats["min"],
        "feature_l2_norm_max": feature_norm_stats["max"],
        "feature_l2_norm_mean": feature_norm_stats["mean"],
        "feature_l2_norm_std": feature_norm_stats["std"],
        **parameters, **gradients,
        "loss_finite": loss_finite, "prediction_finite": prediction_finite,
        "gradient_norm_before_clip": before_clip,
        "gradient_norm_after_clip": after_clip,
        "gradient_clip_threshold": gradient_clip_norm,
        "optimizer_step_applied": True, "elapsed_seconds": elapsed_seconds,
    }


@torch.no_grad()
def evaluate_records(
    model: CachedFeatureClassifier, records: Sequence[FeatureRecord],
    batch_size: int, device: torch.device, forbidden_subjects: Iterable[str],
) -> dict[str, Any]:
    dataset = FeatureRecordDataset(records, forbidden_subjects)
    loader = DataLoader(
        dataset, batch_size=min(int(batch_size), len(dataset)), shuffle=False,
        num_workers=0, collate_fn=collate_records, drop_last=False,
    )
    model.eval()
    truth: list[float] = []
    predicted: list[float] = []
    for batch in loader:
        if set(batch["subject_id"]) & set(forbidden_subjects):
            raise AssertionError("Test subject appeared in evaluation batch")
        features = batch["feature"].to(device)
        labels = batch["label"].to(device).view(-1)
        outputs = model.classify_features(features).view(-1)
        if not torch.isfinite(outputs).all():
            raise NonFiniteError("Evaluation prediction became NaN/Inf")
        truth.extend(labels.cpu().tolist())
        predicted.extend(outputs.cpu().tolist())
    targets = np.asarray(truth, dtype=np.float64)
    predictions = np.asarray(predicted, dtype=np.float64)
    errors = predictions - targets
    return {
        "mse": float(np.mean(errors ** 2)), "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "prediction_min": float(predictions.min()), "prediction_max": float(predictions.max()),
        "prediction_mean": float(predictions.mean()), "prediction_std": float(predictions.std()),
        "all_finite": bool(np.isfinite(predictions).all()), "sample_count": len(predictions),
    }


def _state_copy(model: CachedFeatureClassifier) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.head.state_dict().items()}


def _last_n_slope(values: Sequence[float], window: int) -> float | None:
    if len(values) < 2:
        return None
    selected = np.asarray(values[-min(len(values), int(window)):], dtype=np.float64)
    return float(np.polyfit(np.arange(selected.size, dtype=np.float64), selected, 1)[0])


def _diagnostic_checkpoint(
    path: Path, state: dict[str, torch.Tensor], experiment: str, epoch: int,
    fold_info: dict[str, Any], args: argparse.Namespace, cache_key: str | None,
) -> None:
    torch.save({
        "checkpoint_kind": CHECKPOINT_KIND,
        "not_for_formal_evaluation": True,
        "test_accessed": False,
        "diagnostic_head_state_dict": state,
        "experiment": experiment, "epoch": int(epoch), "fold": int(fold_info["fold"]),
        "task": TASK, "pooling": POOLING, "feature_dim": FEATURE_DIM,
        "target_transform": TARGET_TRANSFORM, "optimizer": "AdamW",
        "head_lr": float(args.head_lr), "weight_decay": float(args.weight_decay),
        "feature_cache_key": cache_key,
    }, path)


def run_training_experiment(
    *, experiment: str, train_records: Sequence[FeatureRecord],
    validation_records: Sequence[FeatureRecord] | None, output_dir: Path,
    max_epochs: int, convergence_mae: float, args: argparse.Namespace,
    fold_info: dict[str, Any], logger: logging.Logger,
    cache_key: str | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    test_subjects = set(fold_info["test_subjects"])
    if ({row.subject_id for row in train_records} | {
        row.subject_id for row in (validation_records or [])
    }) & test_subjects:
        raise AssertionError("Test record supplied to diagnostic experiment")
    device = torch.device(args.device)
    model = create_age_head(args.train_seed, device)
    initial_parameters = _head_vector(model)
    initial_fingerprint = _head_fingerprint(model)
    criterion = make_criterion(TASK)
    optimizer = torch.optim.AdamW(
        [{"params": model.head.parameters(), "name": "head"}],
        lr=float(args.head_lr), weight_decay=float(args.weight_decay),
    )
    initial_train = evaluate_records(model, train_records, args.batch_size, device, test_subjects)
    initial_validation = (
        None if validation_records is None else
        evaluate_records(model, validation_records, args.batch_size, device, test_subjects)
    )
    dataset = FeatureRecordDataset(train_records, test_subjects)
    generator = torch.Generator().manual_seed(int(args.train_seed))
    loader = DataLoader(
        dataset,
        batch_size=len(dataset) if experiment == "mini_overfit" else min(args.batch_size, len(dataset)),
        shuffle=experiment != "mini_overfit", generator=generator, num_workers=0,
        collate_fn=collate_records, drop_last=False,
    )
    epoch_rows: list[dict[str, Any]] = []
    train_mae_history: list[float] = []
    best_train = {"mae": math.inf, "mse": math.inf, "rmse": math.inf, "epoch": None}
    best_loss = {"mse": math.inf, "epoch": None}
    best_validation = {"mae": math.inf, "mse": math.inf, "rmse": math.inf, "epoch": None}
    best_loss_state: dict[str, torch.Tensor] | None = None
    global_step = 0
    status = "completed_not_converged"
    stop_reason = "epoch_budget_exhausted"
    epoch_reaching_threshold = None
    started = time.monotonic()
    failure: dict[str, Any] | None = None
    last_valid_train = initial_train
    last_valid_validation = initial_validation
    step_path = output_dir / "step_trajectory.csv"
    epoch_path = output_dir / "epoch_trajectory.csv"
    with StreamingCsvWriter(step_path, STEP_FIELDS) as step_writer, StreamingCsvWriter(
        epoch_path, EPOCH_FIELDS
    ) as epoch_writer:
        for epoch in range(1, int(max_epochs) + 1):
            epoch_started = time.monotonic()
            model.train()
            epoch_gradient_norms: list[float] = []
            try:
                for batch_index, batch in enumerate(loader, start=1):
                    if set(batch["subject_id"]) & test_subjects:
                        raise AssertionError("Test subject appeared in optimizer batch")
                    global_step += 1
                    batch["forbidden_subjects"] = test_subjects
                    row = perform_training_step(
                        model=model, optimizer=optimizer, criterion=criterion, batch=batch,
                        device=device, experiment=experiment, epoch=epoch,
                        global_step=global_step, batch_index=batch_index,
                        initial_parameters=initial_parameters,
                        gradient_clip_norm=args.gradient_clip_norm,
                        elapsed_seconds=time.monotonic() - started,
                    )
                    epoch_gradient_norms.append(float(row["gradient_total_l2_norm"]))
                    if global_step % int(args.step_log_interval) == 0:
                        step_writer.write(row)
            except NonFiniteError as exc:
                status = exc.status
                stop_reason = "non_finite"
                failure = {
                    "severity": "error", "stage": "training", "experiment": experiment,
                    "epoch": epoch, "global_step": global_step, "code": "non_finite",
                    "message": str(exc), "value": "non_finite", "threshold": "finite",
                }
                break
            train_metrics = evaluate_records(model, train_records, args.batch_size, device, test_subjects)
            validation_metrics = (
                None if validation_records is None else
                evaluate_records(model, validation_records, args.batch_size, device, test_subjects)
            )
            parameters = _parameter_stats(model, initial_parameters)
            all_finite = bool(
                train_metrics["all_finite"]
                and (validation_metrics is None or validation_metrics["all_finite"])
                and parameters["parameter_finite"]
                and all(math.isfinite(value) for value in epoch_gradient_norms)
            )
            if not all_finite:
                status, stop_reason = "failed_non_finite", "non_finite"
                failure = {
                    "severity": "error", "stage": "epoch_evaluation", "experiment": experiment,
                    "epoch": epoch, "global_step": global_step, "code": "non_finite",
                    "message": "Non-finite epoch statistic", "value": "non_finite", "threshold": "finite",
                }
                break
            epoch_row = {
                "experiment": experiment, "epoch": epoch,
                "optimizer_steps": global_step, "train_mse": train_metrics["mse"],
                "train_mae_years": train_metrics["mae"],
                "train_rmse_years": train_metrics["rmse"],
                "validation_mse": None if validation_metrics is None else validation_metrics["mse"],
                "validation_mae_years": None if validation_metrics is None else validation_metrics["mae"],
                "validation_rmse_years": None if validation_metrics is None else validation_metrics["rmse"],
                "prediction_min": train_metrics["prediction_min"],
                "prediction_max": train_metrics["prediction_max"],
                "prediction_mean": train_metrics["prediction_mean"],
                "prediction_std": train_metrics["prediction_std"],
                **{key: parameters[key] for key in (
                    "head_weight_l2_norm", "head_bias", "head_parameter_delta_from_init"
                )},
                "gradient_norm_mean": float(np.mean(epoch_gradient_norms)),
                "gradient_norm_max": float(np.max(epoch_gradient_norms)),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "epoch_elapsed_seconds": time.monotonic() - epoch_started,
                "all_finite": all_finite,
            }
            epoch_writer.write(epoch_row)
            epoch_rows.append(epoch_row)
            train_mae_history.append(float(train_metrics["mae"]))
            last_valid_train = train_metrics
            last_valid_validation = validation_metrics
            if train_metrics["mae"] < best_train["mae"]:
                best_train = {**{key: float(train_metrics[key]) for key in ("mae", "mse", "rmse")}, "epoch": epoch}
            if train_metrics["mse"] < best_loss["mse"]:
                best_loss = {"mse": float(train_metrics["mse"]), "epoch": epoch}
                best_loss_state = _state_copy(model)
            if validation_metrics is not None and validation_metrics["mae"] < best_validation["mae"]:
                best_validation = {
                    **{key: float(validation_metrics[key]) for key in ("mae", "mse", "rmse")},
                    "epoch": epoch,
                }
            log_event(
                logger, "INFO", "diagnostic.epoch", "diagnostic epoch completed",
                experiment=experiment, epoch=f"{epoch}/{max_epochs}", steps=global_step,
                train_loss=round(train_metrics["mse"], 6),
                train_mae=round(train_metrics["mae"], 6),
                validation_mae=(None if validation_metrics is None else round(validation_metrics["mae"], 6)),
                prediction_range=[round(train_metrics["prediction_min"], 4), round(train_metrics["prediction_max"], 4)],
                weight_norm=round(parameters["head_weight_l2_norm"], 6),
                gradient_mean=round(float(np.mean(epoch_gradient_norms)), 6),
                gradient_max=round(float(np.max(epoch_gradient_norms)), 6),
                finite=all_finite, elapsed_seconds=round(time.monotonic() - started, 3),
            )
            if experiment == "mini_overfit" and train_metrics["mae"] <= float(convergence_mae):
                status, stop_reason = "completed_converged", "mae_threshold_reached"
                epoch_reaching_threshold = epoch
                log_event(
                    logger, "INFO", "diagnostic.converged", "mini-overfit threshold reached",
                    experiment=experiment, epoch=epoch, train_mae=train_metrics["mae"],
                    threshold=convergence_mae,
                )
                break
    if status == "failed_non_finite":
        final_train = last_valid_train
        final_validation = last_valid_validation
    else:
        final_train = evaluate_records(model, train_records, args.batch_size, device, test_subjects)
        final_validation = (
            None if validation_records is None else
            evaluate_records(model, validation_records, args.batch_size, device, test_subjects)
        )
    if status not in {"failed_non_finite", "failed_runtime"}:
        threshold = convergence_mae
        if best_train["mae"] <= threshold:
            status = "completed_converged"
            if epoch_reaching_threshold is None:
                epoch_reaching_threshold = best_train["epoch"]
        else:
            status = "completed_not_converged"
    if best_loss_state is not None and experiment == "full_train":
        _diagnostic_checkpoint(
            output_dir / "minimum_train_loss_diagnostic_head.pt", best_loss_state, experiment,
            int(best_loss["epoch"]), fold_info, args, cache_key,
        )
        _diagnostic_checkpoint(
            output_dir / "final_diagnostic_head.pt", _state_copy(model), experiment,
            int(epoch_rows[-1]["epoch"] if epoch_rows else 0), fold_info, args, cache_key,
        )
    slope = _last_n_slope(train_mae_history, args.plateau_window)
    plateau = bool(slope is not None and abs(slope) <= float(args.plateau_slope_tolerance))
    result = {
        "experiment": experiment, "status": status, "stop_reason": stop_reason,
        "initial_head_fingerprint": initial_fingerprint,
        "final_head_fingerprint": _head_fingerprint(model),
        "initial_loss": initial_train["mse"], "final_loss": final_train["mse"],
        "minimum_loss": None if best_loss["epoch"] is None else best_loss["mse"],
        "epoch_of_minimum_train_loss": best_loss["epoch"],
        "initial_mae": initial_train["mae"], "final_mae": final_train["mae"],
        "minimum_mae": None if best_train["epoch"] is None else best_train["mae"],
        "initial_rmse": initial_train["rmse"], "final_rmse": final_train["rmse"],
        "minimum_rmse": (
            None if not epoch_rows else min(float(row["train_rmse_years"]) for row in epoch_rows)
        ),
        "epoch_of_minimum_train_mae": best_train["epoch"],
        "epoch_reaching_threshold": epoch_reaching_threshold,
        "converged": status == "completed_converged",
        "loss_reduction_fraction": (
            None if initial_train["mse"] == 0 else
            float((initial_train["mse"] - final_train["mse"]) / initial_train["mse"])
        ),
        "initial_validation": initial_validation, "final_validation": final_validation,
        "minimum_validation": None if best_validation["epoch"] is None else best_validation,
        "epoch_of_minimum_validation_mae": best_validation["epoch"],
        "train_validation_gap_at_min_train_epoch": None,
        "train_validation_gap_at_min_validation_epoch": None,
        "last_n_epoch_mae_slope": slope, "plateau_detected": plateau,
        "plateau_window": int(args.plateau_window),
        "all_values_finite": status != "failed_non_finite",
        "optimizer_steps": global_step, "step_trajectory_rows": step_writer.count,
        "epoch_trajectory_rows": len(epoch_rows), "failure": failure,
        "test_accessed": False,
    }
    if status == "failed_non_finite":
        result["failure_mode"] = "non_finite"
    elif status == "completed_converged":
        result["failure_mode"] = None
    elif epoch_rows and max(float(row["gradient_norm_max"]) for row in epoch_rows) > EXPLOSION_DIAGNOSTIC_THRESHOLD:
        result["failure_mode"] = "gradient_explosion"
    elif epoch_rows and max(
        max(abs(float(row["prediction_min"])), abs(float(row["prediction_max"])), abs(float(row["head_weight_l2_norm"])))
        for row in epoch_rows
    ) > EXPLOSION_DIAGNOSTIC_THRESHOLD:
        result["failure_mode"] = "prediction_or_parameter_explosion"
    elif plateau:
        result["failure_mode"] = "loss_plateau"
    else:
        result["failure_mode"] = "epoch_budget_exhausted"
    if validation_records is not None and epoch_rows:
        by_epoch = {int(row["epoch"]): row for row in epoch_rows}
        min_train_row = by_epoch.get(int(best_train["epoch"]))
        min_val_row = by_epoch.get(int(best_validation["epoch"])) if best_validation["epoch"] else None
        if min_train_row:
            result["train_validation_gap_at_min_train_epoch"] = (
                float(min_train_row["validation_mae_years"]) - float(min_train_row["train_mae_years"])
            )
        if min_val_row:
            result["train_validation_gap_at_min_validation_epoch"] = (
                float(min_val_row["validation_mae_years"]) - float(min_val_row["train_mae_years"])
            )
    return result


def automatic_conclusion(mini: dict[str, Any], full: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if mini["status"] == "failed_non_finite" or full.get("status") == "failed_non_finite":
        return {
            "category": "A", "code": "non_finite_training_failure",
            "message_zh": "出现非有限数值，当前不能把正式CV7差结果解释为泛化失败。",
        }
    if not mini.get("converged", False):
        return {
            "category": "A", "code": "mini_overfit_failed",
            "message_zh": "少量训练样本无法过拟合，仍可能存在实现、尺度、优化器或梯度问题。",
        }
    if not full.get("converged", False):
        return {
            "category": "B", "code": "mini_passed_full_train_fit_poor",
            "message_zh": "基本训练链路可工作，但完整训练集拟合仍差，需检查容量、特征尺度、优化设置或样本冲突。",
        }
    initial_val = (full.get("initial_validation") or {}).get("mae")
    final_val = (full.get("final_validation") or {}).get("mae")
    minimum_val = (full.get("minimum_validation") or {}).get("mae")
    sustained_improvement = bool(
        initial_val is not None and final_val is not None and minimum_val is not None
        and minimum_val <= 0.8 * initial_val and final_val <= 0.9 * initial_val
    )
    if sustained_improvement:
        return {
            "category": "D", "code": "train_and_validation_improved",
            "message_zh": "mini和完整train均能拟合且validation持续改善，之前正式训练的epoch、LR或选择设置值得进一步核查。",
        }
    return {
        "category": "C", "code": "train_fit_validation_generalization_gap",
        "message_zh": "线性头能学习训练数据，但validation没有持续改善，主要问题更像极小样本跨被试泛化。",
    }


def _git_state(root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip())
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    mini = summary["mini_overfit"]
    full = summary["full_train"]
    conclusion = summary["automatic_conclusion"]
    report = f"""# 年龄线性头收敛诊断

本次是诊断实验，不是正式性能实验。未创建test dataset/DataLoader，未读取test feature、prediction或label，不能从本结果报告test MAE。

- fold: `{summary['fold']}`
- train / validation / excluded test subject: `{summary['train_subjects']}` / `{summary['validation_subject']}` / `{summary['test_subject']}`
- target: `raw years`
- optimizer: `AdamW`
- gradient clipping: `{summary['gradient_clipping']}`
- mini status / minimum MAE: `{mini['status']}` / `{mini.get('minimum_mae')}`
- full status / minimum train MAE: `{full['status']}` / `{full.get('minimum_mae')}`
- full minimum/final validation MAE: `{(full.get('minimum_validation') or {}).get('mae')}` / `{(full.get('final_validation') or {}).get('mae')}`
- any non-finite value: `{not summary['all_values_finite']}`

自动结论 `{conclusion['category']}`：{conclusion['message_zh']}

此前正式sweep的cached linear-probe实际使用SGD；`lr=0.001`在所有fold表现为约32–45年validation MAE，而`0.0003`保持有限。本诊断按要求改用AdamW并记录逐step梯度/尺度，因此可以诊断AdamW头能否收敛，但不能把优化器差异静默忽略，也不会修改正式CV7结论。

诊断checkpoint仅含`diagnostic_head_state_dict`，标记`not_for_formal_evaluation=true`，不得用于正式模型结论。
"""
    path.write_text(report, encoding="utf-8")


def _access_guards() -> dict[str, Any]:
    return {
        "test_dataset_created": False, "test_dataloader_created": False,
        "test_rows_materialized": 0, "test_forward_count": 0,
        "test_metrics_computed": False,
    }


def verify_trajectory_artifacts(output_dir: Path, summary: dict[str, Any]) -> bool:
    """Fail closed if streamed trajectories disagree with their summaries."""

    required = [
        "feature_diagnostics.json", "feature_dimension_statistics.csv",
        "selected_mini_samples.csv", "split_subjects.csv",
        "mini_overfit/step_trajectory.csv", "mini_overfit/epoch_trajectory.csv",
        "full_train/step_trajectory.csv", "full_train/epoch_trajectory.csv",
        "diagnostic_issues.csv", "diagnostic_errors.log",
    ]
    missing = [name for name in required if not (output_dir / name).exists()]
    if missing:
        raise RuntimeError(f"Missing diagnostic artifacts: {missing}")
    for experiment in ("mini_overfit", "full_train"):
        step_rows = _read_csv(output_dir / experiment / "step_trajectory.csv")
        epoch_rows = _read_csv(output_dir / experiment / "epoch_trajectory.csv")
        expected = summary[experiment]
        if len(step_rows) != int(expected["step_trajectory_rows"]):
            raise RuntimeError(f"{experiment} step trajectory row-count mismatch")
        if len(epoch_rows) != int(expected["epoch_trajectory_rows"]):
            raise RuntimeError(f"{experiment} epoch trajectory row-count mismatch")
        epochs = [int(row["epoch"]) for row in epoch_rows]
        if epochs != list(range(1, len(epochs) + 1)):
            raise RuntimeError(f"{experiment} epoch trajectory is not consecutive")
        if experiment == "mini_overfit" and any(
            row["validation_mse"] or row["validation_mae_years"] or row["validation_rmse_years"]
            for row in epoch_rows
        ):
            raise RuntimeError("Mini-overfit trajectory fabricated validation metrics")
    selected = _read_csv(output_dir / "selected_mini_samples.csv")
    if len(selected) != int(summary["mini_row_count"]):
        raise RuntimeError("Selected mini sample count differs from summary")
    if any(row["subject_id"] == summary["test_subject"] for row in selected):
        raise RuntimeError("Test subject leaked into selected mini samples")
    issues_header = (output_dir / "diagnostic_issues.csv").open(
        encoding="utf-8", newline=""
    ).readline().strip().split(",")
    if issues_header != ISSUE_FIELDS:
        raise RuntimeError("Diagnostic issue schema mismatch")
    for checkpoint in (output_dir / "full_train").glob("*_diagnostic_head.pt"):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (
            payload.get("checkpoint_kind") != CHECKPOINT_KIND
            or payload.get("not_for_formal_evaluation") is not True
            or payload.get("test_accessed") is not False
            or "head_state_dict" in payload
            or "model_state_dict" in payload
        ):
            raise RuntimeError(f"Unsafe diagnostic checkpoint metadata: {checkpoint}")
    return True


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def execute_from_records(
    *, records: Sequence[FeatureRecord], fold_info: dict[str, Any], output_dir: Path,
    args: argparse.Namespace, logger: logging.Logger, cache_key: str | None = None,
) -> dict[str, Any]:
    test_subjects = set(fold_info["test_subjects"])
    if any(row.subject_id in test_subjects for row in records):
        raise AssertionError("Test record was materialized before diagnostic execution")
    train_subjects = set(fold_info["train_subjects"])
    validation_subjects = set(fold_info["validation_subjects"])
    train_records = [row for row in records if row.subject_id in train_subjects]
    validation_records = [row for row in records if row.subject_id in validation_subjects]
    if not train_records or not validation_records:
        raise DataValidationError("Train or validation split is empty")
    if {row.sample_id for row in train_records} & {row.sample_id for row in validation_records}:
        raise DataValidationError("Train/validation sample leakage")
    mini_records = stratified_mini_sample(
        train_records, args.mini_samples_per_subject, args.train_seed
    )
    selected_rows = [
        {"sample_id": row.sample_id, "subject_id": row.subject_id, "age": row.age,
         "recording_id": row.recording_id, "window_index": row.window_index}
        for row in mini_records
    ]
    _write_csv(
        output_dir / "selected_mini_samples.csv", selected_rows,
        ["sample_id", "subject_id", "age", "recording_id", "window_index"],
    )
    diagnostics = write_feature_diagnostics(
        output_dir, mini_records, train_records, validation_records
    )
    log_event(logger, "INFO", "diagnostic.stage4", "[stage 4] mini-overfit")
    mini = run_training_experiment(
        experiment="mini_overfit", train_records=mini_records, validation_records=None,
        output_dir=output_dir / "mini_overfit", max_epochs=args.mini_max_epochs,
        convergence_mae=args.mini_convergence_mae, args=args, fold_info=fold_info,
        logger=logger, cache_key=cache_key,
    )
    if mini["status"] == "failed_non_finite":
        full_dir = output_dir / "full_train"
        full_dir.mkdir(parents=True, exist_ok=False)
        _write_csv(full_dir / "step_trajectory.csv", [], STEP_FIELDS)
        _write_csv(full_dir / "epoch_trajectory.csv", [], EPOCH_FIELDS)
        full = {
            "experiment": "full_train", "status": "failed_non_finite",
            "stop_reason": "skipped_after_mini_non_finite", "converged": False,
            "all_values_finite": False, "optimizer_steps": 0,
            "step_trajectory_rows": 0, "epoch_trajectory_rows": 0,
            "test_accessed": False,
        }
    else:
        log_event(logger, "INFO", "diagnostic.stage5", "[stage 5] full-train convergence")
        full = run_training_experiment(
            experiment="full_train", train_records=train_records,
            validation_records=validation_records, output_dir=output_dir / "full_train",
            max_epochs=args.full_max_epochs, convergence_mae=args.full_train_good_mae,
            args=args, fold_info=fold_info, logger=logger, cache_key=cache_key,
        )
    conclusion = automatic_conclusion(mini, full, args)
    summary = {
        "status": (
            "failed_non_finite" if "failed_non_finite" in {mini["status"], full["status"]}
            else "completed_converged"
            if mini.get("converged") and full.get("converged")
            else "completed_not_converged"
        ),
        "diagnostic_only": True, "not_for_formal_performance": True,
        "task": TASK, "target_transform": TARGET_TRANSFORM, "pooling": POOLING,
        "feature_dim": FEATURE_DIM, "feature_source": "existing_cache",
        "backbone_constructed": False, "backbone_forward_count": 0,
        "fold": int(fold_info["fold"]), "split_seed": int(args.split_seed),
        "train_seed": int(args.train_seed), "train_subjects": fold_info["train_subjects"],
        "validation_subject": fold_info["validation_subject"],
        "test_subject": fold_info["test_subject"],
        "train_row_count": len(train_records), "validation_row_count": len(validation_records),
        "mini_row_count": len(mini_records), "optimizer": "AdamW", "scheduler": None,
        "head_lr": float(args.head_lr), "weight_decay": float(args.weight_decay),
        "gradient_clipping": {
            "enabled": args.gradient_clip_norm is not None,
            "threshold": args.gradient_clip_norm,
        },
        "mini_overfit": mini, "full_train": full,
        "feature_diagnostics": diagnostics, "automatic_conclusion": conclusion,
        "all_values_finite": bool(
            mini.get("all_values_finite", False) and full.get("all_values_finite", False)
        ),
        **_access_guards(),
    }
    return summary


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    errors_path = output_dir / "diagnostic_errors.log"
    errors_path.write_text("", encoding="utf-8")
    _write_csv(output_dir / "diagnostic_issues.csv", [], ISSUE_FIELDS)
    logger = setup_run_logging(output_dir, args.log_level, append=False)
    issues: list[dict[str, Any]] = []
    project_root = Path(__file__).resolve().parents[2]
    try:
        log_event(logger, "INFO", "diagnostic.stage1", "[stage 1] validate inputs and cache")
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise DataValidationError("CUDA requested but unavailable")
        source = LanceDiagnosticSource(
            args.input_lance, args.feature_cache_lance, args.feature_cache_manifest
        )
        eligible_subjects = source.discover_subject_ids()
        log_event(logger, "INFO", "diagnostic.stage2", "[stage 2] resolve fold without materializing test")
        fold_info = resolve_canonical_fold(eligible_subjects, args.split_seed, args.fold)
        records = load_allowed_records(source, fold_info)
        split_rows = [
            {"subject_id": subject, "split": "train", "materialized": True}
            for subject in fold_info["train_subjects"]
        ] + [
            {"subject_id": fold_info["validation_subject"], "split": "validation", "materialized": True},
            {"subject_id": fold_info["test_subject"], "split": "test_excluded", "materialized": False},
            {"subject_id": "SUB5", "split": "excluded_missing_age", "materialized": False},
        ]
        _write_csv(output_dir / "split_subjects.csv", split_rows, ["subject_id", "split", "materialized"])
        log_event(logger, "INFO", "diagnostic.stage3", "[stage 3] feature diagnostics")
        summary = execute_from_records(
            records=records, fold_info=fold_info, output_dir=output_dir,
            args=args, logger=logger, cache_key=source.manifest.get("cache_key"),
        )
        for result in (summary["mini_overfit"], summary["full_train"]):
            if result.get("failure"):
                issues.append(result["failure"])
        _write_csv(output_dir / "diagnostic_issues.csv", issues, ISSUE_FIELDS)
        summary["trajectory_artifact_validation_passed"] = verify_trajectory_artifacts(
            output_dir, summary
        )
        provenance = {
            **summary,
            "protocol": "age_head_convergence_diagnostic_v1",
            "canonical_cv_protocol": PROTOCOL,
            "canonical_fold_algorithm_version": ALGORITHM_VERSION,
            "fold_assignment": fold_info,
            "input_lance": str(Path(args.input_lance).resolve()),
            "feature_cache_lance": str(Path(args.feature_cache_lance).resolve()),
            "feature_cache_manifest": str(Path(args.feature_cache_manifest).resolve()),
            "feature_cache_manifest_sha256": _sha256_file(Path(args.feature_cache_manifest)),
            "feature_cache_key": source.manifest.get("cache_key"),
            "cache_parity_passed": source.manifest.get("parity_passed"),
            "cache_dtype": source.manifest.get("dtype"),
            "prior_formal_context": {
                "optimizer": "SGD",
                "head_lr_0p001_validation_mae_observed_range": [32.13933048248291, 45.18931341952965],
                "head_lr_0p0003_validation_mae_observed_range": [0.5651840597896253, 5.850984191894531],
                "source": "audited 50m_age_cv7_sweep_20261102_135226 run.log",
                "diagnostic_optimizer_differs": True,
            },
            "test_query_executed": False,
            "formal_checkpoint_compatible": False,
            "git": _git_state(project_root),
            "code_sha256": {
                "scripts/downstream_50m/diagnose_age_head_convergence.py": _sha256_file(Path(__file__)),
                "scripts/downstream_50m/age_cv7.py": _sha256_file(project_root / "scripts/downstream_50m/age_cv7.py"),
                "scripts/downstream_50m/feature_cache.py": _sha256_file(project_root / "scripts/downstream_50m/feature_cache.py"),
                "scripts/downstream_50m/largeScale_fineTune.py": _sha256_file(project_root / "scripts/downstream_50m/largeScale_fineTune.py"),
            },
        }
        _write_json(output_dir / "convergence_summary.json", summary)
        _write_json(output_dir / "processing_provenance.json", provenance)
        _write_report(output_dir / "convergence_report.md", summary)
        log_event(logger, "INFO", "diagnostic.stage6", "[stage 6] summarize and verify artifacts",
                  status=summary["status"], output_dir=str(output_dir.resolve()))
        return 0 if not summary["status"].startswith("failed") else 2
    except Exception as exc:
        errors_path.write_text(traceback.format_exc(), encoding="utf-8")
        status = exc.status if isinstance(exc, DiagnosticError) else "failed_runtime"
        issue = {
            "severity": "error", "stage": "input_or_runtime", "experiment": "",
            "epoch": "", "global_step": "", "code": status,
            "message": str(exc), "value": "", "threshold": "",
        }
        issues.append(issue)
        _write_csv(output_dir / "diagnostic_issues.csv", issues, ISSUE_FIELDS)
        failure = {
            "status": status, "diagnostic_only": True,
            "not_for_formal_performance": True, **_access_guards(),
            "error": str(exc),
        }
        _write_json(output_dir / "convergence_summary.json", failure)
        _write_json(output_dir / "processing_provenance.json", failure)
        (output_dir / "convergence_report.md").write_text(
            "# 年龄线性头收敛诊断\n\n诊断失败，未访问test。详见diagnostic_errors.log。\n",
            encoding="utf-8",
        )
        log_exception(logger, "diagnostic.failed", "age head convergence diagnostic failed",
                      status=status)
        return 2


def main(argv: Sequence[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
