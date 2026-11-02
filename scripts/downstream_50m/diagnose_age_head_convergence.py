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
RAW_TARGET_TRANSFORM = "raw_years"
ZSCORE_TARGET_TRANSFORM = "train_subject_zscore"
TARGET_TRANSFORMS = (RAW_TARGET_TRANSFORM, ZSCORE_TARGET_TRANSFORM)
CHECKPOINT_KIND = "age_head_convergence_diagnostic_only"
LEAST_SQUARES_CHECKPOINT_KIND = "diagnostic_least_squares_head_only"
TARGET_STD_EPSILON = 1e-12
EXPLOSION_DIAGNOSTIC_THRESHOLD = 1e6
EXPECTED_SUBJECTS = {"SUB1", "SUB2", "SUB3", "SUB4", "SUB5", "SUB6", "SUB7", "SUB8"}
ISSUE_FIELDS = [
    "severity", "stage", "experiment", "epoch", "global_step", "code",
    "message", "value", "threshold",
]
STEP_FIELDS = [
    "experiment", "epoch", "global_step", "batch_index", "batch_size", "lr",
    "target_transform", "target_mean_years", "target_std_years",
    "loss_mse_z", "batch_mae_z", "batch_rmse_z", "batch_mae_years",
    "batch_rmse_years", "batch_mean_bias_years",
    "prediction_z_min", "prediction_z_max", "prediction_z_mean", "prediction_z_std",
    "prediction_years_min", "prediction_years_max", "prediction_years_mean",
    "prediction_years_std", "target_z_min", "target_z_max", "target_z_mean",
    "target_z_std",
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
    "experiment", "epoch", "optimizer_steps", "target_transform", "target_mean_years",
    "target_std_years", "train_mse_z", "train_mae_z", "train_rmse_z",
    "train_mse", "train_mae_years", "train_rmse_years", "train_mean_bias_years",
    "validation_mse_z", "validation_mae_z", "validation_rmse_z",
    "validation_mse", "validation_mae_years", "validation_rmse_years",
    "validation_mean_bias_years", "prediction_z_min", "prediction_z_max",
    "prediction_z_mean", "prediction_z_std", "prediction_years_min",
    "prediction_years_max", "prediction_years_mean", "prediction_years_std",
    "prediction_min", "prediction_max",
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


@dataclass(frozen=True)
class TargetTransform:
    mode: str
    train_subject_ages: dict[str, float]
    mean_years: float
    std_years: float
    epsilon: float = TARGET_STD_EPSILON

    @property
    def standardized(self) -> bool:
        return self.mode == ZSCORE_TARGET_TRANSFORM

    @property
    def initialization(self) -> str:
        return "zeros" if self.standardized else "legacy_seeded_random"

    def transform_tensor(self, values: torch.Tensor) -> torch.Tensor:
        if not self.standardized:
            return values
        return (values - float(self.mean_years)) / float(self.std_years)

    def inverse_tensor(self, values: torch.Tensor) -> torch.Tensor:
        if not self.standardized:
            return values
        return values * float(self.std_years) + float(self.mean_years)

    def transform_numpy(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if not self.standardized:
            return array
        return (array - self.mean_years) / self.std_years

    def inverse_numpy(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if not self.standardized:
            return array
        return array * self.std_years + self.mean_years

    def as_dict(self, fold_info: dict[str, Any]) -> dict[str, Any]:
        subject_ids = list(fold_info["train_subjects"])
        ages = np.asarray([self.train_subject_ages[subject] for subject in subject_ids], dtype=np.float64)
        round_trip = self.inverse_numpy(self.transform_numpy(ages))
        return {
            "fold": int(fold_info["fold"]),
            "target_transform": self.mode,
            "target_statistics_scope": "unique_train_subjects_only",
            "target_statistics_weighting": "equal_subject_weight",
            "train_subject_ids": subject_ids,
            "train_subject_ages_years": {
                subject: float(self.train_subject_ages[subject]) for subject in subject_ids
            },
            "target_mean_years": float(self.mean_years),
            "target_std_years": float(self.std_years),
            "target_std_ddof": 0,
            "epsilon": float(self.epsilon),
            "formula": "z = (age_years - mu_train) / sigma_train",
            "inverse_formula": "age_years = z * sigma_train + mu_train",
            "target_inverse_transform_applied": bool(self.standardized),
            "validation_used_for_target_statistics": False,
            "test_used_for_target_statistics": False,
            "validation_subject": str(fold_info["validation_subject"]),
            "test_subject": str(fold_info["test_subject"]),
            "test_age_read": False,
            "round_trip_max_abs_error_years": float(np.max(np.abs(round_trip - ages))),
        }


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
    parser.add_argument("--target-transform", choices=TARGET_TRANSFORMS, default=RAW_TARGET_TRANSFORM)
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
    parser.add_argument("--lstsq-driver", choices=["gels", "gelsy", "gelsd", "gelss"], default="gelsd")
    parser.add_argument("--lstsq-rcond", type=float, default=None)
    parser.add_argument("--lstsq-absolute-tolerance", type=float, default=1e-6)
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
    if args.lstsq_rcond is not None and args.lstsq_rcond < 0:
        parser.error("--lstsq-rcond cannot be negative")
    if args.lstsq_absolute_tolerance <= 0:
        parser.error("--lstsq-absolute-tolerance must be positive")
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


def build_train_only_target_transform(
    train_records: Sequence[FeatureRecord], fold_info: dict[str, Any], mode: str,
    epsilon: float = TARGET_STD_EPSILON,
) -> TargetTransform:
    """Build target statistics from one age per canonical train subject only."""

    if mode not in TARGET_TRANSFORMS:
        raise DataValidationError(f"Unsupported target transform: {mode}")
    expected = list(fold_info["train_subjects"])
    expected_set = set(expected)
    forbidden = set(fold_info["validation_subjects"]) | set(fold_info["test_subjects"])
    grouped: dict[str, set[float]] = defaultdict(set)
    for row in train_records:
        if row.subject_id in forbidden or row.subject_id not in expected_set:
            raise DataValidationError(
                f"Non-train subject supplied to target statistics: {row.subject_id}"
            )
        if not math.isfinite(float(row.age)):
            raise DataValidationError(f"Non-finite train age for {row.sample_id}")
        grouped[row.subject_id].add(float(row.age))
    if set(grouped) != expected_set:
        raise DataValidationError(
            f"Target-statistics subject mismatch: got={sorted(grouped)}, expected={expected}"
        )
    inconsistent = {subject: sorted(values) for subject, values in grouped.items() if len(values) != 1}
    if inconsistent:
        raise DataValidationError(f"Inconsistent age within train subject: {inconsistent}")
    subject_ages = {subject: next(iter(grouped[subject])) for subject in expected}
    age_vector = np.asarray([subject_ages[subject] for subject in expected], dtype=np.float64)
    mean_years = float(np.mean(age_vector))
    std_years = float(np.std(age_vector, ddof=0))
    if not math.isfinite(mean_years) or not math.isfinite(std_years):
        raise NonFiniteError("Train-only target statistics became NaN/Inf")
    if mode == ZSCORE_TARGET_TRANSFORM and std_years <= float(epsilon):
        raise DataValidationError(
            f"Train-subject age std {std_years} is not greater than epsilon {epsilon}"
        )
    return TargetTransform(
        mode=mode,
        train_subject_ages=subject_ages,
        mean_years=mean_years,
        std_years=std_years,
        epsilon=float(epsilon),
    )


@dataclass(frozen=True)
class LeastSquaresReference:
    diagnostics: dict[str, Any]
    head_state_dict: dict[str, torch.Tensor]
    predicted_years: np.ndarray
    sample_ids: tuple[str, ...]


def solve_train_only_linear_least_squares(
    records: Sequence[FeatureRecord], target_transform: TargetTransform, *,
    driver: str = "gelsd", rcond: float | None = None,
    absolute_tolerance: float = 1e-6,
) -> LeastSquaresReference:
    """Compute a CPU/float64 affine least-squares reference from train rows only."""

    if not records:
        raise DataValidationError("Least-squares diagnostic requires train records")
    if absolute_tolerance <= 0:
        raise DataValidationError("Least-squares absolute tolerance must be positive")
    sample_ids = tuple(row.sample_id for row in records)
    if len(sample_ids) != len(set(sample_ids)):
        raise DataValidationError("Duplicate sample_id in least-squares rows")
    features_np = np.stack([np.asarray(row.feature, dtype=np.float64) for row in records])
    targets_years_np = np.asarray([row.age for row in records], dtype=np.float64)
    if features_np.shape != (len(records), FEATURE_DIM):
        raise DataValidationError("Least-squares features must be [N,512]")
    if not np.isfinite(features_np).all() or not np.isfinite(targets_years_np).all():
        raise NonFiniteError("Least-squares input contains NaN/Inf")
    targets_z_np = target_transform.transform_numpy(targets_years_np)
    features = torch.from_numpy(features_np).to(device="cpu", dtype=torch.float64)
    targets_z = torch.from_numpy(targets_z_np).to(device="cpu", dtype=torch.float64).reshape(-1, 1)
    ones = torch.ones((features.shape[0], 1), dtype=torch.float64, device="cpu")
    augmented = torch.cat([features, ones], dim=1)
    try:
        solved = torch.linalg.lstsq(
            augmented, targets_z, rcond=rcond, driver=str(driver)
        )
    except RuntimeError as exc:
        raise DataValidationError(f"torch.linalg.lstsq failed: {exc}") from exc
    theta = solved.solution.reshape(-1)
    if theta.numel() != FEATURE_DIM + 1 or not torch.isfinite(theta).all():
        raise NonFiniteError("Least-squares solution is non-finite or has the wrong shape")
    predicted_z = augmented @ theta
    if not torch.isfinite(predicted_z).all():
        raise NonFiniteError("Least-squares prediction became NaN/Inf")
    residual_z = predicted_z - targets_z.reshape(-1)
    predicted_years = target_transform.inverse_numpy(predicted_z.numpy())
    residual_years = predicted_years - targets_years_np
    singular = torch.linalg.svdvals(augmented)
    rank = int(torch.linalg.matrix_rank(augmented).item())
    singular_np = singular.detach().cpu().numpy()
    positive = singular_np[singular_np > np.finfo(np.float64).eps]
    condition = None if positive.size == 0 else float(positive.max() / positive.min())
    residual_z_np = residual_z.detach().cpu().numpy()
    mse_z = float(np.mean(residual_z_np ** 2))
    diagnostics = {
        "solver": "torch.linalg.lstsq",
        "compute_device": "cpu",
        "compute_dtype": "float64",
        "driver": str(driver),
        "rcond": None if rcond is None else float(rcond),
        "absolute_tolerance": float(absolute_tolerance),
        "rows": len(records),
        "feature_dim": FEATURE_DIM,
        "augmented_dim": FEATURE_DIM + 1,
        "matrix_rank": rank,
        "full_row_rank": rank == min(len(records), FEATURE_DIM + 1),
        "condition_number": condition,
        "singular_value_min": None if singular_np.size == 0 else float(singular_np.min()),
        "singular_value_max": None if singular_np.size == 0 else float(singular_np.max()),
        "singular_value_summary": _array_summary(singular_np),
        "target_transform": target_transform.mode,
        "target_mean_years": float(target_transform.mean_years),
        "target_std_years": float(target_transform.std_years),
        "solution_finite": True,
        "prediction_finite": True,
        "train_mse_z": mse_z,
        "train_mae_z": float(np.mean(np.abs(residual_z_np))),
        "train_rmse_z": float(np.sqrt(mse_z)),
        "train_mse_years": float(np.mean(residual_years ** 2)),
        "train_mae_years": float(np.mean(np.abs(residual_years))),
        "train_rmse_years": float(np.sqrt(np.mean(residual_years ** 2))),
        "train_mean_bias_years": float(np.mean(residual_years)),
        "train_prediction_min_years": float(np.min(predicted_years)),
        "train_prediction_max_years": float(np.max(predicted_years)),
        "explicit_residual_l2": float(torch.linalg.vector_norm(residual_z).item()),
        "explicit_residual_abs_max": float(torch.max(torch.abs(residual_z)).item()),
        "exact_interpolation_within_tolerance": bool(
            torch.max(torch.abs(residual_z)).item() <= float(absolute_tolerance)
        ),
        "residual_source": "explicit_recomputation_not_lstsq_residuals",
        "train_sample_count": len(sample_ids),
        "train_sample_id_sha256": hashlib.sha256("\n".join(sample_ids).encode()).hexdigest(),
        "validation_rows_used": 0,
        "test_rows_used": 0,
        "test_accessed": False,
    }
    state = {
        "weight": theta[:FEATURE_DIM].to(dtype=torch.float32).reshape(1, FEATURE_DIM).clone(),
        "bias": theta[FEATURE_DIM:].to(dtype=torch.float32).reshape(1).clone(),
    }
    return LeastSquaresReference(
        diagnostics=diagnostics,
        head_state_dict=state,
        predicted_years=np.asarray(predicted_years, dtype=np.float64),
        sample_ids=sample_ids,
    )


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


def create_age_head(
    seed: int, device: torch.device, initialization: str = "legacy_seeded_random"
) -> CachedFeatureClassifier:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        model = CachedFeatureClassifier(
            input_dim=FEATURE_DIM, num_classes=1, pooling=POOLING,
            use_mlp_head=False, head_dropout=0.0,
        )
    model = model.to(device)
    if initialization == "zeros":
        with torch.no_grad():
            model.head.weight.zero_()
            model.head.bias.zero_()
    elif initialization != "legacy_seeded_random":
        raise DataValidationError(f"Unsupported head initialization: {initialization}")
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
    elapsed_seconds: float, target_transform: TargetTransform | None = None,
) -> dict[str, Any]:
    forbidden = batch.get("forbidden_subjects", set())
    if set(batch["subject_id"]) & set(forbidden):
        raise AssertionError("Test subject appeared in a training batch")
    features = batch["feature"].to(device)
    targets_years = batch["label"].to(device).view(-1)
    transform = target_transform or TargetTransform(
        RAW_TARGET_TRANSFORM, {}, 0.0, 1.0
    )
    targets_z = transform.transform_tensor(targets_years)
    feature_finite = bool(torch.isfinite(features).all().item())
    if not feature_finite:
        raise NonFiniteError("Training batch feature contains NaN/Inf")
    optimizer.zero_grad(set_to_none=True)
    predictions_z = model.classify_features(features).view(-1)
    prediction_finite = bool(torch.isfinite(predictions_z).all().item())
    if not prediction_finite:
        raise NonFiniteError("Prediction became NaN/Inf")
    loss = criterion(predictions_z, targets_z)
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
    predictions_years = transform.inverse_tensor(predictions_z.detach())
    target_year_stats = _stats(targets_years)
    target_z_stats = _stats(targets_z)
    prediction_z_stats = _stats(predictions_z)
    prediction_year_stats = _stats(predictions_years)
    feature_norm_stats = _stats(torch.linalg.vector_norm(features.float(), dim=1))
    errors_z = predictions_z.detach() - targets_z
    errors_years = predictions_years - targets_years
    return {
        "experiment": experiment, "epoch": epoch, "global_step": global_step,
        "batch_index": batch_index, "batch_size": int(targets_years.numel()),
        "lr": float(optimizer.param_groups[0]["lr"]),
        "target_transform": transform.mode,
        "target_mean_years": float(transform.mean_years),
        "target_std_years": float(transform.std_years),
        "loss_mse_z": float(loss.item()),
        "batch_mae_z": float(errors_z.abs().mean().item()),
        "batch_rmse_z": float(torch.sqrt(torch.mean(errors_z ** 2)).item()),
        "batch_mae_years": float(errors_years.abs().mean().item()),
        "batch_rmse_years": float(torch.sqrt(torch.mean(errors_years ** 2)).item()),
        "batch_mean_bias_years": float(errors_years.mean().item()),
        "prediction_z_min": prediction_z_stats["min"],
        "prediction_z_max": prediction_z_stats["max"],
        "prediction_z_mean": prediction_z_stats["mean"],
        "prediction_z_std": prediction_z_stats["std"],
        "prediction_years_min": prediction_year_stats["min"],
        "prediction_years_max": prediction_year_stats["max"],
        "prediction_years_mean": prediction_year_stats["mean"],
        "prediction_years_std": prediction_year_stats["std"],
        "target_z_min": target_z_stats["min"], "target_z_max": target_z_stats["max"],
        "target_z_mean": target_z_stats["mean"], "target_z_std": target_z_stats["std"],
        "loss_mse": float(loss.item()), "mae_years": float(errors_years.abs().mean().item()),
        "rmse_years": float(torch.sqrt(torch.mean(errors_years ** 2)).item()),
        "target_min": target_year_stats["min"], "target_max": target_year_stats["max"],
        "target_mean": target_year_stats["mean"], "target_std": target_year_stats["std"],
        "prediction_min": prediction_year_stats["min"],
        "prediction_max": prediction_year_stats["max"],
        "prediction_mean": prediction_year_stats["mean"],
        "prediction_std": prediction_year_stats["std"],
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
    target_transform: TargetTransform | None = None,
    return_predictions: bool = False,
) -> dict[str, Any]:
    dataset = FeatureRecordDataset(records, forbidden_subjects)
    loader = DataLoader(
        dataset, batch_size=min(int(batch_size), len(dataset)), shuffle=False,
        num_workers=0, collate_fn=collate_records, drop_last=False,
    )
    model.eval()
    transform = target_transform or TargetTransform(RAW_TARGET_TRANSFORM, {}, 0.0, 1.0)
    truth_years: list[float] = []
    predicted_z: list[float] = []
    for batch in loader:
        if set(batch["subject_id"]) & set(forbidden_subjects):
            raise AssertionError("Test subject appeared in evaluation batch")
        features = batch["feature"].to(device)
        labels = batch["label"].to(device).view(-1)
        outputs_z = model.classify_features(features).view(-1)
        if not torch.isfinite(outputs_z).all():
            raise NonFiniteError("Evaluation prediction became NaN/Inf")
        truth_years.extend(labels.cpu().tolist())
        predicted_z.extend(outputs_z.cpu().tolist())
    targets_years = np.asarray(truth_years, dtype=np.float64)
    targets_z = transform.transform_numpy(targets_years)
    predictions_z = np.asarray(predicted_z, dtype=np.float64)
    predictions_years = transform.inverse_numpy(predictions_z)
    errors_z = predictions_z - targets_z
    errors_years = predictions_years - targets_years
    mse_z = float(np.mean(errors_z ** 2))
    mse_years = float(np.mean(errors_years ** 2))
    result = {
        "mse": mse_years, "mae": float(np.mean(np.abs(errors_years))),
        "rmse": float(np.sqrt(mse_years)),
        "mse_z": mse_z, "mae_z": float(np.mean(np.abs(errors_z))),
        "rmse_z": float(np.sqrt(mse_z)),
        "mae_years": float(np.mean(np.abs(errors_years))),
        "rmse_years": float(np.sqrt(mse_years)),
        "mean_bias_years": float(np.mean(errors_years)),
        "prediction_min": float(predictions_years.min()),
        "prediction_max": float(predictions_years.max()),
        "prediction_mean": float(predictions_years.mean()),
        "prediction_std": float(predictions_years.std()),
        "prediction_z_min": float(predictions_z.min()),
        "prediction_z_max": float(predictions_z.max()),
        "prediction_z_mean": float(predictions_z.mean()),
        "prediction_z_std": float(predictions_z.std()),
        "all_finite": bool(np.isfinite(predictions_years).all()),
        "sample_count": len(predictions_years),
    }
    if return_predictions:
        result["_prediction_years"] = predictions_years
        result["_sample_ids"] = tuple(row.sample_id for row in records)
    return result


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
    target_transform: TargetTransform | None = None,
) -> None:
    transform = target_transform or TargetTransform(RAW_TARGET_TRANSFORM, {}, 0.0, 1.0)
    torch.save({
        "checkpoint_kind": CHECKPOINT_KIND,
        "not_for_formal_evaluation": True,
        "test_accessed": False,
        "diagnostic_head_state_dict": state,
        "experiment": experiment, "epoch": int(epoch), "fold": int(fold_info["fold"]),
        "task": TASK, "pooling": POOLING, "feature_dim": FEATURE_DIM,
        "target_transform": transform.mode,
        "target_mean_years": float(transform.mean_years),
        "target_std_years": float(transform.std_years),
        "initialization": transform.initialization,
        "optimizer": "AdamW",
        "head_lr": float(args.head_lr), "weight_decay": float(args.weight_decay),
        "feature_cache_key": cache_key,
    }, path)


def _least_squares_checkpoint(
    path: Path, reference: LeastSquaresReference, experiment: str,
    fold_info: dict[str, Any], target_transform: TargetTransform,
    cache_key: str | None,
) -> None:
    torch.save({
        "checkpoint_kind": LEAST_SQUARES_CHECKPOINT_KIND,
        "not_for_formal_evaluation": True,
        "test_accessed": False,
        "diagnostic_least_squares_head_state_dict": reference.head_state_dict,
        "experiment": experiment,
        "fold": int(fold_info["fold"]),
        "task": TASK,
        "pooling": POOLING,
        "feature_dim": FEATURE_DIM,
        "target_transform": target_transform.mode,
        "target_mean_years": float(target_transform.mean_years),
        "target_std_years": float(target_transform.std_years),
        "solver": reference.diagnostics["solver"],
        "driver": reference.diagnostics["driver"],
        "feature_cache_key": cache_key,
    }, path)


def run_training_experiment(
    *, experiment: str, train_records: Sequence[FeatureRecord],
    validation_records: Sequence[FeatureRecord] | None, output_dir: Path,
    max_epochs: int, convergence_mae: float, args: argparse.Namespace,
    fold_info: dict[str, Any], logger: logging.Logger,
    target_transform: TargetTransform, cache_key: str | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    test_subjects = set(fold_info["test_subjects"])
    if ({row.subject_id for row in train_records} | {
        row.subject_id for row in (validation_records or [])
    }) & test_subjects:
        raise AssertionError("Test record supplied to diagnostic experiment")
    if any(row.subject_id not in set(fold_info["train_subjects"]) for row in train_records):
        raise AssertionError("Non-train record supplied to optimizer or least-squares diagnostic")
    lstsq = solve_train_only_linear_least_squares(
        train_records,
        target_transform,
        driver=str(getattr(args, "lstsq_driver", "gelsd")),
        rcond=getattr(args, "lstsq_rcond", None),
        absolute_tolerance=float(getattr(args, "lstsq_absolute_tolerance", 1e-6)),
    )
    if set(lstsq.sample_ids) != {row.sample_id for row in train_records}:
        raise AssertionError("Least-squares sample IDs differ from train records")
    _write_json(output_dir / "least_squares_diagnostics.json", lstsq.diagnostics)
    _least_squares_checkpoint(
        output_dir / "least_squares_diagnostic_head.pt", lstsq, experiment,
        fold_info, target_transform, cache_key,
    )
    device = torch.device(args.device)
    model = create_age_head(args.train_seed, device, target_transform.initialization)
    initial_parameters = _head_vector(model)
    initial_fingerprint = _head_fingerprint(model)
    initial_weight_norm = float(torch.linalg.vector_norm(model.head.weight.detach()).item())
    initial_bias = float(model.head.bias.detach().reshape(-1)[0].item())
    criterion = make_criterion(TASK)
    optimizer = torch.optim.AdamW(
        [{"params": model.head.parameters(), "name": "head"}],
        lr=float(args.head_lr), weight_decay=float(args.weight_decay),
    )
    initial_train = evaluate_records(
        model, train_records, args.batch_size, device, test_subjects, target_transform
    )
    initial_validation = (
        None if validation_records is None else
        evaluate_records(
            model, validation_records, args.batch_size, device, test_subjects, target_transform
        )
    )
    if target_transform.standardized:
        if initial_weight_norm != 0.0 or initial_bias != 0.0:
            raise DataValidationError("Z-score diagnostic head was not zero initialized")
        if max(
            abs(initial_train["prediction_z_min"]), abs(initial_train["prediction_z_max"])
        ) > 1e-7:
            raise DataValidationError("Zero-initialized head did not predict z=0")
        if abs(initial_train["prediction_mean"] - target_transform.mean_years) > 1e-6:
            raise DataValidationError("Zero-initialized head did not predict train mean age")
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
    best_train = {
        "mae": math.inf, "mse": math.inf, "rmse": math.inf,
        "mae_z": math.inf, "mse_z": math.inf, "rmse_z": math.inf, "epoch": None,
    }
    best_loss = {"mse_z": math.inf, "epoch": None}
    best_validation = {
        "mae": math.inf, "mse": math.inf, "rmse": math.inf,
        "mae_z": math.inf, "mse_z": math.inf, "rmse_z": math.inf, "epoch": None,
    }
    best_loss_state: dict[str, torch.Tensor] | None = None
    global_step = 0
    status = "completed_not_converged"
    stop_reason = "epoch_budget_exhausted"
    epoch_reaching_threshold = None
    epoch_reaching_mae_threshold = None
    epoch_reaching_lstsq_tolerance = None
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
                        target_transform=target_transform,
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
            train_metrics = evaluate_records(
                model, train_records, args.batch_size, device, test_subjects, target_transform
            )
            validation_metrics = (
                None if validation_records is None else
                evaluate_records(
                    model, validation_records, args.batch_size, device, test_subjects,
                    target_transform,
                )
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
                "optimizer_steps": global_step,
                "target_transform": target_transform.mode,
                "target_mean_years": target_transform.mean_years,
                "target_std_years": target_transform.std_years,
                "train_mse_z": train_metrics["mse_z"],
                "train_mae_z": train_metrics["mae_z"],
                "train_rmse_z": train_metrics["rmse_z"],
                "train_mse": train_metrics["mse"],
                "train_mae_years": train_metrics["mae"],
                "train_rmse_years": train_metrics["rmse"],
                "train_mean_bias_years": train_metrics["mean_bias_years"],
                "validation_mse_z": None if validation_metrics is None else validation_metrics["mse_z"],
                "validation_mae_z": None if validation_metrics is None else validation_metrics["mae_z"],
                "validation_rmse_z": None if validation_metrics is None else validation_metrics["rmse_z"],
                "validation_mse": None if validation_metrics is None else validation_metrics["mse"],
                "validation_mae_years": None if validation_metrics is None else validation_metrics["mae"],
                "validation_rmse_years": None if validation_metrics is None else validation_metrics["rmse"],
                "validation_mean_bias_years": (
                    None if validation_metrics is None else validation_metrics["mean_bias_years"]
                ),
                "prediction_z_min": train_metrics["prediction_z_min"],
                "prediction_z_max": train_metrics["prediction_z_max"],
                "prediction_z_mean": train_metrics["prediction_z_mean"],
                "prediction_z_std": train_metrics["prediction_z_std"],
                "prediction_years_min": train_metrics["prediction_min"],
                "prediction_years_max": train_metrics["prediction_max"],
                "prediction_years_mean": train_metrics["prediction_mean"],
                "prediction_years_std": train_metrics["prediction_std"],
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
                best_train = {
                    **{key: float(train_metrics[key]) for key in (
                        "mae", "mse", "rmse", "mae_z", "mse_z", "rmse_z"
                    )},
                    "epoch": epoch,
                }
            if train_metrics["mse_z"] < best_loss["mse_z"]:
                best_loss = {"mse_z": float(train_metrics["mse_z"]), "epoch": epoch}
                best_loss_state = _state_copy(model)
            if validation_metrics is not None and validation_metrics["mae"] < best_validation["mae"]:
                best_validation = {
                    **{key: float(validation_metrics[key]) for key in (
                        "mae", "mse", "rmse", "mae_z", "mse_z", "rmse_z"
                    )},
                    "epoch": epoch,
                }
            log_event(
                logger, "INFO", "diagnostic.epoch", "diagnostic epoch completed",
                experiment=experiment, epoch=f"{epoch}/{max_epochs}", steps=global_step,
                train_mse_z=round(train_metrics["mse_z"], 8),
                train_mae_years=round(train_metrics["mae"], 6),
                validation_mae=(None if validation_metrics is None else round(validation_metrics["mae"], 6)),
                best_train_mae_years=round(best_train["mae"], 6),
                least_squares_train_mae_years=round(lstsq.diagnostics["train_mae_years"], 10),
                adamw_to_lstsq_mae_gap=round(
                    best_train["mae"] - lstsq.diagnostics["train_mae_years"], 10
                ),
                prediction_range=[round(train_metrics["prediction_min"], 4), round(train_metrics["prediction_max"], 4)],
                weight_norm=round(parameters["head_weight_l2_norm"], 6),
                gradient_mean=round(float(np.mean(epoch_gradient_norms)), 6),
                gradient_max=round(float(np.max(epoch_gradient_norms)), 6),
                finite=all_finite, elapsed_seconds=round(time.monotonic() - started, 3),
            )
            mae_reached = train_metrics["mae"] <= float(convergence_mae)
            lstsq_reached = train_metrics["mse_z"] <= (
                float(lstsq.diagnostics["train_mse_z"])
                + float(getattr(args, "lstsq_absolute_tolerance", 1e-6))
            )
            if mae_reached and epoch_reaching_mae_threshold is None:
                epoch_reaching_mae_threshold = epoch
            if lstsq_reached and epoch_reaching_lstsq_tolerance is None:
                epoch_reaching_lstsq_tolerance = epoch
            required_lstsq_reached = lstsq_reached if target_transform.standardized else True
            if experiment == "mini_overfit" and mae_reached and required_lstsq_reached:
                status, stop_reason = "completed_converged", "mae_and_lstsq_threshold_reached"
                epoch_reaching_threshold = epoch
                log_event(
                    logger, "INFO", "diagnostic.converged", "mini-overfit threshold reached",
                    experiment=experiment, epoch=epoch, train_mae=train_metrics["mae"],
                    threshold=convergence_mae,
                    lstsq_absolute_tolerance=getattr(args, "lstsq_absolute_tolerance", 1e-6),
                )
                break
    if status == "failed_non_finite":
        final_train = last_valid_train
        final_validation = last_valid_validation
    else:
        final_train = evaluate_records(
            model, train_records, args.batch_size, device, test_subjects,
            target_transform, return_predictions=True,
        )
        final_validation = (
            None if validation_records is None else
            evaluate_records(
                model, validation_records, args.batch_size, device, test_subjects,
                target_transform,
            )
        )
    if status not in {"failed_non_finite", "failed_runtime"}:
        threshold = convergence_mae
        reached_required_optimum = (
            best_loss["mse_z"] <= float(lstsq.diagnostics["train_mse_z"])
            + float(getattr(args, "lstsq_absolute_tolerance", 1e-6))
        ) if target_transform.standardized and experiment == "mini_overfit" else True
        if best_train["mae"] <= threshold and reached_required_optimum:
            status = "completed_converged"
            if epoch_reaching_threshold is None:
                epoch_reaching_threshold = best_train["epoch"]
        else:
            status = "completed_not_converged"
    if best_loss_state is not None and experiment == "full_train":
        _diagnostic_checkpoint(
            output_dir / "minimum_train_loss_diagnostic_head.pt", best_loss_state, experiment,
            int(best_loss["epoch"]), fold_info, args, cache_key, target_transform,
        )
        _diagnostic_checkpoint(
            output_dir / "final_diagnostic_head.pt", _state_copy(model), experiment,
            int(epoch_rows[-1]["epoch"] if epoch_rows else 0), fold_info, args, cache_key,
            target_transform,
        )
    slope = _last_n_slope(train_mae_history, args.plateau_window)
    plateau = bool(slope is not None and abs(slope) <= float(args.plateau_slope_tolerance))
    final_prediction_years = np.asarray(
        final_train.get("_prediction_years", np.asarray([], dtype=np.float64)), dtype=np.float64
    )
    final_sample_ids = tuple(final_train.get("_sample_ids", ()))
    if final_sample_ids and final_sample_ids != lstsq.sample_ids:
        raise AssertionError("AdamW and least-squares prediction row order differs")
    prediction_diff = (
        final_prediction_years - lstsq.predicted_years
        if final_prediction_years.size == lstsq.predicted_years.size else None
    )
    final_train_public = {key: value for key, value in final_train.items() if not key.startswith("_")}
    lstsq_mse_z = float(lstsq.diagnostics["train_mse_z"])
    tolerance = float(getattr(args, "lstsq_absolute_tolerance", 1e-6))
    lstsq_optimum_reached = bool(
        best_loss["epoch"] is not None and best_loss["mse_z"] <= lstsq_mse_z + tolerance
    )
    result = {
        "experiment": experiment, "status": status, "stop_reason": stop_reason,
        "target_transform": target_transform.mode,
        "target_mean_years": float(target_transform.mean_years),
        "target_std_years": float(target_transform.std_years),
        "initialization": target_transform.initialization,
        "initial_weight_norm": initial_weight_norm,
        "initial_bias": initial_bias,
        "initial_prediction_z": initial_train["prediction_z_mean"],
        "initial_prediction_age_years": initial_train["prediction_mean"],
        "initial_head_fingerprint": initial_fingerprint,
        "final_head_fingerprint": _head_fingerprint(model),
        "initial_loss": initial_train["mse_z"], "final_loss": final_train_public["mse_z"],
        "minimum_loss": None if best_loss["epoch"] is None else best_loss["mse_z"],
        "epoch_of_minimum_train_loss": best_loss["epoch"],
        "initial_mae": initial_train["mae"], "final_mae": final_train_public["mae"],
        "minimum_mae": None if best_train["epoch"] is None else best_train["mae"],
        "initial_rmse": initial_train["rmse"], "final_rmse": final_train_public["rmse"],
        "initial_mse_z": initial_train["mse_z"],
        "final_mse_z": final_train_public["mse_z"],
        "minimum_mse_z": None if best_loss["epoch"] is None else best_loss["mse_z"],
        "initial_mae_z": initial_train["mae_z"],
        "final_mae_z": final_train_public["mae_z"],
        "minimum_mae_z": None if best_train["epoch"] is None else best_train["mae_z"],
        "minimum_rmse": (
            None if not epoch_rows else min(float(row["train_rmse_years"]) for row in epoch_rows)
        ),
        "epoch_of_minimum_train_mae": best_train["epoch"],
        "epoch_reaching_threshold": epoch_reaching_threshold,
        "epoch_reaching_mae_threshold": epoch_reaching_mae_threshold,
        "epoch_reaching_lstsq_tolerance": epoch_reaching_lstsq_tolerance,
        "converged": status == "completed_converged",
        "loss_reduction_fraction": (
            None if initial_train["mse_z"] == 0 else
            float((initial_train["mse_z"] - final_train_public["mse_z"]) / initial_train["mse_z"])
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
        "least_squares": lstsq.diagnostics,
        "least_squares_train_mae_years": lstsq.diagnostics["train_mae_years"],
        "adamw_min_train_mae_years": None if best_train["epoch"] is None else best_train["mae"],
        "adamw_final_train_mae_years": final_train_public["mae"],
        "adamw_min_minus_lstsq_mae": (
            None if best_train["epoch"] is None else
            float(best_train["mae"] - lstsq.diagnostics["train_mae_years"])
        ),
        "adamw_final_minus_lstsq_mae": float(
            final_train_public["mae"] - lstsq.diagnostics["train_mae_years"]
        ),
        "least_squares_train_mse_z": lstsq_mse_z,
        "adamw_min_train_mse_z": None if best_loss["epoch"] is None else best_loss["mse_z"],
        "adamw_final_train_mse_z": final_train_public["mse_z"],
        "adamw_min_minus_lstsq_mse_z": (
            None if best_loss["epoch"] is None else float(best_loss["mse_z"] - lstsq_mse_z)
        ),
        "adamw_final_minus_lstsq_mse_z": float(final_train_public["mse_z"] - lstsq_mse_z),
        "adamw_vs_lstsq_prediction_mae_years": (
            None if prediction_diff is None else float(np.mean(np.abs(prediction_diff)))
        ),
        "adamw_vs_lstsq_prediction_max_abs_diff_years": (
            None if prediction_diff is None else float(np.max(np.abs(prediction_diff)))
        ),
        "lstsq_absolute_tolerance": tolerance,
        "lstsq_optimum_reached": lstsq_optimum_reached,
        "train_mae_thresholds_reached": {
            "0.10": bool(best_train["mae"] <= 0.10),
            "0.25": bool(best_train["mae"] <= 0.25),
            "0.50": bool(best_train["mae"] <= 0.50),
        },
        "final_gradient_norm_mean": (
            None if not epoch_rows else float(epoch_rows[-1]["gradient_norm_mean"])
        ),
        "final_gradient_significantly_nonzero": bool(
            epoch_rows and float(epoch_rows[-1]["gradient_norm_mean"]) > 1e-8
        ),
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
    references = [mini.get("least_squares") or {}, full.get("least_squares") or {}]
    if not all(reference.get("exact_interpolation_within_tolerance", False) for reference in references):
        return {
            "category": "A", "code": "least_squares_cannot_fit_train",
            "message_zh": "train-only解析最小二乘也未达到低残差；需检查rank、feature/target关联、solver或数据。",
        }
    if (
        not mini.get("converged", False)
        or not mini.get("lstsq_optimum_reached", False)
        or not full.get("train_mae_thresholds_reached", {}).get("0.50", False)
    ):
        return {
            "category": "B", "code": "adamw_above_reachable_train_optimum",
            "message_zh": "解析解可近零误差拟合，但AdamW仍明显高于可达解；主要问题是优化速度、尺度或训练配置，不能据此判断年龄泛化。",
        }
    final_val = (full.get("final_validation") or {}).get("mae_years")
    minimum_train = full.get("adamw_min_train_mae_years")
    validation_gap = bool(
        final_val is not None and minimum_train is not None
        and final_val > max(0.50, 2.0 * float(minimum_train))
    )
    if validation_gap:
        return {
            "category": "D", "code": "low_train_error_validation_gap",
            "message_zh": "train误差已低但单个validation被试仍差，更像记忆训练集或跨被试泛化失败；因validation仅一名被试，结论需谨慎。",
        }
    return {
        "category": "C", "code": "adamw_near_reachable_train_optimum",
        "message_zh": "AdamW已接近解析训练最优且train误差低，迭代优化问题基本解决；之后才可用validation/OOF评估跨被试泛化。",
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
    transform = summary["target_transform_metadata"]
    ages = ", ".join(
        f"{subject}={age:g}" for subject, age in transform["train_subject_ages_years"].items()
    )
    mini_ls = mini.get("least_squares") or {}
    full_ls = full.get("least_squares") or {}
    initial_val = full.get("initial_validation") or {}
    minimum_val = full.get("minimum_validation") or {}
    final_val = full.get("final_validation") or {}
    supports_formal_change = bool(
        mini.get("converged")
        and (mini.get("minimum_mae") or math.inf) <= 0.10
        and mini.get("lstsq_optimum_reached")
        and summary.get("all_values_finite")
        and not mini.get("test_accessed")
    )
    report = f"""# 年龄线性头收敛诊断

本次是诊断实验，不是正式性能实验。未创建test dataset/DataLoader，未读取test feature、prediction或label，不能从本结果报告test MAE。

## Fold与target transform

- fold: `{summary['fold']}`
- train subjects及年龄: `{ages}`
- validation / excluded test subject: `{summary['validation_subject']}` / `{summary['test_subject']}`
- target transform: `{summary['target_transform']}`
- train-only subject mean/std: `{summary['target_mean_years']}` / `{summary['target_std_years']}`（ddof=0）
- transform: `z = (age_years - mu_train) / sigma_train`
- inverse: `age_years = z * sigma_train + mu_train`
- 统计权重: `unique train subjects, equal subject weight`
- validation/test用于统计: `False / False`
- 初始化: `{summary['initialization']}`；初始z/年龄预测: `{mini.get('initial_prediction_z')}` / `{mini.get('initial_prediction_age_years')}`

## Train-only解析最小二乘参考

| 实验 | rows / rank | condition number | LS train MAE / RMSE (years) | exact interpolation |
|---|---:|---:|---:|---:|
| mini | {mini_ls.get('rows')} / {mini_ls.get('matrix_rank')} | {mini_ls.get('condition_number')} | {mini_ls.get('train_mae_years')} / {mini_ls.get('train_rmse_years')} | {mini_ls.get('exact_interpolation_within_tolerance')} |
| full | {full_ls.get('rows')} / {full_ls.get('matrix_rank')} | {full_ls.get('condition_number')} | {full_ls.get('train_mae_years')} / {full_ls.get('train_rmse_years')} | {full_ls.get('exact_interpolation_within_tolerance')} |

解析解仅证明这些train rows在线性系统中可拟合，不证明feature具有可泛化的年龄规律。512维feature面对不足513个train窗口时具有很强的记忆能力。

## AdamW与可达解

| 实验 | AdamW minimum / final MAE (years) | LS MAE | minimum / final gap | last-{mini.get('plateau_window')} epoch slope | final gradient mean |
|---|---:|---:|---:|---:|---:|
| mini | {mini.get('adamw_min_train_mae_years')} / {mini.get('adamw_final_train_mae_years')} | {mini.get('least_squares_train_mae_years')} | {mini.get('adamw_min_minus_lstsq_mae')} / {mini.get('adamw_final_minus_lstsq_mae')} | {mini.get('last_n_epoch_mae_slope')} | {mini.get('final_gradient_norm_mean')} |
| full | {full.get('adamw_min_train_mae_years')} / {full.get('adamw_final_train_mae_years')} | {full.get('least_squares_train_mae_years')} | {full.get('adamw_min_minus_lstsq_mae')} / {full.get('adamw_final_minus_lstsq_mae')} | {full.get('last_n_epoch_mae_slope')} | {full.get('final_gradient_norm_mean')} |

- optimizer / LR / weight decay: `AdamW / {summary['head_lr']} / {summary['weight_decay']}`
- gradient clipping: `{summary['gradient_clipping']}`
- mini status / reached 0.10 years / reached LS tolerance: `{mini['status']}` / `{mini.get('train_mae_thresholds_reached', {}).get('0.10')}` / `{mini.get('lstsq_optimum_reached')}`
- full reached 0.10 / 0.25 / 0.50 years: `{full.get('train_mae_thresholds_reached')}`
- any non-finite value: `{not summary['all_values_finite']}`

## Validation观察

- initial validation MAE: `{initial_val.get('mae_years')}` years
- minimum validation MAE / epoch: `{minimum_val.get('mae')}` / `{minimum_val.get('epoch')}`
- final validation MAE: `{final_val.get('mae_years')}` years

Validation只有一名被试，只用于观察轨迹，不用于解析求解、target统计、checkpoint选择或正式性能结论。

## Test隔离与结论

- test dataset/DataLoader created: `{summary['test_dataset_created']} / {summary['test_dataloader_created']}`
- test query/materialized/forward/metrics: `False / {summary['test_rows_materialized']} / {summary['test_forward_count']} / {summary['test_metrics_computed']}`

自动结论 `{conclusion['category']}`：{conclusion['message_zh']}

本次是否支持下一阶段考虑修改正式CV7：`{supports_formal_change}`。此前正式cached linear-probe真实使用`SGD`，本诊断使用`AdamW`；两者不能静默等同，本次不会修改正式CV7。

所有迭代与解析checkpoint均标记`not_for_formal_evaluation=true`，不得用于正式模型结论。
"""
    path.write_text(report, encoding="utf-8")


def _access_guards() -> dict[str, Any]:
    return {
        "test_dataset_created": False, "test_dataloader_created": False,
        "test_query_executed": False,
        "test_rows_materialized": 0, "test_forward_count": 0,
        "test_metrics_computed": False,
    }


def verify_trajectory_artifacts(output_dir: Path, summary: dict[str, Any]) -> bool:
    """Fail closed if streamed trajectories disagree with their summaries."""

    required = [
        "feature_diagnostics.json", "feature_dimension_statistics.csv",
        "selected_mini_samples.csv", "split_subjects.csv", "target_transform.json",
        "mini_overfit/step_trajectory.csv", "mini_overfit/epoch_trajectory.csv",
        "mini_overfit/least_squares_diagnostics.json",
        "full_train/step_trajectory.csv", "full_train/epoch_trajectory.csv",
        "full_train/least_squares_diagnostics.json",
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
    checkpoints = [
        checkpoint
        for experiment in ("mini_overfit", "full_train")
        for checkpoint in (output_dir / experiment).glob("*_diagnostic_head.pt")
    ]
    for checkpoint in checkpoints:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        kind = payload.get("checkpoint_kind")
        safe_kind = kind in {CHECKPOINT_KIND, LEAST_SQUARES_CHECKPOINT_KIND}
        forbidden_formal_state = set(payload) & {"head_state_dict", "model_state_dict"}
        expected_state = (
            "diagnostic_head_state_dict" if kind == CHECKPOINT_KIND
            else "diagnostic_least_squares_head_state_dict"
        )
        if (
            not safe_kind
            or expected_state not in payload
            or payload.get("not_for_formal_evaluation") is not True
            or payload.get("test_accessed") is not False
            or forbidden_formal_state
        ):
            raise RuntimeError(f"Unsafe diagnostic checkpoint metadata: {checkpoint}")
    transform = json.loads((output_dir / "target_transform.json").read_text(encoding="utf-8"))
    if (
        set(transform["train_subject_ids"]) != set(summary["train_subjects"])
        or transform["validation_subject"] in transform["train_subject_ids"]
        or transform["test_subject"] in transform["train_subject_ids"]
        or transform["validation_used_for_target_statistics"] is not False
        or transform["test_used_for_target_statistics"] is not False
    ):
        raise RuntimeError("Unsafe target transform statistics scope")
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
    target_transform = build_train_only_target_transform(
        train_records, fold_info, getattr(args, "target_transform", RAW_TARGET_TRANSFORM)
    )
    target_transform_payload = target_transform.as_dict(fold_info)
    _write_json(output_dir / "target_transform.json", target_transform_payload)
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
        logger=logger, target_transform=target_transform, cache_key=cache_key,
    )
    if mini["status"] == "failed_non_finite":
        full_dir = output_dir / "full_train"
        full_dir.mkdir(parents=True, exist_ok=False)
        _write_csv(full_dir / "step_trajectory.csv", [], STEP_FIELDS)
        _write_csv(full_dir / "epoch_trajectory.csv", [], EPOCH_FIELDS)
        full_lstsq = solve_train_only_linear_least_squares(
            train_records, target_transform,
            driver=str(getattr(args, "lstsq_driver", "gelsd")),
            rcond=getattr(args, "lstsq_rcond", None),
            absolute_tolerance=float(getattr(args, "lstsq_absolute_tolerance", 1e-6)),
        )
        _write_json(full_dir / "least_squares_diagnostics.json", full_lstsq.diagnostics)
        _least_squares_checkpoint(
            full_dir / "least_squares_diagnostic_head.pt", full_lstsq, "full_train",
            fold_info, target_transform, cache_key,
        )
        full = {
            "experiment": "full_train", "status": "failed_non_finite",
            "stop_reason": "skipped_after_mini_non_finite", "converged": False,
            "all_values_finite": False, "optimizer_steps": 0,
            "step_trajectory_rows": 0, "epoch_trajectory_rows": 0,
            "test_accessed": False, "least_squares": full_lstsq.diagnostics,
        }
    else:
        log_event(logger, "INFO", "diagnostic.stage5", "[stage 5] full-train convergence")
        full = run_training_experiment(
            experiment="full_train", train_records=train_records,
            validation_records=validation_records, output_dir=output_dir / "full_train",
            max_epochs=args.full_max_epochs, convergence_mae=args.full_train_good_mae,
            args=args, fold_info=fold_info, logger=logger,
            target_transform=target_transform, cache_key=cache_key,
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
        "task": TASK, "target_transform": target_transform.mode, "pooling": POOLING,
        "target_transform_metadata": target_transform_payload,
        "target_statistics_scope": "unique_train_subjects_only",
        "target_statistics_weighting": "equal_subject_weight",
        "target_mean_years": float(target_transform.mean_years),
        "target_std_years": float(target_transform.std_years),
        "target_std_ddof": 0,
        "target_inverse_transform_applied": bool(target_transform.standardized),
        "validation_used_for_target_statistics": False,
        "test_used_for_target_statistics": False,
        "initialization": target_transform.initialization,
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
    summary["formal_age_cv7_change_supported"] = bool(
        mini.get("converged")
        and (mini.get("minimum_mae") or math.inf) <= 0.10
        and mini.get("lstsq_optimum_reached")
        and summary["all_values_finite"]
        and summary["test_rows_materialized"] == 0
    )
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
            "protocol": "age_head_convergence_diagnostic_v2_train_zscore_lstsq",
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
                "optimizer_resolved_from": "scripts/downstream_50m/largeScale_fineTune.py:make_optimizer linear_probe branch",
                "diagnostic_optimizer_differs": True,
            },
            "formal_age_cv7_modified": False,
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
