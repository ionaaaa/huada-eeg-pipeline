"""Subject-mean frozen-feature age Ridge with strictly nested subject LOSO.

This entry point intentionally contains no backbone, optimizer, epoch, gradient-head,
or window-level evaluation path.  Windows are used once to construct deterministic
subject means; all regression and metrics operate on seven independent subjects.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import lance
import numpy as np
import pyarrow as pa
import torch

from feature_cache import (
    CACHE_DTYPE,
    MANIFEST_NAME,
    MEAN_CACHE_NAME,
    PROHIBITED_CACHE_COLUMNS,
    sha256_file,
    sha256_json,
    validate_cache_manifest,
)
from run_logging import log_event, log_exception, setup_run_logging


FEATURE_DIM = 512
DEFAULT_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
EXPECTED_SUBJECTS = {f"SUB{index}" for index in range(1, 9)}
ELIGIBLE_SUBJECT_COUNT = 7
METRIC_TOLERANCE = 1e-9
TARGET_STD_EPSILON = 1e-12
PROTOCOL_VERSION = "subject-mean-ridge-nested-loso-v1"
ISSUE_FIELDS = ["severity", "stage", "outer_fold", "code", "message", "value", "threshold"]
ERROR_HEADER = "timestamp_utc\tstage\terror_type\tmessage\n"


class RidgeValidationError(ValueError):
    """Fail-closed input, isolation, numerical, or artifact validation error."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def subject_number(subject_id: str) -> int:
    match = re.fullmatch(r"SUB(\d+)", str(subject_id).upper())
    if match is None:
        raise RidgeValidationError(f"Expected SUB<number> subject ID, got {subject_id!r}")
    return int(match.group(1))


def canonical_subject_order(subject_ids: Iterable[str]) -> list[str]:
    return sorted({str(value) for value in subject_ids}, key=subject_number)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(array))
    return hashlib.sha256(value.tobytes()).hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: str | Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def git_state(project_root: Path) -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_root, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=project_root, text=True,
            stderr=subprocess.DEVNULL,
        ).strip())
        return commit, dirty
    except Exception:
        return None, None


@dataclass(frozen=True)
class WindowFeatureRow:
    sample_id: str
    subject_id: str
    recording_id: str | None
    window_index: int
    age: float | None
    feature: np.ndarray


@dataclass(frozen=True)
class SubjectFeatureRow:
    subject_id: str
    age: float | None
    eligibility: bool
    exclusion_reason: str
    window_count: int
    feature: np.ndarray
    sample_ids: tuple[str, ...]
    window_feature_dtype: str = "float32"
    aggregation_dtype: str = "float64"


def aggregate_subject_features(rows: Sequence[WindowFeatureRow]) -> list[SubjectFeatureRow]:
    if not rows:
        raise RidgeValidationError("No window feature rows were supplied")
    sample_ids = [row.sample_id for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise RidgeValidationError("Duplicate sample_id in source/cache join")
    grouped: dict[str, list[WindowFeatureRow]] = {}
    for row in rows:
        feature = np.asarray(row.feature)
        if feature.shape != (FEATURE_DIM,):
            raise RidgeValidationError(f"Feature dimension is not 512 for {row.sample_id}")
        if feature.dtype != np.float32:
            raise RidgeValidationError(f"Window feature dtype is not float32 for {row.sample_id}")
        if not np.isfinite(feature).all():
            raise RidgeValidationError(f"Non-finite window feature for {row.sample_id}")
        grouped.setdefault(row.subject_id, []).append(row)
    result: list[SubjectFeatureRow] = []
    for subject in canonical_subject_order(grouped):
        subject_rows = grouped[subject]
        non_null_ages = {float(row.age) for row in subject_rows if row.age is not None}
        if any(not math.isfinite(age) for age in non_null_ages):
            raise RidgeValidationError(f"Non-finite age within {subject}")
        if len(non_null_ages) > 1:
            raise RidgeValidationError(f"Conflicting ages within {subject}: {sorted(non_null_ages)}")
        if non_null_ages and any(row.age is None for row in subject_rows):
            raise RidgeValidationError(f"Partially missing age within {subject}")
        age = next(iter(non_null_ages)) if non_null_ages else None
        stacked = np.stack([np.asarray(row.feature, dtype=np.float64) for row in subject_rows])
        mean_feature = np.mean(stacked, axis=0, dtype=np.float64)
        if mean_feature.shape != (FEATURE_DIM,) or not np.isfinite(mean_feature).all():
            raise RidgeValidationError(f"Non-finite subject mean feature for {subject}")
        result.append(SubjectFeatureRow(
            subject_id=subject,
            age=age,
            eligibility=age is not None,
            exclusion_reason="" if age is not None else "missing_age",
            window_count=len(subject_rows),
            feature=mean_feature,
            sample_ids=tuple(sorted(row.sample_id for row in subject_rows)),
        ))
    return result


class ExistingMeanCacheSource:
    """Require and fully validate an existing label-free mean cache."""

    def __init__(self, input_lance: str | Path, cache_lance: str | Path, manifest_path: str | Path):
        self.input_lance = Path(input_lance)
        self.cache_lance = Path(cache_lance)
        self.manifest_path = Path(manifest_path)
        if not self.input_lance.exists():
            raise RidgeValidationError(f"Input Lance does not exist: {self.input_lance}")
        if not self.cache_lance.exists():
            raise RidgeValidationError(f"Mean feature cache does not exist: {self.cache_lance}")
        if not self.manifest_path.is_file():
            raise RidgeValidationError(f"Cache manifest does not exist: {self.manifest_path}")
        cache_dir = self.cache_lance.parent
        if self.cache_lance.resolve() != (cache_dir / MEAN_CACHE_NAME).resolve():
            raise RidgeValidationError("Only canonical mean_features.lance is accepted")
        if self.manifest_path.resolve() != (cache_dir / MANIFEST_NAME).resolve():
            raise RidgeValidationError("Cache manifest must be cache_manifest.json beside mean cache")
        try:
            self.manifest = validate_cache_manifest(cache_dir)
        except Exception as exc:
            raise RidgeValidationError(str(exc)) from exc
        requested = set(self.manifest.get("requested_poolings", ()))
        if requested != {"mean"}:
            raise RidgeValidationError("Ridge experiment requires a mean-only feature cache")
        if int(self.manifest.get("mean_feature_dim", -1)) != FEATURE_DIM:
            raise RidgeValidationError("Mean cache feature dimension must be 512")
        if str(self.manifest.get("dtype")) != CACHE_DTYPE:
            raise RidgeValidationError("Mean cache dtype must be float32")
        identity = self.manifest.get("cache_identity")
        if identity is not None and sha256_json(identity) != self.manifest.get("cache_key"):
            raise RidgeValidationError("Feature cache identity hash mismatch")
        manifest_ids = self.manifest.get("sample_ids")
        if not isinstance(manifest_ids, list):
            raise RidgeValidationError("Feature cache manifest has no sample_ids integrity list")
        normalized_manifest_ids = [str(value) for value in manifest_ids]
        if len(normalized_manifest_ids) != len(set(normalized_manifest_ids)):
            raise RidgeValidationError("Feature cache manifest contains duplicate sample_id")
        self.source = lance.dataset(str(self.input_lance))
        self.cache = lance.dataset(str(self.cache_lance))
        source_required = {"sample_id", "subject_id", "recording_id", "window_index", "age"}
        cache_required = {"sample_id", "subject_id", "recording_id", "feature", "num_valid_tokens"}
        if not source_required.issubset(self.source.schema.names):
            raise RidgeValidationError(
                f"Input Lance missing fields: {sorted(source_required-set(self.source.schema.names))}"
            )
        if not cache_required.issubset(self.cache.schema.names):
            raise RidgeValidationError("Mean cache schema is missing required fields")
        if set(self.cache.schema.names) & PROHIBITED_CACHE_COLUMNS:
            raise RidgeValidationError("Feature cache contains task/split/prediction columns")
        feature_type = self.cache.schema.field("feature").type
        if (
            not pa.types.is_fixed_size_list(feature_type)
            or feature_type.list_size != FEATURE_DIM
            or feature_type.value_type != pa.float32()
        ):
            raise RidgeValidationError("Mean cache feature must be fixed_size_list<float32>[512]")
        source_count = int(self.source.count_rows())
        cache_count = int(self.cache.count_rows())
        expected_count = int(self.manifest.get("input_row_count", -1))
        if source_count != cache_count or source_count != expected_count:
            raise RidgeValidationError("Source/cache/manifest row counts differ")
        if int(self.manifest.get("mean_row_count", -1)) != cache_count:
            raise RidgeValidationError("Mean cache row count differs from manifest")
        if len(normalized_manifest_ids) != expected_count:
            raise RidgeValidationError("Manifest sample_id count differs from row count")

    def load_and_join(self) -> list[WindowFeatureRow]:
        source_rows = self.source.to_table(
            columns=["sample_id", "subject_id", "recording_id", "window_index", "age"]
        ).to_pylist()
        cache_rows = self.cache.to_table(
            columns=["sample_id", "subject_id", "recording_id", "feature"]
        ).to_pylist()
        source_ids = [str(row["sample_id"]) for row in source_rows]
        cache_ids = [str(row["sample_id"]) for row in cache_rows]
        manifest_ids = [str(value) for value in self.manifest["sample_ids"]]
        if len(source_ids) != len(set(source_ids)) or len(cache_ids) != len(set(cache_ids)):
            raise RidgeValidationError("Duplicate sample_id in input Lance or feature cache")
        if set(source_ids) != set(cache_ids) or set(source_ids) != set(manifest_ids):
            raise RidgeValidationError("Input/cache/manifest sample_id sets differ")
        cache_by_id = {str(row["sample_id"]): row for row in cache_rows}
        joined: list[WindowFeatureRow] = []
        for source in source_rows:
            sample_id = str(source["sample_id"])
            cached = cache_by_id[sample_id]
            subject = str(source["subject_id"])
            if str(cached["subject_id"]) != subject:
                raise RidgeValidationError(f"Cache/source subject mismatch for {sample_id}")
            source_recording = source.get("recording_id")
            cached_recording = cached.get("recording_id")
            if source_recording != cached_recording:
                raise RidgeValidationError(f"Cache/source recording mismatch for {sample_id}")
            feature = np.asarray(cached["feature"], dtype=np.float32).reshape(-1)
            joined.append(WindowFeatureRow(
                sample_id=sample_id,
                subject_id=subject,
                recording_id=None if source_recording is None else str(source_recording),
                window_index=int(source["window_index"]),
                age=None if source["age"] is None else float(source["age"]),
                feature=feature,
            ))
        return joined


def validate_huada_subject_inventory(subject_rows: Sequence[SubjectFeatureRow]) -> list[SubjectFeatureRow]:
    by_subject = {row.subject_id: row for row in subject_rows}
    if set(by_subject) != EXPECTED_SUBJECTS:
        raise RidgeValidationError(f"Unexpected Huada subject set: {canonical_subject_order(by_subject)}")
    if by_subject["SUB5"].age is not None or by_subject["SUB5"].eligibility:
        raise RidgeValidationError("SUB5 must be excluded because age is null")
    eligible = [row for row in subject_rows if row.eligibility]
    if len(eligible) != ELIGIBLE_SUBJECT_COUNT:
        raise RidgeValidationError(f"Expected 7 age-eligible subjects, got {len(eligible)}")
    if any(row.subject_id == "SUB5" for row in eligible):
        raise RidgeValidationError("SUB5 unexpectedly entered eligible subjects")
    return sorted(eligible, key=lambda row: subject_number(row.subject_id))


def build_nested_loso_splits(subject_ids: Iterable[str]) -> list[dict[str, Any]]:
    subjects = canonical_subject_order(subject_ids)
    if len(subjects) != ELIGIBLE_SUBJECT_COUNT:
        raise RidgeValidationError("Nested LOSO requires exactly seven eligible subjects")
    rows: list[dict[str, Any]] = []
    for outer_fold, outer_test in enumerate(subjects):
        development = [subject for subject in subjects if subject != outer_test]
        for inner_fold, inner_validation in enumerate(development):
            inner_train = [subject for subject in development if subject != inner_validation]
            if (len(inner_train), len(development)) != (5, 6):
                raise AssertionError("Invalid nested LOSO fold sizes")
            rows.append({
                "outer_fold": outer_fold,
                "outer_test_subject_id": outer_test,
                "outer_development_subject_ids": list(development),
                "inner_fold": inner_fold,
                "inner_train_subject_ids": inner_train,
                "inner_validation_subject_id": inner_validation,
            })
    if len(rows) != 42:
        raise AssertionError("Nested LOSO must contain 42 inner folds")
    return rows


def fit_train_preprocessing(
    features: np.ndarray, targets: np.ndarray, *, epsilon: float = TARGET_STD_EPSILON,
) -> dict[str, Any]:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64).reshape(-1)
    if x.ndim != 2 or x.shape != (y.size, FEATURE_DIM) or y.size == 0:
        raise RidgeValidationError("Invalid subject-level preprocessing inputs")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise RidgeValidationError("Non-finite preprocessing input")
    center = np.mean(x, axis=0, dtype=np.float64)
    target_mean = float(np.mean(y))
    target_std = float(np.std(y, ddof=0))
    if not math.isfinite(target_std) or target_std <= float(epsilon):
        raise RidgeValidationError("Train subject target standard deviation is zero or invalid")
    return {
        "feature_center": center,
        "feature_center_sha256": array_sha256(center),
        "target_mean": target_mean,
        "target_std": target_std,
        "target_std_ddof": 0,
        "feature_transform": "train_center",
        "target_transform": "train_subject_zscore",
    }


def solve_dual_ridge(features_centered: np.ndarray, targets_z: np.ndarray, alpha: float) -> dict[str, Any]:
    x_np = np.asarray(features_centered, dtype=np.float64)
    y_np = np.asarray(targets_z, dtype=np.float64).reshape(-1)
    if x_np.ndim != 2 or x_np.shape != (y_np.size, FEATURE_DIM):
        raise RidgeValidationError("Invalid Ridge design matrix shape")
    if not math.isfinite(float(alpha)) or float(alpha) <= 0:
        raise RidgeValidationError("Ridge alpha must be finite and positive")
    if not np.isfinite(x_np).all() or not np.isfinite(y_np).all():
        raise RidgeValidationError("Non-finite Ridge input")
    x = torch.from_numpy(np.ascontiguousarray(x_np)).to(device="cpu", dtype=torch.float64)
    y = torch.from_numpy(np.ascontiguousarray(y_np)).to(device="cpu", dtype=torch.float64)
    kernel = x @ x.T
    regularized = kernel + float(alpha) * torch.eye(x.shape[0], dtype=torch.float64)
    try:
        dual = torch.linalg.solve(regularized, y)
    except Exception as exc:
        raise RidgeValidationError(f"Ridge linear solve failed: {exc}") from exc
    coefficient = x.T @ dual
    fitted = x @ coefficient
    if not torch.isfinite(coefficient).all() or not torch.isfinite(fitted).all():
        raise RidgeValidationError("Ridge coefficient or prediction is non-finite")
    return {
        "coefficient": coefficient.numpy(),
        "train_prediction_z": fitted.numpy(),
        "solver": "torch.linalg.solve",
        "solver_form": "dual",
        "compute_device": "cpu",
        "compute_dtype": "float64",
        "fit_intercept": False,
        "alpha": float(alpha),
    }


def fit_ridge_for_subjects(features: np.ndarray, targets: np.ndarray, alpha: float) -> dict[str, Any]:
    preprocessing = fit_train_preprocessing(features, targets)
    centered = np.asarray(features, dtype=np.float64) - preprocessing["feature_center"]
    targets_z = (np.asarray(targets, dtype=np.float64) - preprocessing["target_mean"]) / preprocessing["target_std"]
    solution = solve_dual_ridge(centered, targets_z, alpha)
    return {**preprocessing, **solution}


def predict_age(model: dict[str, Any], feature: np.ndarray) -> float:
    value = np.asarray(feature, dtype=np.float64).reshape(-1)
    if value.shape != (FEATURE_DIM,) or not np.isfinite(value).all():
        raise RidgeValidationError("Invalid subject feature presented for prediction")
    prediction_z = float((value - model["feature_center"]) @ model["coefficient"])
    prediction = prediction_z * float(model["target_std"]) + float(model["target_mean"])
    if not math.isfinite(prediction):
        raise RidgeValidationError("Ridge prediction is non-finite")
    return prediction


def regression_metrics(true_values: Sequence[float], predicted_values: Sequence[float]) -> dict[str, Any]:
    truth = np.asarray(true_values, dtype=np.float64).reshape(-1)
    predicted = np.asarray(predicted_values, dtype=np.float64).reshape(-1)
    if truth.shape != predicted.shape or truth.size == 0:
        raise RidgeValidationError("Metric inputs must have matching non-empty shapes")
    if not np.isfinite(truth).all() or not np.isfinite(predicted).all():
        raise RidgeValidationError("Metric inputs must be finite")
    error = predicted - truth
    denominator = float(np.sum((truth - truth.mean()) ** 2))
    r2_defined = truth.size >= 2 and denominator > 0
    return {
        "subject_count": int(truth.size),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "r2": float(1.0 - np.sum(error ** 2) / denominator) if r2_defined else None,
        "r2_defined": bool(r2_defined),
        "mean_signed_error": float(np.mean(error)),
        "median_absolute_error": float(np.median(np.abs(error))),
        "max_absolute_error": float(np.max(np.abs(error))),
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    index = 0
    while index < values.size:
        end = index + 1
        while end < values.size and values[order[end]] == values[order[index]]:
            end += 1
        ranks[order[index:end]] = (index + end - 1) / 2.0 + 1.0
        index = end
    return ranks


def correlation_metric(left: Sequence[float], right: Sequence[float], *, spearman: bool) -> dict[str, Any]:
    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    name = "spearman_rho" if spearman else "pearson_r"
    if x.shape != y.shape or x.size < 2:
        return {name: None, f"{name}_defined": False, f"{name}_reason": "fewer_than_two_subjects"}
    if spearman:
        x, y = _average_ranks(x), _average_ranks(y)
    x_centered, y_centered = x - x.mean(), y - y.mean()
    denominator = float(np.sqrt(np.sum(x_centered ** 2) * np.sum(y_centered ** 2)))
    if denominator <= 0 or not math.isfinite(denominator):
        return {name: None, f"{name}_defined": False, f"{name}_reason": "constant_or_nonfinite_input"}
    value = float(np.sum(x_centered * y_centered) / denominator)
    return {name: value, f"{name}_defined": True, f"{name}_reason": None}


def complete_subject_metrics(truth: Sequence[float], predicted: Sequence[float]) -> dict[str, Any]:
    return {
        **regression_metrics(truth, predicted),
        **correlation_metric(truth, predicted, spearman=False),
        **correlation_metric(truth, predicted, spearman=True),
        "primary_metric": "subject_mae",
        "independent_unit": "subject",
    }


def select_alpha(results: Sequence[dict[str, Any]], tolerance: float = METRIC_TOLERANCE) -> dict[str, Any]:
    if not results:
        raise RidgeValidationError("No alpha results to select from")
    winner: dict[str, Any] | None = None
    for candidate in results:
        if winner is None:
            winner = candidate
            continue
        decided = False
        for field in ("mae", "rmse", "max_absolute_error"):
            left, right = float(candidate[field]), float(winner[field])
            if left < right - tolerance:
                winner, decided = candidate, True
                break
            if left > right + tolerance:
                decided = True
                break
        if not decided and float(candidate["alpha"]) > float(winner["alpha"]):
            winner = candidate
    assert winner is not None
    return dict(winner)


class GuardedOuterAccessor:
    """Prevent any split-specific outer-test use before selection is locked."""

    def __init__(self, rows: Sequence[SubjectFeatureRow], outer_test_subject_id: str):
        self._rows = {row.subject_id: row for row in rows}
        self.outer_test_subject_id = str(outer_test_subject_id)
        self.selection_locked = False
        self.selection_locked_at: str | None = None
        self.test_feature_accessed_at: str | None = None
        self.test_age_accessed_at: str | None = None
        self.test_prediction_computed_at: str | None = None
        self.test_feature_access_count = 0
        self.test_age_access_count = 0
        self.test_evaluation_count = 0
        self.selection_fingerprint: str | None = None

    def _row(self, subject_id: str) -> SubjectFeatureRow:
        if subject_id not in self._rows:
            raise RidgeValidationError(f"Unknown subject: {subject_id}")
        return self._rows[subject_id]

    def feature(self, subject_id: str) -> np.ndarray:
        subject_id = str(subject_id)
        if subject_id == self.outer_test_subject_id:
            if not self.selection_locked:
                raise RidgeValidationError("Outer test feature accessed before selection lock")
            if self.test_feature_access_count != 0:
                raise RidgeValidationError("Outer test feature accessed more than once")
            self.test_feature_access_count += 1
            self.test_feature_accessed_at = utc_now()
        return self._row(subject_id).feature.copy()

    def age(self, subject_id: str) -> float:
        subject_id = str(subject_id)
        if subject_id == self.outer_test_subject_id:
            if not self.selection_locked:
                raise RidgeValidationError("Outer test age accessed before selection lock")
            if self.test_age_access_count != 0:
                raise RidgeValidationError("Outer test age accessed more than once")
            self.test_age_access_count += 1
            self.test_age_accessed_at = utc_now()
        age = self._row(subject_id).age
        if age is None:
            raise RidgeValidationError(f"Missing age requested for {subject_id}")
        return float(age)

    def lock(self, selection_fingerprint: str) -> None:
        if self.selection_locked:
            raise RidgeValidationError("Selection was already locked")
        self.selection_fingerprint = str(selection_fingerprint)
        self.selection_locked = True
        self.selection_locked_at = utc_now()

    def mark_test_prediction(self) -> None:
        if not self.selection_locked or self.test_feature_access_count != 1 or self.test_age_access_count != 1:
            raise RidgeValidationError("Test prediction attempted without exactly one locked feature/age access")
        if self.test_evaluation_count != 0:
            raise RidgeValidationError("Outer test prediction computed more than once")
        self.test_evaluation_count = 1
        self.test_prediction_computed_at = utc_now()

    def isolation_record(self) -> dict[str, Any]:
        selection_precedes = bool(
            self.selection_locked_at and self.test_feature_accessed_at and self.test_age_accessed_at
            and self.selection_locked_at <= self.test_feature_accessed_at
            and self.selection_locked_at <= self.test_age_accessed_at
        )
        return {
            "outer_test_feature_accessed_during_selection": False,
            "outer_test_age_accessed_during_selection": False,
            "outer_test_prediction_computed_during_selection": False,
            "selection_locked": self.selection_locked,
            "selection_locked_at": self.selection_locked_at,
            "test_feature_accessed_at": self.test_feature_accessed_at,
            "test_age_accessed_at": self.test_age_accessed_at,
            "test_prediction_computed_at": self.test_prediction_computed_at,
            "selection_precedes_test_access": selection_precedes,
            "test_feature_access_count": self.test_feature_access_count,
            "test_age_access_count": self.test_age_access_count,
            "test_evaluation_count": self.test_evaluation_count,
        }


def _matrix(accessor: GuardedOuterAccessor, subject_ids: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    features = np.stack([accessor.feature(subject) for subject in subject_ids]).astype(np.float64)
    ages = np.asarray([accessor.age(subject) for subject in subject_ids], dtype=np.float64)
    return features, ages


def _save_outer_model_artifact(
    output_dir: Path, outer_fold: int, outer_train: Sequence[str], outer_test: str,
    model: dict[str, Any], selected_alpha: float, selection_fingerprint: str,
    cache_key: str, cache_manifest_sha256: str, selection_payload: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    artifact_dir = output_dir / "model_artifacts" / f"outer_fold_{outer_fold}"
    artifact_dir.mkdir(parents=True, exist_ok=False)
    selection_artifact = {
        **selection_payload,
        "selection_fingerprint": selection_fingerprint,
        "selection_locked": True,
        "outer_test_feature_accessed": False,
        "outer_test_age_accessed": False,
        "created_before_outer_test_access": True,
    }
    write_json(artifact_dir / "selection.json", selection_artifact)
    model_path = artifact_dir / "model.npz"
    np.savez(
        model_path,
        coef=np.asarray(model["coefficient"], dtype=np.float64),
        feature_center=np.asarray(model["feature_center"], dtype=np.float64),
        target_mean=np.asarray([model["target_mean"]], dtype=np.float64),
        target_std=np.asarray([model["target_std"]], dtype=np.float64),
    )
    manifest = {
        "artifact_kind": "age_subject_ridge_outer_fold",
        "not_compatible_with_gradient_head_loader": True,
        "outer_fold": int(outer_fold),
        "outer_train_subject_ids": list(outer_train),
        "outer_test_subject_id": outer_test,
        "selected_alpha": float(selected_alpha),
        "coef_shape": [FEATURE_DIM],
        "feature_center_shape": [FEATURE_DIM],
        "target_mean": float(model["target_mean"]),
        "target_std": float(model["target_std"]),
        "feature_transform": "train_center",
        "target_transform": "train_subject_zscore",
        "solver": model["solver"],
        "solver_form": model["solver_form"],
        "dtype": "float64",
        "fit_intercept": False,
        "input_cache_key": cache_key,
        "input_cache_manifest_sha256": cache_manifest_sha256,
        "selection_fingerprint": selection_fingerprint,
        "model_npz_sha256": sha256_file(model_path),
    }
    manifest_path = artifact_dir / "model_manifest.json"
    write_json(manifest_path, manifest)
    if sha256_file(model_path) != manifest["model_npz_sha256"]:
        raise RidgeValidationError("Outer model artifact SHA mismatch")
    return model_path, manifest


def run_nested_loso_from_subjects(
    subject_rows: Sequence[SubjectFeatureRow], alphas: Sequence[float], output_dir: Path,
    *, cache_key: str, cache_manifest_sha256: str, logger=None,
) -> dict[str, Any]:
    eligible = validate_huada_subject_inventory(subject_rows)
    eligible_ids = [row.subject_id for row in eligible]
    splits = build_nested_loso_splits(eligible_ids)
    normalized_alphas = [float(value) for value in alphas]
    if not normalized_alphas or len(normalized_alphas) != len(set(normalized_alphas)):
        raise RidgeValidationError("Alphas must be non-empty and unique")
    if any(not math.isfinite(value) or value <= 0 for value in normalized_alphas):
        raise RidgeValidationError("All alphas must be finite and positive")

    inner_predictions: list[dict[str, Any]] = []
    inner_results: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    oof_rows: list[dict[str, Any]] = []
    isolation_rows: list[dict[str, Any]] = []

    for outer_fold, outer_test in enumerate(eligible_ids):
        development = [subject for subject in eligible_ids if subject != outer_test]
        accessor = GuardedOuterAccessor(subject_rows, outer_test)
        fold_splits = [row for row in splits if row["outer_fold"] == outer_fold]
        candidate_rows: list[dict[str, Any]] = []
        for alpha in normalized_alphas:
            alpha_predictions: list[dict[str, Any]] = []
            for split in fold_splits:
                inner_train = split["inner_train_subject_ids"]
                inner_validation = split["inner_validation_subject_id"]
                train_x, train_y = _matrix(accessor, inner_train)
                model = fit_ridge_for_subjects(train_x, train_y, alpha)
                validation_feature = accessor.feature(inner_validation)
                true_age = accessor.age(inner_validation)
                predicted_age = predict_age(model, validation_feature)
                error = predicted_age - true_age
                row = {
                    "outer_fold": outer_fold,
                    "outer_test_subject_id": outer_test,
                    "alpha": alpha,
                    "inner_fold": split["inner_fold"],
                    "inner_train_subject_ids": ";".join(inner_train),
                    "inner_validation_subject_id": inner_validation,
                    "true_age": true_age,
                    "predicted_age": predicted_age,
                    "signed_error": error,
                    "absolute_error": abs(error),
                    "squared_error": error ** 2,
                    "target_mean": model["target_mean"],
                    "target_std": model["target_std"],
                    "feature_center_sha256": model["feature_center_sha256"],
                    "feature_center_scope": "inner_train_subjects_only",
                    "target_statistics_scope": "inner_train_subjects_only",
                    "outer_test_used": False,
                }
                inner_predictions.append(row)
                alpha_predictions.append(row)
                log_event(
                    logger, "INFO", "ridge.inner", "inner LOSO validation prediction",
                    outer_fold=outer_fold, alpha=alpha, inner_fold=split["inner_fold"],
                    train_subjects=";".join(inner_train), validation_subject=inner_validation,
                    validation_prediction=round(predicted_age, 6), absolute_error=round(abs(error), 6),
                )
            if len(alpha_predictions) != 6:
                raise RidgeValidationError("An alpha is missing one or more inner folds")
            metrics = regression_metrics(
                [row["true_age"] for row in alpha_predictions],
                [row["predicted_age"] for row in alpha_predictions],
            )
            candidate = {
                "outer_fold": outer_fold,
                "outer_test_subject_id": outer_test,
                "alpha": alpha,
                "inner_prediction_count": len(alpha_predictions),
                **metrics,
            }
            inner_results.append(candidate)
            candidate_rows.append(candidate)
        selected = select_alpha(candidate_rows)
        selection_payload = {
            "outer_fold": outer_fold,
            "outer_test_subject_id": outer_test,
            "selection_metric": "inner_pooled_subject_mae",
            "metric_tolerance": METRIC_TOLERANCE,
            "selected_alpha": selected["alpha"],
            "selection_metrics": {
                "mae": selected["mae"],
                "rmse": selected["rmse"],
                "max_absolute_error": selected["max_absolute_error"],
            },
            "candidate_results": candidate_rows,
        }
        selection_fingerprint = hashlib.sha256(canonical_json(selection_payload).encode()).hexdigest()
        development_x, development_y = _matrix(accessor, development)
        outer_model = fit_ridge_for_subjects(development_x, development_y, selected["alpha"])
        model_path, model_manifest = _save_outer_model_artifact(
            output_dir, outer_fold, development, outer_test, outer_model,
            selected["alpha"], selection_fingerprint, cache_key, cache_manifest_sha256,
            selection_payload,
        )
        log_event(logger, "INFO", "ridge.stage5", "[stage 5] refit six-subject outer model",
                  outer_fold=outer_fold, selected_alpha=selected["alpha"])
        accessor.lock(selection_fingerprint)
        log_event(
            logger, "INFO", "ridge.outer", "selection locked and outer model refitted",
            outer_fold=outer_fold, selected_alpha=selected["alpha"],
            inner_pooled_mae=selected["mae"], selection_locked=True,
        )
        test_feature = accessor.feature(outer_test)
        test_age = accessor.age(outer_test)
        test_prediction = predict_age(outer_model, test_feature)
        accessor.mark_test_prediction()
        log_event(logger, "INFO", "ridge.stage6", "[stage 6] lock selection and evaluate outer test once",
                  outer_fold=outer_fold, test_subject=outer_test)
        baseline_prediction = float(np.mean(development_y))
        model_error = test_prediction - test_age
        baseline_error = baseline_prediction - test_age
        true_distance_from_train_mean = abs(test_age - baseline_prediction)
        prediction_distance_from_train_mean = abs(test_prediction - baseline_prediction)
        shrinkage_ratio = (
            prediction_distance_from_train_mean / true_distance_from_train_mean
            if true_distance_from_train_mean > 0 else None
        )
        prediction_between_train_mean_and_true = bool(
            min(baseline_prediction, test_age) <= test_prediction <= max(baseline_prediction, test_age)
        )
        isolation = {"outer_fold": outer_fold, "outer_test_subject_id": outer_test, **accessor.isolation_record()}
        if not isolation["selection_precedes_test_access"] or isolation["test_evaluation_count"] != 1:
            raise RidgeValidationError("Outer test isolation verification failed")
        isolation_rows.append(isolation)
        selected_rows.append({
            "outer_fold": outer_fold,
            "outer_test_subject_id": outer_test,
            "outer_development_subject_ids": ";".join(development),
            "selected_alpha": selected["alpha"],
            "inner_pooled_mae": selected["mae"],
            "inner_pooled_rmse": selected["rmse"],
            "inner_pooled_max_absolute_error": selected["max_absolute_error"],
            "metric_tolerance": METRIC_TOLERANCE,
            "selection_fingerprint": selection_fingerprint,
            "selection_locked": True,
            "outer_feature_center_sha256": outer_model["feature_center_sha256"],
            "outer_target_mean": outer_model["target_mean"],
            "outer_target_std": outer_model["target_std"],
            "model_artifact_path": str(model_path),
            "model_artifact_sha256": model_manifest["model_npz_sha256"],
            **accessor.isolation_record(),
        })
        oof_rows.append({
            "subject_id": outer_test,
            "true_age": test_age,
            "predicted_age": test_prediction,
            "baseline_predicted_age": baseline_prediction,
            "model_signed_error": model_error,
            "model_absolute_error": abs(model_error),
            "model_squared_error": model_error ** 2,
            "baseline_signed_error": baseline_error,
            "baseline_absolute_error": abs(baseline_error),
            "baseline_squared_error": baseline_error ** 2,
            "outer_train_mean_age": baseline_prediction,
            "true_distance_from_train_mean": true_distance_from_train_mean,
            "prediction_distance_from_train_mean": prediction_distance_from_train_mean,
            "shrinkage_ratio": shrinkage_ratio,
            "prediction_between_train_mean_and_true": prediction_between_train_mean_and_true,
            "outer_fold": outer_fold,
            "selected_alpha": selected["alpha"],
        })
        log_event(
            logger, "INFO", "ridge.outer", "outer test evaluated exactly once after lock",
            outer_fold=outer_fold, test_subject=outer_test,
            predicted_age=round(test_prediction, 6), true_age=test_age,
            absolute_error=round(abs(model_error), 6), test_evaluation_count=1,
        )

    if len(inner_predictions) != len(eligible_ids) * len(normalized_alphas) * 6:
        raise RidgeValidationError("Unexpected inner prediction count")
    oof_subjects = [row["subject_id"] for row in oof_rows]
    if len(oof_rows) != 7 or len(set(oof_subjects)) != 7 or set(oof_subjects) != set(eligible_ids):
        raise RidgeValidationError("OOF subjects are duplicated or incomplete")
    truth = [row["true_age"] for row in oof_rows]
    predictions = [row["predicted_age"] for row in oof_rows]
    baseline_predictions = [row["baseline_predicted_age"] for row in oof_rows]
    model_metrics = complete_subject_metrics(truth, predictions)
    baseline_metrics = complete_subject_metrics(truth, baseline_predictions)
    improved = [row["subject_id"] for row in oof_rows if row["model_absolute_error"] < row["baseline_absolute_error"]]
    worsened = [row["subject_id"] for row in oof_rows if row["model_absolute_error"] > row["baseline_absolute_error"]]
    equal = [row["subject_id"] for row in oof_rows if row["model_absolute_error"] == row["baseline_absolute_error"]]
    comparison = {
        "model_mae_minus_baseline_mae": model_metrics["mae"] - baseline_metrics["mae"],
        "model_rmse_minus_baseline_rmse": model_metrics["rmse"] - baseline_metrics["rmse"],
        "model_r2_minus_baseline_r2": (
            model_metrics["r2"] - baseline_metrics["r2"]
            if model_metrics["r2"] is not None and baseline_metrics["r2"] is not None else None
        ),
        "subjects_improved": improved,
        "subjects_worsened": worsened,
        "subjects_equal": equal,
        "number_of_subjects_improved": len(improved),
        "number_of_subjects_worsened": len(worsened),
        "number_of_subjects_equal": len(equal),
        "model_better_than_baseline_by_mae": model_metrics["mae"] < baseline_metrics["mae"],
        "subjects_with_prediction_between_train_mean_and_true": [
            row["subject_id"] for row in oof_rows if row["prediction_between_train_mean_and_true"]
        ],
        "number_with_prediction_between_train_mean_and_true": sum(
            bool(row["prediction_between_train_mean_and_true"]) for row in oof_rows
        ),
        "median_shrinkage_ratio": float(np.median([
            row["shrinkage_ratio"] for row in oof_rows if row["shrinkage_ratio"] is not None
        ])),
    }
    return {
        "splits": splits,
        "inner_predictions": inner_predictions,
        "inner_results": inner_results,
        "selected_rows": selected_rows,
        "oof_rows": oof_rows,
        "isolation_rows": isolation_rows,
        "model_metrics": model_metrics,
        "baseline_metrics": baseline_metrics,
        "comparison": comparison,
        "eligible_subject_ids": eligible_ids,
        "inner_prediction_count": len(inner_predictions),
        "outer_prediction_count": len(oof_rows),
    }


def subject_inventory_rows(subject_rows: Sequence[SubjectFeatureRow]) -> list[dict[str, Any]]:
    return [{
        "subject_id": row.subject_id,
        "age": row.age,
        "eligibility": row.eligibility,
        "exclusion_reason": row.exclusion_reason,
        "window_count": row.window_count,
        "feature_dim": FEATURE_DIM,
        "window_feature_dtype": row.window_feature_dtype,
        "aggregation_dtype": row.aggregation_dtype,
        "subject_feature_l2_norm": float(np.linalg.norm(row.feature)),
        "subject_feature_abs_max": float(np.max(np.abs(row.feature))),
        "all_finite": bool(np.isfinite(row.feature).all()),
        "sample_id_unique": len(row.sample_ids) == len(set(row.sample_ids)),
    } for row in subject_rows]


def _report(subject_rows: Sequence[SubjectFeatureRow], result: dict[str, Any]) -> str:
    model = result["model_metrics"]
    baseline = result["baseline_metrics"]
    comparison = result["comparison"]
    eligible = [row for row in subject_rows if row.eligibility]
    selected_lines = "\n".join(
        f"- fold {row['outer_fold']} / {row['outer_test_subject_id']}: alpha={row['selected_alpha']}, "
        f"inner MAE={row['inner_pooled_mae']:.4f}"
        for row in result["selected_rows"]
    )
    inner_lines = "\n".join(
        f"- fold {row['outer_fold']} / alpha={row['alpha']}: "
        f"MAE={row['mae']:.4f}, RMSE={row['rmse']:.4f}, max|error|={row['max_absolute_error']:.4f}"
        for row in result["inner_results"]
    )
    prediction_lines = "\n".join(
        f"- {row['subject_id']}: true={row['true_age']:.2f}, ridge={row['predicted_age']:.4f}, "
        f"baseline={row['baseline_predicted_age']:.4f}, |error|={row['model_absolute_error']:.4f}"
        for row in result["oof_rows"]
    )
    outcome = (
        "Ridge 的 OOF subject MAE 优于逐折 train-mean baseline。"
        if comparison["model_better_than_baseline_by_mae"] else
        "Ridge 未优于逐折 train-mean baseline；这是负结果，不能描述为有效年龄预测模型。"
    )
    return f"""# 年龄 subject-mean Ridge nested LOSO 报告

## 实验边界

本实验只有 **7 名独立年龄有效被试**。SUB5 的 age 为空，按 `missing_age` 排除，未插补。
每名被试的全部窗口先以 float64 做 512 维算术均值；窗口仅构造 subject feature，未作为独立年龄标签。
本实验没有 optimizer、learning rate、epoch、backbone forward、gradient head 或 window-level 性能指标。

## 被试与窗口

{chr(10).join(f'- {row.subject_id}: age={row.age:.2f}, windows={row.window_count}' for row in eligible)}

## Nested LOSO 与防泄漏

外层 7 折，每折 1 名 test、6 名 development；每个外层折内部再做 6 折 LOSO。
每个 alpha 在每个外层折有 6 个 validation subject 预测，总 inner 预测数为 {result['inner_prediction_count']}。
feature center 和 target mean/std 在每个 inner fold 只由 5 名 inner-train 被试计算；outer refit 只由 6 名 development 被试计算。
alpha 先按 inner pooled subject MAE（再按 RMSE、最大绝对误差、更强正则）锁定，之后 outer test feature/age 才各访问一次。

## 每折 alpha

{selected_lines}

完整 inner validation 聚合（每项均为 6 名 validation subject）：

{inner_lines}

## OOF 逐被试预测

{prediction_lines}

## 主结果与 baseline

- Ridge: MAE={model['mae']:.4f}, RMSE={model['rmse']:.4f}, R²={model['r2']:.4f}, bias={model['mean_signed_error']:.4f}
- Train-mean baseline: MAE={baseline['mae']:.4f}, RMSE={baseline['rmse']:.4f}, R²={baseline['r2']:.4f}
- MAE 差（model-baseline）={comparison['model_mae_minus_baseline_mae']:.4f}
- 改善被试：{', '.join(comparison['subjects_improved']) or '无'}
- 恶化被试：{', '.join(comparison['subjects_worsened']) or '无'}
- 预测位于 outer train mean 与真实年龄之间：{comparison['number_with_prediction_between_train_mean_and_true']}/7
- 中位 shrinkage ratio（|pred-train mean| / |true-train mean|）={comparison['median_shrinkage_ratio']:.4f}

{outcome}

Pearson r 和 Spearman rho 仅为辅助排序诊断。N=7 极小，指标方差很高，不能包装成稳定人群性能。
上述 shrinkage ratio 小于 1 表示预测相对真实年龄更靠近训练均值；它是收缩诊断，不是性能指标。
Ridge 消除了迭代优化/学习率/epoch 的歧义，但不会创造新的独立年龄标签，也不能保证冻结 feature 含有可泛化年龄信息。
"""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-lance", required=True)
    parser.add_argument("--feature-cache-lance", required=True)
    parser.add_argument("--feature-cache-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--alphas", nargs="+", type=float, default=list(DEFAULT_ALPHAS))
    parser.add_argument("--aggregation", choices=["mean"], default="mean")
    parser.add_argument("--feature-transform", choices=["train_center"], default="train_center")
    parser.add_argument("--target-transform", choices=["train_subject_zscore"], default="train_subject_zscore")
    parser.add_argument("--selection-metric", choices=["mae"], default="mae")
    parser.add_argument("--device", choices=["cpu"], default="cpu")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    if not args.alphas or len(args.alphas) != len(set(args.alphas)):
        parser.error("--alphas must be non-empty and unique")
    if any(not math.isfinite(value) or value <= 0 for value in args.alphas):
        parser.error("--alphas must contain finite positive values")
    return args


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise RidgeValidationError(f"Output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    logger = setup_run_logging(output_dir, args.log_level, append=False)
    write_csv(output_dir / "ridge_issues.csv", [], ISSUE_FIELDS)
    (output_dir / "ridge_errors.log").write_text(ERROR_HEADER, encoding="utf-8")
    started = time.monotonic()
    project_root = Path(__file__).resolve().parents[2]
    try:
        log_event(logger, "INFO", "ridge.stage1", "[stage 1] validate Lance and mean feature cache")
        source = ExistingMeanCacheSource(
            args.input_lance, args.feature_cache_lance, args.feature_cache_manifest
        )
        windows = source.load_and_join()
        log_event(logger, "INFO", "ridge.stage2", "[stage 2] aggregate deterministic subject features")
        subjects = aggregate_subject_features(windows)
        eligible = validate_huada_subject_inventory(subjects)
        inventory = subject_inventory_rows(subjects)
        write_csv(output_dir / "subject_feature_inventory.csv", inventory, list(inventory[0]))
        subject_npz = output_dir / "subject_features.npz"
        np.savez(
            subject_npz,
            subject_ids=np.asarray([row.subject_id for row in subjects]),
            features=np.stack([row.feature for row in subjects]).astype(np.float64),
            window_counts=np.asarray([row.window_count for row in subjects], dtype=np.int64),
        )
        subject_npz_sha = sha256_file(subject_npz)
        resolved = {
            "protocol_version": PROTOCOL_VERSION,
            "task": "age_regression",
            "feature_source": "existing_frozen_backbone_cache",
            "pooling": "masked_mean",
            "window_feature_dim": FEATURE_DIM,
            "subject_feature_aggregation": "arithmetic_mean_over_all_subject_windows",
            "aggregation_dtype": "float64",
            "estimator": "ridge_regression",
            "solver": "torch.linalg.solve",
            "solver_form": "dual",
            "compute_device": "cpu",
            "compute_dtype": "float64",
            "outer_protocol": "leave_one_subject_out",
            "inner_protocol": "leave_one_subject_out",
            "alphas": [float(value) for value in args.alphas],
            "feature_transform": args.feature_transform,
            "target_transform": args.target_transform,
            "selection_metric": "inner_pooled_subject_mae",
            "metric_tolerance": METRIC_TOLERANCE,
            "subject_order": [row.subject_id for row in eligible],
            "fit_intercept": False,
            "backbone_constructed": False,
            "optimizer_created": False,
        }
        write_json(output_dir / "resolved_config.json", resolved)
        log_event(logger, "INFO", "ridge.stage3", "[stage 3] build seven outer LOSO folds")
        cache_manifest_sha = sha256_file(args.feature_cache_manifest)
        log_event(logger, "INFO", "ridge.stage4", "[stage 4] run nested inner LOSO alpha selection")
        result = run_nested_loso_from_subjects(
            subjects, args.alphas, output_dir,
            cache_key=str(source.manifest["cache_key"]),
            cache_manifest_sha256=cache_manifest_sha,
            logger=logger,
        )
        split_rows = [{
            **row,
            "outer_development_subject_ids": ";".join(row["outer_development_subject_ids"]),
            "inner_train_subject_ids": ";".join(row["inner_train_subject_ids"]),
        } for row in result["splits"]]
        write_csv(output_dir / "nested_cv_splits.csv", split_rows, list(split_rows[0]))
        write_csv(output_dir / "inner_cv_predictions.csv", result["inner_predictions"], list(result["inner_predictions"][0]))
        write_csv(output_dir / "inner_cv_results.csv", result["inner_results"], list(result["inner_results"][0]))
        write_csv(output_dir / "outer_fold_selected_alpha.csv", result["selected_rows"], list(result["selected_rows"][0]))
        write_csv(output_dir / "oof_subject_predictions.csv", result["oof_rows"], list(result["oof_rows"][0]))
        write_json(output_dir / "oof_subject_metrics.json", result["model_metrics"])
        write_json(output_dir / "baseline_subject_metrics.json", result["baseline_metrics"])
        write_json(output_dir / "model_vs_baseline.json", result["comparison"])
        log_event(logger, "INFO", "ridge.stage7", "[stage 7] aggregate seven-subject OOF")
        log_event(logger, "INFO", "ridge.stage8", "[stage 8] compare against train-mean baseline")
        commit, dirty = git_state(project_root)
        provenance = {
            "protocol_version": PROTOCOL_VERSION,
            "evaluation_unit": "subject",
            "N_subjects": 7,
            "eligible_subjects": result["eligible_subject_ids"],
            "excluded_subjects": ["SUB5"],
            "exclusion_reason": {"SUB5": "missing_age"},
            "input_lance": str(Path(args.input_lance).resolve()),
            "feature_cache_lance": str(Path(args.feature_cache_lance).resolve()),
            "feature_cache_manifest": str(Path(args.feature_cache_manifest).resolve()),
            "feature_cache_key": source.manifest["cache_key"],
            "feature_cache_manifest_sha256": cache_manifest_sha,
            "subject_features_npz_sha256": subject_npz_sha,
            "global_subject_aggregation_before_split": True,
            "global_aggregation_label_free": True,
            "global_aggregation_window_selection_depends_on_age_or_fold": False,
            "inner_statistics_scope": "five_inner_train_subjects_only",
            "outer_refit_statistics_scope": "six_outer_development_subjects_only",
            "outer_test_used_for_selection": False,
            "outer_test_feature_access_count_per_fold": 1,
            "outer_test_age_access_count_per_fold": 1,
            "outer_test_evaluation_count_per_fold": 1,
            "selection_precedes_test_access_all_folds": all(
                row["selection_precedes_test_access"] for row in result["isolation_rows"]
            ),
            "inner_prediction_count": result["inner_prediction_count"],
            "outer_prediction_count": result["outer_prediction_count"],
            "backbone_constructed": False,
            "backbone_forward_count": 0,
            "optimizer_created": False,
            "scheduler_created": False,
            "epochs": None,
            "learning_rate": None,
            "git_commit": commit,
            "git_dirty": dirty,
            "code_sha256": sha256_file(__file__),
            "elapsed_seconds": time.monotonic() - started,
        }
        write_json(output_dir / "processing_provenance.json", provenance)
        (output_dir / "age_subject_ridge_report.md").write_text(
            _report(subjects, result), encoding="utf-8"
        )
        log_event(logger, "INFO", "ridge.stage9", "[stage 9] verify artifacts and provenance")
        required = [
            "resolved_config.json", "subject_feature_inventory.csv", "subject_features.npz",
            "nested_cv_splits.csv", "inner_cv_predictions.csv", "inner_cv_results.csv",
            "outer_fold_selected_alpha.csv", "oof_subject_predictions.csv",
            "oof_subject_metrics.json", "baseline_subject_metrics.json", "model_vs_baseline.json",
            "ridge_issues.csv", "ridge_errors.log", "processing_provenance.json",
            "age_subject_ridge_report.md", "run.log",
        ]
        missing = [name for name in required if not (output_dir / name).exists()]
        if missing or sha256_file(subject_npz) != subject_npz_sha:
            raise RidgeValidationError(f"Final artifact verification failed; missing={missing}")
        log_event(
            logger, "INFO", "ridge.complete", "subject Ridge nested LOSO completed",
            subject_mae=result["model_metrics"]["mae"],
            baseline_mae=result["baseline_metrics"]["mae"],
            subject_rmse=result["model_metrics"]["rmse"],
            elapsed_seconds=round(time.monotonic() - started, 3), output_dir=str(output_dir),
        )
        return 0
    except Exception as exc:
        with (output_dir / "ridge_errors.log").open("a", encoding="utf-8") as handle:
            handle.write(f"{utc_now()}\trun\t{type(exc).__name__}\t{str(exc).replace(chr(9), ' ')}\n")
            handle.write(traceback.format_exc() + "\n")
        write_csv(output_dir / "ridge_issues.csv", [{
            "severity": "error", "stage": "run", "outer_fold": None,
            "code": type(exc).__name__, "message": str(exc), "value": None, "threshold": None,
        }], ISSUE_FIELDS)
        log_exception(logger, "ridge.failed", "subject Ridge nested LOSO failed", error=str(exc))
        raise


def main(argv: Sequence[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
