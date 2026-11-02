#!/usr/bin/env python3
"""NeuralBench HBN age full-checkpoint zero-shot evaluation on Huada tokens.

This entry point performs one ordered online pass over age-eligible windows.
It has no parameter update, calibration, split, cache, or checkpoint search.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import lance
import numpy as np
import torch
from torch.utils.data import DataLoader

try:  # package imports in tests
    from .dataset import collate_fn
    from .lance_dataset import LanceEEGPretrainDataset
    from .largeScale_fineTune import EEGDownstreamClassifier
    from .neuralbench_checkpoint import (
        AGE_KEY_REMAP_VERSION,
        DEFAULT_NEURALBENCH_HBN_AGE_SHA256,
        load_neuralbench_hbn_age_model_state,
        sha256_file,
        verify_checkpoint_sha256,
        write_checkpoint_key_mapping,
    )
    from .neuralbench_zero_shot_common import (
        FEATURE_DIM,
        POOLING,
        PROBE_LAYER_IDX,
        PROBE_LAYER_RATIO,
        build_meta_zero_shot_model as _build_common_meta_model,
        cuda_allocated_mb,
        git_info,
        model_state_fingerprint,
        path_content_fingerprint,
        sample_id_fingerprint,
    )
    from .run_logging import log_event, log_exception, setup_run_logging
except ImportError:  # direct ``python scripts/...py`` execution
    from dataset import collate_fn
    from lance_dataset import LanceEEGPretrainDataset
    from largeScale_fineTune import EEGDownstreamClassifier
    from neuralbench_checkpoint import (
        AGE_KEY_REMAP_VERSION,
        DEFAULT_NEURALBENCH_HBN_AGE_SHA256,
        load_neuralbench_hbn_age_model_state,
        sha256_file,
        verify_checkpoint_sha256,
        write_checkpoint_key_mapping,
    )
    from neuralbench_zero_shot_common import (
        FEATURE_DIM,
        POOLING,
        PROBE_LAYER_IDX,
        PROBE_LAYER_RATIO,
        build_meta_zero_shot_model as _build_common_meta_model,
        cuda_allocated_mb,
        git_info,
        model_state_fingerprint,
        path_content_fingerprint,
        sample_id_fingerprint,
    )
    from run_logging import log_event, log_exception, setup_run_logging


RUN_MODE = "external_checkpoint_zero_shot"
TASK = "age_regression"
OUTPUT_DIM = 1
SOURCE_FORMAT = "neuralbench_lightning_state_dict"
SOURCE_CHECKPOINT_KIND = "neuralbench_hbn_age_full_finetuned"
SOURCE_DATASET = "HBN"
SOURCE_RELEASE = "R5"
SOURCE_AGE_MIN = 5.17
SOURCE_AGE_MAX = 20.88
EXPECTED_ELIGIBLE_SUBJECTS = {"SUB1", "SUB2", "SUB3", "SUB4", "SUB6", "SUB7", "SUB8"}
EXPECTED_EXCLUDED_SUBJECTS = {"SUB5"}
ZERO_SHOT_COLUMNS = list(LanceEEGPretrainDataset.TENSOR_COLUMNS) + ["window_index"]
WINDOW_FIELDS = [
    "sample_id", "subject_id", "recording_id", "window_index", "true_age",
    "predicted_age", "signed_error", "absolute_error", "squared_error",
    "source_age_min", "source_age_max", "true_age_outside_source_range",
]
SUBJECT_FIELDS = [
    "subject_id", "true_age", "mean_predicted_age", "median_predicted_age",
    "prediction_std", "prediction_min", "prediction_max", "mean_signed_error",
    "mean_absolute_error", "signed_error", "absolute_error", "median_signed_error",
    "median_absolute_error", "window_count", "true_age_outside_source_range",
]
ELIGIBILITY_FIELDS = [
    "subject_id", "age", "eligible_for_age", "exclusion_reason", "window_count",
    "true_age_outside_source_range",
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-lance", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--expected-checkpoint-sha256", default=DEFAULT_NEURALBENCH_HBN_AGE_SHA256
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--log-every-batches", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    if args.log_every_batches <= 0:
        parser.error("--log-every-batches must be positive")
    if str(args.log_level).upper() not in logging._nameToLevel:
        parser.error("--log-level is invalid")
    return args


def normalize_age(value: Any) -> float | None:
    if hasattr(value, "as_py"):
        value = value.as_py()
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    age = float(value)
    if not math.isfinite(age):
        raise ValueError(f"Age must be finite or null, got {value!r}")
    return age


def _subject_sort_key(subject_id: str) -> tuple[int, str]:
    text = str(subject_id).upper()
    if text.startswith("SUB") and text[3:].isdigit():
        return int(text[3:]), text
    return sys.maxsize, text


class NeuralBenchHbnAgeZeroShotDataset(LanceEEGPretrainDataset):
    """Age-eligible ordered Lance view with explicit subject eligibility."""

    def __init__(self, lance_path: str | Path):
        self.excluded_samples: list[dict[str, Any]] = []
        self.subject_eligibility: list[dict[str, Any]] = []
        self.input_row_count = 0
        super().__init__(
            lance_path=str(lance_path),
            columns=ZERO_SHOT_COLUMNS,
            task=TASK,
            subject_split_manifest=None,
            split=None,
            max_samples=None,
        )

    def _select_indices(self, max_samples=None):
        del max_samples
        self.input_row_count = len(self._metadata)
        selected: list[int] = []
        seen_samples: set[str] = set()
        subjects: dict[str, dict[str, Any]] = {}
        for index, row in enumerate(self._metadata):
            sample_id = str(row.get("sample_id", "")).strip()
            if not sample_id:
                raise ValueError(f"Input row {index} has an empty sample_id")
            if sample_id in seen_samples:
                raise ValueError(f"Duplicate sample_id in input Lance: {sample_id}")
            seen_samples.add(sample_id)
            subject_id = self._normalize_subject_id(row.get("subject_id")).strip()
            if not subject_id:
                raise ValueError(f"Input sample {sample_id} has an empty subject_id")
            age = normalize_age(row.get("age"))
            info = subjects.setdefault(
                subject_id,
                {"subject_id": subject_id, "age": age, "window_count": 0, "missing": age is None},
            )
            info["window_count"] += 1
            if (info["age"] is None) != (age is None):
                raise ValueError(f"Mixed missing and non-missing age within subject {subject_id}")
            if age is not None and not math.isclose(float(info["age"]), age, rel_tol=0.0, abs_tol=0.0):
                raise ValueError(f"Inconsistent ages within subject {subject_id}")
            if age is None:
                self.excluded_samples.append({
                    "sample_id": sample_id,
                    "subject_id": subject_id,
                    "reason": "missing_age",
                })
            else:
                selected.append(index)
        if not selected:
            raise ValueError("Input Lance has no age-eligible samples")
        self.subject_eligibility = []
        for subject_id in sorted(subjects, key=_subject_sort_key):
            info = subjects[subject_id]
            age = info["age"]
            self.subject_eligibility.append({
                "subject_id": subject_id,
                "age": "" if age is None else float(age),
                "eligible_for_age": age is not None,
                "exclusion_reason": "missing_age" if age is None else "",
                "window_count": int(info["window_count"]),
                "true_age_outside_source_range": (
                    "" if age is None else bool(age < SOURCE_AGE_MIN or age > SOURCE_AGE_MAX)
                ),
            })
        return selected

    def _row_to_sample(self, row):
        sample = super()._row_to_sample(row)
        sample["window_index"] = torch.tensor(int(row["window_index"]), dtype=torch.int64)
        return sample


def assert_authoritative_contract(model: torch.nn.Module | None = None) -> None:
    assert POOLING == "mean"
    assert PROBE_LAYER_IDX == 8
    assert math.isclose(PROBE_LAYER_RATIO, 0.75)
    assert FEATURE_DIM == 512
    assert OUTPUT_DIM == 1
    if model is not None:
        if model.pooling != POOLING or model.probe_layer_idx != PROBE_LAYER_IDX:
            raise RuntimeError("Constructed model violates pooling/layer contract")
        if model.head.in_features != FEATURE_DIM or model.head.out_features != OUTPUT_DIM:
            raise RuntimeError("Constructed age probe violates [512,1] contract")


def build_meta_zero_shot_model() -> EEGDownstreamClassifier:
    assert_authoritative_contract()
    model = _build_common_meta_model(OUTPUT_DIM)
    assert_authoritative_contract(model)
    return model


def prepare_model_for_inference(
    model: torch.nn.Module, device: torch.device
) -> torch.nn.Module:
    """Freeze the checkpoint-populated model and enter inference state."""

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("At least one model parameter remains trainable")
    model = model.to(device)
    model.eval()
    if model.training or model.backbone.training or model.head.training:
        raise RuntimeError("Model did not enter eval mode")
    return model


def _correlation(left: np.ndarray, right: np.ndarray) -> tuple[float | None, bool]:
    defined = bool(
        left.size >= 2
        and float(np.sum((left - left.mean()) ** 2)) > 0.0
        and float(np.sum((right - right.mean()) ** 2)) > 0.0
    )
    return (float(np.corrcoef(left, right)[0, 1]), True) if defined else (None, False)


def regression_metrics(true_ages: Sequence[float], predicted_ages: Sequence[float]) -> dict[str, Any]:
    truth = np.asarray(true_ages, dtype=np.float64).reshape(-1)
    predicted = np.asarray(predicted_ages, dtype=np.float64).reshape(-1)
    if truth.shape != predicted.shape or truth.size == 0:
        raise ValueError("Regression metrics require equal non-empty inputs")
    if not np.isfinite(truth).all() or not np.isfinite(predicted).all():
        raise ValueError("Regression metric inputs must be finite")
    errors = predicted - truth
    denominator = float(np.sum((truth - truth.mean()) ** 2))
    r2_defined = bool(truth.size >= 2 and denominator > 0.0)
    pearson, pearson_defined = _correlation(truth, predicted)
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "r2": float(1.0 - np.sum(errors ** 2) / denominator) if r2_defined else None,
        "r2_defined": r2_defined,
        "pearson_r": pearson,
        "pearson_r_defined": pearson_defined,
        "mean_signed_error": float(np.mean(errors)),
        "sample_count": int(truth.size),
    }


def _average_ranks(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=np.float64)
    start = 0
    while start < array.size:
        end = start + 1
        while end < array.size and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def spearman_rho(left: Sequence[float], right: Sequence[float]) -> tuple[float | None, bool]:
    return _correlation(_average_ranks(left), _average_ranks(right))


def aggregate_subject_predictions(window_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in window_rows:
        grouped[str(row["subject_id"])].append(row)
    result = []
    for subject_id in sorted(grouped, key=_subject_sort_key):
        rows = grouped[subject_id]
        true_ages = {float(row["true_age"]) for row in rows}
        if len(true_ages) != 1:
            raise ValueError(f"Inconsistent true ages within subject {subject_id}")
        truth = true_ages.pop()
        predictions = np.asarray([float(row["predicted_age"]) for row in rows])
        errors = predictions - truth
        mean_prediction = float(np.mean(predictions))
        median_prediction = float(np.median(predictions))
        result.append({
            "subject_id": subject_id,
            "true_age": truth,
            "mean_predicted_age": mean_prediction,
            "median_predicted_age": median_prediction,
            "prediction_std": float(np.std(predictions)),
            "prediction_min": float(np.min(predictions)),
            "prediction_max": float(np.max(predictions)),
            "mean_signed_error": float(np.mean(errors)),
            "mean_absolute_error": float(np.mean(np.abs(errors))),
            "signed_error": mean_prediction - truth,
            "absolute_error": abs(mean_prediction - truth),
            "median_signed_error": median_prediction - truth,
            "median_absolute_error": abs(median_prediction - truth),
            "window_count": len(rows),
            "true_age_outside_source_range": bool(
                truth < SOURCE_AGE_MIN or truth > SOURCE_AGE_MAX
            ),
        })
    return result


def extrapolation_diagnostics(subject_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    truth = np.asarray([float(row["true_age"]) for row in subject_rows])
    predicted = np.asarray([float(row["mean_predicted_age"]) for row in subject_rows])
    pearson, pearson_defined = _correlation(truth, predicted)
    spearman, spearman_defined = spearman_rho(truth, predicted)
    slope = intercept = None
    if truth.size >= 2 and float(np.sum((truth - truth.mean()) ** 2)) > 0.0:
        slope, intercept = (float(value) for value in np.polyfit(truth, predicted, 1))
    true_order = sorted(subject_rows, key=lambda row: (float(row["true_age"]), row["subject_id"]))
    predicted_order = sorted(
        subject_rows, key=lambda row: (float(row["mean_predicted_age"]), row["subject_id"])
    )
    errors = predicted - truth
    outside_subjects = [
        str(row["subject_id"])
        for row in subject_rows
        if float(row["true_age"]) < SOURCE_AGE_MIN
        or float(row["true_age"]) > SOURCE_AGE_MAX
    ]
    return {
        "diagnostic_only": True,
        "calibration_applied": False,
        "linear_fit_applied_to_predictions": False,
        "source_age_min": SOURCE_AGE_MIN,
        "source_age_max": SOURCE_AGE_MAX,
        "target_age_min": float(np.min(truth)),
        "target_age_max": float(np.max(truth)),
        "target_subjects_outside_source_range": outside_subjects,
        "target_subjects_outside_source_range_count": len(outside_subjects),
        "all_target_subjects_outside_source_range": bool(
            all(age < SOURCE_AGE_MIN or age > SOURCE_AGE_MAX for age in truth)
        ),
        "transfer_is_out_of_source_age_range": True,
        "pearson_r": pearson,
        "pearson_r_defined": pearson_defined,
        "spearman_rho": spearman,
        "spearman_rho_defined": spearman_defined,
        "mean_signed_error": float(np.mean(errors)),
        "prediction_min": float(np.min(predicted)),
        "prediction_max": float(np.max(predicted)),
        "prediction_mean": float(np.mean(predicted)),
        "prediction_std": float(np.std(predicted)),
        "prediction_within_source_range_count": int(
            np.sum((predicted >= SOURCE_AGE_MIN) & (predicted <= SOURCE_AGE_MAX))
        ),
        "prediction_within_one_year_of_source_upper_bound_count": int(
            np.sum(np.abs(predicted - SOURCE_AGE_MAX) <= 1.0)
        ),
        "prediction_concentration_reference": "absolute distance <= 1 year from source_age_max",
        "linear_fit_predicted_from_true": {
            "slope": slope,
            "intercept": intercept,
            "diagnostic_only": True,
            "calibration_applied": False,
        },
        "true_age_order_ascending": [
            {"rank": index, "subject_id": row["subject_id"], "true_age": row["true_age"]}
            for index, row in enumerate(true_order, start=1)
        ],
        "predicted_age_order_ascending": [
            {
                "rank": index,
                "subject_id": row["subject_id"],
                "mean_predicted_age": row["mean_predicted_age"],
            }
            for index, row in enumerate(predicted_order, start=1)
        ],
        "subject_errors": [
            {
                "subject_id": row["subject_id"],
                "true_age": row["true_age"],
                "mean_predicted_age": row["mean_predicted_age"],
                "signed_error": row["signed_error"],
                "absolute_error": row["absolute_error"],
            }
            for row in subject_rows
        ],
    }


def run_zero_shot_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    logger: logging.Logger,
    log_every_batches: int,
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    total_rows = len(loader.dataset)
    total_batches = len(loader)
    started = time.monotonic()
    forward_count = 0
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, start=1):
            model_inputs = {
                key: batch[key].to(device, non_blocking=device.type == "cuda")
                for key in (
                    "token_inputs", "token_channel_indices", "token_time_indices",
                    "token_valid_mask",
                )
            }
            outputs = model(**model_inputs)
            forward_count += 1
            if outputs.ndim != 2 or outputs.shape[1] != 1 or not torch.isfinite(outputs).all():
                raise RuntimeError("Raw age outputs must be finite with shape [B,1]")
            predictions = outputs[:, 0].detach().cpu()
            labels = batch["label"].view(-1).detach().cpu()
            for item_index, sample_id in enumerate(batch["sample_id"]):
                sample_id = str(sample_id)
                if sample_id in seen:
                    raise RuntimeError(f"Sample was forwarded more than once: {sample_id}")
                seen.add(sample_id)
                truth = float(labels[item_index])
                prediction = float(predictions[item_index])
                error = prediction - truth
                rows.append({
                    "sample_id": sample_id,
                    "subject_id": str(batch["subject_id"][item_index]),
                    "recording_id": str(batch["recording_id"][item_index]),
                    "window_index": int(batch["window_index"][item_index]),
                    "true_age": truth,
                    "predicted_age": prediction,
                    "signed_error": error,
                    "absolute_error": abs(error),
                    "squared_error": error ** 2,
                    "source_age_min": SOURCE_AGE_MIN,
                    "source_age_max": SOURCE_AGE_MAX,
                    "true_age_outside_source_range": bool(
                        truth < SOURCE_AGE_MIN or truth > SOURCE_AGE_MAX
                    ),
                })
            processed = len(rows)
            if batch_index == 1 or batch_index == total_batches or batch_index % log_every_batches == 0:
                elapsed = max(time.monotonic() - started, 1e-9)
                throughput = processed / elapsed
                eta = (total_rows - processed) / throughput if throughput > 0 else None
                fields = {
                    "batch": f"{batch_index}/{total_batches}",
                    "current_batch": batch_index,
                    "total_batches": total_batches,
                    "processed_rows": processed,
                    "total_rows": total_rows,
                    "progress_percent": round(100.0 * processed / total_rows, 2),
                    "elapsed_seconds": round(elapsed, 3),
                    "throughput_rows_per_second": round(throughput, 3),
                    "eta_seconds": None if eta is None else round(eta, 3),
                    "forward_count": forward_count,
                    "cuda_allocated_mb": round(cuda_allocated_mb(device), 3),
                }
                log_event(logger, "INFO", "inference.progress", "zero-shot inference progress", **fields)
    if len(rows) != total_rows or len(seen) != total_rows:
        raise RuntimeError("Inference coverage does not equal eligible dataset size")
    if forward_count != total_batches:
        raise RuntimeError("Forward count does not equal DataLoader batch count")
    return rows, forward_count


def validate_window_rows(
    rows: Sequence[dict[str, Any]], expected_sample_ids: Sequence[str]
) -> None:
    if len(rows) != len(expected_sample_ids):
        raise ValueError("Window prediction count differs from eligible sample count")
    row_ids = [str(row["sample_id"]) for row in rows]
    if len(row_ids) != len(set(row_ids)):
        raise ValueError("window_predictions contains duplicate sample_id values")
    if row_ids != list(expected_sample_ids):
        raise ValueError("Window prediction order/coverage differs from eligible Lance order")
    for row in rows:
        values = [
            float(row[key])
            for key in (
                "true_age", "predicted_age", "signed_error", "absolute_error", "squared_error"
            )
        ]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Non-finite age output for {row['sample_id']}")
        error = values[1] - values[0]
        if not math.isclose(values[2], error, rel_tol=1e-7, abs_tol=1e-7):
            raise ValueError(f"Signed error mismatch for {row['sample_id']}")
        if not math.isclose(values[3], abs(error), rel_tol=1e-7, abs_tol=1e-7):
            raise ValueError(f"Absolute error mismatch for {row['sample_id']}")
        if str(row["subject_id"]).upper() == "SUB5":
            raise ValueError("SUB5 must not appear in age zero-shot predictions")


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_zero_shot_artifacts(
    output_dir: Path,
    window_rows: list[dict[str, Any]],
    expected_sample_ids: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    validate_window_rows(window_rows, expected_sample_ids)
    window_path = output_dir / "window_predictions.csv"
    _write_csv(window_path, window_rows, WINDOW_FIELDS)
    with window_path.open("r", encoding="utf-8", newline="") as handle:
        round_trip = list(csv.DictReader(handle))
    validate_window_rows(round_trip, expected_sample_ids)

    window_metrics = regression_metrics(
        [row["true_age"] for row in window_rows],
        [row["predicted_age"] for row in window_rows],
    )
    window_metrics.update({
        "window_count": len(window_rows),
        "subject_count": len({row["subject_id"] for row in window_rows}),
        "windows_are_independent_subjects": False,
        "metric_role": "auxiliary",
    })
    subject_rows = aggregate_subject_predictions(window_rows)
    _write_csv(output_dir / "subject_predictions.csv", subject_rows, SUBJECT_FIELDS)
    truth = [row["true_age"] for row in subject_rows]
    subject_mean_metrics = regression_metrics(
        truth, [row["mean_predicted_age"] for row in subject_rows]
    )
    subject_mean_metrics["subject_count"] = len(subject_rows)
    subject_mean_metrics["metric_role"] = "primary"
    subject_median_metrics = regression_metrics(
        truth, [row["median_predicted_age"] for row in subject_rows]
    )
    subject_median_metrics["subject_count"] = len(subject_rows)
    subject_median_metrics["metric_role"] = "robustness"
    diagnostics = extrapolation_diagnostics(subject_rows)
    _write_json(output_dir / "window_metrics.json", window_metrics)
    _write_json(output_dir / "subject_mean_metrics.json", subject_mean_metrics)
    _write_json(output_dir / "subject_median_metrics.json", subject_median_metrics)
    _write_json(output_dir / "extrapolation_diagnostics.json", diagnostics)
    return window_metrics, subject_mean_metrics, subject_median_metrics, subject_rows, diagnostics


def _inspect_lance(path: str | Path) -> dict[str, Any]:
    if not Path(path).exists():
        raise FileNotFoundError(f"Input Lance does not exist: {path}")
    dataset = lance.dataset(str(path))
    missing = sorted(set(ZERO_SHOT_COLUMNS).difference(dataset.schema.names))
    if missing:
        raise ValueError(f"Input Lance is missing required columns: {missing}")
    return {"version": int(dataset.version), "row_count": int(dataset.count_rows())}


def _validate_authoritative_eligibility(dataset: NeuralBenchHbnAgeZeroShotDataset) -> None:
    eligible = {
        row["subject_id"] for row in dataset.subject_eligibility if row["eligible_for_age"]
    }
    excluded = {
        row["subject_id"] for row in dataset.subject_eligibility if not row["eligible_for_age"]
    }
    if eligible != EXPECTED_ELIGIBLE_SUBJECTS:
        raise RuntimeError(
            f"Age-eligible subjects differ from authority: {sorted(eligible)}"
        )
    if excluded != EXPECTED_EXCLUDED_SUBJECTS:
        raise RuntimeError(f"Excluded age subjects differ from authority: {sorted(excluded)}")
    sub5 = next(row for row in dataset.subject_eligibility if row["subject_id"] == "SUB5")
    if sub5["age"] != "" or sub5["exclusion_reason"] != "missing_age":
        raise RuntimeError("SUB5 must have age=null and exclusion_reason=missing_age")


def _code_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    paths = [
        Path(__file__).resolve(),
        root / "scripts/downstream_50m/neuralbench_checkpoint.py",
        root / "scripts/downstream_50m/neuralbench_zero_shot_common.py",
        root / "scripts/downstream_50m/largeScale_fineTune.py",
        root / "scripts/downstream_50m/model.py",
        root / "scripts/downstream_50m/lance_dataset.py",
        root / "scripts/downstream_50m/dataset.py",
    ]
    return {str(path.relative_to(root).as_posix()): sha256_file(path) for path in paths}


def dry_run(args: argparse.Namespace) -> int:
    """Validate path/schema/SHA only; do not scan rows or create output."""

    assert_authoritative_contract()
    checkpoint = verify_checkpoint_sha256(args.checkpoint, args.expected_checkpoint_sha256)
    lance_info = _inspect_lance(args.input_lance)
    plan = {
        "dry_run": True,
        "formal_output_created": False,
        "full_lance_rows_read": False,
        "checkpoint_deserialized": False,
        "run_mode": RUN_MODE,
        "task": TASK,
        "training_enabled": False,
        "calibration_enabled": False,
        "checkpoint": checkpoint,
        "input_lance": {"path": str(Path(args.input_lance).resolve()), **lance_info},
        "device": args.device,
        "batch_size": args.batch_size,
        "pooling": POOLING,
        "probe_layer_idx": PROBE_LAYER_IDX,
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def run(args: argparse.Namespace) -> int:
    if args.dry_run:
        return dry_run(args)
    # Fail on missing/mismatched checkpoint bytes before creating output or
    # reading target rows. The safe loader verifies these bytes again directly
    # before its restricted torch.load call.
    checkpoint_preflight = verify_checkpoint_sha256(
        args.checkpoint, args.expected_checkpoint_sha256
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    error_path = output_dir / "zero_shot_errors.log"
    error_path.write_text("", encoding="utf-8")
    logger = setup_run_logging(output_dir, args.log_level, append=False)
    started = time.monotonic()
    try:
        assert_authoritative_contract()
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        log_event(
            logger,
            "INFO",
            "run.start",
            "starting HBN age full-checkpoint zero-shot evaluation",
            run_mode=RUN_MODE,
            task=TASK,
            training_enabled=False,
            calibration_enabled=False,
            optimizer_created=False,
            scheduler_created=False,
            backward_called=False,
            checkpoint_selection=False,
            head_source="neuralbench_hbn_age_probe",
            backbone_source="neuralbench_hbn_age_full_finetuned",
        )

        model = build_meta_zero_shot_model()
        checkpoint, checkpoint_info, mapping_rows, mapping_summary = (
            load_neuralbench_hbn_age_model_state(
                model, args.checkpoint, args.expected_checkpoint_sha256
            )
        )
        del checkpoint
        if checkpoint_info != checkpoint_preflight:
            raise RuntimeError("Checkpoint changed between preflight and restricted loading")
        write_checkpoint_key_mapping(output_dir / "checkpoint_key_mapping.csv", mapping_rows)
        if mapping_summary["source_backbone_tensor_count"] != 154:
            raise RuntimeError("HBN age checkpoint must contain exactly 154 backbone tensors")
        if mapping_summary["source_probe_tensor_count"] != 2:
            raise RuntimeError("HBN age checkpoint must contain exactly two probe tensors")
        if mapping_summary["age_probe_parameter_count"] != 513:
            raise RuntimeError("HBN age probe must contain exactly 513 parameters")
        model = prepare_model_for_inference(model, device)
        before_forward_fingerprint = model_state_fingerprint(model)
        log_event(
            logger,
            "INFO",
            "checkpoint.loaded",
            "strict HBN age checkpoint mapping completed",
            checkpoint_sha256=checkpoint_info["sha256"],
            source_backbone_tensor_count=mapping_summary["source_backbone_tensor_count"],
            source_probe_tensor_count=mapping_summary["source_probe_tensor_count"],
            age_probe_parameter_count=mapping_summary["age_probe_parameter_count"],
            core_backbone_key_match_fraction=mapping_summary[
                "core_backbone_key_match_fraction"
            ],
            full_backbone_key_match_fraction=mapping_summary[
                "full_backbone_key_match_fraction"
            ],
            probe_key_match_fraction=mapping_summary["probe_key_match_fraction"],
            all_parameters_requires_grad_false=True,
        )

        input_before = path_content_fingerprint(args.input_lance)
        lance_info = _inspect_lance(args.input_lance)
        dataset = NeuralBenchHbnAgeZeroShotDataset(args.input_lance)
        _validate_authoritative_eligibility(dataset)
        eligible_metadata = dataset.get_subject_label_rows()
        expected_sample_ids = [str(row["sample_id"]) for row in eligible_metadata]
        all_input_sample_ids = [str(row["sample_id"]) for row in dataset._metadata]
        eligible_subject_ages = {
            str(row["subject_id"]): float(row["label"]) for row in eligible_metadata
        }
        target_age_min = min(eligible_subject_ages.values())
        target_age_max = max(eligible_subject_ages.values())
        outside_subjects = sorted(
            (
                subject_id for subject_id, age in eligible_subject_ages.items()
                if age < SOURCE_AGE_MIN or age > SOURCE_AGE_MAX
            ),
            key=_subject_sort_key,
        )
        all_outside = len(outside_subjects) == len(eligible_subject_ages)
        if not all_outside:
            raise RuntimeError("Expected all Huada age subjects to be outside the HBN source range")
        _write_csv(
            output_dir / "age_eligibility.csv", dataset.subject_eligibility, ELIGIBILITY_FIELDS
        )
        log_event(
            logger,
            "INFO",
            "age.eligibility",
            "validated authoritative age eligibility",
            eligible_subject_count=len(eligible_subject_ages),
            eligible_window_count=len(dataset),
            excluded_subjects=["SUB5"],
            exclusion_reason="missing_age",
            target_age_min=target_age_min,
            target_age_max=target_age_max,
            target_subjects_outside_source_range=outside_subjects,
            all_target_subjects_outside_source_range=all_outside,
        )

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            drop_last=False,
            pin_memory=device.type == "cuda",
        )
        window_rows, forward_count = run_zero_shot_inference(
            model, loader, device, logger, args.log_every_batches
        )
        after_forward_fingerprint = model_state_fingerprint(model)
        if before_forward_fingerprint != after_forward_fingerprint:
            raise RuntimeError("Model parameter/state fingerprint changed during inference")
        (
            window_metrics,
            subject_mean_metrics,
            subject_median_metrics,
            subject_rows,
            diagnostics,
        ) = write_zero_shot_artifacts(output_dir, window_rows, expected_sample_ids)

        input_after = path_content_fingerprint(args.input_lance)
        if input_before != input_after:
            raise RuntimeError("Input Lance content fingerprint changed during evaluation")
        checkpoint_after = sha256_file(args.checkpoint)
        if checkpoint_after != checkpoint_info["sha256"]:
            raise RuntimeError("Checkpoint content fingerprint changed during evaluation")

        issues: list[dict[str, Any]] = []
        if lance_info["row_count"] != 480:
            issues.append({
                "severity": "warning",
                "code": "unexpected_input_row_count",
                "detail": f"expected 480, observed {lance_info['row_count']}",
            })
        if len(dataset) != 420:
            issues.append({
                "severity": "warning",
                "code": "unexpected_eligible_window_count",
                "detail": f"expected 420, observed {len(dataset)}",
            })
        _write_csv(output_dir / "zero_shot_issues.csv", issues, ["severity", "code", "detail"])

        exclusion_counts = dict(Counter(row["reason"] for row in dataset.excluded_samples))
        elapsed_seconds = time.monotonic() - started
        summary = {
            "evaluation_kind": RUN_MODE,
            "run_mode": RUN_MODE,
            "task": TASK,
            "status": "completed",
            "target_training_samples": 0,
            "target_validation_samples": 0,
            "target_label_used_for_training": False,
            "target_label_used_for_calibration": False,
            "target_label_used_for_metrics_only": True,
            "training_enabled": False,
            "calibration_enabled": False,
            "calibration_applied": False,
            "prediction_clipping": "none",
            "checkpoint_selection": False,
            "head_source": "neuralbench_hbn_age_probe",
            "backbone_source": "neuralbench_hbn_age_full_finetuned",
            "eligible_subject_count": len(subject_rows),
            "eligible_window_count": len(dataset),
            "excluded_subject_count": 1,
            "excluded_subjects": ["SUB5"],
            "batch_count": len(loader),
            "forward_count": forward_count,
            "source_age_range": [SOURCE_AGE_MIN, SOURCE_AGE_MAX],
            "target_age_range": [target_age_min, target_age_max],
            "target_subjects_outside_source_range": outside_subjects,
            "target_subjects_outside_source_range_count": len(outside_subjects),
            "all_target_subjects_outside_source_range": all_outside,
            "transfer_is_out_of_source_age_range": True,
            "window_metrics": window_metrics,
            "subject_mean_metrics": subject_mean_metrics,
            "subject_median_metrics": subject_median_metrics,
            "elapsed_seconds": elapsed_seconds,
        }
        provenance = {
            **summary,
            "checkpoint_format": SOURCE_FORMAT,
            "source_checkpoint_kind": SOURCE_CHECKPOINT_KIND,
            "source_checkpoint_sha256": checkpoint_info["sha256"],
            "source_dataset": SOURCE_DATASET,
            "source_release": SOURCE_RELEASE,
            "source_age_range_years": [SOURCE_AGE_MIN, SOURCE_AGE_MAX],
            "source_age_min": SOURCE_AGE_MIN,
            "source_age_max": SOURCE_AGE_MAX,
            "source_test_subject_count": 329,
            "source_target_scaler": "none",
            "source_loss": "MSE",
            "source_loss_config_name": "MSELoss",
            "source_selection_metric": "val/pearsonr",
            "source_selection_mode": "max",
            "source_optimizer": "AdamW",
            "source_learning_rate": 1e-4,
            "source_weight_decay": 0.05,
            "source_freeze_backbone": False,
            "source_test_subject_mean_audit_metrics": {
                "mae": 1.9526,
                "rmse": 2.5894,
                "pearson_r": 0.6627,
                "r2": 0.4174,
                "provenance_only_not_target_result": True,
            },
            "target_unit": "years",
            "target_scaler": "none",
            "optimizer_created": False,
            "scheduler_created": False,
            "backward_called": False,
            "feature_cache_created": False,
            "input_lance": {
                "path": str(Path(args.input_lance).resolve()),
                "version": lance_info["version"],
                "row_count": lance_info["row_count"],
                "sample_id_fingerprint_sha256": sample_id_fingerprint(all_input_sample_ids),
                "eligible_sample_id_fingerprint_sha256": sample_id_fingerprint(
                    expected_sample_ids
                ),
                "content_fingerprint_before_sha256": input_before,
                "content_fingerprint_after_sha256": input_after,
                "unmodified": input_before == input_after,
            },
            "checkpoint": {
                **checkpoint_info,
                "source_format": SOURCE_FORMAT,
                "source_checkpoint_kind": SOURCE_CHECKPOINT_KIND,
                "expected_sha256": args.expected_checkpoint_sha256.lower(),
                "content_fingerprint_after_sha256": checkpoint_after,
                "unmodified": checkpoint_after == checkpoint_info["sha256"],
            },
            "key_remap_version": AGE_KEY_REMAP_VERSION,
            "checkpoint_mapping": mapping_summary,
            "representation": {
                "probe_layer_idx": PROBE_LAYER_IDX,
                "probe_layer_ratio": PROBE_LAYER_RATIO,
                "pooling": POOLING,
                "masked_mean_pooling": True,
                "feature_dim": FEATURE_DIM,
                "output_dim": OUTPUT_DIM,
            },
            "parity": {
                "dynamic_source_model_parity_performed": False,
                "parity_source": "strict audited structure/key/shape/semantic compatibility",
                "reason": "original NeuralBench source runtime was not instantiated in this target run",
                "requested_atol": 1e-5,
                "requested_rtol": 1e-5,
                "structural_weight_parity": True,
                "fabricated_dynamic_parity": False,
            },
            "device": str(device),
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "excluded_samples_by_reason": exclusion_counts,
            "git": git_info(Path(__file__).resolve().parents[2]),
            "code_sha256": _code_hashes(),
            "model_state_fingerprint_before_forward_sha256": before_forward_fingerprint,
            "model_state_fingerprint_after_forward_sha256": after_forward_fingerprint,
            "model_state_unchanged": before_forward_fingerprint == after_forward_fingerprint,
            "all_parameters_requires_grad_false": True,
            "trainable_parameter_count": 0,
            "model_parameter_count": sum(
                int(parameter.numel()) for parameter in model.parameters()
            ),
            "random_materialized_parameter_count": 0,
            "extrapolation_diagnostics": diagnostics,
        }
        _write_json(output_dir / "zero_shot_summary.json", summary)
        _write_json(output_dir / "zero_shot_provenance.json", provenance)
        report = f"""# NeuralBench HBN age full-checkpoint zero-shot report

- evaluation kind: `external_checkpoint_zero_shot`
- checkpoint SHA256: `{checkpoint_info['sha256']}` (strictly matched expected)
- eligible/excluded subjects: `{len(subject_rows)}` / `1` (`SUB5`, `missing_age`)
- eligible windows: `{len(dataset)}`
- batches/forward calls: `{len(loader)}` / `{forward_count}`
- pooling/layer: `masked mean` / zero-based `{PROBE_LAYER_IDX}`
- source age range: `{SOURCE_AGE_MIN}`–`{SOURCE_AGE_MAX}` years
- target age range: `{target_age_min}`–`{target_age_max}` years
- all target subjects outside source range: `{all_outside}`
- exact full-backbone key/parameter match: `{mapping_summary['full_backbone_key_match_fraction']}` / `{mapping_summary['full_backbone_parameter_match_fraction']}`
- exact age-probe key/parameter match: `{mapping_summary['probe_key_match_fraction']}` / `{mapping_summary['probe_parameter_match_fraction']}`
- model fingerprint unchanged by inference: `{before_forward_fingerprint == after_forward_fingerprint}`
- target training, calibration, clipping, checkpoint selection: `false`

Subject mean is the primary result (`n={len(subject_rows)}`): MAE
`{subject_mean_metrics['mae']:.6f}`, RMSE `{subject_mean_metrics['rmse']:.6f}`,
R² `{subject_mean_metrics['r2']}`, Pearson r `{subject_mean_metrics['pearson_r']}`,
and mean signed error `{subject_mean_metrics['mean_signed_error']:.6f}`. Subject
median metrics are a robustness result. Pearson and the diagnostic linear fit are
descriptive only and were not used for checkpoint selection or calibration.

The `{len(window_rows)}` windows are repeated observations from seven subjects,
not independent age subjects. Window-level metrics are auxiliary; subject-level
metrics are the formal result. Raw linear predictions were saved without target
scaling, calibration, sigmoid/softmax, rounding, or clipping.
"""
        (output_dir / "zero_shot_report.md").write_text(report, encoding="utf-8")
        log_event(
            logger,
            "INFO",
            "run.complete",
            "HBN age zero-shot evaluation completed",
            eligible_window_count=len(dataset),
            eligible_subject_count=len(subject_rows),
            forward_count=forward_count,
            elapsed_seconds=round(elapsed_seconds, 3),
            output_dir=str(output_dir.resolve()),
        )
        return 0
    except Exception:
        error_path.write_text(traceback.format_exc(), encoding="utf-8")
        log_exception(logger, "run.failed", "HBN age zero-shot evaluation failed")
        raise


def main(argv: Sequence[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
