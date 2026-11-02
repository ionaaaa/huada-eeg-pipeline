"""Deterministic subject-level CV7 and OOF utilities for age regression."""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


PROTOCOL = "age_subject_cv7"
ALGORITHM_VERSION = "numeric-subject-shuffle-next-validation-v1"
AGE_UNIT = "years"
TARGET_NORMALIZATION = "none"
PREDICTION_CLIPPING = "none"
SELECTION_ATOL = 1e-10


def _subject_number(subject_id):
    match = re.fullmatch(r"SUB(\d+)", str(subject_id), flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"Age CV7 requires SUB<number> subject IDs: {subject_id!r}")
    return int(match.group(1))


def regression_metrics(true_ages, predicted_ages):
    truth = np.asarray(true_ages, dtype=np.float64).reshape(-1)
    predicted = np.asarray(predicted_ages, dtype=np.float64).reshape(-1)
    if truth.shape != predicted.shape or truth.size == 0:
        raise ValueError("Regression metric inputs must be non-empty and have equal shape")
    if not np.isfinite(truth).all() or not np.isfinite(predicted).all():
        raise ValueError("Regression metric inputs must be finite")
    errors = predicted - truth
    denominator = float(np.sum((truth - truth.mean()) ** 2))
    r2_defined = bool(truth.size >= 2 and denominator > 0.0)
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "r2": float(1.0 - np.sum(errors ** 2) / denominator) if r2_defined else None,
        "r2_defined": r2_defined,
        "sample_count": int(truth.size),
    }


def build_age_subject_cv7(subject_ages, seed):
    normalized = {str(subject): float(age) for subject, age in subject_ages.items()}
    if len(normalized) != 7:
        raise ValueError(f"age_subject_cv7 requires exactly 7 eligible subjects; got {len(normalized)}")
    if not all(math.isfinite(age) for age in normalized.values()):
        raise ValueError("Age CV7 subject ages must be finite")
    ordered = sorted(normalized, key=_subject_number)
    rng = np.random.default_rng(int(seed))
    permutation = [ordered[index] for index in rng.permutation(len(ordered))]
    folds = []
    for fold in range(7):
        test_subject = permutation[fold]
        validation_subject = permutation[(fold + 1) % 7]
        train_subjects = [
            subject for subject in permutation
            if subject not in {test_subject, validation_subject}
        ]
        train_ages = [normalized[subject] for subject in train_subjects]
        test_age = normalized[test_subject]
        folds.append({
            "fold": fold,
            "train_subjects": train_subjects,
            "validation_subject": validation_subject,
            "validation_subjects": [validation_subject],
            "test_subject": test_subject,
            "test_subjects": [test_subject],
            "train_age_min": min(train_ages),
            "train_age_max": max(train_ages),
            "train_age_mean": float(np.mean(train_ages)),
            "validation_age": normalized[validation_subject],
            "test_age": test_age,
            "test_is_outside_train_age_range": bool(
                test_age < min(train_ages) or test_age > max(train_ages)
            ),
        })
    payload = {
        "protocol": PROTOCOL,
        "algorithm_version": ALGORITHM_VERSION,
        "fold_count": 7,
        "split_seed": int(seed),
        "numeric_subject_order": ordered,
        "permutation": permutation,
        "subject_ages": normalized,
        "folds": folds,
    }
    validate_age_cv7_payload(payload)
    return payload


def validate_age_cv7_payload(payload):
    if payload.get("protocol") != PROTOCOL or len(payload.get("folds", [])) != 7:
        raise ValueError("Expected age_subject_cv7 with exactly seven folds")
    subjects = set(payload["subject_ages"])
    test_counts = Counter()
    validation_counts = Counter()
    train_counts = Counter()
    for fold_info in payload["folds"]:
        train = set(fold_info["train_subjects"])
        validation = set(fold_info["validation_subjects"])
        test = set(fold_info["test_subjects"])
        if (len(train), len(validation), len(test)) != (5, 1, 1):
            raise ValueError(f"Invalid CV7 fold sizes: {fold_info}")
        if train & validation or train & test or validation & test:
            raise ValueError(f"Subject overlap in age fold {fold_info['fold']}")
        if train | validation | test != subjects:
            raise ValueError(f"Age fold {fold_info['fold']} does not cover all subjects")
        train_counts.update(train)
        validation_counts.update(validation)
        test_counts.update(test)
    if any(test_counts[subject] != 1 for subject in subjects):
        raise ValueError(f"Every eligible subject must be test once: {dict(test_counts)}")
    if any(validation_counts[subject] != 1 for subject in subjects):
        raise ValueError(
            f"Every eligible subject must be validation once: {dict(validation_counts)}"
        )
    if any(train_counts[subject] != 5 for subject in subjects):
        raise ValueError(f"Every eligible subject must be train five times: {dict(train_counts)}")
    return True


def fold_assignment_rows(payload, window_counts=None):
    window_counts = {} if window_counts is None else window_counts
    rows = []
    for fold_info in payload["folds"]:
        split_by_subject = {subject: "train" for subject in fold_info["train_subjects"]}
        split_by_subject[fold_info["validation_subject"]] = "validation"
        split_by_subject[fold_info["test_subject"]] = "test"
        rows.append({
            "fold": fold_info["fold"],
            "train_subjects": ";".join(fold_info["train_subjects"]),
            "validation_subject": fold_info["validation_subject"],
            "test_subject": fold_info["test_subject"],
            "train_age_min": fold_info["train_age_min"],
            "train_age_max": fold_info["train_age_max"],
            "train_age_mean": fold_info["train_age_mean"],
            "validation_age": fold_info["validation_age"],
            "test_age": fold_info["test_age"],
            "test_is_outside_train_age_range": fold_info[
                "test_is_outside_train_age_range"
            ],
            "train_window_count": sum(window_counts.get(s, 0) for s in fold_info["train_subjects"]),
            "validation_window_count": window_counts.get(fold_info["validation_subject"], 0),
            "test_window_count": window_counts.get(fold_info["test_subject"], 0),
            "split_seed": payload["split_seed"],
            "algorithm_version": payload["algorithm_version"],
            "subject_split_json": json.dumps(split_by_subject, sort_keys=True),
        })
    return rows


def write_fold_artifacts(output_dir, payload, window_counts=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "age_cv7_folds.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rows = fold_assignment_rows(payload, window_counts)
    with (output_dir / "age_fold_assignments.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _compare_min(left, right, atol):
    if left < right - atol:
        return -1
    if left > right + atol:
        return 1
    return 0


def validation_result_is_better(candidate, incumbent, atol=SELECTION_ATOL):
    """MAE, RMSE, loss, then earlier epoch; never R²."""
    if incumbent is None:
        return True
    for key in ("mae", "rmse", "loss"):
        comparison = _compare_min(float(candidate[key]), float(incumbent[key]), atol)
        if comparison:
            return comparison < 0
    return int(candidate["epoch"]) < int(incumbent["epoch"])


def candidate_selection_key(row):
    return (
        float(row["best_validation_mae"]),
        float(row["best_validation_rmse"]),
        float(row["best_validation_loss"]),
        str(row["candidate_id"]),
    )


def rank_candidates(rows):
    ranked = sorted(rows, key=candidate_selection_key)
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    return ranked


def validate_oof_coverage(rows, eligible_samples, subject_test_folds):
    if not rows:
        raise ValueError("OOF predictions are empty")
    sample_ids = [str(row["sample_id"]) for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("OOF sample_id values are not unique")
    if set(sample_ids) != set(eligible_samples):
        missing = sorted(set(eligible_samples).difference(sample_ids))
        extra = sorted(set(sample_ids).difference(eligible_samples))
        raise ValueError(f"OOF coverage mismatch; missing={missing}, extra={extra}")
    for row in rows:
        subject = str(row["subject_id"])
        if subject not in subject_test_folds:
            raise ValueError(f"Unknown OOF subject: {subject}")
        if int(row["fold"]) != int(subject_test_folds[subject]):
            raise ValueError(f"OOF row is assigned to the wrong test fold: {row}")
        if subject.upper() == "SUB5":
            raise ValueError("SUB5 must never appear in age OOF predictions")
    return True


def aggregate_subject_predictions(window_rows):
    grouped = defaultdict(list)
    for row in window_rows:
        grouped[str(row["subject_id"])].append(row)
    result = []
    for subject in sorted(grouped, key=_subject_number):
        rows = grouped[subject]
        true_ages = {float(row["true_age"]) for row in rows}
        folds = {int(row["fold"]) for row in rows}
        outside = {bool(row["test_is_outside_train_age_range"]) for row in rows}
        if len(true_ages) != 1 or len(folds) != 1 or len(outside) != 1:
            raise ValueError(f"Inconsistent OOF metadata within subject {subject}")
        predictions = np.asarray([float(row["predicted_age"]) for row in rows])
        true_age = true_ages.pop()
        mean_prediction = float(np.mean(predictions))
        median_prediction = float(np.median(predictions))
        result.append({
            "subject_id": subject,
            "true_age": true_age,
            "mean_predicted_age": mean_prediction,
            "median_predicted_age": median_prediction,
            "prediction_std": float(np.std(predictions)),
            "prediction_min": float(np.min(predictions)),
            "prediction_max": float(np.max(predictions)),
            "window_count": len(rows),
            "signed_error": mean_prediction - true_age,
            "absolute_error": abs(mean_prediction - true_age),
            "median_signed_error": median_prediction - true_age,
            "median_absolute_error": abs(median_prediction - true_age),
            "test_fold": folds.pop(),
            "test_is_outside_train_age_range": outside.pop(),
        })
    return result


def subject_aggregation_metrics(subject_rows):
    truth = [row["true_age"] for row in subject_rows]
    return {
        "mean": regression_metrics(truth, [row["mean_predicted_age"] for row in subject_rows]),
        "median": regression_metrics(
            truth, [row["median_predicted_age"] for row in subject_rows]
        ),
    }


def train_mean_baseline(fold_info, subject_ages):
    return float(np.mean([float(subject_ages[s]) for s in fold_info["train_subjects"]]))

