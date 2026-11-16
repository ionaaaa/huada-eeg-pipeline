"""Shared multi-source age-training adapters and reporting.

This module deliberately contains no optimizer or model loop.  It provides the
thin dataset/split/metric layer used by the existing Dortmund full-finetune
runner so single-source and joint experiments cannot drift into separate
training implementations.
"""

from __future__ import annotations

import bisect
import csv
import json
import math
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from torch.utils.data import Dataset

from dortmund_age_finetune import (
    TargetTransform,
    aggregate_subject_predictions,
    build_target_transform,
    metric_bundle,
    regression_metrics,
    sha256_file,
)


JOINT_PROFILE_IDS = ("joint_matched18", "joint_native64_vs_sparse18")
JOINT_SPLIT_COUNTS = {
    "dortmund": {"train": 172, "validation": 21, "test": 21},
    "mipdb": {"train": 15, "validation": 2, "test": 2},
}
MIPDB_BLOCK_FILTER = "block01"


class MultiSourceAgeDataset(Dataset):
    """A virtual concatenation of independent token caches.

    Child datasets remain physically separate.  Global sample and sampling
    subject identifiers are namespaced to make accidental cross-dataset
    collisions impossible.  Child order and length are independent of channel
    profile application, which keeps B/C sampling schedules identical.
    """

    def __init__(self, datasets: Mapping[str, Dataset]):
        if not datasets:
            raise ValueError("MultiSourceAgeDataset requires at least one source")
        self.dataset_ids = tuple(str(key).casefold() for key in datasets)
        if len(self.dataset_ids) != len(set(self.dataset_ids)):
            raise ValueError("Duplicate dataset_id after normalization")
        self.datasets = tuple(datasets[key] for key in datasets)
        self.cumulative_sizes: list[int] = []
        total = 0
        for dataset in self.datasets:
            if len(dataset) <= 0:
                raise ValueError("Every joint source dataset must be non-empty")
            total += len(dataset)
            self.cumulative_sizes.append(total)

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    def _locate(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        source_index = bisect.bisect_right(self.cumulative_sizes, index)
        previous = 0 if source_index == 0 else self.cumulative_sizes[source_index - 1]
        return source_index, index - previous

    @staticmethod
    def _namespace(dataset_id: str, sample: Mapping[str, Any]) -> dict[str, Any]:
        output = dict(sample)
        raw_sample_id = str(sample["sample_id"])
        output["source_sample_id"] = raw_sample_id
        output["sample_id"] = f"{dataset_id}::{raw_sample_id}"
        output["dataset_id"] = dataset_id
        return output

    def __getitem__(self, index: int) -> dict[str, Any]:
        source_index, local_index = self._locate(int(index))
        return self._namespace(
            self.dataset_ids[source_index], self.datasets[source_index][local_index]
        )

    def __getitems__(self, indices: Sequence[int]) -> list[dict[str, Any]]:
        # Preserve requested order.  Lance children already batch reads when a
        # DataLoader addresses them directly; correctness and stable B/C order
        # are more important than reordering a mixed-source batch here.
        return [self[int(index)] for index in indices]

    def get_sampling_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        offset = 0
        for dataset_id, dataset in zip(self.dataset_ids, self.datasets):
            for row in dataset.get_sampling_rows():
                output = dict(row)
                raw_subject = str(row["subject_id"])
                raw_sample = str(row["sample_id"])
                output.update({
                    "index": offset + int(row["index"]),
                    "dataset_id": dataset_id,
                    "source_subject_id": raw_subject,
                    "sampling_subject_id": f"{dataset_id}::{raw_subject}",
                    "source_sample_id": raw_sample,
                    "sample_id": f"{dataset_id}::{raw_sample}",
                })
                rows.append(output)
            offset += len(dataset)
        if len(rows) != len(self):
            raise RuntimeError("Joint sampling metadata length mismatch")
        sample_ids = [str(row["sample_id"]) for row in rows]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("Namespaced sample_id values are not globally unique")
        return rows


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    return rows


def validate_joint_manifest(
    rows: Sequence[Mapping[str, Any]], *, strict_counts: bool = True
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    keys = set()
    split_sets: dict[tuple[str, str], set[str]] = {}
    for raw in rows:
        row = dict(raw)
        dataset_id = str(row.get("dataset_id", "")).casefold()
        subject_id = str(row.get("subject_id", "")).strip()
        split = str(row.get("split", "")).strip()
        if dataset_id not in JOINT_SPLIT_COUNTS or not subject_id:
            raise ValueError(f"Invalid joint manifest identity: {row}")
        if split not in ("train", "validation", "test"):
            raise ValueError(f"Invalid joint split: {row}")
        try:
            age = float(row["age"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid joint age: {row}") from error
        if not math.isfinite(age):
            raise ValueError(f"Non-finite joint age: {row}")
        key = (dataset_id, subject_id)
        if key in keys:
            raise ValueError(f"Duplicate joint subject: {key}")
        keys.add(key)
        split_sets.setdefault((dataset_id, split), set()).add(subject_id)
        row.update({"dataset_id": dataset_id, "subject_id": subject_id, "split": split, "age": age})
        normalized.append(row)
    if strict_counts:
        actual = {
            dataset: {split: len(split_sets.get((dataset, split), set())) for split in expected}
            for dataset, expected in JOINT_SPLIT_COUNTS.items()
        }
        if actual != JOINT_SPLIT_COUNTS:
            raise ValueError(f"Joint split count mismatch: actual={actual}, expected={JOINT_SPLIT_COUNTS}")
    return normalized


def build_joint_target_transform(
    joint_rows: Sequence[Mapping[str, Any]], manifest_sha256: str
) -> TargetTransform:
    namespaced = []
    for raw in joint_rows:
        row = dict(raw)
        row["subject_id"] = f"{str(row['dataset_id']).casefold()}::{row['subject_id']}"
        namespaced.append(row)
    return build_target_transform(namespaced, manifest_sha256)


def validate_frozen_joint_contract(
    *, joint_manifest_path: str | Path, target_transform_path: str | Path,
    contract_path: str | Path, strict_counts: bool = True,
) -> tuple[list[dict[str, Any]], TargetTransform, dict[str, Any]]:
    contract = json.loads(Path(contract_path).read_text(encoding="utf-8"))
    expected = contract.get("sha256", {})
    actual = {
        "joint_subject_split_manifest": sha256_file(joint_manifest_path),
        "joint_target_transform": sha256_file(target_transform_path),
    }
    mismatches = {
        key: {"expected": expected.get(key), "actual": value}
        for key, value in actual.items() if expected.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Frozen joint contract SHA mismatch: {mismatches}")
    rows = validate_joint_manifest(read_csv_rows(joint_manifest_path), strict_counts=strict_counts)
    recomputed = build_joint_target_transform(rows, actual["joint_subject_split_manifest"])
    frozen = json.loads(Path(target_transform_path).read_text(encoding="utf-8"))
    if asdict(recomputed) != frozen:
        raise ValueError(
            "Frozen joint target transform does not exactly match the joint train subjects"
        )
    return rows, recomputed, contract


def joint_metric_bundle(window_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    datasets = sorted({str(row.get("dataset_id", "")).casefold() for row in window_rows})
    if not datasets or "" in datasets:
        raise ValueError("Joint metrics require dataset_id on every prediction row")
    per_dataset = {
        dataset_id: metric_bundle(
            [row for row in window_rows if str(row.get("dataset_id", "")).casefold() == dataset_id]
        )
        for dataset_id in datasets
    }
    macro: dict[str, Any] = {}
    for aggregation in ("subject_mean", "subject_median", "window"):
        for metric in ("mae", "rmse", "bias"):
            macro[f"{aggregation}_{metric}"] = float(np.mean([
                float(per_dataset[dataset][aggregation][metric]) for dataset in datasets
            ]))
    by_age_band: dict[str, Any] = {}
    for dataset_id in datasets:
        subject_rows = aggregate_subject_predictions([
            row for row in window_rows if str(row.get("dataset_id", "")).casefold() == dataset_id
        ])
        bands = {}
        for label, low, high in (("20-24", 20, 24), ("25-29", 25, 29), ("30-35", 30, 35)):
            selected = [row for row in subject_rows if low <= float(row["true_age"]) <= high]
            bands[label] = None if not selected else regression_metrics(
                [row["true_age"] for row in selected],
                [row["mean_predicted_age"] for row in selected],
            )
        by_age_band[dataset_id] = bands
    return {
        "datasets": per_dataset,
        "joint_macro": macro,
        "joint_macro_MAE": macro["subject_mean_mae"],
        "joint_macro_RMSE": macro["subject_mean_rmse"],
        "by_age_band": by_age_band,
        "dataset_ids": datasets,
        "aggregation_policy": "dataset_macro_unweighted",
    }


def split_prediction_rows(
    rows: Sequence[dict[str, Any]], dataset_id: str
) -> list[dict[str, Any]]:
    key = str(dataset_id).casefold()
    return [row for row in rows if str(row.get("dataset_id", "")).casefold() == key]


def dataset_draw_counts(rows: Sequence[Mapping[str, Any]], sequence: Sequence[int]) -> dict[str, int]:
    by_index = {int(row["index"]): str(row.get("dataset_id", "")) for row in rows}
    return dict(sorted(Counter(by_index[int(index)] for index in sequence).items()))
