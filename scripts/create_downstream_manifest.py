#!/usr/bin/env python
"""Create task and subject-level split manifests from a shared token Lance."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import lance
import numpy as np


SEX_LABEL_MAPPING = {"female": 0, "male": 1}
TASKS = {"sex_classification", "age_regression"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-lance", required=True)
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-splits", type=Path, required=True)
    parser.add_argument("--storage-options")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def _write(path, rows, fields):
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _storage_options(path):
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return {key: str(value) for key, value in json.load(handle).items()}


def _subject_rows(dataset, task):
    subject_values = {}
    for row in dataset.to_table(columns=["subject_id", "age", "sex"]).to_pylist():
        subject_id = str(row["subject_id"])
        sex = str(row["sex"]).strip().lower()
        if sex not in SEX_LABEL_MAPPING:
            raise ValueError(f"Unknown sex value {sex!r} for {subject_id}")
        if task == "sex_classification":
            value = (str(SEX_LABEL_MAPPING[sex]), sex, "")
        elif row["age"] is None:
            value = ("", "", "age_missing")
        else:
            value = (str(float(row["age"])), "", "")
        previous = subject_values.setdefault(subject_id, value)
        if previous != value:
            raise ValueError(f"Inconsistent subject label for {subject_id}: {previous} != {value}")
    return [
        {
            "subject_id": subject_id,
            "task": task,
            "label": label,
            "label_name": label_name,
            "exclusion_reason": exclusion_reason,
        }
        for subject_id, (label, label_name, exclusion_reason) in sorted(subject_values.items())
    ]


def _assign_splits(rows, seed):
    included = [row["subject_id"] for row in rows if not row["exclusion_reason"]]
    if len(included) < 3:
        raise ValueError("Need at least three eligible subjects for train/validation/test")
    rng = np.random.default_rng(seed)
    shuffled = [included[index] for index in rng.permutation(len(included))]
    n_validation = 1
    n_test = 1
    assignment = {subject: "train" for subject in shuffled[: len(shuffled) - 2]}
    assignment[shuffled[-2]] = "validation"
    assignment[shuffled[-1]] = "test"
    split_rows = []
    for row in rows:
        copied = dict(row)
        copied["split"] = assignment.get(row["subject_id"], "")
        copied["split_seed"] = seed
        split_rows.append(copied)
    sets = {
        name: {row["subject_id"] for row in split_rows if row["split"] == name}
        for name in ("train", "validation", "test")
    }
    if sets["train"] & sets["validation"] or sets["train"] & sets["test"] or sets["validation"] & sets["test"]:
        raise AssertionError("Generated subject splits overlap")
    return split_rows


def main(argv=None):
    args = parse_args(argv)
    options = _storage_options(args.storage_options)
    dataset = lance.dataset(args.input_lance, **({} if options is None else {"storage_options": options}))
    required = {"subject_id", "age", "sex"}
    missing = sorted(required.difference(dataset.schema.names))
    if missing:
        raise ValueError(f"Input Lance missing fields: {missing}")
    rows = _subject_rows(dataset, args.task)
    split_rows = _assign_splits(rows, args.seed)
    manifest_fields = ["subject_id", "task", "label", "label_name", "exclusion_reason"]
    split_fields = manifest_fields + ["split", "split_seed"]
    _write(args.output_manifest, rows, manifest_fields)
    _write(args.output_splits, split_rows, split_fields)
    print(json.dumps({
        "task": args.task,
        "subjects": len(rows),
        "eligible_subjects": sum(not row["exclusion_reason"] for row in rows),
        "excluded": [row for row in rows if row["exclusion_reason"]],
        "output_manifest": str(args.output_manifest.resolve()),
        "output_splits": str(args.output_splits.resolve()),
        "seed": args.seed,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
