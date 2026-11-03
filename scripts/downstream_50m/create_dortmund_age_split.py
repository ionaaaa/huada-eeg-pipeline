#!/usr/bin/env python3
"""Create a frozen, deterministic Dortmund subject-level age split manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import lance

try:
    from .dortmund_age_split import (
        MANIFEST_FIELDS, SPLITS, build_subject_split, distribution_rows,
        dump_json, subjects_from_window_rows, write_manifest,
    )
except ImportError:
    from dortmund_age_split import (
        MANIFEST_FIELDS, SPLITS, build_subject_split, distribution_rows,
        dump_json, subjects_from_window_rows, write_manifest,
    )


def write_csv(path: Path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-lance", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    dataset = lance.dataset(args.input_lance)
    required = ["sample_id", "subject_id", "age", "sex", "recording_id", "eyes_condition"]
    missing = sorted(set(required).difference(dataset.schema.names))
    if missing:
        raise ValueError(f"Token Lance is missing split metadata: {missing}")
    window_rows = dataset.to_table(columns=required).to_pylist()
    subjects = subjects_from_window_rows(window_rows)
    manifest, summary = build_subject_split(
        subjects,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        split_seed=args.split_seed,
    )
    manifest_path = args.output_dir / "subject_split_manifest.csv"
    digest = write_manifest(manifest_path, manifest)
    summary["manifest_sha256"] = digest
    summary["input_lance"] = str(args.input_lance.resolve())
    summary["input_lance_version"] = int(dataset.version)
    dump_json(args.output_dir / "split_summary.json", summary)
    config = {
        "input_lance": str(args.input_lance.resolve()),
        "train_ratio": args.train_ratio,
        "validation_ratio": args.validation_ratio,
        "test_ratio": args.test_ratio,
        "split_seed": args.split_seed,
        "integer_allocation_rule": "validation=round(N*validation_ratio); test=round(N*test_ratio); train=remainder",
        "manifest_sha256": digest,
    }
    dump_json(args.output_dir / "resolved_split_config.json", config)
    age_rows, age_sex_rows = distribution_rows(manifest)
    write_csv(args.output_dir / "split_age_distribution.csv", age_rows, ["split", "age", "subject_count"])
    write_csv(args.output_dir / "split_age_sex_distribution.csv", age_sex_rows, ["split", "age", "sex", "subject_count"])
    issues = []
    for row in manifest:
        if row["stratification_level"] != "exact_age_x_sex":
            issues.append({
                "severity": "info", "subject_id": row["subject_id"],
                "issue_type": "sparse_stratum_fallback",
                "detail": row["stratification_reason"],
            })
    write_csv(args.output_dir / "split_issues.csv", issues, ["severity", "subject_id", "issue_type", "detail"])
    print(json.dumps({"output_dir": str(args.output_dir), "manifest_sha256": digest, "splits": summary["splits"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
