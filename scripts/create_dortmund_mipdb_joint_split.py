#!/usr/bin/env python3
"""Freeze the Dortmund+MIPDB subject split and joint age transform."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOWNSTREAM = HERE / "downstream_50m"
if str(DOWNSTREAM) not in sys.path:
    sys.path.insert(0, str(DOWNSTREAM))

from age_joint_training import (
    build_joint_target_transform,
    canonical_csv_sha256,
    canonical_json_sha256,
    validate_joint_manifest,
)
from dortmund_age_finetune import sha256_file


MIPDB_VALIDATION = {"A00054039", "A00062125"}  # ages 22, 32
MIPDB_TEST = {"A00058775", "A00062165"}  # ages 23, 31
UNRELEASED = {"A00062919", "A00063051", "A00063117", "A00063377"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite frozen artifact: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dortmund-manifest", type=Path, required=True)
    parser.add_argument("--mipdb-metadata", type=Path, required=True)
    parser.add_argument("--mipdb-allowlist", type=Path, required=True)
    parser.add_argument("--mipdb-data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    allowlist = [line.strip() for line in args.mipdb_allowlist.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    metadata = {row["ID"].strip(): row for row in read_csv(args.mipdb_metadata)}
    released = [subject for subject in allowlist if (args.mipdb_data_dir / f"sub-{subject}").is_dir()]
    if set(allowlist) - set(released) != UNRELEASED:
        raise ValueError(
            f"Released MIPDB set changed; review and version a new split. "
            f"Currently absent={sorted(set(allowlist) - set(released))}"
        )
    mipdb_rows: list[dict[str, object]] = []
    for subject in released:
        age = float(metadata[subject]["Age"])
        split = "validation" if subject in MIPDB_VALIDATION else "test" if subject in MIPDB_TEST else "train"
        mipdb_rows.append({
            "dataset_id": "mipdb", "subject_id": f"sub-{subject}", "age": f"{age:g}",
            "split": split, "task": "age_regression", "label": str(float(age)),
            "block_filter": "block01", "split_seed": "42",
            "stratification": "deterministic_age_only_near_stratified_v1",
            "sex_used_for_stratification": "false",
        })

    source_dortmund = read_csv(args.dortmund_manifest)
    dortmund_rows: list[dict[str, object]] = []
    for row in source_dortmund:
        age = float(row["age"])
        if 20 <= age <= 35:
            dortmund_rows.append({
                "dataset_id": "dortmund", "subject_id": row["subject_id"],
                "age": f"{age:g}", "split": row["split"], "task": "age_regression",
                "label": str(float(age)), "block_filter": "",
                "split_seed": row.get("split_seed", "42"),
                "stratification": "inherited_frozen_dortmund_manifest",
                "sex_used_for_stratification": "inherited",
            })
    joint_rows = [*dortmund_rows, *mipdb_rows]
    validate_joint_manifest(joint_rows, strict_counts=True)

    fields = [
        "dataset_id", "subject_id", "age", "split", "task", "label",
        "block_filter", "split_seed", "stratification", "sex_used_for_stratification",
    ]
    mipdb_path = args.output_dir / "mipdb_subject_split_manifest_v1.csv"
    joint_path = args.output_dir / "joint_subject_split_manifest_v1.csv"
    transform_path = args.output_dir / "joint_target_transform_v1.json"
    contract_path = args.output_dir / "joint_age_training_contract_v1.json"
    for path in (mipdb_path, joint_path, transform_path, contract_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite frozen artifact: {path}")
    write_csv(mipdb_path, mipdb_rows, fields)
    write_csv(joint_path, joint_rows, fields)

    transform = build_joint_target_transform(joint_rows, canonical_csv_sha256(joint_path))
    transform_path.write_text(
        json.dumps(asdict(transform), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    contract = {
        "contract_version": "dortmund-mipdb-joint-age-v1",
        "dortmund_split_policy": "inherited_172_21_21_unchanged",
        "mipdb_split_policy": "age_only_deterministic_15_2_2",
        "mipdb_block_filter": "block01",
        "mipdb_sex_used_for_stratification": False,
        "joint_train_subject_count": 187,
        "target_transform": asdict(transform),
        "fingerprint_policy": "canonical_content_v1",
        "source": {
            "dortmund_manifest": str(args.dortmund_manifest),
            "dortmund_manifest_sha256": sha256_file(args.dortmund_manifest),
            "mipdb_metadata": str(args.mipdb_metadata),
            "mipdb_metadata_sha256": sha256_file(args.mipdb_metadata),
            "mipdb_allowlist": str(args.mipdb_allowlist),
            "mipdb_allowlist_sha256": sha256_file(args.mipdb_allowlist),
        },
        "sha256": {
            "mipdb_subject_split_manifest": sha256_file(mipdb_path),
            "joint_subject_split_manifest": sha256_file(joint_path),
            "joint_target_transform": sha256_file(transform_path),
        },
        "canonical_sha256": {
            "mipdb_subject_split_manifest": canonical_csv_sha256(mipdb_path),
            "joint_subject_split_manifest": canonical_csv_sha256(joint_path),
            "joint_target_transform": canonical_json_sha256(transform_path),
        },
    }
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(contract, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
