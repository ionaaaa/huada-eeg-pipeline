#!/usr/bin/env python3
"""Audit Dortmund common windows and adapt selected rows to 50M FBMAE tokens.

Signal/token math is delegated to ``build_huada_model_tokens``.  This module
owns dataset discovery, Profile-A audit, authoritative electrode identity,
Dortmund metadata, smoke selection, reporting, and transactional output.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import tomllib
import traceback
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import lance
import numpy as np
import pyarrow as pa

import build_huada_model_tokens as token_core
from preprocess_dortmund_age_windows import (
    DORTMUND_CHANNELS,
    PROFILE_A_RECORDINGS,
    PROFILE_A_SUBJECTS,
    build_cohort_id,
    dortmund_lance_schema,
)


VERSION = "dortmund-fbmae-adapter-v1.1"
REQUIRED_OUTPUTS = {
    "adapter_report.md", "adapter_summary.json", "processing_provenance.json",
    "resolved_config.json", "input_dataset_audit.json", "input_schema.json",
    "input_subject_summary.csv", "input_recording_summary.csv",
    "input_age_distribution.csv", "input_issues.csv", "channel_mapping.csv",
    "channel_mapping_summary.json", "electrode_vocabulary_snapshot.json",
    "channel_profile_resolved.json", "sample_mapping_inventory.csv",
    "token_schema.json", "token_statistics.json", "mask_statistics.json",
    "label_statistics.json", "adaptation_issues.csv", "adaptation_errors.log",
    "source_snapshot_before.json", "source_snapshot_after.json",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_metadata(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(["git", "-C", str(root), *args], check=True, text=True, capture_output=True).stdout.strip()
    try:
        return {"commit": run("rev-parse", "HEAD"), "branch": run("branch", "--show-current") or None, "dirty": bool(run("status", "--porcelain"))}
    except Exception as exc:
        return {"commit": None, "branch": None, "dirty": None, "error": str(exc)}


def snapshot_tree(path: Path) -> dict[str, Any]:
    rows = []
    for item in sorted(path.rglob("*"), key=lambda value: value.as_posix()):
        if item.is_file():
            stat = item.stat()
            rows.append({"path": item.relative_to(path).as_posix(), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    fingerprint = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"root": str(path.resolve()), "file_count": len(rows), "fingerprint": fingerprint, "files": rows}


def discover_input_lance(input_dir: Path) -> tuple[Path, list[dict[str, Any]]]:
    candidates = sorted(path for path in input_dir.rglob("*.lance") if path.is_dir())
    inspected: list[dict[str, Any]] = []
    valid: list[Path] = []
    expected = dortmund_lance_schema()
    for candidate in candidates:
        row: dict[str, Any] = {"path": str(candidate.resolve())}
        try:
            dataset = lance.dataset(candidate)
            row.update({"readable": True, "rows": dataset.count_rows(), "schema_matches": dataset.schema == expected})
            if dataset.schema == expected:
                valid.append(candidate)
        except Exception as exc:
            row.update({"readable": False, "error": str(exc), "schema_matches": False})
        inspected.append(row)
    if len(valid) != 1:
        raise RuntimeError(
            f"Expected exactly one readable Dortmund common-window Lance under {input_dir}; "
            f"found valid={len(valid)}, candidates={inspected}"
        )
    return valid[0], inspected


def load_subject_sidecar(input_dir: Path) -> dict[str, dict[str, str]]:
    candidates = list(input_dir.rglob("subject_manifest.csv"))
    if len(candidates) > 1:
        raise RuntimeError(f"Multiple subject_manifest.csv files: {candidates}")
    if not candidates:
        return {}
    with candidates[0].open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for row in rows:
        subject_id = str(row.get("subject_id", "")).strip()
        if subject_id:
            if subject_id in result:
                raise RuntimeError(f"Duplicate subject in manifest: {subject_id}")
            result[subject_id] = row
    return result


def parse_smoke_subjects(values: Sequence[str] | None) -> list[str]:
    result: list[str] = []
    for value in values or []:
        for item in value.split(","):
            normalized = item.strip().lower()
            if normalized and normalized not in result:
                result.append(normalized)
    return result


def choose_smoke_indices(
    metadata: Sequence[dict[str, Any]],
    requested_subjects: Sequence[str],
    max_windows_per_recording: int,
    min_age: int = 20,
    max_age: int = 35,
) -> tuple[list[int], dict[str, Any]]:
    if max_windows_per_recording <= 0:
        raise ValueError("--smoke-max-windows-per-recording must be positive")
    by_subject: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(metadata):
        by_subject[str(row["subject_id"]).lower()].append((index, row))
    requested = list(requested_subjects)
    if not requested:
        # Deterministic age representatives; include known edge cases when present.
        chosen = [subject for subject in ("sub-230", "sub-600") if subject in by_subject]
        subject_age = {subject: float(rows[0][1]["age"]) for subject, rows in by_subject.items()}
        midpoint = (float(min_age) + float(max_age)) / 2.0
        for target in (float(min_age), midpoint, float(max_age)):
            candidate = min(subject_age, key=lambda subject: (abs(subject_age[subject] - target), subject))
            if candidate not in chosen:
                chosen.append(candidate)
        requested = chosen[:5]
    missing = sorted(set(requested) - set(by_subject))
    if missing:
        raise ValueError(f"Requested smoke subjects absent from input: {missing}")
    selected: list[int] = []
    recording_counts: dict[str, int] = {}
    for subject in requested:
        grouped: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
        for item in by_subject[subject]:
            grouped[str(item[1]["recording_id"])].append(item)
        for recording, rows in sorted(grouped.items()):
            rows.sort(key=lambda item: (int(item[1].get("window_index", 0)), float(item[1]["start_time"]), item[0]))
            chosen_rows = rows[:max_windows_per_recording]
            selected.extend(index for index, _ in chosen_rows)
            recording_counts[recording] = len(chosen_rows)
    selected.sort()
    return selected, {
        "subjects": requested,
        "rows": len(selected),
        "recording_counts": recording_counts,
        "selection_basis": "explicit subject list" if requested_subjects else "deterministic age targets plus known edge cases",
    }


def audit_input(
    dataset: lance.LanceDataset,
    input_dir: Path,
    inspected_candidates: Sequence[dict[str, Any]],
    data_indices: Sequence[int] | None,
    expected_subjects: int,
    expected_recordings: int,
    min_age: int = 20,
    max_age: int = 35,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, str]], list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []

    def issue(severity: str, kind: str, detail: str, sample_id: str = "", subject_id: str = "") -> None:
        issues.append({"severity": severity, "stage": "input_audit", "issue_type": kind, "sample_id": sample_id, "subject_id": subject_id, "detail": detail})

    expected_schema = dortmund_lance_schema()
    if dataset.schema != expected_schema:
        issue("error", "schema_mismatch", f"actual={dataset.schema}; expected={expected_schema}")
    if "split" in dataset.schema.names:
        issue("error", "forbidden_split", "Input contains split")
    metadata_columns = [name for name in dataset.schema.names if name != "data"]
    metadata = dataset.to_table(columns=metadata_columns).to_pylist()
    sample_ids = [str(row.get("sample_id", "")) for row in metadata]
    if any(not value for value in sample_ids):
        issue("error", "empty_sample_id", "One or more sample_id values are empty")
    duplicate_ids = [value for value, count in Counter(sample_ids).items() if count > 1]
    if duplicate_ids:
        issue("error", "duplicate_sample_id", f"count={len(duplicate_ids)} examples={duplicate_ids[:10]}")

    subject_sidecar = load_subject_sidecar(input_dir)
    subject_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    recording_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in metadata:
        sid = str(row["subject_id"])
        sample_id = str(row["sample_id"])
        subject_rows[sid].append(row)
        recording_rows[(sid, str(row["recording_id"]))].append(row)
        if str(row.get("session_id", "")).lower() != "ses-1":
            issue("error", "non_profile_session", str(row.get("session_id")), sample_id, sid)
        if str(row.get("pre_or_post", "")).lower() != "pre_cognitive":
            issue("error", "non_profile_acquisition", str(row.get("pre_or_post")), sample_id, sid)
        if "acq-pre" not in sample_id.lower() or "acq-post" in sample_id.lower():
            issue("error", "non_profile_acquisition_identity", sample_id, sample_id, sid)
        if str(row.get("eyes_condition", "")).lower() not in {"eyes_open", "eyes_closed", "eyes-open", "eyes-closed"}:
            issue("error", "non_profile_condition", str(row.get("eyes_condition")), sample_id, sid)
        if row.get("qc_pass") is not True:
            issue("error", "qc_pass", str(row.get("qc_pass")), sample_id, sid)
        if str(row.get("modality", "")).lower() != "eeg":
            issue("error", "modality", str(row.get("modality")), sample_id, sid)
        if not str(row.get("recording_id", "")).strip() or not str(row.get("source_path", "")).strip():
            issue("error", "missing_provenance_field", "recording_id/source_path empty", sample_id, sid)
        if str(row.get("task_type", "")) != "regression" or str(row.get("target_name", "")) != "age" or str(row.get("target_unit", "")) != "year":
            issue("error", "task_contract", f"{row.get('task_type')}/{row.get('target_name')}/{row.get('target_unit')}", sample_id, sid)
        names = [str(value) for value in row.get("channel_names", [])]
        if "Status" in names or any(name.lower() == "status" for name in names):
            issue("error", "status_channel_present", "Status must be excluded", sample_id, sid)
        if len(names) != 64:
            issue("error", "channel_count", f"expected=64 actual={len(names)}", sample_id, sid)
        if tuple(row.get("shape", [])) != (64, 2000):
            issue("error", "shape", str(row.get("shape")), sample_id, sid)
        if tuple(row.get("original_shape", [])) != (64, 2000) or int(row.get("valid_length", 0)) != 128000:
            issue("error", "original_shape_or_valid_length", f"shape={row.get('original_shape')} valid_length={row.get('valid_length')}", sample_id, sid)
        if not math.isclose(float(row.get("target_sampling_rate", 0)), 200.0):
            issue("error", "target_sampling_rate", str(row.get("target_sampling_rate")), sample_id, sid)
        if not math.isclose(float(row.get("duration", 0)), 10.0):
            issue("error", "duration", str(row.get("duration")), sample_id, sid)
        age = float(row["age"])
        label = float(row["label"])
        if not min_age <= age <= max_age or age != label:
            issue("error", "age_label", f"age={age} label={label}", sample_id, sid)
        masks = np.asarray(row["channel_mask"], dtype=bool)
        bad = np.asarray(row["bad_channel_mask"], dtype=bool)
        status = [str(value).lower() for value in row["channel_status"]]
        if masks.shape != (64,) or bad.shape != (64,) or len(status) != 64:
            issue("error", "mask_shape", f"mask={masks.shape} bad={bad.shape} status={len(status)}", sample_id, sid)
        elif any(
            (bad[i] != (status[i] == "bad"))
            or (not masks[i] and status[i] not in {"absent", "missing"})
            or (masks[i] and not bad[i] and status[i] != "good")
            for i in range(64)
        ):
            issue("error", "mask_status_inconsistent", "channel masks and status disagree", sample_id, sid)
        if int(row.get("bad_channel_count", -1)) != int(bad.sum()):
            issue("error", "bad_channel_count", f"field={row.get('bad_channel_count')} mask={int(bad.sum())}", sample_id, sid)

    if len(subject_rows) != expected_subjects:
        issue("error", "subject_count", f"expected={expected_subjects} actual={len(subject_rows)}")
    if len(recording_rows) != expected_recordings:
        issue("error", "recording_count", f"expected={expected_recordings} actual={len(recording_rows)}")
    for sid, rows in subject_rows.items():
        ages = {float(row["age"]) for row in rows}
        conditions = {str(row["eyes_condition"]).lower().replace("-", "_") for row in rows}
        recording_ids = {str(row["recording_id"]) for row in rows}
        if len(ages) != 1:
            issue("error", "subject_age_inconsistent", str(sorted(ages)), subject_id=sid)
        if conditions != {"eyes_open", "eyes_closed"}:
            issue("error", "subject_conditions", str(sorted(conditions)), subject_id=sid)
        if len(recording_ids) != 2:
            issue("error", "subject_recording_count", f"expected=2 actual={len(recording_ids)}", subject_id=sid)
        sidecar_age = subject_sidecar.get(sid, {}).get("age", "")
        if str(sidecar_age).strip() and float(sidecar_age) != next(iter(ages)):
            issue("error", "subject_sidecar_age_mismatch", f"sidecar={sidecar_age} lance={sorted(ages)}", subject_id=sid)

    selected = list(range(len(metadata))) if data_indices is None else list(data_indices)
    data_hasher = hashlib.sha256()
    for offset in range(0, len(selected), 32):
        indices = selected[offset : offset + 32]
        rows = dataset.take(indices=indices, columns=["sample_id", "subject_id", "data", "bad_channel_mask"]).to_pylist()
        for row in rows:
            values = np.asarray(row["data"], dtype=np.float32).reshape(64, 2000)
            data_hasher.update(str(row["sample_id"]).encode("utf-8"))
            data_hasher.update(values.astype("<f4", copy=False).tobytes(order="C"))
            bad = np.asarray(row["bad_channel_mask"], dtype=bool)
            if not np.isfinite(values).all():
                issue("error", "non_finite_data", "NaN/Inf", str(row["sample_id"]), str(row["subject_id"]))
            if values.min() < -1.000001 or values.max() > 1.000001:
                issue("error", "value_range", f"min={values.min()} max={values.max()}", str(row["sample_id"]), str(row["subject_id"]))
            if bad.any() and np.any(values[bad] != 0):
                issue("error", "bad_channel_not_zero", "bad-channel samples are nonzero", str(row["sample_id"]), str(row["subject_id"]))

    subject_summary = []
    for sid, rows in sorted(subject_rows.items()):
        sidecar = subject_sidecar.get(sid, {})
        subject_summary.append({
            "subject_id": sid, "age": float(rows[0]["age"]), "sex": sidecar.get("sex", ""),
            "window_count": len(rows), "recording_count": len({row["recording_id"] for row in rows}),
            "conditions": ";".join(sorted({str(row["eyes_condition"]) for row in rows})),
        })
    recording_summary = []
    for (sid, rid), rows in sorted(recording_rows.items()):
        recording_summary.append({
            "subject_id": sid, "recording_id": rid, "condition": rows[0]["eyes_condition"],
            "session_id": rows[0]["session_id"], "acquisition": rows[0]["pre_or_post"],
            "source_path": rows[0]["source_path"], "age": float(rows[0]["age"]),
            "window_count": len(rows), "start_min": min(float(row["start_time"]) for row in rows),
            "start_max": max(float(row["start_time"]) for row in rows),
        })
    age_distribution = [{"age": age, "subject_count": count} for age, count in sorted(Counter(float(rows[0]["age"]) for rows in subject_rows.values()).items())]
    hard_errors = [row for row in issues if row["severity"] == "error"]
    audit = {
        "status": "passed" if not hard_errors and data_indices is None else "passed_smoke_scope" if not hard_errors else "failed",
        "data_validation_scope": "all_rows" if data_indices is None else "selected_smoke_rows",
        "configured_min_age": min_age, "configured_max_age": max_age,
        "expected_subject_count": expected_subjects,
        "expected_recording_count": expected_recordings,
        "schema_matches": dataset.schema == expected_schema,
        "row_count": len(metadata), "subject_count": len(subject_rows), "recording_count": len(recording_rows),
        "sample_id_unique": len(sample_ids) == len(set(sample_ids)), "age_min": min(float(row["age"]) for row in metadata),
        "age_max": max(float(row["age"]) for row in metadata), "split_absent": "split" not in dataset.schema.names,
        "status_channel_excluded": all("Status" not in row["channel_names"] for row in metadata),
        "inspected_lance_candidates": list(inspected_candidates), "hard_error_count": len(hard_errors),
        "input_data_fingerprint_sha256": data_hasher.hexdigest(),
        "input_data_fingerprint_scope": "all_rows" if data_indices is None else "selected_smoke_rows",
        "input_data_fingerprint_algorithm": "sha256(sample_id UTF-8 + little-endian float32 [64,2000] bytes, row order)",
    }
    return audit, metadata, subject_summary, recording_summary, age_distribution, subject_sidecar, issues


def load_channel_profile(path: Path, vocab_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], list[str]]:
    profile = tomllib.loads(path.read_text(encoding="utf-8"))
    source_names = [str(value) for value in profile["source"]["channel_names"]]
    profile_ids = [int(value) for value in profile["source"]["electrode_ids"]]
    model_channels = [str(value) for value in profile["model"]["standard_channels"]]
    if len(source_names) != len(profile_ids) or len(source_names) != 64:
        raise ValueError("Dortmund channel profile must contain 64 source names and IDs")
    if len(model_channels) != 64 or len(set(model_channels)) != 64:
        raise ValueError("FBMAE model profile must contain 64 unique standard channels")
    vocab_raw = json.loads(vocab_path.read_text(encoding="utf-8"))
    id_to_name = {int(key): str(value) for key, value in vocab_raw.items()}
    canonical_candidates: dict[str, list[int]] = defaultdict(list)
    authoritative: dict[str, int] = {}
    for key, value in vocab_raw.items():
        canonical = token_core.canonical_channel_name(value).lower()
        canonical_candidates[canonical].append(int(key))
        authoritative[canonical] = int(key)  # exact current 50M last-wins behavior
    rows = []
    resolved_ids = []
    for index, name in enumerate(source_names):
        canonical = token_core.canonical_channel_name(name)
        candidates = canonical_candidates.get(canonical.lower(), [])
        electrode_id = authoritative.get(canonical.lower())
        resolved_ids.append(electrode_id)
        rows.append({
            "source_index": index, "source_channel_name": name, "canonical_channel_name": canonical,
            "electrode_id": "" if electrode_id is None else electrode_id,
            "mapping_source": "50M electrode_vocab.json last-wins canonical lookup",
            "mapping_status": "unknown" if electrode_id is None else "mapped",
            "notes": "vocab candidate IDs=" + ",".join(str(value) for value in candidates),
            "fbmae_standard_slot": model_channels.index(canonical) if canonical in model_channels else "",
        })
    unknown = [row["source_channel_name"] for row in rows if row["mapping_status"] != "mapped"]
    duplicate_ids = [value for value, count in Counter(value for value in resolved_ids if value is not None).items() if count > 1]
    profile_mismatches = [
        {"channel": name, "profile_id": expected, "resolved_id": actual}
        for name, expected, actual in zip(source_names, profile_ids, resolved_ids) if expected != actual
    ]
    if unknown or duplicate_ids or profile_mismatches:
        raise ValueError(f"Channel mapping blocked: unknown={unknown}, duplicate_ids={duplicate_ids}, profile_mismatches={profile_mismatches}")
    dropped = [row["source_channel_name"] for row in rows if row["fbmae_standard_slot"] == ""]
    missing_model = [name for name in model_channels if name not in {row["canonical_channel_name"] for row in rows}]
    summary = {
        "channel_mapping_complete": True, "source_channel_count": 64,
        "unknown_channel_count": 0, "mapping_collision_count": 0,
        "duplicate_electrode_id_count": 0, "positional_huada_mapping_used": False,
        "fbmae_standard_slots_covered": 64 - len(missing_model),
        "dortmund_channels_not_in_fixed_fbmae_slots": dropped,
        "missing_fixed_fbmae_slots": missing_model,
        "authoritative_scatter_required": True,
        "reference": profile["profile"],
    }
    vocab_snapshot = {"source_path": str(vocab_path.resolve()), "sha256": sha256_file(vocab_path), "entry_count": len(vocab_raw), "entries": vocab_raw}
    model_ids = [authoritative.get(token_core.canonical_channel_name(name).lower()) for name in model_channels]
    missing_model_vocab = [name for name, value in zip(model_channels, model_ids) if value is None]
    if missing_model_vocab:
        raise ValueError(f"Current model channel_config contains channels absent from electrode vocabulary: {missing_model_vocab}")
    resolved = {
        **profile,
        "resolved_source_electrode_ids": resolved_ids,
        "resolved_model_electrode_ids": model_ids,
        "vocabulary_sha256": vocab_snapshot["sha256"],
    }
    return resolved, rows, summary, model_channels


def token_schema() -> pa.Schema:
    return pa.schema([
        pa.field("num_tokens", pa.int64(), False), pa.field("patch_len", pa.int64(), False),
        pa.field("n_bands", pa.int64(), False), pa.field("frames_per_patch", pa.int64(), False),
        pa.field("n_channels", pa.int64(), False), pa.field("num_patches", pa.int64(), False),
        pa.field("token_inputs", pa.list_(pa.float32(), 64000), False),
        pa.field("targets", pa.list_(pa.float32(), 16000), False),
        pa.field("token_channel_indices", pa.list_(pa.int64(), 640), False),
        pa.field("token_time_indices", pa.list_(pa.int64(), 640), False),
        pa.field("token_valid_mask", pa.list_(pa.float32(), 640), False),
        pa.field("channel_valid_mask", pa.list_(pa.float32(), 64), False),
        pa.field("sample_id", pa.string(), False), pa.field("subject_id", pa.string(), False),
        pa.field("source_path", pa.string(), False), pa.field("recording_id", pa.string(), False),
        pa.field("window_index", pa.int64(), False), pa.field("start_time", pa.float64(), False),
        pa.field("qc_pass", pa.bool_(), False), pa.field("age", pa.float32(), False),
        pa.field("label", pa.float32(), False), pa.field("sex", pa.string(), False),
        pa.field("valid_channel_count", pa.int64(), False), pa.field("source_sfreq", pa.float32(), False),
        pa.field("model_sfreq", pa.float32(), False), pa.field("model_preprocessing_version", pa.string(), False),
        pa.field("source_preprocessing_config_hash", pa.string(), False),
        pa.field("session_id", pa.string(), False), pa.field("eyes_condition", pa.string(), False),
        pa.field("run_id", pa.string(), False), pa.field("acquisition", pa.string(), False),
        pa.field("pre_or_post", pa.string(), False), pa.field("duration", pa.float64(), False),
        pa.field("source_start_sample", pa.int64(), False), pa.field("target_start_sample", pa.int64(), False),
        pa.field("task_type", pa.string(), False),
        pa.field("target_name", pa.string(), False), pa.field("target_unit", pa.string(), False),
        pa.field("source_channel_names", pa.list_(pa.string()), False),
        pa.field("electrode_ids", pa.list_(pa.int64(), 64), False),
        pa.field("token_electrode_ids", pa.list_(pa.int64(), 640), False),
        pa.field("source_electrode_ids", pa.list_(pa.int64(), 64), False),
        # Float masks match the established model-token schema and avoid a
        # fixed-size-list<bool> writer bug in some Lance stable encoders.
        pa.field("source_channel_mask", pa.list_(pa.float32(), 64), False),
        pa.field("source_bad_channel_mask", pa.list_(pa.float32(), 64), False),
        pa.field("model_channel_names", pa.list_(pa.string()), False),
    ])


def rows_to_table(rows: Sequence[dict[str, Any]]) -> pa.Table:
    schema = token_schema()
    arrays = []
    for field in schema:
        values = [row[field.name] for row in rows]
        if pa.types.is_fixed_size_list(field.type):
            size = field.type.list_size
            if pa.types.is_float32(field.type.value_type):
                flat = np.concatenate([np.asarray(value, dtype=np.float32).reshape(-1) for value in values])
            elif pa.types.is_int64(field.type.value_type):
                flat = np.concatenate([np.asarray(value, dtype=np.int64).reshape(-1) for value in values])
            elif pa.types.is_boolean(field.type.value_type):
                flat = np.concatenate([np.asarray(value, dtype=bool).reshape(-1) for value in values])
            else:
                flat = [item for value in values for item in value]
            arrays.append(pa.FixedSizeListArray.from_arrays(pa.array(flat, type=field.type.value_type), size))
        else:
            arrays.append(pa.array(values, type=field.type))
    return pa.Table.from_arrays(arrays, schema=schema)


def build_token_row(
    row: dict[str, Any],
    model_channels: Sequence[str],
    source_electrode_ids: Sequence[int],
    model_electrode_ids: Sequence[int],
    sex: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    # Dortmund common windows encode the index through target_start_sample;
    # the shared Huada tokenizer expects an explicit window_index metadata key.
    token_source = dict(row)
    token_source["window_index"] = int(
        row.get("window_index", int(row["target_start_sample"]) // int(10 * token_core.SOURCE_SFREQ))
    )
    processed, audit = token_core.process_row(token_source, standard_channels=model_channels)
    age = np.float32(row["age"])
    processed.update({
        "age": age, "label": age, "sex": str(sex),
        "session_id": str(row["session_id"]), "eyes_condition": str(row["eyes_condition"]),
        "run_id": str(row["run_id"]), "acquisition": "pre",
        "pre_or_post": str(row["pre_or_post"]), "duration": float(row["duration"]),
        "source_start_sample": int(row["source_start_sample"]),
        "target_start_sample": int(row["target_start_sample"]), "task_type": "regression",
        "target_name": "age", "target_unit": "year",
        "source_channel_names": [str(value) for value in row["channel_names"]],
        # ``electrode_ids`` aligns with the 64 model data slots; source IDs are
        # retained separately so Dortmund-only electrodes are never relabelled.
        "electrode_ids": np.asarray(model_electrode_ids, dtype=np.int64),
        "token_electrode_ids": np.repeat(np.asarray(model_electrode_ids, dtype=np.int64), token_core.NUM_TIME_PATCHES),
        "source_electrode_ids": np.asarray(source_electrode_ids, dtype=np.int64),
        "source_channel_mask": np.asarray(row["channel_mask"], dtype=np.float32),
        "source_bad_channel_mask": np.asarray(row["bad_channel_mask"], dtype=np.float32),
        "model_channel_names": list(model_channels),
    })
    return processed, audit


def validate_token_dataset(path: Path, expected_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    dataset = lance.dataset(path)
    if dataset.schema != token_schema():
        raise AssertionError(f"Token schema mismatch: {dataset.schema}")
    if "split" in dataset.schema.names:
        raise AssertionError("Token output contains split")
    expected_by_id = {str(row["sample_id"]): row for row in expected_rows}
    actual_ids: set[str] = set()
    all_finite = True
    mask_ok = True
    actual_count = 0
    for batch in dataset.scanner(batch_size=32).to_batches():
        for row in batch.to_pylist():
            actual_count += 1
            sample_id = str(row["sample_id"])
            if sample_id in actual_ids:
                raise AssertionError(f"Duplicate output sample_id: {sample_id}")
            actual_ids.add(sample_id)
            source = expected_by_id.get(sample_id)
            if source is None:
                raise AssertionError(f"Unexpected output sample_id: {sample_id}")
            if float(row["label"]) != float(source["age"]) or float(row["age"]) != float(source["age"]):
                raise AssertionError(f"Age/label parity failed: {sample_id}")
            if (
                row["recording_id"] != source["recording_id"]
                or row["eyes_condition"] != source["eyes_condition"]
                or row["session_id"] != source["session_id"]
                or row["run_id"] != source["run_id"]
                or row["acquisition"] != "pre"
                or row["source_path"] != source["source_path"]
            ):
                raise AssertionError(f"Metadata parity failed: {sample_id}")
            inputs = np.asarray(row["token_inputs"], dtype=np.float32).reshape(640, 100)
            targets = np.asarray(row["targets"], dtype=np.float32).reshape(640, 5, 5)
            mask = np.asarray(row["token_valid_mask"], dtype=np.float32)
            channels = np.asarray(row["channel_valid_mask"], dtype=np.float32)
            all_finite &= bool(np.isfinite(inputs).all() and np.isfinite(targets).all())
            mask_ok &= bool(np.array_equal(mask, channels[np.repeat(np.arange(64), 10)]))
            if np.any(inputs[mask == 0] != 0) or np.any(targets[mask == 0] != 0):
                raise AssertionError(f"Invalid token zero semantics failed: {sample_id}")
    if actual_ids != set(expected_by_id):
        raise AssertionError("Input/output sample_id parity failed")
    # Exercise the actual ported 50M dataset and collate path.
    from downstream_50m.dataset import collate_fn
    from downstream_50m.lance_dataset import LanceEEGPretrainDataset
    loader_dataset = LanceEEGPretrainDataset(path, task="age_regression")
    samples = [loader_dataset[index] for index in range(min(2, len(loader_dataset)))]
    batch = collate_fn(samples)
    loader_ok = tuple(batch["token_inputs"].shape[1:]) == (640, 100) and batch["label"].dtype.is_floating_point
    return {
        "rows": actual_count, "sample_id_parity": "passed", "metadata_parity": "passed",
        "age_label_parity": "passed", "all_finite": all_finite, "mask_propagation": mask_ok,
        "lance_roundtrip": "passed", "loader_batch_validation": "passed" if loader_ok else "failed",
        "loader_batch_shape": list(batch["token_inputs"].shape),
    }


def write_stage_outputs(
    root: Path,
    *,
    input_audit: dict[str, Any],
    schema: pa.Schema,
    subject_summary: Sequence[dict[str, Any]],
    recording_summary: Sequence[dict[str, Any]],
    age_distribution: Sequence[dict[str, Any]],
    input_issues: Sequence[dict[str, Any]],
) -> None:
    json_dump(root / "input_dataset_audit.json", input_audit)
    json_dump(root / "input_schema.json", {"schema": str(schema), "fields": [{"name": field.name, "type": str(field.type), "nullable": field.nullable} for field in schema]})
    write_csv(root / "input_subject_summary.csv", subject_summary, ["subject_id", "age", "sex", "window_count", "recording_count", "conditions"])
    write_csv(root / "input_recording_summary.csv", recording_summary, ["subject_id", "recording_id", "condition", "session_id", "acquisition", "source_path", "age", "window_count", "start_min", "start_max"])
    write_csv(root / "input_age_distribution.csv", age_distribution, ["age", "subject_count"])
    write_csv(root / "input_issues.csv", input_issues, ["severity", "stage", "issue_type", "sample_id", "subject_id", "detail"])


def finalize_reports(
    root: Path,
    summary: dict[str, Any],
    provenance: dict[str, Any],
    resolved_config: dict[str, Any],
    adaptation_issues: Sequence[dict[str, Any]],
) -> None:
    json_dump(root / "adapter_summary.json", summary)
    json_dump(root / "processing_provenance.json", provenance)
    json_dump(root / "resolved_config.json", resolved_config)
    write_csv(root / "adaptation_issues.csv", adaptation_issues, ["severity", "stage", "issue_type", "sample_id", "subject_id", "detail"])
    input_summary = summary.get("input", {})
    mapping = summary.get("channel_mapping", {})
    validation = summary.get("token_validation", {})
    selection = summary.get("smoke_selection") or {}
    report = [
        "# Dortmund → 50M FBMAE adapter report", "",
        f"- adapter_ready: {summary.get('adapter_ready')}",
        f"- run_success: {summary.get('run_success')}",
        f"- cohort_id: `{summary.get('cohort_id')}`",
        f"- configured age range: {summary.get('configured_min_age')}–{summary.get('configured_max_age')}",
        f"- expected subjects/recordings: {summary.get('expected_subject_count')}/{summary.get('expected_recording_count')}",
        f"- actual input: `{provenance.get('actual_input_local_path', 'not_resolved')}`",
        f"- source URI: `{provenance.get('source_uri')}`",
        f"- input data fingerprint: `{provenance.get('input_data_fingerprint_sha256')}` ({provenance.get('input_data_fingerprint_scope')})",
        f"- input audit: {summary['stages']['input_audit']}",
        f"- channel semantic audit: {summary['stages']['channel_semantic_audit']}",
        f"- token smoke: {summary['stages']['token_smoke']}",
        f"- full conversion: {summary['stages']['full_conversion']}",
        f"- model forward smoke: {summary['stages']['model_forward_smoke']}", "",
        "## Input audit", "",
        f"rows={input_summary.get('row_count', 'not_available')}; subjects={input_summary.get('subject_count', 'not_available')}; recordings={input_summary.get('recording_count', 'not_available')}; schema_matches={input_summary.get('schema_matches', 'not_available')}; source_data_unchanged={summary.get('source_data_unchanged', False)}.",
        f"Smoke subjects: {selection.get('subjects', 'not_performed')}; selected rows: {selection.get('rows', 'not_performed')}.", "",
        "## Fixed reference policy", "",
        "source_reference=FCz; ground=AFz; rereference_applied=false; processing_reference=unchanged; reference_channel_present_in_data=false.", "",
        "## Channel semantics", "",
        f"Authoritative Dortmund vocabulary mapping complete={mapping.get('channel_mapping_complete')}; unknown={mapping.get('unknown_channel_count')}; collisions={mapping.get('mapping_collision_count')}; fixed FBMAE slots covered={mapping.get('fbmae_standard_slots_covered')}.",
        "The current public channel_config.py uses Iz/F9/F10. The fixed 64-slot name scatter covers 58 Dortmund channels; Dortmund-only TP9/TP10/PO9/PO10/FT9/FT10 retain source IDs but are not remapped to unrelated slots. Missing Fpz/AFz/FCz/Iz/F9/F10 model slots remain invalid and zero.", "",
        "## Token output", "",
        f"rows={validation.get('rows', 'not_performed')}; token_inputs=[640,100]; targets=[640,5,5]; loader batch={validation.get('loader_batch_shape', 'not_performed')}; sample_id parity={validation.get('sample_id_parity', 'not_performed')}; mask propagation={validation.get('mask_propagation', 'not_performed')}.", "",
        "## Non-actions", "",
        "No rereferencing, filtering, clip/divide normalization, split, downstream training, backbone feature cache, pooling, or checkpoint loading is performed. Full conversion is not performed in smoke mode.", "",
    ]
    (root / "adapter_report.md").write_text("\n".join(report), encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--input-lance", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--channel-profile", type=Path, default=root / "configs" / "dortmund_fbmae_channel_profile.toml")
    parser.add_argument("--electrode-vocab", type=Path, default=root / "configs" / "fbmae_electrode_vocab_snapshot.json")
    parser.add_argument("--source-uri", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-subjects", nargs="*", default=[])
    parser.add_argument("--smoke-max-windows-per-recording", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--min-age", type=int, default=20)
    parser.add_argument("--max-age", type=int, default=35)
    parser.add_argument("--cohort-id", default=None)
    parser.add_argument("--expected-subjects", type=int, default=None)
    parser.add_argument("--expected-recordings", type=int, default=None)
    args = parser.parse_args(argv)
    if args.min_age < 0 or args.max_age < args.min_age:
        parser.error("require 0 <= --min-age <= --max-age")
    try:
        args.cohort_id = build_cohort_id(args.min_age, args.max_age, args.cohort_id)
    except ValueError as exc:
        parser.error(str(exc))
    default_cohort = args.min_age == 20 and args.max_age == 35
    if args.expected_subjects is None and default_cohort:
        args.expected_subjects = PROFILE_A_SUBJECTS
    if args.expected_recordings is None and default_cohort:
        args.expected_recordings = PROFILE_A_RECORDINGS
    if args.expected_subjects is None:
        parser.error("--expected-subjects is required for a non-default age range")
    if args.expected_recordings is None:
        parser.error("--expected-recordings is required for a non-default age range")
    if args.expected_subjects <= 0:
        parser.error("--expected-subjects must be >0")
    if args.expected_recordings <= 0:
        parser.error("--expected-recordings must be >0")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    if args.input_lance is None and args.input_dir is None:
        raise ValueError("Pass --input-lance or --input-dir")
    input_dir = (args.input_dir or args.input_lance.parent).resolve()
    if not input_dir.exists():
        raise FileNotFoundError(input_dir)
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    output_resolved = args.output_dir.resolve()
    if output_resolved == input_dir or input_dir in output_resolved.parents:
        raise ValueError("Output directory cannot be the input directory or its child")
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("Invalid batch-size/num-workers")

    temp = args.output_dir.parent / f".{args.output_dir.name}.tmp-{uuid.uuid4().hex}"
    temp.mkdir(parents=True, exist_ok=False)
    errors: list[str] = []
    adaptation_issues: list[dict[str, Any]] = []
    input_lance: Path | None = None
    source_before: dict[str, Any] = {}
    source_after: dict[str, Any] = {}
    summary: dict[str, Any] = {"adapter_ready": False, "smoke_passed": False, "run_success": False, "blocked_by_channel_mapping": False, "cohort_id": args.cohort_id, "configured_min_age": args.min_age, "configured_max_age": args.max_age, "expected_subject_count": args.expected_subjects, "expected_recording_count": args.expected_recordings, "stages": {"input_audit": "not_performed", "channel_semantic_audit": "not_performed", "token_smoke": "not_performed", "full_conversion": "not_performed", "model_forward_smoke": "not_performed", "downstream_training": "not_performed"}}
    resolved_config = vars(args).copy()
    resolved_config = {key: str(value) if isinstance(value, Path) else value for key, value in resolved_config.items()}
    provenance: dict[str, Any] = {
        "version": VERSION, "created_at": now_iso(), "source_uri": args.source_uri,
        "cohort_id": args.cohort_id, "min_age": args.min_age, "max_age": args.max_age,
        "expected_subjects": args.expected_subjects, "expected_recordings": args.expected_recordings,
        "git": git_metadata(project_root), "rereference_applied": False, "processing_reference": "unchanged",
        "source_reference": "FCz", "ground": "AFz", "reference_channel_present_in_data": False,
        "full_conversion_performed": False, "model_forward_smoke": "not_performed",
        "critical_code_sha256": {},
    }
    for relative in (
        "scripts/adapt_dortmund_fbmae.py", "scripts/build_huada_model_tokens.py",
        "scripts/preprocess_dortmund_age_windows.py", "scripts/downstream_50m/lance_dataset.py",
        "scripts/downstream_50m/model.py", "configs/dortmund_fbmae_channel_profile.toml",
        "configs/fbmae_electrode_vocab_snapshot.json",
    ):
        path = project_root / relative
        provenance["critical_code_sha256"][relative] = sha256_file(path)

    try:
        print("[stage 1] inspect input", flush=True)
        if args.input_lance is not None:
            input_lance = args.input_lance.resolve()
            if not input_lance.exists():
                raise FileNotFoundError(input_lance)
            inspected = [{"path": str(input_lance), "explicit": True}]
        else:
            input_lance, inspected = discover_input_lance(input_dir)
        provenance["actual_input_local_path"] = str(input_lance)
        source_before = snapshot_tree(input_dir)
        json_dump(temp / "source_snapshot_before.json", source_before)
        dataset = lance.dataset(input_lance)
        metadata_columns = [name for name in dataset.schema.names if name != "data"]
        metadata_preview = dataset.to_table(columns=metadata_columns).to_pylist()
        smoke_indices = None
        smoke_selection = None
        if args.smoke:
            smoke_indices, smoke_selection = choose_smoke_indices(
                metadata_preview, parse_smoke_subjects(args.smoke_subjects),
                args.smoke_max_windows_per_recording, args.min_age, args.max_age
            )
        audit, metadata, subject_rows, recording_rows, ages, subject_sidecar, input_issues = audit_input(
            dataset, input_dir, inspected, smoke_indices,
            args.expected_subjects, args.expected_recordings,
            args.min_age, args.max_age,
        )
        write_stage_outputs(temp, input_audit=audit, schema=dataset.schema, subject_summary=subject_rows, recording_summary=recording_rows, age_distribution=ages, input_issues=input_issues)
        adaptation_issues.extend(input_issues)
        summary["input"] = audit
        summary["smoke_selection"] = smoke_selection
        summary["stages"]["input_audit"] = audit["status"]
        if audit["hard_error_count"]:
            raise RuntimeError(f"Input audit has {audit['hard_error_count']} hard errors")

        print("[stage 2] validate channel mapping", flush=True)
        resolved_profile, mapping_rows, mapping_summary, model_channels = load_channel_profile(args.channel_profile.resolve(), args.electrode_vocab.resolve())
        actual_channel_sets = {tuple(row["channel_names"]) for row in metadata}
        if len(actual_channel_sets) != 1:
            raise RuntimeError(f"Input has multiple channel layouts: {len(actual_channel_sets)}")
        actual_names = list(next(iter(actual_channel_sets)))
        expected_names = resolved_profile["source"]["channel_names"]
        if [token_core.canonical_channel_name(value) for value in actual_names] != [token_core.canonical_channel_name(value) for value in expected_names]:
            raise RuntimeError("Input channel order/names differ from resolved Dortmund channel profile")
        write_csv(temp / "channel_mapping.csv", mapping_rows, ["source_index", "source_channel_name", "canonical_channel_name", "electrode_id", "mapping_source", "mapping_status", "notes", "fbmae_standard_slot"])
        json_dump(temp / "channel_mapping_summary.json", mapping_summary)
        vocab_raw = json.loads(args.electrode_vocab.read_text(encoding="utf-8"))
        json_dump(temp / "electrode_vocabulary_snapshot.json", {"source_path": str(args.electrode_vocab.resolve()), "sha256": sha256_file(args.electrode_vocab), "entry_count": len(vocab_raw), "entries": vocab_raw})
        json_dump(temp / "channel_profile_resolved.json", resolved_profile)
        summary["channel_mapping"] = mapping_summary
        summary["stages"]["channel_semantic_audit"] = "passed"

        print("[stage 3] build token rows", flush=True)
        selected_indices = list(range(len(metadata))) if smoke_indices is None else list(smoke_indices)
        selected_expected_rows = [metadata[index] for index in selected_indices]
        source_electrode_ids = resolved_profile["resolved_source_electrode_ids"]
        model_electrode_ids = resolved_profile["resolved_model_electrode_ids"]
        output_lance = temp / "dortmund_fbmae_tokens.lance"
        inventory: list[dict[str, Any]] = []
        valid_channel_counts: Counter[int] = Counter()
        valid_token_counts: Counter[int] = Counter()
        output_ages: list[float] = []
        first_source_row: dict[str, Any] | None = None
        first_token_row: dict[str, Any] | None = None
        first_write = True
        started = time.monotonic()
        total = len(selected_indices)
        executor = ThreadPoolExecutor(max_workers=args.num_workers) if args.num_workers else None
        try:
            for offset in range(0, total, args.batch_size):
                batch_indices = selected_indices[offset : offset + args.batch_size]
                batch_rows = dataset.take(indices=batch_indices).to_pylist()

                def convert(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
                    sid = str(row["subject_id"])
                    sidecar = subject_sidecar.get(sid, {})
                    return build_token_row(
                        row, model_channels, source_electrode_ids, model_electrode_ids, sidecar.get("sex", "")
                    )

                converted = list(executor.map(convert, batch_rows)) if executor else [convert(row) for row in batch_rows]
                batch_token_rows = [item[0] for item in converted]
                batch_audits = [item[1] for item in converted]
                if first_source_row is None:
                    first_source_row, first_token_row = batch_rows[0], batch_token_rows[0]
                lance.write_dataset(
                    rows_to_table(batch_token_rows), output_lance,
                    mode="create" if first_write else "append",
                    max_rows_per_group=min(32, args.batch_size), max_rows_per_file=256,
                    data_storage_version="stable",
                )
                first_write = False
                for row, row_audit in zip(batch_token_rows, batch_audits):
                    valid_channels = int(row["valid_channel_count"])
                    valid_tokens = int(np.asarray(row["token_valid_mask"]).sum())
                    valid_channel_counts[valid_channels] += 1
                    valid_token_counts[valid_tokens] += 1
                    output_ages.append(float(row["age"]))
                    inventory.append({
                        "sample_id": row["sample_id"], "subject_id": row["subject_id"], "recording_id": row["recording_id"],
                        "eyes_condition": row["eyes_condition"], "age": float(row["age"]), "label": float(row["label"]),
                        "valid_channel_count": valid_channels, "valid_token_count": valid_tokens,
                        "source_bad_channel_count": int(np.asarray(row["source_bad_channel_mask"]).sum()),
                        "unknown_channels": ";".join(row_audit["unknown_channels"]),
                        "missing_model_channels": ";".join(row_audit["missing_channels"]),
                    })
                done = min(offset + len(batch_rows), total)
                elapsed = max(time.monotonic() - started, 1e-9)
                rate = done / elapsed
                remaining = (total - done) / rate if rate else math.inf
                current = batch_rows[-1]
                print(
                    f"processed {done} / {total} rows ({done / total * 100:.1f}%) "
                    f"elapsed={elapsed:.1f}s rows/s={rate:.2f} eta={remaining:.1f}s "
                    f"current_subject={current['subject_id']} current_recording={current['recording_id']}",
                    flush=True,
                )
        finally:
            if executor:
                executor.shutdown(wait=True, cancel_futures=True)

        if first_source_row is None or first_token_row is None:
            raise RuntimeError("No rows selected for token conversion")
        # Determinism: repeat one row through exactly the shared tokenizer.
        repeated, _ = build_token_row(
            first_source_row, model_channels, source_electrode_ids, model_electrode_ids,
            subject_sidecar.get(str(first_source_row["subject_id"]), {}).get("sex", ""),
        )
        if not np.array_equal(repeated["token_inputs"], first_token_row["token_inputs"]) or not np.array_equal(repeated["targets"], first_token_row["targets"]):
            raise AssertionError("Token determinism failed")

        print("[stage 4] validate output", flush=True)
        validation = validate_token_dataset(output_lance, selected_expected_rows)
        summary["token_validation"] = validation
        summary["stages"]["token_smoke" if args.smoke else "full_conversion"] = "passed"
        json_dump(temp / "token_schema.json", {"schema": str(token_schema()), "fields": [{"name": field.name, "type": str(field.type), "nullable": field.nullable} for field in token_schema()]})
        json_dump(temp / "token_statistics.json", {"row_count": total, "token_inputs_shape": [640, 100], "targets_shape": [640, 5, 5], "all_finite": validation["all_finite"], "determinism": "passed", "batch_size": args.batch_size, "num_workers": args.num_workers, "write_mode": "streaming_create_then_append"})
        json_dump(temp / "mask_statistics.json", {"valid_channel_count_distribution": dict(valid_channel_counts), "valid_token_count_distribution": dict(valid_token_counts), "mask_propagation": "passed", "invalid_tokens_zero": True})
        json_dump(temp / "label_statistics.json", {"task_type": "regression", "target_name": "age", "target_unit": "year", "row_count": total, "age_min": min(output_ages), "age_max": max(output_ages), "age_label_parity": "passed", "age_standardized": False})
        write_csv(temp / "sample_mapping_inventory.csv", inventory, ["sample_id", "subject_id", "recording_id", "eyes_condition", "age", "label", "valid_channel_count", "valid_token_count", "source_bad_channel_count", "unknown_channels", "missing_model_channels"])
        summary["adapter_ready"] = bool(
            mapping_summary["channel_mapping_complete"] and validation["sample_id_parity"] == "passed"
            and validation["metadata_parity"] == "passed" and validation["mask_propagation"]
            and validation["lance_roundtrip"] == "passed" and audit["status"] == "passed"
        )
        summary["smoke_passed"] = bool(args.smoke)
        summary["run_success"] = True
        provenance["full_conversion_performed"] = not args.smoke

        source_after = snapshot_tree(input_dir)
        json_dump(temp / "source_snapshot_after.json", source_after)
        summary["source_data_unchanged"] = source_before == source_after
        if not summary["source_data_unchanged"]:
            raise RuntimeError("Input snapshot changed during adaptation")
        summary["acceptance"] = {
            "input_common_dataset_audit": audit["status"],
            "profile_a_validation": "passed" if audit["hard_error_count"] == 0 else "failed",
            "source_data_unchanged": True,
            "channel_mapping_complete": mapping_summary["channel_mapping_complete"],
            "unknown_channel_count": mapping_summary["unknown_channel_count"],
            "mapping_collision_count": mapping_summary["mapping_collision_count"],
            "positional_huada_mapping_used": False,
            "reference_unchanged": True,
            "status_channel_excluded": audit["status_channel_excluded"],
            "sample_id_parity": validation["sample_id_parity"],
            "metadata_parity": validation["metadata_parity"],
            "age_label_parity": validation["age_label_parity"],
            "mask_propagation": "passed" if validation["mask_propagation"] else "failed",
            "token_schema_validation": "passed",
            "token_determinism": "passed",
            "lance_roundtrip": validation["lance_roundtrip"],
            "huada_regression_tests": "external_repository_test_gate",
            "local_or_cfs_smoke": "passed" if args.smoke else "not_performed",
            "cloud_full_conversion": "not_performed" if args.smoke else "passed",
            "downstream_training": "not_performed",
        }
        (temp / "_SUCCESS").write_text(now_iso() + "\n", encoding="utf-8")
    except BaseException as exc:
        errors.append(traceback.format_exc())
        summary["adapter_ready"] = False
        summary["run_success"] = False
        summary["error"] = str(exc)
        adaptation_issues.append({"severity": "error", "stage": "adapter", "issue_type": type(exc).__name__, "sample_id": "", "subject_id": "", "detail": str(exc)})
        source_after = snapshot_tree(input_dir)
        json_dump(temp / "source_snapshot_after.json", source_after)
    finally:
        print("[stage 5] write report", flush=True)
        provenance["input_fingerprint"] = source_before.get("fingerprint")
        provenance["input_fingerprint_kind"] = "sha256 of sorted relative path, byte size, and mtime_ns inventory"
        provenance["input_data_fingerprint_sha256"] = summary.get("input", {}).get("input_data_fingerprint_sha256")
        provenance["input_data_fingerprint_scope"] = summary.get("input", {}).get("input_data_fingerprint_scope", "not_performed")
        provenance["source_data_unchanged"] = bool(source_before and source_before == source_after)
        provenance["actual_output_local_path"] = str(args.output_dir.resolve())
        (temp / "adaptation_errors.log").write_text("\n".join(errors), encoding="utf-8")
        # Ensure not-performed output descriptors exist even on fail-closed paths.
        for name, value in (
            ("token_schema.json", {"status": "not_performed"}),
            ("token_statistics.json", {"status": "not_performed"}),
            ("mask_statistics.json", {"status": "not_performed"}),
            ("label_statistics.json", {"status": "not_performed"}),
        ):
            if not (temp / name).exists():
                json_dump(temp / name, value)
        if not (temp / "input_dataset_audit.json").exists():
            json_dump(temp / "input_dataset_audit.json", {"status": "failed_before_audit", "error": summary.get("error")})
        if not (temp / "input_schema.json").exists():
            json_dump(temp / "input_schema.json", {"status": "not_available"})
        for name, fields in (
            ("input_subject_summary.csv", ["subject_id", "age", "sex", "window_count", "recording_count", "conditions"]),
            ("input_recording_summary.csv", ["subject_id", "recording_id", "condition", "session_id", "acquisition", "source_path", "age", "window_count", "start_min", "start_max"]),
            ("input_age_distribution.csv", ["age", "subject_count"]),
            ("input_issues.csv", ["severity", "stage", "issue_type", "sample_id", "subject_id", "detail"]),
        ):
            if not (temp / name).exists():
                write_csv(temp / name, [], fields)
        if not (temp / "source_snapshot_before.json").exists():
            json_dump(temp / "source_snapshot_before.json", source_before or {"status": "not_performed"})
        if not (temp / "source_snapshot_after.json").exists():
            json_dump(temp / "source_snapshot_after.json", source_after or {"status": "not_performed"})
        if not (temp / "sample_mapping_inventory.csv").exists():
            write_csv(temp / "sample_mapping_inventory.csv", [], ["sample_id", "subject_id", "recording_id", "eyes_condition", "age", "label", "valid_channel_count", "valid_token_count", "source_bad_channel_count", "unknown_channels", "missing_model_channels"])
        if not (temp / "channel_mapping.csv").exists():
            write_csv(temp / "channel_mapping.csv", [], ["source_index", "source_channel_name", "canonical_channel_name", "electrode_id", "mapping_source", "mapping_status", "notes", "fbmae_standard_slot"])
        for name in ("channel_mapping_summary.json", "electrode_vocabulary_snapshot.json", "channel_profile_resolved.json"):
            if not (temp / name).exists():
                json_dump(temp / name, {"status": "not_performed"})
        finalize_reports(temp, summary, provenance, resolved_config, adaptation_issues)
        missing_outputs = REQUIRED_OUTPUTS - {path.name for path in temp.iterdir()}
        if missing_outputs:
            raise AssertionError(f"Missing required report outputs: {sorted(missing_outputs)}")
        os.replace(temp, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if summary.get("run_success") else 2


if __name__ == "__main__":
    raise SystemExit(main())
