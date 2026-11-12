#!/usr/bin/env python3
"""Dortmund Vital Study Profile-A age preprocessing adapter.

This module owns only Dortmund/BIDS selection, age semantics, EDF adaptation,
and supervised base-profile metadata. Signal processing, hard window QC,
bad-channel masking, normalization, smoke limiting, source snapshots, and
Lance batch writing are imported from the validated Huada implementation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import lance
import mne
import numpy as np
import pyarrow as pa
import scipy

from preprocess_huada_common_windows import (
    FILTER_PARAMETERS,
    READER_OUTPUT_UNIT,
    READER_TO_UV_MULTIPLIER,
    _config_hash,
    _git_metadata,
    _prepare_output_dir,
    _snapshot_raw_files,
    _write_csv,
    _write_lance_batch,
    apply_common_signal_processing,
    convert_mne_v_to_uv,
    dropped_tail,
    evaluate_window_qc,
    identify_window_bad_channels,
    limit_window_slices,
    merge_bad_channel_masks,
    normalize_clip_divide_200,
    now_iso,
    parse_smoke_subjects,
    window_slices,
    zero_bad_channels,
)
from validate_huada_raw_bdf import load_and_validate_config, unit_to_uv_multiplier


SCRIPT_VERSION = "1.1.0"
DEFAULT_SOURCE_URI = "cos://jianyu-1442740494/huada/data/Dortmund_Vital_Study"
SOURCE_SFREQ = 1000.0
N_EEG_CHANNELS = 64
STATUS_NAME = "Status"
SOURCE_REFERENCE = "FCz"
SOURCE_GROUND = "AFz"
SOURCE_MONTAGE = "10-20 / source-declared placement"
PROFILE_A_SUBJECTS = 214
PROFILE_A_RECORDINGS = 428

DORTMUND_CHANNELS = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2", "FC6",
    "T7", "C3", "Cz", "C4", "T8", "TP9", "CP5", "CP1", "CP2", "CP6", "TP10",
    "P7", "P3", "Pz", "P4", "P8", "PO9", "O1", "Oz", "O2", "PO10", "AF7", "AF3",
    "AF4", "AF8", "F5", "F1", "F2", "F6", "FT9", "FT7", "FC3", "FC4", "FT8",
    "FT10", "C5", "C1", "C2", "C6", "TP7", "CP3", "CPz", "CP4", "TP8", "P5",
    "P1", "P2", "P6", "PO7", "PO3", "POz", "PO4", "PO8",
]

BIDS_RE = re.compile(
    r"^(?P<subject>sub-[^_]+)_(?P<session>ses-[^_]+)_task-(?P<task>[^_]+)_acq-(?P<acquisition>[^_]+)(?:_run-(?P<run>[^_]+))?_eeg\.edf$",
    re.IGNORECASE,
)

CSV_FIELDS: dict[str, list[str]] = {
    "subject_manifest.csv": ["subject_id", "age", "label", "age_dtype", "age_unit", "age_scope", "age_status", "sex", "handedness", "selected_profile", "recording_count", "eligible"],
    "recording_manifest.csv": ["subject_id", "session_id", "task", "acquisition", "eyes_condition", "run_id", "recording_id", "source_path", "sidecar_path", "channels_path", "age", "label", "selected_reason"],
    "excluded_recordings.csv": ["subject_id", "session_id", "recording_id", "source_path", "entity_type", "entity_name", "exclusion_reason"],
    "recording_validation.csv": ["subject_id", "recording_id", "source_path", "header_read_success", "fully_readable", "sampling_rate_hz", "n_channels_total", "n_eeg_channels", "status_channel_count", "n_samples_header", "n_samples_read", "duration_edf_seconds", "duration_sidecar_seconds", "duration_matches", "channel_order_matches", "unit_conversion_verified", "has_nan", "has_inf", "recording_bad_channel_count", "near_zero_variance_channel_count", "physical_limit_hit_count", "dc_offset_channel_count", "extreme_amplitude_channel_count", "validation_pass", "error"],
    "recording_processing_summary.csv": ["subject_id", "recording_id", "source_path", "eyes_condition", "run_mode", "source_samples", "source_duration_seconds", "resampled_samples", "theoretical_complete_windows", "emitted_windows", "written_windows", "smoke_truncated_windows", "real_dropped_tail_samples", "real_dropped_tail_seconds", "recording_bad_channels", "qc_pass_windows", "qc_fail_windows", "mean_clip_fraction", "max_clip_fraction", "processing_success", "error"],
    "channel_header_inventory.csv": ["subject_id", "recording_id", "source_path", "channel_index", "channel_name", "channel_type", "physical_dimension", "physical_min", "physical_max", "digital_min", "digital_max", "raw_min_uv", "raw_max_uv", "raw_mean_uv", "raw_std_uv", "post_filter_min_uv", "post_filter_max_uv", "post_filter_mean_uv", "post_filter_std_uv", "physical_limit_hit_count", "near_zero_variance_flag", "dc_offset_flag", "extreme_amplitude_flag", "is_bad", "bad_reason"],
    "bad_channel_inventory.csv": ["subject_id", "recording_id", "source_path", "sample_id", "channel_index", "channel_name", "scope", "bad_reason", "affected_windows", "data_zeroed", "mask_verified"],
    "window_inventory.csv": ["sample_id", "subject_id", "session_id", "recording_id", "eyes_condition", "source_path", "window_index", "start_time", "duration", "start_sample_source", "start_sample_target", "shape", "dtype", "age", "label", "qc_pass", "qc_failure_reason", "clip_fraction", "positive_clip_fraction", "negative_clip_fraction", "bad_channel_count", "bad_channel_names", "written_to_lance"],
    "dropped_tails.csv": ["subject_id", "recording_id", "source_path", "source_samples", "resampled_samples", "theoretical_complete_windows", "emitted_windows", "smoke_truncated_windows", "real_dropped_tail_samples", "real_dropped_tail_seconds"],
    "preprocessing_issues.csv": ["severity", "stage", "subject_id", "recording_id", "source_path", "sample_id", "channel_name", "issue_type", "description", "action"],
    "source_snapshot_before.csv": ["relative_path", "size_bytes", "mtime_ns"],
    "source_snapshot_after.csv": ["relative_path", "size_bytes", "mtime_ns"],
}

INVENTORY_CSV_FIELDS: dict[str, list[str]] = {
    "candidate_subjects.csv": [
        "subject_id", "age", "sex", "handedness", "profile_recording_count",
        "eyes_open_count", "eyes_closed_count", "eligible", "exclusion_reason",
    ],
    "eligible_subjects.csv": [
        "subject_id", "age", "sex", "handedness", "profile_recording_count",
        "eyes_open_count", "eyes_closed_count",
    ],
    "eligible_recordings.csv": [
        "subject_id", "session_id", "task", "acquisition", "eyes_condition",
        "run_id", "recording_id", "source_path", "sidecar_path", "channels_path",
        "age", "label",
    ],
    "excluded_subjects.csv": [
        "subject_id", "age", "sex", "handedness", "exclusion_reason",
    ],
}


def parse_bids_edf(path: str | Path) -> dict[str, str]:
    match = BIDS_RE.match(Path(path).name)
    if not match:
        return {}
    return {key: (value or "") for key, value in match.groupdict().items()}


def resolve_dataset_root(data_dir: Path) -> Path:
    data_dir = data_dir.resolve()
    if any(data_dir.glob("sub-*")):
        return data_dir
    candidates = [p for p in data_dir.iterdir() if p.is_dir() and any(p.glob("sub-*"))]
    if len(candidates) != 1:
        raise ValueError(f"Could not uniquely resolve BIDS subject root below {data_dir}: {candidates}")
    return candidates[0]


def find_participants_file(data_dir: Path, dataset_root: Path) -> Path:
    candidates = [
        data_dir / "participants.tsv", data_dir / "participants.txt",
        dataset_root / "participants.tsv", dataset_root / "participants.txt",
        dataset_root.parent / "participants.tsv", dataset_root.parent / "participants.txt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("participants metadata not found")


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def parse_individual_numeric_age(value: Any) -> int:
    text = str(value).strip()
    if not re.fullmatch(r"\d+(?:\.0+)?", text):
        raise ValueError(f"Age must be an unbinned integer-year value, got {value!r}")
    return int(float(text))


def parse_numeric_age(value: Any, min_age: int = 20, max_age: int = 35) -> int:
    """Parse a documented integer-year age and enforce the configured cohort."""
    age = parse_individual_numeric_age(value)
    if not min_age <= age <= max_age:
        raise ValueError(f"Age outside configured range {min_age}-{max_age}: {age}")
    return age


def build_cohort_id(min_age: int, max_age: int, cohort_id: str | None = None) -> str:
    value = cohort_id or f"age{min_age}_{max_age}_ses1_pre_profile_a"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise ValueError(
            "cohort_id must contain only letters, digits, dot, underscore, or hyphen"
        )
    return value


def build_dataset_id(cohort_id: str) -> str:
    return f"ds005385_dortmund_{cohort_id}"


def validate_full_read(actual_samples: int, header_samples: int, has_nan: bool, has_inf: bool) -> None:
    if actual_samples != header_samples:
        raise RuntimeError(f"Actual samples {actual_samples} != EDF header {header_samples}")
    if has_nan or has_inf:
        raise RuntimeError("NaN/Inf detected during full recording read")


def resolve_duration(header_samples: int, sfreq: float, sidecar_duration: float) -> tuple[float, bool]:
    actual = header_samples / float(sfreq)
    return actual, math.isclose(actual, float(sidecar_duration), abs_tol=1.1)


def resolve_dortmund_channel_indices(
    channel_names: Sequence[str], channel_types: Sequence[str]
) -> tuple[list[int], list[int]]:
    """Resolve native EEG and Status slots without trusting Status type inference."""
    if len(channel_names) != len(channel_types):
        raise RuntimeError(
            f"Channel name/type length mismatch: {len(channel_names)} != {len(channel_types)}"
        )
    status_indices = [
        idx for idx, name in enumerate(channel_names)
        if name.casefold() == STATUS_NAME.casefold()
    ]
    eeg_indices = [
        idx for idx, channel_type in enumerate(channel_types)
        if channel_type == "eeg" and idx not in status_indices
    ]
    if len(channel_names) != 65 or len(eeg_indices) != 64 or len(status_indices) != 1:
        typed_names = list(zip(channel_names, channel_types))
        raise RuntimeError(
            "Expected 64 EEG + one exact-name Status channel, got "
            f"names={len(channel_names)} eeg_excluding_status={len(eeg_indices)} "
            f"status={len(status_indices)} typed_names={typed_names!r}"
        )
    return eeg_indices, status_indices


def read_edf_signal_header(path: Path) -> list[dict[str, Any]]:
    """Read the EDF fixed/signal header without relying on MNE private extras."""

    def decode_field(value: bytes) -> str:
        return value.decode("latin-1").strip()

    with path.open("rb") as handle:
        fixed = handle.read(256)
        if len(fixed) != 256:
            raise RuntimeError(f"Truncated EDF fixed header: expected 256 bytes, got {len(fixed)}")
        try:
            header_bytes = int(decode_field(fixed[184:192]))
            signal_count = int(decode_field(fixed[252:256]))
        except ValueError as exc:
            raise RuntimeError("Invalid EDF fixed-header numeric fields") from exc
        expected_header_bytes = 256 + signal_count * 256
        if signal_count <= 0 or header_bytes != expected_header_bytes:
            raise RuntimeError(
                f"EDF header size mismatch: declared={header_bytes} expected={expected_header_bytes} "
                f"signals={signal_count}"
            )
        signal_block = handle.read(header_bytes - 256)
        if len(signal_block) != header_bytes - 256:
            raise RuntimeError(
                f"Truncated EDF signal header: expected {header_bytes - 256} bytes, got {len(signal_block)}"
            )

    offset = 0

    def take(width: int) -> list[str]:
        nonlocal offset
        size = width * signal_count
        block = signal_block[offset : offset + size]
        offset += size
        return [decode_field(block[index * width : (index + 1) * width]) for index in range(signal_count)]

    labels = take(16)
    transducers = take(80)
    physical_dimensions = take(8)
    physical_mins = take(8)
    physical_maxs = take(8)
    digital_mins = take(8)
    digital_maxs = take(8)
    prefilters = take(80)
    samples_per_record = take(8)
    reserved = take(32)
    if offset != len(signal_block):
        raise RuntimeError(f"EDF signal-header parse mismatch: consumed={offset} available={len(signal_block)}")

    rows: list[dict[str, Any]] = []
    for index in range(signal_count):
        try:
            row = {
                "channel_index": index,
                "label": labels[index],
                "transducer": transducers[index],
                "physical_dimension": physical_dimensions[index],
                "physical_min": float(physical_mins[index]),
                "physical_max": float(physical_maxs[index]),
                "digital_min": float(digital_mins[index]),
                "digital_max": float(digital_maxs[index]),
                "prefilter": prefilters[index],
                "samples_per_record": int(samples_per_record[index]),
                "reserved": reserved[index],
            }
        except ValueError as exc:
            raise RuntimeError(f"Invalid EDF signal-header numeric field at channel {index}") from exc
        if not row["label"]:
            raise RuntimeError(f"Empty EDF signal label at channel {index}")
        if row["physical_min"] >= row["physical_max"]:
            raise RuntimeError(f"Invalid EDF physical range at channel {index}: {row}")
        if row["digital_min"] >= row["digital_max"]:
            raise RuntimeError(f"Invalid EDF digital range at channel {index}: {row}")
        if row["samples_per_record"] <= 0:
            raise RuntimeError(f"Invalid EDF samples-per-record at channel {index}: {row}")
        rows.append(row)
    return rows


def condition_from_entities(entities: dict[str, str]) -> str:
    task = entities.get("task", "").lower()
    if task == "eyesopen":
        return "eyes_open"
    if task == "eyesclosed":
        return "eyes_closed"
    return "unknown"


def profile_a_decision(entities: dict[str, str], sidecar: dict[str, Any]) -> tuple[bool, str]:
    if entities.get("session", "").lower() != "ses-1":
        return False, "excluded_age_semantics_unsafe"
    if entities.get("acquisition", "").lower() != "pre":
        return False, "excluded_profile_a"
    if entities.get("task", "").lower() not in {"eyesopen", "eyesclosed"}:
        return False, "excluded_condition_unknown"
    if str(sidecar.get("TaskName", "")).strip().lower() != "restingstate":
        return False, "excluded_task_semantics_unconfirmed"
    return True, "selected_profile_a"


def validate_profile_a_counts(recordings: Sequence[dict[str, Any]], expected_subjects: int = PROFILE_A_SUBJECTS, expected_recordings: int = PROFILE_A_RECORDINGS) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in recordings:
        grouped[row["subject_id"]].append(row)
    errors = []
    if len(grouped) != expected_subjects:
        errors.append(f"subjects expected={expected_subjects} actual={len(grouped)}")
    if len(recordings) != expected_recordings:
        errors.append(f"recordings expected={expected_recordings} actual={len(recordings)}")
    for subject_id, rows in sorted(grouped.items()):
        conditions = Counter(row["eyes_condition"] for row in rows)
        if conditions != Counter({"eyes_open": 1, "eyes_closed": 1}):
            errors.append(f"{subject_id}: expected one EO and one EC, got {dict(conditions)}")
    if errors:
        raise ValueError("Profile A count gate failed: " + " | ".join(errors[:30]))


def select_smoke_recordings(recordings: Sequence[dict[str, Any]], smoke: bool, smoke_subjects: Sequence[str]) -> list[dict[str, Any]]:
    if not smoke:
        return list(recordings)
    wanted = {item.lower() for item in smoke_subjects}
    selected = [row for row in recordings if row["subject_id"].lower() in wanted]
    found = {row["subject_id"].lower() for row in selected}
    if found != wanted:
        raise ValueError(f"Smoke subjects missing from Profile A: {sorted(wanted - found)}")
    return selected


def make_dortmund_sample_id(subject_id: str, session_id: str, eyes_condition: str, acquisition: str, recording_id: str, window_index: int, source_start_sample: int) -> str:
    return f"ds005385__{subject_id}__{session_id}__{eyes_condition}__acq-{acquisition}__{recording_id}__w{window_index:05d}__s{source_start_sample:09d}"


def dortmund_lance_schema() -> pa.Schema:
    return pa.schema([
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("subject_id", pa.string(), nullable=False),
        pa.field("session_id", pa.string(), nullable=False),
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("recording_id", pa.string(), nullable=False),
        pa.field("modality", pa.string(), nullable=False),
        pa.field("data", pa.list_(pa.float32(), 128000), nullable=False),
        pa.field("shape", pa.list_(pa.int32()), nullable=False),
        pa.field("original_shape", pa.list_(pa.int32()), nullable=False),
        pa.field("valid_length", pa.int64(), nullable=False),
        pa.field("qc_pass", pa.bool_(), nullable=False),
        pa.field("label", pa.float32(), nullable=False),
        pa.field("age", pa.float32(), nullable=False),
        pa.field("task_type", pa.string(), nullable=False),
        pa.field("target_name", pa.string(), nullable=False),
        pa.field("target_unit", pa.string(), nullable=False),
        pa.field("eyes_condition", pa.string(), nullable=False),
        pa.field("pre_or_post", pa.string(), nullable=False),
        pa.field("source_path", pa.string(), nullable=False),
        pa.field("start_time", pa.float64(), nullable=False),
        pa.field("duration", pa.float64(), nullable=False),
        pa.field("source_start_sample", pa.int64(), nullable=False),
        pa.field("target_start_sample", pa.int64(), nullable=False),
        pa.field("channel_names", pa.list_(pa.string()), nullable=False),
        pa.field("channel_mask", pa.list_(pa.bool_(), 64), nullable=False),
        pa.field("bad_channel_mask", pa.list_(pa.bool_(), 64), nullable=False),
        pa.field("channel_status", pa.list_(pa.string()), nullable=False),
        pa.field("channel_layout", pa.string(), nullable=False),
        pa.field("bad_channel_count", pa.int32(), nullable=False),
        pa.field("clip_fraction", pa.float32(), nullable=False),
        pa.field("positive_clip_fraction", pa.float32(), nullable=False),
        pa.field("negative_clip_fraction", pa.float32(), nullable=False),
        pa.field("pre_normalization_abs_max_uv", pa.float32(), nullable=False),
        pa.field("source_sampling_rate", pa.float32(), nullable=False),
        pa.field("target_sampling_rate", pa.float32(), nullable=False),
        pa.field("preprocessing_config_hash", pa.string(), nullable=False),
    ])


def validate_dortmund_lance(
    path: Path, expected_rows: int, min_age: int = 20, max_age: int = 35
) -> dict[str, Any]:
    dataset = lance.dataset(path)
    schema = dataset.schema
    if schema != dortmund_lance_schema():
        raise AssertionError(f"Lance schema mismatch\nexpected={dortmund_lance_schema()}\nactual={schema}")
    forbidden = {"split", "fold", "electrode_ids", "model_tokens", "prediction", "feature"} & set(schema.names)
    if forbidden:
        raise AssertionError(f"Forbidden fields: {sorted(forbidden)}")
    ids: set[str] = set()
    rows = 0
    all_valid = True
    for batch in dataset.scanner(batch_size=8).to_batches():
        for i in range(batch.num_rows):
            sample_id = batch.column("sample_id")[i].as_py()
            if sample_id in ids:
                raise AssertionError(f"Duplicate sample_id: {sample_id}")
            ids.add(sample_id)
            shape = batch.column("shape")[i].as_py()
            data = np.asarray(batch.column("data")[i].values, dtype=np.float32).reshape(64, 2000)
            channel_mask = np.asarray(batch.column("channel_mask")[i].values, dtype=bool)
            bad = np.asarray(batch.column("bad_channel_mask")[i].values, dtype=bool)
            age = float(batch.column("age")[i].as_py())
            label = float(batch.column("label")[i].as_py())
            valid = shape == [64, 2000] and data.dtype == np.float32 and np.isfinite(data).all() and data.min() >= -1 and data.max() <= 1 and channel_mask.shape == (64,) and channel_mask.all() and bad.shape == (64,) and (not bad.any() or np.all(data[bad] == 0)) and age == label and min_age <= age <= max_age
            all_valid &= bool(valid)
            if not valid:
                raise AssertionError(f"Invalid Lance row {sample_id}")
            rows += 1
    if rows != expected_rows:
        raise AssertionError(f"Expected {expected_rows} Lance rows, got {rows}")
    return {"row_count": rows, "sample_id_unique": len(ids) == rows, "schema_matches": True, "all_rows_valid": all_valid, "forbidden_fields_absent": not forbidden, "schema": str(schema)}


def snapshot_rows(snapshot: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"relative_path": path, **values} for path, values in sorted(snapshot.items())]


def _toml_quote(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


class DortmundPipeline:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.data_dir = args.data_dir.resolve()
        self.dataset_root = resolve_dataset_root(self.data_dir)
        self.output_dir = args.output_dir.resolve()
        self.cohort_id = build_cohort_id(args.min_age, args.max_age, args.cohort_id)
        self.dataset_id = build_dataset_id(self.cohort_id)
        self.config = load_and_validate_config(args.config)
        self.pre = self.config["preprocessing"]
        self.windowing = self.config["windowing"]
        self.config_hash = _config_hash(args.config)
        self.schema = dortmund_lance_schema()
        self.lance_path = self.output_dir / "dataset.lance"
        self.first_lance_write = True
        self.rows = {name: [] for name in CSV_FIELDS}
        self.issues: list[dict[str, Any]] = self.rows["preprocessing_issues.csv"]
        self.errors: list[str] = []
        self.subjects: dict[str, dict[str, Any]] = {}
        self.all_profile_recordings: list[dict[str, Any]] = []
        self.recordings: list[dict[str, Any]] = []
        self.written_rows = 0
        self.selected_paths: list[str] = []
        self.inventory_rows = {name: [] for name in INVENTORY_CSV_FIELDS}

    def issue(self, severity: str, stage: str, issue_type: str, description: str, action: str, rec: dict[str, Any] | None = None, channel: str = "", sample_id: str = "") -> None:
        rec = rec or {}
        self.issues.append({"severity": severity, "stage": stage, "subject_id": rec.get("subject_id", ""), "recording_id": rec.get("recording_id", ""), "source_path": rec.get("source_path", ""), "sample_id": sample_id, "channel_name": channel, "issue_type": issue_type, "description": description, "action": action})

    def build_manifests(self) -> None:
        participants_path = find_participants_file(self.data_dir, self.dataset_root)
        participants = {row["participant_id"]: row for row in read_tsv(participants_path)}
        selected: list[dict[str, Any]] = []
        downloaded_in_scope: dict[str, dict[str, Any]] = {}
        excluded_subjects: dict[str, dict[str, Any]] = {}
        for edf in sorted(self.dataset_root.rglob("*.edf"), key=lambda p: p.as_posix().lower()):
            entities = parse_bids_edf(edf)
            if not entities:
                self.issue("error", "profile_selection", "bids_entities_invalid", f"Cannot parse {edf.name}", "Fix BIDS filename")
                continue
            rel = edf.relative_to(self.data_dir).as_posix()
            sid = entities["subject"]
            recording_id = edf.stem
            if sid not in participants:
                raise ValueError(f"Downloaded subject missing participants metadata: {sid}")
            source = participants[sid]
            try:
                age = parse_individual_numeric_age(source.get("age"))
            except ValueError as exc:
                raise ValueError(f"Invalid age metadata for downloaded subject {sid}: {exc}") from exc
            if not self.args.min_age <= age <= self.args.max_age:
                self.rows["excluded_recordings.csv"].append({"subject_id": sid, "session_id": entities["session"], "recording_id": recording_id, "source_path": rel, "entity_type": "recording", "entity_name": recording_id, "exclusion_reason": "excluded_age_out_of_scope"})
                excluded_subjects[sid] = {
                    "subject_id": sid, "age": age, "sex": source.get("sex", ""),
                    "handedness": source.get("handedness", ""),
                    "exclusion_reason": "excluded_age_out_of_scope",
                }
                continue
            downloaded_in_scope[sid] = source
            stem = edf.name[:-8]
            sidecar_path = edf.with_name(stem + "_eeg.json")
            channels_path = edf.with_name(stem + "_channels.tsv")
            if not sidecar_path.exists() or not channels_path.exists():
                raise FileNotFoundError(f"Missing sidecar/channels file for {edf}")
            sidecar = read_json(sidecar_path)
            is_selected, reason = profile_a_decision(entities, sidecar)
            if not is_selected:
                self.rows["excluded_recordings.csv"].append({"subject_id": sid, "session_id": entities["session"], "recording_id": recording_id, "source_path": rel, "entity_type": "recording", "entity_name": recording_id, "exclusion_reason": reason})
                continue
            condition = condition_from_entities(entities)
            rec = {"subject_id": sid, "session_id": entities["session"], "task": entities["task"], "acquisition": entities["acquisition"], "eyes_condition": condition, "run_id": entities.get("run") or "run-01", "recording_id": recording_id, "source_path": rel, "sidecar_path": sidecar_path.relative_to(self.data_dir).as_posix(), "channels_path": channels_path.relative_to(self.data_dir).as_posix(), "age": age, "label": age, "selected_reason": reason, "absolute_path": edf, "sidecar": sidecar}
            selected.append(rec)
        self.all_profile_recordings = selected
        self.recordings = select_smoke_recordings(selected, self.args.smoke, self.args.smoke_subjects)
        selected_subject_ids = sorted({r["subject_id"] for r in self.recordings})
        for sid in selected_subject_ids:
            source = participants[sid]
            age = parse_numeric_age(source.get("age"), self.args.min_age, self.args.max_age)
            count = sum(r["subject_id"] == sid for r in self.recordings)
            self.subjects[sid] = {"subject_id": sid, "age": age, "label": age, "age_dtype": "integer", "age_unit": "year", "age_scope": "baseline/ses-1", "age_status": "valid_numeric_baseline", "sex": source.get("sex", ""), "handedness": source.get("handedness", ""), "selected_profile": "A", "recording_count": count, "eligible": count == 2}
        self.rows["subject_manifest.csv"] = list(self.subjects.values())
        self.rows["recording_manifest.csv"] = [{key: value for key, value in rec.items() if key not in {"absolute_path", "sidecar"}} for rec in self.recordings]
        for rec in self.recordings:
                self.rows["excluded_recordings.csv"].append({"subject_id": rec["subject_id"], "session_id": rec["session_id"], "recording_id": rec["recording_id"], "source_path": rec["source_path"], "entity_type": "channel", "entity_name": STATUS_NAME, "exclusion_reason": "excluded_non_eeg_channel"})
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in selected:
            grouped[row["subject_id"]].append(row)
        eligible_subject_ids: set[str] = set()
        for sid, source in sorted(downloaded_in_scope.items()):
            rows = grouped.get(sid, [])
            conditions = Counter(row["eyes_condition"] for row in rows)
            eligible = conditions == Counter({"eyes_open": 1, "eyes_closed": 1})
            if eligible:
                eligible_subject_ids.add(sid)
            reason = "" if eligible else (
                "no_profile_a_recording" if not rows else "profile_a_pair_missing_or_duplicate"
            )
            candidate = {
                "subject_id": sid,
                "age": parse_numeric_age(source.get("age"), self.args.min_age, self.args.max_age),
                "sex": source.get("sex", ""),
                "handedness": source.get("handedness", ""),
                "profile_recording_count": len(rows),
                "eyes_open_count": conditions.get("eyes_open", 0),
                "eyes_closed_count": conditions.get("eyes_closed", 0),
                "eligible": eligible,
                "exclusion_reason": reason,
            }
            self.inventory_rows["candidate_subjects.csv"].append(candidate)
            if eligible:
                self.inventory_rows["eligible_subjects.csv"].append(
                    {key: candidate[key] for key in INVENTORY_CSV_FIELDS["eligible_subjects.csv"]}
                )
            else:
                excluded_subjects[sid] = {
                    "subject_id": sid, "age": candidate["age"], "sex": candidate["sex"],
                    "handedness": candidate["handedness"], "exclusion_reason": reason,
                }
        self.inventory_rows["eligible_recordings.csv"] = [
            {key: row[key] for key in INVENTORY_CSV_FIELDS["eligible_recordings.csv"]}
            for row in selected if row["subject_id"] in eligible_subject_ids
        ]
        self.inventory_rows["excluded_subjects.csv"] = list(excluded_subjects.values())
        if not self.args.smoke and not self.args.inventory_only:
            try:
                validate_profile_a_counts(selected, self.args.expected_subjects, self.args.expected_recordings)
            except ValueError as exc:
                self.issue("error", "profile_selection", "profile_a_count_gate_failed", str(exc), "Resolve missing/duplicate Profile-A recordings before signal processing")
                grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for row in selected:
                    grouped[row["subject_id"]].append(row)
                for subject_id, rows in sorted(grouped.items()):
                    conditions = Counter(row["eyes_condition"] for row in rows)
                    if conditions != Counter({"eyes_open": 1, "eyes_closed": 1}):
                        self.issue("error", "profile_selection", "profile_a_subject_pair_invalid", f"Expected one eyes_open and one eyes_closed, got {dict(conditions)}", "Retrieve or deduplicate Profile-A recording", {"subject_id": subject_id})
                for name in ("subject_manifest.csv", "recording_manifest.csv", "excluded_recordings.csv", "preprocessing_issues.csv"):
                    _write_csv(self.output_dir / name, self.rows[name], CSV_FIELDS[name])
                raise
        metadata_paths = {participants_path.relative_to(self.data_dir).as_posix()}
        for rec in self.recordings:
            metadata_paths.update({rec["source_path"], rec["sidecar_path"], rec["channels_path"]})
        self.selected_paths = sorted(metadata_paths)

    def _scan_recording(self, raw: mne.io.BaseRaw, eeg_raw_indices: list[int], rec: dict[str, Any], channel_meta: list[dict[str, str]]) -> tuple[np.ndarray, list[dict[str, Any]], np.ndarray]:
        n_samples = int(raw.n_times)
        n_channels = len(eeg_raw_indices)
        chunk = max(1, int(round(self.args.chunk_seconds * raw.info["sfreq"])))
        mins = np.full(n_channels, np.inf); maxs = np.full(n_channels, -np.inf)
        sums = np.zeros(n_channels); squares = np.zeros(n_channels); counts = np.zeros(n_channels, dtype=np.int64)
        first = np.full(n_channels, np.nan); all_same = np.ones(n_channels, dtype=bool)
        has_nan = np.zeros(n_channels, dtype=bool); has_inf = np.zeros(n_channels, dtype=bool)
        hit_min = np.zeros(n_channels, dtype=np.int64); hit_max = np.zeros(n_channels, dtype=np.int64)
        edf_headers = read_edf_signal_header(Path(rec["absolute_path"]))
        if len(edf_headers) != len(raw.ch_names):
            raise RuntimeError(
                f"EDF signal-header count {len(edf_headers)} != MNE channel count {len(raw.ch_names)}"
            )
        edf_labels = [row["label"] for row in edf_headers]
        if edf_labels != list(raw.ch_names):
            differences = [
                f"{idx}: EDF={expected!r} MNE={actual!r}"
                for idx, (expected, actual) in enumerate(zip(edf_labels, raw.ch_names))
                if expected != actual
            ]
            raise RuntimeError("EDF/MNE channel label mismatch: " + "; ".join(differences[:12]))
        eeg_headers = [edf_headers[index] for index in eeg_raw_indices]
        header_multipliers = [unit_to_uv_multiplier(row["physical_dimension"]) for row in eeg_headers]
        if any(multiplier is None for multiplier in header_multipliers):
            failures = [
                f"{DORTMUND_CHANNELS[idx]}={eeg_headers[idx]['physical_dimension']!r}"
                for idx, multiplier in enumerate(header_multipliers)
                if multiplier is None
            ]
            raise RuntimeError("Unrecognized EDF physical dimension: " + "; ".join(failures))
        physical_min_raw = np.asarray([row["physical_min"] for row in eeg_headers], dtype=float)
        physical_max_raw = np.asarray([row["physical_max"] for row in eeg_headers], dtype=float)
        header_multipliers_array = np.asarray(header_multipliers, dtype=float)
        physical_min_uv = physical_min_raw * header_multipliers_array
        physical_max_uv = physical_max_raw * header_multipliers_array
        digital_min = np.asarray([row["digital_min"] for row in eeg_headers], dtype=float)
        digital_max = np.asarray([row["digital_max"] for row in eeg_headers], dtype=float)
        read_samples = 0
        for start in range(0, n_samples, chunk):
            stop = min(n_samples, start + chunk)
            data_uv = convert_mne_v_to_uv(raw.get_data(picks=eeg_raw_indices, start=start, stop=stop))
            if data_uv.shape != (64, stop - start):
                raise RuntimeError(f"Chunk shape mismatch {data_uv.shape} at {start}:{stop}")
            read_samples += data_uv.shape[1]
            for idx, values in enumerate(data_uv):
                has_nan[idx] |= bool(np.isnan(values).any()); has_inf[idx] |= bool(np.isinf(values).any())
                finite = values[np.isfinite(values)]
                if finite.size == 0: continue
                if counts[idx] == 0: first[idx] = finite[0]
                if not np.all(finite == first[idx]): all_same[idx] = False
                mins[idx] = min(mins[idx], float(finite.min())); maxs[idx] = max(maxs[idx], float(finite.max()))
                sums[idx] += finite.sum(dtype=np.float64); squares[idx] += np.dot(finite, finite); counts[idx] += finite.size
                hit_min[idx] += np.count_nonzero(np.isclose(finite, physical_min_uv[idx], rtol=0, atol=1e-9))
                hit_max[idx] += np.count_nonzero(np.isclose(finite, physical_max_uv[idx], rtol=0, atol=1e-9))
        validate_full_read(read_samples, n_samples, bool(has_nan.any()), bool(has_inf.any()))
        if np.any(counts != n_samples):
            raise RuntimeError("Incomplete finite sample count for one or more channels")
        bad = np.zeros(64, dtype=bool)
        inventory: list[dict[str, Any]] = []
        for idx in range(64):
            mean = sums[idx] / counts[idx]
            variance = max(0.0, squares[idx] / counts[idx] - mean * mean)
            reasons = []
            if all_same[idx]:
                if first[idx] == 0: reasons.append("all_zero")
                reasons.extend(["strict_constant", "exact_zero_variance"])
            bad[idx] = bool(reasons)
            unit = eeg_headers[idx]["physical_dimension"]
            sidecar_unit = channel_meta[idx].get("units", "")
            header_multiplier = unit_to_uv_multiplier(unit)
            sidecar_multiplier = unit_to_uv_multiplier(sidecar_unit)
            if header_multiplier is None or sidecar_multiplier is None or header_multiplier != sidecar_multiplier:
                raise RuntimeError(
                    f"Unverified/conflicting EEG unit for {DORTMUND_CHANNELS[idx]}: "
                    f"EDF={unit!r} channels.tsv={sidecar_unit!r}"
                )
            physical_hits = int(hit_min[idx] + hit_max[idx])
            near_zero = variance <= 1e-12 and not all_same[idx]
            dc = abs(mean) > 1000.0
            extreme = max(abs(mins[idx]), abs(maxs[idx])) > 10000.0
            row = {"subject_id": rec["subject_id"], "recording_id": rec["recording_id"], "source_path": rec["source_path"], "channel_index": idx, "channel_name": DORTMUND_CHANNELS[idx], "channel_type": "EEG", "physical_dimension": unit, "physical_min": physical_min_raw[idx], "physical_max": physical_max_raw[idx], "digital_min": digital_min[idx], "digital_max": digital_max[idx], "raw_min_uv": mins[idx], "raw_max_uv": maxs[idx], "raw_mean_uv": mean, "raw_std_uv": math.sqrt(variance), "post_filter_min_uv": "", "post_filter_max_uv": "", "post_filter_mean_uv": "", "post_filter_std_uv": "", "physical_limit_hit_count": physical_hits, "near_zero_variance_flag": near_zero, "dc_offset_flag": dc, "extreme_amplitude_flag": extreme, "is_bad": bool(bad[idx]), "bad_reason": ";".join(reasons)}
            inventory.append(row)
            if physical_hits:
                self.issue("warning", "raw_signal_qc", "physical_limit_hit", f"{physical_hits} samples equal EDF physical limits; source limits are documented as potentially invalid", "Record only; do not reject on this evidence alone", rec, DORTMUND_CHANNELS[idx])
            if dc:
                self.issue("warning", "raw_signal_qc", "dc_offset", f"Raw mean={mean:.3f} µV", "Record post-filter statistics and clip fraction", rec, DORTMUND_CHANNELS[idx])
            if near_zero:
                self.issue("warning", "raw_signal_qc", "near_zero_variance", f"Raw variance={variance:.12g} µV²", "Review during formal QC; do not interpolate in public preprocessing", rec, DORTMUND_CHANNELS[idx])
            if extreme:
                self.issue("warning", "raw_signal_qc", "extreme_amplitude", f"Raw range=[{mins[idx]:.3f},{maxs[idx]:.3f}] µV", "Record post-filter statistics and clip fraction", rec, DORTMUND_CHANNELS[idx])
            if bad[idx]:
                self.issue("warning", "raw_signal_qc", "recording_bad_channel", ";".join(reasons), "Keep real slot, propagate mask, and zero cached data", rec, DORTMUND_CHANNELS[idx])
        if bad.all():
            raise RuntimeError("All 64 EEG channels are bad")
        return bad, inventory, np.array([read_samples], dtype=np.int64)

    def run_inventory(self) -> dict[str, Any]:
        """Resolve cohort/profile membership without opening any EDF signal."""
        self.build_manifests()
        for name, fields in INVENTORY_CSV_FIELDS.items():
            _write_csv(self.output_dir / name, self.inventory_rows[name], fields)
        eligible_subjects = self.inventory_rows["eligible_subjects.csv"]
        eligible_recordings = self.inventory_rows["eligible_recordings.csv"]
        summary = {
            "run_mode": "inventory_only",
            "inventory_only": True,
            "cohort_id": self.cohort_id,
            "dataset_id": self.dataset_id,
            "min_age": self.args.min_age,
            "max_age": self.args.max_age,
            "candidate_subject_count": len(self.inventory_rows["candidate_subjects.csv"]),
            "eligible_subject_count": len(eligible_subjects),
            "eligible_recording_count": len(eligible_recordings),
            "excluded_subject_count": len(self.inventory_rows["excluded_subjects.csv"]),
            "expected_subjects": self.args.expected_subjects,
            "expected_recordings": self.args.expected_recordings,
            "expected_counts_match": (
                self.args.expected_subjects is not None
                and self.args.expected_recordings is not None
                and len(eligible_subjects) == self.args.expected_subjects
                and len(eligible_recordings) == self.args.expected_recordings
            ),
            "edf_signal_opened": False,
            "lance_generated": False,
            "generated_at": now_iso(),
        }
        (self.output_dir / "inventory_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return summary

    def process_one(self, rec: dict[str, Any]) -> None:
        summary = {"subject_id": rec["subject_id"], "recording_id": rec["recording_id"], "source_path": rec["source_path"], "eyes_condition": rec["eyes_condition"], "run_mode": "smoke" if self.args.smoke else "full", "processing_success": False, "error": ""}
        validation = {"subject_id": rec["subject_id"], "recording_id": rec["recording_id"], "source_path": rec["source_path"], "header_read_success": False, "fully_readable": False, "validation_pass": False, "error": ""}
        try:
            raw = mne.io.read_raw_edf(rec["absolute_path"], preload=False, infer_types=True, verbose="ERROR")
            validation["header_read_success"] = True
            types = raw.get_channel_types()
            # Status is an exact source channel name. Do not make exclusion
            # depend on MNE's inferred type because that inference has varied
            # between MNE releases (stim/eeg/misc). The other 64 channels must
            # still be explicitly typed as EEG.
            eeg_indices, status_indices = resolve_dortmund_channel_indices(raw.ch_names, types)
            status_type = types[status_indices[0]]
            if status_type != "stim":
                self.issue(
                    "warning",
                    "header_qc",
                    "status_type_inference_differs",
                    f"MNE inferred Status as {status_type!r}; excluded by exact channel name",
                    "Exclude Status from EEG data; retain the inferred type in diagnostics",
                    rec,
                    STATUS_NAME,
                )
            eeg_names = [raw.ch_names[i] for i in eeg_indices]
            if eeg_names != DORTMUND_CHANNELS:
                differences = [
                    f"{idx}: expected={expected!r} actual={actual!r}"
                    for idx, (expected, actual) in enumerate(zip(DORTMUND_CHANNELS, eeg_names))
                    if expected != actual
                ]
                if len(eeg_names) != len(DORTMUND_CHANNELS):
                    differences.append(f"length: expected=64 actual={len(eeg_names)}")
                raise RuntimeError(
                    "Dortmund native 64-channel order mismatch: " + "; ".join(differences[:12])
                )
            sfreq = float(raw.info["sfreq"])
            if not math.isclose(sfreq, SOURCE_SFREQ, rel_tol=0, abs_tol=1e-6):
                raise RuntimeError(f"Expected 1000 Hz, got {sfreq}")
            channel_meta_all = read_tsv(self.data_dir / rec["channels_path"])
            channel_meta = [
                row for row in channel_meta_all
                if row.get("name", "").casefold() != STATUS_NAME.casefold()
                and row.get("type", "EEG").casefold() == "eeg"
            ]
            if [row.get("name") for row in channel_meta] != DORTMUND_CHANNELS:
                actual_names = [row.get("name") for row in channel_meta]
                differences = [
                    f"{idx}: expected={expected!r} actual={actual!r}"
                    for idx, (expected, actual) in enumerate(zip(DORTMUND_CHANNELS, actual_names))
                    if expected != actual
                ]
                if len(actual_names) != len(DORTMUND_CHANNELS):
                    differences.append(f"length: expected=64 actual={len(actual_names)}")
                raise RuntimeError("channels.tsv order does not match native profile: " + "; ".join(differences[:12]))
            bad, channel_rows, read_count = self._scan_recording(raw, eeg_indices, rec, channel_meta)
            n_samples = int(raw.n_times)
            sidecar_duration = float(rec["sidecar"].get("RecordingDuration", n_samples / sfreq))
            duration, duration_matches = resolve_duration(n_samples, sfreq, sidecar_duration)
            if not duration_matches:
                self.issue("warning", "metadata", "duration_metadata_mismatch", f"EDF={duration}s, sidecar={sidecar_duration}s; using readable EDF samples", "Do not pad or synthesize signal", rec)
            good = np.flatnonzero(~bad)
            good_uv = convert_mne_v_to_uv(raw.get_data(picks=[eeg_indices[i] for i in good]))
            raw.close()
            filtered = apply_common_signal_processing(good_uv, sfreq, float(self.pre["target_sfreq"]), float(self.pre["notch_hz"]), float(self.pre["bandpass_low"]), float(self.pre["bandpass_high"]))
            processed = np.zeros((64, filtered.shape[1]), dtype=np.float64)
            processed[good] = filtered
            for local, idx in enumerate(good):
                values = filtered[local]
                channel_rows[idx].update({"post_filter_min_uv": float(values.min()), "post_filter_max_uv": float(values.max()), "post_filter_mean_uv": float(values.mean()), "post_filter_std_uv": float(values.std(dtype=np.float64))})
            self.rows["channel_header_inventory.csv"].extend(channel_rows)
            possible = window_slices(processed.shape[1], int(self.pre["segment_samples"]), int(round(float(self.pre["hop_seconds"]) * float(self.pre["target_sfreq"]))))
            emitted, smoke_truncated = limit_window_slices(possible, self.args.smoke, self.args.smoke_max_windows_per_recording)
            tail_samples, tail_seconds = dropped_tail(processed.shape[1], int(self.pre["segment_samples"]), int(round(float(self.pre["hop_seconds"]) * float(self.pre["target_sfreq"]))), float(self.pre["target_sfreq"]))
            cache_rows: list[dict[str, Any]] = []
            clips: list[float] = []
            qc_pass_count = 0
            for window_index, (start, stop) in enumerate(emitted):
                window = processed[:, start:stop].copy()
                window_bad, reasons_by_channel, preprocessing_finite = identify_window_bad_channels(window, bad)
                combined = merge_bad_channel_masks(bad, window_bad)
                qc_pass, qc_reasons = evaluate_window_qc(window, combined, (64, 2000), preprocessing_finite)
                zeroed, channel_mask = zero_bad_channels(window, combined)
                good_values = zeroed[~combined]
                clip = float(np.mean(np.abs(good_values) >= 200.0)) if good_values.size else 0.0
                pos_clip = float(np.mean(good_values >= 200.0)) if good_values.size else 0.0
                neg_clip = float(np.mean(good_values <= -200.0)) if good_values.size else 0.0
                absmax = float(np.max(np.abs(good_values))) if good_values.size else 0.0
                normalized = normalize_clip_divide_200(zeroed); normalized[combined] = 0.0
                if normalized.shape != (64, 2000) or normalized.dtype != np.float32 or not np.isfinite(normalized).all() or normalized.min() < -1 or normalized.max() > 1 or (combined.any() and not np.all(normalized[combined] == 0)):
                    qc_pass = False; qc_reasons.append("normalized_hard_qc_failed")
                source_start = int(round(start / float(self.pre["target_sfreq"]) * sfreq))
                sample_id = make_dortmund_sample_id(rec["subject_id"], rec["session_id"], rec["eyes_condition"], rec["acquisition"], rec["recording_id"], window_index, source_start)
                for idx in np.flatnonzero(window_bad):
                    self.rows["bad_channel_inventory.csv"].append({"subject_id": rec["subject_id"], "recording_id": rec["recording_id"], "source_path": rec["source_path"], "sample_id": sample_id, "channel_index": int(idx), "channel_name": DORTMUND_CHANNELS[idx], "scope": "window", "bad_reason": ";".join(reasons_by_channel[idx]), "affected_windows": 1, "data_zeroed": bool(np.all(normalized[idx] == 0)), "mask_verified": bool(combined[idx])})
                for idx in np.flatnonzero(bad):
                    if window_index == 0:
                        self.rows["bad_channel_inventory.csv"].append({"subject_id": rec["subject_id"], "recording_id": rec["recording_id"], "source_path": rec["source_path"], "sample_id": "", "channel_index": int(idx), "channel_name": DORTMUND_CHANNELS[idx], "scope": "recording", "bad_reason": channel_rows[idx]["bad_reason"], "affected_windows": len(emitted), "data_zeroed": True, "mask_verified": True})
                written = bool(qc_pass)
                self.rows["window_inventory.csv"].append({"sample_id": sample_id, "subject_id": rec["subject_id"], "session_id": rec["session_id"], "recording_id": rec["recording_id"], "eyes_condition": rec["eyes_condition"], "source_path": rec["source_path"], "window_index": window_index, "start_time": start / float(self.pre["target_sfreq"]), "duration": float(self.pre["window_seconds"]), "start_sample_source": source_start, "start_sample_target": start, "shape": "[64,2000]", "dtype": "float32", "age": rec["age"], "label": rec["label"], "qc_pass": qc_pass, "qc_failure_reason": ";".join(qc_reasons), "clip_fraction": clip, "positive_clip_fraction": pos_clip, "negative_clip_fraction": neg_clip, "bad_channel_count": int(combined.sum()), "bad_channel_names": ";".join(DORTMUND_CHANNELS[i] for i in np.flatnonzero(combined)), "written_to_lance": written})
                if written:
                    cache_rows.append({"sample_id": sample_id, "subject_id": rec["subject_id"], "session_id": rec["session_id"], "run_id": rec["run_id"], "recording_id": rec["recording_id"], "modality": "eeg", "data": normalized.reshape(-1), "shape": [64, 2000], "original_shape": [64, 2000], "valid_length": 128000, "qc_pass": True, "label": np.float32(rec["age"]), "age": np.float32(rec["age"]), "task_type": "regression", "target_name": "age", "target_unit": "year", "eyes_condition": rec["eyes_condition"], "pre_or_post": "pre_cognitive", "source_path": rec["source_path"], "start_time": start / float(self.pre["target_sfreq"]), "duration": float(self.pre["window_seconds"]), "source_start_sample": source_start, "target_start_sample": start, "channel_names": DORTMUND_CHANNELS, "channel_mask": channel_mask, "bad_channel_mask": combined, "channel_status": ["bad" if x else "good" for x in combined], "channel_layout": "dataset_level", "bad_channel_count": int(combined.sum()), "clip_fraction": np.float32(clip), "positive_clip_fraction": np.float32(pos_clip), "negative_clip_fraction": np.float32(neg_clip), "pre_normalization_abs_max_uv": np.float32(absmax), "source_sampling_rate": np.float32(sfreq), "target_sampling_rate": np.float32(self.pre["target_sfreq"]), "preprocessing_config_hash": self.config_hash})
                    qc_pass_count += 1
                else:
                    self.issue("error", "window_qc", "window_hard_qc_failed", ";".join(qc_reasons), "Do not write this window to main Lance", rec, sample_id=sample_id)
                clips.append(clip)
            if cache_rows:
                _write_lance_batch(self.lance_path, cache_rows, self.first_lance_write, self.schema)
                self.first_lance_write = False; self.written_rows += len(cache_rows)
            self.rows["dropped_tails.csv"].append({"subject_id": rec["subject_id"], "recording_id": rec["recording_id"], "source_path": rec["source_path"], "source_samples": n_samples, "resampled_samples": processed.shape[1], "theoretical_complete_windows": len(possible), "emitted_windows": len(emitted), "smoke_truncated_windows": len(possible) - len(emitted), "real_dropped_tail_samples": tail_samples, "real_dropped_tail_seconds": tail_seconds})
            summary.update({"source_samples": n_samples, "source_duration_seconds": duration, "resampled_samples": processed.shape[1], "theoretical_complete_windows": len(possible), "emitted_windows": len(emitted), "written_windows": len(cache_rows), "smoke_truncated_windows": len(possible) - len(emitted), "real_dropped_tail_samples": tail_samples, "real_dropped_tail_seconds": tail_seconds, "recording_bad_channels": ";".join(DORTMUND_CHANNELS[i] for i in np.flatnonzero(bad)), "qc_pass_windows": qc_pass_count, "qc_fail_windows": len(emitted) - qc_pass_count, "mean_clip_fraction": float(np.mean(clips)) if clips else 0.0, "max_clip_fraction": max(clips, default=0.0), "processing_success": True})
            validation.update({"fully_readable": True, "sampling_rate_hz": sfreq, "n_channels_total": 65, "n_eeg_channels": 64, "status_channel_count": 1, "n_samples_header": n_samples, "n_samples_read": int(read_count[0]), "duration_edf_seconds": duration, "duration_sidecar_seconds": sidecar_duration, "duration_matches": duration_matches, "channel_order_matches": True, "unit_conversion_verified": True, "has_nan": False, "has_inf": False, "recording_bad_channel_count": int(bad.sum()), "near_zero_variance_channel_count": sum(bool(row["near_zero_variance_flag"]) for row in channel_rows), "physical_limit_hit_count": sum(int(row["physical_limit_hit_count"]) for row in channel_rows), "dc_offset_channel_count": sum(bool(row["dc_offset_flag"]) for row in channel_rows), "extreme_amplitude_channel_count": sum(bool(row["extreme_amplitude_flag"]) for row in channel_rows), "validation_pass": True})
        except Exception as exc:
            summary["error"] = f"{type(exc).__name__}: {exc}"; validation["error"] = summary["error"]
            self.errors.append(f"[{now_iso()}] {rec['subject_id']} {rec['source_path']}\n{traceback.format_exc()}\n")
            self.issue("error", "recording_processing", "recording_failed_closed", summary["error"], "Exclude recording from main Lance", rec)
            print(
                f"[recording failed] {rec['subject_id']} {rec['eyes_condition']}: {summary['error']}",
                file=sys.stderr,
                flush=True,
            )
        finally:
            self.rows["recording_processing_summary.csv"].append(summary)
            self.rows["recording_validation.csv"].append(validation)

    def run(self) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        self.build_manifests()
        before = _snapshot_raw_files(self.data_dir, self.selected_paths)
        for index, rec in enumerate(self.recordings, 1):
            print(f"[{index}/{len(self.recordings)}] {rec['subject_id']} {rec['eyes_condition']}", flush=True)
            self.process_one(rec)
        after = _snapshot_raw_files(self.data_dir, self.selected_paths)
        if self.first_lance_write:
            self.issue(
                "error",
                "window_qc",
                "no_hard_qc_passing_windows",
                "No selected recording produced a hard-QC-passing window",
                "Inspect recording_validation.csv, recording_processing_summary.csv, preprocessing_issues.csv, and preprocessing_errors.log",
            )
            diagnostic_roundtrip = {
                "row_count": 0,
                "sample_id_unique": True,
                "schema_matches": False,
                "all_rows_valid": False,
                "forbidden_fields_absent": True,
                "dataset_present": False,
                "schema": str(self.schema),
            }
            self.write_outputs(before, after, diagnostic_roundtrip)
            raise RuntimeError("No hard-QC-passing windows were written")
        roundtrip = validate_dortmund_lance(
            self.lance_path, self.written_rows, self.args.min_age, self.args.max_age
        )
        self.write_outputs(before, after, roundtrip)
        return before, after, roundtrip

    def write_outputs(self, before: dict[str, Any], after: dict[str, Any], roundtrip: dict[str, Any]) -> None:
        for name, fields in CSV_FIELDS.items():
            if name.startswith("source_snapshot_"): continue
            _write_csv(self.output_dir / name, self.rows[name], fields)
        _write_csv(self.output_dir / "source_snapshot_before.csv", snapshot_rows(before), CSV_FIELDS["source_snapshot_before.csv"])
        _write_csv(self.output_dir / "source_snapshot_after.csv", snapshot_rows(after), CSV_FIELDS["source_snapshot_after.csv"])
        (self.output_dir / "preprocessing_errors.log").write_text("timestamp\tstage\tsubject_id\tsource_path\terror\n" + "".join(self.errors), encoding="utf-8")
        source_unchanged = before == after
        qc_pass_windows = sum(str(row["qc_pass"]).lower() == "true" or row["qc_pass"] is True for row in self.rows["window_inventory.csv"])
        dataset_written = self.lance_path.exists() and self.written_rows > 0
        summary = {"dataset_id": self.dataset_id, "cohort_id": self.cohort_id, "configured_min_age": self.args.min_age, "configured_max_age": self.args.max_age, "profile": "A", "run_mode": "smoke" if self.args.smoke else "full", "smoke": self.args.smoke, "selected_subjects": sorted(self.subjects), "selected_subject_count": len(self.subjects), "selected_recordings": len(self.recordings), "profile_a_total_subjects": len({r['subject_id'] for r in self.all_profile_recordings}), "profile_a_total_recordings": len(self.all_profile_recordings), "theoretical_complete_windows": sum(int(r.get("theoretical_complete_windows", 0) or 0) for r in self.rows["recording_processing_summary.csv"]), "emitted_windows": len(self.rows["window_inventory.csv"]), "written_windows": self.written_rows, "dataset_written": dataset_written, "smoke_truncated_windows": sum(int(r.get("smoke_truncated_windows", 0) or 0) for r in self.rows["recording_processing_summary.csv"]), "real_dropped_tail_seconds": sum(float(r.get("real_dropped_tail_seconds", 0) or 0) for r in self.rows["recording_processing_summary.csv"]), "qc_pass_windows": qc_pass_windows, "qc_fail_windows": len(self.rows["window_inventory.csv"]) - qc_pass_windows, "recording_success_count": sum(bool(r.get("processing_success")) for r in self.rows["recording_processing_summary.csv"]), "recording_failure_count": sum(not bool(r.get("processing_success")) for r in self.rows["recording_processing_summary.csv"]), "blocking_error_count": sum(r["severity"] == "error" for r in self.issues), "warning_count": sum(r["severity"] == "warning" for r in self.issues), "age_min": min(r["age"] for r in self.subjects.values()), "age_max": max(r["age"] for r in self.subjects.values()), "age_distribution": dict(Counter(r["age"] for r in self.subjects.values())), "eyes_open_windows": sum(r["eyes_condition"] == "eyes_open" and r["written_to_lance"] for r in self.rows["window_inventory.csv"]), "eyes_closed_windows": sum(r["eyes_condition"] == "eyes_closed" and r["written_to_lance"] for r in self.rows["window_inventory.csv"]), "source_files_unchanged": source_unchanged, "source_files_modified": not source_unchanged, "source_files_deleted": bool(set(before) - set(after)), "source_files_added": bool(set(after) - set(before)), "status_absent_from_eeg": True, "ses2_absent": all(r["session_id"] == "ses-1" for r in self.recordings), "post_cognitive_absent": all(r["acquisition"] == "pre" for r in self.recordings), "split_absent": "split" not in self.schema.names, "fbmae_tokens_absent": "model_tokens" not in self.schema.names and "electrode_ids" not in self.schema.names, "source_reference": SOURCE_REFERENCE, "source_ground": SOURCE_GROUND, "rereference_applied": False, "processing_reference": "unchanged", "channel_layout": "dataset_level", "native_channel_names": DORTMUND_CHANNELS, "roundtrip": roundtrip, "lance_row_count_matches_qc_pass": dataset_written and self.written_rows == qc_pass_windows == roundtrip["row_count"], "generated_at": now_iso()}
        (self.output_dir / "preprocessing_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        resolved = f'''[preprocessing]\ntarget_sfreq = {float(self.pre["target_sfreq"])}\nwindow_seconds = {float(self.pre["window_seconds"])}\nhop_seconds = {float(self.pre["hop_seconds"])}\nsegment_samples = {int(self.pre["segment_samples"])}\nbandpass_low = {float(self.pre["bandpass_low"])}\nbandpass_high = {float(self.pre["bandpass_high"])}\nnotch_hz = {float(self.pre["notch_hz"])}\nunit_before_normalization = "uV"\nnormalization = "clip_divide_200"\nnormalization_detail = "x_uV / 200, then clip to [-1, 1]"\n\n[windowing]\ndrop_incomplete_tail = true\npadding = false\n\n[source_expectations]\nformat = "EDF"\nsampling_rate_hz = 1000.0\nn_eeg_channels = 64\nstatus_channel = "Status"\n'''
        (self.output_dir / "resolved_config.toml").write_text(resolved, encoding="utf-8")
        provenance = {"created_at": now_iso(), "source_uri": self.args.source_uri, "resolved_local_data_dir": str(self.data_dir), "dataset_root": str(self.dataset_root), "output_dir": str(self.output_dir), "run_mode": summary["run_mode"], "cohort_id": self.cohort_id, "dataset_id": self.dataset_id, "min_age": self.args.min_age, "max_age": self.args.max_age, "expected_subjects": self.args.expected_subjects, "expected_recordings": self.args.expected_recordings, "selected_subjects": summary["selected_subjects"], "smoke_max_windows_per_recording": self.args.smoke_max_windows_per_recording if self.args.smoke else None, "python_version": platform.python_version(), "mne_version": mne.__version__, "numpy_version": np.__version__, "scipy_version": scipy.__version__, "pyarrow_version": pa.__version__, "lance_version": lance.__version__, "script_path": str(Path(__file__).resolve()), "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "config_path": str(self.args.config.resolve()), "config_sha256": self.config_hash, "actual_command": self.args.actual_command, "filter_parameters": FILTER_PARAMETERS, "actual_processing_order": ["MNE V to uV", "recording hard QC", "50 Hz notch", "0.3-75 Hz bandpass", "1000 to 200 Hz polyphase resample", "10 s non-overlap window", "window hard QC", "bad channel zero", "clip_divide_200"], "source_reference": SOURCE_REFERENCE, "source_ground": SOURCE_GROUND, "rereference_applied": False, "processing_reference": "unchanged", "montage": SOURCE_MONTAGE, "channel_layout": "dataset_level", "fbmae_channel_mapping": "pending", "source_snapshot_before": before, "source_snapshot_after": after, "source_files_unchanged": source_unchanged}
        provenance.update(_git_metadata(Path(__file__).resolve().parents[1]))
        (self.output_dir / "processing_provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        metadata = f'''spec_version = "0.1.0"\nprofile = "base"\ndataset_id = "{self.dataset_id}"\ncohort_id = "{self.cohort_id}"\nmin_age = {self.args.min_age}\nmax_age = {self.args.max_age}\nmodality = "eeg"\nsample_unit = "window"\ntask_type = "regression"\ntarget_name = "age"\ntarget_unit = "year"\nchannel_layout = "dataset_level"\nsource_reference = "FCz"\nsource_ground = "AFz"\nrereference_applied = false\nprocessing_reference = "unchanged"\nfbmae_channel_mapping = "pending"\n'''
        (self.output_dir / "metadata.toml").write_text(metadata, encoding="utf-8")
        (self.output_dir / "versions.toml").write_text(f'script_version = "{SCRIPT_VERSION}"\nmne = "{mne.__version__}"\nnumpy = "{np.__version__}"\nscipy = "{scipy.__version__}"\npyarrow = "{pa.__version__}"\nlance = "{lance.__version__}"\n', encoding="utf-8")
        (self.output_dir / "description.toml").write_text('title = "Dortmund Vital Study adult age Profile A"\ndescription = "ses-1 pre-cognitive eyes-open/eyes-closed 10-second EEG windows"\nsource_dataset = "OpenNeuro ds005385"\n', encoding="utf-8")
        report = self.render_report(summary)
        (self.output_dir / "preprocessing_report.md").write_text(report, encoding="utf-8")
        qc_dir = self.output_dir / "derivatives" / "qc"; report_dir = self.output_dir / "derivatives" / "reports"
        qc_dir.mkdir(parents=True, exist_ok=True); report_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.output_dir / "preprocessing_issues.csv", qc_dir / "preprocessing_issues.csv")
        shutil.copy2(self.output_dir / "preprocessing_summary.json", report_dir / "preprocessing_summary.json")
        shutil.copy2(self.output_dir / "preprocessing_report.md", report_dir / "preprocessing_report.md")

    def render_report(self, summary: dict[str, Any]) -> str:
        return f'''# Dortmund Vital Study 成人年龄公共预处理报告\n\n- 运行模式：{summary["run_mode"]}\n- cohort_id：{summary["cohort_id"]}；配置年龄范围：{summary["configured_min_age"]}–{summary["configured_max_age"]}。\n- Profile A 总范围：{summary["profile_a_total_subjects"]} 名被试，{summary["profile_a_total_recordings"]} 条 recording。\n- 本次选择：{summary["selected_subject_count"]} 名被试，{summary["selected_recordings"]} 条 recording。\n- 理论完整窗口：{summary["theoretical_complete_windows"]}；实际发出：{summary["emitted_windows"]}；写入 Lance：{summary["written_windows"]}。\n- eyes-open / eyes-closed：{summary["eyes_open_windows"]} / {summary["eyes_closed_windows"]}。\n- QC pass/fail：{summary["qc_pass_windows"]} / {summary["qc_fail_windows"]}。\n- smoke 截断窗口：{summary["smoke_truncated_windows"]}；真实 dropped tail：{summary["real_dropped_tail_seconds"]:.3f} 秒。\n- 实际年龄范围：{summary["age_min"]}–{summary["age_max"]}；age 与 label 均为逐被试 baseline 整数年龄。\n\n## 固定语义\n\n仅包含 ses-1、pre-cognitive、eyes-open/eyes-closed。ses-2、post-cognitive、年龄范围外被试和 Status 不进入主数据。source_reference=FCz，source_ground=AFz，rereference_applied=false，processing_reference=unchanged。保留 Dortmund 原生 64 通道顺序；FBMAE channel mapping=pending。\n\n## 处理链\n\nMNE V→µV → recording 硬 QC → 50 Hz notch → 0.3–75 Hz bandpass → 1000→200 Hz polyphase → 10 秒无重叠窗口 → window 硬 QC → 坏道置零 → x/200 clip[-1,1]。没有重参考、ICA、插值、padding、split 或模型 token。\n\nsub-230 的 60 秒 EyesClosed/pre recording 在完整读取与硬 QC 通过时保留并产生 6 窗。sub-600 的 duration 冲突以 EDF 实际可读样本为准，只记录 metadata warning，不补零。\n\n## 验收\n\n- Lance round-trip：{summary["roundtrip"]["all_rows_valid"]}\n- Lance 行数=window_inventory QC pass=summary written_windows：{summary["lance_row_count_matches_qc_pass"]}\n- 原始文件大小/mtime 未改变：{summary["source_files_unchanged"]}\n- blocking errors：{summary["blocking_error_count"]}；warnings：{summary["warning_count"]}\n'''


def write_empty_outputs(output_dir: Path) -> None:
    for name, fields in CSV_FIELDS.items():
        _write_csv(output_dir / name, [], fields)


def run_synthetic_smoke(args: argparse.Namespace) -> dict[str, Any]:
    """Exercise the shared processing chain and supervised Lance schema locally."""
    _prepare_output_dir(args.output_dir, args.overwrite, args.data_dir)
    write_empty_outputs(args.output_dir)
    config = load_and_validate_config(args.config); pre = config["preprocessing"]
    schema = dortmund_lance_schema(); lance_path = args.output_dir / "dataset.lance"
    cohort_id = build_cohort_id(args.min_age, args.max_age, args.cohort_id)
    dataset_id = build_dataset_id(cohort_id)
    rng = np.random.default_rng(385)
    rows = []
    subject_rows: list[dict[str, Any]] = []
    recording_rows: list[dict[str, Any]] = []
    processing_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    tail_rows: list[dict[str, Any]] = []
    midpoint_age = (args.min_age + args.max_age) // 2
    subjects = [
        ("sub-synth001", args.min_age),
        ("sub-230", midpoint_age),
        ("sub-600", args.max_age),
    ]
    for subject_id, age in subjects:
        subject_rows.append({"subject_id": subject_id, "age": age, "label": age, "age_dtype": "integer", "age_unit": "year", "age_scope": "baseline/ses-1", "age_status": "valid_numeric_baseline", "sex": "synthetic", "handedness": "synthetic", "selected_profile": "A", "recording_count": 2, "eligible": True})
        for condition in ("eyes_open", "eyes_closed"):
            recording_id = f"{subject_id}_{condition}_pre"
            recording_rows.append({"subject_id": subject_id, "session_id": "ses-1", "task": condition, "acquisition": "pre", "eyes_condition": condition, "run_id": "run-01", "recording_id": recording_id, "source_path": f"synthetic/{subject_id}/{condition}.edf", "sidecar_path": f"synthetic/{subject_id}/{condition}.json", "channels_path": f"synthetic/{subject_id}/{condition}_channels.tsv", "age": age, "label": age, "selected_reason": "selected_profile_a"})
            source = rng.normal(0, 20, size=(64, 31_000)).astype(np.float64)
            filtered = apply_common_signal_processing(source, 1000.0, 200.0, 50.0, 0.3, 75.0)
            possible = window_slices(filtered.shape[1], 2000, 2000)
            slices, _ = limit_window_slices(possible, True, 2)
            tail_samples, tail_seconds = dropped_tail(filtered.shape[1], 2000, 2000, 200)
            for index, (start, stop) in enumerate(slices):
                normalized = normalize_clip_divide_200(filtered[:, start:stop])
                sample_id = make_dortmund_sample_id(subject_id, "ses-1", condition, "pre", f"{subject_id}_ses-1_task-{condition}_acq-pre_eeg", index, int(start * 5))
                clip = float(np.mean(np.abs(filtered[:, start:stop]) >= 200))
                pos = float(np.mean(filtered[:, start:stop] >= 200)); neg = float(np.mean(filtered[:, start:stop] <= -200))
                rows.append({"sample_id": sample_id, "subject_id": subject_id, "session_id": "ses-1", "run_id": "run-01", "recording_id": recording_id, "modality": "eeg", "data": normalized.reshape(-1), "shape": [64, 2000], "original_shape": [64, 2000], "valid_length": 128000, "qc_pass": True, "label": np.float32(age), "age": np.float32(age), "task_type": "regression", "target_name": "age", "target_unit": "year", "eyes_condition": condition, "pre_or_post": "pre_cognitive", "source_path": f"synthetic/{subject_id}/{condition}.edf", "start_time": start / 200.0, "duration": 10.0, "source_start_sample": int(start * 5), "target_start_sample": start, "channel_names": DORTMUND_CHANNELS, "channel_mask": np.ones(64, bool), "bad_channel_mask": np.zeros(64, bool), "channel_status": ["good"] * 64, "channel_layout": "dataset_level", "bad_channel_count": 0, "clip_fraction": np.float32(clip), "positive_clip_fraction": np.float32(pos), "negative_clip_fraction": np.float32(neg), "pre_normalization_abs_max_uv": np.float32(np.max(np.abs(filtered[:, start:stop]))), "source_sampling_rate": np.float32(1000), "target_sampling_rate": np.float32(200), "preprocessing_config_hash": _config_hash(args.config)})
                window_rows.append({"sample_id": sample_id, "subject_id": subject_id, "session_id": "ses-1", "recording_id": recording_id, "eyes_condition": condition, "source_path": f"synthetic/{subject_id}/{condition}.edf", "window_index": index, "start_time": start / 200, "duration": 10.0, "start_sample_source": start * 5, "start_sample_target": start, "shape": "[64,2000]", "dtype": "float32", "age": age, "label": age, "qc_pass": True, "qc_failure_reason": "", "clip_fraction": clip, "positive_clip_fraction": pos, "negative_clip_fraction": neg, "bad_channel_count": 0, "bad_channel_names": "", "written_to_lance": True})
            processing_rows.append({"subject_id": subject_id, "recording_id": recording_id, "source_path": f"synthetic/{subject_id}/{condition}.edf", "eyes_condition": condition, "run_mode": "synthetic_smoke", "source_samples": 31000, "source_duration_seconds": 31.0, "resampled_samples": filtered.shape[1], "theoretical_complete_windows": len(possible), "emitted_windows": len(slices), "written_windows": len(slices), "smoke_truncated_windows": len(possible) - len(slices), "real_dropped_tail_samples": tail_samples, "real_dropped_tail_seconds": tail_seconds, "recording_bad_channels": "", "qc_pass_windows": len(slices), "qc_fail_windows": 0, "mean_clip_fraction": 0.0, "max_clip_fraction": 0.0, "processing_success": True, "error": ""})
            tail_rows.append({"subject_id": subject_id, "recording_id": recording_id, "source_path": f"synthetic/{subject_id}/{condition}.edf", "source_samples": 31000, "resampled_samples": filtered.shape[1], "theoretical_complete_windows": len(possible), "emitted_windows": len(slices), "smoke_truncated_windows": len(possible) - len(slices), "real_dropped_tail_samples": tail_samples, "real_dropped_tail_seconds": tail_seconds})
    _write_lance_batch(lance_path, rows, True, schema)
    roundtrip = validate_dortmund_lance(
        lance_path, len(rows), args.min_age, args.max_age
    )
    _write_csv(args.output_dir / "subject_manifest.csv", subject_rows, CSV_FIELDS["subject_manifest.csv"])
    _write_csv(args.output_dir / "recording_manifest.csv", recording_rows, CSV_FIELDS["recording_manifest.csv"])
    _write_csv(args.output_dir / "recording_processing_summary.csv", processing_rows, CSV_FIELDS["recording_processing_summary.csv"])
    _write_csv(args.output_dir / "window_inventory.csv", window_rows, CSV_FIELDS["window_inventory.csv"])
    _write_csv(args.output_dir / "dropped_tails.csv", tail_rows, CSV_FIELDS["dropped_tails.csv"])
    summary = {"synthetic": True, "smoke": True, "cohort_id": cohort_id, "dataset_id": dataset_id, "configured_min_age": args.min_age, "configured_max_age": args.max_age, "selected_subject_count": 3, "selected_recordings": 6, "theoretical_complete_windows": sum(r["theoretical_complete_windows"] for r in processing_rows), "emitted_windows": len(rows), "written_windows": len(rows), "smoke_truncated_windows": sum(r["smoke_truncated_windows"] for r in processing_rows), "real_dropped_tail_seconds": sum(r["real_dropped_tail_seconds"] for r in processing_rows), "window_inventory_qc_pass": sum(r["qc_pass"] for r in window_rows), "source_files_unchanged": True, "roundtrip": roundtrip, "formal_dortmund_data_processed_locally": False}
    (args.output_dir / "preprocessing_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "processing_provenance.json").write_text(json.dumps({"run_mode": "synthetic_smoke", "cohort_id": cohort_id, "dataset_id": dataset_id, "min_age": args.min_age, "max_age": args.max_age, "source_uri": None, "resolved_local_data_dir": str(args.data_dir.resolve()), "formal_data_opened": False, "actual_processing_order": ["notch", "bandpass", "resample", "window", "normalize"], "created_at": now_iso()}, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "resolved_config.toml").write_text(Path(args.config).read_text(encoding="utf-8"), encoding="utf-8")
    (args.output_dir / "metadata.toml").write_text(f'spec_version = "0.1.0"\nprofile = "base"\ndataset_id = "{dataset_id}"\ncohort_id = "{cohort_id}"\nmin_age = {args.min_age}\nmax_age = {args.max_age}\ntask_type = "regression"\n', encoding="utf-8")
    (args.output_dir / "versions.toml").write_text(f'script_version = "{SCRIPT_VERSION}"\n', encoding="utf-8")
    (args.output_dir / "description.toml").write_text('description = "synthetic smoke only"\n', encoding="utf-8")
    (args.output_dir / "preprocessing_report.md").write_text("# Dortmund synthetic smoke\n\n正式 Dortmund EDF 未在本地打开。Lance round-trip 通过。\n", encoding="utf-8")
    (args.output_dir / "preprocessing_errors.log").write_text("timestamp\tstage\tsubject_id\tsource_path\terror\n", encoding="utf-8")
    (args.output_dir / "derivatives" / "qc").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "derivatives" / "reports").mkdir(parents=True, exist_ok=True)
    return summary


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-uri", default=os.environ.get("DORTMUND_SOURCE_URI", DEFAULT_SOURCE_URI))
    parser.add_argument("--chunk-seconds", type=float, default=30.0)
    parser.add_argument("--min-age", type=int, default=20)
    parser.add_argument("--max-age", type=int, default=35)
    parser.add_argument("--cohort-id", default=None)
    parser.add_argument("--expected-subjects", type=int, default=None)
    parser.add_argument("--expected-recordings", type=int, default=None)
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-subjects", nargs="+", default=None)
    parser.add_argument("--smoke-max-windows-per-recording", type=int, default=None)
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    tokens = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args(tokens)
    if args.chunk_seconds <= 0: parser.error("--chunk-seconds must be >0")
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
    if not (args.smoke or args.synthetic_smoke or args.inventory_only):
        if args.expected_subjects is None:
            parser.error("--expected-subjects is required for a non-default age range")
        if args.expected_recordings is None:
            parser.error("--expected-recordings is required for a non-default age range")
    if args.expected_subjects is not None and args.expected_subjects <= 0:
        parser.error("--expected-subjects must be >0")
    if args.expected_recordings is not None and args.expected_recordings <= 0:
        parser.error("--expected-recordings must be >0")
    if args.inventory_only and (args.smoke or args.synthetic_smoke):
        parser.error("--inventory-only cannot be combined with smoke modes")
    if args.synthetic_smoke:
        args.smoke = True
    if args.smoke:
        args.smoke_subjects = parse_smoke_subjects(args.smoke_subjects or ["sub-004,sub-230,sub-600"])
        args.smoke_subjects = [x.lower() for x in args.smoke_subjects]
        args.smoke_max_windows_per_recording = args.smoke_max_windows_per_recording or 2
        if args.smoke_max_windows_per_recording <= 0: parser.error("smoke window limit must be >0")
    elif args.smoke_subjects or args.smoke_max_windows_per_recording:
        parser.error("smoke selection arguments require --smoke")
    else:
        args.smoke_subjects = []
    args.actual_command = subprocess.list2cmdline([sys.executable, str(Path(__file__).resolve()), *tokens])
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.synthetic_smoke:
            summary = run_synthetic_smoke(args)
        else:
            _prepare_output_dir(args.output_dir, args.overwrite, args.data_dir)
            pipeline = DortmundPipeline(args)
            if args.inventory_only:
                summary = pipeline.run_inventory()
            else:
                pipeline.run()
                summary = json.loads((args.output_dir / "preprocessing_summary.json").read_text(encoding="utf-8"))
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return 0 if summary.get("blocking_error_count", 0) == 0 and summary.get("source_files_unchanged", True) else 2
    except Exception:
        # Always preserve the original failure on stderr. In particular, an
        # unwritable Ti-ONE output mount must not mask the useful traceback
        # with a second exception while trying to create the error log.
        original_traceback = traceback.format_exc()
        sys.stderr.write(original_traceback)
        try:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            with (args.output_dir / "preprocessing_errors.log").open("a", encoding="utf-8") as handle:
                handle.write(f"{now_iso()}\tfatal\t\t\t{original_traceback}\n")
        except Exception as log_error:
            print(
                f"Could not write preprocessing_errors.log below {args.output_dir}: {log_error}",
                file=sys.stderr,
            )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
