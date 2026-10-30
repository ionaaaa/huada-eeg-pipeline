#!/usr/bin/env python3
"""Stages 4-9 of the Huada resting-state EEG preprocessing workflow.

The raw BDF files are opened read-only. Processing is recording-local and
produces a shared, task-neutral Lance window cache. No labels, dataset split,
rereferencing, montage inference, interpolation, ICA, or soft-threshold
rejection are performed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
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
from mne.filter import filter_data, notch_filter, resample

from validate_huada_raw_bdf import load_and_validate_config, unit_to_uv_multiplier


SCRIPT_VERSION = "1.1.0"
READER_OUTPUT_UNIT = "V"
READER_TO_UV_MULTIPLIER = 1e6
KNOWN_BAD_CHANNELS = {("SUB5", "FC1"), ("SUB8", "FT7")}
REQUIRED_OUTPUT_NAMES = (
    "common_windows.lance", "resolved_preprocessing_config.toml",
    "processing_provenance.json", "recording_processing_summary.csv",
    "window_inventory.csv", "bad_channel_inventory.csv", "dropped_tails.csv",
    "preprocessing_issues.csv", "preprocessing_summary.json",
    "preprocessing_report.md", "preprocessing_errors.log",
)

FILTER_PARAMETERS = {
    "notch": {
        "frequency_hz": 50.0,
        "method": "fir",
        "phase": "zero-double",
        "fir_design": "firwin",
        "fir_window": "hamming",
        "filter_length": "auto",
        "transition_bandwidth_hz": 1.0,
        "padding": "reflect_limited",
    },
    "bandpass": {
        "method": "fir",
        "phase": "zero-double",
        "fir_design": "firwin",
        "fir_window": "hamming",
        "filter_length": "auto",
        "low_transition_bandwidth": "auto",
        "high_transition_bandwidth": "auto",
        "padding": "reflect_limited",
    },
    "resample": {
        "method": "polyphase",
        "window": "auto",
        "padding": "reflect",
        "npad": 100,
    },
}


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def convert_mne_v_to_uv(data_v: np.ndarray) -> np.ndarray:
    """Convert MNE reader output in volts to microvolts exactly once."""
    return np.asarray(data_v, dtype=np.float64) * READER_TO_UV_MULTIPLIER


def _same_finite_value(value: float, expected: float | None) -> bool:
    return expected is not None and math.isfinite(expected) and value == expected


def detect_hard_bad_channel(
    values_uv: np.ndarray,
    physical_min_uv: float | None = None,
    physical_max_uv: float | None = None,
    digital_min_physical_uv: float | None = None,
    digital_max_physical_uv: float | None = None,
) -> list[str]:
    """Return threshold-free hard bad-channel reasons for one full signal."""
    x = np.asarray(values_uv)
    reasons: list[str] = []
    if x.ndim != 1 or x.size == 0:
        return ["missing_or_empty_channel"]
    if np.isnan(x).any():
        reasons.append("contains_nan")
    if np.isinf(x).any():
        reasons.append("contains_inf")
    finite = x[np.isfinite(x)]
    if finite.size != x.size or finite.size == 0:
        return reasons
    first = float(finite[0])
    is_constant = bool(np.all(finite == finite[0]))
    if bool(np.all(finite == 0)):
        reasons.append("all_zero")
    if is_constant:
        reasons.append("strict_constant")
        if float(np.var(finite, dtype=np.float64)) == 0.0:
            reasons.append("exact_zero_variance")
        if _same_finite_value(first, physical_min_uv):
            reasons.append("fixed_at_physical_minimum")
        if _same_finite_value(first, physical_max_uv):
            reasons.append("fixed_at_physical_maximum")
        if _same_finite_value(first, digital_min_physical_uv):
            reasons.append("fixed_at_digital_minimum_physical_value")
        if _same_finite_value(first, digital_max_physical_uv):
            reasons.append("fixed_at_digital_maximum_physical_value")
    return list(dict.fromkeys(reasons))


def window_slices(n_samples: int, window_samples: int, hop_samples: int) -> list[tuple[int, int]]:
    if window_samples <= 0 or hop_samples <= 0:
        raise ValueError("window_samples and hop_samples must be positive")
    if n_samples < window_samples:
        return []
    return [(start, start + window_samples) for start in range(0, n_samples - window_samples + 1, hop_samples)]


def dropped_tail(n_samples: int, window_samples: int, hop_samples: int, sfreq: float) -> tuple[int, float]:
    slices = window_slices(n_samples, window_samples, hop_samples)
    used_stop = slices[-1][1] if slices else 0
    dropped = n_samples - used_stop
    return dropped, dropped / float(sfreq)


def merge_bad_channel_masks(recording_bad: Sequence[bool], window_bad: Sequence[bool]) -> np.ndarray:
    a = np.asarray(recording_bad, dtype=bool)
    b = np.asarray(window_bad, dtype=bool)
    if a.shape != b.shape:
        raise ValueError(f"bad-channel mask shape mismatch: {a.shape} vs {b.shape}")
    return np.logical_or(a, b)


def zero_bad_channels(data: np.ndarray, bad_mask: Sequence[bool]) -> tuple[np.ndarray, np.ndarray]:
    out = np.asarray(data).copy()
    mask = np.asarray(bad_mask, dtype=bool)
    if out.ndim != 2 or mask.shape != (out.shape[0],):
        raise ValueError("bad mask must match the channel dimension")
    out[mask, :] = 0.0
    channel_mask = np.ones(out.shape[0], dtype=bool)
    return out, channel_mask


def normalize_clip_divide_200(data_uv: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(data_uv, dtype=np.float64) / 200.0, -1.0, 1.0).astype(np.float32)


def evaluate_window_qc(
    data_before_zeroing: np.ndarray,
    combined_bad_mask: Sequence[bool],
    expected_shape: tuple[int, int] = (64, 2000),
    preprocessing_success: bool = True,
) -> tuple[bool, list[str]]:
    x = np.asarray(data_before_zeroing)
    bad = np.asarray(combined_bad_mask, dtype=bool)
    reasons: list[str] = []
    if x.shape != expected_shape:
        reasons.append("shape_mismatch")
    if x.size != expected_shape[0] * expected_shape[1]:
        reasons.append("flattened_length_mismatch")
    if bad.shape != (expected_shape[0],):
        reasons.append("bad_mask_shape_mismatch")
    if not preprocessing_success:
        reasons.append("preprocessing_incomplete")
    if x.shape == expected_shape and bad.shape == (expected_shape[0],):
        good = ~bad
        if not np.any(good):
            reasons.append("all_channels_bad")
        elif not np.isfinite(x[good]).all():
            reasons.append("nonfinite_nonbad_data")
        finite_for_zero = np.where(np.isfinite(x), x, 0.0)
        if bool(np.all(finite_for_zero == 0)):
            reasons.append("entire_window_zero")
        try:
            x.reshape(expected_shape)
        except ValueError:
            reasons.append("reshape_failed")
    return not reasons, reasons


def make_sample_id(subject_id: str, recording_id: str, window_index: int) -> str:
    return f"{subject_id}__{recording_id}__w{window_index:05d}"


def identify_window_bad_channels(window_uv: np.ndarray, recording_bad: Sequence[bool]) -> tuple[np.ndarray, list[list[str]], bool]:
    x = np.asarray(window_uv)
    rec_bad = np.asarray(recording_bad, dtype=bool)
    if x.ndim != 2 or rec_bad.shape != (x.shape[0],):
        raise ValueError("window and recording bad mask are not aligned")
    new_bad = np.zeros(x.shape[0], dtype=bool)
    reasons: list[list[str]] = [[] for _ in range(x.shape[0])]
    preprocessing_finite = True
    for idx in range(x.shape[0]):
        if rec_bad[idx]:
            continue
        channel = x[idx]
        if not np.isfinite(channel).all():
            new_bad[idx] = True
            reasons[idx].append("window_nonfinite_after_preprocessing")
            preprocessing_finite = False
            continue
        if bool(np.all(channel == 0)):
            new_bad[idx] = True
            reasons[idx].append("window_all_zero")
        elif bool(np.all(channel == channel[0])):
            new_bad[idx] = True
            reasons[idx].append("window_strict_constant")
    return new_bad, reasons, preprocessing_finite


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _optional_float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def parse_smoke_subjects(values: Sequence[str] | None) -> list[str]:
    """Parse comma-separated and/or space-separated subject IDs deterministically."""
    if not values:
        return ["SUB1", "SUB5"]
    parsed: list[str] = []
    for value in values:
        for item in value.split(","):
            subject_id = item.strip().upper()
            if subject_id and subject_id not in parsed:
                parsed.append(subject_id)
    if not parsed:
        raise ValueError("Smoke subject selection is empty")
    return parsed


def select_recordings(
    recordings: Sequence[dict[str, str]], smoke: bool, smoke_subjects: Sequence[str]
) -> list[dict[str, str]]:
    """Select smoke recordings before any BDF is opened."""
    if not smoke:
        return list(recordings)
    wanted = {subject.upper() for subject in smoke_subjects}
    selected = [row for row in recordings if row["subject_id"].upper() in wanted]
    found = {row["subject_id"].upper() for row in selected}
    missing = sorted(wanted - found)
    if missing:
        raise ValueError(f"Smoke subjects not found in recording inventory: {missing}")
    return selected


def limit_window_slices(
    slices: Sequence[tuple[int, int]], smoke: bool, max_windows_per_recording: int | None
) -> tuple[list[tuple[int, int]], bool]:
    all_slices = list(slices)
    if not smoke:
        return all_slices, False
    if max_windows_per_recording is None or max_windows_per_recording <= 0:
        raise ValueError("Smoke max windows per recording must be > 0")
    emitted = all_slices[:max_windows_per_recording]
    return emitted, len(emitted) < len(all_slices)


def _snapshot_raw_files(
    data_dir: Path, relative_paths: Sequence[str] | None = None
) -> dict[str, dict[str, Any]]:
    snapshot: dict[str, dict[str, Any]] = {}
    paths = (
        [data_dir / Path(relative_path) for relative_path in relative_paths]
        if relative_paths is not None
        else list(data_dir.rglob("*.bdf"))
    )
    for path in sorted(paths, key=lambda p: p.as_posix().lower()):
        stat = path.stat()
        snapshot[path.relative_to(data_dir).as_posix()] = {
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    return snapshot


def _prepare_output_dir(output_dir: Path, overwrite: bool, data_dir: Path) -> None:
    resolved_out = output_dir.resolve()
    resolved_data = data_dir.resolve()
    if resolved_out == resolved_data or resolved_data in resolved_out.parents:
        raise ValueError("Output directory must not be the raw data directory or a child of it")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {output_dir}. Use --overwrite explicitly.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _config_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


def _git_metadata(project_root: Path) -> dict[str, Any]:
    commit = _git_commit(project_root)
    if commit is None:
        return {"git_commit": None, "git_branch": None, "git_dirty": None}
    try:
        branch = subprocess.run(
            ["git", "-C", str(project_root), "branch", "--show-current"],
            capture_output=True, text=True, check=True,
        ).stdout.strip() or None
        dirty = bool(subprocess.run(
            ["git", "-C", str(project_root), "status", "--porcelain"],
            capture_output=True, text=True, check=True,
        ).stdout.strip())
        return {"git_commit": commit, "git_branch": branch, "git_dirty": dirty}
    except Exception:
        return {"git_commit": commit, "git_branch": None, "git_dirty": None}


def validate_smoke_output_dir(output_dir: Path, project_root: Path) -> None:
    """Prevent a smoke run from writing into the formal full-run output tree."""
    formal_root = (project_root / "preprocessed_common").resolve()
    resolved = output_dir.resolve()
    if resolved == formal_root or formal_root in resolved.parents:
        raise ValueError("Smoke output must not be inside the formal preprocessed_common directory")


def build_full_run_command(args: argparse.Namespace, output_dir: Path) -> str:
    """Build the exact PowerShell full-run command without smoke or overwrite flags."""
    values = [
        ("--data-dir", args.data_dir.resolve()),
        ("--config", args.config.resolve()),
        ("--subject-manifest", args.subject_manifest.resolve()),
        ("--recording-validation", args.recording_validation.resolve()),
        ("--channel-header-inventory", args.channel_header_inventory.resolve()),
        ("--output-dir", output_dir.resolve()),
        ("--chunk-seconds", args.chunk_seconds),
    ]
    lines = [f'& "{Path(sys.executable).resolve()}" "{Path(__file__).resolve()}" `']
    for index, (flag, value) in enumerate(values):
        suffix = " `" if index < len(values) - 1 else ""
        rendered = str(value)
        if isinstance(value, Path) or " " in rendered:
            rendered = f'"{rendered}"'
        lines.append(f"  {flag} {rendered}{suffix}")
    return "\n".join(lines) + "\n"


def common_window_schema() -> pa.Schema:
    return pa.schema([
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("subject_id", pa.string(), nullable=False),
        pa.field("modality", pa.string(), nullable=False),
        pa.field("source_path", pa.string(), nullable=False),
        pa.field("recording_id", pa.string(), nullable=False),
        pa.field("window_index", pa.int32(), nullable=False),
        pa.field("start_time", pa.float64(), nullable=False),
        pa.field("duration", pa.float64(), nullable=False),
        pa.field("source_start_sample", pa.int64(), nullable=False),
        pa.field("source_stop_sample", pa.int64(), nullable=False),
        pa.field("data", pa.list_(pa.float32(), 128000), nullable=False),
        pa.field("shape", pa.list_(pa.int32()), nullable=False),
        pa.field("original_shape", pa.list_(pa.int32()), nullable=False),
        pa.field("valid_length", pa.int64(), nullable=False),
        pa.field("channel_names", pa.list_(pa.string()), nullable=False),
        pa.field("channel_mask", pa.list_(pa.bool_(), 64), nullable=False),
        pa.field("bad_channel_mask", pa.list_(pa.bool_(), 64), nullable=False),
        pa.field("channel_status", pa.list_(pa.string()), nullable=False),
        pa.field("qc_pass", pa.bool_(), nullable=False),
        pa.field("bad_channel_count", pa.int32(), nullable=False),
        pa.field("age", pa.int32(), nullable=True),
        pa.field("sex", pa.string(), nullable=False),
        pa.field("original_directory_name", pa.string(), nullable=False),
        pa.field("clip_fraction", pa.float32(), nullable=False),
        pa.field("positive_clip_fraction", pa.float32(), nullable=False),
        pa.field("negative_clip_fraction", pa.float32(), nullable=False),
        pa.field("pre_normalization_abs_max_uv", pa.float32(), nullable=False),
        pa.field("source_sampling_rate", pa.float32(), nullable=False),
        pa.field("target_sampling_rate", pa.float32(), nullable=False),
        pa.field("preprocessing_config_hash", pa.string(), nullable=False),
    ])


def rows_to_arrow(rows: list[dict[str, Any]], schema: pa.Schema | None = None) -> pa.Table:
    schema = schema or common_window_schema()
    if not rows:
        return pa.Table.from_pylist([], schema=schema)
    columns: dict[str, pa.Array] = {}
    for field in schema:
        values = [row[field.name] for row in rows]
        if field.name == "data":
            flat = np.concatenate([np.asarray(v, dtype=np.float32).reshape(-1) for v in values])
            columns[field.name] = pa.FixedSizeListArray.from_arrays(pa.array(flat, type=pa.float32()), 128000)
        elif field.name in {"channel_mask", "bad_channel_mask"}:
            flat = np.concatenate([np.asarray(v, dtype=bool).reshape(-1) for v in values])
            columns[field.name] = pa.FixedSizeListArray.from_arrays(pa.array(flat, type=pa.bool_()), 64)
        elif field.name in {"channel_names", "channel_status"} and pa.types.is_fixed_size_list(field.type):
            flat = [item for value in values for item in value]
            columns[field.name] = pa.FixedSizeListArray.from_arrays(pa.array(flat, type=field.type.value_type), 64)
        else:
            columns[field.name] = pa.array(values, type=field.type)
    return pa.Table.from_arrays([columns[field.name] for field in schema], schema=schema)


def _write_lance_batch(path: Path, rows: list[dict[str, Any]], first: bool) -> None:
    table = rows_to_arrow(rows)
    lance.write_dataset(
        table,
        path,
        mode="create" if first else "append",
        max_rows_per_group=32,
        max_rows_per_file=256,
        data_storage_version="stable",
    )


def validate_lance_roundtrip(path: Path, expected_rows: int | None = None) -> dict[str, Any]:
    dataset = lance.dataset(path)
    schema = dataset.schema
    forbidden = {"label", "label_name", "split"}.intersection(schema.names)
    if forbidden:
        raise AssertionError(f"Forbidden supervised fields in cache: {sorted(forbidden)}")
    expected_schema = common_window_schema()
    if schema != expected_schema:
        raise AssertionError(f"Lance schema mismatch:\nexpected={expected_schema}\nactual={schema}")
    count = dataset.count_rows()
    if expected_rows is not None and count != expected_rows:
        raise AssertionError(f"Expected {expected_rows} rows, got {count}")

    sample_ids: set[str] = set()
    row_count = 0
    bad_zero_verified = True
    all_shapes_valid = True
    all_values_finite = True
    all_values_in_range = True
    min_value = math.inf
    max_value = -math.inf
    for batch in dataset.scanner(batch_size=8).to_batches():
        ids = batch.column(batch.schema.get_field_index("sample_id")).to_pylist()
        data_arr = batch.column(batch.schema.get_field_index("data"))
        bad_arr = batch.column(batch.schema.get_field_index("bad_channel_mask"))
        shape_arr = batch.column(batch.schema.get_field_index("shape")).to_pylist()
        for idx, sample_id in enumerate(ids):
            if sample_id in sample_ids:
                raise AssertionError(f"Duplicate sample_id: {sample_id}")
            sample_ids.add(sample_id)
            if shape_arr[idx] != [64, 2000]:
                all_shapes_valid = False
                raise AssertionError(f"Unexpected shape for {sample_id}: {shape_arr[idx]}")
            data = np.asarray(data_arr[idx].values, dtype=np.float32).reshape(64, 2000)
            bad = np.asarray(bad_arr[idx].values, dtype=bool)
            if data.dtype != np.float32 or not np.isfinite(data).all():
                all_values_finite = False
                raise AssertionError(f"Invalid dtype or nonfinite data for {sample_id}")
            if data.min() < -1.0 or data.max() > 1.0:
                all_values_in_range = False
                raise AssertionError(f"Normalized range violation for {sample_id}")
            if bad.any() and not np.all(data[bad] == 0.0):
                bad_zero_verified = False
                raise AssertionError(f"Bad channel not zero for {sample_id}")
            min_value = min(min_value, float(data.min()))
            max_value = max(max_value, float(data.max()))
            row_count += 1
    return {
        "row_count": row_count,
        "unique_sample_id_count": len(sample_ids),
        "sample_id_unique": len(sample_ids) == row_count,
        "schema_matches": schema == expected_schema,
        "all_shapes_valid": all_shapes_valid,
        "all_values_finite": all_values_finite,
        "all_values_in_range": all_values_in_range,
        "bad_slots_zero": bad_zero_verified,
        "normalized_min": min_value if row_count else None,
        "normalized_max": max_value if row_count else None,
        "forbidden_fields_absent": not forbidden,
        "contains_label": "label" in schema.names,
        "contains_label_name": "label_name" in schema.names,
        "contains_split": "split" in schema.names,
        "data_type": str(schema.field("data").type),
        "channel_mask_type": str(schema.field("channel_mask").type),
        "bad_mask_type": str(schema.field("bad_channel_mask").type),
        "age_nullable": schema.field("age").nullable,
    }


class Pipeline:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.data_dir = args.data_dir.resolve()
        self.output_dir = args.output_dir.resolve()
        self.config = load_and_validate_config(args.config)
        self.pre = self.config["preprocessing"]
        self.exp = self.config["source_expectations"]
        self.windowing = self.config["windowing"]
        self.config_hash = _config_hash(args.config)
        self.manifest = {row["subject_id"]: row for row in _read_csv(args.subject_manifest)}
        all_recordings = _read_csv(args.recording_validation)
        self.recordings = select_recordings(
            all_recordings, args.smoke, args.smoke_subjects,
        )
        self.selected_subjects = list(dict.fromkeys(row["subject_id"] for row in self.recordings))
        header_rows = _read_csv(args.channel_header_inventory)
        self.headers: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in header_rows:
            self.headers[row["source_path"]].append(row)
        for rows in self.headers.values():
            rows.sort(key=lambda row: int(row["channel_index"]))
        self.issues: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self.bad_rows: list[dict[str, Any]] = []
        self.recording_rows: list[dict[str, Any]] = []
        self.window_rows: list[dict[str, Any]] = []
        self.tail_rows: list[dict[str, Any]] = []
        self.subject_stats: dict[str, dict[str, Any]] = defaultdict(lambda: {
            "window_count": 0, "qc_pass": 0, "qc_fail": 0,
            "recording_bad_channels": set(), "window_new_bad_channels": set(),
            "clip_fractions": [], "dropped_tail_seconds": 0.0,
        })
        self.lance_path = self.output_dir / "common_windows.lance"
        self._first_lance_write = True

    def issue(self, severity: str, subject_id: str, source_path: str, issue_type: str,
              description: str, recommendation: str, channel_name: str = "",
              sample_id: str = "") -> None:
        stage_by_type = {
            "raw_hard_bad_channel": "raw_signal_qc",
            "window_hard_bad_channel": "window_qc",
            "window_qc_failed": "window_qc",
            "recording_processing_failed": "preprocessing",
            "raw_source_changed": "source_integrity",
        }
        self.issues.append({
            "severity": severity, "stage": stage_by_type.get(issue_type, "preprocessing"),
            "subject_id": subject_id, "source_path": source_path,
            "sample_id": sample_id, "channel_name": channel_name,
            "issue_type": issue_type, "description": description,
            "action": recommendation,
        })

    def validate_inputs(self) -> None:
        if not self.manifest:
            raise ValueError("Subject manifest is empty")
        if not self.recordings:
            raise ValueError("Recording validation inventory is empty")
        for row in self.recordings:
            if not _parse_bool(row.get("validation_pass")):
                raise ValueError(f"Recording did not pass stages 1-3: {row['source_path']}")
            if row["subject_id"] not in self.manifest:
                raise ValueError(f"Missing manifest subject: {row['subject_id']}")
            headers = self.headers.get(row["source_path"], [])
            if len(headers) != int(self.exp["n_channels"]):
                raise ValueError(f"Expected 64 header rows for {row['source_path']}, got {len(headers)}")
            for header in headers:
                unit = header.get("physical_dimension", "")
                if unit_to_uv_multiplier(unit) != 1.0:
                    raise ValueError(f"Header physical unit is not verified µV: {row['source_path']} {unit!r}")
                if header.get("mne_data_unit") != READER_OUTPUT_UNIT:
                    raise ValueError(f"Unexpected reader unit in inventory: {header.get('mne_data_unit')!r}")

    def _scan_raw_qc(self, raw: mne.io.BaseRaw, headers: list[dict[str, str]],
                     subject_id: str, source_path: str) -> tuple[np.ndarray, list[dict[str, Any]]]:
        n_channels = len(headers)
        n_samples = raw.n_times
        chunk_samples = max(1, int(round(float(self.args.chunk_seconds) * float(raw.info["sfreq"]))))
        mins = np.full(n_channels, np.inf)
        maxs = np.full(n_channels, -np.inf)
        sums = np.zeros(n_channels, dtype=np.float64)
        sumsquares = np.zeros(n_channels, dtype=np.float64)
        counts = np.zeros(n_channels, dtype=np.int64)
        has_nan = np.zeros(n_channels, dtype=bool)
        has_inf = np.zeros(n_channels, dtype=bool)
        first_values = np.full(n_channels, np.nan)
        all_same = np.ones(n_channels, dtype=bool)

        for start in range(0, n_samples, chunk_samples):
            stop = min(start + chunk_samples, n_samples)
            data_uv = convert_mne_v_to_uv(raw.get_data(start=start, stop=stop, picks=np.arange(n_channels)))
            if data_uv.shape != (n_channels, stop - start):
                raise RuntimeError(f"Chunk shape mismatch at {start}:{stop}: {data_uv.shape}")
            for idx in range(n_channels):
                x = data_uv[idx]
                has_nan[idx] |= np.isnan(x).any()
                has_inf[idx] |= np.isinf(x).any()
                finite = x[np.isfinite(x)]
                if finite.size == 0:
                    continue
                if counts[idx] == 0:
                    first_values[idx] = finite[0]
                if not np.all(finite == first_values[idx]):
                    all_same[idx] = False
                mins[idx] = min(mins[idx], float(finite.min()))
                maxs[idx] = max(maxs[idx], float(finite.max()))
                sums[idx] += float(finite.sum(dtype=np.float64))
                sumsquares[idx] += float(np.dot(finite, finite))
                counts[idx] += finite.size

        bad_mask = np.zeros(n_channels, dtype=bool)
        rows: list[dict[str, Any]] = []
        for idx, header in enumerate(headers):
            physical_min = _optional_float(header.get("physical_min"))
            physical_max = _optional_float(header.get("physical_max"))
            reasons: list[str] = []
            if counts[idx] != n_samples:
                reasons.append("contains_nan" if has_nan[idx] else "")
                reasons.append("contains_inf" if has_inf[idx] else "")
                if counts[idx] + int(has_nan[idx] or has_inf[idx]) < n_samples:
                    reasons.append("channel_length_or_finiteness_mismatch")
            if n_samples == 0:
                reasons.append("missing_or_empty_channel")
            elif counts[idx] == n_samples and all_same[idx]:
                value = float(first_values[idx])
                if value == 0.0:
                    reasons.append("all_zero")
                reasons.extend(["strict_constant", "exact_zero_variance"])
                if _same_finite_value(value, physical_min):
                    reasons.append("fixed_at_physical_minimum")
                    reasons.append("fixed_at_digital_minimum_physical_value")
                if _same_finite_value(value, physical_max):
                    reasons.append("fixed_at_physical_maximum")
                    reasons.append("fixed_at_digital_maximum_physical_value")
            reasons = [reason for reason in dict.fromkeys(reasons) if reason]
            is_bad = bool(reasons)
            bad_mask[idx] = is_bad
            mean = sums[idx] / counts[idx] if counts[idx] else math.nan
            variance = max(0.0, sumsquares[idx] / counts[idx] - mean * mean) if counts[idx] else math.nan
            row = {
                "subject_id": subject_id, "source_path": source_path,
                "channel_index": idx, "channel_name": header["channel_name"],
                "is_bad": is_bad, "bad_reason": ";".join(reasons),
                "raw_min_uv": mins[idx] if counts[idx] else math.nan,
                "raw_max_uv": maxs[idx] if counts[idx] else math.nan,
                "raw_mean_uv": mean, "raw_std_uv": math.sqrt(variance) if counts[idx] else math.nan,
                "physical_min_uv": physical_min, "physical_max_uv": physical_max,
                "source_header_unit": header.get("physical_dimension", ""),
                "reader_output_unit": READER_OUTPUT_UNIT,
                "reader_to_uv_multiplier": READER_TO_UV_MULTIPLIER,
                "unit_conversion_verified": unit_to_uv_multiplier(header.get("physical_dimension")) == 1.0,
            }
            rows.append(row)
            if is_bad:
                self.issue("warning", subject_id, source_path, "raw_hard_bad_channel",
                           f"Hard bad channel detected before filtering: {row['bad_reason']}",
                           "Keep the channel slot, propagate the bad mask, and zero the cached data.",
                           header["channel_name"])
        return bad_mask, rows

    def _process_one(self, rec: dict[str, str], recording_id: str) -> None:
        subject_id = rec["subject_id"]
        source_path = rec["source_path"]
        raw_path = self.data_dir / Path(source_path)
        headers = self.headers[source_path]
        manifest = self.manifest[subject_id]
        started = now_iso()
        summary: dict[str, Any] = {
            "subject_id": subject_id, "source_path": source_path, "recording_id": recording_id,
            "run_mode": self.args.run_mode,
            "processing_started": started, "processing_success": False,
            "source_samples": rec.get("n_samples", ""),
            "source_duration_seconds": rec.get("duration_seconds", ""),
            "resampled_samples": "", "possible_complete_windows": 0,
            "emitted_windows": 0, "smoke_truncated": False,
            "dropped_tail_samples": "", "dropped_tail_seconds": "",
            "n_raw_samples": rec.get("n_samples", ""), "n_resampled_samples": "",
            "complete_windows": 0, "qc_pass_windows": 0, "qc_fail_windows": 0,
            "recording_bad_channel_count": 0, "recording_bad_channels": "",
            "window_new_bad_channels": "", "mean_clip_fraction": "", "max_clip_fraction": "",
            "source_reference": "unknown", "rereference_applied": False,
            "processing_reference": "unchanged", "montage": "unknown",
            "source_header_unit": "µV", "reader_output_unit": "V",
            "reader_to_uv_multiplier": READER_TO_UV_MULTIPLIER,
            "unit_conversion_verified": True, "error": "",
        }
        try:
            raw = mne.io.read_raw_bdf(raw_path, preload=False, verbose="ERROR")
            if len(raw.ch_names) != 64 or raw.ch_names != [h["channel_name"] for h in headers]:
                raise RuntimeError("Raw channel layout no longer matches the validated 64-channel inventory")
            raw_bad, bad_rows = self._scan_raw_qc(raw, headers, subject_id, source_path)
            self.bad_rows.extend(bad_rows)
            detected = {(subject_id, headers[i]["channel_name"]) for i in np.flatnonzero(raw_bad)}
            for expected in sorted(KNOWN_BAD_CHANNELS):
                if expected[0] == subject_id and expected not in detected:
                    raise RuntimeError(f"Known audit bad channel was not automatically detected: {expected}")
            summary["recording_bad_channel_count"] = int(raw_bad.sum())
            summary["recording_bad_channels"] = ";".join(headers[i]["channel_name"] for i in np.flatnonzero(raw_bad))
            self.subject_stats[subject_id]["recording_bad_channels"].update(
                headers[i]["channel_name"] for i in np.flatnonzero(raw_bad)
            )

            good_idx = np.flatnonzero(~raw_bad)
            if good_idx.size == 0:
                raise RuntimeError("All channels failed recording-level hard QC")
            good_uv = convert_mne_v_to_uv(raw.get_data(picks=good_idx))
            del raw
            good_uv = notch_filter(
                good_uv, Fs=float(self.exp["sampling_rate_hz"]), freqs=[float(self.pre["notch_hz"])],
                filter_length="auto", trans_bandwidth=1.0, method="fir", copy=True,
                phase="zero-double", fir_window="hamming", fir_design="firwin",
                pad="reflect_limited", verbose="ERROR",
            )
            good_uv = filter_data(
                good_uv, sfreq=float(self.exp["sampling_rate_hz"]),
                l_freq=float(self.pre["bandpass_low"]), h_freq=float(self.pre["bandpass_high"]),
                filter_length="auto", l_trans_bandwidth="auto", h_trans_bandwidth="auto",
                method="fir", copy=False, phase="zero-double", fir_window="hamming",
                fir_design="firwin", pad="reflect_limited", verbose="ERROR",
            )
            good_uv = resample(
                good_uv, up=float(self.pre["target_sfreq"]), down=float(self.exp["sampling_rate_hz"]),
                axis=-1, window="auto", pad="reflect", npad=100, method="polyphase", verbose="ERROR",
            )
            n_resampled = good_uv.shape[1]
            processed = np.zeros((64, n_resampled), dtype=np.float64)
            processed[good_idx] = good_uv
            del good_uv
            summary["n_resampled_samples"] = n_resampled
            summary["resampled_samples"] = n_resampled

            window_samples = int(self.pre["segment_samples"])
            hop_samples = int(round(float(self.pre["hop_seconds"]) * float(self.pre["target_sfreq"])))
            possible_slices = window_slices(n_resampled, window_samples, hop_samples)
            slices, smoke_truncated = limit_window_slices(
                possible_slices, self.args.smoke, self.args.smoke_max_windows_per_recording,
            )
            dropped_samples, dropped_seconds = dropped_tail(n_resampled, window_samples, hop_samples, float(self.pre["target_sfreq"]))
            raw_n = int(rec["n_samples"])
            source_used = min(raw_n, len(possible_slices) * int(round(float(self.exp["sampling_rate_hz"]) * float(self.pre["window_seconds"]))))
            source_dropped = raw_n - source_used
            summary["possible_complete_windows"] = len(possible_slices)
            summary["emitted_windows"] = len(slices)
            summary["smoke_truncated"] = smoke_truncated
            summary["dropped_tail_samples"] = dropped_samples
            summary["dropped_tail_seconds"] = dropped_seconds
            self.tail_rows.append({
                "subject_id": subject_id, "source_path": source_path, "recording_id": recording_id,
                "raw_samples": raw_n, "resampled_samples": n_resampled,
                "possible_complete_windows": len(possible_slices), "emitted_windows": len(slices),
                "smoke_truncated": smoke_truncated,
                "smoke_omitted_complete_windows": len(possible_slices) - len(slices),
                "complete_windows": len(possible_slices), "dropped_resampled_samples": dropped_samples,
                "dropped_source_samples": source_dropped, "dropped_tail_seconds": dropped_seconds,
            })
            self.subject_stats[subject_id]["dropped_tail_seconds"] += dropped_seconds

            cache_rows: list[dict[str, Any]] = []
            recording_clip: list[float] = []
            window_new_names: set[str] = set()
            channel_names = [h["channel_name"] for h in headers]
            for window_index, (start, stop) in enumerate(slices):
                sample_id = make_sample_id(subject_id, recording_id, window_index)
                window = processed[:, start:stop].copy()
                window_new_bad, window_reasons, preprocessing_finite = identify_window_bad_channels(window, raw_bad)
                combined_bad = merge_bad_channel_masks(raw_bad, window_new_bad)
                qc_pass, qc_reasons = evaluate_window_qc(
                    window, combined_bad, (64, window_samples), preprocessing_success=preprocessing_finite,
                )
                for idx in np.flatnonzero(window_new_bad):
                    window_new_names.add(channel_names[idx])
                    self.issue(
                        "error" if "window_nonfinite_after_preprocessing" in window_reasons[idx] else "warning",
                        subject_id, source_path, "window_hard_bad_channel",
                        ";".join(window_reasons[idx]),
                        "Keep the slot zeroed and inspect this channel in later statistical QC.",
                        channel_names[idx], sample_id,
                    )
                zeroed, channel_mask = zero_bad_channels(window, combined_bad)
                good_values = zeroed[~combined_bad]
                if good_values.size:
                    absmax = float(np.max(np.abs(good_values)))
                    positive_clip = float(np.mean(good_values >= 200.0))
                    negative_clip = float(np.mean(good_values <= -200.0))
                    clip_fraction = float(np.mean(np.abs(good_values) >= 200.0))
                else:
                    absmax = 0.0
                    positive_clip = negative_clip = clip_fraction = 0.0
                normalized = normalize_clip_divide_200(zeroed)
                normalized[combined_bad] = 0.0
                if normalized.shape != (64, window_samples) or normalized.dtype != np.float32:
                    qc_pass = False
                    qc_reasons.append("normalized_shape_or_dtype_invalid")
                if not np.isfinite(normalized).all() or normalized.min() < -1 or normalized.max() > 1:
                    qc_pass = False
                    qc_reasons.append("normalized_range_or_finiteness_invalid")
                if combined_bad.any() and not np.all(normalized[combined_bad] == 0.0):
                    qc_pass = False
                    qc_reasons.append("bad_channel_zeroing_failed")
                for idx in np.flatnonzero(window_new_bad):
                    self.bad_rows.append({
                        "subject_id": subject_id, "source_path": source_path,
                        "channel_index": int(idx), "channel_name": channel_names[idx],
                        "scope": "window", "bad_reason": ";".join(window_reasons[idx]),
                        "sample_id": sample_id, "affected_emitted_windows": 1,
                        "data_zeroed": bool(np.all(normalized[idx] == 0.0)),
                        "mask_verified": bool(combined_bad[idx]), "is_bad": True,
                    })
                if qc_reasons:
                    self.issue("error", subject_id, source_path, "window_qc_failed",
                               ";".join(dict.fromkeys(qc_reasons)),
                               "Do not use this window for downstream labels or modeling.", sample_id=sample_id)

                start_time = start / float(self.pre["target_sfreq"])
                source_start = int(round(start_time * float(self.exp["sampling_rate_hz"])))
                source_stop = int(round((start_time + float(self.pre["window_seconds"])) * float(self.exp["sampling_rate_hz"])))
                source_stop = min(source_stop, raw_n)
                status = ["bad" if flag else "good" for flag in combined_bad]
                row = {
                    "sample_id": sample_id, "subject_id": subject_id, "modality": "eeg",
                    "source_path": source_path, "recording_id": recording_id,
                    "window_index": window_index, "start_time": start_time,
                    "duration": float(self.pre["window_seconds"]),
                    "source_start_sample": source_start, "source_stop_sample": source_stop,
                    "data": normalized.reshape(-1), "shape": [64, window_samples],
                    "original_shape": [64, window_samples], "valid_length": int(normalized.size),
                    "channel_names": channel_names, "channel_mask": channel_mask,
                    "bad_channel_mask": combined_bad, "channel_status": status,
                    "qc_pass": bool(qc_pass), "bad_channel_count": int(combined_bad.sum()),
                    "age": int(manifest["age"]) if manifest.get("age", "").strip() else None,
                    "sex": manifest["sex"], "original_directory_name": manifest["original_directory_name"],
                    "clip_fraction": np.float32(clip_fraction),
                    "positive_clip_fraction": np.float32(positive_clip),
                    "negative_clip_fraction": np.float32(negative_clip),
                    "pre_normalization_abs_max_uv": np.float32(absmax),
                    "source_sampling_rate": np.float32(self.exp["sampling_rate_hz"]),
                    "target_sampling_rate": np.float32(self.pre["target_sfreq"]),
                    "preprocessing_config_hash": self.config_hash,
                }
                cache_rows.append(row)
                recording_clip.append(clip_fraction)
                self.window_rows.append({
                    "sample_id": sample_id, "subject_id": subject_id, "source_path": source_path,
                    "recording_id": recording_id, "window_index": window_index,
                    "start_time": start_time, "duration": float(self.pre["window_seconds"]),
                    "start_sample_resampled": start, "stop_sample_resampled": stop,
                    "start_time_seconds": start_time, "duration_seconds": float(self.pre["window_seconds"]),
                    "source_start_sample_500hz": source_start, "source_stop_sample_500hz": source_stop,
                    "shape": "[64,2000]", "valid_length": int(normalized.size),
                    "qc_pass": bool(qc_pass), "qc_reasons": ";".join(dict.fromkeys(qc_reasons)),
                    "bad_channel_count": int(combined_bad.sum()),
                    "bad_channel_names": ";".join(channel_names[i] for i in np.flatnonzero(combined_bad)),
                    "recording_bad_channels": ";".join(channel_names[i] for i in np.flatnonzero(raw_bad)),
                    "window_new_bad_channels": ";".join(channel_names[i] for i in np.flatnonzero(window_new_bad)),
                    "clip_fraction": clip_fraction, "positive_clip_fraction": positive_clip,
                    "negative_clip_fraction": negative_clip, "pre_normalization_abs_max_uv": absmax,
                })
                stats = self.subject_stats[subject_id]
                stats["window_count"] += 1
                stats["qc_pass" if qc_pass else "qc_fail"] += 1
                stats["clip_fractions"].append(clip_fraction)
                stats["window_new_bad_channels"].update(channel_names[i] for i in np.flatnonzero(window_new_bad))

            if cache_rows:
                _write_lance_batch(self.lance_path, cache_rows, self._first_lance_write)
                self._first_lance_write = False
            summary["processing_success"] = True
            summary["complete_windows"] = len(cache_rows)
            summary["qc_pass_windows"] = sum(bool(row["qc_pass"]) for row in cache_rows)
            summary["qc_fail_windows"] = len(cache_rows) - summary["qc_pass_windows"]
            summary["window_new_bad_channels"] = ";".join(sorted(window_new_names))
            summary["mean_clip_fraction"] = float(np.mean(recording_clip)) if recording_clip else 0.0
            summary["max_clip_fraction"] = max(recording_clip, default=0.0)
            for bad_row in self.bad_rows:
                if (
                    bad_row.get("subject_id") == subject_id
                    and bad_row.get("source_path") == source_path
                    and bad_row.get("is_bad")
                    and not bad_row.get("scope")
                ):
                    bad_row["scope"] = "recording"
                    bad_row["affected_emitted_windows"] = len(cache_rows)
                    bad_row["data_zeroed"] = True
                    bad_row["mask_verified"] = True
        except Exception as exc:
            summary["error"] = f"{type(exc).__name__}: {exc}"
            self.errors.append(f"[{now_iso()}] {subject_id} {source_path}\n{traceback.format_exc()}\n")
            self.issue("error", subject_id, source_path, "recording_processing_failed",
                       summary["error"], "Resolve the error before label assignment or modeling.")
        finally:
            summary["processing_finished"] = now_iso()
            self.recording_rows.append(summary)

    def run(self) -> dict[str, Any]:
        self.validate_inputs()
        recording_counts: dict[str, int] = defaultdict(int)
        for rec in self.recordings:
            recording_counts[rec["subject_id"]] += 1
            recording_id = f"recording{recording_counts[rec['subject_id']]:02d}"
            self._process_one(rec, recording_id)
        if self._first_lance_write:
            raise RuntimeError("No complete windows were written to Lance")
        return validate_lance_roundtrip(self.lance_path, len(self.window_rows))

    def write_outputs(self, roundtrip: dict[str, Any], raw_before: dict[str, Any], raw_after: dict[str, Any]) -> dict[str, Any]:
        raw_unchanged = raw_before == raw_after
        if not raw_unchanged:
            self.issue("error", "", "", "raw_source_changed",
                       "Raw BDF file size or modification time changed during processing.",
                       "Stop and investigate source integrity before any downstream work.")
        fields_recording = [
            "subject_id", "source_path", "run_mode", "recording_id",
            "source_samples", "source_duration_seconds", "resampled_samples",
            "possible_complete_windows", "emitted_windows", "smoke_truncated",
            "dropped_tail_samples", "dropped_tail_seconds", "recording_bad_channels",
            "qc_pass_windows", "qc_fail_windows", "mean_clip_fraction", "max_clip_fraction",
            "processing_success", "processing_started", "processing_finished",
            "recording_bad_channel_count", "window_new_bad_channels",
            "source_reference", "rereference_applied", "processing_reference", "montage",
            "source_header_unit", "reader_output_unit", "reader_to_uv_multiplier",
            "unit_conversion_verified", "error",
        ]
        fields_window = [
            "sample_id", "subject_id", "source_path", "recording_id", "window_index",
            "start_time", "duration", "shape", "qc_pass", "bad_channel_count", "bad_channel_names",
            "clip_fraction", "positive_clip_fraction", "negative_clip_fraction",
            "pre_normalization_abs_max_uv",
            "start_sample_resampled", "stop_sample_resampled", "start_time_seconds", "duration_seconds",
            "source_start_sample_500hz", "source_stop_sample_500hz", "valid_length",
            "qc_reasons", "recording_bad_channels", "window_new_bad_channels",
        ]
        fields_bad = [
            "subject_id", "source_path", "channel_index", "channel_name", "scope", "bad_reason",
            "affected_emitted_windows", "data_zeroed", "mask_verified", "sample_id",
            "is_bad",
            "raw_min_uv", "raw_max_uv", "raw_mean_uv", "raw_std_uv", "physical_min_uv",
            "physical_max_uv", "source_header_unit", "reader_output_unit", "reader_to_uv_multiplier",
            "unit_conversion_verified",
        ]
        fields_tail = [
            "subject_id", "source_path", "recording_id", "raw_samples", "resampled_samples",
            "possible_complete_windows", "emitted_windows", "smoke_truncated",
            "smoke_omitted_complete_windows", "complete_windows", "dropped_resampled_samples",
            "dropped_source_samples", "dropped_tail_seconds",
        ]
        fields_issue = [
            "severity", "stage", "subject_id", "source_path", "sample_id", "channel_name",
            "issue_type", "description", "action",
        ]
        bad_output_rows = [row for row in self.bad_rows if row.get("is_bad")]
        _write_csv(self.output_dir / "recording_processing_summary.csv", self.recording_rows, fields_recording)
        _write_csv(self.output_dir / "window_inventory.csv", self.window_rows, fields_window)
        _write_csv(self.output_dir / "bad_channel_inventory.csv", bad_output_rows, fields_bad)
        _write_csv(self.output_dir / "dropped_tails.csv", self.tail_rows, fields_tail)
        _write_csv(self.output_dir / "preprocessing_issues.csv", self.issues, fields_issue)
        error_header = "timestamp\tstage\tsubject_id\tsource_path\terror\n"
        (self.output_dir / "preprocessing_errors.log").write_text(
            error_header + "".join(self.errors), encoding="utf-8",
        )

        shutil.copy2(self.args.config, self.output_dir / "resolved_preprocessing_config.toml")
        subject_summary = {}
        for subject_id in sorted(self.selected_subjects, key=lambda value: int(value[3:])):
            stats = self.subject_stats[subject_id]
            clips = stats["clip_fractions"]
            subject_summary[subject_id] = {
                "complete_windows": stats["window_count"],
                "dropped_tail_seconds": stats["dropped_tail_seconds"],
                "qc_pass": stats["qc_pass"], "qc_fail": stats["qc_fail"],
                "recording_bad_channels": sorted(stats["recording_bad_channels"]),
                "window_new_bad_channels": sorted(stats["window_new_bad_channels"]),
                "mean_clip_fraction": float(np.mean(clips)) if clips else 0.0,
                "max_clip_fraction": max(clips, default=0.0),
            }
        recording_bad_count = sum(row.get("scope") == "recording" for row in bad_output_rows)
        window_bad_count = sum(row.get("scope") == "window" for row in bad_output_rows)
        lance_roundtrip_pass = all([
            roundtrip["row_count"] == len(self.window_rows), roundtrip["sample_id_unique"],
            roundtrip["schema_matches"], roundtrip["all_shapes_valid"],
            roundtrip["all_values_finite"], roundtrip["all_values_in_range"],
            roundtrip["bad_slots_zero"], roundtrip["forbidden_fields_absent"],
        ])
        summary = {
            "run_mode": self.args.run_mode,
            "selected_subject_count": len(self.selected_subjects),
            "selected_recording_count": len(self.recordings),
            "emitted_window_count": len(self.window_rows),
            "qc_pass_count": sum(_parse_bool(r["qc_pass"]) for r in self.window_rows),
            "qc_fail_count": sum(not _parse_bool(r["qc_pass"]) for r in self.window_rows),
            "recording_bad_channel_count": recording_bad_count,
            "window_bad_channel_count": window_bad_count,
            "unique_sample_id_count": roundtrip["unique_sample_id_count"],
            "all_shapes_valid": roundtrip["all_shapes_valid"],
            "all_values_finite": roundtrip["all_values_finite"],
            "all_values_in_range": roundtrip["all_values_in_range"],
            "bad_channels_zeroed": roundtrip["bad_slots_zero"],
            "contains_label": roundtrip["contains_label"],
            "contains_label_name": roundtrip["contains_label_name"],
            "contains_split": roundtrip["contains_split"],
            "lance_roundtrip_pass": lance_roundtrip_pass,
            "source_files_unchanged": raw_unchanged,
            "generated_at": now_iso(), "script_version": SCRIPT_VERSION,
            "subjects": len(self.selected_subjects), "recordings": len(self.recordings),
            "recordings_processed_successfully": sum(_parse_bool(r["processing_success"]) for r in self.recording_rows),
            "complete_windows": len(self.window_rows),
            "qc_pass": sum(_parse_bool(r["qc_pass"]) for r in self.window_rows),
            "qc_fail": sum(not _parse_bool(r["qc_pass"]) for r in self.window_rows),
            "recording_level_bad_channels": [
                {"subject_id": r["subject_id"], "channel_name": r["channel_name"], "reason": r["bad_reason"]}
                for r in bad_output_rows if r.get("scope") == "recording"
            ],
            "subject_summary": subject_summary,
            "roundtrip_validation": roundtrip,
            "raw_bdf_unchanged": raw_unchanged,
            "issue_counts": dict(Counter(issue["severity"] for issue in self.issues)),
            "blocking_issue_count": sum(issue["severity"] == "error" for issue in self.issues),
        }
        _json_dump(self.output_dir / "preprocessing_summary.json", summary)

        project_root = Path(__file__).resolve().parents[1]
        script_path = Path(__file__).resolve()
        created_at = now_iso()
        provenance = {
            "run_mode": self.args.run_mode, "created_at": created_at,
            "command": self.args.actual_command,
            "python_version": platform.python_version(), "mne_version": mne.__version__,
            "numpy_version": np.__version__, "scipy_version": scipy.__version__,
            "pyarrow_version": pa.__version__, "lance_version": lance.__version__,
            "script_path": str(script_path), "script_sha256": hashlib.sha256(script_path.read_bytes()).hexdigest(),
            "config_path": str(self.args.config.resolve()), "config_sha256": self.config_hash,
            "subject_manifest_path": str(self.args.subject_manifest.resolve()),
            "recording_validation_path": str(self.args.recording_validation.resolve()),
            "channel_header_inventory_path": str(self.args.channel_header_inventory.resolve()),
            "data_dir": str(self.data_dir), "output_dir": str(self.output_dir),
            "selected_subjects": self.selected_subjects,
            "max_windows_per_recording": self.args.smoke_max_windows_per_recording if self.args.smoke else None,
            "filter_parameters": {
                "notch": {**FILTER_PARAMETERS["notch"], "frequency_hz": float(self.pre["notch_hz"])},
                "bandpass": {
                    **FILTER_PARAMETERS["bandpass"],
                    "low_hz": float(self.pre["bandpass_low"]),
                    "high_hz": float(self.pre["bandpass_high"]),
                },
            },
            "resample_parameters": {
                **FILTER_PARAMETERS["resample"],
                "source_sampling_rate_hz": float(self.exp["sampling_rate_hz"]),
                "target_sampling_rate_hz": float(self.pre["target_sfreq"]),
            },
            "normalization": {
                "method": self.pre["normalization"], "detail": self.pre["normalization_detail"],
                "clip_fraction_denominator": "all samples from channels not marked bad in that window",
            },
            "source_reference": "unknown", "processing_reference": "unchanged",
            "rereference_applied": False, "montage": "unknown",
            "run_started": self.args.run_started, "run_finished": created_at,
            "script_version": SCRIPT_VERSION,
            "versions": {
                "python": platform.python_version(), "mne": mne.__version__, "numpy": np.__version__,
                "scipy": scipy.__version__, "lance": lance.__version__, "pyarrow": pa.__version__,
            },
            "input_config_sha256": self.config_hash,
            "raw_data_directory": str(self.data_dir),
            "subject_manifest": str(self.args.subject_manifest.resolve()),
            "recording_validation": str(self.args.recording_validation.resolve()),
            "channel_header_inventory": str(self.args.channel_header_inventory.resolve()),
            "output_directory": str(self.output_dir),
            "filter_and_resample": {
                **FILTER_PARAMETERS,
                "notch_frequency_hz": float(self.pre["notch_hz"]),
                "bandpass_hz": [float(self.pre["bandpass_low"]), float(self.pre["bandpass_high"])],
                "source_sampling_rate_hz": float(self.exp["sampling_rate_hz"]),
                "target_sampling_rate_hz": float(self.pre["target_sfreq"]),
            },
            "windowing": {
                "window_seconds": float(self.pre["window_seconds"]),
                "hop_seconds": float(self.pre["hop_seconds"]),
                "segment_samples": int(self.pre["segment_samples"]),
                "drop_incomplete_tail": bool(self.windowing["drop_incomplete_tail"]),
                "padding": bool(self.windowing["padding"]),
            },
            "unit_policy": {
                "source_header_unit": "µV", "reader_output_unit": "V",
                "reader_to_uv_multiplier": READER_TO_UV_MULTIPLIER,
                "cross_checked_against_channel_header_inventory": True,
            },
            "reference_and_montage": {
                "source_reference": "unknown", "rereference_applied": False,
                "processing_reference": "unchanged", "montage": "unknown",
            },
            "raw_file_snapshot_before": raw_before,
            "raw_file_snapshot_after": raw_after,
            "raw_bdf_unchanged": raw_unchanged,
            "actual_command": self.args.actual_command,
            "roundtrip_validation": roundtrip,
        }
        provenance.update(_git_metadata(project_root))
        _json_dump(self.output_dir / "processing_provenance.json", provenance)
        if self.args.smoke:
            (self.output_dir / "full_run_command.ps1").write_text(self.args.full_run_command, encoding="utf-8")
            self.args.full_run_command_path.write_text(self.args.full_run_command, encoding="utf-8")
        self._write_report(summary, provenance)
        return summary

    def _write_report(self, summary: dict[str, Any], provenance: dict[str, Any]) -> None:
        lines = [
            "# 华大静息态 EEG 公共窗口预处理报告", "",
            f"- 运行模式：{summary['run_mode']}",
            f"- 实际选择被试：{', '.join(provenance['selected_subjects'])}",
            f"- 运行时间：{provenance['run_started']} 至 {provenance['run_finished']}",
            f"- 输入 recording：{summary['recordings']}，成功处理：{summary['recordings_processed_successfully']}",
            f"- 完整窗口：{summary['complete_windows']}，QC 通过：{summary['qc_pass']}，QC 失败：{summary['qc_fail']}",
            f"- 原始 BDF 大小和修改时间保持不变：{'是' if summary['raw_bdf_unchanged'] else '否'}", "",
            "## 运行范围", "",
            (
                f"本次是 smoke 运行。仅在打开 BDF 前选择 {', '.join(provenance['selected_subjects'])}，"
                f"每条 recording 最多写出 {provenance['max_windows_per_recording']} 个完整窗口。"
                "未写出的正常完整窗口属于 smoke 截断，不计入 dropped tail。本次没有运行 8 名被试的全量处理。"
                if summary["run_mode"] == "smoke"
                else "本次是 full 运行，处理输入清单中的全部 recording。"
            ), "",
            "## 固定处理顺序", "",
            "读取 64 个 EEG 数据通道 → MNE V 转 µV → 原始硬性 QC → 非坏通道 50 Hz notch → 0.3–75 Hz bandpass → 200 Hz polyphase 重采样 → 10 秒无重叠切窗 → 坏道置零 → clip/divide 归一化。", "",
            "未应用重新参考、montage 推断、空间插值、ICA、经验振幅/PSD 阈值、标签映射或数据集划分。source_reference=unknown，rereference_applied=false，processing_reference=unchanged，montage=unknown。", "",
            "## 被试汇总", "",
            "| 被试 | 完整窗 | 丢弃尾部秒数 | QC pass | QC fail | recording 坏道 | 窗口新增坏道 | 平均 clip | 最大 clip |",
            "|---|---:|---:|---:|---:|---|---|---:|---:|",
        ]
        for subject_id, row in summary["subject_summary"].items():
            lines.append(
                f"| {subject_id} | {row['complete_windows']} | {row['dropped_tail_seconds']:.3f} | "
                f"{row['qc_pass']} | {row['qc_fail']} | {', '.join(row['recording_bad_channels']) or '无'} | "
                f"{', '.join(row['window_new_bad_channels']) or '无'} | {row['mean_clip_fraction']:.6f} | {row['max_clip_fraction']:.6f} |"
            )
        lines.extend(["", "## Lance 验证", ""])
        rt = summary["roundtrip_validation"]
        lines.extend([
            f"- schema 匹配：{rt['schema_matches']}",
            f"- round-trip 行数：{rt['row_count']}",
            f"- sample_id 全部唯一：{rt['sample_id_unique']}",
            f"- 坏道槽位全部严格为零：{rt['bad_slots_zero']}",
            f"- 数据范围：[{rt['normalized_min']}, {rt['normalized_max']}]",
            f"- 不含 label、label_name、split：{rt['forbidden_fields_absent']}", "",
            f"- Lance schema：data={rt['data_type']}，channel_mask={rt['channel_mask_type']}，bad_channel_mask={rt['bad_mask_type']}，age nullable={rt['age_nullable']}",
            f"- round-trip 总体验收：{summary['lance_roundtrip_pass']}", "",
            "## 坏道结果", "",
            f"recording 级坏道 {summary['recording_bad_channel_count']} 个，窗口级坏道记录 {summary['window_bad_channel_count']} 个。坏道槽位 mask 与严格置零验证：{summary['bad_channels_zeroed']}。", "",
            "## 问题与进入下一阶段判断", "",
            f"记录到 error {summary['issue_counts'].get('error', 0)} 条、warning {summary['issue_counts'].get('warning', 0)} 条。原始硬性坏道以 mask 表达且缓存中置零；它们本身不阻止其他通道窗口进入标签分配。只有 preprocessing_issues.csv 中的 error 会阻止相应 recording/window 进入下一阶段。", "",
            "## 实际执行命令", "", "```powershell", provenance["actual_command"], "```", "",
            "## 后续手动全量命令（本次未执行）", "", "```powershell",
            self.args.full_run_command.rstrip(), "```", "",
        ])
        (self.output_dir / "preprocessing_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--subject-manifest", type=Path, required=True)
    parser.add_argument("--recording-validation", type=Path, required=True)
    parser.add_argument("--channel-header-inventory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-seconds", type=float, default=30.0)
    parser.add_argument("--smoke", action="store_true", help="Run only the selected smoke subjects")
    parser.add_argument(
        "--smoke-subjects", nargs="+", default=None,
        help="Smoke subjects as comma-separated and/or space-separated IDs (default: SUB1,SUB5)",
    )
    parser.add_argument("--smoke-max-windows-per-recording", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    cli_tokens = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args(cli_tokens)
    if args.chunk_seconds <= 0:
        parser.error("--chunk-seconds must be > 0")
    if args.smoke:
        try:
            args.smoke_subjects = parse_smoke_subjects(args.smoke_subjects)
        except ValueError as exc:
            parser.error(str(exc))
        if args.smoke_max_windows_per_recording is None:
            args.smoke_max_windows_per_recording = 2
        if args.smoke_max_windows_per_recording <= 0:
            parser.error("--smoke-max-windows-per-recording must be > 0")
    else:
        if args.smoke_subjects is not None or args.smoke_max_windows_per_recording is not None:
            parser.error("Smoke selection arguments require --smoke")
        args.smoke_subjects = []
    args.run_mode = "smoke" if args.smoke else "full"
    args.run_started = now_iso()
    args.actual_command = subprocess.list2cmdline([sys.executable, str(Path(__file__).resolve()), *cli_tokens])
    project_root = Path(__file__).resolve().parents[1]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_output = project_root / "preprocessed_common" / f"{stamp}_manual_full"
    serial = 1
    while full_output.exists():
        full_output = project_root / "preprocessed_common" / f"{stamp}_manual_full_{serial:02d}"
        serial += 1
    args.full_run_output_dir = full_output
    args.full_run_command = build_full_run_command(args, full_output)
    args.full_run_command_path = project_root / "full_run_command.ps1"
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        project_root = Path(__file__).resolve().parents[1]
        if args.smoke:
            validate_smoke_output_dir(args.output_dir, project_root)
        _prepare_output_dir(args.output_dir, args.overwrite, args.data_dir)
        pipeline = Pipeline(args)
        selected_source_paths = [row["source_path"] for row in pipeline.recordings]
        raw_before = _snapshot_raw_files(args.data_dir, selected_source_paths)
        roundtrip = pipeline.run()
        raw_after = _snapshot_raw_files(args.data_dir, selected_source_paths)
        summary = pipeline.write_outputs(roundtrip, raw_before, raw_after)
        print(json.dumps({"output_dir": str(args.output_dir.resolve()), **summary}, ensure_ascii=False, indent=2, default=str))
        return 0 if summary["blocking_issue_count"] == 0 and summary["raw_bdf_unchanged"] else 2
    except Exception:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        error_path = args.output_dir / "preprocessing_errors.log"
        if not error_path.exists() or error_path.stat().st_size == 0:
            error_path.write_text("timestamp\tstage\tsubject_id\tsource_path\terror\n", encoding="utf-8")
        with error_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{now_iso()}\tfatal\t\t\t{traceback.format_exc()}\n")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
