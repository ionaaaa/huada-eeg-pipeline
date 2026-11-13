#!/usr/bin/env python3
"""MIPDB 20-35 age raw-to-Lance preprocessing adapter.

The adapter discovers subjects and BrainVision recordings at run time. It
keeps the native E1-E128 EEG order, excludes all auxiliary/event channels,
does not rereference or infer task semantics, and reuses the validated common
signal-processing/QC/Lance helpers used by the Huada and Dortmund pipelines.
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
import tomllib
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import lance
import mne
import numpy as np
import pyarrow as pa
import scipy

from preprocess_huada_common_windows import (
    FILTER_PARAMETERS,
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
from validate_huada_raw_bdf import unit_to_uv_multiplier


SCRIPT_VERSION = "1.0.0"
DEFAULT_SOURCE_URI = "cos://jianyu-1442740494/huada/data/MIPDB/mipdb_download/nm000153_py"
SOURCE_SFREQ = 500.0
TARGET_SFREQ = 200.0
N_EEG_CHANNELS = 128
WINDOW_SAMPLES = 2000
DATA_LENGTH = N_EEG_CHANNELS * WINDOW_SAMPLES
CORE_CHANNELS = [f"E{index}" for index in range(1, N_EEG_CHANNELS + 1)]
SOURCE_REFERENCE = "unknown"
PROCESSING_REFERENCE = "unchanged"
TASK_SEMANTICS = "unknown"
BLOCK_RE = re.compile(r"^(?P<subject>sub-[^_]+)_task-(?P<block>block\d+)_eeg\.vhdr$", re.I)


CSV_FIELDS: dict[str, list[str]] = {
    "subject_manifest.csv": [
        "subject_id", "age", "label", "age_unit", "age_source", "sex_raw",
        "recording_count", "selected",
    ],
    "recording_manifest.csv": [
        "subject_id", "recording_id", "block_id", "source_path", "binary_path",
        "marker_path", "sidecar_path", "channels_path", "events_path", "age",
        "header_signature", "task_semantics", "resting_state_confirmed",
        "block01_candidate", "aux_channel_count", "aux_channel_names",
        "aux_channel_types", "power_line_frequency_sidecar",
    ],
    "recording_validation.csv": [
        "subject_id", "recording_id", "block_id", "source_path",
        "header_read_success", "fully_readable", "sampling_rate_hz",
        "n_channels_total", "n_eeg_channels", "n_aux_channels", "n_samples_header",
        "n_samples_read", "channel_order_matches", "unit_conversion_verified",
        "has_nan", "has_inf", "recording_bad_channel_count",
        "near_zero_variance_channel_count", "dc_offset_channel_count",
        "extreme_amplitude_channel_count", "validation_pass", "error",
    ],
    "recording_processing_summary.csv": [
        "subject_id", "recording_id", "block_id", "source_path", "run_mode",
        "source_samples", "source_duration_seconds", "resampled_samples",
        "theoretical_complete_windows", "emitted_windows", "written_windows",
        "smoke_truncated_windows", "real_dropped_tail_samples",
        "real_dropped_tail_seconds", "recording_bad_channels", "qc_pass_windows",
        "qc_fail_windows", "mean_clip_fraction", "max_clip_fraction",
        "processing_success", "error",
    ],
    "channel_statistics.csv": [
        "subject_id", "recording_id", "block_id", "source_path", "channel_index",
        "channel_name", "channel_type", "unit", "raw_min_uv", "raw_max_uv",
        "raw_mean_uv", "raw_std_uv", "post_filter_min_uv", "post_filter_max_uv",
        "post_filter_mean_uv", "post_filter_std_uv", "near_zero_variance_flag",
        "dc_offset_flag", "extreme_amplitude_flag", "is_bad", "bad_reason",
    ],
    "aux_channel_inventory.csv": [
        "subject_id", "recording_id", "block_id", "source_path", "source_index",
        "channel_name", "channel_type", "unit", "excluded_from_lance",
    ],
    "bad_channel_inventory.csv": [
        "subject_id", "recording_id", "block_id", "source_path", "sample_id",
        "channel_index", "channel_name", "scope", "bad_reason", "affected_windows",
        "data_zeroed", "mask_verified",
    ],
    "window_inventory.csv": [
        "sample_id", "subject_id", "recording_id", "block_id", "source_path",
        "age", "header_signature", "task_semantics", "resting_state_confirmed",
        "block01_candidate", "window_index", "start_sample", "start_seconds",
        "target_start_sample", "duration", "shape", "dtype", "qc_pass",
        "qc_failure_reason", "clip_fraction", "positive_clip_fraction",
        "negative_clip_fraction", "bad_channel_count", "bad_channel_names",
        "written_to_lance",
    ],
    "dropped_tails.csv": [
        "subject_id", "recording_id", "block_id", "source_path", "source_samples",
        "resampled_samples", "theoretical_complete_windows", "emitted_windows",
        "smoke_truncated_windows", "real_dropped_tail_samples",
        "real_dropped_tail_seconds",
    ],
    "preprocessing_issues.csv": [
        "severity", "stage", "subject_id", "recording_id", "block_id",
        "source_path", "sample_id", "channel_name", "issue_type", "description",
        "action",
    ],
    "source_snapshot_before.csv": ["relative_path", "size_bytes", "mtime_ns"],
    "source_snapshot_after.csv": ["relative_path", "size_bytes", "mtime_ns"],
}


def normalize_subject(value: str) -> str:
    text = str(value).strip()
    return text if text.startswith("sub-") else f"sub-{text}"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def load_allowlist(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    subjects = {
        normalize_subject(line.split("#", 1)[0])
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.split("#", 1)[0].strip()
    }
    if not subjects:
        raise ValueError("subject allowlist is empty")
    return subjects


def parse_exact_age(value: Any) -> float:
    text = str(value).strip()
    if not re.fullmatch(r"\d+(?:\.\d+)?", text):
        raise ValueError(f"exact numeric age required, got {value!r}")
    return float(text)


def load_age_metadata(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_tsv(path) if path.suffix.lower() == ".tsv" else read_csv(path)
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        lowered = {str(key).lower(): value for key, value in row.items()}
        raw_id = lowered.get("participant_id") or lowered.get("id") or lowered.get("subject_id")
        raw_age = lowered.get("age")
        if not raw_id or raw_age in (None, "", "n/a", "na"):
            continue
        sid = normalize_subject(str(raw_id))
        result[sid] = {
            "age": parse_exact_age(raw_age),
            "sex_raw": str(lowered.get("sex", "")),
            "source": str(path.resolve()),
        }
    return result


def parse_recording_name(path: Path) -> dict[str, str]:
    match = BLOCK_RE.match(path.name)
    if not match:
        return {}
    return match.groupdict()


def load_mipdb_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    pre = config.get("preprocessing", {})
    windowing = config.get("windowing", {})
    source = config.get("source_expectations", {})
    line = config.get("line_frequency", {})
    required_pre = {
        "target_sfreq", "window_seconds", "hop_seconds", "segment_samples",
        "bandpass_low", "bandpass_high", "notch_hz", "unit_before_normalization",
        "normalization", "normalization_detail",
    }
    missing = required_pre - set(pre)
    if missing:
        raise ValueError(f"Missing preprocessing fields: {sorted(missing)}")
    if str(source.get("format", "")).casefold() != "brainvision":
        raise ValueError("source_expectations.format must be BrainVision")
    if float(source.get("sampling_rate_hz", 0)) != SOURCE_SFREQ:
        raise ValueError(f"source sampling rate must be {SOURCE_SFREQ}")
    if int(source.get("n_eeg_channels", 0)) != N_EEG_CHANNELS:
        raise ValueError(f"source n_eeg_channels must be {N_EEG_CHANNELS}")
    if float(pre["target_sfreq"]) != TARGET_SFREQ:
        raise ValueError(f"target_sfreq must be {TARGET_SFREQ}")
    if int(pre["segment_samples"]) != int(float(pre["target_sfreq"]) * float(pre["window_seconds"])):
        raise ValueError("segment_samples must equal target_sfreq * window_seconds")
    if float(pre["hop_seconds"]) <= 0 or float(pre["bandpass_low"]) <= 0:
        raise ValueError("hop_seconds and bandpass_low must be positive")
    if float(pre["bandpass_high"]) >= float(pre["target_sfreq"]) / 2:
        raise ValueError("bandpass_high must be below target Nyquist")
    notch = float(pre["notch_hz"])
    if notch not in {50.0, 60.0}:
        raise ValueError("notch_hz must be explicitly resolved to 50 or 60 Hz")
    if float(line.get("selected_hz", -1)) != notch:
        raise ValueError("line_frequency.selected_hz must equal preprocessing.notch_hz")
    if not str(line.get("decision", "")).strip():
        raise ValueError("line_frequency.decision must document the evidence source")
    if windowing.get("drop_incomplete_tail") is not True or windowing.get("padding") is not False:
        raise ValueError("MIPDB requires dropped incomplete tails and no padding")
    return config


def resolve_core_channels(
    names: Sequence[str], types: Sequence[str]
) -> tuple[list[int], list[int]]:
    if len(names) != len(types):
        raise RuntimeError("channel name/type length mismatch")
    if len(set(names)) != len(names):
        duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
        raise RuntimeError(f"duplicate source channel names: {duplicates}")
    missing = [name for name in CORE_CHANNELS if name not in names]
    if missing:
        raise RuntimeError(f"missing core EEG channels: {missing}")
    core_indices = [names.index(name) for name in CORE_CHANNELS]
    source_core_order = [name for name in names if name in set(CORE_CHANNELS)]
    if source_core_order != CORE_CHANNELS:
        raise RuntimeError("E1-E128 source order is not canonical")
    non_eeg = [CORE_CHANNELS[index] for index, raw_index in enumerate(core_indices) if types[raw_index] != "eeg"]
    if non_eeg:
        raise RuntimeError(f"core channels not typed EEG: {non_eeg}")
    aux_indices = [index for index in range(len(names)) if index not in set(core_indices)]
    return core_indices, aux_indices


def parse_power_line_frequency(value: Any) -> float | None:
    if value is None or str(value).strip().casefold() in {"", "n/a", "na", "none", "unknown"}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid PowerLineFrequency: {value!r}") from exc


def make_sample_id(subject_id: str, recording_id: str, window_index: int, source_start: int) -> str:
    return f"nm000153__{subject_id}__{recording_id}__w{window_index:05d}__s{source_start:09d}"


def mipdb_lance_schema() -> pa.Schema:
    return pa.schema([
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("subject_id", pa.string(), nullable=False),
        pa.field("recording_id", pa.string(), nullable=False),
        pa.field("block_id", pa.string(), nullable=False),
        pa.field("source_path", pa.string(), nullable=False),
        pa.field("age", pa.float32(), nullable=False),
        pa.field("label", pa.float32(), nullable=False),
        pa.field("task_type", pa.string(), nullable=False),
        pa.field("task_semantics", pa.string(), nullable=False),
        pa.field("resting_state_confirmed", pa.bool_(), nullable=False),
        pa.field("block01_candidate", pa.bool_(), nullable=False),
        pa.field("header_signature", pa.string(), nullable=False),
        pa.field("window_index", pa.int32(), nullable=False),
        pa.field("start_sample", pa.int64(), nullable=False),
        pa.field("start_seconds", pa.float64(), nullable=False),
        pa.field("target_start_sample", pa.int64(), nullable=False),
        pa.field("duration", pa.float64(), nullable=False),
        pa.field("modality", pa.string(), nullable=False),
        # Lance stable encoding currently panics for a >1 MiB fixed-size-list
        # cell (128 * 2000 float32). Keep an ordinary list and enforce the
        # exact length through valid_length plus round-trip validation below.
        pa.field("data", pa.list_(pa.float32()), nullable=False),
        pa.field("shape", pa.list_(pa.int32(), 2), nullable=False),
        pa.field("original_shape", pa.list_(pa.int32(), 2), nullable=False),
        pa.field("valid_length", pa.int64(), nullable=False),
        pa.field("channel_names", pa.list_(pa.string()), nullable=False),
        pa.field("channel_mask", pa.list_(pa.bool_(), N_EEG_CHANNELS), nullable=False),
        pa.field("bad_channel_mask", pa.list_(pa.bool_(), N_EEG_CHANNELS), nullable=False),
        pa.field("channel_status", pa.list_(pa.string()), nullable=False),
        pa.field("bad_channel_count", pa.int32(), nullable=False),
        pa.field("qc_pass", pa.bool_(), nullable=False),
        pa.field("clip_fraction", pa.float32(), nullable=False),
        pa.field("positive_clip_fraction", pa.float32(), nullable=False),
        pa.field("negative_clip_fraction", pa.float32(), nullable=False),
        pa.field("pre_normalization_abs_max_uv", pa.float32(), nullable=False),
        pa.field("source_sampling_rate", pa.float32(), nullable=False),
        pa.field("target_sampling_rate", pa.float32(), nullable=False),
        pa.field("source_reference", pa.string(), nullable=False),
        pa.field("processing_reference", pa.string(), nullable=False),
        pa.field("preprocessing_config_hash", pa.string(), nullable=False),
    ])


def validate_mipdb_lance(path: Path, expected_rows: int) -> dict[str, Any]:
    dataset = lance.dataset(path)
    if dataset.schema != mipdb_lance_schema():
        raise AssertionError("MIPDB Lance schema mismatch")
    forbidden = {"split", "fold", "electrode_ids", "model_tokens", "prediction", "feature"} & set(dataset.schema.names)
    if forbidden:
        raise AssertionError(f"forbidden fields: {sorted(forbidden)}")
    seen: set[str] = set()
    rows = 0
    for batch in dataset.scanner(batch_size=4).to_batches():
        for index in range(batch.num_rows):
            sample_id = batch.column("sample_id")[index].as_py()
            if sample_id in seen:
                raise AssertionError(f"duplicate sample_id: {sample_id}")
            seen.add(sample_id)
            data = np.asarray(batch.column("data")[index].values, dtype=np.float32).reshape(N_EEG_CHANNELS, WINDOW_SAMPLES)
            bad = np.asarray(batch.column("bad_channel_mask")[index].values, dtype=bool)
            names = batch.column("channel_names")[index].as_py()
            statuses = batch.column("channel_status")[index].as_py()
            valid = (
                data.shape == (N_EEG_CHANNELS, WINDOW_SAMPLES)
                and int(batch.column("valid_length")[index].as_py()) == DATA_LENGTH
                and np.isfinite(data).all() and data.min() >= -1 and data.max() <= 1
                and bad.shape == (N_EEG_CHANNELS,)
                and (not bad.any() or np.all(data[bad] == 0))
                and names == CORE_CHANNELS
                and len(statuses) == N_EEG_CHANNELS
                and batch.column("task_semantics")[index].as_py() == TASK_SEMANTICS
                and batch.column("resting_state_confirmed")[index].as_py() is False
                and float(batch.column("age")[index].as_py()) == float(batch.column("label")[index].as_py())
            )
            if not valid:
                raise AssertionError(f"invalid MIPDB Lance row: {sample_id}")
            rows += 1
    if rows != expected_rows:
        raise AssertionError(f"expected {expected_rows} Lance rows, got {rows}")
    return {
        "row_count": rows, "sample_id_unique": len(seen) == rows,
        "schema_matches": True, "all_rows_valid": True,
        "forbidden_fields_absent": not forbidden,
    }


def snapshot_rows(snapshot: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"relative_path": path, **value} for path, value in sorted(snapshot.items())]


class MIPDBPipeline:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.data_dir = args.data_dir.resolve()
        self.output_dir = args.output_dir.resolve()
        self.config = load_mipdb_config(args.config)
        self.pre = self.config["preprocessing"]
        self.config_hash = _config_hash(args.config)
        self.allowlist = load_allowlist(args.subject_allowlist)
        self.age_metadata = load_age_metadata(args.age_metadata)
        self.schema = mipdb_lance_schema()
        self.lance_path = self.output_dir / "dataset.lance"
        self.first_lance_write = True
        self.written_rows = 0
        self.rows: dict[str, list[dict[str, Any]]] = {name: [] for name in CSV_FIELDS}
        self.issues = self.rows["preprocessing_issues.csv"]
        self.errors: list[str] = []
        self.recordings: list[dict[str, Any]] = []
        self.subjects: dict[str, dict[str, Any]] = {}
        self.selected_paths: list[str] = []

    def issue(
        self, severity: str, stage: str, issue_type: str, description: str,
        action: str, rec: dict[str, Any] | None = None, channel: str = "",
        sample_id: str = "",
    ) -> None:
        rec = rec or {}
        self.issues.append({
            "severity": severity, "stage": stage,
            "subject_id": rec.get("subject_id", ""),
            "recording_id": rec.get("recording_id", ""),
            "block_id": rec.get("block_id", ""),
            "source_path": rec.get("source_path", ""),
            "sample_id": sample_id, "channel_name": channel,
            "issue_type": issue_type, "description": description, "action": action,
        })

    def build_manifests(self) -> None:
        discovered: list[dict[str, Any]] = []
        for path in sorted(self.data_dir.glob("sub-*/eeg/*_eeg.vhdr"), key=lambda value: value.as_posix().lower()):
            entities = parse_recording_name(path)
            if not entities:
                self.issue("error", "selection", "filename_unparseable", path.name, "Fix BIDS filename")
                continue
            sid = entities["subject"]
            if self.allowlist is not None and sid not in self.allowlist:
                continue
            if sid not in self.age_metadata:
                raise ValueError(f"Exact age metadata missing for discovered subject {sid}")
            age_row = self.age_metadata[sid]
            age = float(age_row["age"])
            if not self.args.min_age <= age <= self.args.max_age:
                continue
            block = entities["block"].lower()
            stem = path.name[:-9]
            sidecar = path.with_suffix(".json")
            channels = path.with_name(stem + "_channels.tsv")
            events = path.with_name(stem + "_events.tsv")
            binary = path.with_suffix(".eeg")
            marker = path.with_suffix(".vmrk")
            required = [sidecar, channels, binary, marker]
            missing = [str(item) for item in required if not item.is_file()]
            if missing:
                raise FileNotFoundError(f"Missing MIPDB recording companions for {path}: {missing}")
            sidecar_data = read_json(sidecar)
            sidecar_line = parse_power_line_frequency(sidecar_data.get("PowerLineFrequency"))
            configured_notch = float(self.pre["notch_hz"])
            if sidecar_line is not None and not math.isclose(sidecar_line, configured_notch, abs_tol=1e-9):
                raise ValueError(
                    f"Sidecar PowerLineFrequency={sidecar_line} conflicts with explicit notch={configured_notch}: {sidecar}"
                )
            rec = {
                "subject_id": sid, "recording_id": path.stem, "block_id": block,
                "source_path": path.relative_to(self.data_dir).as_posix(),
                "binary_path": binary.relative_to(self.data_dir).as_posix(),
                "marker_path": marker.relative_to(self.data_dir).as_posix(),
                "sidecar_path": sidecar.relative_to(self.data_dir).as_posix(),
                "channels_path": channels.relative_to(self.data_dir).as_posix(),
                "events_path": events.relative_to(self.data_dir).as_posix() if events.is_file() else "",
                "absolute_path": path, "sidecar": sidecar_data, "age": age,
                "label": age, "header_signature": "", "task_semantics": TASK_SEMANTICS,
                "resting_state_confirmed": False, "block01_candidate": block == "block01",
                "aux_channel_count": "", "aux_channel_names": [], "aux_channel_types": [],
                "power_line_frequency_sidecar": sidecar_data.get("PowerLineFrequency", ""),
            }
            discovered.append(rec)
        if not discovered:
            raise ValueError("No eligible MIPDB BrainVision recordings discovered")
        if self.args.smoke:
            wanted = {value.casefold() for value in self.args.smoke_subjects}
            self.recordings = [row for row in discovered if row["subject_id"].casefold() in wanted]
            found = {row["subject_id"].casefold() for row in self.recordings}
            if found != wanted:
                raise ValueError(f"Smoke subjects not found in eligible recordings: {sorted(wanted - found)}")
        else:
            self.recordings = discovered
        for sid in sorted({row["subject_id"] for row in self.recordings}):
            age_row = self.age_metadata[sid]
            rows = [row for row in self.recordings if row["subject_id"] == sid]
            self.subjects[sid] = {
                "subject_id": sid, "age": age_row["age"], "label": age_row["age"],
                "age_unit": "year", "age_source": age_row["source"],
                "sex_raw": age_row["sex_raw"], "recording_count": len(rows), "selected": True,
            }
        self.rows["subject_manifest.csv"] = list(self.subjects.values())
        # Snapshot every file below each selected subject, plus root metadata.
        # This remains dynamic when subjects/recordings are added and makes the
        # source-unchanged assertion broader than the files opened directly.
        paths: set[str] = set()
        for sid in self.subjects:
            subject_dir = self.data_dir / sid
            paths.update(
                path.relative_to(self.data_dir).as_posix()
                for path in subject_dir.rglob("*") if path.is_file()
            )
        paths.update(
            path.relative_to(self.data_dir).as_posix()
            for path in self.data_dir.iterdir() if path.is_file()
        )
        self.selected_paths = sorted(paths)

    def _scan_recording(
        self, raw: mne.io.BaseRaw, core_indices: list[int], rec: dict[str, Any],
        channel_meta: list[dict[str, str]],
    ) -> tuple[np.ndarray, list[dict[str, Any]], int]:
        n_samples = int(raw.n_times)
        chunk = max(1, int(round(self.args.chunk_seconds * float(raw.info["sfreq"]))))
        mins = np.full(N_EEG_CHANNELS, np.inf)
        maxs = np.full(N_EEG_CHANNELS, -np.inf)
        sums = np.zeros(N_EEG_CHANNELS)
        squares = np.zeros(N_EEG_CHANNELS)
        counts = np.zeros(N_EEG_CHANNELS, dtype=np.int64)
        first = np.full(N_EEG_CHANNELS, np.nan)
        all_same = np.ones(N_EEG_CHANNELS, dtype=bool)
        has_nan = np.zeros(N_EEG_CHANNELS, dtype=bool)
        has_inf = np.zeros(N_EEG_CHANNELS, dtype=bool)
        read_samples = 0
        for start in range(0, n_samples, chunk):
            stop = min(n_samples, start + chunk)
            data_uv = convert_mne_v_to_uv(raw.get_data(picks=core_indices, start=start, stop=stop))
            if data_uv.shape != (N_EEG_CHANNELS, stop - start):
                raise RuntimeError(f"chunk shape mismatch: {data_uv.shape} at {start}:{stop}")
            read_samples += data_uv.shape[1]
            for index, values in enumerate(data_uv):
                has_nan[index] |= bool(np.isnan(values).any())
                has_inf[index] |= bool(np.isinf(values).any())
                finite = values[np.isfinite(values)]
                if finite.size == 0:
                    continue
                if counts[index] == 0:
                    first[index] = finite[0]
                if not np.all(finite == first[index]):
                    all_same[index] = False
                mins[index] = min(mins[index], float(finite.min()))
                maxs[index] = max(maxs[index], float(finite.max()))
                sums[index] += finite.sum(dtype=np.float64)
                squares[index] += np.dot(finite, finite)
                counts[index] += finite.size
        if read_samples != n_samples:
            raise RuntimeError(f"decoded samples {read_samples} != header samples {n_samples}")
        if has_nan.any() or has_inf.any() or np.any(counts != n_samples):
            raise RuntimeError("NaN/Inf or incomplete finite sample count in full recording")
        bad = np.zeros(N_EEG_CHANNELS, dtype=bool)
        inventory: list[dict[str, Any]] = []
        for index, name in enumerate(CORE_CHANNELS):
            mean = sums[index] / counts[index]
            variance = max(0.0, squares[index] / counts[index] - mean * mean)
            reasons: list[str] = []
            if all_same[index]:
                if first[index] == 0:
                    reasons.append("all_zero")
                reasons.extend(["strict_constant", "exact_zero_variance"])
            bad[index] = bool(reasons)
            unit = channel_meta[index].get("units", "")
            if unit_to_uv_multiplier(unit) != 1.0:
                raise RuntimeError(f"Unverified core channel unit {name}={unit!r}")
            near_zero = variance <= 1e-12 and not all_same[index]
            dc = abs(mean) > 1000.0
            extreme = max(abs(mins[index]), abs(maxs[index])) > 10000.0
            row = {
                "subject_id": rec["subject_id"], "recording_id": rec["recording_id"],
                "block_id": rec["block_id"], "source_path": rec["source_path"],
                "channel_index": index, "channel_name": name, "channel_type": "EEG",
                "unit": unit, "raw_min_uv": mins[index], "raw_max_uv": maxs[index],
                "raw_mean_uv": mean, "raw_std_uv": math.sqrt(variance),
                "post_filter_min_uv": "", "post_filter_max_uv": "",
                "post_filter_mean_uv": "", "post_filter_std_uv": "",
                "near_zero_variance_flag": near_zero, "dc_offset_flag": dc,
                "extreme_amplitude_flag": extreme, "is_bad": bool(bad[index]),
                "bad_reason": ";".join(reasons),
            }
            inventory.append(row)
            if near_zero:
                self.issue("warning", "raw_signal_qc", "near_zero_variance", f"variance={variance:.12g} µV²", "Record warning; existing company rule does not reject on this statistic alone", rec, name)
            if dc:
                self.issue("warning", "raw_signal_qc", "dc_offset", f"raw mean={mean:.3f} µV", "Record pre/post-filter statistics and clip fraction", rec, name)
            if extreme:
                self.issue("warning", "raw_signal_qc", "extreme_amplitude", f"raw range=[{mins[index]:.3f},{maxs[index]:.3f}] µV", "Record pre/post-filter statistics and clip fraction", rec, name)
            if bad[index]:
                self.issue("warning", "raw_signal_qc", "recording_bad_channel", ";".join(reasons), "Keep native slot, set bad mask, and zero cached samples", rec, name)
        if bad.all():
            raise RuntimeError("all 128 core EEG channels are bad")
        return bad, inventory, read_samples

    def process_one(self, rec: dict[str, Any]) -> None:
        summary = {
            "subject_id": rec["subject_id"], "recording_id": rec["recording_id"],
            "block_id": rec["block_id"], "source_path": rec["source_path"],
            "run_mode": "smoke" if self.args.smoke else "full",
            "processing_success": False, "error": "",
        }
        validation = {
            "subject_id": rec["subject_id"], "recording_id": rec["recording_id"],
            "block_id": rec["block_id"], "source_path": rec["source_path"],
            "header_read_success": False, "fully_readable": False,
            "validation_pass": False, "error": "",
        }
        try:
            raw = mne.io.read_raw_brainvision(rec["absolute_path"], preload=False, verbose="ERROR")
            validation["header_read_success"] = True
            sfreq = float(raw.info["sfreq"])
            if not math.isclose(sfreq, SOURCE_SFREQ, abs_tol=1e-9):
                raise RuntimeError(f"expected {SOURCE_SFREQ} Hz, got {sfreq}")
            types = raw.get_channel_types()
            core_indices, aux_indices = resolve_core_channels(raw.ch_names, types)
            channel_rows_all = read_tsv(self.data_dir / rec["channels_path"])
            channel_by_name = {row.get("name", ""): row for row in channel_rows_all}
            channel_meta = [channel_by_name.get(name, {}) for name in CORE_CHANNELS]
            if [row.get("name") for row in channel_meta] != CORE_CHANNELS:
                raise RuntimeError("channels.tsv does not contain E1-E128 in canonical order")
            if any(str(row.get("type", "")).casefold() != "eeg" for row in channel_meta):
                raise RuntimeError("channels.tsv core channel type mismatch")
            orig_units = dict(getattr(raw, "_orig_units", {}))
            if any(unit_to_uv_multiplier(orig_units.get(name)) != 1.0 for name in CORE_CHANNELS):
                raise RuntimeError("BrainVision header units are not uniformly µV for E1-E128")
            aux_names = [raw.ch_names[index] for index in aux_indices]
            aux_types = [types[index] for index in aux_indices]
            rec.update({
                "aux_channel_count": len(aux_indices), "aux_channel_names": aux_names,
                "aux_channel_types": aux_types,
            })
            for index in aux_indices:
                name = raw.ch_names[index]
                self.rows["aux_channel_inventory.csv"].append({
                    "subject_id": rec["subject_id"], "recording_id": rec["recording_id"],
                    "block_id": rec["block_id"], "source_path": rec["source_path"],
                    "source_index": index, "channel_name": name,
                    "channel_type": types[index], "unit": orig_units.get(name, "n/a"),
                    "excluded_from_lance": True,
                })
            signature_payload = {
                "sfreq": sfreq, "core_names": CORE_CHANNELS,
                "core_types": [types[index] for index in core_indices],
                "core_units": [orig_units.get(name, "") for name in CORE_CHANNELS],
                "aux_names": aux_names, "aux_types": aux_types,
            }
            rec["header_signature"] = hashlib.sha256(
                json.dumps(signature_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
            ).hexdigest()[:16]
            bad, channel_stats, read_samples = self._scan_recording(raw, core_indices, rec, channel_meta)
            n_samples = int(raw.n_times)
            good = np.flatnonzero(~bad)
            good_uv = convert_mne_v_to_uv(raw.get_data(picks=[core_indices[index] for index in good]))
            raw.close()
            filtered = apply_common_signal_processing(
                good_uv, sfreq, float(self.pre["target_sfreq"]),
                float(self.pre["notch_hz"]), float(self.pre["bandpass_low"]),
                float(self.pre["bandpass_high"]),
            )
            processed = np.zeros((N_EEG_CHANNELS, filtered.shape[1]), dtype=np.float64)
            processed[good] = filtered
            for local, index in enumerate(good):
                values = filtered[local]
                channel_stats[index].update({
                    "post_filter_min_uv": float(values.min()),
                    "post_filter_max_uv": float(values.max()),
                    "post_filter_mean_uv": float(values.mean()),
                    "post_filter_std_uv": float(values.std(dtype=np.float64)),
                })
            self.rows["channel_statistics.csv"].extend(channel_stats)
            hop_samples = int(round(float(self.pre["hop_seconds"]) * float(self.pre["target_sfreq"])))
            possible = window_slices(processed.shape[1], int(self.pre["segment_samples"]), hop_samples)
            emitted, smoke_truncated = limit_window_slices(
                possible, self.args.smoke, self.args.smoke_max_windows_per_recording
            )
            tail_samples, tail_seconds = dropped_tail(
                processed.shape[1], int(self.pre["segment_samples"]), hop_samples,
                float(self.pre["target_sfreq"]),
            )
            cache_rows: list[dict[str, Any]] = []
            clips: list[float] = []
            qc_pass_count = 0
            for window_index, (start, stop) in enumerate(emitted):
                window = processed[:, start:stop].copy()
                window_bad, reasons_by_channel, preprocessing_finite = identify_window_bad_channels(window, bad)
                combined = merge_bad_channel_masks(bad, window_bad)
                qc_pass, qc_reasons = evaluate_window_qc(
                    window, combined, (N_EEG_CHANNELS, WINDOW_SAMPLES), preprocessing_finite
                )
                zeroed, channel_mask = zero_bad_channels(window, combined)
                good_values = zeroed[~combined]
                clip = float(np.mean(np.abs(good_values) >= 200.0)) if good_values.size else 0.0
                positive_clip = float(np.mean(good_values >= 200.0)) if good_values.size else 0.0
                negative_clip = float(np.mean(good_values <= -200.0)) if good_values.size else 0.0
                absmax = float(np.max(np.abs(good_values))) if good_values.size else 0.0
                normalized = normalize_clip_divide_200(zeroed)
                normalized[combined] = 0.0
                if (
                    normalized.shape != (N_EEG_CHANNELS, WINDOW_SAMPLES)
                    or normalized.dtype != np.float32 or not np.isfinite(normalized).all()
                    or normalized.min() < -1 or normalized.max() > 1
                    or (combined.any() and not np.all(normalized[combined] == 0))
                ):
                    qc_pass = False
                    qc_reasons.append("normalized_hard_qc_failed")
                source_start = int(round(start / float(self.pre["target_sfreq"]) * sfreq))
                sample_id = make_sample_id(rec["subject_id"], rec["recording_id"], window_index, source_start)
                for index in np.flatnonzero(window_bad):
                    self.rows["bad_channel_inventory.csv"].append({
                        "subject_id": rec["subject_id"], "recording_id": rec["recording_id"],
                        "block_id": rec["block_id"], "source_path": rec["source_path"],
                        "sample_id": sample_id, "channel_index": int(index),
                        "channel_name": CORE_CHANNELS[index], "scope": "window",
                        "bad_reason": ";".join(reasons_by_channel[index]),
                        "affected_windows": 1, "data_zeroed": bool(np.all(normalized[index] == 0)),
                        "mask_verified": bool(combined[index]),
                    })
                for index in np.flatnonzero(bad):
                    if window_index == 0:
                        self.rows["bad_channel_inventory.csv"].append({
                            "subject_id": rec["subject_id"], "recording_id": rec["recording_id"],
                            "block_id": rec["block_id"], "source_path": rec["source_path"],
                            "sample_id": "", "channel_index": int(index),
                            "channel_name": CORE_CHANNELS[index], "scope": "recording",
                            "bad_reason": channel_stats[index]["bad_reason"],
                            "affected_windows": len(emitted), "data_zeroed": True,
                            "mask_verified": True,
                        })
                written = bool(qc_pass)
                inventory_row = {
                    "sample_id": sample_id, "subject_id": rec["subject_id"],
                    "recording_id": rec["recording_id"], "block_id": rec["block_id"],
                    "source_path": rec["source_path"], "age": rec["age"],
                    "header_signature": rec["header_signature"],
                    "task_semantics": TASK_SEMANTICS, "resting_state_confirmed": False,
                    "block01_candidate": rec["block01_candidate"], "window_index": window_index,
                    "start_sample": source_start,
                    "start_seconds": source_start / sfreq,
                    "target_start_sample": start, "duration": float(self.pre["window_seconds"]),
                    "shape": f"[{N_EEG_CHANNELS},{WINDOW_SAMPLES}]", "dtype": "float32",
                    "qc_pass": qc_pass, "qc_failure_reason": ";".join(qc_reasons),
                    "clip_fraction": clip, "positive_clip_fraction": positive_clip,
                    "negative_clip_fraction": negative_clip,
                    "bad_channel_count": int(combined.sum()),
                    "bad_channel_names": ";".join(CORE_CHANNELS[index] for index in np.flatnonzero(combined)),
                    "written_to_lance": written,
                }
                self.rows["window_inventory.csv"].append(inventory_row)
                if written:
                    cache_rows.append({
                        "sample_id": sample_id, "subject_id": rec["subject_id"],
                        "recording_id": rec["recording_id"], "block_id": rec["block_id"],
                        "source_path": rec["source_path"], "age": np.float32(rec["age"]),
                        "label": np.float32(rec["age"]), "task_type": "regression",
                        "task_semantics": TASK_SEMANTICS, "resting_state_confirmed": False,
                        "block01_candidate": rec["block01_candidate"],
                        "header_signature": rec["header_signature"], "window_index": window_index,
                        "start_sample": source_start, "start_seconds": source_start / sfreq,
                        "target_start_sample": start, "duration": float(self.pre["window_seconds"]),
                        "modality": "eeg", "data": normalized.reshape(-1),
                        "shape": [N_EEG_CHANNELS, WINDOW_SAMPLES],
                        "original_shape": [N_EEG_CHANNELS, WINDOW_SAMPLES],
                        "valid_length": DATA_LENGTH, "channel_names": CORE_CHANNELS,
                        "channel_mask": channel_mask, "bad_channel_mask": combined,
                        "channel_status": ["bad" if value else "good" for value in combined],
                        "bad_channel_count": int(combined.sum()), "qc_pass": True,
                        "clip_fraction": np.float32(clip),
                        "positive_clip_fraction": np.float32(positive_clip),
                        "negative_clip_fraction": np.float32(negative_clip),
                        "pre_normalization_abs_max_uv": np.float32(absmax),
                        "source_sampling_rate": np.float32(sfreq),
                        "target_sampling_rate": np.float32(self.pre["target_sfreq"]),
                        "source_reference": SOURCE_REFERENCE,
                        "processing_reference": PROCESSING_REFERENCE,
                        "preprocessing_config_hash": self.config_hash,
                    })
                    qc_pass_count += 1
                else:
                    self.issue("error", "window_qc", "window_hard_qc_failed", ";".join(qc_reasons), "Do not write this window to Lance", rec, sample_id=sample_id)
                clips.append(clip)
            if cache_rows:
                _write_lance_batch(self.lance_path, cache_rows, self.first_lance_write, self.schema)
                self.first_lance_write = False
                self.written_rows += len(cache_rows)
            self.rows["dropped_tails.csv"].append({
                "subject_id": rec["subject_id"], "recording_id": rec["recording_id"],
                "block_id": rec["block_id"], "source_path": rec["source_path"],
                "source_samples": n_samples, "resampled_samples": processed.shape[1],
                "theoretical_complete_windows": len(possible), "emitted_windows": len(emitted),
                "smoke_truncated_windows": len(possible) - len(emitted),
                "real_dropped_tail_samples": tail_samples,
                "real_dropped_tail_seconds": tail_seconds,
            })
            summary.update({
                "source_samples": n_samples, "source_duration_seconds": n_samples / sfreq,
                "resampled_samples": processed.shape[1],
                "theoretical_complete_windows": len(possible), "emitted_windows": len(emitted),
                "written_windows": len(cache_rows),
                "smoke_truncated_windows": len(possible) - len(emitted),
                "real_dropped_tail_samples": tail_samples,
                "real_dropped_tail_seconds": tail_seconds,
                "recording_bad_channels": ";".join(CORE_CHANNELS[index] for index in np.flatnonzero(bad)),
                "qc_pass_windows": qc_pass_count,
                "qc_fail_windows": len(emitted) - qc_pass_count,
                "mean_clip_fraction": float(np.mean(clips)) if clips else 0.0,
                "max_clip_fraction": max(clips, default=0.0), "processing_success": True,
            })
            validation.update({
                "fully_readable": True, "sampling_rate_hz": sfreq,
                "n_channels_total": len(types), "n_eeg_channels": N_EEG_CHANNELS,
                "n_aux_channels": len(aux_indices), "n_samples_header": n_samples,
                "n_samples_read": read_samples, "channel_order_matches": True,
                "unit_conversion_verified": True, "has_nan": False, "has_inf": False,
                "recording_bad_channel_count": int(bad.sum()),
                "near_zero_variance_channel_count": sum(bool(row["near_zero_variance_flag"]) for row in channel_stats),
                "dc_offset_channel_count": sum(bool(row["dc_offset_flag"]) for row in channel_stats),
                "extreme_amplitude_channel_count": sum(bool(row["extreme_amplitude_flag"]) for row in channel_stats),
                "validation_pass": True,
            })
        except Exception as exc:
            summary["error"] = f"{type(exc).__name__}: {exc}"
            validation["error"] = summary["error"]
            self.errors.append(
                f"[{now_iso()}] {rec['subject_id']} {rec['source_path']}\n{traceback.format_exc()}\n"
            )
            self.issue("error", "recording_processing", "recording_failed_closed", summary["error"], "Exclude recording from main Lance", rec)
            print(f"[recording failed] {rec['source_path']}: {summary['error']}", file=sys.stderr, flush=True)
        finally:
            self.rows["recording_processing_summary.csv"].append(summary)
            self.rows["recording_validation.csv"].append(validation)

    def run(self) -> None:
        self.build_manifests()
        before = _snapshot_raw_files(self.data_dir, self.selected_paths)
        for index, rec in enumerate(self.recordings, 1):
            print(f"[{index}/{len(self.recordings)}] {rec['subject_id']} {rec['block_id']}", flush=True)
            self.process_one(rec)
        after = _snapshot_raw_files(self.data_dir, self.selected_paths)
        if self.first_lance_write:
            self.issue("error", "window_qc", "no_hard_qc_passing_windows", "No recording produced a passing window", "Inspect errors and QC outputs")
            self.write_outputs(before, after, {"row_count": 0, "all_rows_valid": False})
            raise RuntimeError("No hard-QC-passing windows were written")
        roundtrip = validate_mipdb_lance(self.lance_path, self.written_rows)
        self.write_outputs(before, after, roundtrip)

    def write_outputs(
        self, before: dict[str, Any], after: dict[str, Any], roundtrip: dict[str, Any]
    ) -> None:
        self.rows["recording_manifest.csv"] = [
            {key: value for key, value in rec.items() if key not in {"absolute_path", "sidecar"}}
            for rec in self.recordings
        ]
        for name, fields in CSV_FIELDS.items():
            if name.startswith("source_snapshot_"):
                continue
            _write_csv(self.output_dir / name, self.rows[name], fields)
        _write_csv(self.output_dir / "source_snapshot_before.csv", snapshot_rows(before), CSV_FIELDS["source_snapshot_before.csv"])
        _write_csv(self.output_dir / "source_snapshot_after.csv", snapshot_rows(after), CSV_FIELDS["source_snapshot_after.csv"])
        (self.output_dir / "preprocessing_errors.log").write_text("".join(self.errors), encoding="utf-8")
        source_unchanged = before == after
        summary = {
            "dataset_id": "nm000153_mipdb_age_common",
            "run_mode": "smoke" if self.args.smoke else "full",
            "selected_subjects": sorted(self.subjects),
            "selected_subject_count": len(self.subjects),
            "selected_recording_count": len(self.recordings),
            "recording_success_count": sum(bool(row.get("processing_success")) for row in self.rows["recording_processing_summary.csv"]),
            "recording_failure_count": sum(not bool(row.get("processing_success")) for row in self.rows["recording_processing_summary.csv"]),
            "theoretical_complete_windows": sum(int(row.get("theoretical_complete_windows") or 0) for row in self.rows["recording_processing_summary.csv"]),
            "emitted_windows": len(self.rows["window_inventory.csv"]),
            "written_windows": self.written_rows,
            "qc_pass_windows": sum(bool(row.get("written_to_lance")) for row in self.rows["window_inventory.csv"]),
            "qc_fail_windows": sum(not bool(row.get("written_to_lance")) for row in self.rows["window_inventory.csv"]),
            "smoke_truncated_windows": sum(int(row.get("smoke_truncated_windows") or 0) for row in self.rows["recording_processing_summary.csv"]),
            "age_min": min(row["age"] for row in self.subjects.values()),
            "age_max": max(row["age"] for row in self.subjects.values()),
            "age_distribution": dict(Counter(row["age"] for row in self.subjects.values())),
            "block_recording_distribution": dict(Counter(row["block_id"] for row in self.recordings)),
            "header_signature_count": len({row.get("header_signature") for row in self.recordings if row.get("header_signature")}),
            "native_eeg_channels": CORE_CHANNELS,
            "aux_channels_excluded": True,
            "task_semantics": TASK_SEMANTICS,
            "resting_state_confirmed": False,
            "block01_candidate_available": any(row["block_id"] == "block01" for row in self.recordings),
            "notch_hz": float(self.pre["notch_hz"]),
            "line_frequency_evidence": self.config["line_frequency"],
            "source_reference": SOURCE_REFERENCE,
            "processing_reference": PROCESSING_REFERENCE,
            "rereference_applied": False,
            "source_files_unchanged": source_unchanged,
            "source_files_modified": not source_unchanged,
            "blocking_error_count": sum(row["severity"] == "error" for row in self.issues),
            "warning_count": sum(row["severity"] == "warning" for row in self.issues),
            "roundtrip": roundtrip,
            "lance_row_count_matches_qc_pass": self.written_rows == sum(bool(row.get("written_to_lance")) for row in self.rows["window_inventory.csv"]),
            "generated_at": now_iso(),
        }
        (self.output_dir / "preprocessing_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        shutil.copy2(self.args.config, self.output_dir / "resolved_preprocessing_config.toml")
        provenance = {
            "created_at": now_iso(), "source_uri": self.args.source_uri,
            "resolved_local_data_dir": str(self.data_dir), "output_dir": str(self.output_dir),
            "run_mode": summary["run_mode"], "selected_subjects": summary["selected_subjects"],
            "selected_recordings": summary["selected_recording_count"],
            "python_version": platform.python_version(), "mne_version": mne.__version__,
            "numpy_version": np.__version__, "scipy_version": scipy.__version__,
            "pyarrow_version": pa.__version__, "lance_version": lance.__version__,
            "script_path": str(Path(__file__).resolve()),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "config_path": str(self.args.config.resolve()), "config_sha256": self.config_hash,
            "age_metadata_path": str(self.args.age_metadata.resolve()),
            "age_metadata_sha256": hashlib.sha256(self.args.age_metadata.read_bytes()).hexdigest(),
            "subject_allowlist_path": str(self.args.subject_allowlist.resolve()) if self.args.subject_allowlist else None,
            "subject_allowlist_sha256": (
                hashlib.sha256(self.args.subject_allowlist.read_bytes()).hexdigest()
                if self.args.subject_allowlist else None
            ),
            "actual_command": self.args.actual_command,
            "filter_parameters": {
                **FILTER_PARAMETERS,
                "notch": {**FILTER_PARAMETERS["notch"], "frequency_hz": float(self.pre["notch_hz"])},
            },
            "line_frequency_resolution": self.config["line_frequency"],
            "actual_processing_order": [
                "MNE V to uV", "recording hard QC", f"{float(self.pre['notch_hz']):g} Hz notch",
                "0.3-75 Hz bandpass", "500 to 200 Hz polyphase resample",
                "10 s non-overlap window", "window hard QC", "bad channel zero",
                "clip_divide_200",
            ],
            "source_reference": SOURCE_REFERENCE, "processing_reference": PROCESSING_REFERENCE,
            "rereference_applied": False, "task_semantics": TASK_SEMANTICS,
            "resting_state_confirmed": False, "source_snapshot_before": before,
            "source_snapshot_after": after, "source_files_unchanged": source_unchanged,
        }
        provenance.update(_git_metadata(Path(__file__).resolve().parents[1]))
        (self.output_dir / "processing_provenance.json").write_text(
            json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (self.output_dir / "metadata.toml").write_text(
            'spec_version = "0.1.0"\nprofile = "common"\ndataset_id = "nm000153_mipdb_age_common"\n'
            'task_semantics = "unknown"\nresting_state_confirmed = false\n'
            'block01_candidate_profile = "derive downstream using block_id == block01"\n'
            'all_blocks_profile = "derive downstream using all block_id values"\n'
            'channel_layout = "EGI E1-E128 native order"\nsource_reference = "unknown"\n'
            'processing_reference = "unchanged"\nrereference_applied = false\n'
            'model_channel_mapping = "pending"\n',
            encoding="utf-8",
        )
        (self.output_dir / "versions.toml").write_text(
            f'script_version = "{SCRIPT_VERSION}"\nmne = "{mne.__version__}"\n'
            f'numpy = "{np.__version__}"\nscipy = "{scipy.__version__}"\n'
            f'pyarrow = "{pa.__version__}"\nlance = "{lance.__version__}"\n',
            encoding="utf-8",
        )
        report = self.render_report(summary)
        (self.output_dir / "preprocessing_report.md").write_text(report, encoding="utf-8")
        qc_dir = self.output_dir / "derivatives" / "qc"
        report_dir = self.output_dir / "derivatives" / "reports"
        qc_dir.mkdir(parents=True, exist_ok=True)
        report_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.output_dir / "preprocessing_issues.csv", qc_dir / "preprocessing_issues.csv")
        shutil.copy2(self.output_dir / "preprocessing_summary.json", report_dir / "preprocessing_summary.json")
        shutil.copy2(self.output_dir / "preprocessing_report.md", report_dir / "preprocessing_report.md")

    def render_report(self, summary: dict[str, Any]) -> str:
        return f'''# MIPDB 成人年龄公共 raw-to-Lance 预处理报告

- 运行模式：{summary["run_mode"]}
- 动态发现并选择：{summary["selected_subject_count"]} 名被试、{summary["selected_recording_count"]} 条 recording。
- recording success/failure：{summary["recording_success_count"]}/{summary["recording_failure_count"]}。
- 理论完整窗口：{summary["theoretical_complete_windows"]}；实际发出：{summary["emitted_windows"]}；写入 Lance：{summary["written_windows"]}。
- QC pass/fail：{summary["qc_pass_windows"]}/{summary["qc_fail_windows"]}。
- 年龄范围：{summary["age_min"]}–{summary["age_max"]}。
- notch：{summary["notch_hz"]} Hz；sidecar 为 n/a，依据配置中记录的抽样 PSD 明确选择，不自动猜测。

## 固定语义

仅保留 E1–E128 原顺序。所有 auxiliary/event 通道从 Lance 排除并记录在 `aux_channel_inventory.csv`。不添加 E129/Cz，不做通道映射、插值、补零或重参考。source_reference=unknown，processing_reference=unchanged。

每个窗口保留 subject、recording、block、source path、age、header signature、window index、source start sample/seconds。task_semantics=unknown，resting_state_confirmed=false。`block01_candidate` 只是下游筛选字段，不声称其已确认为静息态。

## 处理链

MNE V→µV → recording 硬 QC → {summary["notch_hz"]:g} Hz notch → 0.3–75 Hz bandpass → 500→200 Hz polyphase → 10 秒无重叠窗口 → window 硬 QC → 坏道置零 → x/200 clip[-1,1]。不足10秒尾部丢弃；无 padding、重参考、ICA、插值、split 或模型电极映射。

DC offset 与极端幅值只记录 pre/post-filter 统计和 clip fraction，不据此淘汰。严格常数/全零通道进入 bad mask 并置零；无法完整解码、NaN/Inf、核心通道/采样率/单位错误的 recording fail closed。

## 验收

- Lance round-trip：{summary["roundtrip"].get("all_rows_valid")}
- Lance 行数与 QC pass 一致：{summary["lance_row_count_matches_qc_pass"]}
- 原始文件大小/mtime 未改变：{summary["source_files_unchanged"]}
- blocking errors：{summary["blocking_error_count"]}；warnings：{summary["warning_count"]}
'''


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--age-metadata", type=Path, required=True)
    parser.add_argument("--subject-allowlist", type=Path)
    parser.add_argument("--source-uri", default=os.environ.get("MIPDB_SOURCE_URI", DEFAULT_SOURCE_URI))
    parser.add_argument("--chunk-seconds", type=float, default=30.0)
    parser.add_argument("--min-age", type=float, default=20.0)
    parser.add_argument("--max-age", type=float, default=35.0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-subjects", nargs="+")
    parser.add_argument("--smoke-max-windows-per-recording", type=int)
    parser.add_argument("--overwrite", action="store_true")
    tokens = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args(tokens)
    if args.chunk_seconds <= 0:
        parser.error("--chunk-seconds must be positive")
    if args.min_age < 0 or args.max_age < args.min_age:
        parser.error("require 0 <= min-age <= max-age")
    if args.smoke:
        args.smoke_subjects = parse_smoke_subjects(
            args.smoke_subjects or ["sub-A00054039,sub-A00062219,sub-A00062435"]
        )
        args.smoke_max_windows_per_recording = args.smoke_max_windows_per_recording or 2
        if not 2 <= len(args.smoke_subjects) <= 3:
            parser.error("MIPDB smoke requires 2-3 subjects")
        if args.smoke_max_windows_per_recording <= 0:
            parser.error("smoke window limit must be positive")
    elif args.smoke_subjects or args.smoke_max_windows_per_recording:
        parser.error("smoke selection arguments require --smoke")
    else:
        args.smoke_subjects = []
    args.actual_command = subprocess.list2cmdline(
        [sys.executable, str(Path(__file__).resolve()), *tokens]
    )
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        _prepare_output_dir(args.output_dir, args.overwrite, args.data_dir)
        pipeline = MIPDBPipeline(args)
        pipeline.run()
        summary = json.loads(
            (args.output_dir / "preprocessing_summary.json").read_text(encoding="utf-8")
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if summary["blocking_error_count"] == 0 and summary["source_files_unchanged"] else 2
    except Exception:
        error = traceback.format_exc()
        sys.stderr.write(error)
        try:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            with (args.output_dir / "preprocessing_errors.log").open("a", encoding="utf-8") as handle:
                handle.write(error)
        except Exception as log_error:
            print(f"Could not write preprocessing error log: {log_error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
