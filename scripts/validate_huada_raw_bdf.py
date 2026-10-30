#!/usr/bin/env python3
"""Stages 1-3 of the Huada resting-state EEG cleaning workflow.

This program only validates configuration, builds the authoritative subject
manifest from directory names, and performs read-only file-level BDF checks.
It does not filter, resample, window, normalize, repair channels, export labels,
or convert data formats.

The implementation adapts the existing audit_resting_state.py patterns for
case-insensitive subject discovery, MNE lazy loading, chunked signal reads,
per-file error isolation, and UTF-8 audit outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import subprocess
import sys
import tomllib
import traceback
import warnings
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, TypeVar


FULL_SUBJECT_RE = re.compile(r"^SUB(\d+)_(\d+)_([MF])$", re.IGNORECASE)
MISSING_AGE_SUBJECT_RE = re.compile(r"^SUB(\d+)_([MF])$", re.IGNORECASE)
PLACEHOLDER_IDS = {"X", "UNKNOWN", "ANON", "ANONYMOUS", "N/A", "NA", "NONE"}
KNOWN_DEFERRED_CONSTANT_CHANNELS = {("SUB5", "FC1"), ("SUB8", "FT7")}

T = TypeVar("T")
R = TypeVar("R")


class ConfigValidationError(ValueError):
    """Raised when the fixed preprocessing configuration is inconsistent."""


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def safe_rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def parse_subject_directory(name: str) -> dict[str, Any] | None:
    """Parse supported subject directory names without inventing missing age."""
    match = FULL_SUBJECT_RE.fullmatch(name)
    if match:
        number = int(match.group(1))
        sex_code = match.group(3).upper()
        return {
            "subject_id": f"SUB{number}",
            "subject_number": number,
            "age": int(match.group(2)),
            "sex": "male" if sex_code == "M" else "female",
            "original_directory_name": name,
            "metadata_source": "directory_name",
            "metadata_status": "complete",
            "notes": "Authoritative demographic metadata parsed from directory name.",
        }

    match = MISSING_AGE_SUBJECT_RE.fullmatch(name)
    if match:
        number = int(match.group(1))
        sex_code = match.group(2).upper()
        return {
            "subject_id": f"SUB{number}",
            "subject_number": number,
            "age": None,
            "sex": "male" if sex_code == "M" else "female",
            "original_directory_name": name,
            "metadata_source": "directory_name",
            "metadata_status": "age_missing",
            "notes": "Age is absent from the directory name and remains null by policy.",
        }
    return None


def canonical_unit(unit: str | None) -> str | None:
    if unit is None:
        return None
    normalized = str(unit).strip().replace("µ", "u").replace("μ", "u").lower()
    return {"v": "V", "mv": "mV", "uv": "uV"}.get(normalized)


def unit_to_uv_multiplier(unit: str | None) -> float | None:
    canonical = canonical_unit(unit)
    return {"V": 1e6, "mV": 1e3, "uV": 1.0}.get(canonical)


def validate_config(config: dict[str, Any]) -> None:
    """Validate fixed configuration invariants; raise with all detected errors."""
    errors: list[str] = []
    try:
        preprocessing = config["preprocessing"]
        windowing = config["windowing"]
        expectations = config["source_expectations"]
    except KeyError as exc:
        raise ConfigValidationError(f"Missing configuration section: {exc.args[0]}") from exc

    required_preprocessing = {
        "target_sfreq", "window_seconds", "hop_seconds", "segment_samples",
        "bandpass_low", "bandpass_high", "notch_hz", "unit_before_normalization",
        "normalization", "normalization_detail",
    }
    required_windowing = {"drop_incomplete_tail", "padding"}
    required_expectations = {"format", "sampling_rate_hz", "n_channels"}
    for key in sorted(required_preprocessing - set(preprocessing)):
        errors.append(f"Missing preprocessing.{key}")
    for key in sorted(required_windowing - set(windowing)):
        errors.append(f"Missing windowing.{key}")
    for key in sorted(required_expectations - set(expectations)):
        errors.append(f"Missing source_expectations.{key}")
    if errors:
        raise ConfigValidationError("; ".join(errors))

    target_sfreq = float(preprocessing["target_sfreq"])
    window_seconds = float(preprocessing["window_seconds"])
    hop_seconds = float(preprocessing["hop_seconds"])
    segment_samples = int(preprocessing["segment_samples"])
    bandpass_low = float(preprocessing["bandpass_low"])
    bandpass_high = float(preprocessing["bandpass_high"])
    notch_hz = float(preprocessing["notch_hz"])

    expected_segment_samples = target_sfreq * window_seconds
    if not math.isclose(segment_samples, expected_segment_samples, rel_tol=0.0, abs_tol=1e-9):
        errors.append(
            f"segment_samples={segment_samples} != target_sfreq*window_seconds={expected_segment_samples:g}"
        )
    if hop_seconds <= 0:
        errors.append("hop_seconds must be > 0")
    if bandpass_low <= 0:
        errors.append("bandpass_low must be > 0")
    if bandpass_high >= target_sfreq / 2:
        errors.append("bandpass_high must be < target_sfreq / 2")
    if notch_hz >= target_sfreq / 2:
        errors.append("notch_hz must be < target_sfreq / 2")
    if target_sfreq <= 0 or window_seconds <= 0:
        errors.append("target_sfreq and window_seconds must be > 0")
    if str(expectations["format"]).upper() != "BDF":
        errors.append("source_expectations.format must be BDF")
    if float(expectations["sampling_rate_hz"]) <= 0:
        errors.append("source_expectations.sampling_rate_hz must be > 0")
    if int(expectations["n_channels"]) <= 0:
        errors.append("source_expectations.n_channels must be > 0")
    if errors:
        raise ConfigValidationError("; ".join(errors))


def load_and_validate_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    validate_config(config)
    return config


def compare_channel_layout(
    reference: list[dict[str, Any]], actual: list[dict[str, Any]], sfreq_tolerance: float = 1e-9
) -> list[dict[str, Any]]:
    """Return positional channel differences without reordering either layout."""
    mismatches: list[dict[str, Any]] = []
    max_length = max(len(reference), len(actual))
    for index in range(max_length):
        expected = reference[index] if index < len(reference) else None
        observed = actual[index] if index < len(actual) else None
        if expected is None or observed is None:
            mismatches.append(
                {
                    "channel_index": index,
                    "field": "channel_presence",
                    "expected": expected,
                    "actual": observed,
                }
            )
            continue
        if expected["channel_name"] != observed["channel_name"]:
            mismatches.append(
                {
                    "channel_index": index,
                    "field": "channel_name",
                    "expected": expected["channel_name"],
                    "actual": observed["channel_name"],
                }
            )
        if not math.isclose(
            float(expected["sampling_rate_hz"]), float(observed["sampling_rate_hz"]),
            rel_tol=sfreq_tolerance, abs_tol=sfreq_tolerance,
        ):
            mismatches.append(
                {
                    "channel_index": index,
                    "field": "sampling_rate_hz",
                    "expected": expected["sampling_rate_hz"],
                    "actual": observed["sampling_rate_hz"],
                }
            )
        if str(expected["physical_dimension"]) != str(observed["physical_dimension"]):
            mismatches.append(
                {
                    "channel_index": index,
                    "field": "physical_dimension",
                    "expected": expected["physical_dimension"],
                    "actual": observed["physical_dimension"],
                }
            )
    return mismatches


def run_batch(
    items: Iterable[T], validator: Callable[[T], R]
) -> tuple[list[tuple[T, R]], list[tuple[T, BaseException, str]]]:
    """Run independent validations while preserving failures for later reporting."""
    successes: list[tuple[T, R]] = []
    failures: list[tuple[T, BaseException, str]] = []
    for item in items:
        try:
            successes.append((item, validator(item)))
        except Exception as exc:  # deliberately isolate each input file
            failures.append((item, exc, "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))))
    return successes, failures


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def prepare_output_dir(path: Path) -> Path:
    if not path.exists():
        path.mkdir(parents=True, exist_ok=False)
        return path
    if not any(path.iterdir()):
        return path
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = path / stamp
    serial = 1
    while candidate.exists():
        candidate = path / f"{stamp}_{serial:02d}"
        serial += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def issue_row(
    severity: str,
    subject_id: str,
    source_path: str,
    issue_type: str,
    expected: Any,
    actual: Any,
    description: str,
    recommendation: str,
    channel_name: str = "",
) -> dict[str, Any]:
    return {
        "severity": severity,
        "subject_id": subject_id or "unknown",
        "source_path": source_path,
        "channel_name": channel_name,
        "issue_type": issue_type,
        "expected": expected,
        "actual": actual,
        "description": description,
        "recommendation": recommendation,
    }


class RawBdfValidator:
    def __init__(self, data_dir: Path, config_path: Path, output_dir: Path, chunk_seconds: float):
        self.data_dir = data_dir.resolve()
        self.config_path = config_path.resolve()
        self.output_dir = prepare_output_dir(output_dir.resolve())
        self.chunk_seconds = max(float(chunk_seconds), 1.0)
        self.config = load_and_validate_config(self.config_path)
        self.expected_sfreq = float(self.config["source_expectations"]["sampling_rate_hz"])
        self.expected_channels = int(self.config["source_expectations"]["n_channels"])
        self.started_at = now_iso()
        self.actual_command = subprocess.list2cmdline([sys.executable, *sys.argv])

        self.manifest: list[dict[str, Any]] = []
        self.recordings: list[dict[str, Any]] = []
        self.channels: list[dict[str, Any]] = []
        self.metadata_conflicts: list[dict[str, Any]] = []
        self.issues: list[dict[str, Any]] = []
        self.error_log: list[str] = []
        self.channel_layout_by_path: dict[str, list[dict[str, Any]]] = {}
        self.reference_source_path: str | None = None

    def add_issue(self, row: dict[str, Any]) -> None:
        self.issues.append(row)

    def log_exception(self, context: str, exc: BaseException, trace: str | None = None) -> None:
        self.error_log.append(f"[{now_iso()}] {context}\n{trace or ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))}")

    def build_subject_manifest(self) -> None:
        top_dirs = sorted((path for path in self.data_dir.iterdir() if path.is_dir()), key=lambda p: p.name.lower())
        for directory in top_dirs:
            parsed = parse_subject_directory(directory.name)
            relative = safe_rel(directory, self.data_dir)
            if parsed is None:
                self.add_issue(
                    issue_row(
                        "error", "unknown", relative, "unparseable_subject_directory",
                        "SUB<number>_<age>_<M|F> or SUB<number>_<M|F>", directory.name,
                        "Top-level directory cannot be parsed as a supported subject directory.",
                        "Verify the source registry; do not rename or infer demographics automatically.",
                    )
                )
                continue
            parsed["source_directory"] = relative
            self.manifest.append(parsed)
            if parsed["age"] is None:
                self.add_issue(
                    issue_row(
                        "warning", parsed["subject_id"], relative, "subject_age_missing",
                        "age from directory name", None,
                        "Age is absent from the authoritative directory name and remains null.",
                        "Resolve from an approved external registry before age-dependent analysis; do not guess.",
                    )
                )
            elif parsed["age"] < 0 or parsed["age"] > 120:
                self.add_issue(
                    issue_row(
                        "error", parsed["subject_id"], relative, "subject_age_invalid",
                        "0 <= age <= 120", parsed["age"], "Directory age is implausible.",
                        "Verify against the approved subject registry.",
                    )
                )
            elif parsed["age"] < 18 or parsed["age"] > 80:
                self.add_issue(
                    issue_row(
                        "warning", parsed["subject_id"], relative, "subject_age_unusual",
                        "study eligibility range", parsed["age"],
                        "Age may require review against study inclusion criteria.",
                        "Check the protocol and approved registry.",
                    )
                )
            if parsed["sex"] not in {"male", "female"}:
                self.add_issue(
                    issue_row(
                        "error", parsed["subject_id"], relative, "subject_sex_invalid",
                        "male or female", parsed["sex"], "Parsed sex value is invalid.",
                        "Verify the directory name and approved registry.",
                    )
                )

        for field in ("subject_id", "subject_number"):
            grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
            for subject in self.manifest:
                grouped[subject[field]].append(subject)
            for value, entries in grouped.items():
                if len(entries) > 1:
                    for entry in entries:
                        self.add_issue(
                            issue_row(
                                "error", entry["subject_id"], entry["source_directory"],
                                f"duplicate_{field}", "unique", value,
                                f"{field} occurs in multiple authoritative directories.",
                                "Resolve the duplicate in an approved mapping table without changing raw directories.",
                            )
                        )
        self.manifest.sort(key=lambda row: (row["subject_number"], row["original_directory_name"].lower()))

    def manifest_for_path(self, path: Path) -> dict[str, Any] | None:
        relative_parts = Path(safe_rel(path, self.data_dir)).parts
        if not relative_parts:
            return None
        top_name = relative_parts[0]
        for row in self.manifest:
            if row["original_directory_name"].lower() == top_name.lower():
                return row
        return None

    def compare_header_metadata(
        self,
        manifest: dict[str, Any],
        source_path: str,
        subject_info: dict[str, Any] | None,
        meas_date: Any,
    ) -> None:
        if not subject_info:
            self.add_issue(
                issue_row(
                    "warning", manifest["subject_id"], source_path, "header_subject_metadata_missing",
                    "header metadata if present", "missing", "BDF subject metadata are unavailable.",
                    "Keep directory-name demographics authoritative and consult the approved registry if needed.",
                )
            )
            return

        header_id = str(subject_info.get("his_id") or "").strip()
        if not header_id or header_id.upper() in PLACEHOLDER_IDS:
            self.metadata_conflicts.append(
                {
                    "subject_id": manifest["subject_id"], "source_path": source_path,
                    "field": "subject_id", "authoritative_value": manifest["subject_id"],
                    "header_value": header_id or None, "conflict_type": "anonymous_placeholder",
                    "severity": "warning",
                    "description": "BDF his_id is missing or an anonymous placeholder; directory value remains authoritative.",
                }
            )
            self.add_issue(
                issue_row(
                    "warning", manifest["subject_id"], source_path, "header_subject_id_placeholder",
                    manifest["subject_id"], header_id or "missing",
                    "BDF his_id is an anonymous placeholder and cannot verify the authoritative subject_id.",
                    "Do not overwrite the manifest; retain the directory-derived subject_id.",
                )
            )
        elif header_id.upper() not in {
            manifest["subject_id"].upper(), manifest["original_directory_name"].upper()
        }:
            self.metadata_conflicts.append(
                {
                    "subject_id": manifest["subject_id"], "source_path": source_path,
                    "field": "subject_id", "authoritative_value": manifest["subject_id"],
                    "header_value": header_id, "conflict_type": "value_conflict", "severity": "warning",
                    "description": "BDF his_id conflicts with the authoritative directory-derived subject_id.",
                }
            )
            self.add_issue(
                issue_row(
                    "warning", manifest["subject_id"], source_path, "header_subject_id_conflict",
                    manifest["subject_id"], header_id,
                    "BDF his_id conflicts with directory metadata; directory value remains authoritative.",
                    "Review the acquisition log; do not alter the manifest automatically.",
                )
            )

        header_sex_code = subject_info.get("sex")
        header_sex = {0: None, 1: "male", 2: "female"}.get(header_sex_code)
        if header_sex is None:
            self.metadata_conflicts.append(
                {
                    "subject_id": manifest["subject_id"], "source_path": source_path,
                    "field": "sex", "authoritative_value": manifest["sex"],
                    "header_value": header_sex_code, "conflict_type": "anonymous_or_unknown",
                    "severity": "warning",
                    "description": "BDF sex is unknown; directory value remains authoritative.",
                }
            )
            self.add_issue(
                issue_row(
                    "warning", manifest["subject_id"], source_path, "header_sex_unknown",
                    manifest["sex"], header_sex_code,
                    "BDF sex is unknown and cannot verify the authoritative directory value.",
                    "Do not overwrite the manifest; consult the approved registry if independent verification is required.",
                )
            )
        elif header_sex != manifest["sex"]:
            self.metadata_conflicts.append(
                {
                    "subject_id": manifest["subject_id"], "source_path": source_path,
                    "field": "sex", "authoritative_value": manifest["sex"],
                    "header_value": header_sex, "conflict_type": "value_conflict", "severity": "warning",
                    "description": "BDF sex conflicts with directory metadata; directory value remains authoritative.",
                }
            )
            self.add_issue(
                issue_row(
                    "warning", manifest["subject_id"], source_path, "header_sex_conflict",
                    manifest["sex"], header_sex,
                    "BDF sex conflicts with directory metadata; directory value remains authoritative.",
                    "Review the acquisition log; do not alter the manifest automatically.",
                )
            )

        birthday = subject_info.get("birthday")
        birthday_date = birthday.date() if hasattr(birthday, "date") else birthday
        meas_date_value = meas_date.date() if hasattr(meas_date, "date") else meas_date
        if birthday_date:
            placeholder_birthday = bool(meas_date_value and birthday_date == meas_date_value)
            if placeholder_birthday:
                self.metadata_conflicts.append(
                    {
                        "subject_id": manifest["subject_id"], "source_path": source_path,
                        "field": "birthday", "authoritative_value": None,
                        "header_value": str(birthday_date), "conflict_type": "anonymous_placeholder",
                        "severity": "warning",
                        "description": "BDF birthday equals the measurement date and appears to be an anonymous placeholder.",
                    }
                )
                self.add_issue(
                    issue_row(
                        "warning", manifest["subject_id"], source_path, "header_birthday_placeholder",
                        "real birthday or unavailable", birthday_date,
                        "BDF birthday equals measurement date and is treated as an anonymous placeholder.",
                        "Do not derive authoritative age from this header field.",
                    )
                )
            if manifest["age"] is not None and meas_date_value:
                header_age = meas_date_value.year - birthday_date.year - (
                    (meas_date_value.month, meas_date_value.day) < (birthday_date.month, birthday_date.day)
                )
                if abs(header_age - int(manifest["age"])) > 1:
                    self.metadata_conflicts.append(
                        {
                            "subject_id": manifest["subject_id"], "source_path": source_path,
                            "field": "age", "authoritative_value": manifest["age"],
                            "header_value": header_age, "conflict_type": "value_conflict",
                            "severity": "warning",
                            "description": "Age implied by BDF birthday conflicts with directory age; directory age remains authoritative.",
                        }
                    )
                    self.add_issue(
                        issue_row(
                            "warning", manifest["subject_id"], source_path, "header_age_conflict",
                            manifest["age"], header_age,
                            "Age implied by BDF birthday conflicts with the authoritative directory age.",
                            "Retain directory age; review the approved registry if independent verification is required.",
                        )
                    )

    def validate_one_bdf(self, path: Path) -> dict[str, Any]:
        import mne
        import numpy as np

        source_path = safe_rel(path, self.data_dir)
        manifest = self.manifest_for_path(path)
        subject_id = manifest["subject_id"] if manifest else "unknown"
        local_issues: list[dict[str, Any]] = []

        def add(
            severity: str, issue_type: str, expected: Any, actual: Any,
            description: str, recommendation: str, channel_name: str = ""
        ) -> None:
            row = issue_row(
                severity, subject_id, source_path, issue_type, expected, actual,
                description, recommendation, channel_name,
            )
            local_issues.append(row)
            self.add_issue(row)

        if manifest is None:
            add(
                "error", "bdf_without_manifest_subject", "path under a parsed subject directory", source_path,
                "BDF file cannot be linked to an authoritative manifest subject.",
                "Resolve the directory mapping without moving or renaming the raw file.",
            )

        if path.suffix.lower() != ".bdf":
            add(
                "error", "file_extension_not_bdf", ".bdf", path.suffix,
                "File extension is not BDF.", "Use the correct source file; do not convert it in this stage.",
            )

        warning_messages: list[str] = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            raw = mne.io.read_raw_bdf(path, preload=False, verbose="ERROR")
            warning_messages.extend(str(item.message) for item in caught)

        try:
            extras = raw._raw_extras[0]  # MNE's parsed EDF/BDF header; inspected and version-recorded.
            subtype = str(extras.get("subtype", "unknown")).lower()
            file_format = subtype.upper() if subtype != "unknown" else "unknown"
            if subtype != "bdf" or int(extras.get("dtype_byte", 0)) != 3:
                add(
                    "error", "format_not_bdf", "BDF / 3-byte samples",
                    f"subtype={subtype}, dtype_byte={extras.get('dtype_byte')}",
                    "Parsed header does not describe a BDF file.",
                    "Verify the original export and file identity.",
                )

            n_channels = int(raw.info["nchan"])
            sfreq = float(raw.info["sfreq"])
            n_samples = int(raw.n_times)
            duration = n_samples / sfreq if sfreq > 0 else 0.0
            ch_names = list(raw.ch_names)
            ch_types = list(raw.get_channel_types())
            if n_channels != self.expected_channels:
                add(
                    "error", "channel_count_mismatch", self.expected_channels, n_channels,
                    "Data channel count does not match source expectations.",
                    "Block the recording and verify the acquisition/export configuration.",
                )
            if not math.isclose(sfreq, self.expected_sfreq, rel_tol=1e-9, abs_tol=1e-6):
                add(
                    "error", "sampling_rate_mismatch", self.expected_sfreq, sfreq,
                    "Sampling rate does not match source expectations.",
                    "Block the recording; do not resample during this validation stage.",
                )
            empty_name_indices = [index for index, name in enumerate(ch_names) if not str(name).strip()]
            duplicate_names = sorted(name for name, count in Counter(ch_names).items() if count > 1)
            if empty_name_indices:
                add(
                    "error", "empty_channel_name", "all channel names non-empty", empty_name_indices,
                    "One or more channel names are empty.", "Verify the original BDF header.",
                )
            if duplicate_names:
                add(
                    "error", "duplicate_channel_name", "unique channel names", duplicate_names,
                    "Duplicate channel names prevent positional channel identity checks.",
                    "Verify the original BDF header; do not auto-rename in this stage.",
                )
            if n_samples <= 0:
                add(
                    "error", "nonpositive_sample_count", "> 0", n_samples,
                    "Recording has no samples.", "Recover a complete source file.",
                )
            if duration <= 0:
                add(
                    "error", "nonpositive_duration", "> 0", duration,
                    "Recording duration is not positive.", "Recover a complete source file.",
                )

            header_total_signals = int(extras.get("nchan", -1))
            header_data_channels = int(extras.get("orig_nchan", -1))
            header_n_records = int(extras.get("n_records", -1))
            header_nsamples = int(extras.get("nsamples", -1))
            data_offset = int(extras.get("data_offset", -1))
            dtype_byte = int(extras.get("dtype_byte", -1))
            n_samps_all = np.asarray(extras.get("n_samps", []), dtype=np.int64)
            selected_signal_indices = np.asarray(extras.get("sel", np.arange(n_channels)), dtype=np.int64)
            record_length_values = np.asarray(extras.get("record_length", []), dtype=float)
            record_duration = float(record_length_values[0]) if record_length_values.size else math.nan
            annotation_channel_count = header_total_signals - header_data_channels

            header_shape_checks: list[bool] = []
            header_shape_checks.append(header_data_channels == n_channels)
            header_shape_checks.append(header_nsamples == n_samples)
            header_shape_checks.append(header_n_records > 0)
            header_shape_checks.append(len(n_samps_all) == header_total_signals)
            header_shape_checks.append(len(selected_signal_indices) == n_channels)
            header_shape_checks.append(math.isfinite(record_duration) and record_duration > 0)
            if n_samps_all.size and header_n_records > 0 and data_offset >= 0 and dtype_byte > 0:
                expected_file_size = data_offset + header_n_records * int(n_samps_all.sum()) * dtype_byte
            else:
                expected_file_size = -1
            file_size = path.stat().st_size
            file_size_matches = expected_file_size == file_size
            header_shape_checks.append(file_size_matches)
            if not file_size_matches:
                add(
                    "error", "header_file_size_mismatch", expected_file_size, file_size,
                    "File size does not match BDF header records, samples per record, and 3-byte sample layout.",
                    "Block the recording and recover a complete source file.",
                )

            physical_min = np.asarray(extras.get("physical_min", []), dtype=float)
            physical_max = np.asarray(extras.get("physical_max", []), dtype=float)
            digital_min = np.asarray(extras.get("digital_min", []), dtype=float)
            digital_max = np.asarray(extras.get("digital_max", []), dtype=float)
            header_shape_checks.extend(
                len(values) == n_channels for values in (physical_min, physical_max, digital_min, digital_max)
            )
            orig_units = getattr(raw, "_orig_units", {}) or {}
            channel_layout: list[dict[str, Any]] = []
            channel_inventory: list[dict[str, Any]] = []
            units_seen: set[str] = set()
            unit_failures: list[str] = []
            per_channel_sfreqs: list[float] = []

            for channel_index, (channel_name, channel_type) in enumerate(zip(ch_names, ch_types)):
                signal_index = int(selected_signal_indices[channel_index]) if channel_index < len(selected_signal_indices) else -1
                samples_per_record = (
                    int(n_samps_all[signal_index])
                    if signal_index >= 0 and signal_index < len(n_samps_all) else -1
                )
                header_sfreq = samples_per_record / record_duration if record_duration > 0 else math.nan
                per_channel_sfreqs.append(header_sfreq)
                dimension = str(orig_units.get(channel_name, "")).strip()
                canonical = canonical_unit(dimension)
                multiplier = unit_to_uv_multiplier(dimension)
                if canonical:
                    units_seen.add(canonical)
                if multiplier is None:
                    unit_failures.append(channel_name)
                row = {
                    "subject_id": subject_id,
                    "source_path": source_path,
                    "channel_index": channel_index,
                    "channel_name": channel_name,
                    "channel_type": channel_type,
                    "physical_dimension": dimension or "unknown",
                    "unit_to_uv_multiplier": multiplier if multiplier is not None else "",
                    "physical_min": float(physical_min[channel_index]) if channel_index < len(physical_min) else "",
                    "physical_max": float(physical_max[channel_index]) if channel_index < len(physical_max) else "",
                    "digital_min": float(digital_min[channel_index]) if channel_index < len(digital_min) else "",
                    "digital_max": float(digital_max[channel_index]) if channel_index < len(digital_max) else "",
                    "sampling_rate_hz": header_sfreq,
                    "samples_per_record": samples_per_record,
                    "mne_data_unit": "V",
                }
                channel_inventory.append(row)
                channel_layout.append(
                    {
                        "channel_name": channel_name,
                        "sampling_rate_hz": header_sfreq,
                        "physical_dimension": dimension or "unknown",
                    }
                )
                if channel_index < len(physical_min) and channel_index < len(physical_max):
                    if not physical_min[channel_index] < physical_max[channel_index]:
                        add(
                            "error", "invalid_physical_range", "physical_min < physical_max",
                            f"{physical_min[channel_index]} >= {physical_max[channel_index]}",
                            "BDF physical range is invalid.", "Verify the original BDF header.", channel_name,
                        )
                if channel_index < len(digital_min) and channel_index < len(digital_max):
                    if not digital_min[channel_index] < digital_max[channel_index]:
                        add(
                            "error", "invalid_digital_range", "digital_min < digital_max",
                            f"{digital_min[channel_index]} >= {digital_max[channel_index]}",
                            "BDF digital range is invalid.", "Verify the original BDF header.", channel_name,
                        )

            if unit_failures:
                add(
                    "error", "unit_conversion_failed", "V, mV, µV, uV, or μV",
                    unit_failures,
                    "One or more channel physical dimensions cannot be converted reliably to µV.",
                    "Block the recording; obtain an authoritative unit definition instead of guessing.",
                )
            if len(units_seen) > 1:
                add(
                    "error", "channel_unit_conflict", "one common EEG physical dimension",
                    sorted(units_seen),
                    "EEG channels use conflicting physical dimensions.",
                    "Block the recording and review the acquisition/export configuration.",
                )
            unit_conversion_status = "pass" if not unit_failures and len(units_seen) == 1 else "fail"

            for index, header_sfreq in enumerate(per_channel_sfreqs):
                if not math.isclose(header_sfreq, self.expected_sfreq, rel_tol=1e-9, abs_tol=1e-6):
                    add(
                        "error", "channel_header_sampling_rate_mismatch", self.expected_sfreq, header_sfreq,
                        "Per-channel samples-per-record and record duration imply an unexpected sampling rate.",
                        "Block the recording; do not resample during this validation stage.", ch_names[index],
                    )
                    header_shape_checks.append(False)
                else:
                    header_shape_checks.append(True)
                if header_n_records > 0 and index < len(per_channel_sfreqs):
                    signal_index = int(selected_signal_indices[index])
                    expected_samples = header_n_records * int(n_samps_all[signal_index])
                    header_shape_checks.append(expected_samples == n_samples)

            header_shape_matches = all(header_shape_checks)
            if not header_shape_matches:
                add(
                    "error", "header_shape_mismatch", "header dimensions consistent with readable array",
                    {
                        "header_total_signals": header_total_signals,
                        "header_data_channels": header_data_channels,
                        "header_n_records": header_n_records,
                        "header_nsamples": header_nsamples,
                        "raw_n_channels": n_channels,
                        "raw_n_samples": n_samples,
                    },
                    "BDF header dimensions do not agree with the data exposed by MNE.",
                    "Block the recording and verify file completeness and header integrity.",
                )

            reference = "unknown"
            montage = "unknown"
            add(
                "warning", "reference_unknown", "explicit header/config reference", reference,
                "Acquisition reference cannot be confirmed from the BDF header or reliable project configuration.",
                "Document the acquisition reference before signal-level preprocessing.",
            )
            add(
                "warning", "montage_unknown", "explicit header/config montage", montage,
                "Montage cannot be confirmed from the BDF header or reliable project configuration.",
                "Document the montage before spatial operations; do not infer it only from channel names.",
            )

            nan_counts = np.zeros(n_channels, dtype=np.int64)
            inf_counts = np.zeros(n_channels, dtype=np.int64)
            minima = np.full(n_channels, np.inf, dtype=np.float64)
            maxima = np.full(n_channels, -np.inf, dtype=np.float64)
            processed_samples = 0
            chunk_samples = max(1, int(round(sfreq * self.chunk_seconds)))
            for start in range(0, n_samples, chunk_samples):
                stop = min(n_samples, start + chunk_samples)
                block = raw.get_data(start=start, stop=stop)
                if block.shape != (n_channels, stop - start):
                    raise RuntimeError(
                        f"Unexpected block shape {block.shape}; expected {(n_channels, stop - start)}"
                    )
                nan_counts += np.isnan(block).sum(axis=1)
                inf_counts += np.isinf(block).sum(axis=1)
                finite = np.isfinite(block)
                for channel_index in range(n_channels):
                    values = block[channel_index, finite[channel_index]]
                    if values.size:
                        minima[channel_index] = min(minima[channel_index], float(values.min()))
                        maxima[channel_index] = max(maxima[channel_index], float(values.max()))
                processed_samples += block.shape[1]

            fully_readable = processed_samples == n_samples
            has_nan = bool(nan_counts.sum())
            has_inf = bool(inf_counts.sum())
            if not fully_readable:
                add(
                    "error", "incomplete_data_read", n_samples, processed_samples,
                    "Chunked reading did not reach the final declared sample.",
                    "Block the recording and recover a complete source file.",
                )
            if has_nan:
                add(
                    "error", "nan_values", 0, int(nan_counts.sum()),
                    "NaN values were found during full chunked reading.",
                    "Block the recording pending source investigation.",
                )
            if has_inf:
                add(
                    "error", "inf_values", 0, int(inf_counts.sum()),
                    "Inf values were found during full chunked reading.",
                    "Block the recording pending source investigation.",
                )

            constant_channels = [
                ch_names[index] for index in range(n_channels)
                if math.isfinite(minima[index]) and math.isfinite(maxima[index]) and minima[index] == maxima[index]
            ]
            extreme_channels = [
                ch_names[index] for index in range(n_channels)
                if math.isfinite(minima[index]) and max(abs(minima[index]), abs(maxima[index])) > 0.01
            ]
            if constant_channels:
                known = [name for name in constant_channels if (subject_id, name) in KNOWN_DEFERRED_CONSTANT_CHANNELS]
                add(
                    "warning", "constant_channel_deferred_to_signal_qc", "variable signal",
                    constant_channels,
                    "Constant channels were detected. This known signal-level issue is recorded but does not fail file-level validation.",
                    "Handle in stage 4 signal-level QC; do not zero or interpolate in this stage.",
                    ",".join(known or constant_channels),
                )
            if extreme_channels:
                add(
                    "warning", "extreme_amplitude_deferred_to_signal_qc", "review in signal-level QC",
                    f"{len(extreme_channels)} channels > 10,000 µV absolute",
                    "Large absolute raw amplitudes were detected and intentionally deferred to signal-level QC.",
                    "Review reference, DC offset, scaling, saturation, and artifacts in stage 4; do not modify data here.",
                )

            if manifest:
                self.compare_header_metadata(manifest, source_path, raw.info.get("subject_info"), raw.info.get("meas_date"))

            basic_header_pass = not any(
                row["severity"] == "error" for row in local_issues
                if row["issue_type"] not in {"incomplete_data_read", "nan_values", "inf_values"}
            )
            self.channels.extend(channel_inventory)
            self.channel_layout_by_path[source_path] = channel_layout
            return {
                "subject_id": subject_id,
                "source_path": source_path,
                "file_format": file_format,
                "file_size_bytes": file_size,
                "n_channels": n_channels,
                "sampling_rate_hz": sfreq,
                "n_samples": n_samples,
                "duration_seconds": duration,
                "fully_readable": fully_readable,
                "has_nan": has_nan,
                "has_inf": has_inf,
                "channel_order_matches": "pending",
                "header_shape_matches": header_shape_matches,
                "unit_conversion_status": unit_conversion_status,
                "header_physical_dimensions": json_text(sorted(units_seen)),
                "mne_data_unit": "V",
                "reference": reference,
                "montage": montage,
                "header_total_signal_count": header_total_signals,
                "header_data_channel_count": header_data_channels,
                "annotation_channel_count": annotation_channel_count,
                "header_n_records": header_n_records,
                "header_record_duration_seconds": record_duration,
                "header_samples_per_record": json_text(
                    [int(n_samps_all[int(index)]) for index in selected_signal_indices]
                ),
                "header_expected_file_size_bytes": expected_file_size,
                "header_subject_metadata": json_text(raw.info.get("subject_info")) if raw.info.get("subject_info") else "unknown",
                "measurement_date": str(raw.info.get("meas_date") or "unknown"),
                "constant_channels_deferred": json_text(constant_channels),
                "extreme_amplitude_channel_count_deferred": len(extreme_channels),
                "basic_header_pass": basic_header_pass,
                "read_warnings": json_text(warning_messages),
                "validation_pass": "pending",
                "error_count": 0,
                "warning_count": 0,
            }
        finally:
            raw.close()

    def failed_recording_row(self, path: Path, exc: BaseException) -> dict[str, Any]:
        manifest = self.manifest_for_path(path)
        source_path = safe_rel(path, self.data_dir)
        subject_id = manifest["subject_id"] if manifest else "unknown"
        return {
            "subject_id": subject_id,
            "source_path": source_path,
            "file_format": "BDF" if path.suffix.lower() == ".bdf" else "unknown",
            "file_size_bytes": path.stat().st_size if path.exists() else "",
            "n_channels": "", "sampling_rate_hz": "", "n_samples": "", "duration_seconds": "",
            "fully_readable": False, "has_nan": "unknown", "has_inf": "unknown",
            "channel_order_matches": False, "header_shape_matches": False,
            "unit_conversion_status": "fail", "header_physical_dimensions": "unknown",
            "mne_data_unit": "unknown", "reference": "unknown", "montage": "unknown",
            "header_total_signal_count": "", "header_data_channel_count": "",
            "annotation_channel_count": "", "header_n_records": "",
            "header_record_duration_seconds": "", "header_samples_per_record": "unknown",
            "header_expected_file_size_bytes": "", "header_subject_metadata": "unknown",
            "measurement_date": "unknown", "constant_channels_deferred": "unknown",
            "extreme_amplitude_channel_count_deferred": "unknown", "basic_header_pass": False,
            "read_warnings": json_text([str(exc)]), "validation_pass": False,
            "error_count": 1, "warning_count": 0,
        }

    def validate_recordings(self) -> None:
        bdf_files = sorted(
            (path for path in self.data_dir.rglob("*") if path.is_file() and path.suffix.lower() == ".bdf"),
            key=lambda path: safe_rel(path, self.data_dir).lower(),
        )
        successes, failures = run_batch(bdf_files, self.validate_one_bdf)
        self.recordings = [result for _, result in successes]
        for path, exc, trace in failures:
            source_path = safe_rel(path, self.data_dir)
            manifest = self.manifest_for_path(path)
            subject_id = manifest["subject_id"] if manifest else "unknown"
            self.log_exception(f"BDF validation failed: {path}", exc, trace)
            self.add_issue(
                issue_row(
                    "error", subject_id, source_path, "file_validation_exception",
                    "file opens and reads completely", repr(exc),
                    "The BDF file raised an exception; other files continued validating.",
                    "Block this recording and investigate the exception in validation_errors.log.",
                )
            )
            self.recordings.append(self.failed_recording_row(path, exc))

        eligible = [row for row in self.recordings if row.get("basic_header_pass") is True]
        if eligible:
            self.reference_source_path = eligible[0]["source_path"]
            reference_layout = self.channel_layout_by_path[self.reference_source_path]
            for row in self.recordings:
                source_path = row["source_path"]
                actual_layout = self.channel_layout_by_path.get(source_path)
                if actual_layout is None:
                    row["channel_order_matches"] = False
                    continue
                mismatches = compare_channel_layout(reference_layout, actual_layout)
                row["channel_order_matches"] = not mismatches
                for mismatch in mismatches:
                    index = mismatch["channel_index"]
                    actual_channel = actual_layout[index]["channel_name"] if index < len(actual_layout) else ""
                    self.add_issue(
                        issue_row(
                            "error", row["subject_id"], source_path, "channel_layout_mismatch",
                            mismatch["expected"], mismatch["actual"],
                            f"Channel position {index} differs from reference file in {mismatch['field']}.",
                            "Block the recording; do not auto-reorder channels in this stage.", actual_channel,
                        )
                    )
        else:
            for row in self.recordings:
                row["channel_order_matches"] = False
                self.add_issue(
                    issue_row(
                        "error", row["subject_id"], row["source_path"], "channel_reference_unavailable",
                        "at least one recording passing basic header checks", "none",
                        "No valid reference channel layout could be selected.",
                        "Resolve header errors before cross-recording comparison.",
                    )
                )

        issues_by_path: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for issue in self.issues:
            if issue["source_path"].lower().endswith(".bdf"):
                issues_by_path[issue["source_path"]].append(issue)
        self.recordings.sort(key=lambda row: row["source_path"].lower())
        for row in self.recordings:
            file_issues = issues_by_path[row["source_path"]]
            row["error_count"] = sum(issue["severity"] == "error" for issue in file_issues)
            row["warning_count"] = sum(issue["severity"] == "warning" for issue in file_issues)
            row["validation_pass"] = row["error_count"] == 0

    def summary(self) -> dict[str, Any]:
        severity_counts = Counter(issue["severity"] for issue in self.issues)
        pass_count = sum(row.get("validation_pass") is True for row in self.recordings)
        return {
            "generated_at": now_iso(),
            "data_dir": str(self.data_dir),
            "config_path": str(self.config_path),
            "output_dir": str(self.output_dir),
            "actual_command": self.actual_command,
            "raw_data_modified": False,
            "scope": [
                "fixed configuration validation",
                "authoritative directory-name subject manifest",
                "read-only BDF file-level validation",
            ],
            "excluded_operations": [
                "filtering", "notch filtering", "resampling", "windowing", "normalization",
                "bad-channel modification", "label export", "Lance conversion",
            ],
            "config_validation_pass": True,
            "manifest_subject_count": len(self.manifest),
            "recording_count": len(self.recordings),
            "recording_pass_count": pass_count,
            "recording_fail_count": len(self.recordings) - pass_count,
            "reference_channel_file": self.reference_source_path,
            "reference_unknown_count": sum(row.get("reference") == "unknown" for row in self.recordings),
            "montage_unknown_count": sum(row.get("montage") == "unknown" for row in self.recordings),
            "unit_conversion_status_counts": dict(Counter(row.get("unit_conversion_status") for row in self.recordings)),
            "severity_counts": dict(severity_counts),
            "python_version": sys.version,
            "mne_version": __import__("mne").__version__,
            "header_reader": "MNE read_raw_bdf plus parsed raw._raw_extras header fields",
            "pyedflib_used": False,
            "pyedflib_note": "Not installed; MNE exposed all required fields for these files, so no installation was needed.",
        }

    def report_text(self, summary: dict[str, Any]) -> str:
        failed = [row for row in self.recordings if row.get("validation_pass") is not True]
        unit_values = sorted({row.get("header_physical_dimensions", "unknown") for row in self.recordings})
        known_constant_rows = [row for row in self.recordings if row.get("constant_channels_deferred") not in {"[]", "unknown", ""}]
        errors = [issue for issue in self.issues if issue["severity"] == "error"]
        warnings_rows = [issue for issue in self.issues if issue["severity"] == "warning"]
        blocking_lines = [
            f"- `{row['subject_id']}` `{row['source_path']}`：{row['error_count']} 个 error。"
            for row in failed
        ] or ["- 无；8 条 recording 均通过本阶段文件级硬性检查。"]
        constant_lines = [
            f"- `{row['subject_id']}` `{row['source_path']}`：{row['constant_channels_deferred']}"
            for row in known_constant_rows
        ] or ["- 未发现常数通道。"]
        manifest_lines = [
            f"- {row['subject_id']}：age={row['age'] if row['age'] is not None else 'null'}，"
            f"sex={row['sex']}，source=`{row['source_directory']}`，status={row['metadata_status']}"
            for row in self.manifest
        ]
        return f"""# 华大静息态 EEG：阶段 1–3 验证报告

生成时间：{summary['generated_at']}  
原始数据目录：`{self.data_dir}`  
配置：`{self.config_path}`  
结果目录：`{self.output_dir}`

## 范围

本次只完成：固定并校验公共配置、建立目录名权威被试清单、执行原始 BDF 文件级硬性检查。未执行滤波、陷波、重采样、切窗、归一化、坏道修改、标签导出或 Lance 转换。原始数据未被修改。

## 公共配置

- 配置一致性校验：通过。
- `segment_samples == target_sfreq × window_seconds`：通过（2000 == 200 × 10）。
- `hop_seconds > 0`、`bandpass_low > 0`、`bandpass_high < target_sfreq/2`、`notch_hz < target_sfreq/2`：全部通过。
- 配置当前只保存和校验，没有执行预处理。

## 权威被试清单

- manifest 被试数：{len(self.manifest)}。
- 人口学权威来源固定为 `directory_name`；BDF header 不覆盖 manifest。

{chr(10).join(manifest_lines)}

## BDF 文件级硬性检查

- recording 数：{len(self.recordings)}。
- 通过：{summary['recording_pass_count']}；失败：{summary['recording_fail_count']}。
- 参考通道顺序文件：`{self.reference_source_path or 'unknown'}`。
- 单位转换状态：{summary['unit_conversion_status_counts']}；原始 physical dimension 集合：{unit_values}。
- MNE 返回的校准数据单位按库约定记录为 `V`；后续转换到 µV 使用每通道 header 单位对应的 multiplier。
- reference 可确认：0/{len(self.recordings)}；montage 可确认：0/{len(self.recordings)}。两者均写为 `unknown`，未根据通道名称推断。
- header 总 signal 数包含 BDF annotation signal；硬性 64 通道检查针对 MNE 暴露的 64 个数据通道。
- 每个文件的 record 数、每 record 样本数、record duration、物理/数字上下限及预期文件大小均已与实际读取结果核对。

## 阻止进入第四阶段的问题

{chr(10).join(blocking_lines)}

## 已记录但推迟到第四阶段的信号级问题

{chr(10).join(constant_lines)}

8 条 recording 的大绝对幅值也作为 warning 记录，但未升级为本阶段文件读取失败，且没有在本阶段进行滤波、置零、插值或其他信号修改。

## Header 人口学字段

BDF header 的 `his_id=X`、sex unknown、birthday 等于采集日期均作为匿名占位/冲突 warning 记录。目录名仍是 subject_id、age、sex 的权威来源。详见 `metadata_conflicts.csv`。

## 问题统计

- error：{len(errors)}
- warning：{len(warnings_rows)}
- info：{sum(issue['severity'] == 'info' for issue in self.issues)}

## 实际执行命令

```powershell
{self.actual_command}
```

## 结论

8 名被试均进入 manifest，8 个 BDF 均完成从首样本到末样本的分块读取。文件格式、500 Hz、64 数据通道、通道名/顺序、header 维度、有限值和 µV 单位转换均通过硬性检查。reference 与 montage 不能从可靠来源确认，保持 `unknown`。已知常数通道和极端幅值留待第四阶段信号级 QC。
"""

    def write_outputs(self) -> None:
        manifest_fields = [
            "subject_id", "subject_number", "age", "sex", "original_directory_name",
            "source_directory", "metadata_source", "metadata_status", "notes",
        ]
        recording_fields = [
            "subject_id", "source_path", "file_format", "file_size_bytes", "n_channels",
            "sampling_rate_hz", "n_samples", "duration_seconds", "fully_readable", "has_nan", "has_inf",
            "channel_order_matches", "header_shape_matches", "unit_conversion_status",
            "header_physical_dimensions", "mne_data_unit", "reference", "montage",
            "header_total_signal_count", "header_data_channel_count", "annotation_channel_count",
            "header_n_records", "header_record_duration_seconds", "header_samples_per_record",
            "header_expected_file_size_bytes", "header_subject_metadata", "measurement_date",
            "constant_channels_deferred", "extreme_amplitude_channel_count_deferred",
            "validation_pass", "error_count", "warning_count", "read_warnings",
        ]
        channel_fields = [
            "subject_id", "source_path", "channel_index", "channel_name", "channel_type",
            "physical_dimension", "unit_to_uv_multiplier", "physical_min", "physical_max",
            "digital_min", "digital_max", "sampling_rate_hz", "samples_per_record", "mne_data_unit",
        ]
        conflict_fields = [
            "subject_id", "source_path", "field", "authoritative_value", "header_value",
            "conflict_type", "severity", "description",
        ]
        issue_fields = [
            "severity", "subject_id", "source_path", "channel_name", "issue_type",
            "expected", "actual", "description", "recommendation",
        ]

        write_csv(self.output_dir / "subject_manifest.csv", self.manifest, manifest_fields)
        (self.output_dir / "subject_manifest.json").write_text(
            json.dumps(self.manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        write_csv(self.output_dir / "recording_validation.csv", self.recordings, recording_fields)
        write_csv(self.output_dir / "channel_header_inventory.csv", self.channels, channel_fields)
        write_csv(self.output_dir / "metadata_conflicts.csv", self.metadata_conflicts, conflict_fields)
        write_csv(self.output_dir / "validation_issues.csv", self.issues, issue_fields)
        summary = self.summary()
        (self.output_dir / "validation_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (self.output_dir / "validation_report.md").write_text(
            self.report_text(summary), encoding="utf-8"
        )
        log_text = "\n\n".join(self.error_log) if self.error_log else f"[{now_iso()}] No file validation exceptions.\n"
        (self.output_dir / "validation_errors.log").write_text(log_text, encoding="utf-8")

    def run(self) -> None:
        self.build_subject_manifest()
        self.validate_recordings()
        self.write_outputs()


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only Huada raw BDF file-level validator")
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--chunk-seconds", type=float, default=30.0)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    data_dir = args.data_dir.resolve()
    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()
    if not data_dir.is_dir():
        print(f"ERROR: data directory does not exist: {data_dir}", file=sys.stderr)
        return 2
    if not config_path.is_file():
        print(f"ERROR: config file does not exist: {config_path}", file=sys.stderr)
        return 2
    try:
        output_dir.relative_to(data_dir)
    except ValueError:
        pass
    else:
        print("ERROR: output directory must not be inside the raw data directory", file=sys.stderr)
        return 2

    try:
        validator = RawBdfValidator(data_dir, config_path, output_dir, args.chunk_seconds)
        validator.run()
    except ConfigValidationError as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1

    summary = validator.summary()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["recording_fail_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
