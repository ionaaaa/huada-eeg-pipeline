#!/usr/bin/env python3
"""Read-only staged audit for the authoritative MIPDB 20–35 age subset.

The audit fails closed before signal access when the download/root metadata is
incomplete.  Header inspection never preloads signal data.  Signal smoke is
deterministic, capped, and decodes selected recordings from start to finish in
bounded chunks without preprocessing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import re
import subprocess
import sys
import traceback
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import mne
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_dortmund_age_subset import (
    RunningStats,
    classify_materialization,
    compare_snapshots,
    ensure_new_output,
    git_info,
    sha256_file,
    snapshot_tree,
    utc_now,
    validate_numeric_age,
    write_csv,
)


SCRIPT_VERSION = "1.1.0"
DATASET_ID = "nm000153"
OFFICIAL_DATASET_PAGE = "https://fcon_1000.projects.nitrc.org/indi/cmi_eeg/eeg.html"
OFFICIAL_NEMAR_RELEASE = "https://www.nemar.org/dataset/nm000153?v=v1.0.2"
OFFICIAL_PROTOCOL_DOC = "https://fcon_1000.projects.nitrc.org/indi/cmi_eeg/_static/MIPDB_EEG_Readme.zip"
OFFICIAL_CONVERSION_CODE = "https://data.nemar.org/nm000153/v1.0.2/code/convert_mipdb.py"
DATA_PAPER = "https://doi.org/10.1038/sdata.2017.40"
PRIORITY_RECORDINGS = (
    "sub-A00058775/eeg/sub-A00058775_task-block01_eeg.eeg",
    "sub-A00062842/eeg/sub-A00062842_task-block03_eeg.eeg",
)
ROOT_METADATA = ("dataset_description.json", "participants.tsv", "participants.json")
TEMP_SUFFIXES = (".part", ".tmp", ".partial", ".crdownload")
TASK_RE = re.compile(r"^(?P<subject>sub-[^_]+)_task-(?P<task>[^_]+)(?:_run-(?P<run>[^_]+))?_eeg\.vhdr$", re.I)


CSV_FIELDS: dict[str, list[str]] = {
    "subject_manifest.csv": [
        "subject_id", "allowlisted", "directory_exists", "recording_count", "tasks", "exact_age",
        "age_unit", "age_source", "age_status", "sex", "sex_status", "download_status",
    ],
    "download_integrity.csv": [
        "subject_id", "relative_path", "file_role", "size_bytes", "expected_size_bytes",
        "manifest_status", "materialization_status", "companion_status", "frame_alignment_status",
        "priority_recording", "checksum", "checksum_source", "status", "detail",
    ],
    "metadata_label_inventory.csv": [
        "subject_id", "field", "value", "unit", "source_path", "source_priority", "validation_status", "detail",
    ],
    "recording_inventory.csv": [
        "subject_id", "task", "recording", "session", "run", "source_path", "format", "binary_path",
        "marker_path", "sidecar_path", "events_path", "sampling_frequency_hz", "n_channels_total",
        "n_eeg_channels", "n_aux_channels", "channel_names", "channel_order", "channel_types", "units",
        "reference", "ground", "montage", "coordinate_system", "n_samples_header", "header_duration_seconds",
        "binary_size_bytes", "binary_samples_inferred", "binary_duration_seconds", "sidecar_duration_seconds",
        "physical_limits", "digital_limits", "marker_status", "events_status", "header_signature",
        "header_read_success", "duplicate_channels", "canonicalization_collisions", "short_or_truncated_status",
    ],
    "task_inventory.csv": [
        "task", "recording_count", "subject_count", "paradigm", "eyes_condition", "continuity",
        "pre_post_session_semantics", "mapping_status", "confidence", "evidence", "notes",
    ],
    "channel_header_inventory.csv": [
        "subject_id", "task", "recording", "source_path", "channel_index", "channel_name", "channel_type",
        "unit", "resolution", "reference", "sampling_frequency_hz", "header_signature",
    ],
    "header_signature_inventory.csv": [
        "header_signature", "format", "sampling_frequency_hz", "n_channels_total", "n_eeg_channels",
        "n_aux_channels", "ordered_channel_names", "channel_types", "units", "reference", "ground",
        "montage", "recording_count", "subject_count", "tasks",
    ],
    "signal_sampling_plan.csv": [
        "selection_order", "subject_id", "exact_age", "sex", "task", "recording", "source_path",
        "duration_seconds", "sampling_frequency_hz", "header_signature", "selection_reason", "age_sex_stratified",
        "mandatory", "selected", "read_completed",
    ],
    "signal_smoke_recordings.csv": [
        "subject_id", "task", "recording", "source_path", "source_samples_header", "actual_samples_read",
        "duration_seconds", "chunks_read", "fully_decoded", "samples_match_header", "unit_conversion_status",
        "has_nan", "has_inf", "all_zero_channels", "constant_channels", "near_zero_variance_channels",
        "duplicate_channel_groups", "candidate_saturation_channels", "marker_or_status_in_eeg_picks",
        "extreme_abs_max_uv", "hard_error_count", "warning_count", "smoke_status",
    ],
    "signal_smoke_channel_statistics.csv": [
        "subject_id", "task", "recording", "source_path", "channel_index", "channel_name", "channel_type",
        "unit", "n_samples", "min_uv", "max_uv", "mean_uv", "std_uv", "has_nan", "has_inf", "all_zero",
        "strict_constant", "near_zero_variance", "physical_min_uv", "physical_max_uv", "physical_limit_hit_count",
        "extreme_amplitude_warning", "dc_offset_warning", "duplicate_group",
    ],
    "adaptive_expansion_log.csv": [
        "adaptive_expansion_triggered", "adaptive_expansion_reason", "recommended_expansion_scope",
        "approval_required", "auto_executed",
    ],
    "audit_issues.csv": [
        "severity", "stage", "subject_id", "task", "recording", "source_path", "issue_type", "description",
        "recommendation",
    ],
}


def normalize_subject(value: str) -> str:
    value = value.strip()
    if not value:
        return ""
    return value if value.startswith("sub-") else f"sub-{value}"


def load_allowlist(path: Path) -> list[str]:
    values = [normalize_subject(line.split("#", 1)[0]) for line in path.read_text(encoding="utf-8-sig").splitlines()]
    values = [value for value in values if value]
    if not values:
        raise ValueError("subject allowlist is empty")
    if len(values) != len(set(values)):
        raise ValueError("subject allowlist contains duplicates")
    return values


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_task(path: Path) -> dict[str, str]:
    match = TASK_RE.match(path.name)
    if not match:
        return {"subject": path.parents[1].name if len(path.parents) > 1 else "", "task": "", "run": ""}
    return {key: (value or "") for key, value in match.groupdict().items()}


def parse_vhdr(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8-sig")
    section = ""
    common: dict[str, str] = {}
    binary: dict[str, str] = {}
    channels: list[dict[str, Any]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip("\ufeff\r\n")
        if not line or line.lstrip().startswith(";"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip().lower()
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if section == "common infos":
            common[key] = value
        elif section == "binary infos":
            binary[key] = value
        elif section == "channel infos" and key.lower().startswith("ch"):
            parts = value.split(",")
            channels.append({
                "name": parts[0].replace("\\1", ",").rstrip(),
                "reference": parts[1].strip() if len(parts) > 1 else "",
                "resolution": parts[2].strip() if len(parts) > 2 else "",
                "unit": parts[3].strip() if len(parts) > 3 else "",
            })
    n_channels = int(common.get("NumberOfChannels", len(channels)))
    interval_us = float(common.get("SamplingInterval", "nan"))
    sfreq = 1_000_000.0 / interval_us if interval_us > 0 else math.nan
    binary_format = binary.get("BinaryFormat", "")
    bytes_per_sample = {"INT_16": 2, "INT_32": 4, "IEEE_FLOAT_32": 4}.get(binary_format.upper())
    return {
        "data_file": common.get("DataFile", ""),
        "marker_file": common.get("MarkerFile", ""),
        "data_format": common.get("DataFormat", ""),
        "orientation": common.get("DataOrientation", ""),
        "n_channels": n_channels,
        "sampling_frequency": sfreq,
        "binary_format": binary_format,
        "bytes_per_sample": bytes_per_sample,
        "channels": channels,
    }


def binary_geometry(vhdr: dict[str, Any], binary_size: int) -> tuple[int | None, int | None]:
    bps = vhdr.get("bytes_per_sample")
    n_channels = vhdr.get("n_channels")
    if not bps or not n_channels:
        return None, None
    frame = int(bps) * int(n_channels)
    return binary_size // frame, binary_size % frame


def canonical_name(name: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", name.upper())


def canonicalization_collisions(names: Sequence[str]) -> list[list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for name in names:
        grouped[canonical_name(name)].append(name)
    return [values for values in grouped.values() if len(values) > 1 and len(set(values)) > 1]


def channel_flags(stat: RunningStats) -> dict[str, bool]:
    return {
        "has_nan": stat.has_nan,
        "has_inf": stat.has_inf,
        "all_zero": stat.minimum == 0.0 and stat.maximum == 0.0,
        "strict_constant": stat.minimum == stat.maximum,
        "near_zero_variance": stat.variance <= 1e-12,
        "extreme_amplitude_warning": max(abs(stat.minimum), abs(stat.maximum)) > 10_000 if stat.count else False,
        "dc_offset_warning": abs(stat.mean) > 1_000 if stat.count else False,
    }


def _manifest_records(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, list):
        for item in value:
            yield from _manifest_records(item)
    elif isinstance(value, dict):
        keys = {str(key).lower(): key for key in value}
        path_key = next((keys[key] for key in ("relative_path", "path", "filename", "key", "name") if key in keys), None)
        size_key = next((keys[key] for key in ("size_bytes", "size", "bytes", "content_length") if key in keys), None)
        if path_key is not None and size_key is not None:
            yield value
        for item in value.values():
            yield from _manifest_records(item)


def load_local_manifest(data_dir: Path, explicit_path: Path | None = None) -> tuple[Path | None, dict[str, dict[str, Any]]]:
    candidates = [
        explicit_path,
        data_dir.parents[1] / "manifest_v1.0.2.json" if len(data_dir.parents) > 1 else None,
        data_dir.parents[1] / "manifest" / "_v1.0.2.json" if len(data_dir.parents) > 1 else None,
        data_dir / ".nemar" / "availability-report.json",
        data_dir.parent / ".nemar" / "availability-report.json",
        data_dir.parents[1] / DATASET_ID / ".nemar" / "availability-report.json" if len(data_dir.parents) > 1 else data_dir,
    ]
    path = next((candidate for candidate in candidates if candidate is not None and candidate.is_file()), None)
    if path is None:
        return None, {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return path, {}
    lookup: dict[str, dict[str, Any]] = {}
    for row in _manifest_records(payload):
        lowered = {str(key).lower(): value for key, value in row.items()}
        raw_path = next((lowered[key] for key in ("relative_path", "path", "filename", "key", "name") if key in lowered), "")
        rel = str(raw_path).replace("\\", "/").lstrip("./")
        if rel:
            lookup[rel] = row
    return path, lookup


def discover_nemar_support(data_dir: Path) -> dict[str, Any]:
    candidates = [
        data_dir / ".nemar",
        data_dir.parent / ".nemar",
        data_dir.parents[1] / DATASET_ID / ".nemar" if len(data_dir.parents) > 1 else data_dir,
    ]
    directories: list[Path] = []
    for candidate in candidates:
        if candidate.is_dir() and candidate not in directories:
            directories.append(candidate)
    files = sorted(path for directory in directories for path in directory.iterdir() if path.is_file())
    snapshot = next((path for path in files if path.name == "http-snapshot.json"), None)
    failures = next((path for path in files if path.name == "http-download-failures.txt"), None)
    snapshot_payload = None
    if snapshot:
        try:
            snapshot_payload = json.loads(snapshot.read_text(encoding="utf-8-sig"))
        except Exception:
            snapshot_payload = None
    return {
        "directories": [str(path) for path in directories],
        "files": [str(path) for path in files],
        "snapshot_path": str(snapshot) if snapshot else None,
        "snapshot": snapshot_payload,
        "download_failure_log_path": str(failures) if failures else None,
        "download_failure_log_size_bytes": failures.stat().st_size if failures else 0,
    }


def manifest_size(row: dict[str, Any] | None) -> int | None:
    if not row:
        return None
    lowered = {str(key).lower(): value for key, value in row.items()}
    value = next((lowered[key] for key in ("size_bytes", "size", "bytes", "content_length") if key in lowered), None)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def manifest_checksum(row: dict[str, Any] | None) -> tuple[str, str]:
    if not row:
        return "", ""
    lowered = {str(key).lower(): value for key, value in row.items()}
    for key in ("sha256", "checksum", "md5", "etag"):
        if lowered.get(key):
            return str(lowered[key]), key
    return "", ""


def manifest_checksum_algorithm(row: dict[str, Any] | None, checksum_source: str) -> str:
    if not row:
        return ""
    lowered = {str(key).lower(): value for key, value in row.items()}
    algorithm = str(lowered.get("checksum_algorithm", "")).strip().lower()
    if algorithm:
        return algorithm
    return "sha256" if checksum_source == "sha256" else ("md5" if checksum_source == "md5" else "")


def preflight_dataset(
    data_dir: Path,
    allowlist: Sequence[str],
    explicit_manifest_path: Path | None = None,
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    integrity: list[dict[str, Any]] = []
    subjects: list[dict[str, Any]] = []

    def issue(severity: str, kind: str, description: str, recommendation: str = "", **context: Any) -> None:
        issues.append({
            "severity": severity, "stage": "0", "issue_type": kind, "description": description,
            "recommendation": recommendation, **context,
        })

    if not data_dir.is_dir():
        issue("error", "data_directory_missing", f"Data directory does not exist: {data_dir}", "Locate the completed local download")
        return {
            "passed": False, "issues": issues, "integrity": integrity, "subjects": subjects,
            "actual_subjects": [], "audit_subjects": [], "missing_subjects": list(allowlist),
            "missing_published_subjects": list(allowlist), "not_published_subjects": [], "extra_subjects": [],
            "manifest_path": None, "manifest_available": False, "dataset_description": {},
            "priority_checks": {}, "recording_count_discovered": 0, "nemar_support": {},
        }

    actual_subjects = sorted(path.name for path in data_dir.glob("sub-*") if path.is_dir())
    extra_subjects = sorted(set(actual_subjects) - set(allowlist))
    manifest_path, manifest = load_local_manifest(data_dir, explicit_manifest_path)
    manifest_subjects = sorted({rel.split("/", 1)[0] for rel in manifest if rel.startswith("sub-") and "/" in rel})
    if manifest:
        audit_subjects = [sid for sid in allowlist if sid in set(manifest_subjects)]
        not_published_subjects = sorted(set(allowlist) - set(manifest_subjects))
    else:
        audit_subjects = list(allowlist)
        not_published_subjects = []
    missing_subjects = sorted(set(allowlist) - set(actual_subjects))
    missing_published_subjects = sorted(set(audit_subjects) - set(actual_subjects))
    nemar_support = discover_nemar_support(data_dir)

    dataset_description: dict[str, Any] = {}
    if nemar_support["download_failure_log_size_bytes"]:
        issue(
            "warning", "local_download_failure_log_present",
            f"Non-empty local NEMAR download failure log: {nemar_support['download_failure_log_path']}",
            "Use as download-history evidence; it is not an availability manifest",
            source_path=nemar_support["download_failure_log_path"],
        )
    for name in ROOT_METADATA:
        path = data_dir / name
        if not path.is_file():
            integrity.append({"relative_path": name, "file_role": "root_metadata", "status": "missing", "detail": "required root metadata missing"})
            issue("error", "root_metadata_missing", f"Missing root metadata: {name}", "Restore the official root metadata", source_path=name)
        else:
            integrity.append({"relative_path": name, "file_role": "root_metadata", "size_bytes": path.stat().st_size, "status": "present"})
    description_path = data_dir / "dataset_description.json"
    if description_path.is_file():
        try:
            dataset_description = json.loads(description_path.read_text(encoding="utf-8-sig"))
            if not dataset_description.get("Version") or not dataset_description.get("Name"):
                issue("error", "dataset_version_undetermined", "dataset_description.json lacks Name or Version", "Restore official dataset_description.json", source_path="dataset_description.json")
        except Exception as exc:
            issue("error", "dataset_description_invalid", str(exc), "Restore a valid official dataset_description.json", source_path="dataset_description.json")

    for sid in missing_published_subjects:
        issue(
            "error", "published_allowlisted_subject_missing" if manifest else "allowlisted_subject_missing",
            f"Published allowlisted subject directory is missing: {sid}" if manifest else f"Missing allowlisted subject directory: {sid}",
            "Complete the local download; do not run signal smoke", subject_id=sid,
        )
    for sid in not_published_subjects:
        issue("info", "allowlisted_subject_not_published", f"Allowlisted subject is absent from the supplied release manifest: {sid}", "Retain in the authoritative allowlist but exclude from this release audit", subject_id=sid)
    if extra_subjects:
        issue("info", "non_allowlisted_subjects_present", f"Ignored subjects outside allowlist: {extra_subjects}", "Keep excluded from this audit")

    rows_by_rel: dict[str, dict[str, Any]] = {row["relative_path"]: row for row in integrity}
    priority_checks: dict[str, dict[str, Any]] = {}
    discovered_recordings = 0
    fatal = bool(missing_published_subjects) or any(row.get("severity") == "error" for row in issues)

    for sid in allowlist:
        subject_dir = data_dir / sid
        published = sid in set(audit_subjects)
        vhdrs = sorted(subject_dir.rglob("*_eeg.vhdr")) if subject_dir.is_dir() and published else []
        tasks = sorted({parse_task(path)["task"] for path in vhdrs})
        subjects.append({
            "subject_id": sid, "allowlisted": True, "directory_exists": subject_dir.is_dir(),
            "recording_count": len(vhdrs), "tasks": tasks,
            "download_status": (
                "not_published_in_release_manifest" if not published
                else ("present" if subject_dir.is_dir() else "missing_published_subject")
            ),
        })
        discovered_recordings += len(vhdrs)
        if published and subject_dir.is_dir() and not vhdrs:
            fatal = True
            issue("error", "subject_has_no_recordings", f"No BrainVision headers found for {sid}", "Complete the subject download", subject_id=sid)

        if not published or not subject_dir.is_dir():
            continue
        for path in sorted(p for p in subject_dir.rglob("*") if p.is_file()):
            rel = path.relative_to(data_dir).as_posix()
            status, _, detail = classify_materialization(path)
            manifest_row = manifest.get(rel)
            expected = manifest_size(manifest_row)
            checksum, checksum_source = manifest_checksum(manifest_row)
            manifest_status = "not_available" if not manifest else ("not_listed" if manifest_row is None else ("size_mismatch" if expected != path.stat().st_size else "present_size_match"))
            row = {
                "subject_id": sid, "relative_path": rel, "file_role": path.suffix.lower().lstrip("."),
                "size_bytes": path.stat().st_size, "expected_size_bytes": expected, "manifest_status": manifest_status,
                "materialization_status": status, "companion_status": "", "frame_alignment_status": "",
                "priority_recording": rel in PRIORITY_RECORDINGS, "checksum": checksum, "checksum_source": checksum_source,
                "status": "present", "detail": detail,
            }
            integrity.append(row)
            rows_by_rel[rel] = row
            if path.suffix.lower() in TEMP_SUFFIXES or any(path.name.lower().endswith(suffix) for suffix in TEMP_SUFFIXES):
                fatal = True
                row["status"] = "incomplete_temp_file"
                issue("error", "partial_or_temp_file", f"Incomplete/temp file present: {rel}", "Complete or repair the download outside this audit", subject_id=sid, source_path=rel)
            if status != "materialized":
                fatal = True
                row["status"] = status
                issue("error", f"file_{status}", f"{rel}: {detail}", "Repair the download outside this audit", subject_id=sid, source_path=rel)
            if expected is not None and expected != path.stat().st_size:
                fatal = True
                row["status"] = "size_mismatch"
                issue("error", "manifest_size_mismatch", f"{rel}: actual={path.stat().st_size}, expected={expected}", "Re-download the affected file", subject_id=sid, source_path=rel)

        for vhdr_path in vhdrs:
            rel_header = vhdr_path.relative_to(data_dir).as_posix()
            try:
                header = parse_vhdr(vhdr_path)
                data_path = vhdr_path.with_name(header["data_file"])
                marker_path = vhdr_path.with_name(header["marker_file"])
                companions = {"binary": data_path, "marker": marker_path}
                missing_companions = [role for role, path in companions.items() if not path.is_file()]
                companion_status = "complete" if not missing_companions else "missing:" + ",".join(missing_companions)
                if rel_header in rows_by_rel:
                    rows_by_rel[rel_header]["companion_status"] = companion_status
                if missing_companions:
                    fatal = True
                    issue("error", "brainvision_companion_missing", f"{rel_header} missing {missing_companions}", "Restore the complete .vhdr/.eeg/.vmrk triplet", subject_id=sid, source_path=rel_header)
                if data_path.is_file():
                    rel_data = data_path.relative_to(data_dir).as_posix()
                    samples, remainder = binary_geometry(header, data_path.stat().st_size)
                    data_row = rows_by_rel.get(rel_data)
                    if data_row is not None:
                        data_row["companion_status"] = companion_status
                        data_row["frame_alignment_status"] = "aligned" if remainder == 0 else f"remainder:{remainder}"
                    if remainder not in (0, None):
                        fatal = True
                        issue("error", "binary_frame_size_mismatch", f"{rel_data} has {remainder} trailing bytes outside complete frames", "Re-download the EEG binary", subject_id=sid, source_path=rel_data)
                    if samples == 0:
                        fatal = True
                        issue("error", "zero_sample_binary", f"{rel_data} has zero complete samples", "Re-download the EEG binary", subject_id=sid, source_path=rel_data)
                    if rel_data in PRIORITY_RECORDINGS:
                        checksum_algorithm = manifest_checksum_algorithm(manifest.get(rel_data), data_row.get("checksum_source", "") if data_row else "")
                        actual_checksum = ""
                        checksum_match: bool | str = "not_available"
                        expected_checksum = data_row.get("checksum", "") if data_row else ""
                        if expected_checksum and checksum_algorithm == "sha256":
                            actual_checksum = sha256_file(data_path)
                            checksum_match = actual_checksum.lower() == str(expected_checksum).lower()
                            if not checksum_match:
                                fatal = True
                                issue("error", "priority_checksum_mismatch", f"{rel_data}: SHA256 differs from release manifest", "Repair the affected file outside this audit", subject_id=sid, source_path=rel_data)
                        priority_checks[rel_data] = {
                            "present": True, "size_bytes": data_path.stat().st_size, "manifest_status": data_row.get("manifest_status") if data_row else "unknown",
                            "frame_aligned": remainder == 0, "companions_complete": not missing_companions,
                            "header_parseable": True, "signal_readable": "not_tested_in_preflight",
                            "checksum_algorithm": checksum_algorithm, "expected_checksum": expected_checksum,
                            "actual_checksum": actual_checksum, "checksum_match": checksum_match,
                        }
            except Exception as exc:
                fatal = True
                issue("error", "vhdr_parse_failed", f"{rel_header}: {exc}", "Repair the header or complete download", subject_id=sid, source_path=rel_header)

    for rel in PRIORITY_RECORDINGS:
        if rel.split("/", 1)[0] not in set(audit_subjects):
            continue
        if rel not in priority_checks:
            exists = (data_dir / Path(rel)).is_file()
            priority_checks[rel] = {
                "present": exists, "size_bytes": (data_dir / Path(rel)).stat().st_size if exists else None,
                "manifest_status": rows_by_rel.get(rel, {}).get("manifest_status", "not_available"),
                "frame_aligned": False, "companions_complete": False, "header_parseable": False,
                "signal_readable": "not_tested_in_preflight",
            }
            if not exists:
                fatal = True
                issue("error", "priority_recording_missing", f"Priority recording missing: {rel}", "Complete the local download", subject_id=rel.split("/")[0], source_path=rel)

    optional_manifest_missing = 0
    if manifest:
        allow_prefixes = tuple(f"{sid}/" for sid in audit_subjects)
        for rel, row in manifest.items():
            if rel.startswith(allow_prefixes) and not (data_dir / Path(rel)).is_file():
                required_raw = Path(rel).suffix.lower() in {".eeg", ".vhdr", ".vmrk"}
                if required_raw:
                    fatal = True
                else:
                    optional_manifest_missing += 1
                integrity.append({
                    "subject_id": rel.split("/", 1)[0], "relative_path": rel, "file_role": "manifest_expected",
                    "expected_size_bytes": manifest_size(row), "manifest_status": "missing",
                    "status": "missing_required_raw" if required_raw else "missing_optional_metadata",
                    "detail": "release-manifest entry absent locally",
                })
                if required_raw:
                    issue("error", "manifest_required_raw_file_missing", f"Required raw file missing locally: {rel}", "Complete the local download", subject_id=rel.split("/", 1)[0], source_path=rel)
    if optional_manifest_missing:
        issue(
            "warning", "release_metadata_files_not_downloaded",
            f"{optional_manifest_missing} release-manifest metadata/sidecar files are absent locally; raw BrainVision triplets are assessed separately",
            "Obtain the official sidecars if event-level or block-semantic validation requires them",
        )

    return {
        "passed": not fatal, "issues": issues, "integrity": integrity, "subjects": subjects,
        "actual_subjects": actual_subjects, "audit_subjects": audit_subjects,
        "manifest_subjects": manifest_subjects, "missing_subjects": missing_subjects,
        "missing_published_subjects": missing_published_subjects,
        "not_published_subjects": not_published_subjects, "extra_subjects": extra_subjects,
        "manifest_path": str(manifest_path) if manifest_path else None, "manifest_available": bool(manifest),
        "nemar_support": nemar_support,
        "dataset_description": dataset_description, "priority_checks": priority_checks,
        "recording_count_discovered": discovered_recordings,
        "optional_manifest_missing_count": optional_manifest_missing,
    }


def resolve_age_labels(data_dir: Path, allowlist: Sequence[str]) -> dict[str, Any]:
    participant_path = data_dir / "participants.tsv"
    participant_rows = {row.get("participant_id", ""): row for row in read_tsv(participant_path)}
    public_candidates = [data_dir.parents[1] / "MIPDB_PublicFile.csv"] if len(data_dir.parents) > 1 else []
    public_path = next((path for path in public_candidates if path.is_file()), None)
    public_rows = {normalize_subject(row.get("ID", "")): row for row in read_csv(public_path)} if public_path else {}
    labels: dict[str, dict[str, Any]] = {}
    inventory: list[dict[str, Any]] = []
    for sid in allowlist:
        participant = participant_rows.get(sid, {})
        valid, age, reason = validate_numeric_age(participant.get("age"))
        age_source = str(participant_path) if valid else ""
        priority = "participants.tsv"
        if not valid:
            public = public_rows.get(sid, {})
            valid, age, reason = validate_numeric_age(public.get("Age"))
            if valid:
                age_source = str(public_path)
                priority = "explicit_local_public_metadata_fallback"
        sex = str(participant.get("sex", "")).strip()
        sex_status = "documented" if sex.upper() in {"F", "M"} else "missing"
        if sex_status == "missing" and sid in public_rows and str(public_rows[sid].get("Sex", "")).strip().lower() not in {"", "n/a", "na"}:
            sex = str(public_rows[sid].get("Sex", "")).strip()
            sex_status = "raw_numeric_code_unmapped_no_local_schema"
        labels[sid] = {
            "age": age, "age_unit": "years" if valid else "", "age_source": age_source,
            "age_status": "exact_numeric" if valid else reason, "sex": sex, "sex_status": sex_status,
        }
        inventory.append({
            "subject_id": sid, "field": "age", "value": age if valid else participant.get("age", ""),
            "unit": "years" if valid else "", "source_path": age_source or str(participant_path),
            "source_priority": priority, "validation_status": "passed" if valid else reason,
            "detail": "Exact subject-level age; no midpoint/imputation" if valid else "Exact age unavailable",
        })
        inventory.append({
            "subject_id": sid, "field": "sex", "value": sex, "unit": "", "source_path": str(participant_path) if sex_status == "documented" else (str(public_path) if public_path else str(participant_path)),
            "source_priority": "participants.tsv" if sex_status == "documented" else "local fallback",
            "validation_status": sex_status, "detail": "Numeric sex codes are retained but not interpreted without a schema",
        })
    exact_ready = all(labels[sid]["age_status"] == "exact_numeric" for sid in allowlist)
    sex_ready = all(labels[sid]["sex_status"] == "documented" for sid in allowlist)
    return {
        "labels": labels, "inventory": inventory, "exact_age_label_validation": "passed" if exact_ready else "failed",
        "exact_age_labels_ready": exact_ready, "sex_labels_ready": sex_ready,
        "age_source": str(participant_path) if all(labels[sid]["age_source"] == str(participant_path) for sid in allowlist) else (str(public_path) if public_path else str(participant_path)),
        "public_metadata_fallback": str(public_path) if public_path else None,
    }


def build_sampling_plan(
    recordings: Sequence[dict[str, Any]],
    max_recordings: int = 12,
    smoke_subjects: int = 5,
    exact_age_ready: bool = False,
    sex_ready: bool = False,
) -> list[dict[str, Any]]:
    if max_recordings < 1 or max_recordings > 12:
        raise ValueError("max_signal_recordings must be between 1 and 12")
    if smoke_subjects < 3 or smoke_subjects > 5:
        raise ValueError("smoke_subjects must be between 3 and 5")
    available = sorted((dict(row) for row in recordings if row.get("header_read_success")), key=lambda row: (row["subject_id"], row["task"], row["source_path"]))
    priority_vhdr = {path[:-4] + ".vhdr" for path in PRIORITY_RECORDINGS}
    mandatory = [row for row in available if row["source_path"] in priority_vhdr]
    selected_subjects = {row["subject_id"] for row in mandatory}
    subject_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in available:
        subject_rows[row["subject_id"]].append(row)
    subject_meta = []
    for sid, rows in subject_rows.items():
        subject_meta.append({"subject_id": sid, "age": rows[0].get("exact_age"), "sex": rows[0].get("sex", "")})
    subject_meta.sort(key=lambda row: row["subject_id"])
    target_subject_count = min(smoke_subjects, len(subject_meta))
    if exact_age_ready and subject_meta:
        numeric = [row for row in subject_meta if row.get("age") is not None]
        ages = sorted(float(row["age"]) for row in numeric)
        anchors = [ages[0], float(np.median(ages)), ages[-1]] if ages else []
        for anchor in anchors:
            candidates = [row for row in numeric if row["subject_id"] not in selected_subjects]
            if candidates:
                chosen = min(candidates, key=lambda row: (abs(float(row["age"]) - anchor), row["subject_id"]))
                selected_subjects.add(chosen["subject_id"])
    if sex_ready:
        for target_sex in ("F", "M"):
            if not any(row["subject_id"] in selected_subjects and str(row.get("sex", "")).upper() == target_sex for row in subject_meta):
                candidate = next((row for row in subject_meta if row["subject_id"] not in selected_subjects and str(row.get("sex", "")).upper() == target_sex), None)
                if candidate:
                    selected_subjects.add(candidate["subject_id"])
    for row in subject_meta:
        if len(selected_subjects) >= target_subject_count:
            break
        selected_subjects.add(row["subject_id"])

    pool = [row for row in available if row["subject_id"] in selected_subjects]
    chosen: list[tuple[dict[str, Any], str, bool]] = []
    seen: set[str] = set()

    def duration_seconds(row: dict[str, Any]) -> float | None:
        value = row.get("duration_seconds")
        if value is None:
            value = row.get("header_duration_seconds")
        return float(value) if value is not None else None

    def add(row: dict[str, Any], reason: str, required: bool = False) -> None:
        if row["source_path"] not in seen and len(chosen) < max_recordings:
            chosen.append((row, reason, required))
            seen.add(row["source_path"])

    for row in mandatory:
        add(row, "mandatory priority recording", True)
    # Reserve subject coverage before greedy task/signature coverage can fill
    # the bounded plan. Prefer block01 so the inferred resting candidate is
    # represented across the deterministic age anchors.
    covered_subjects = {row["subject_id"] for row, _, _ in chosen}
    for sid in sorted(selected_subjects):
        if sid in covered_subjects:
            continue
        subject_candidates = [row for row in pool if row["subject_id"] == sid]
        candidate = next((row for row in subject_candidates if row.get("task") == "block01"), subject_candidates[0] if subject_candidates else None)
        if candidate:
            add(candidate, "subject/age-anchor coverage")
            covered_subjects.add(sid)
    if pool:
        add(min(pool, key=lambda row: (duration_seconds(row) if duration_seconds(row) is not None else math.inf, row["source_path"])), "shortest recording among selected subjects")
        add(max(pool, key=lambda row: (duration_seconds(row) if duration_seconds(row) is not None else -math.inf, row["source_path"])), "longest recording among selected subjects")
    covered_signatures = {row.get("header_signature") for row, _, _ in chosen}
    covered_tasks = {row.get("task") for row, _, _ in chosen}
    covered_rates = {row.get("sampling_frequency_hz") for row, _, _ in chosen}
    for row in pool:
        features = []
        if row.get("header_signature") not in covered_signatures:
            features.append("header signature")
        if row.get("sampling_frequency_hz") not in covered_rates:
            features.append("sampling rate")
        if row.get("task") not in covered_tasks:
            features.append("task/block")
        if features:
            add(row, "coverage: " + ", ".join(features))
            covered_signatures.add(row.get("header_signature"))
            covered_rates.add(row.get("sampling_frequency_hz"))
            covered_tasks.add(row.get("task"))
    for row in pool:
        add(row, "deterministic fill by subject/task/path")

    stratified = exact_age_ready and sex_ready
    plan: list[dict[str, Any]] = []
    for order, (row, reason, required) in enumerate(chosen, 1):
        plan.append({
            "selection_order": order, "subject_id": row["subject_id"], "exact_age": row.get("exact_age"),
            "sex": row.get("sex", ""), "task": row.get("task", ""), "recording": row.get("recording", ""),
            "source_path": row["source_path"], "duration_seconds": duration_seconds(row),
            "sampling_frequency_hz": row.get("sampling_frequency_hz"), "header_signature": row.get("header_signature", ""),
            "selection_reason": reason, "age_sex_stratified": stratified, "mandatory": required,
            "selected": True, "read_completed": False,
        })
    return plan


def recommend_expansion(signal_rows: Sequence[dict[str, Any]], channel_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    reasons: list[str] = []
    scopes: list[str] = []
    for row in signal_rows:
        if not row.get("fully_decoded") or not row.get("samples_match_header"):
            reasons.append(f"decode/truncation: {row.get('recording')}")
            scopes.append("associated BrainVision triplet and same task/block")
        if row.get("has_nan") or row.get("has_inf"):
            reasons.append(f"non-finite values: {row.get('recording')}")
            scopes.append("same header signature")
        if row.get("constant_channels") or row.get("all_zero_channels"):
            reasons.append(f"constant channels: {row.get('recording')}")
            scopes.append("same channels across additional stratified subjects")
    repeated = Counter(row.get("channel_name") for row in channel_rows if row.get("near_zero_variance") or row.get("extreme_amplitude_warning"))
    repeated = Counter({key: count for key, count in repeated.items() if key and count >= 2})
    if repeated:
        reasons.append(f"repeated channel warnings: {dict(repeated)}")
        scopes.append("affected channels in additional subjects")
    if not reasons:
        return [{
            "adaptive_expansion_triggered": False, "adaptive_expansion_reason": "",
            "recommended_expansion_scope": "", "approval_required": True, "auto_executed": False,
        }]
    return [{
        "adaptive_expansion_triggered": True, "adaptive_expansion_reason": "; ".join(sorted(set(reasons))),
        "recommended_expansion_scope": "; ".join(sorted(set(scopes))), "approval_required": True,
        "auto_executed": False,
    }]


class Audit:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.repo = Path(__file__).resolve().parents[1]
        self.data_dir = Path(args.data_dir).resolve()
        self.allowlist_path = Path(args.subject_allowlist).resolve()
        self.allowlist = load_allowlist(self.allowlist_path)
        self.manifest_path = Path(args.manifest).resolve() if getattr(args, "manifest", None) else None
        self.audit_subjects = list(self.allowlist)
        self.output_dir = ensure_new_output(Path(args.output_dir).resolve())
        self.rows: dict[str, list[dict[str, Any]]] = {name: [] for name in CSV_FIELDS}
        self.summary: dict[str, Any] = {}
        self.errors: list[str] = []
        self.logs: list[str] = []
        self.recordings: list[dict[str, Any]] = []
        self.labels: dict[str, dict[str, Any]] = {}
        self.source_before = snapshot_tree(self.data_dir) if self.data_dir.is_dir() else {}
        self.start_time = utc_now()
        self.preflight_passed = False

    def log(self, message: str) -> None:
        line = f"{datetime.now().isoformat(timespec='seconds')} {message}"
        self.logs.append(line)
        print(message, flush=True)

    def issue(self, severity: str, stage: str, issue_type: str, description: str, recommendation: str = "", **context: Any) -> None:
        self.rows["audit_issues.csv"].append({
            "severity": severity, "stage": stage, "issue_type": issue_type, "description": description,
            "recommendation": recommendation, **context,
        })

    def run(self) -> Path:
        self.log("[stage 0] download and directory integrity preflight")
        self.run_preflight()
        if not self.preflight_passed:
            self.log("[fail closed] preflight blocked metadata/header/signal stages")
            self.summary.update({
                "metadata_audit_status": "blocked_by_preflight", "header_audit_status": "blocked_by_preflight",
                "signal_smoke_status": "not_run_blocked_by_preflight", "exact_age_label_validation": "blocked",
            })
        else:
            self.log("[stage 1] metadata audit")
            self.audit_metadata()
            self.log("[stage 2] full header audit")
            header_ok = self.audit_headers()
            if header_ok and not self.args.skip_signal_smoke:
                self.log("[stage 3] create deterministic signal sampling plan")
                self.create_sampling_plan()
                self.log("[stage 4] bounded chunked signal smoke")
                self.run_signal_smoke()
            else:
                reason = "requested" if self.args.skip_signal_smoke else "header gate failed"
                self.summary["signal_smoke_status"] = f"not_run: {reason}"
        self.log("[stage 5] joint-training readiness")
        self.assess_joint_readiness()
        self.log("[stage 6] verify source snapshot and write outputs")
        self.verify_source()
        self.write_outputs()
        return self.output_dir

    def run_preflight(self) -> None:
        result = preflight_dataset(self.data_dir, self.allowlist, self.manifest_path)
        self.preflight_passed = bool(result["passed"])
        self.audit_subjects = list(result["audit_subjects"])
        self.rows["subject_manifest.csv"] = result["subjects"]
        self.rows["download_integrity.csv"] = result["integrity"]
        self.rows["audit_issues.csv"].extend(result["issues"])
        dataset = result["dataset_description"]
        self.summary.update({
            "dataset_id": DATASET_ID, "dataset_name": dataset.get("Name"), "dataset_version": dataset.get("Version"),
            "bids_version": dataset.get("BIDSVersion"), "license": dataset.get("License"),
            "dataset_source": dataset.get("SourceDatasets"), "allowlist_subject_count": len(self.allowlist),
            "actual_local_subject_count": len(result["actual_subjects"]), "actual_local_subjects": result["actual_subjects"],
            "missing_allowlist_subject_count": len(result["missing_subjects"]), "missing_allowlist_subjects": result["missing_subjects"],
            "release_manifest_subject_count": len(result.get("manifest_subjects", [])),
            "published_allowlist_subject_count": len(self.audit_subjects), "published_allowlist_subjects": self.audit_subjects,
            "not_published_allowlist_subject_count": len(result.get("not_published_subjects", [])),
            "not_published_allowlist_subjects": result.get("not_published_subjects", []),
            "missing_published_subject_count": len(result.get("missing_published_subjects", [])),
            "missing_published_subjects": result.get("missing_published_subjects", []),
            "non_allowlisted_subject_count": len(result["extra_subjects"]), "non_allowlisted_subjects": result["extra_subjects"],
            "recording_count_discovered_from_paths": result["recording_count_discovered"],
            "local_manifest_available": result["manifest_available"], "local_manifest_path": result["manifest_path"],
            "optional_release_metadata_missing_count": result.get("optional_manifest_missing_count", 0),
            "local_nemar_support": result["nemar_support"],
            "priority_recording_checks": result["priority_checks"],
            "preflight_status": "passed" if self.preflight_passed else "failed_closed",
        })

    def audit_metadata(self) -> None:
        resolved = resolve_age_labels(self.data_dir, self.audit_subjects)
        self.labels = resolved["labels"]
        self.rows["metadata_label_inventory.csv"] = resolved["inventory"]
        by_subject = {row["subject_id"]: row for row in self.rows["subject_manifest.csv"]}
        for sid, label in self.labels.items():
            by_subject[sid].update({
                "exact_age": label["age"], "age_unit": label["age_unit"], "age_source": label["age_source"],
                "age_status": label["age_status"], "sex": label["sex"], "sex_status": label["sex_status"],
            })
        self.summary.update({
            "metadata_audit_status": "completed", "exact_age_label_validation": resolved["exact_age_label_validation"],
            "exact_age_labels_ready": resolved["exact_age_labels_ready"], "sex_labels_ready": resolved["sex_labels_ready"],
            "age_label_source": resolved["age_source"], "public_metadata_fallback": resolved["public_metadata_fallback"],
            "exact_age_label_count": sum(label.get("age") is not None for label in self.labels.values()),
            "age_min": min((label.get("age") for label in self.labels.values() if label.get("age") is not None), default=None),
            "age_max": max((label.get("age") for label in self.labels.values() if label.get("age") is not None), default=None),
            "age_distribution": dict(sorted(Counter(label.get("age") for label in self.labels.values() if label.get("age") is not None).items())),
        })
        if not resolved["exact_age_labels_ready"]:
            self.issue("error", "1", "exact_age_labels_unavailable", "Exact numeric age is unavailable for one or more allowlisted subjects", "Provide a documented subject-to-age mapping")

    def audit_headers(self) -> bool:
        signature_members: dict[str, list[dict[str, Any]]] = defaultdict(list)
        failures = 0
        vhdr_paths = [
            path for sid in self.audit_subjects
            for path in sorted((self.data_dir / sid).rglob("*_eeg.vhdr"))
        ]
        for index, path in enumerate(vhdr_paths, 1):
            if index == 1 or index % 25 == 0 or index == len(vhdr_paths):
                self.log(f"  header {index}/{len(vhdr_paths)}; failures={failures}")
            rel = path.relative_to(self.data_dir).as_posix()
            entities = parse_task(path)
            sid, task = entities["subject"], entities["task"]
            label = self.labels.get(sid, {})
            try:
                parsed = parse_vhdr(path)
                binary_path = path.with_name(parsed["data_file"])
                marker_path = path.with_name(parsed["marker_file"])
                raw = mne.io.read_raw_brainvision(path, preload=False, verbose="ERROR")
                names = list(raw.ch_names)
                types = list(raw.get_channel_types())
                sfreq = float(raw.info["sfreq"])
                n_samples = int(raw.n_times)
                duration = n_samples / sfreq if sfreq else 0.0
                highpass = float(raw.info["highpass"])
                lowpass = float(raw.info["lowpass"])
                orig_units = dict(getattr(raw, "_orig_units", {}))
                raw.close()
                channel_meta = parsed["channels"]
                meta_by_name = {row["name"]: row for row in channel_meta}
                units = [meta_by_name.get(name, {}).get("unit") or orig_units.get(name, "unknown") for name in names]
                references = sorted({row.get("reference", "") for row in channel_meta[:128] if row.get("reference")})
                reference = ",".join(references) if references else "not_declared"
                ground = "not_declared"
                montage = "dataset-level: EGI HydroCel Geodesic Sensor Net 128; local coordinates absent"
                coordinate_system = "not_declared_in_local_files"
                eeg_indices = [i for i, value in enumerate(types) if value == "eeg"]
                duplicate = sorted(name for name, count in Counter(names).items() if count > 1)
                collisions = canonicalization_collisions(names)
                binary_samples, remainder = binary_geometry(parsed, binary_path.stat().st_size)
                binary_duration = binary_samples / sfreq if binary_samples is not None and sfreq else None
                sidecar_path = path.with_suffix(".json")
                events_path = path.with_name(path.name.replace("_eeg.vhdr", "_events.tsv"))
                sidecar_duration = None
                if sidecar_path.is_file():
                    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8-sig"))
                    sidecar_duration = sidecar.get("RecordingDuration")
                    reference = sidecar.get("EEGReference", reference) or reference
                    ground = sidecar.get("EEGGround", ground) or ground
                signature_payload = {
                    "format": "BrainVision", "binary_format": parsed["binary_format"], "sfreq": sfreq,
                    "names": names, "types": types, "units": units, "reference": reference,
                }
                signature = hashlib.sha256(json.dumps(signature_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]
                short_status = "ok"
                if duration <= 0:
                    short_status = "zero_duration"
                elif duration < 5:
                    short_status = "abnormally_short_warning"
                if remainder not in (0, None) or binary_samples != n_samples:
                    short_status = "binary_header_mismatch"
                row = {
                    "subject_id": sid, "task": task, "recording": path.stem, "session": "", "run": entities["run"],
                    "source_path": rel, "format": "BrainVision", "binary_path": binary_path.relative_to(self.data_dir).as_posix(),
                    "marker_path": marker_path.relative_to(self.data_dir).as_posix(),
                    "sidecar_path": sidecar_path.relative_to(self.data_dir).as_posix() if sidecar_path.is_file() else "",
                    "events_path": events_path.relative_to(self.data_dir).as_posix() if events_path.is_file() else "",
                    "sampling_frequency_hz": sfreq, "n_channels_total": len(names), "n_eeg_channels": len(eeg_indices),
                    "n_aux_channels": len(names) - len(eeg_indices), "channel_names": names, "channel_order": names,
                    "channel_types": types, "units": units, "reference": reference, "ground": ground, "montage": montage,
                    "coordinate_system": coordinate_system, "n_samples_header": n_samples, "header_duration_seconds": duration,
                    "binary_size_bytes": binary_path.stat().st_size, "binary_samples_inferred": binary_samples,
                    "binary_duration_seconds": binary_duration, "sidecar_duration_seconds": sidecar_duration,
                    "physical_limits": "not_provided_by_BrainVision_float32_header",
                    "digital_limits": f"{parsed['binary_format']}; explicit limits not provided",
                    "marker_status": "present" if marker_path.is_file() else "missing",
                    "events_status": "present" if events_path.is_file() else "absent; BrainVision marker file only",
                    "header_signature": signature, "header_read_success": True, "duplicate_channels": duplicate,
                    "canonicalization_collisions": collisions, "short_or_truncated_status": short_status,
                    "exact_age": label.get("age"), "sex": label.get("sex", ""),
                    "highpass_hz": highpass, "lowpass_hz": lowpass,
                }
                self.rows["recording_inventory.csv"].append(row)
                self.recordings.append(row)
                signature_members[signature].append(row)
                for channel_index, name in enumerate(names):
                    meta = meta_by_name.get(name, {})
                    self.rows["channel_header_inventory.csv"].append({
                        "subject_id": sid, "task": task, "recording": path.stem, "source_path": rel,
                        "channel_index": channel_index, "channel_name": name, "channel_type": types[channel_index],
                        "unit": units[channel_index], "resolution": meta.get("resolution", ""),
                        "reference": meta.get("reference", "") or reference, "sampling_frequency_hz": sfreq,
                        "header_signature": signature,
                    })
                if duplicate:
                    self.issue("error", "2", "duplicate_channel_names", f"Duplicate channel names: {duplicate}", "Resolve channel identity before modeling", subject_id=sid, task=task, recording=path.stem, source_path=rel)
                if collisions:
                    self.issue("warning", "2", "canonicalization_collision", f"Canonicalization collisions: {collisions}", "Use an explicit channel mapping", subject_id=sid, task=task, recording=path.stem, source_path=rel)
                if short_status == "binary_header_mismatch":
                    self.issue("error", "2", "binary_header_size_mismatch", f"Header samples={n_samples}, binary inferred={binary_samples}, remainder={remainder}", "Repair the recording before signal access", subject_id=sid, task=task, recording=path.stem, source_path=rel)
                elif short_status != "ok":
                    self.issue("warning", "2", "short_recording", f"Duration={duration:.3f}s", "Review task expectations before preprocessing", subject_id=sid, task=task, recording=path.stem, source_path=rel)
            except Exception as exc:
                failures += 1
                self.errors.append(f"HEADER {rel}\n{traceback.format_exc()}")
                self.issue("error", "2", "header_read_failed", str(exc), "Repair the BrainVision triplet", subject_id=sid, task=task, recording=path.stem, source_path=rel)
                self.rows["recording_inventory.csv"].append({
                    "subject_id": sid, "task": task, "recording": path.stem, "source_path": rel,
                    "format": "BrainVision", "header_read_success": False,
                })

        for signature, members in sorted(signature_members.items()):
            first = members[0]
            self.rows["header_signature_inventory.csv"].append({
                "header_signature": signature, "format": first["format"],
                "sampling_frequency_hz": first["sampling_frequency_hz"], "n_channels_total": first["n_channels_total"],
                "n_eeg_channels": first["n_eeg_channels"], "n_aux_channels": first["n_aux_channels"],
                "ordered_channel_names": first["channel_names"], "channel_types": first["channel_types"],
                "units": first["units"], "reference": first["reference"], "ground": first["ground"],
                "montage": first["montage"], "recording_count": len(members),
                "subject_count": len({row["subject_id"] for row in members}), "tasks": sorted({row["task"] for row in members}),
            })
        task_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in self.recordings:
            task_groups[row["task"]].append(row)
        expected_tasks = [f"block{index:02d}" for index in range(1, 20)] + ["block398"]
        for task in expected_tasks:
            members = task_groups.get(task, [])
            resting_candidate = task == "block01"
            self.rows["task_inventory.csv"].append({
                "task": task, "recording_count": len(members), "subject_count": len({row["subject_id"] for row in members}),
                "paradigm": "resting-state candidate" if resting_candidate else "unknown at block level",
                "eyes_condition": (
                    "EO/EC expected in official resting protocol; converted markers do not preserve original trigger 20/30 labels"
                    if resting_candidate else "unknown"
                ),
                "continuity": "continuous resting acquisition candidate" if resting_candidate else "unknown",
                "pre_post_session_semantics": "not encoded in local filename/sidecar",
                "mapping_status": "inferred" if resting_candidate else "unknown",
                "confidence": "moderate" if resting_candidate else "unknown",
                "evidence": f"{OFFICIAL_PROTOCOL_DOC}; {OFFICIAL_CONVERSION_CODE}; {OFFICIAL_DATASET_PAGE}; {DATA_PAPER}",
                "notes": (
                    "Official protocol fixes resting first and local block01 marker structure is consistent, but the official conversion code explicitly states that raw-block-index mapping is not enumerated; therefore not confirmed."
                    if resting_candidate else
                    "The official protocol confirms the dataset-level battery (surround suppression, naturalistic viewing, contrast change, sequence learning, symbol search), but the official conversion code says block-index mapping is not enumerated."
                ),
            })

        tasks_by_subject = {sid: {row["task"] for row in self.recordings if row["subject_id"] == sid} for sid in self.audit_subjects}
        union_tasks = set().union(*tasks_by_subject.values()) if tasks_by_subject else set()
        for sid, tasks in tasks_by_subject.items():
            missing = sorted(union_tasks - tasks)
            if missing:
                self.issue("warning", "2", "subject_missing_observed_blocks", f"Missing relative to local union: {missing}", "Interpret as completion heterogeneity until official completion overview is available", subject_id=sid)
        durations = [float(row["header_duration_seconds"]) for row in self.recordings]
        self.summary.update({
            "header_audit_status": "passed" if failures == 0 else "failed", "header_read_success_count": len(self.recordings),
            "header_read_failure_count": failures, "header_signature_count": len(signature_members),
            "sampling_frequencies_hz": sorted({row["sampling_frequency_hz"] for row in self.recordings}),
            "channel_count_profiles": sorted({(row["n_channels_total"], row["n_eeg_channels"], row["n_aux_channels"]) for row in self.recordings}),
            "reference_values": sorted({row["reference"] for row in self.recordings}),
            "duration_min_seconds": min(durations) if durations else None, "duration_max_seconds": max(durations) if durations else None,
            "task_recording_counts": dict(Counter(row["task"] for row in self.recordings)),
            "resting_state_block_mapping": "block01 inferred candidate; not confirmed",
            "resting_state_mapping_status": "inferred",
        })
        return failures == 0 and not any(row["short_or_truncated_status"] == "binary_header_mismatch" for row in self.recordings)

    def create_sampling_plan(self) -> None:
        plan = build_sampling_plan(
            self.recordings, max_recordings=self.args.max_signal_recordings, smoke_subjects=self.args.smoke_subjects,
            exact_age_ready=self.summary.get("exact_age_labels_ready", False), sex_ready=self.summary.get("sex_labels_ready", False),
        )
        priority_vhdr = {path[:-4] + ".vhdr" for path in PRIORITY_RECORDINGS if path.split("/", 1)[0] in set(self.audit_subjects)}
        selected_paths = {row["source_path"] for row in plan}
        missing_priority = sorted(priority_vhdr - selected_paths)
        if missing_priority:
            for rel in missing_priority:
                self.issue("error", "3", "mandatory_smoke_recording_unavailable", f"Mandatory smoke recording unavailable: {rel}", "Complete/repair the download before signal smoke", source_path=rel)
            self.summary["signal_smoke_status"] = "blocked_missing_mandatory_recording"
            plan = []
        self.rows["signal_sampling_plan.csv"] = plan
        write_csv(self.output_dir / "signal_sampling_plan.csv", plan, CSV_FIELDS["signal_sampling_plan.csv"])
        self.summary.update({
            "signal_sampling_plan_count": len(plan), "signal_sampling_subject_count": len({row["subject_id"] for row in plan}),
            "signal_sampling_age_sex_stratified": bool(plan and plan[0]["age_sex_stratified"]),
            "signal_sampling_age_stratified": bool(plan and self.summary.get("exact_age_labels_ready", False)),
            "signal_sampling_sex_stratified": bool(plan and self.summary.get("sex_labels_ready", False)),
        })

    def run_signal_smoke(self) -> None:
        plan = self.rows["signal_sampling_plan.csv"]
        if not plan:
            self.summary["signal_smoke_status"] = self.summary.get("signal_smoke_status", "not_run_empty_plan")
            return
        record_by_path = {row["source_path"]: row for row in self.recordings}
        for plan_row in plan:
            rec = record_by_path[plan_row["source_path"]]
            self.log(f"  signal {plan_row['selection_order']}/{len(plan)} {rec['recording']}")
            self.smoke_recording(rec)
            plan_row["read_completed"] = bool(self.rows["signal_smoke_recordings.csv"][-1].get("fully_decoded"))
        self.rows["adaptive_expansion_log.csv"] = recommend_expansion(
            self.rows["signal_smoke_recordings.csv"], self.rows["signal_smoke_channel_statistics.csv"]
        )
        failures = [row for row in self.rows["signal_smoke_recordings.csv"] if row.get("smoke_status") != "passed"]
        self.summary.update({
            "signal_smoke_status": "passed" if not failures else "completed_with_failures",
            "signal_smoke_recording_count": len(self.rows["signal_smoke_recordings.csv"]),
            "signal_smoke_fully_decoded_count": sum(bool(row.get("fully_decoded")) for row in self.rows["signal_smoke_recordings.csv"]),
            "signal_smoke_hard_error_count": sum(int(row.get("hard_error_count") or 0) for row in self.rows["signal_smoke_recordings.csv"]),
            "signal_smoke_warning_count": sum(int(row.get("warning_count") or 0) for row in self.rows["signal_smoke_recordings.csv"]),
            "signal_smoke_nan_recording_count": sum(bool(row.get("has_nan")) for row in self.rows["signal_smoke_recordings.csv"]),
            "signal_smoke_inf_recording_count": sum(bool(row.get("has_inf")) for row in self.rows["signal_smoke_recordings.csv"]),
            "signal_smoke_all_zero_recording_count": sum(bool(row.get("all_zero_channels")) for row in self.rows["signal_smoke_recordings.csv"]),
            "signal_smoke_constant_recording_count": sum(bool(row.get("constant_channels")) for row in self.rows["signal_smoke_recordings.csv"]),
            "signal_smoke_near_zero_recording_count": sum(bool(row.get("near_zero_variance_channels")) for row in self.rows["signal_smoke_recordings.csv"]),
            "signal_smoke_duplicate_recording_count": sum(bool(row.get("duplicate_channel_groups")) for row in self.rows["signal_smoke_recordings.csv"]),
            "signal_smoke_marker_misclassification_count": sum(bool(row.get("marker_or_status_in_eeg_picks")) for row in self.rows["signal_smoke_recordings.csv"]),
            "signal_smoke_unit_failure_count": sum(str(row.get("unit_conversion_status", "")) == "failed" for row in self.rows["signal_smoke_recordings.csv"]),
            "signal_smoke_files": [row["source_path"] for row in self.rows["signal_smoke_recordings.csv"]],
            "adaptive_expansion_triggered": self.rows["adaptive_expansion_log.csv"][0]["adaptive_expansion_triggered"],
            "adaptive_expansion_reason": self.rows["adaptive_expansion_log.csv"][0]["adaptive_expansion_reason"],
            "recommended_expansion_scope": self.rows["adaptive_expansion_log.csv"][0]["recommended_expansion_scope"],
            "approval_required": True,
        })
        priority_checks = self.summary.get("priority_recording_checks", {})
        for row in self.rows["signal_smoke_recordings.csv"]:
            binary_rel = str(row.get("source_path", ""))[:-5] + ".eeg" if str(row.get("source_path", "")).endswith(".vhdr") else ""
            if binary_rel in priority_checks:
                priority_checks[binary_rel].update({
                    "signal_readable": bool(row.get("fully_decoded") and row.get("samples_match_header")),
                    "fully_decoded": bool(row.get("fully_decoded")),
                    "decoded_samples": row.get("actual_samples_read"),
                    "header_samples": row.get("source_samples_header"),
                    "signal_smoke_status": row.get("smoke_status"),
                })

    def smoke_recording(self, rec: dict[str, Any]) -> None:
        path = self.data_dir / Path(rec["source_path"])
        sid, task = rec["subject_id"], rec["task"]
        hard = 0
        warnings = 0
        try:
            raw = mne.io.read_raw_brainvision(path, preload=False, verbose="ERROR")
            types = raw.get_channel_types()
            picks = [index for index, value in enumerate(types) if value == "eeg"]
            names = [raw.ch_names[index] for index in picks]
            eeg_types = [types[index] for index in picks]
            stats = [RunningStats() for _ in picks]
            digests = [hashlib.sha256() for _ in picks]
            chunk_samples = max(1, int(round(self.args.chunk_seconds * float(raw.info["sfreq"]))))
            chunks = 0
            actual_samples = 0
            for start in range(0, raw.n_times, chunk_samples):
                stop = min(raw.n_times, start + chunk_samples)
                data_uv = raw.get_data(picks=picks, start=start, stop=stop) * 1e6
                chunks += 1
                actual_samples += data_uv.shape[1]
                for channel_index in range(len(picks)):
                    values = np.ascontiguousarray(data_uv[channel_index])
                    stats[channel_index].update(values)
                    digests[channel_index].update(values.tobytes())
            raw.close()
            digest_groups: dict[str, list[int]] = defaultdict(list)
            for index, digest in enumerate(digests):
                digest_groups[digest.hexdigest()].append(index)
            duplicate_groups = [[names[index] for index in group] for group in digest_groups.values() if len(group) > 1]
            all_zero: list[str] = []
            constant: list[str] = []
            near_zero: list[str] = []
            extreme_max = 0.0
            for index, stat in enumerate(stats):
                flags = channel_flags(stat)
                if flags["all_zero"]:
                    all_zero.append(names[index])
                if flags["strict_constant"]:
                    constant.append(names[index])
                if flags["near_zero_variance"]:
                    near_zero.append(names[index])
                if stat.count and math.isfinite(stat.minimum) and math.isfinite(stat.maximum):
                    extreme_max = max(extreme_max, abs(stat.minimum), abs(stat.maximum))
                duplicate_group = next((group_index + 1 for group_index, group in enumerate(duplicate_groups) if names[index] in group), "")
                self.rows["signal_smoke_channel_statistics.csv"].append({
                    "subject_id": sid, "task": task, "recording": rec["recording"], "source_path": rec["source_path"],
                    "channel_index": index, "channel_name": names[index], "channel_type": eeg_types[index],
                    "unit": "µV after MNE V→µV x1e6", "n_samples": stat.count,
                    "min_uv": stat.minimum, "max_uv": stat.maximum, "mean_uv": stat.mean,
                    "std_uv": math.sqrt(stat.variance), "has_nan": flags["has_nan"], "has_inf": flags["has_inf"],
                    "all_zero": flags["all_zero"], "strict_constant": flags["strict_constant"],
                    "near_zero_variance": flags["near_zero_variance"], "physical_min_uv": "unavailable",
                    "physical_max_uv": "unavailable", "physical_limit_hit_count": "unavailable",
                    "extreme_amplitude_warning": flags["extreme_amplitude_warning"],
                    "dc_offset_warning": flags["dc_offset_warning"], "duplicate_group": duplicate_group,
                })
                for severity, condition, issue_type, description in (
                    ("error", flags["has_nan"], "nan", "NaN encountered during complete chunked decode"),
                    ("error", flags["has_inf"], "inf", "Inf encountered during complete chunked decode"),
                    ("error", flags["all_zero"], "all_zero_channel", "Channel is zero for the full recording"),
                    ("error", flags["strict_constant"] and not flags["all_zero"], "strict_constant_channel", "Channel is strictly constant for the full recording"),
                    ("warning", flags["near_zero_variance"] and not flags["strict_constant"], "near_zero_variance", "Statistical warning only; no ad-hoc exclusion applied"),
                    ("warning", flags["extreme_amplitude_warning"], "extreme_amplitude", "Absolute amplitude exceeds 10000 µV; warning only"),
                    ("warning", flags["dc_offset_warning"], "dc_offset", "Absolute mean exceeds 1000 µV; warning only"),
                ):
                    if condition:
                        hard += int(severity == "error")
                        warnings += int(severity == "warning")
                        self.issue(severity, "4", issue_type, description, "Targeted review; do not auto-expand without approval", subject_id=sid, task=task, recording=rec["recording"], source_path=rec["source_path"])
            if duplicate_groups:
                hard += len(duplicate_groups)
                self.issue("error", "4", "duplicate_channels", f"Exact duplicate channel groups: {duplicate_groups}", "Investigate channel export before preprocessing", subject_id=sid, task=task, recording=rec["recording"], source_path=rec["source_path"])
            samples_match = actual_samples == int(rec["n_samples_header"])
            if not samples_match:
                hard += 1
                self.issue("error", "4", "decoded_sample_count_mismatch", f"Decoded {actual_samples}; header {rec['n_samples_header']}", "Inspect the BrainVision triplet", subject_id=sid, task=task, recording=rec["recording"], source_path=rec["source_path"])
            source_eeg_units = [unit for unit, typ in zip(rec["units"], rec["channel_types"]) if typ == "eeg"]
            unit_ok = all(str(unit).replace("μ", "µ").lower() in {"µv", "uv"} for unit in source_eeg_units)
            if not unit_ok:
                hard += 1
                self.issue("error", "4", "unit_semantics_unverified", f"EEG units are not uniformly µV: {sorted(set(source_eeg_units))}", "Resolve units before modeling", subject_id=sid, task=task, recording=rec["recording"], source_path=rec["source_path"])
            marker_or_status = [name for name in names if not re.fullmatch(r"E(?:[1-9]|[1-9][0-9]|1[01][0-9]|12[0-8])", name)]
            if marker_or_status:
                hard += 1
                self.issue("error", "4", "marker_status_misclassified_as_eeg", f"Unexpected EEG picks: {marker_or_status}", "Correct channel typing before preprocessing", subject_id=sid, task=task, recording=rec["recording"], source_path=rec["source_path"])
            has_nan = any(stat.has_nan for stat in stats)
            has_inf = any(stat.has_inf for stat in stats)
            self.rows["signal_smoke_recordings.csv"].append({
                "subject_id": sid, "task": task, "recording": rec["recording"], "source_path": rec["source_path"],
                "source_samples_header": rec["n_samples_header"], "actual_samples_read": actual_samples,
                "duration_seconds": rec["header_duration_seconds"], "chunks_read": chunks, "fully_decoded": True,
                "samples_match_header": samples_match, "unit_conversion_status": "MNE outputs V; multiplied by 1e6 to µV; source header µV verified" if unit_ok else "failed",
                "has_nan": has_nan, "has_inf": has_inf, "all_zero_channels": all_zero,
                "constant_channels": constant, "near_zero_variance_channels": near_zero,
                "duplicate_channel_groups": duplicate_groups, "candidate_saturation_channels": [],
                "marker_or_status_in_eeg_picks": marker_or_status, "extreme_abs_max_uv": extreme_max,
                "hard_error_count": hard, "warning_count": warnings, "smoke_status": "passed" if hard == 0 else "failed",
            })
        except Exception as exc:
            self.errors.append(f"SIGNAL {rec['source_path']}\n{traceback.format_exc()}")
            self.issue("error", "4", "signal_decode_failed", str(exc), "Inspect the associated header/binary/marker set", subject_id=sid, task=task, recording=rec["recording"], source_path=rec["source_path"])
            self.rows["signal_smoke_recordings.csv"].append({
                "subject_id": sid, "task": task, "recording": rec["recording"], "source_path": rec["source_path"],
                "fully_decoded": False, "samples_match_header": False, "hard_error_count": 1,
                "warning_count": 0, "smoke_status": "failed",
            })

    def assess_joint_readiness(self) -> None:
        latest_dortmund = None
        dortmund_root = self.repo / "audit_dortmund_age"
        if dortmund_root.is_dir():
            for directory in sorted((path for path in dortmund_root.iterdir() if path.is_dir()), reverse=True):
                summary_path = directory / "audit_summary.json"
                if summary_path.is_file():
                    try:
                        latest_dortmund = json.loads(summary_path.read_text(encoding="utf-8-sig"))
                        latest_dortmund["source_path"] = str(summary_path)
                        break
                    except Exception:
                        continue
        smoke_rows = self.rows["signal_smoke_recordings.csv"]
        raw_readable = bool(
            self.preflight_passed and self.summary.get("header_audit_status") == "passed" and smoke_rows
            and all(row.get("fully_decoded") and row.get("samples_match_header") for row in smoke_rows)
        )
        exact_ready = bool(self.summary.get("exact_age_labels_ready", False))
        resting_ready = self.summary.get("resting_state_mapping_status") == "confirmed"
        channel_reference_harmonization_ready = False
        joint_ready = bool(raw_readable and exact_ready and resting_ready and channel_reference_harmonization_ready)
        readiness = {
            "raw_data_readable": raw_readable,
            "exact_age_labels_ready": exact_ready,
            "resting_state_profile_ready": resting_ready,
            "channel_reference_harmonization_ready": channel_reference_harmonization_ready,
            "joint_training_ready": joint_ready,
            "joint_training_ready_reason": "false: requires complete allowlist download, verified smoke, exact labels, confirmed resting block/EO-EC semantics, and explicit 128→64 channel/reference harmonization",
            "mipdb_subject_count": len(self.audit_subjects),
            "mipdb_authoritative_allowlist_count": len(self.allowlist),
            "mipdb_not_published_in_release": self.summary.get("not_published_allowlist_subjects", []),
            "mipdb_age_range": [min((label.get("age") for label in self.labels.values() if label.get("age") is not None), default=None), max((label.get("age") for label in self.labels.values() if label.get("age") is not None), default=None)],
            "mipdb_task_profile": self.summary.get("task_recording_counts", {}),
            "mipdb_sampling_frequencies_hz": self.summary.get("sampling_frequencies_hz", []),
            "mipdb_channel_profiles": self.summary.get("channel_count_profiles", []),
            "mipdb_reference": self.summary.get("reference_values", []),
            "mipdb_duration_range_seconds": [self.summary.get("duration_min_seconds"), self.summary.get("duration_max_seconds")],
            "potential_10s_windows": sum(int(float(row.get("header_duration_seconds") or 0) // 10) for row in self.recordings),
            "dortmund_latest_audit": latest_dortmund,
            "dortmund_reference_profile": {
                "subject_count": latest_dortmund.get("in_scope_subject_count") if latest_dortmund else None,
                "age_range": [latest_dortmund.get("age_min"), latest_dortmund.get("age_max")] if latest_dortmund else None,
                "sampling_rates_hz": latest_dortmund.get("sampling_rates_hz") if latest_dortmund else None,
                "eeg_channel_counts": latest_dortmund.get("eeg_channel_counts") if latest_dortmund else None,
                "reference": latest_dortmund.get("references") if latest_dortmund else None,
            },
            "domain_confounding_risks": [
                "MIPDB task/block semantics may differ from Dortmund resting EO/EC recordings",
                "MIPDB uses 128 EGI EEG channels plus auxiliary channels; Dortmund uses 64 EEG channels",
                "reference, montage coordinates, equipment, sampling rate, duration, and cohort/source dataset can reveal dataset identity",
                "window counts per subject can differ substantially and must not reweight subjects silently",
            ],
            "next_requirements": [
                "obtain a later official release if the four allowlisted but unpublished subjects are required; do not treat them as local download failures",
                "obtain an authoritative block-to-paradigm mapping including resting/EO/EC semantics",
                "define and validate channel, montage, reference, unit, sampling-rate, task, and subject-weight harmonization against Dortmund",
            ],
        }
        self.readiness = readiness
        self.summary.update({key: readiness[key] for key in ("raw_data_readable", "exact_age_labels_ready", "resting_state_profile_ready", "joint_training_ready")})

    def verify_source(self) -> None:
        after = snapshot_tree(self.data_dir) if self.data_dir.is_dir() else {}
        unchanged, deleted, added = compare_snapshots(self.source_before, after)
        self.summary.update({
            "source_snapshot_method": "relative path + byte size + nanosecond mtime; no full 22GB content hash",
            "source_data_unchanged": unchanged, "source_files_deleted": deleted, "source_files_added": added,
            "source_file_count_before": len(self.source_before), "source_file_count_after": len(after),
        })

    def write_outputs(self) -> None:
        if not self.rows["adaptive_expansion_log.csv"]:
            self.rows["adaptive_expansion_log.csv"] = recommend_expansion([], [])
        for name, fields in CSV_FIELDS.items():
            write_csv(self.output_dir / name, self.rows[name], fields)
        self.summary["audit_completed_at"] = utc_now()
        self.summary["output_dir"] = str(self.output_dir)
        provenance = {
            "created_at": self.start_time, "script_version": SCRIPT_VERSION, "script_path": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()), "allowlist_path": str(self.allowlist_path),
            "allowlist_sha256": sha256_file(self.allowlist_path), "command": subprocess.list2cmdline(sys.argv),
            "python_version": platform.python_version(), "mne_version": mne.__version__, "numpy_version": np.__version__,
            "data_dir": str(self.data_dir), "output_dir": str(self.output_dir), "git": git_info(self.repo),
            "read_only_source": True, "preprocessing_performed": False, "lance_generated": False,
            "training_performed": False, "external_sync_performed": False,
            "official_sources": [OFFICIAL_NEMAR_RELEASE, OFFICIAL_DATASET_PAGE, OFFICIAL_PROTOCOL_DOC, OFFICIAL_CONVERSION_CODE, DATA_PAPER],
            "arguments": vars(self.args),
        }
        (self.output_dir / "audit_summary.json").write_text(json.dumps(self.summary, ensure_ascii=False, indent=2), encoding="utf-8")
        (self.output_dir / "processing_provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
        (self.output_dir / "joint_training_readiness.json").write_text(json.dumps(self.readiness, ensure_ascii=False, indent=2), encoding="utf-8")
        (self.output_dir / "audit_errors.log").write_text("\n\n".join(self.errors), encoding="utf-8")
        (self.output_dir / "run.log").write_text("\n".join(self.logs) + "\n", encoding="utf-8")
        (self.output_dir / "audit_report.md").write_text(self.render_report(), encoding="utf-8")

    def render_report(self) -> str:
        s = self.summary
        priority_lines = "\n".join(f"- `{path}`: {json.dumps(status, ensure_ascii=False)}" for path, status in s.get("priority_recording_checks", {}).items()) or "- 无记录。"
        issue_counts = Counter(row["issue_type"] for row in self.rows["audit_issues.csv"])
        tasks = s.get("task_recording_counts", {})
        sampled = s.get("signal_smoke_files", [])
        return f"""# MIPDB 20–35 岁权威 23 人子集 raw data 抽样审计

## 结论

- 数据目录：`{self.data_dir}`。
- 权威 allowlist：{s.get('allowlist_subject_count')} 人；v1.0.2 manifest 发布其中 {s.get('published_allowlist_subject_count')} 人，本地找到 {s.get('actual_local_subject_count')} 个 subject 目录。
- allowlist 中未见于本 release manifest 的 {s.get('not_published_allowlist_subject_count')} 人：{s.get('not_published_allowlist_subjects')}；已发布但本地缺失 {s.get('missing_published_subject_count')} 人：{s.get('missing_published_subjects')}。
- 阶段 0：**{s.get('preflight_status')}**。下载完整性未通过时按要求 fail closed，未进入 metadata/header/signal 读取。
- 数据集：ID `{s.get('dataset_id')}`，版本 `{s.get('dataset_version')}`，BIDS `{s.get('bids_version')}`，许可证 `{s.get('license')}`。
- 本地 manifest：available={s.get('local_manifest_available')}，path={s.get('local_manifest_path')}。无可靠本地 manifest 时未声称完成 manifest 尺寸/checksum 对账。
- 已发布19人的 BrainVision raw 三件套按 manifest 完整；本地未下载的 release metadata/sidecar 共 {s.get('optional_release_metadata_missing_count')} 项，已记录为 warning，不与 raw binary 缺失混淆。
- 本地 NEMAR 辅助证据：{s.get('local_nemar_support')}。`http-snapshot.json` 只作版本旁证；`http-download-failures.txt` 是历史失败日志，不当作 availability manifest。
- exact age validation：`{s.get('exact_age_label_validation')}`；raw_data_readable={s.get('raw_data_readable')}；resting_state_profile_ready={s.get('resting_state_profile_ready')}；joint_training_ready={s.get('joint_training_ready')}。

## 两个重点 recording

{priority_lines}

重点文件在 preflight 中先做尺寸、三件套、frame alignment 和 SHA256 对账；若进入 signal smoke，上述状态还会补充 `fully_decoded` 与实际解码样本数。

## Metadata 与 task/block

- metadata audit：{s.get('metadata_audit_status')}；年龄来源：{s.get('age_label_source')}。
- 精确年龄：{s.get('exact_age_label_count')} 人，范围 {s.get('age_min')}–{s.get('age_max')} 岁，分布 {s.get('age_distribution')}。
- 根 `participants.tsv/json` 是首选来源。若其精确年龄缺失，脚本只会使用有明确路径记录的本地 `MIPDB_PublicFile.csv`，不会反推、取年龄段中点或从 ID 猜测。
- 路径发现 recording 数：{s.get('recording_count_discovered_from_paths')}；完成 header 后的 task 分布：{tasks}。
- 官方 CMI protocol 确认范式集合、固定顺序和 resting trigger（20=EO、30=EC）；官方 NEMAR 转换代码同时明确声明 raw block index→范式没有被权威资料枚举。`block01` 仅作为 moderate-confidence resting 候选（inferred，非 confirmed），其余 block 保持 unknown。
- 官方资料：{OFFICIAL_PROTOCOL_DOC}；{OFFICIAL_CONVERSION_CODE}；{OFFICIAL_DATASET_PAGE}；{DATA_PAPER}。

## Header 审计

- 状态：{s.get('header_audit_status')}；成功/失败={s.get('header_read_success_count', 0)}/{s.get('header_read_failure_count', 0)}；signature={s.get('header_signature_count', 0)}。
- 采样率：{s.get('sampling_frequencies_hz', [])}；通道 profile：{s.get('channel_count_profiles', [])}；reference：{s.get('reference_values', [])}。
- 时长范围：{s.get('duration_min_seconds')}–{s.get('duration_max_seconds')} 秒。

## Signal smoke

- 状态：{s.get('signal_smoke_status')}；计划/完成={s.get('signal_sampling_plan_count', 0)}/{s.get('signal_smoke_recording_count', 0)}。
- 全程解码={s.get('signal_smoke_fully_decoded_count', 0)}；hard errors={s.get('signal_smoke_hard_error_count', 0)}；warnings={s.get('signal_smoke_warning_count', 0)}。
- NaN/Inf recording={s.get('signal_smoke_nan_recording_count', 0)}/{s.get('signal_smoke_inf_recording_count', 0)}；全零/常数/近零方差 recording={s.get('signal_smoke_all_zero_recording_count', 0)}/{s.get('signal_smoke_constant_recording_count', 0)}/{s.get('signal_smoke_near_zero_recording_count', 0)}；重复通道={s.get('signal_smoke_duplicate_recording_count', 0)}；marker/status 误归 EEG={s.get('signal_smoke_marker_misclassification_count', 0)}；单位语义失败={s.get('signal_smoke_unit_failure_count', 0)}。
- 实际读取文件：{sampled}。
- 未做滤波、notch、重采样、重参考、修复、切窗、归一化、模型 forward、Lance 或训练。
- adaptive_expansion_triggered={s.get('adaptive_expansion_triggered', False)}；reason={s.get('adaptive_expansion_reason', '')}；scope={s.get('recommended_expansion_scope', '')}；approval_required=true；未自动扩查。

## Dortmund 联合训练 readiness

- `raw_data_readable={s.get('raw_data_readable')}`
- `exact_age_labels_ready={s.get('exact_age_labels_ready')}`
- `resting_state_profile_ready={s.get('resting_state_profile_ready')}`
- `joint_training_ready={s.get('joint_training_ready')}`
- 文件可读不代表可联合训练。当前仍需取得权威 block→范式/EO-EC 映射，并验证 EGI 128 EEG + auxiliary 到 Dortmund 64 EEG 的 montage/reference/unit/task harmonization；若必须覆盖23人，还需等待/取得包含另外4人的正式 release。

## 问题与原始数据保护

- issue counts：{dict(issue_counts)}。
- 原始数据前后快照一致：{s.get('source_data_unchanged')}；文件数 {s.get('source_file_count_before')}→{s.get('source_file_count_after')}；新增={s.get('source_files_added')}；删除={s.get('source_files_deleted')}。
- 快照方法：{s.get('source_snapshot_method')}。按要求没有为了本次抽样审计对全部数据做完整 SHA256，因此“未变化”严格指路径、大小和纳秒 mtime 快照一致。
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--subject-allowlist", required=True)
    parser.add_argument("--manifest", help="Optional explicit release manifest JSON; otherwise auto-detected")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chunk-seconds", type=float, default=30.0)
    parser.add_argument("--max-signal-recordings", type=int, default=12)
    parser.add_argument("--smoke-subjects", type=int, default=5)
    parser.add_argument("--skip-signal-smoke", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be positive")
    if not 1 <= args.max_signal_recordings <= 12:
        raise ValueError("max_signal_recordings must be between 1 and 12")
    if not 3 <= args.smoke_subjects <= 5:
        raise ValueError("smoke_subjects must be between 3 and 5")
    data_dir = Path(args.data_dir).resolve()
    output_root = Path(args.output_dir).resolve()
    if data_dir == output_root or data_dir in output_root.parents:
        raise ValueError("output directory must not be inside source data")
    audit = Audit(args)
    output = audit.run()
    print(f"audit output: {output}")
    return 0 if audit.preflight_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
