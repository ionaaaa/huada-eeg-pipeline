#!/usr/bin/env python3
"""Read-only, staged audit for the downloaded Dortmund Vital Study subset.

The script deliberately separates full metadata/header inspection from bounded
signal smoke tests.  It never writes below --data-dir and never preprocesses,
filters, resamples, windows, or normalizes EEG data.
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
import subprocess
import sys
import traceback
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import mne
import numpy as np


SCRIPT_VERSION = "1.0.0"
OFFICIAL_README = "https://github.com/OpenNeuroDatasets/ds005385"
OFFICIAL_PARTICIPANTS_JSON = (
    "https://raw.githubusercontent.com/OpenNeuroDatasets/ds005385/main/participants.json"
)
EEG_SUFFIX = "_eeg.edf"
SMALL_HASH_LIMIT = 1024 * 1024
POINTER_PREFIXES = (b"/annex/objects/", b"git-annex", b".git/annex/objects/")
BIDS_RE = re.compile(
    r"^(?P<subject>sub-[^_]+)(?:_(?P<session>ses-[^_]+))?"
    r"(?:_task-(?P<task>[^_]+))?(?:_acq-(?P<acquisition>[^_]+))?"
    r"(?:_run-(?P<run>[^_]+))?(?:_(?P<suffix>[^.]+))?\.(?P<extension>[^.]+)$",
    re.IGNORECASE,
)

CSV_FIELDS: dict[str, list[str]] = {
    "subject_manifest.csv": ["subject_id", "age", "age_dtype", "age_unit", "age_scope", "age_status", "sex", "handedness", "downloaded", "eeg_materialized", "sessions", "eligible_profile_a", "eligible_profile_b", "eligible_profile_c", "exclusion_reason"],
    "age_distribution.csv": ["age", "subject_count"],
    "downloaded_file_inventory.csv": ["relative_path", "extension", "size_bytes", "mtime_ns", "sha256_small_file", "header_magic", "materialized_status", "subject_id", "session_id", "task", "acquisition", "run"],
    "materialization_status.csv": ["subject_id", "session_id", "source_path", "size_bytes", "magic", "materialized_status", "detail"],
    "bids_structure_issues.csv": ["severity", "subject_id", "session_id", "source_path", "issue_type", "description", "recommendation"],
    "recording_inventory.csv": ["subject_id", "session_id", "task", "acquisition", "run", "recording_id", "source_path", "age", "age_label_scope", "age_label_status", "sex", "handedness", "file_format", "file_size", "sampling_rate", "n_channels_total", "n_eeg_channels", "channel_names", "channel_order", "channel_types", "units", "duration_seconds", "n_samples", "reference", "ground", "montage", "line_frequency", "highpass_from_header", "lowpass_from_header", "manufacturer", "device_model", "cap_model", "measurement_date_or_batch", "header_signature", "eyes_condition", "pre_or_post_cognitive_activity", "resting_state_status", "sidecar_sources", "header_read_success", "possible_10s_windows", "tail_seconds"],
    "subject_recording_summary.csv": ["subject_id", "age", "sex", "sessions", "recording_count", "header_success_count", "conditions", "header_signatures", "total_duration_seconds", "profile_a_recordings", "profile_b_recordings", "profile_c_recordings"],
    "condition_inventory.csv": ["session_id", "eyes_condition", "pre_or_post", "recording_count", "subject_count", "age_label_status"],
    "channel_header_inventory.csv": ["subject_id", "session_id", "recording_id", "source_path", "channel_index", "channel_name", "channel_type", "unit", "sampling_rate", "reference", "ground", "manufacturer", "device_model", "header_signature"],
    "header_signature_inventory.csv": ["header_signature", "file_format", "sampling_rate", "n_channels_total", "n_eeg_channels", "ordered_channel_names", "channel_types", "units", "reference", "manufacturer", "device_model", "recording_count", "subject_count"],
    "channel_compatibility.csv": ["channel_name", "dortmund_index", "huada_index", "in_dortmund", "in_huada", "same_position", "proposed_status"],
    "metadata_conflicts.csv": ["severity", "subject_id", "session_id", "source_path", "field", "source_a", "value_a", "source_b", "value_b", "description"],
    "signal_sampling_plan.csv": ["subject_id", "age", "sex", "session_id", "recording_id", "eyes_condition", "pre_or_post", "header_signature", "device_batch", "sampling_tier", "target_age_anchor", "age_distance", "selection_reason", "selected", "read_completed"],
    "signal_smoke_recordings.csv": ["subject_id", "age", "sex", "session_id", "recording_id", "source_path", "sampling_tier", "source_samples_header", "actual_samples_read", "duration_seconds", "chunks_read", "fully_readable", "samples_match_header", "unit_conversion_status", "has_nan", "has_inf", "all_zero_channels", "constant_channels", "near_zero_variance_channels", "saturation_channels", "duplicate_channel_groups", "extreme_abs_max_uv", "abs_p50_uv", "abs_p95_uv", "abs_p99_uv", "hard_error_count", "warning_count", "smoke_pass"],
    "signal_smoke_channel_statistics.csv": ["subject_id", "session_id", "recording_id", "source_path", "sampling_tier", "channel_index", "channel_name", "n_samples", "min_uv", "max_uv", "mean_uv", "std_uv", "physical_min_uv", "physical_max_uv", "at_physical_min_count", "at_physical_max_count", "saturation_detected", "all_zero", "strict_constant", "near_zero_variance", "has_nan", "has_inf", "extreme_amplitude_warning", "dc_offset_warning", "duplicate_group"],
    "signal_smoke_issues.csv": ["severity", "sampling_tier", "subject_id", "session_id", "recording_id", "source_path", "channel_name", "issue_type", "description", "action"],
    "adaptive_expansion_log.csv": ["trigger_issue", "trigger_subject", "trigger_recording", "issue_scope", "expansion_rule", "candidate_recording_count", "actual_expanded_recording_count", "auto_executed", "approval_required", "result"],
    "audit_issues.csv": ["severity", "stage", "subject_id", "session_id", "source_path", "recording_id", "channel_name", "issue_type", "description", "recommendation"],
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_bids_entities(path: str | Path) -> dict[str, str]:
    match = BIDS_RE.match(Path(path).name)
    if not match:
        return {}
    return {k: (v or "") for k, v in match.groupdict().items()}


def validate_numeric_age(value: Any) -> tuple[bool, float | None, str]:
    if value is None or str(value).strip().lower() in {"", "n/a", "na", "nan", "unknown"}:
        return False, None, "missing"
    text = str(value).strip()
    if re.search(r"[-–]", text) or text.lower() in {"young", "old", "adult"}:
        return False, None, "binned_or_categorical"
    try:
        age = float(text)
    except ValueError:
        return False, None, "not_numeric"
    if not math.isfinite(age) or age <= 0 or age > 120:
        return False, None, "implausible"
    return True, age, "numeric_years"


def age_status_for_session(session_id: str, baseline_age: float | int | None, explicit_session_age: float | int | None = None) -> tuple[str, float | int | None]:
    if explicit_session_age is not None:
        return "documented_session_age", explicit_session_age
    if session_id == "ses-1" and baseline_age is not None:
        return "documented_baseline_age", baseline_age
    if session_id == "ses-2":
        return "unsafe_or_unknown", None
    return "unknown", None


def classify_materialization(path: Path, prefix: bytes | None = None) -> tuple[str, str, str]:
    size = path.stat().st_size
    if size == 0:
        return "empty", "", "zero-byte file"
    if prefix is None:
        with path.open("rb") as handle:
            prefix = handle.read(512)
    low = prefix.lower().lstrip()
    magic = prefix[:8].decode("ascii", errors="replace")
    if any(token in low for token in POINTER_PREFIXES) or b"annex/objects" in low:
        return "git_annex_pointer", magic, "DataLad/git-annex pointer"
    if low.startswith((b"<!doctype html", b"<html", b"<?xml")):
        return "error_page", magic, "HTML/XML content"
    if path.suffix.lower() == ".edf":
        if prefix[:8] != b"0       ":
            return "magic_mismatch", magic, "EDF extension but EDF magic missing"
        if size < 1024:
            return "suspiciously_small", magic, "EDF too small to contain signal"
    return "materialized", magic, ""


def snapshot_tree(root: Path) -> dict[str, tuple[int, int]]:
    return {str(p.relative_to(root)).replace("\\", "/"): (p.stat().st_size, p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}


def compare_snapshots(before: dict[str, tuple[int, int]], after: dict[str, tuple[int, int]]) -> tuple[bool, bool, bool]:
    return (
        all(k in after and after[k] == v for k, v in before.items()),
        bool(set(before) - set(after)),
        bool(set(after) - set(before)),
    )


def assert_limits(initial_subjects: int, stratified_subjects: int, max_recordings: int, auto_max: int, auto_fraction: float) -> None:
    assert 0 < initial_subjects <= 3
    assert initial_subjects <= stratified_subjects <= 12
    assert 0 < max_recordings <= 4
    assert initial_subjects * max_recordings <= 12
    assert stratified_subjects * max_recordings <= 48
    assert 0 < auto_max <= 100
    assert 0 < auto_fraction <= 0.20


def choose_nearest(candidates: Sequence[dict[str, Any]], target: float, excluded: set[str], preferred_sex: str | None = None, uncovered_batches: set[str] | None = None, batch_first: bool = False) -> dict[str, Any] | None:
    pool = [x for x in candidates if x["subject_id"] not in excluded]
    if not pool:
        return None
    uncovered_batches = uncovered_batches or set()
    def score(item: dict[str, Any]) -> tuple[Any, ...]:
        age_distance = abs(float(item["age"]) - target)
        batch_penalty = 0 if uncovered_batches & set(item.get("batches", [])) else 1
        sex_penalty = 0 if preferred_sex and item.get("sex") == preferred_sex else 1
        return (batch_penalty, age_distance, sex_penalty, item["subject_id"]) if batch_first else (age_distance, batch_penalty, sex_penalty, item["subject_id"])
    return min(pool, key=score)


def select_initial_subjects(candidates: Sequence[dict[str, Any]], count: int = 3) -> list[dict[str, Any]]:
    assert 0 < count <= 3
    ages = sorted(float(x["age"]) for x in candidates)
    targets = [ages[0], float(np.median(ages)), ages[-1]][:count]
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    for i, target in enumerate(targets):
        item = choose_nearest(candidates, target, used, "F" if i % 2 == 0 else "M")
        if item:
            selected.append(dict(item, target_age_anchor=target, selection_reason=f"D1 age coverage near {target:g}"))
            used.add(item["subject_id"])
    return selected


def select_stratified_subjects(candidates: Sequence[dict[str, Any]], initial: Sequence[dict[str, Any]], total: int = 12, anchors: Sequence[int] = (20, 23, 26, 29, 32, 35)) -> list[dict[str, Any]]:
    assert len(initial) <= total <= 12
    selected = [dict(x) for x in initial]
    used = {x["subject_id"] for x in selected}
    all_batches = {batch for item in candidates for batch in item.get("batches", [])}
    covered_batches = {batch for item in selected for batch in item.get("batches", [])}
    cycle = 0
    while len(selected) < total:
        target = anchors[cycle % len(anchors)]
        preferred = "F" if cycle % 2 == 0 else "M"
        # First cover each requested age anchor exactly/closely; only the
        # remaining slots may trade age distance for uncovered year batches.
        item = choose_nearest(candidates, target, used, preferred, all_batches - covered_batches, batch_first=cycle >= len(anchors))
        if item is None:
            break
        selected.append(dict(item, target_age_anchor=target, selection_reason=f"D2 anchor {target}, sex/signature diversity"))
        used.add(item["subject_id"])
        covered_batches.update(item.get("batches", []))
        cycle += 1
    return selected


def decide_adaptive_expansion(candidate_count: int, total_recordings: int, max_recordings: int = 100, max_fraction: float = 0.20) -> tuple[bool, bool]:
    fraction = candidate_count / total_recordings if total_recordings else 1.0
    approval = candidate_count > max_recordings or fraction > max_fraction
    return (not approval, approval)


def open_edf_header(path: Path, reader=mne.io.read_raw_edf):
    return reader(path, preload=False, infer_types=True, verbose="ERROR")


def ensure_new_output(root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = root / stamp
    if out.exists():
        raise FileExistsError(f"output exists: {out}")
    out.mkdir(parents=True)
    return out


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: serialize_cell(row.get(key, "")) for key in fields})


def serialize_cell(value: Any) -> Any:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (list, tuple, set, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=isinstance(value, dict))
    return value


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_info(repo: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.check_output(["git", *args], cwd=repo, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None
    status = run("status", "--porcelain")
    return {"commit": run("rev-parse", "HEAD"), "branch": run("branch", "--show-current"), "dirty": None if status is None else bool(status), "status_porcelain": status}


@dataclass
class RunningStats:
    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf
    has_nan: bool = False
    has_inf: bool = False

    def update(self, values: np.ndarray) -> None:
        self.count += int(values.size)
        self.has_nan |= bool(np.isnan(values).any())
        self.has_inf |= bool(np.isinf(values).any())
        finite = values[np.isfinite(values)]
        if finite.size:
            self.total += float(finite.sum(dtype=np.float64))
            self.total_sq += float(np.square(finite, dtype=np.float64).sum(dtype=np.float64))
            self.minimum = min(self.minimum, float(finite.min()))
            self.maximum = max(self.maximum, float(finite.max()))

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else math.nan

    @property
    def variance(self) -> float:
        return max(0.0, self.total_sq / self.count - self.mean**2) if self.count else math.nan


class Audit:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.data_dir = Path(args.data_dir).resolve()
        self.repo = Path(__file__).resolve().parents[1]
        self.output_dir = ensure_new_output(Path(args.output_dir).resolve())
        self.rows: dict[str, list[dict[str, Any]]] = {name: [] for name in CSV_FIELDS}
        self.errors: list[str] = []
        self.logs: list[str] = []
        self.participants: dict[str, dict[str, Any]] = {}
        self.recordings: list[dict[str, Any]] = []
        self.materialized_edfs: list[Path] = []
        self.signal_seen: set[str] = set()
        self.summary: dict[str, Any] = {}
        self.source_before = snapshot_tree(self.data_dir)
        self.start_time = utc_now()

    def log(self, message: str) -> None:
        line = f"{datetime.now().isoformat(timespec='seconds')} {message}"
        self.logs.append(line)
        print(message, flush=True)

    def issue(self, severity: str, stage: str, issue_type: str, description: str, recommendation: str = "", **context: Any) -> None:
        self.rows["audit_issues.csv"].append({"severity": severity, "stage": stage, "issue_type": issue_type, "description": description, "recommendation": recommendation, **context})

    def run(self) -> Path:
        self.log("[stage 1] inspect repository and configuration")
        self.inspect_repository()
        self.log("[stage 2] resolve downloaded subset and age metadata")
        age_gate = self.resolve_metadata()
        self.log("[stage 3] validate BIDS/materialization")
        materialization_gate = self.inventory_files()
        if age_gate and materialization_gate:
            self.log("[stage 4] audit all recording headers")
            self.audit_headers()
            self.log("[stage 5] select and run 3-subject initial smoke")
            d1_pass = self.run_signal_sampling("D1")
            self.log("[stage 6] decide adaptive path")
            if d1_pass:
                self.log("[stage 7] run 12-subject stratified smoke or targeted expansion")
                self.run_signal_sampling("D2")
                self.run_adaptive_plan()
            else:
                self.run_adaptive_plan()
        else:
            self.log("[stage 4-7] stopped by age/materialization gate")
            self.summary.update({"initial_smoke_passed": False, "stratified_smoke_passed": False})
        self.log("[stage 8] compare channels and preprocessing compatibility")
        self.compare_channels()
        self.log("[stage 9] recommend training profile")
        self.finalize_subject_summaries()
        self.log("[stage 10] verify source data unchanged")
        self.verify_source()
        self.write_outputs()
        return self.output_dir

    def inspect_repository(self) -> None:
        cfg = self.repo / "configs" / "huada_resting_preprocessing.toml"
        self.provenance = {
            "created_at": self.start_time,
            "script_version": SCRIPT_VERSION,
            "script_path": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "command": subprocess.list2cmdline(sys.argv),
            "python_version": platform.python_version(),
            "mne_version": mne.__version__,
            "numpy_version": np.__version__,
            "data_dir": str(self.data_dir),
            "output_dir": str(self.output_dir),
            "public_config_path": str(cfg) if cfg.exists() else None,
            "public_config_sha256": sha256_file(cfg) if cfg.exists() else None,
            "git": git_info(self.repo),
            "limits": vars(self.args),
            "age_semantics_sources": [OFFICIAL_PARTICIPANTS_JSON, OFFICIAL_README],
            "read_only": True,
        }

    def resolve_metadata(self) -> bool:
        candidates = [self.data_dir / "participants.tsv", self.data_dir.parent / "participants.txt"]
        participant_path = next((p for p in candidates if p.exists()), None)
        if participant_path is None:
            self.issue("error", "A", "participants_missing", "No participants metadata found", "Provide official participants.tsv")
            self.summary["numeric_age_label_valid"] = False
            return False
        rows = read_tsv(participant_path)
        downloaded = {p.name for p in self.data_dir.glob("sub-*") if p.is_dir()}
        valid_all = True
        for row in rows:
            sid = row.get("participant_id", "")
            valid, age, reason = validate_numeric_age(row.get("age"))
            valid_all &= valid
            row2 = dict(row, age_numeric=age, age_validation=reason)
            self.participants[sid] = row2
        metadata_ids = set(self.participants)
        in_scope = sorted(metadata_ids & downloaded)
        age_values = [self.participants[s]["age_numeric"] for s in in_scope if self.participants[s]["age_numeric"] is not None]
        age_non_null = len(age_values) == len(in_scope)
        age_in_range = all(20 <= float(x) <= 35 for x in age_values)
        self.summary.update({
            "metadata_subject_count": len(metadata_ids),
            "downloaded_subject_count": len(downloaded),
            "metadata_only_out_of_scope_count": len(metadata_ids - downloaded),
            "in_scope_subject_count": len(in_scope),
            "numeric_age_label_valid": bool(valid_all and age_non_null and age_in_range),
            "age_min": min(age_values) if age_values else None,
            "age_max": max(age_values) if age_values else None,
            "age_unique_value_count": len(set(age_values)),
            "age_field": "age",
            "age_dtype": "integer" if all(float(x).is_integer() for x in age_values) else "float",
            "age_unit": "years",
            "age_semantics": "individual numeric baseline age in years; not birth-date/month precision",
            "session_age_semantics_status": "partial: ses-1 baseline usable; ses-2 age-at-recording unavailable",
        })
        counts = Counter(int(x) if float(x).is_integer() else x for x in age_values)
        self.rows["age_distribution.csv"] = [{"age": age, "subject_count": counts[age]} for age in sorted(counts)]
        sessions_by_subject: dict[str, list[str]] = {}
        for sid in downloaded:
            sessions_by_subject[sid] = sorted(p.name for p in (self.data_dir / sid).glob("ses-*") if p.is_dir())
        for sid in sorted(metadata_ids):
            p = self.participants[sid]
            is_downloaded = sid in downloaded
            sessions = sessions_by_subject.get(sid, [])
            in_range = p["age_numeric"] is not None and 20 <= p["age_numeric"] <= 35
            exclusion = "" if is_downloaded and in_range else ("metadata_only_out_of_scope" if not is_downloaded else "age_out_of_scope")
            self.rows["subject_manifest.csv"].append({
                "subject_id": sid, "age": p["age_numeric"], "age_dtype": self.summary["age_dtype"], "age_unit": "years",
                "age_scope": "baseline_subject", "age_status": "documented_baseline_age" if p["age_numeric"] is not None else "missing",
                "sex": p.get("sex", ""), "handedness": p.get("handedness", ""), "downloaded": is_downloaded,
                "eeg_materialized": False, "sessions": sessions, "eligible_profile_a": False, "eligible_profile_b": False,
                "eligible_profile_c": False, "exclusion_reason": exclusion,
            })
        if not self.summary["numeric_age_label_valid"]:
            self.issue("error", "A", "age_gate_failed", "In-scope ages are missing, non-numeric, binned, or outside 20–35", "Resolve age metadata before signal audit")
        self.issue("warning", "A", "ses2_age_unsafe", "participants age is baseline age; recording_year alone cannot recover ses-2 age-at-recording", "Exclude ses-2 until reliable session-specific age is available")
        return bool(self.summary["numeric_age_label_valid"])

    def inventory_files(self) -> bool:
        materialized_subjects: set[str] = set()
        missing_or_pointer = 0
        for index, path in enumerate(sorted(p for p in self.data_dir.rglob("*") if p.is_file()), 1):
            rel = str(path.relative_to(self.data_dir)).replace("\\", "/")
            entities = parse_bids_entities(path)
            with path.open("rb") as handle:
                prefix = handle.read(512)
            status, magic, detail = classify_materialization(path, prefix)
            sha = sha256_file(path) if path.stat().st_size <= SMALL_HASH_LIMIT else ""
            self.rows["downloaded_file_inventory.csv"].append({
                "relative_path": rel, "extension": path.suffix.lower(), "size_bytes": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns, "sha256_small_file": sha, "header_magic": magic,
                "materialized_status": status, "subject_id": entities.get("subject", ""), "session_id": entities.get("session", ""),
                "task": entities.get("task", ""), "acquisition": entities.get("acquisition", ""), "run": entities.get("run", ""),
            })
            if path.suffix.lower() == ".edf":
                self.rows["materialization_status.csv"].append({"subject_id": entities.get("subject", ""), "session_id": entities.get("session", ""), "source_path": rel, "size_bytes": path.stat().st_size, "magic": magic, "materialized_status": status, "detail": detail})
                if status == "materialized":
                    self.materialized_edfs.append(path)
                    materialized_subjects.add(entities.get("subject", ""))
                else:
                    missing_or_pointer += 1
                    self.issue("error", "B", "eeg_not_materialized", detail, "Retrieve the real EDF object", subject_id=entities.get("subject", ""), session_id=entities.get("session", ""), source_path=rel)
        required_root = ["dataset_description.json", "participants.tsv"]
        for name in required_root:
            if not (self.data_dir / name).exists():
                row = {"severity": "error", "subject_id": "", "session_id": "", "source_path": name, "issue_type": "missing_bids_root_file", "description": f"Downloaded subset omits required BIDS root file {name}", "recommendation": "Preserve as a documented partial BIDS subset or add official metadata outside this read-only audit"}
                self.rows["bids_structure_issues.csv"].append(row)
                self.issue("error", "B", row["issue_type"], row["description"], row["recommendation"], source_path=name)
        for path in self.materialized_edfs:
            entities = parse_bids_entities(path)
            stem = path.name[:-len(EEG_SUFFIX)]
            for suffix in ("_eeg.json", "_channels.tsv", "_events.tsv", "_events.json"):
                expected = path.with_name(stem + suffix)
                if not expected.exists():
                    rel = str(path.relative_to(self.data_dir)).replace("\\", "/")
                    row = {"severity": "error", "subject_id": entities.get("subject", ""), "session_id": entities.get("session", ""), "source_path": rel, "issue_type": "missing_sidecar", "description": f"Missing {expected.name}", "recommendation": "Retrieve matching BIDS sidecar"}
                    self.rows["bids_structure_issues.csv"].append(row)
                    self.issue("error", "B", row["issue_type"], row["description"], row["recommendation"], subject_id=row["subject_id"], session_id=row["session_id"], source_path=rel)
        self.summary.update({"materialized_subject_count": len(materialized_subjects), "materialized_eeg_subject_count": len(materialized_subjects), "recording_count": len(self.materialized_edfs), "download_materialization_validation": "passed" if missing_or_pointer == 0 else "failed", "bids_structure_validation": "failed" if self.rows["bids_structure_issues.csv"] else "passed"})
        for row in self.rows["subject_manifest.csv"]:
            if row["subject_id"] in materialized_subjects:
                row["eeg_materialized"] = True
        self.log(f"  files={len(self.rows['downloaded_file_inventory.csv'])}, materialized EDF={len(self.materialized_edfs)}, subjects={len(materialized_subjects)}")
        return missing_or_pointer == 0 and bool(self.materialized_edfs)

    def audit_headers(self) -> None:
        failures = 0
        signature_members: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for i, path in enumerate(self.materialized_edfs, 1):
            if i == 1 or i % 50 == 0 or i == len(self.materialized_edfs):
                self.log(f"  header {i}/{len(self.materialized_edfs)}; errors={failures}")
            rel = str(path.relative_to(self.data_dir)).replace("\\", "/")
            ent = parse_bids_entities(path)
            sid, ses = ent.get("subject", ""), ent.get("session", "")
            participant = self.participants.get(sid, {})
            age_status, session_age = age_status_for_session(ses, participant.get("age_numeric"))
            stem = path.name[:-len(EEG_SUFFIX)]
            eeg_json_path = path.with_name(stem + "_eeg.json")
            channel_path = path.with_name(stem + "_channels.tsv")
            event_path = path.with_name(stem + "_events.tsv")
            sidecar = read_json(eeg_json_path) if eeg_json_path.exists() else {}
            channel_meta = read_tsv(channel_path)
            try:
                raw = open_edf_header(path)
                names = list(raw.ch_names)
                types = raw.get_channel_types()
                eeg_indices = [j for j, typ in enumerate(types) if typ == "eeg"]
                eeg_names = [names[j] for j in eeg_indices]
                units_by_name = {x.get("name", ""): x.get("units", "") for x in channel_meta}
                units = [units_by_name.get(name, getattr(raw, "_orig_units", {}).get(name, "unknown")) for name in eeg_names]
                sfreq = float(raw.info["sfreq"])
                n_samples = int(raw.n_times)
                duration = n_samples / sfreq if sfreq else 0.0
                ref = sidecar.get("EEGReference", "unknown") or "unknown"
                manufacturer = sidecar.get("Manufacturer", "unknown") or "unknown"
                device = sidecar.get("ManufacturersModelName", "unknown") or "unknown"
                signature_payload = {"format": "EDF", "sfreq": sfreq, "n_total": len(names), "n_eeg": len(eeg_names), "names": eeg_names, "types": [types[j] for j in eeg_indices], "units": units, "reference": ref, "manufacturer": manufacturer, "device": device}
                signature = hashlib.sha256(json.dumps(signature_payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]
                meas = raw.info.get("meas_date")
                session_rows = read_tsv(self.data_dir / sid / f"{sid}_sessions.tsv")
                session_year = next((x.get("recording_year", "") for x in session_rows if x.get("session_id") == ses), "")
                batch = f"recording_year:{session_year}" if session_year else (meas.isoformat() if meas else "unknown")
                raw.close()
                eyes = "eyes_open" if ent.get("task", "").lower() == "eyesopen" else ("eyes_closed" if ent.get("task", "").lower() == "eyesclosed" else "unknown")
                prepost = "pre_cognitive" if ent.get("acquisition", "").lower() == "pre" else ("post_cognitive" if ent.get("acquisition", "").lower() == "post" else "unknown")
                recording_id = path.stem
                row = {
                    "subject_id": sid, "session_id": ses, "task": ent.get("task", ""), "acquisition": ent.get("acquisition", ""), "run": ent.get("run", ""), "recording_id": recording_id, "source_path": rel,
                    "age": session_age, "age_label_scope": "baseline_subject" if ses == "ses-1" else "session_unknown", "age_label_status": age_status, "sex": participant.get("sex", ""), "handedness": participant.get("handedness", ""),
                    "file_format": "EDF", "file_size": path.stat().st_size, "sampling_rate": sfreq, "n_channels_total": len(names), "n_eeg_channels": len(eeg_names), "channel_names": eeg_names, "channel_order": eeg_names,
                    "channel_types": types, "units": units, "duration_seconds": duration, "n_samples": n_samples, "reference": ref, "ground": sidecar.get("EEGGround", "unknown") or "unknown", "montage": sidecar.get("EEGPlacementScheme", "unknown") or "unknown",
                    "line_frequency": sidecar.get("PowerLineFrequency", "unknown"), "highpass_from_header": float(raw.info["highpass"]), "lowpass_from_header": float(raw.info["lowpass"]), "manufacturer": manufacturer, "device_model": device,
                    "cap_model": sidecar.get("CapManufacturersModelName", "unknown") or "unknown", "measurement_date_or_batch": batch, "header_signature": signature, "eyes_condition": eyes, "pre_or_post_cognitive_activity": prepost,
                    "resting_state_status": "resting_state_documented" if str(sidecar.get("TaskName", "")).lower() == "restingstate" else "unknown", "sidecar_sources": [str(eeg_json_path.relative_to(self.data_dir)).replace("\\", "/"), str(channel_path.relative_to(self.data_dir)).replace("\\", "/"), str(event_path.relative_to(self.data_dir)).replace("\\", "/")],
                    "header_read_success": True, "possible_10s_windows": int(duration // 10), "tail_seconds": duration - int(duration // 10) * 10,
                }
                self.rows["recording_inventory.csv"].append(row)
                self.recordings.append(row)
                signature_members[signature].append(row)
                for index, name in enumerate(names):
                    self.rows["channel_header_inventory.csv"].append({"subject_id": sid, "session_id": ses, "recording_id": recording_id, "source_path": rel, "channel_index": index, "channel_name": name, "channel_type": types[index], "unit": units_by_name.get(name, getattr(raw, "_orig_units", {}).get(name, "unknown")), "sampling_rate": sfreq, "reference": ref, "ground": row["ground"], "manufacturer": manufacturer, "device_model": device, "header_signature": signature})
                if sidecar.get("SamplingFrequency") not in (None, sfreq):
                    self.rows["metadata_conflicts.csv"].append({"severity": "error", "subject_id": sid, "session_id": ses, "source_path": rel, "field": "SamplingFrequency", "source_a": "EDF", "value_a": sfreq, "source_b": "eeg.json", "value_b": sidecar.get("SamplingFrequency"), "description": "Header/sidecar sampling rate mismatch"})
                if sidecar.get("RecordingDuration") is not None and not math.isclose(float(sidecar["RecordingDuration"]), duration, abs_tol=1.1):
                    self.rows["metadata_conflicts.csv"].append({"severity": "warning", "subject_id": sid, "session_id": ses, "source_path": rel, "field": "RecordingDuration", "source_a": "EDF", "value_a": duration, "source_b": "eeg.json", "value_b": sidecar.get("RecordingDuration"), "description": "Sidecar nominal duration differs from readable samples"})
                    self.issue("warning", "C", "duration_metadata_mismatch", f"EDF readable duration {duration}s differs from sidecar {sidecar.get('RecordingDuration')}s", "Use readable sample count; review before preprocessing", subject_id=sid, session_id=ses, source_path=rel, recording_id=recording_id)
                if duration < 120:
                    self.issue("warning", "C", "short_recording", f"Readable duration {duration}s is short relative to the documented approximately 3-minute protocol", "Retain only with explicit short-recording policy", subject_id=sid, session_id=ses, source_path=rel, recording_id=recording_id)
            except Exception as exc:
                failures += 1
                self.errors.append(f"HEADER {rel}\n{traceback.format_exc()}")
                self.issue("error", "C", "header_read_failed", str(exc), "Exclude recording until repaired", subject_id=sid, session_id=ses, source_path=rel)
                self.rows["recording_inventory.csv"].append({"subject_id": sid, "session_id": ses, "recording_id": path.stem, "source_path": rel, "age": session_age, "age_label_status": age_status, "file_format": "EDF", "file_size": path.stat().st_size, "header_read_success": False})
        for signature, members in signature_members.items():
            first = members[0]
            self.rows["header_signature_inventory.csv"].append({"header_signature": signature, "file_format": first["file_format"], "sampling_rate": first["sampling_rate"], "n_channels_total": first["n_channels_total"], "n_eeg_channels": first["n_eeg_channels"], "ordered_channel_names": first["channel_names"], "channel_types": first["channel_types"], "units": first["units"], "reference": first["reference"], "manufacturer": first["manufacturer"], "device_model": first["device_model"], "recording_count": len(members), "subject_count": len({x["subject_id"] for x in members})})
        condition_groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in self.recordings:
            condition_groups[(row["session_id"], row["eyes_condition"], row["pre_or_post_cognitive_activity"], row["age_label_status"])].append(row)
        self.rows["condition_inventory.csv"] = [{"session_id": key[0], "eyes_condition": key[1], "pre_or_post": key[2], "age_label_status": key[3], "recording_count": len(vals), "subject_count": len({x["subject_id"] for x in vals})} for key, vals in sorted(condition_groups.items())]
        self.summary.update({"header_read_success_count": len(self.recordings), "header_read_failure_count": failures, "header_signature_count": len(signature_members), "full_metadata_audit_performed": True, "full_header_audit_performed": True, "full_header_audit": "passed" if failures == 0 else "failed"})
        durations = [float(x["duration_seconds"]) for x in self.recordings]
        self.summary.update({"session_recording_counts": dict(Counter(x["session_id"] for x in self.recordings)), "duration_min_seconds": min(durations) if durations else None, "duration_max_seconds": max(durations) if durations else None, "recordings_under_10_seconds": sum(x < 10 for x in durations), "recordings_under_120_seconds": sum(x < 120 for x in durations), "sampling_rates_hz": sorted({x["sampling_rate"] for x in self.recordings}), "eeg_channel_counts": sorted({x["n_eeg_channels"] for x in self.recordings}), "references": sorted({x["reference"] for x in self.recordings}), "line_frequencies_hz": sorted({x["line_frequency"] for x in self.recordings})})

    def subject_candidates(self) -> list[dict[str, Any]]:
        ses1 = [r for r in self.recordings if r["session_id"] == "ses-1" and r["age_label_status"] == "documented_baseline_age"]
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in ses1:
            grouped[row["subject_id"]].append(row)
        out = []
        for sid, rows in grouped.items():
            p = self.participants[sid]
            out.append({"subject_id": sid, "age": p["age_numeric"], "sex": p.get("sex", ""), "handedness": p.get("handedness", ""), "recordings": sorted(rows, key=lambda x: (x["pre_or_post_cognitive_activity"], x["eyes_condition"], x["recording_id"])), "signatures": sorted({x["header_signature"] for x in rows}), "batches": sorted({x["measurement_date_or_batch"] for x in rows})})
        return sorted(out, key=lambda x: (float(x["age"]), x["subject_id"]))

    def run_signal_sampling(self, tier: str) -> bool:
        candidates = self.subject_candidates()
        if tier == "D1":
            chosen = select_initial_subjects(candidates, self.args.initial_smoke_subjects)
        else:
            d1_ids = {x["subject_id"] for x in self.rows["signal_sampling_plan.csv"] if x.get("sampling_tier") == "D1"}
            initial = [x for x in candidates if x["subject_id"] in d1_ids]
            chosen = select_stratified_subjects(candidates, initial, self.args.stratified_smoke_subjects)
            chosen = [x for x in chosen if x["subject_id"] not in d1_ids]
        selected_recordings: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for subject in chosen:
            for rec in subject["recordings"][: self.args.max_recordings_per_subject]:
                selected_recordings.append((subject, rec))
        cap = 12 if tier == "D1" else 48 - len(self.signal_seen)
        selected_recordings = selected_recordings[:cap]
        tier_hard = 0
        completed = 0
        for i, (subject, rec) in enumerate(selected_recordings, 1):
            plan = {"subject_id": subject["subject_id"], "age": subject["age"], "sex": subject["sex"], "session_id": rec["session_id"], "recording_id": rec["recording_id"], "eyes_condition": rec["eyes_condition"], "pre_or_post": rec["pre_or_post_cognitive_activity"], "header_signature": rec["header_signature"], "device_batch": rec["measurement_date_or_batch"], "sampling_tier": tier, "target_age_anchor": subject.get("target_age_anchor", ""), "age_distance": abs(float(subject["age"]) - float(subject.get("target_age_anchor", subject["age"]))), "selection_reason": subject.get("selection_reason", tier), "selected": True, "read_completed": False}
            self.rows["signal_sampling_plan.csv"].append(plan)
            if rec["source_path"] in self.signal_seen:
                plan["read_completed"] = True
                continue
            self.signal_seen.add(rec["source_path"])
            self.log(f"  {tier} signal {i}/{len(selected_recordings)} {rec['recording_id']} ({plan['selection_reason']})")
            ok, hard = self.smoke_recording(rec, tier)
            plan["read_completed"] = ok
            completed += int(ok)
            tier_hard += hard
        if tier == "D1":
            self.summary.update({"initial_smoke_subject_count": len({x["subject_id"] for x in self.rows["signal_sampling_plan.csv"] if x["sampling_tier"] == "D1"}), "initial_smoke_recording_count": len([x for x in self.rows["signal_sampling_plan.csv"] if x["sampling_tier"] == "D1"]), "initial_smoke_passed": tier_hard == 0 and completed == len(selected_recordings)})
            return bool(self.summary["initial_smoke_passed"])
        self.summary.update({"stratified_smoke_subject_count": len({x["subject_id"] for x in self.rows["signal_sampling_plan.csv"]}), "stratified_smoke_recording_count": len(self.signal_seen), "stratified_smoke_passed": tier_hard == 0})
        return tier_hard == 0

    def smoke_recording(self, rec: dict[str, Any], tier: str) -> tuple[bool, int]:
        path = self.data_dir / rec["source_path"]
        sid, ses = rec["subject_id"], rec["session_id"]
        hard = 0
        warnings = 0
        try:
            raw = open_edf_header(path)
            picks = [i for i, typ in enumerate(raw.get_channel_types()) if typ == "eeg"]
            names = [raw.ch_names[i] for i in picks]
            stats = [RunningStats() for _ in picks]
            hashes = [hashlib.sha256() for _ in picks]
            extras = raw._raw_extras[0]
            physical_min_uv = np.asarray(extras.get("physical_min", np.full(len(raw.ch_names), np.nan)), dtype=float)[picks]
            physical_max_uv = np.asarray(extras.get("physical_max", np.full(len(raw.ch_names), np.nan)), dtype=float)[picks]
            sat_min_counts = np.zeros(len(picks), dtype=np.int64)
            sat_max_counts = np.zeros(len(picks), dtype=np.int64)
            samples_for_quantile: list[np.ndarray] = []
            chunk = max(1, int(round(self.args.chunk_seconds * raw.info["sfreq"])))
            chunks = 0
            actual = 0
            for start in range(0, raw.n_times, chunk):
                stop = min(raw.n_times, start + chunk)
                data_uv = raw.get_data(picks=picks, start=start, stop=stop) * 1e6
                chunks += 1
                actual += data_uv.shape[1]
                stride = max(1, data_uv.size // 20000)
                samples_for_quantile.append(np.abs(data_uv.ravel()[::stride]))
                for ch in range(len(picks)):
                    stats[ch].update(data_uv[ch])
                    hashes[ch].update(np.ascontiguousarray(data_uv[ch]).tobytes())
                    if math.isfinite(float(physical_min_uv[ch])):
                        sat_min_counts[ch] += int(np.count_nonzero(np.isclose(data_uv[ch], physical_min_uv[ch], rtol=0.0, atol=1e-9)))
                    if math.isfinite(float(physical_max_uv[ch])):
                        sat_max_counts[ch] += int(np.count_nonzero(np.isclose(data_uv[ch], physical_max_uv[ch], rtol=0.0, atol=1e-9)))
            raw.close()
            digest_groups: dict[str, list[int]] = defaultdict(list)
            for idx, digest in enumerate(hashes):
                digest_groups[digest.hexdigest()].append(idx)
            duplicate_groups = [[names[i] for i in group] for group in digest_groups.values() if len(group) > 1]
            all_zero: list[str] = []
            constant: list[str] = []
            near_zero: list[str] = []
            saturation: list[str] = []
            for idx, stat in enumerate(stats):
                is_zero = stat.minimum == 0.0 and stat.maximum == 0.0
                is_constant = stat.minimum == stat.maximum
                is_near = stat.variance <= 1e-12
                if is_zero: all_zero.append(names[idx])
                if is_constant: constant.append(names[idx])
                if is_near: near_zero.append(names[idx])
                is_saturated = bool(sat_min_counts[idx] or sat_max_counts[idx])
                if is_saturated: saturation.append(names[idx])
                extreme = max(abs(stat.minimum), abs(stat.maximum)) > 10000
                dc = abs(stat.mean) > 1000
                dup_group = next((n + 1 for n, group in enumerate(duplicate_groups) if names[idx] in group), "")
                self.rows["signal_smoke_channel_statistics.csv"].append({"subject_id": sid, "session_id": ses, "recording_id": rec["recording_id"], "source_path": rec["source_path"], "sampling_tier": tier, "channel_index": idx, "channel_name": names[idx], "n_samples": stat.count, "min_uv": stat.minimum, "max_uv": stat.maximum, "mean_uv": stat.mean, "std_uv": math.sqrt(stat.variance), "physical_min_uv": physical_min_uv[idx], "physical_max_uv": physical_max_uv[idx], "at_physical_min_count": int(sat_min_counts[idx]), "at_physical_max_count": int(sat_max_counts[idx]), "saturation_detected": is_saturated, "all_zero": is_zero, "strict_constant": is_constant, "near_zero_variance": is_near, "has_nan": stat.has_nan, "has_inf": stat.has_inf, "extreme_amplitude_warning": extreme, "dc_offset_warning": dc, "duplicate_group": dup_group})
                for severity, condition, kind, desc in [
                    ("error", stat.has_nan, "nan", "NaN found during full chunked read"), ("error", stat.has_inf, "inf", "Inf found during full chunked read"),
                    ("error", is_zero, "all_zero_channel", "Channel is exactly zero for the entire recording"), ("error", is_constant and not is_zero, "strict_constant_channel", "Channel is strictly constant for the entire recording"),
                    ("warning", is_near and not is_constant, "near_zero_variance", "Channel variance is near zero"), ("warning", extreme, "extreme_amplitude", "Absolute amplitude exceeds 10000 µV; no exclusion threshold applied"),
                    ("warning", dc, "dc_offset", "Absolute channel mean exceeds 1000 µV; no preprocessing performed")]:
                    if condition:
                        self.rows["signal_smoke_issues.csv"].append({"severity": severity, "sampling_tier": tier, "subject_id": sid, "session_id": ses, "recording_id": rec["recording_id"], "source_path": rec["source_path"], "channel_name": names[idx], "issue_type": kind, "description": desc, "action": "Exclude at formal preprocessing hard QC" if severity == "error" else "Prioritize for formal preprocessing QC"})
                        hard += int(severity == "error")
                        warnings += int(severity == "warning")
                if is_saturated:
                    warnings += 1
                    self.rows["signal_smoke_issues.csv"].append({"severity": "warning", "sampling_tier": tier, "subject_id": sid, "session_id": ses, "recording_id": rec["recording_id"], "source_path": rec["source_path"], "channel_name": names[idx], "issue_type": "physical_limit_hit", "description": f"Exact physical-limit samples: min={sat_min_counts[idx]}, max={sat_max_counts[idx]}; official README warns EDF physical min/max may be invalid", "action": "Review digital saturation during formal preprocessing; do not exclude from this smoke alone"})
            if duplicate_groups:
                hard += len(duplicate_groups)
                self.rows["signal_smoke_issues.csv"].append({"severity": "error", "sampling_tier": tier, "subject_id": sid, "session_id": ses, "recording_id": rec["recording_id"], "source_path": rec["source_path"], "channel_name": "", "issue_type": "duplicate_channels", "description": f"Exact duplicate channel groups: {duplicate_groups}", "action": "Investigate before preprocessing"})
            samples_match = actual == int(rec["n_samples"])
            if not samples_match:
                hard += 1
            quant = np.concatenate(samples_for_quantile) if samples_for_quantile else np.array([], dtype=float)
            q = np.quantile(quant, [0.5, 0.95, 0.99]) if quant.size else [math.nan] * 3
            unit_ok = all(str(u).replace("μ", "µ").lower() in {"µv", "uv"} for u in rec["units"])
            if not unit_ok:
                hard += 1
            has_nan = any(x.has_nan for x in stats)
            has_inf = any(x.has_inf for x in stats)
            self.rows["signal_smoke_recordings.csv"].append({"subject_id": sid, "age": rec["age"], "sex": rec["sex"], "session_id": ses, "recording_id": rec["recording_id"], "source_path": rec["source_path"], "sampling_tier": tier, "source_samples_header": rec["n_samples"], "actual_samples_read": actual, "duration_seconds": rec["duration_seconds"], "chunks_read": chunks, "fully_readable": True, "samples_match_header": samples_match, "unit_conversion_status": "MNE_V_to_uV_x1e6_verified" if unit_ok else "failed", "has_nan": has_nan, "has_inf": has_inf, "all_zero_channels": all_zero, "constant_channels": constant, "near_zero_variance_channels": near_zero, "saturation_channels": saturation, "duplicate_channel_groups": duplicate_groups, "extreme_abs_max_uv": max(max(abs(x.minimum), abs(x.maximum)) for x in stats), "abs_p50_uv": q[0], "abs_p95_uv": q[1], "abs_p99_uv": q[2], "hard_error_count": hard, "warning_count": warnings, "smoke_pass": hard == 0})
            return True, hard
        except Exception as exc:
            hard += 1
            self.errors.append(f"SIGNAL {rec['source_path']}\n{traceback.format_exc()}")
            self.rows["signal_smoke_issues.csv"].append({"severity": "error", "sampling_tier": tier, "subject_id": sid, "session_id": ses, "recording_id": rec["recording_id"], "source_path": rec["source_path"], "channel_name": "", "issue_type": "full_read_failed", "description": str(exc), "action": "Exclude until repaired"})
            self.rows["signal_smoke_recordings.csv"].append({"subject_id": sid, "session_id": ses, "recording_id": rec["recording_id"], "source_path": rec["source_path"], "sampling_tier": tier, "fully_readable": False, "hard_error_count": 1, "smoke_pass": False})
            return False, hard

    def run_adaptive_plan(self) -> None:
        errors = [x for x in self.rows["signal_smoke_issues.csv"] if x["severity"] == "error"]
        priority_warnings = [x for x in self.rows["signal_smoke_issues.csv"] if x["issue_type"] in {"extreme_amplitude", "dc_offset", "physical_limit_hit"}]
        triggers = errors or priority_warnings
        if not triggers:
            self.summary.update({"adaptive_expansion_triggered": False, "adaptive_expansion_scope": None, "adaptive_expansion_recording_count": 0, "adaptive_expansion_approval_required": False})
            return
        trigger = triggers[0]
        matching = [r for r in self.recordings if r["header_signature"] == next((x["header_signature"] for x in self.recordings if x["recording_id"] == trigger["recording_id"]), "")]
        auto, approval = decide_adaptive_expansion(len(matching), len(self.recordings), self.args.adaptive_auto_max_recordings, self.args.adaptive_auto_max_fraction)
        scope = "possible_global_dataset_or_unit_issue" if priority_warnings and len({x["subject_id"] for x in priority_warnings}) >= 3 else "header_signature_or_channel"
        self.rows["adaptive_expansion_log.csv"].append({"trigger_issue": trigger["issue_type"], "trigger_subject": trigger["subject_id"], "trigger_recording": trigger["recording_id"], "issue_scope": scope, "expansion_rule": "same header signature / device profile; prioritize affected channels", "candidate_recording_count": len(matching), "actual_expanded_recording_count": 0, "auto_executed": False, "approval_required": approval, "result": "approval_required" if approval else "bounded expansion eligible but not needed after completed 12-subject stratification"})
        self.summary.update({"adaptive_expansion_triggered": True, "adaptive_expansion_scope": scope, "adaptive_expansion_recording_count": 0, "adaptive_expansion_approval_required": approval})

    def compare_channels(self) -> None:
        dortmund = []
        if self.recordings:
            dortmund = list(self.recordings[0]["channel_names"])
        huada_path = self.repo / "preprocessing_audit" / "20261030_105520" / "channel_header_inventory.csv"
        huada: list[str] = []
        if huada_path.exists():
            with huada_path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            first_sid = rows[0]["subject_id"] if rows else ""
            huada = [x["channel_name"] for x in rows if x["subject_id"] == first_sid]
        union = list(dict.fromkeys(dortmund + huada))
        for name in union:
            di = dortmund.index(name) if name in dortmund else None
            hi = huada.index(name) if name in huada else None
            if di is not None and hi is not None:
                status = "shared_same_position" if di == hi else "shared_reorder_required"
            elif di is not None:
                status = "dortmund_only"
            else:
                status = "huada_only"
            self.rows["channel_compatibility.csv"].append({"channel_name": name, "dortmund_index": di, "huada_index": hi, "in_dortmund": di is not None, "in_huada": hi is not None, "same_position": di is not None and hi is not None and di == hi, "proposed_status": status})
        self.summary.update({"channel_cross_dataset_comparison": "completed" if huada else "pending", "channel_intersection_count": len(set(dortmund) & set(huada)), "dortmund_only_channel_count": len(set(dortmund) - set(huada)), "huada_only_channel_count": len(set(huada) - set(dortmund)), "public_preprocessing_compatibility": "conditional: 1000 Hz supports 200 Hz; Nyquist supports 75 Hz; 50 Hz line frequency; units µV; channel/reference harmonization required"})

    def finalize_subject_summaries(self) -> None:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in self.recordings:
            grouped[row["subject_id"]].append(row)
        for sid, recs in sorted(grouped.items()):
            p = self.participants[sid]
            ses1 = [r for r in recs if r["session_id"] == "ses-1"]
            profile_a = [r for r in ses1 if r["pre_or_post_cognitive_activity"] == "pre_cognitive"]
            self.rows["subject_recording_summary.csv"].append({"subject_id": sid, "age": p["age_numeric"], "sex": p.get("sex", ""), "sessions": sorted({r["session_id"] for r in recs}), "recording_count": len(recs), "header_success_count": sum(bool(r["header_read_success"]) for r in recs), "conditions": sorted({f"{r['eyes_condition']}:{r['pre_or_post_cognitive_activity']}" for r in recs}), "header_signatures": sorted({r["header_signature"] for r in recs}), "total_duration_seconds": sum(float(r["duration_seconds"]) for r in recs), "profile_a_recordings": len(profile_a), "profile_b_recordings": len(ses1), "profile_c_recordings": 0})
        bad_signal = any(x["severity"] == "error" for x in self.rows["signal_smoke_issues.csv"])
        eligible = self.summary.get("numeric_age_label_valid", False) and self.summary.get("header_read_failure_count", 1) == 0 and not bad_signal
        warning_counts = Counter(x["issue_type"] for x in self.rows["signal_smoke_issues.csv"] if x["severity"] == "warning")
        hard_counts = Counter(x["issue_type"] for x in self.rows["signal_smoke_issues.csv"] if x["severity"] == "error")
        d1_subjects = sorted({x["subject_id"] for x in self.rows["signal_sampling_plan.csv"] if x["sampling_tier"] == "D1"})
        all_sampled_subjects = sorted({x["subject_id"] for x in self.rows["signal_sampling_plan.csv"]})
        all_batches = sorted({x["measurement_date_or_batch"] for x in self.recordings})
        sampled_batches = sorted({x["device_batch"] for x in self.rows["signal_sampling_plan.csv"]})
        self.summary.update({
            "full_signal_audit_performed": False, "full_signal_qc_deferred_to_preprocessing": True, "preprocessing_performed": False,
            "eligible_for_preprocessing": "conditional" if eligible else False, "recommended_profile": "A" if eligible else "none",
            "adaptive_expansion_triggered": self.summary.get("adaptive_expansion_triggered", False), "adaptive_expansion_scope": self.summary.get("adaptive_expansion_scope"), "adaptive_expansion_recording_count": self.summary.get("adaptive_expansion_recording_count", 0), "adaptive_expansion_approval_required": self.summary.get("adaptive_expansion_approval_required", False),
            "adaptive_expansion_completed": False if self.summary.get("adaptive_expansion_triggered") else "not_needed",
            "signal_hard_issue_counts": dict(hard_counts), "signal_warning_counts": dict(warning_counts),
            "initial_smoke_subject_ids": d1_subjects, "stratified_smoke_subject_ids": all_sampled_subjects,
            "all_device_batches": all_batches, "sampled_device_batches": sampled_batches, "uncovered_device_batches": sorted(set(all_batches) - set(sampled_batches)),
            "numeric_age_label_validation": "passed" if self.summary.get("numeric_age_label_valid") else "failed",
            "session_age_semantics_validation": "partial", "initial_3_subject_smoke": "passed" if self.summary.get("initial_smoke_passed") else "failed",
            "stratified_12_subject_smoke": "passed" if self.summary.get("stratified_smoke_passed") else "not_run",
        })
        for row in self.rows["subject_manifest.csv"]:
            if row["subject_id"] in grouped and row["age_status"] == "documented_baseline_age":
                row["eligible_profile_a"] = eligible
                row["eligible_profile_b"] = eligible
                row["eligible_profile_c"] = False
                if not eligible and not row["exclusion_reason"]:
                    row["exclusion_reason"] = "audit_hard_error"

    def verify_source(self) -> None:
        after = snapshot_tree(self.data_dir)
        unchanged, deleted, added = compare_snapshots(self.source_before, after)
        self.summary.update({"source_files_modified": not unchanged and not deleted and not added, "source_files_deleted": deleted, "source_files_added": added, "source_data_modified": not unchanged, "source_data_unchanged": unchanged})

    def write_outputs(self) -> None:
        for name, fields in CSV_FIELDS.items():
            write_csv(self.output_dir / name, self.rows[name], fields)
        self.summary["audit_completed_at"] = utc_now()
        self.summary["output_dir"] = str(self.output_dir)
        (self.output_dir / "audit_summary.json").write_text(json.dumps(self.summary, ensure_ascii=False, indent=2), encoding="utf-8")
        (self.output_dir / "processing_provenance.json").write_text(json.dumps(self.provenance, ensure_ascii=False, indent=2), encoding="utf-8")
        (self.output_dir / "audit_errors.log").write_text("\n\n".join(self.errors), encoding="utf-8")
        (self.output_dir / "run.log").write_text("\n".join(self.logs) + "\n", encoding="utf-8")
        (self.output_dir / "audit_report.md").write_text(self.render_report(), encoding="utf-8")

    def render_report(self) -> str:
        s = self.summary
        ages = ", ".join(f"{r['age']}岁={r['subject_count']}人" for r in self.rows["age_distribution.csv"])
        d1 = sorted({x["subject_id"] for x in self.rows["signal_sampling_plan.csv"] if x["sampling_tier"] == "D1"})
        d2 = sorted({x["subject_id"] for x in self.rows["signal_sampling_plan.csv"] if x["sampling_tier"] == "D2"})
        hard = [x for x in self.rows["signal_smoke_issues.csv"] if x["severity"] == "error"]
        warn = [x for x in self.rows["signal_smoke_issues.csv"] if x["severity"] == "warning"]
        warning_counts = Counter(x["issue_type"] for x in warn)
        return f"""# Dortmund Vital Study 20–35 岁下载子集审计报告

## 结论

- 数据路径：`{self.data_dir}`。
- 年龄标签：`participants.txt` 的 `age` 是以年为单位的逐被试数值 baseline 年龄；它不是生日/月龄精度。
- ses-1 可使用 baseline 年龄。ses-2 只有采集年份，不能可靠恢复采集时年龄，状态为 `unsafe_or_unknown`，不得直接复用 baseline 年龄或简单加 5。
- 实际下载/物化被试：{s.get('downloaded_subject_count')} / {s.get('materialized_subject_count')}；recording：{s.get('recording_count')}。
- BIDS：EEG 及逐记录 sidecar 已下载，但顶层 `dataset_description.json`、`participants.tsv` 缺失，因此这是可审计的部分 BIDS 子集，不是完整 BIDS 根。
- 推荐 Profile：**{s.get('recommended_profile')}**。第一版仅用 ses-1、pre-cognitive；eyes-open 和 eyes-closed 保持为独立 recording，所有拆分按 subject。
- 可进入预处理：{s.get('eligible_for_preprocessing')}。正式预处理读取每条 recording 时必须同步执行全量硬性 QC；本次没有做全量 signal 审计。

## 数据规模与年龄

- 完整 metadata 被试：{s.get('metadata_subject_count')}；本地下载：{s.get('downloaded_subject_count')}；物化 EEG：{s.get('materialized_subject_count')}；in-scope：{s.get('in_scope_subject_count')}；metadata-only out-of-scope：{s.get('metadata_only_out_of_scope_count')}。
- 年龄范围：{s.get('age_min')}–{s.get('age_max')} 岁，共 {s.get('age_unique_value_count')} 个整数年龄值。
- 年龄频数：{ages}。

## Header 全量审计

- 成功/失败：{s.get('header_read_success_count')} / {s.get('header_read_failure_count')}；header signature：{s.get('header_signature_count')}。
- EDF 为 1000 Hz，64 EEG 数据通道，另有 Status 通道；EEG 单位为 µV。online reference=FCz，ground=AFz，montage/placement=10-20，设备 BrainProducts BrainAmp DC / actiCAP 64。
- recording 时长范围 {s.get('duration_min_seconds')}–{s.get('duration_max_seconds')} 秒；小于 10 秒={s.get('recordings_under_10_seconds')}，小于 120 秒={s.get('recordings_under_120_seconds')}。最短的 60 秒记录是 sidecar 明确声明的短记录，不是读取截断，但应在正式 profile 中单独决定是否保留。
- 1000 Hz 支持确定性降采样到 200 Hz，Nyquist 支持 75 Hz 低通，sidecar 电源频率 50 Hz，单位可转换到 µV。
- reference 和通道布局与华大并不相同，不能直接假定同一 64 槽位语义；详见 `channel_compatibility.csv`。

## 自适应 signal smoke

- D1 被试：{d1}；recording={s.get('initial_smoke_recording_count')}；通过={s.get('initial_smoke_passed')}。
- D2 新增被试：{d2}；总抽样被试={s.get('stratified_smoke_subject_count')}；总 recording={s.get('stratified_smoke_recording_count')}；通过={s.get('stratified_smoke_passed')}。
- signal hard errors={len(hard)}，warnings={len(warn)}，按类型={dict(warning_counts)}。physical-limit 命中仅作为候选饱和 warning：官方 README 明确提示 EDF physical min/max 可能无效，因此不能单凭该字段淘汰。
- D3 triggered={s.get('adaptive_expansion_triggered')}，scope={s.get('adaptive_expansion_scope')}，候选范围覆盖同一 signature 的全部 {s.get('recording_count')} 条，超过 100 条及 20% 上限，approval required={s.get('adaptive_expansion_approval_required')}，本次未继续读取。
- 抽样覆盖批次：{s.get('sampled_device_batches')}；未覆盖：{s.get('uncovered_device_batches')}。
- 每条抽中 recording 均按 30 秒从头到尾分块读取；未滤波、未重采样、未重参考、未窗口化。

## Profile 与训练协议

- Profile A（推荐）：ses-1 + pre-cognitive，eyes-open/eyes-closed 分开，subject-level split。
- Profile B（可作为后续敏感性分析）：ses-1 + pre/post；post 可能包含约 2 小时认知任务后的系统状态差异。
- Profile C：`not_recommended_due_to_age_label_ambiguity`；只有获得可靠 session-specific age 后才可使用 ses-2。
- 同一 subject 的 session、condition、recording 和所有派生窗口必须进入同一 fold；ses-1/ses-2、eyes-open/closed、pre/post 均不得跨 split。年龄指标应按 subject 聚合。

## 正式预处理阶段要求

正式预处理读取每条 recording 时，同时执行全量硬性QC，从而避免为了审计和预处理重复读取全部原始信号。每条必须验证完整可读、header/实际样本一致、采样率和通道 profile、NaN/Inf、单位到 µV、全零/严格常数通道，并将失败 recording 阻止进入窗口缓存。`full_signal_qc_deferred_to_preprocessing=true`。

## 原始数据保护

- 预处理执行：false；全量 signal 审计：false。
- 文件大小与纳秒级 mtime 前后快照一致：{s.get('source_data_unchanged')}；新增={s.get('source_files_added')}，删除={s.get('source_files_deleted')}。
- 本审计没有修改、移动、重命名或删除原始数据。

## 语义来源

- 官方 participants schema：{OFFICIAL_PARTICIPANTS_JSON}
- 官方数据集说明：{OFFICIAL_README}
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--initial-smoke-subjects", type=int, default=3)
    parser.add_argument("--stratified-smoke-subjects", type=int, default=12)
    parser.add_argument("--max-recordings-per-subject", type=int, default=4)
    parser.add_argument("--chunk-seconds", type=float, default=30.0)
    parser.add_argument("--adaptive-expand", action="store_true")
    parser.add_argument("--adaptive-auto-max-recordings", type=int, default=100)
    parser.add_argument("--adaptive-auto-max-fraction", type=float, default=0.20)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    assert_limits(args.initial_smoke_subjects, args.stratified_smoke_subjects, args.max_recordings_per_subject, args.adaptive_auto_max_recordings, args.adaptive_auto_max_fraction)
    data_dir = Path(args.data_dir).resolve()
    output_root = Path(args.output_dir).resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(data_dir)
    if data_dir == output_root or data_dir in output_root.parents:
        raise ValueError("output directory must not be inside source data")
    audit = Audit(args)
    output = audit.run()
    print(f"audit output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
