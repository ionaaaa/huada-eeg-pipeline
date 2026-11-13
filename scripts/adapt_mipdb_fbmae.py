#!/usr/bin/env python3
"""Adapt MIPDB common 128-channel windows to checkpoint-compatible FBMAE tokens.

The adapter performs only a frozen sparse18 channel scatter before delegating
all token math to the existing Huada/Dortmund tokenizer.  It never modifies
the common Lance and never repeats filtering, notch, common resampling,
windowing, rereferencing, or clip/divide-200 normalization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import lance
import numpy as np
import pyarrow as pa

import build_huada_model_tokens as token_core
from adapt_dortmund_fbmae import git_metadata, json_dump, sha256_file, snapshot_tree, write_csv
from fbmae_training_profiles import load_training_profiles, resolve_training_profile


VERSION = "mipdb-fbmae-sparse18-adapter-v1"
SOURCE_CHANNELS = [f"E{index}" for index in range(1, 129)]
SOURCE_SHAPE = (128, 2000)
SOURCE_SFREQ = 200.0
EXPECTED_MAPPING_NAME = "mipdb_egi128_to_standard64_sparse18_v1"
TOKEN_LANCE_NAME = "mipdb_fbmae_tokens.lance"
REQUIRED_INPUT_FIELDS = {
    "sample_id", "subject_id", "recording_id", "block_id", "source_path",
    "age", "label", "task_semantics", "resting_state_confirmed",
    "block01_candidate", "header_signature", "window_index", "start_sample",
    "start_seconds", "data", "shape", "channel_names", "channel_mask",
    "bad_channel_mask", "qc_pass", "clip_fraction", "target_sampling_rate",
    "preprocessing_config_hash",
}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sequence_fingerprint(values: Sequence[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode("utf-8")).hexdigest()


def load_mapping_profile(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("profile_name") != EXPECTED_MAPPING_NAME:
        raise ValueError(f"mapping profile_name must be {EXPECTED_MAPPING_NAME}")
    mappings = payload.get("mappings")
    if not isinstance(mappings, list) or len(mappings) != 18:
        raise ValueError("mapping profile must contain exactly 18 ordered mappings")
    sources = [str(row.get("source", "")) for row in mappings]
    targets = [str(row.get("target", "")) for row in mappings]
    if len(set(sources)) != 18 or len(set(targets)) != 18:
        raise ValueError("mapping source and target names must each be unique")
    unknown_source = sorted(set(sources) - set(SOURCE_CHANNELS))
    unknown_target = sorted(set(targets) - set(token_core.STANDARD_64_CHANNELS))
    if unknown_source or unknown_target:
        raise ValueError(f"mapping has unknown source={unknown_source} target={unknown_target}")
    invariants = {
        "observed_source_channels": 18,
        "target_slots": 64,
        "missing_target_slots": 46,
        "interpolation_applied": False,
        "zero_fill_missing_slots": True,
        "zero_fill_semantics": "missing_and_masked",
    }
    mismatches = {key: (payload.get(key), expected) for key, expected in invariants.items() if payload.get(key) != expected}
    if mismatches:
        raise ValueError(f"mapping invariant mismatch: {mismatches}")
    return payload


def load_electrode_ids(path: Path) -> tuple[list[int], list[int]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    authoritative: dict[str, int] = {}
    for key, name in raw.items():
        authoritative[token_core.canonical_channel_name(name).casefold()] = int(key)
    source_ids = [authoritative.get(name.casefold()) for name in SOURCE_CHANNELS]
    target_ids = [authoritative.get(name.casefold()) for name in token_core.STANDARD_64_CHANNELS]
    if any(value is None for value in source_ids) or len(set(source_ids)) != 128:
        raise ValueError("electrode vocabulary does not uniquely cover E1-E128")
    if any(value is None for value in target_ids) or len(set(target_ids)) != 64:
        raise ValueError("electrode vocabulary does not uniquely cover STANDARD_64_CHANNELS")
    return [int(value) for value in source_ids], [int(value) for value in target_ids]


def discover_input_lance(input_dir: Path) -> tuple[Path, list[dict[str, Any]]]:
    inspected: list[dict[str, Any]] = []
    valid: list[Path] = []
    for candidate in sorted(path for path in input_dir.rglob("*.lance") if path.is_dir()):
        row: dict[str, Any] = {"path": str(candidate.resolve())}
        try:
            dataset = lance.dataset(candidate)
            missing = sorted(REQUIRED_INPUT_FIELDS - set(dataset.schema.names))
            row.update({"readable": True, "rows": dataset.count_rows(), "missing_required_fields": missing})
            if not missing:
                valid.append(candidate)
        except Exception as exc:
            row.update({"readable": False, "error": str(exc)})
        inspected.append(row)
    if len(valid) != 1:
        raise RuntimeError(f"Expected exactly one MIPDB common Lance; valid={len(valid)} inspected={inspected}")
    return valid[0], inspected


def choose_smoke_indices(
    metadata: Sequence[dict[str, Any]], requested: Sequence[str], max_windows_per_recording: int,
) -> tuple[list[int], dict[str, Any]]:
    if max_windows_per_recording <= 0:
        raise ValueError("smoke max windows per recording must be positive")
    by_subject: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(metadata):
        by_subject[str(row["subject_id"]).casefold()].append((index, row))
    subjects = [value.casefold() for value in requested]
    if not subjects:
        priorities = [value for value in ("sub-a00058775", "sub-a00062842") if value in by_subject]
        subjects = (priorities + [value for value in sorted(by_subject) if value not in priorities])[:2]
    missing = sorted(set(subjects) - set(by_subject))
    if missing:
        raise ValueError(f"smoke subjects absent from input: {missing}")
    selected: list[int] = []
    recording_counts: dict[str, int] = {}
    for subject in subjects:
        by_recording: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
        for item in by_subject[subject]:
            by_recording[str(item[1]["recording_id"])].append(item)
        for recording, rows in sorted(by_recording.items()):
            rows.sort(key=lambda item: (int(item[1]["window_index"]), item[0]))
            chosen = rows[:max_windows_per_recording]
            selected.extend(index for index, _ in chosen)
            recording_counts[recording] = len(chosen)
    return sorted(selected), {
        "subjects": subjects, "rows": len(selected), "recording_counts": recording_counts,
        "selection_basis": "explicit" if requested else "priority edge-case subjects then lexical",
    }


def align_sparse18(
    row: dict[str, Any], mapping_profile: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    shape = tuple(int(value) for value in row["shape"])
    if shape != SOURCE_SHAPE:
        raise ValueError(f"expected MIPDB shape {SOURCE_SHAPE}, got {shape}")
    if list(row["channel_names"]) != SOURCE_CHANNELS:
        raise ValueError("source channel names/order must be exactly E1-E128")
    if not math.isclose(float(row["target_sampling_rate"]), SOURCE_SFREQ, abs_tol=1e-9):
        raise ValueError(f"expected common-cache sampling rate {SOURCE_SFREQ}")
    if row["qc_pass"] is not True:
        raise ValueError(f"source qc_pass is not true for {row['sample_id']}")
    if str(row["task_semantics"]) != "unknown" or bool(row["resting_state_confirmed"]):
        raise ValueError("MIPDB task/rest semantics must remain unknown/false")
    data = np.asarray(row["data"], dtype=np.float32).reshape(SOURCE_SHAPE)
    present = np.asarray(row["channel_mask"], dtype=bool)
    bad = np.asarray(row["bad_channel_mask"], dtype=bool)
    if present.shape != (128,) or bad.shape != (128,):
        raise ValueError("source channel masks must have 128 entries")
    if not np.isfinite(data).all() or data.min() < -1.000001 or data.max() > 1.000001:
        raise ValueError("source data must be finite and normalized to [-1,1]")

    target_data = np.zeros((64, 2000), dtype=np.float32)
    target_present = np.zeros(64, dtype=bool)
    target_bad = np.zeros(64, dtype=bool)
    selected_sources: list[str] = []
    selected_targets: list[str] = []
    for mapping in mapping_profile["mappings"]:
        source_name, target_name = str(mapping["source"]), str(mapping["target"])
        source_index = SOURCE_CHANNELS.index(source_name)
        target_index = token_core.STANDARD_64_CHANNELS.index(target_name)
        target_present[target_index] = bool(present[source_index])
        target_bad[target_index] = bool(present[source_index] and bad[source_index])
        if target_present[target_index] and not target_bad[target_index]:
            target_data[target_index] = data[source_index]
        selected_sources.append(source_name)
        selected_targets.append(target_name)

    target_valid = target_present & ~target_bad
    missing = ~target_present
    if int(target_present.sum()) > 18 or np.any(target_data[missing] != 0):
        raise AssertionError("sparse18 presence/zero-fill invariant failed")
    if np.any(target_data[target_bad] != 0):
        raise AssertionError("mapped bad target slots are not strictly zero")
    mapped_values = target_data[target_valid]
    mapped_clip = float(np.mean(np.abs(mapped_values) >= 1.0)) if mapped_values.size else 0.0
    selected_set = set(selected_sources)
    source_bad_names = [SOURCE_CHANNELS[index] for index in np.flatnonzero(bad)]
    excluded_bad = [name for name in source_bad_names if name not in selected_set]
    mapped_bad = [name for name in source_bad_names if name in selected_set]
    aligned = {
        "data": target_data,
        "channel_names": list(token_core.STANDARD_64_CHANNELS),
        "channel_mask": target_present,
        "bad_channel_mask": target_bad,
    }
    audit = {
        "selected_source_channels": selected_sources,
        "selected_target_channels": selected_targets,
        "target_presence_count": int(target_present.sum()),
        "target_bad_count": int(target_bad.sum()),
        "target_valid_count": int(target_valid.sum()),
        "missing_target_count": int(missing.sum()),
        "source_bad_channels": source_bad_names,
        "source_bad_channels_excluded_by_mapping": excluded_bad,
        "source_mapped_bad_channels": mapped_bad,
        "mapped_18_clip_fraction": mapped_clip,
    }
    return aligned, audit


def build_token_row(
    row: dict[str, Any], mapping_profile: dict[str, Any], mapping_hash: str,
    source_electrode_ids: Sequence[int], target_electrode_ids: Sequence[int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    aligned, mapping_audit = align_sparse18(row, mapping_profile)
    token_source = {
        "sample_id": str(row["sample_id"]), "subject_id": str(row["subject_id"]),
        "source_path": str(row["source_path"]), "recording_id": str(row["recording_id"]),
        "window_index": int(row["window_index"]), "start_time": float(row["start_seconds"]),
        "data": aligned["data"].reshape(-1), "shape": [64, 2000],
        "channel_names": aligned["channel_names"], "channel_mask": aligned["channel_mask"],
        "bad_channel_mask": aligned["bad_channel_mask"], "qc_pass": True,
        "age": float(row["age"]), "sex": "", "target_sampling_rate": SOURCE_SFREQ,
        "preprocessing_config_hash": str(row["preprocessing_config_hash"]),
    }
    processed, token_audit = token_core.process_row(
        token_source, standard_channels=token_core.STANDARD_64_CHANNELS
    )
    age = np.float32(row["age"])
    processed.update({
        "age": age, "label": age, "sex": "",
        "block_id": str(row["block_id"]), "window_start_sample": int(row["start_sample"]),
        "window_start_seconds": float(row["start_seconds"]),
        "header_signature": str(row["header_signature"]),
        "task_semantics": "unknown", "resting_state_confirmed": False,
        "block01_candidate": bool(row["block01_candidate"]),
        "source_128_clip_fraction": np.float32(row["clip_fraction"]),
        "mapped_18_clip_fraction": np.float32(mapping_audit["mapped_18_clip_fraction"]),
        # Keep the inherited field byte-for-byte in meaning: clip_fraction is
        # the common-cache 128-channel statistic.  The mapped18 statistic is
        # stored separately and never silently substitutes for it.
        "clip_fraction": np.float32(row["clip_fraction"]),
        "target_channel_mask": np.asarray(aligned["channel_mask"], dtype=np.float32),
        "target_bad_channel_mask": np.asarray(aligned["bad_channel_mask"], dtype=np.float32),
        "source_channel_names": list(SOURCE_CHANNELS),
        "source_channel_mask": np.asarray(row["channel_mask"], dtype=np.float32),
        "source_bad_channel_mask": np.asarray(row["bad_channel_mask"], dtype=np.float32),
        "source_electrode_ids": np.asarray(source_electrode_ids, dtype=np.int64),
        "electrode_ids": np.asarray(target_electrode_ids, dtype=np.int64),
        "token_electrode_ids": np.repeat(np.asarray(target_electrode_ids, dtype=np.int64), token_core.NUM_TIME_PATCHES),
        "selected_source_channels": list(mapping_audit["selected_source_channels"]),
        "selected_target_channels": list(mapping_audit["selected_target_channels"]),
        "source_bad_channels_excluded_by_mapping": list(mapping_audit["source_bad_channels_excluded_by_mapping"]),
        "source_mapped_bad_channels": list(mapping_audit["source_mapped_bad_channels"]),
        "mapping_profile": EXPECTED_MAPPING_NAME, "mapping_canonical_sha256": mapping_hash,
        "adapter_version": VERSION,
    })
    audit = {
        "sample_id": str(row["sample_id"]), "subject_id": str(row["subject_id"]),
        "recording_id": str(row["recording_id"]), "block_id": str(row["block_id"]),
        "valid_channel_count": int(processed["valid_channel_count"]),
        "valid_token_count": int(np.asarray(processed["token_valid_mask"]).sum()),
        "target_presence_count": mapping_audit["target_presence_count"],
        "target_bad_count": mapping_audit["target_bad_count"],
        "missing_target_count": mapping_audit["missing_target_count"],
        "source_bad_channels": ";".join(mapping_audit["source_bad_channels"]),
        "source_bad_channels_excluded_by_mapping": ";".join(mapping_audit["source_bad_channels_excluded_by_mapping"]),
        "source_mapped_bad_channels": ";".join(mapping_audit["source_mapped_bad_channels"]),
        "source_128_clip_fraction": float(row["clip_fraction"]),
        "mapped_18_clip_fraction": mapping_audit["mapped_18_clip_fraction"],
        "token_core_source_valid_count": token_audit["source_valid_count"],
    }
    return processed, audit


def output_schema() -> pa.Schema:
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
        pa.field("target_channel_mask", pa.list_(pa.float32(), 64), False),
        pa.field("target_bad_channel_mask", pa.list_(pa.float32(), 64), False),
        pa.field("sample_id", pa.string(), False), pa.field("subject_id", pa.string(), False),
        pa.field("recording_id", pa.string(), False), pa.field("block_id", pa.string(), False),
        pa.field("source_path", pa.string(), False), pa.field("age", pa.float32(), False),
        pa.field("label", pa.float32(), False), pa.field("sex", pa.string(), False),
        pa.field("window_index", pa.int64(), False), pa.field("window_start_sample", pa.int64(), False),
        pa.field("window_start_seconds", pa.float64(), False), pa.field("header_signature", pa.string(), False),
        pa.field("task_semantics", pa.string(), False), pa.field("resting_state_confirmed", pa.bool_(), False),
        pa.field("block01_candidate", pa.bool_(), False), pa.field("qc_pass", pa.bool_(), False),
        pa.field("clip_fraction", pa.float32(), False),
        pa.field("source_128_clip_fraction", pa.float32(), False),
        pa.field("mapped_18_clip_fraction", pa.float32(), False),
        pa.field("valid_channel_count", pa.int64(), False),
        pa.field("source_sfreq", pa.float32(), False), pa.field("model_sfreq", pa.float32(), False),
        pa.field("model_preprocessing_version", pa.string(), False),
        pa.field("source_preprocessing_config_hash", pa.string(), False),
        pa.field("source_channel_names", pa.list_(pa.string()), False),
        pa.field("source_channel_mask", pa.list_(pa.float32(), 128), False),
        pa.field("source_bad_channel_mask", pa.list_(pa.float32(), 128), False),
        pa.field("source_electrode_ids", pa.list_(pa.int64(), 128), False),
        pa.field("electrode_ids", pa.list_(pa.int64(), 64), False),
        pa.field("token_electrode_ids", pa.list_(pa.int64(), 640), False),
        pa.field("selected_source_channels", pa.list_(pa.string()), False),
        pa.field("selected_target_channels", pa.list_(pa.string()), False),
        pa.field("source_bad_channels_excluded_by_mapping", pa.list_(pa.string()), False),
        pa.field("source_mapped_bad_channels", pa.list_(pa.string()), False),
        pa.field("mapping_profile", pa.string(), False),
        pa.field("mapping_canonical_sha256", pa.string(), False),
        pa.field("adapter_version", pa.string(), False),
    ])


def rows_to_table(rows: Sequence[dict[str, Any]]) -> pa.Table:
    schema = output_schema()
    arrays: list[pa.Array] = []
    for field in schema:
        values = [row[field.name] for row in rows]
        if pa.types.is_fixed_size_list(field.type):
            size = field.type.list_size
            flat = np.concatenate([np.asarray(value).reshape(-1) for value in values])
            arrays.append(pa.FixedSizeListArray.from_arrays(pa.array(flat, type=field.type.value_type), size))
        else:
            arrays.append(pa.array(values, type=field.type))
    return pa.Table.from_arrays(arrays, schema=schema)


def validate_output(path: Path, expected_rows: int) -> dict[str, Any]:
    dataset = lance.dataset(path)
    if dataset.schema != output_schema() or dataset.count_rows() != expected_rows:
        raise AssertionError("output schema/row-count mismatch")
    if {"split", "fold", "train", "validation", "test"} & set(dataset.schema.names):
        raise AssertionError("output contains forbidden split/fold fields")
    valid_counts: Counter[int] = Counter()
    excluded_bad: Counter[str] = Counter()
    rows = 0
    for batch in dataset.scanner(batch_size=32).to_batches():
        for row in batch.to_pylist():
            inputs = np.asarray(row["token_inputs"], np.float32).reshape(640, 100)
            targets = np.asarray(row["targets"], np.float32).reshape(640, 5, 5)
            present = np.asarray(row["target_channel_mask"], bool)
            bad = np.asarray(row["target_bad_channel_mask"], bool)
            channels = np.asarray(row["channel_valid_mask"], bool)
            tokens = np.asarray(row["token_valid_mask"], bool)
            expected_channels = present & ~bad
            expected_tokens = np.repeat(expected_channels, 10)
            valid = (
                row["num_tokens"] == 640 and row["patch_len"] == 100
                and row["n_channels"] == 64 and row["num_patches"] == 10
                and int(present.sum()) == 18 and int((~present).sum()) == 46
                and not np.any(bad & ~present)
                and np.array_equal(channels, expected_channels)
                and np.array_equal(tokens, expected_tokens)
                and np.all(inputs[~tokens] == 0) and np.all(targets[~tokens] == 0)
                and np.isfinite(inputs).all() and np.isfinite(targets).all()
                and row["task_semantics"] == "unknown" and row["resting_state_confirmed"] is False
                and float(row["age"]) == float(row["label"])
            )
            if not valid:
                raise AssertionError(f"output invariant failed: {row['sample_id']}")
            valid_counts[int(tokens.sum())] += 1
            excluded_bad.update(row["source_bad_channels_excluded_by_mapping"])
            rows += 1
    return {
        "rows": rows, "schema_matches": True, "checkpoint_input_shape": [640, 100],
        "valid_token_count_distribution": dict(valid_counts),
        "source_bad_channels_excluded_by_mapping": dict(excluded_bad),
        "all_rows_valid": True,
    }


def parse_subjects(values: Sequence[str] | None) -> list[str]:
    result: list[str] = []
    for value in values or []:
        for item in value.split(","):
            normalized = item.strip()
            if normalized and normalized not in result:
                result.append(normalized)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--input-lance", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mapping-config", type=Path, default=root / "configs" / "mipdb_egi128_to_standard64_sparse18_v1.json")
    parser.add_argument("--training-profiles", type=Path, default=root / "configs" / "mipdb_dortmund_fbmae_training_profiles_v1.json")
    parser.add_argument("--electrode-vocab", type=Path, default=root / "configs" / "fbmae_electrode_vocab_snapshot.json")
    parser.add_argument("--source-uri", default=os.environ.get("MIPDB_COMMON_SOURCE_URI", "local://mipdb-common"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-subjects", nargs="+")
    parser.add_argument("--smoke-max-windows-per-recording", type=int, default=2)
    args = parser.parse_args(argv)
    if args.input_dir is None and args.input_lance is None:
        parser.error("pass --input-dir or --input-lance")
    if args.batch_size <= 0 or args.num_workers < 0:
        parser.error("invalid batch-size/num-workers")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    input_dir = (args.input_dir or args.input_lance.parent).resolve()
    output_dir = args.output_dir.resolve()
    if not input_dir.exists():
        raise FileNotFoundError(input_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {output_dir}")
    if output_dir == input_dir or input_dir in output_dir.parents:
        raise ValueError("output cannot be input directory or its child")
    temp = output_dir.parent / f".{output_dir.name}.tmp-{uuid.uuid4().hex}"
    temp.mkdir(parents=True, exist_ok=False)
    errors: list[str] = []
    started = time.monotonic()
    try:
        if args.input_lance:
            input_lance = args.input_lance.resolve()
            inspected = [{"path": str(input_lance), "explicit": True}]
        else:
            input_lance, inspected = discover_input_lance(input_dir)
        dataset = lance.dataset(input_lance)
        missing = sorted(REQUIRED_INPUT_FIELDS - set(dataset.schema.names))
        if missing:
            raise ValueError(f"input Lance missing fields: {missing}")
        forbidden = {"split", "fold"} & set(dataset.schema.names)
        if forbidden:
            raise ValueError(f"input common Lance contains forbidden fields: {sorted(forbidden)}")
        source_before = snapshot_tree(input_dir)
        metadata_columns = [name for name in dataset.schema.names if name != "data"]
        metadata = dataset.to_table(columns=metadata_columns).to_pylist()
        if not metadata or len({row["sample_id"] for row in metadata}) != len(metadata):
            raise ValueError("input Lance is empty or has duplicate sample_id")
        subjects = parse_subjects(args.smoke_subjects)
        if args.smoke:
            selected_indices, smoke_selection = choose_smoke_indices(
                metadata, subjects, args.smoke_max_windows_per_recording
            )
        else:
            selected_indices, smoke_selection = list(range(len(metadata))), None

        mapping = load_mapping_profile(args.mapping_config.resolve())
        profiles = load_training_profiles(args.training_profiles.resolve())
        mapping_hash = canonical_json_sha256(mapping)
        source_ids, target_ids = load_electrode_ids(args.electrode_vocab.resolve())
        selected_targets = [row["target"] for row in mapping["mappings"]]
        for name, profile in profiles["profiles"].items():
            if "mipdb" not in profile["datasets"]:
                continue
            mipdb_allowed = resolve_training_profile(profiles, name, "mipdb")["allowed_target_channels"]
            if mipdb_allowed != selected_targets:
                raise ValueError(f"training profile {name} MIPDB channels differ from frozen mapping order")

        resolved_mapping = {
            **mapping,
            "canonical_json_sha256": mapping_hash,
            "source_channel_fingerprint_sha256": sequence_fingerprint(SOURCE_CHANNELS),
            "target_standard64_fingerprint_sha256": sequence_fingerprint(token_core.STANDARD_64_CHANNELS),
            "mapping_order_fingerprint_sha256": sequence_fingerprint(
                [f"{row['source']}->{row['target']}" for row in mapping["mappings"]]
            ),
            "electrode_vocabulary_sha256": sha256_file(args.electrode_vocab.resolve()),
            "source_electrode_ids": source_ids,
            "target_electrode_ids": target_ids,
        }
        json_dump(temp / "mapping_profile_resolved.json", resolved_mapping)
        json_dump(temp / "training_profiles_resolved.json", profiles)
        json_dump(temp / "input_schema.json", {"schema": str(dataset.schema), "inspected_candidates": inspected})
        json_dump(temp / "source_snapshot_before.json", source_before)

        output_lance = temp / TOKEN_LANCE_NAME
        inventory: list[dict[str, Any]] = []
        first_write = True
        executor = ThreadPoolExecutor(max_workers=args.num_workers) if args.num_workers else None
        try:
            for offset in range(0, len(selected_indices), args.batch_size):
                indices = selected_indices[offset : offset + args.batch_size]
                source_rows = dataset.take(indices=indices).to_pylist()

                def convert(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
                    return build_token_row(row, mapping, mapping_hash, source_ids, target_ids)

                converted = list(executor.map(convert, source_rows)) if executor else [convert(row) for row in source_rows]
                token_rows = [item[0] for item in converted]
                inventory.extend(item[1] for item in converted)
                lance.write_dataset(
                    rows_to_table(token_rows), output_lance,
                    mode="create" if first_write else "append",
                    max_rows_per_group=min(32, args.batch_size), max_rows_per_file=256,
                    data_storage_version="stable",
                )
                first_write = False
                print(f"[mipdb-fbmae] {min(offset + len(indices), len(selected_indices))}/{len(selected_indices)}", flush=True)
        finally:
            if executor:
                executor.shutdown(wait=True)

        validation = validate_output(output_lance, len(selected_indices))
        source_after = snapshot_tree(input_dir)
        if source_before != source_after:
            raise RuntimeError("input common cache changed during conversion")
        json_dump(temp / "source_snapshot_after.json", source_after)
        fields = [
            "sample_id", "subject_id", "recording_id", "block_id", "valid_channel_count",
            "valid_token_count", "target_presence_count", "target_bad_count", "missing_target_count",
            "source_bad_channels", "source_bad_channels_excluded_by_mapping", "source_mapped_bad_channels",
            "source_128_clip_fraction", "mapped_18_clip_fraction", "token_core_source_valid_count",
        ]
        write_csv(temp / "sample_mapping_inventory.csv", inventory, fields)
        write_csv(temp / "adaptation_issues.csv", [], ["severity", "stage", "sample_id", "issue_type", "detail"])
        (temp / "adaptation_errors.log").write_text("".join(errors), encoding="utf-8")
        summary = {
            "run_success": True, "run_mode": "smoke" if args.smoke else "full",
            "input_rows": len(metadata), "selected_rows": len(selected_indices),
            "selected_subject_count": len({metadata[index]["subject_id"] for index in selected_indices}),
            "selected_recording_count": len({metadata[index]["recording_id"] for index in selected_indices}),
            "smoke_selection": smoke_selection, "mapping_profile": EXPECTED_MAPPING_NAME,
            "mapping_canonical_sha256": mapping_hash, "observed_source_channels": 18,
            "target_slots": 64, "missing_target_slots": 46,
            "checkpoint_contract": {"n_channels": 64, "num_tokens": 640, "patch_len": 100},
            "tokenizer_reused": "scripts/build_huada_model_tokens.py:process_row",
            "training_profiles": list(profiles["profiles"]),
            "one_physical_token_cache": True,
            "source_data_unchanged": True, "validation": validation,
            "elapsed_seconds": time.monotonic() - started, "created_at": now_iso(),
        }
        json_dump(temp / "adapter_summary.json", summary)
        provenance = {
            "version": VERSION, "created_at": now_iso(), "source_uri": args.source_uri,
            "input_dir": str(input_dir), "input_lance": str(input_lance),
            "output_lance": TOKEN_LANCE_NAME, "git": git_metadata(project_root),
            "mapping_profile": resolved_mapping, "training_profiles": profiles,
            "processing": {
                "repeated_bandpass": False, "repeated_notch": False,
                "repeated_common_resample": False, "repeated_windowing": False,
                "repeated_clip_divide_200": False, "rereference_applied": False,
                "model_tokenizer_resample_200_to_100": True,
                "model_tokenizer_existing_standardization": True,
                "model_tokenizer_time_frequency_targets": True,
                "approximate_inverse_clip_scale_x200": True,
                "tokenizer_implementation": "scripts/build_huada_model_tokens.py:process_row",
            },
            "metadata_semantics": {"task_semantics": "unknown", "resting_state_confirmed": False},
            "age_target": "raw exact age; no scaler and no split/fold in token cache",
            "source_snapshot_before": source_before, "source_snapshot_after": source_after,
            "critical_files_sha256": {
                str(path.relative_to(project_root)): sha256_file(path)
                for path in (
                    Path(__file__).resolve(), project_root / "scripts" / "build_huada_model_tokens.py",
                    args.mapping_config.resolve(), args.training_profiles.resolve(), args.electrode_vocab.resolve(),
                )
            },
        }
        json_dump(temp / "processing_provenance.json", provenance)
        json_dump(temp / "token_schema.json", {"schema": str(output_schema())})
        report = f'''# MIPDB sparse18 → FBMAE token adapter

- mode: {summary["run_mode"]}
- rows: {summary["selected_rows"]}
- mapping: `{EXPECTED_MAPPING_NAME}` (`{mapping_hash}`)
- model contract: `[640,100]`, 64 slots, 10 patches/channel
- normal valid tokens: 180; missing slots: 46
- source common cache unchanged: true
- task_semantics=unknown; resting_state_confirmed=false
- physical cache: one; profiles are applied at training read time from the unified profile catalog

The adapter only scatters sparse18 before reusing the authoritative Huada/Dortmund tokenizer.
It does not repeat bandpass, notch, common preprocessing, `/200`, clipping, rereferencing, or windowing.
The tokenizer's checkpoint-required 200→100 Hz step, existing standardization, and time-frequency targets remain unchanged.
'''
        (temp / "adapter_report.md").write_text(report, encoding="utf-8")
        (temp / "_SUCCESS").write_text(now_iso() + "\n", encoding="utf-8")
        temp.replace(output_dir)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return 0
    except Exception:
        error = traceback.format_exc()
        errors.append(error)
        (temp / "adaptation_errors.log").write_text("".join(errors), encoding="utf-8")
        print(error, file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
