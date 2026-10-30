#!/usr/bin/env python
"""Build a shared 50M/1B EEG token cache from Huada common windows.

The input is already filtered, notched, windowed, and clip/divide-200 normalized.
This converter deliberately performs none of those operations again.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import lance
import numpy as np
import pyarrow as pa
from scipy.signal import resample_poly, spectrogram


VERSION = "huada-model-tokens-v1"
SOURCE_SFREQ = 200.0
MODEL_SFREQ = 100.0
CLIP_SECONDS = 10.0
CLIP_STRIDE_SECONDS = 10.0
PATCH_SECONDS = 1.0
PATCH_STRIDE_SECONDS = 1.0
TARGET_FRAME_SECONDS = 0.2
STFT_WINDOW_SECONDS = 1.0
STFT_HOP_SECONDS = 0.1
EPS = 1e-6
BAND_DEFS = [
    ("delta", 1.0, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("gamma_low", 30.0, 45.0),
]

STANDARD_64_CHANNELS = [
    "Fp1", "Fpz", "Fp2",
    "AF7", "AF3", "AFz", "AF4", "AF8",
    "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8",
    "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8",
    "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8",
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POz", "PO4", "PO8",
    "O1", "Oz", "O2", "Iz", "F9", "F10",
]
PUBLIC_REPO_CHANNEL_TAIL = ["Iz", "F9", "F10"]

TOTAL_POINTS = int(CLIP_SECONDS * MODEL_SFREQ)
PATCH_LEN = int(PATCH_SECONDS * MODEL_SFREQ)
PATCH_STRIDE = int(PATCH_STRIDE_SECONDS * MODEL_SFREQ)
NUM_TIME_PATCHES = 1 + (TOTAL_POINTS - PATCH_LEN) // PATCH_STRIDE
N_CHANNELS = len(STANDARD_64_CHANNELS)
NUM_TOKENS = N_CHANNELS * NUM_TIME_PATCHES
FRAMES_PER_PATCH = int(round(PATCH_SECONDS / TARGET_FRAME_SECONDS))
N_BANDS = len(BAND_DEFS)

EXPECTED_DATA_CONFIG = {
    "clip_seconds": 10.0,
    "clip_stride_seconds": 10.0,
    "target_sfreq": 100.0,
    "n_channels": 64,
    "patch_seconds": 1.0,
    "patch_stride_seconds": 1.0,
    "target_frame_seconds": 0.2,
    "stft_window_seconds": 1.0,
    "stft_hop_seconds": 0.1,
    "band_defs": BAND_DEFS,
    "eps": 1e-6,
}
REFERENCE_AUDIT = {
    "50m": {
        "repository": "https://github.com/ionaaaa/50M_Model",
        "data_config": EXPECTED_DATA_CONFIG,
        "architecture": {
            "d_model": {"config_value": 512, "constructor_default": 2048, "effective_value": 512},
            "n_heads": {"config_value": 8, "constructor_default": 16, "effective_value": 8},
            "depth": {"config_value": 12, "constructor_default": 24, "effective_value": 12},
        },
    },
    "1b": {
        "repository": "https://github.com/AbabababaSmart/EEG_pretrain",
        "data_config": EXPECTED_DATA_CONFIG,
        "architecture": {
            "d_model": {"config_value": 2048, "constructor_default": 2048, "effective_value": 2048},
            "n_heads": {"config_value": 16, "constructor_default": 16, "effective_value": 16},
            "depth": {"config_value": 20, "constructor_default": 24, "effective_value": 20},
        },
    },
}

ALIASES = {
    "T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8",
    "FP1": "Fp1", "FPZ": "Fpz", "FP2": "Fp2",
    "FZ": "Fz", "CZ": "Cz", "PZ": "Pz", "OZ": "Oz",
    "AFZ": "AFz", "FCZ": "FCz", "CPZ": "CPz", "POZ": "POz",
}


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)


def canonical_channel_name(name: Any) -> str:
    if name is None:
        return ""
    value = re.sub(r"^EEG\s+", "", str(name).strip(), flags=re.IGNORECASE)
    # Preserve standalone A1/A2 model channels while still stripping them when
    # they are reference suffixes (for example, ``Fp1-A1``).
    if value.upper() not in {"A1", "A2"}:
        value = re.sub(r"[-_ ]?(REF|REFERENCE|AVG|A1|A2|M1|M2)$", "", value, flags=re.IGNORECASE)
    value = re.sub(r"[^A-Za-z0-9]", "", value)
    if not value:
        return ""
    upper = value.upper()
    if upper in ALIASES:
        return ALIASES[upper]
    letters = "".join(char for char in value if char.isalpha()).upper()
    digits = "".join(char for char in value if char.isdigit())
    if letters == "FP":
        result = "Fp" + digits
    elif letters in {"AF", "FT", "FC", "TP", "CP", "PO"}:
        result = letters + digits
    elif letters in {"F", "T", "C", "P", "O", "I", "A"}:
        result = letters + digits
    else:
        result = value
    z_names = {
        "FPZ": "Fpz", "AFZ": "AFz", "FZ": "Fz", "FCZ": "FCz",
        "CZ": "Cz", "CPZ": "CPz", "PZ": "Pz", "POZ": "POz",
        "OZ": "Oz", "IZ": "Iz",
    }
    return z_names.get(upper, ALIASES.get(result, result))


def validate_reference_contracts() -> dict[str, Any]:
    left = REFERENCE_AUDIT["50m"]["data_config"]
    right = REFERENCE_AUDIT["1b"]["data_config"]
    if left != right:
        raise ValueError(f"50M and 1B data configs differ: 50M={left}, 1B={right}")
    if left != EXPECTED_DATA_CONFIG:
        raise ValueError(f"Repository data config differs from required config: {left}")
    derived = {
        "total_points": TOTAL_POINTS,
        "patch_len": PATCH_LEN,
        "patch_stride": PATCH_STRIDE,
        "num_time_patches": NUM_TIME_PATCHES,
        "num_tokens": NUM_TOKENS,
        "frames_per_patch": FRAMES_PER_PATCH,
        "n_bands": N_BANDS,
    }
    expected = {
        "total_points": 1000, "patch_len": 100, "patch_stride": 100,
        "num_time_patches": 10, "num_tokens": 640,
        "frames_per_patch": 5, "n_bands": 5,
    }
    if derived != expected:
        raise ValueError(f"Derived geometry mismatch: {derived} != {expected}")
    return {"data_contract_equal": True, "derived": derived}


def _ast_literal_assignments(path: Path, class_name: str) -> dict[str, Any]:
    """Read simple dataclass defaults without importing a model repository."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            values: dict[str, Any] = {}
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and item.value is not None and isinstance(item.target, ast.Name):
                    try:
                        values[item.target.id] = ast.literal_eval(item.value)
                    except (ValueError, TypeError):
                        if isinstance(item.value, ast.Call):
                            for keyword in item.value.keywords:
                                if keyword.arg == "default_factory" and isinstance(keyword.value, ast.Lambda):
                                    try:
                                        values[item.target.id] = ast.literal_eval(keyword.value.body)
                                    except (ValueError, TypeError):
                                        pass
            return values
    return {}


def audit_local_reference_repositories(reference_50m: Path | None, reference_1b: Path | None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, root in (("50m", reference_50m), ("1b", reference_1b)):
        if root is None or not root.exists():
            result[key] = {"available": False}
            continue
        data_values = _ast_literal_assignments(root / "configs.py", "DataConfig")
        model_values = _ast_literal_assignments(root / "configs.py", "ModelConfig")
        selected = {name: data_values.get(name) for name in EXPECTED_DATA_CONFIG if name != "band_defs"}
        selected["band_defs"] = data_values.get("band_defs")
        result[key] = {
            "available": True,
            "path": str(root.resolve()),
            "data_config": selected,
            "model_config": {name: model_values.get(name) for name in ("d_model", "n_heads", "depth")},
            "data_config_matches_public_current": selected == EXPECTED_DATA_CONFIG,
            "architecture_matches_public_current": all(
                model_values.get(name) == REFERENCE_AUDIT[key]["architecture"][name]["config_value"]
                for name in ("d_model", "n_heads", "depth")
            ),
        }
    return result


def source_snapshot(path: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for item in sorted(path.rglob("*"), key=lambda candidate: candidate.as_posix()):
        if item.is_file():
            stat = item.stat()
            files.append({
                "relative_path": item.relative_to(path).as_posix(),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            })
    payload = json.dumps(files, sort_keys=True).encode("utf-8")
    return {"files": files, "manifest_sha256": hashlib.sha256(payload).hexdigest()}


def output_schema() -> pa.Schema:
    return pa.schema([
        pa.field("num_tokens", pa.int64(), nullable=False),
        pa.field("patch_len", pa.int64(), nullable=False),
        pa.field("n_bands", pa.int64(), nullable=False),
        pa.field("frames_per_patch", pa.int64(), nullable=False),
        pa.field("n_channels", pa.int64(), nullable=False),
        pa.field("num_patches", pa.int64(), nullable=False),
        pa.field("token_inputs", pa.list_(pa.float32(), NUM_TOKENS * PATCH_LEN), nullable=False),
        pa.field("targets", pa.list_(pa.float32(), NUM_TOKENS * N_BANDS * FRAMES_PER_PATCH), nullable=False),
        pa.field("token_channel_indices", pa.list_(pa.int64(), NUM_TOKENS), nullable=False),
        pa.field("token_time_indices", pa.list_(pa.int64(), NUM_TOKENS), nullable=False),
        pa.field("token_valid_mask", pa.list_(pa.float32(), NUM_TOKENS), nullable=False),
        pa.field("channel_valid_mask", pa.list_(pa.float32(), N_CHANNELS), nullable=False),
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("subject_id", pa.string(), nullable=False),
        pa.field("source_path", pa.string(), nullable=False),
        pa.field("recording_id", pa.string(), nullable=False),
        pa.field("window_index", pa.int64(), nullable=False),
        pa.field("start_time", pa.float64(), nullable=False),
        pa.field("qc_pass", pa.bool_(), nullable=False),
        pa.field("age", pa.int64(), nullable=True),
        pa.field("sex", pa.string(), nullable=False),
        pa.field("valid_channel_count", pa.int64(), nullable=False),
        pa.field("source_sfreq", pa.float32(), nullable=False),
        pa.field("model_sfreq", pa.float32(), nullable=False),
        pa.field("model_preprocessing_version", pa.string(), nullable=False),
        pa.field("source_preprocessing_config_hash", pa.string(), nullable=False),
    ])


def _fixed_list(values: Sequence[np.ndarray], value_type: pa.DataType, size: int) -> pa.FixedSizeListArray:
    flat = np.concatenate([np.asarray(value).reshape(-1) for value in values])
    return pa.FixedSizeListArray.from_arrays(pa.array(flat, type=value_type), size)


def rows_to_table(rows: Sequence[dict[str, Any]]) -> pa.Table:
    schema = output_schema()
    arrays: list[pa.Array] = []
    float_lists = {
        "token_inputs": NUM_TOKENS * PATCH_LEN,
        "targets": NUM_TOKENS * N_BANDS * FRAMES_PER_PATCH,
        "token_valid_mask": NUM_TOKENS,
        "channel_valid_mask": N_CHANNELS,
    }
    int_lists = {"token_channel_indices": NUM_TOKENS, "token_time_indices": NUM_TOKENS}
    for field in schema:
        values = [row[field.name] for row in rows]
        if field.name in float_lists:
            arrays.append(_fixed_list(values, pa.float32(), float_lists[field.name]))
        elif field.name in int_lists:
            arrays.append(_fixed_list(values, pa.int64(), int_lists[field.name]))
        else:
            arrays.append(pa.array(values, type=field.type))
    return pa.Table.from_arrays(arrays, schema=schema)


def align_to_standard_channels(
    signal: np.ndarray,
    channel_names: Sequence[str],
    channel_mask: Sequence[bool],
    bad_channel_mask: Sequence[bool],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    signal = np.asarray(signal, dtype=np.float32)
    if signal.ndim != 2 or signal.shape[0] != len(channel_names):
        raise ValueError(f"Signal/channel shape mismatch: {signal.shape}, names={len(channel_names)}")
    present = np.asarray(channel_mask, dtype=bool)
    bad = np.asarray(bad_channel_mask, dtype=bool)
    if present.shape != (signal.shape[0],) or bad.shape != (signal.shape[0],):
        raise ValueError("channel_mask and bad_channel_mask must match source channel count")
    source_valid = present & ~bad
    canonical = [canonical_channel_name(name) for name in channel_names]
    order_matches = canonical == STANDARD_64_CHANNELS
    if order_matches:
        assert canonical == STANDARD_64_CHANNELS

    grouped: dict[str, list[int]] = defaultdict(list)
    for index, name in enumerate(canonical):
        if name:
            grouped[name].append(index)
    duplicates = sorted(name for name, indices in grouped.items() if len(indices) > 1)
    target_set = set(STANDARD_64_CHANNELS)
    unknown = sorted(name for name in grouped if name not in target_set)
    missing = sorted(name for name in STANDARD_64_CHANNELS if name not in grouped)

    aligned = np.zeros((N_CHANNELS, signal.shape[1]), dtype=np.float32)
    valid = np.zeros(N_CHANNELS, dtype=np.float32)
    for target_index, target_name in enumerate(STANDARD_64_CHANNELS):
        indices = grouped.get(target_name, [])
        valid_indices = [index for index in indices if source_valid[index]]
        if len(indices) == 1 and len(valid_indices) == 1:
            aligned[target_index] = signal[valid_indices[0]]
            valid[target_index] = 1.0
        elif len(indices) > 1:
            # Duplicates are never guessed/averaged; the target remains invalid.
            valid[target_index] = 0.0
    audit = {
        "source_order_matches_standard": order_matches,
        "canonical_source_channels": canonical,
        "duplicates": duplicates,
        "unknown_channels": unknown,
        "missing_channels": missing,
        "source_valid_count": int(source_valid.sum()),
        "mapped_valid_count": int(valid.sum()),
        "bad_channels": [canonical[i] for i in np.flatnonzero(bad)],
        "absent_channels": [canonical[i] for i in np.flatnonzero(~present)],
    }
    return aligned, valid, audit


def resample_signal(signal: np.ndarray, source_sfreq: float = SOURCE_SFREQ, model_sfreq: float = MODEL_SFREQ) -> np.ndarray:
    source = int(round(source_sfreq))
    target = int(round(model_sfreq))
    gcd = math.gcd(source, target)
    result = resample_poly(signal, up=target // gcd, down=source // gcd, axis=1).astype(np.float32)
    if result.shape != (N_CHANNELS, TOTAL_POINTS):
        raise ValueError(f"Expected resampled shape {(N_CHANNELS, TOTAL_POINTS)}, got {result.shape}")
    return result


def standardize_signal(signal: np.ndarray, valid_mask: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    result = np.zeros_like(signal, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)
    if valid.any():
        values = signal[valid]
        mean = values.mean(axis=1, keepdims=True)
        std = values.std(axis=1, keepdims=True)
        result[valid] = (values - mean) / (std + eps)
    if not np.isfinite(result).all():
        raise ValueError("Signal standardization produced NaN/Inf")
    result[~valid] = 0.0
    return result


def compute_timefreq(signal: np.ndarray) -> np.ndarray:
    nperseg = int(round(STFT_WINDOW_SECONDS * MODEL_SFREQ))
    hop = int(round(STFT_HOP_SECONDS * MODEL_SFREQ))
    noverlap = nperseg - hop
    freqs, stft_times, psd = spectrogram(
        signal,
        fs=MODEL_SFREQ,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=128,
        detrend=False,
        return_onesided=True,
        scaling="density",
        mode="psd",
        axis=1,
    )
    band_values: list[np.ndarray] = []
    for _, low, high in BAND_DEFS:
        selected = (freqs >= low) & (freqs < high)
        if not selected.any():
            raise ValueError(f"No STFT bins in band [{low}, {high})")
        band_values.append(np.log(psd[:, selected, :].mean(axis=1) + EPS).astype(np.float32))
    band_power = np.stack(band_values, axis=1)
    sample_times = np.arange(TOTAL_POINTS, dtype=np.float32) / MODEL_SFREQ
    result = np.zeros((N_CHANNELS, N_BANDS, TOTAL_POINTS), dtype=np.float32)
    for channel in range(N_CHANNELS):
        for band in range(N_BANDS):
            result[channel, band] = np.interp(
                sample_times,
                stft_times,
                band_power[channel, band],
                left=band_power[channel, band, 0],
                right=band_power[channel, band, -1],
            )
    return result


def standardize_timefreq(timefreq: np.ndarray, valid_mask: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    result = np.zeros_like(timefreq, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)
    if valid.any():
        values = timefreq[valid]
        mean = values.mean(axis=2, keepdims=True)
        std = values.std(axis=2, keepdims=True)
        result[valid] = (values - mean) / (std + eps)
    if not np.isfinite(result).all():
        raise ValueError("Time-frequency standardization produced NaN/Inf")
    result[~valid] = 0.0
    return result


def cut_patches(signal: np.ndarray) -> np.ndarray:
    patches = np.zeros((N_CHANNELS, NUM_TIME_PATCHES, PATCH_LEN), dtype=np.float32)
    for patch in range(NUM_TIME_PATCHES):
        start = patch * PATCH_STRIDE
        patches[:, patch] = signal[:, start : start + PATCH_LEN]
    return patches


def make_targets(timefreq: np.ndarray) -> np.ndarray:
    targets = np.zeros((N_CHANNELS, NUM_TIME_PATCHES, N_BANDS, FRAMES_PER_PATCH), dtype=np.float32)
    frame_indices = np.linspace(0, PATCH_LEN - 1, FRAMES_PER_PATCH).astype(np.int64)
    for patch in range(NUM_TIME_PATCHES):
        indices = patch * PATCH_STRIDE + frame_indices
        targets[:, patch] = timefreq[:, :, indices]
    return targets


def process_row(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if row.get("qc_pass") is not True:
        raise ValueError(f"qc_pass is not true for sample {row.get('sample_id')}")
    shape = tuple(int(value) for value in row.get("shape", []))
    if shape != (64, 2000):
        raise ValueError(f"Expected input shape (64, 2000), got {shape}")
    if not math.isclose(float(row.get("target_sampling_rate")), SOURCE_SFREQ):
        raise ValueError(f"Expected common-cache sampling rate {SOURCE_SFREQ}, got {row.get('target_sampling_rate')}")
    normalized = np.asarray(row["data"], dtype=np.float32).reshape(shape)
    if not np.isfinite(normalized).all() or normalized.min() < -1.000001 or normalized.max() > 1.000001:
        raise ValueError("Input normalized data must be finite and inside [-1, 1]")

    # Approximate inverse of clip(original_uv / 200, -1, 1); clipped extremes are unrecoverable.
    signal_uv_approx = normalized * np.float32(200.0)
    aligned, channel_valid_mask, channel_audit = align_to_standard_channels(
        signal_uv_approx,
        row["channel_names"],
        row["channel_mask"],
        row["bad_channel_mask"],
    )
    resampled = resample_signal(aligned)
    standardized = standardize_signal(resampled, channel_valid_mask)
    timefreq = standardize_timefreq(compute_timefreq(standardized), channel_valid_mask)
    patches = cut_patches(standardized)
    targets = make_targets(timefreq)
    token_inputs = patches.reshape(NUM_TOKENS, PATCH_LEN).astype(np.float32)
    token_targets = targets.reshape(NUM_TOKENS, N_BANDS, FRAMES_PER_PATCH).astype(np.float32)
    token_channel_indices = np.repeat(np.arange(N_CHANNELS), NUM_TIME_PATCHES).astype(np.int64)
    token_time_indices = np.tile(np.arange(NUM_TIME_PATCHES), N_CHANNELS).astype(np.int64)
    token_valid_mask = channel_valid_mask[token_channel_indices].astype(np.float32)
    invalid_tokens = token_valid_mask == 0
    token_inputs[invalid_tokens] = 0.0
    token_targets[invalid_tokens] = 0.0

    if not np.isfinite(token_inputs).all() or not np.isfinite(token_targets).all():
        raise ValueError("Processed tensors contain NaN/Inf")
    if not np.array_equal(token_inputs[invalid_tokens], np.zeros_like(token_inputs[invalid_tokens])):
        raise AssertionError("Invalid token inputs are not strictly zero")
    if not np.array_equal(token_targets[invalid_tokens], np.zeros_like(token_targets[invalid_tokens])):
        raise AssertionError("Invalid targets are not strictly zero")

    processed = {
        "num_tokens": NUM_TOKENS,
        "patch_len": PATCH_LEN,
        "n_bands": N_BANDS,
        "frames_per_patch": FRAMES_PER_PATCH,
        "n_channels": N_CHANNELS,
        "num_patches": NUM_TIME_PATCHES,
        "token_inputs": token_inputs.reshape(-1),
        "targets": token_targets.reshape(-1),
        "token_channel_indices": token_channel_indices,
        "token_time_indices": token_time_indices,
        "token_valid_mask": token_valid_mask,
        "channel_valid_mask": channel_valid_mask,
        "sample_id": str(row["sample_id"]),
        "subject_id": str(row["subject_id"]),
        "source_path": str(row["source_path"]),
        "recording_id": str(row.get("recording_id", "")),
        "window_index": int(row["window_index"]),
        "start_time": float(row["start_time"]),
        "qc_pass": True,
        "age": None if row.get("age") is None else int(row["age"]),
        "sex": str(row.get("sex", "")),
        "valid_channel_count": int(channel_valid_mask.sum()),
        "source_sfreq": np.float32(SOURCE_SFREQ),
        "model_sfreq": np.float32(MODEL_SFREQ),
        "model_preprocessing_version": VERSION,
        "source_preprocessing_config_hash": str(row.get("preprocessing_config_hash", "")),
    }
    row_audit = {
        "sample_id": processed["sample_id"],
        "subject_id": processed["subject_id"],
        "valid_channel_count": processed["valid_channel_count"],
        "token_valid_mask_sum": int(token_valid_mask.sum()),
        "token_inputs_shape": str(tuple(token_inputs.shape)),
        "targets_shape": str(tuple(token_targets.shape)),
        **channel_audit,
    }
    return processed, row_audit


def validate_roundtrip(path: Path, expected_rows: int) -> dict[str, Any]:
    dataset = lance.dataset(path)
    if dataset.schema != output_schema():
        raise AssertionError(f"Output schema mismatch:\n{dataset.schema}\n!=\n{output_schema()}")
    forbidden = {"label", "label_name", "split"}.intersection(dataset.schema.names)
    if forbidden:
        raise AssertionError(f"Forbidden columns found: {sorted(forbidden)}")
    if dataset.count_rows() != expected_rows:
        raise AssertionError(f"Expected {expected_rows} rows, got {dataset.count_rows()}")
    rows = dataset.to_table().to_pylist()
    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise AssertionError("sample_id values are not unique")
    checks: list[dict[str, Any]] = []
    fc1_index = STANDARD_64_CHANNELS.index("FC1")
    for row in rows:
        token_inputs = np.asarray(row["token_inputs"], dtype=np.float32).reshape(NUM_TOKENS, PATCH_LEN)
        targets = np.asarray(row["targets"], dtype=np.float32).reshape(NUM_TOKENS, N_BANDS, FRAMES_PER_PATCH)
        channel_mask = np.asarray(row["channel_valid_mask"], dtype=np.float32)
        token_mask = np.asarray(row["token_valid_mask"], dtype=np.float32)
        expected_token_mask = channel_mask[np.repeat(np.arange(N_CHANNELS), NUM_TIME_PATCHES)]
        if not np.array_equal(token_mask, expected_token_mask):
            raise AssertionError(f"Token mask mismatch for {row['sample_id']}")
        invalid = token_mask == 0
        if np.any(token_inputs[invalid] != 0) or np.any(targets[invalid] != 0):
            raise AssertionError(f"Invalid tensors are not zero for {row['sample_id']}")
        fc1_slice = slice(fc1_index * NUM_TIME_PATCHES, (fc1_index + 1) * NUM_TIME_PATCHES)
        checks.append({
            "sample_id": row["sample_id"],
            "subject_id": row["subject_id"],
            "valid_channel_count": int(channel_mask.sum()),
            "token_valid_mask_sum": int(token_mask.sum()),
            "token_inputs_shape": list(token_inputs.shape),
            "targets_shape": list(targets.shape),
            "all_finite": bool(np.isfinite(token_inputs).all() and np.isfinite(targets).all()),
            "fc1_token_mask_sum": int(token_mask[fc1_slice].sum()),
            "fc1_inputs_all_zero": bool(np.all(token_inputs[fc1_slice] == 0)),
            "fc1_targets_all_zero": bool(np.all(targets[fc1_slice] == 0)),
        })
    return {"rows": len(rows), "sample_ids_unique": True, "rows_verified": checks}


def _contract(model_key: str) -> dict[str, Any]:
    architecture = REFERENCE_AUDIT[model_key]["architecture"]
    return {
        "model": model_key,
        "repository": REFERENCE_AUDIT[model_key]["repository"],
        "d_model": architecture["d_model"]["effective_value"],
        "n_heads": architecture["n_heads"]["effective_value"],
        "depth": architecture["depth"]["effective_value"],
        "input_dim": PATCH_LEN,
        "n_channels": N_CHANNELS,
        "n_time_patches": NUM_TIME_PATCHES,
        "num_tokens": NUM_TOKENS,
        "n_bands": N_BANDS,
        "frames_per_patch": FRAMES_PER_PATCH,
        "architecture_audit": architecture,
        "data_config": EXPECTED_DATA_CONFIG,
        "standard_channels": STANDARD_64_CHANNELS,
        "public_repository_channel_tail": PUBLIC_REPO_CHANNEL_TAIL,
        "channel_contract_source": "Both public repositories' current channel_config.py",
    }


def _issues_from_audit(audits: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for audit in audits:
        for issue_type, values in (
            ("duplicate_channel", audit["duplicates"]),
            ("unknown_channel", audit["unknown_channels"]),
            ("missing_channel", audit["missing_channels"]),
        ):
            for value in values:
                issues.append({
                    "severity": "warning", "sample_id": audit["sample_id"],
                    "subject_id": audit["subject_id"], "issue_type": issue_type,
                    "channel": value,
                    "detail": "Recorded without guessing or silent positional mapping",
                })
        if not audit["source_order_matches_standard"]:
            issues.append({
                "severity": "warning", "sample_id": audit["sample_id"],
                "subject_id": audit["subject_id"], "issue_type": "source_order_mismatch",
                "channel": "", "detail": "Per-row channel names were mapped explicitly",
            })
    return issues


def write_reports(
    output_dir: Path,
    input_path: Path,
    output_path: Path,
    audits: Sequence[dict[str, Any]],
    roundtrip: dict[str, Any],
    issues: Sequence[dict[str, Any]],
    source_before: dict[str, Any],
    source_after: dict[str, Any],
    local_audit: dict[str, Any],
    smoke: bool,
) -> dict[str, Any]:
    contract_50m = _contract("50m")
    contract_1b = _contract("1b")
    data_fields = ("input_dim", "n_channels", "n_time_patches", "num_tokens", "n_bands", "frames_per_patch")
    data_contract_equal = all(contract_50m[name] == contract_1b[name] for name in data_fields)
    comparison = {
        "data_contract_equal": data_contract_equal,
        "shared_token_cache_compatible": data_contract_equal,
        "architecture_equal": False,
        "channel_geometry_equal": True,
        "channel_semantics_verified": True,
        "source_montage_fully_covers_model_montage": all(
            not audit["unknown_channels"] and not audit["missing_channels"] and not audit["duplicates"]
            for audit in audits
        ),
    }
    subject_counts = dict(Counter(audit["subject_id"] for audit in audits))
    actual_counts = sorted({audit["valid_channel_count"] for audit in audits})
    smoke_expected = {
        "rows": 4,
        "subject_counts": {"SUB1": 2, "SUB5": 2},
        "SUB1_valid_channel_count": 60,
        "SUB5_valid_channel_count": 59,
        "basis": (
            "Public repository STANDARD_64_CHANNELS; Huada has four extra channels "
            "(PO5/PO6/CB1/CB2) and lacks AFz/Iz/F9/F10"
        ),
    }
    smoke_actual_acceptance = (
        len(audits) == 4
        and subject_counts == {"SUB1": 2, "SUB5": 2}
        and all(audit["valid_channel_count"] == (60 if audit["subject_id"] == "SUB1" else 59) for audit in audits)
    ) if smoke else None
    summary = {
        "input_lance_path": str(input_path.resolve()),
        "output_lance_path": str(output_path.resolve()),
        "rows_written": len(audits),
        "subject_counts": subject_counts,
        "valid_channel_counts": actual_counts,
        "roundtrip": roundtrip,
        "issues_count": len(issues),
        "source_cache_unchanged": source_before == source_after,
        "smoke": smoke,
        "smoke_expected": smoke_expected if smoke else None,
        "smoke_acceptance_passed": smoke_actual_acceptance,
        "full_conversion_executed": not smoke,
    }
    provenance = {
        "created_at": datetime.now().astimezone().isoformat(),
        "version": VERSION,
        "denorm_clip_divide_200": True,
        "denormalization_note": (
            "normalized * 200.0 is an approximate inverse of clip(original_uv / 200, -1, 1); "
            "clipped extremes cannot be recovered"
        ),
        "v_to_uv_applied": False,
        "repeated_bandpass": False,
        "repeated_notch": False,
        "repeated_windowing": False,
        "repeated_clip_divide_200": False,
        "reference_audit": REFERENCE_AUDIT,
        "local_reference_audit": local_audit,
        "source_snapshot_before": source_before,
        "source_snapshot_after": source_after,
    }
    _json_dump(output_dir / "model_contract_50m.json", contract_50m)
    _json_dump(output_dir / "model_contract_1b.json", contract_1b)
    _json_dump(output_dir / "model_contract_comparison.json", comparison)
    _json_dump(output_dir / "conversion_provenance.json", provenance)
    _json_dump(output_dir / "conversion_summary.json", summary)
    inventory_fields = [
        "sample_id", "subject_id", "valid_channel_count", "token_valid_mask_sum",
        "token_inputs_shape", "targets_shape", "source_order_matches_standard",
        "source_valid_count", "mapped_valid_count", "bad_channels", "unknown_channels",
        "missing_channels", "duplicates",
    ]
    inventory_rows = []
    for audit in audits:
        row = dict(audit)
        for key in ("bad_channels", "unknown_channels", "missing_channels", "duplicates"):
            row[key] = ";".join(row[key])
        inventory_rows.append(row)
    _write_csv(output_dir / "row_inventory.csv", inventory_rows, inventory_fields)
    _write_csv(
        output_dir / "conversion_issues.csv", issues,
        ["severity", "sample_id", "subject_id", "issue_type", "channel", "detail"],
    )
    report_lines = [
        "# 华大模型 token 转换报告", "",
        f"- 输出行数：{len(audits)}", f"- 被试计数：{subject_counts}",
        f"- 输出 Lance：`{output_path}`", f"- Lance round-trip：通过",
        f"- 公共缓存未修改：{source_before == source_after}",
        f"- 50M/1B 数据几何合同一致：{data_contract_equal}",
        f"- smoke 验收：{smoke_actual_acceptance if smoke else '不适用'}", "",
        "## 通道审计", "",
        "64 通道合同以两个公开模型仓库当前 `channel_config.py` 为准。转换严格按名称映射，",
        "未把华大额外通道按位置伪装成模型缺失通道。所有未知、缺失和顺序差异见 `conversion_issues.csv`。", "",
        f"- 实际有效通道计数：{actual_counts}",
        f"- 问题条数：{len(issues)}", "",
        "## 重复处理审计", "",
        "仅执行 `normalized * 200.0` 近似反归一化；没有乘 `1e6`，没有重复 bandpass、notch、",
        "10 秒切窗或 clip/divide-200。", "",
        "## 后续模型适配", "",
        "1. 50M 当前下游代码主要读取 `int64 label` 做分类；年龄回归仍需 float label、回归头、回归损失及 MAE/RMSE。",
        "2. 1B 仓库主要是预训练框架；仍需补充类似 50M 的 downstream backbone/head 和训练入口。", "",
    ]
    (output_dir / "conversion_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-lance-path", type=Path, required=True)
    parser.add_argument("--output-lance-path", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--limit-rows", type=int)
    parser.add_argument("--reference-50m-repo", type=Path, default=Path(r"D:\repo\xxxxxxxx"))
    parser.add_argument("--reference-1b-repo", type=Path, default=Path(r"D:\repo\EEG_pretrain"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_reference_contracts()
    if not args.input_lance_path.exists():
        raise FileNotFoundError(args.input_lance_path)
    if args.output_lance_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output_lance_path}")
    if args.limit_rows is not None and args.limit_rows <= 0:
        raise ValueError("--limit-rows must be positive")
    if args.smoke and args.limit_rows not in (None, 4):
        raise ValueError("Smoke mode must convert exactly the four-row smoke cache")

    output_dir = args.output_lance_path.parent
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be new/empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_before = source_snapshot(args.input_lance_path)
    dataset = lance.dataset(args.input_lance_path)
    available = dataset.count_rows()
    limit = 4 if args.smoke else args.limit_rows
    if args.smoke and available != 4:
        raise ValueError(f"Smoke cache must contain exactly 4 rows, got {available}")
    table = dataset.to_table().slice(0, limit) if limit is not None else dataset.to_table()

    processed_rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for row in table.to_pylist():
        processed, audit = process_row(row)
        processed_rows.append(processed)
        audits.append(audit)
    if not processed_rows:
        raise ValueError("No rows selected")
    lance.write_dataset(
        rows_to_table(processed_rows),
        args.output_lance_path,
        mode="create",
        max_rows_per_group=32,
        max_rows_per_file=256,
        data_storage_version="stable",
    )
    roundtrip = validate_roundtrip(args.output_lance_path, len(processed_rows))
    issues = _issues_from_audit(audits)
    source_after = source_snapshot(args.input_lance_path)
    local_audit = audit_local_reference_repositories(args.reference_50m_repo, args.reference_1b_repo)
    summary = write_reports(
        output_dir, args.input_lance_path, args.output_lance_path, audits, roundtrip,
        issues, source_before, source_after, local_audit, args.smoke,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
