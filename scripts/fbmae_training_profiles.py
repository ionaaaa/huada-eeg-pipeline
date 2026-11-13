#!/usr/bin/env python3
"""Apply joint-training channel profiles without duplicating token caches."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from build_huada_model_tokens import NUM_TIME_PATCHES, PATCH_LEN, STANDARD_64_CHANNELS


PROFILE_NAMES = {"matched18", "native64_vs_sparse18"}
DATASET_NAMES = {"dortmund", "mipdb"}


def load_training_profiles(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    profiles = payload.get("profiles", {})
    if set(profiles) != PROFILE_NAMES:
        raise ValueError(f"training profile names must be {sorted(PROFILE_NAMES)}")
    standard = set(STANDARD_64_CHANNELS)
    for profile_name, profile in profiles.items():
        for dataset in sorted(DATASET_NAMES):
            value = profile.get(f"{dataset}_allowed_target_channels")
            if value == "all":
                continue
            if not isinstance(value, list) or len(value) != 18 or len(set(value)) != 18:
                raise ValueError(f"{profile_name}/{dataset} must define 18 unique channels or 'all'")
            unknown = sorted(set(value) - standard)
            if unknown:
                raise ValueError(f"{profile_name}/{dataset} has unknown standard channels: {unknown}")
    return payload


def allowed_channel_mask(
    profiles: Mapping[str, Any], profile_name: str, dataset_name: str,
) -> np.ndarray:
    if profile_name not in PROFILE_NAMES:
        raise ValueError(f"unknown training profile: {profile_name}")
    dataset = dataset_name.casefold()
    if dataset not in DATASET_NAMES:
        raise ValueError(f"unknown dataset name: {dataset_name}")
    value = profiles["profiles"][profile_name][f"{dataset}_allowed_target_channels"]
    if value == "all":
        return np.ones(len(STANDARD_64_CHANNELS), dtype=bool)
    allowed = set(value)
    return np.asarray([name in allowed for name in STANDARD_64_CHANNELS], dtype=bool)


def apply_training_profile(
    *,
    token_inputs: np.ndarray,
    targets: np.ndarray,
    channel_valid_mask: Sequence[bool],
    profiles: Mapping[str, Any],
    profile_name: str,
    dataset_name: str,
) -> dict[str, np.ndarray]:
    """Return profile-masked copies; true/one always means valid."""
    inputs = np.asarray(token_inputs, dtype=np.float32).reshape(
        len(STANDARD_64_CHANNELS) * NUM_TIME_PATCHES, PATCH_LEN
    ).copy()
    target_array = np.asarray(targets, dtype=np.float32).copy()
    if target_array.shape[0] != len(STANDARD_64_CHANNELS) * NUM_TIME_PATCHES:
        target_array = target_array.reshape(len(STANDARD_64_CHANNELS) * NUM_TIME_PATCHES, -1)
    else:
        target_array = target_array.copy()
    source_valid = np.asarray(channel_valid_mask, dtype=bool)
    if source_valid.shape != (len(STANDARD_64_CHANNELS),):
        raise ValueError("channel_valid_mask must have 64 entries")
    profile_allowed = allowed_channel_mask(profiles, profile_name, dataset_name)
    channel_valid = source_valid & profile_allowed
    token_valid = np.repeat(channel_valid, NUM_TIME_PATCHES)
    inputs[~token_valid] = 0.0
    target_array[~token_valid] = 0.0
    return {
        "token_inputs": inputs,
        "targets": target_array,
        "channel_valid_mask": channel_valid.astype(np.float32),
        "token_valid_mask": token_valid.astype(np.float32),
    }
