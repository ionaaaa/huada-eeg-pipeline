#!/usr/bin/env python3
"""Resolve and apply FBMAE dataset profiles without duplicating token caches."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from build_huada_model_tokens import NUM_TIME_PATCHES, PATCH_LEN, STANDARD_64_CHANNELS


DATASET_NAMES = {"dortmund", "mipdb"}
CHANNEL_MODES = {"native64", "matched18", "sparse18"}


def load_training_profiles(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    profiles = payload.get("profiles")
    shared = payload.get("shared_target_channels")
    modes = payload.get("channel_modes")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("training profiles must define at least one profile")
    if not isinstance(shared, list) or len(shared) != 18 or len(set(shared)) != 18:
        raise ValueError("shared_target_channels must contain 18 unique channels")
    unknown = sorted(set(shared) - set(STANDARD_64_CHANNELS))
    if unknown:
        raise ValueError(f"unknown shared standard64 channels: {unknown}")
    if not isinstance(modes, dict) or set(modes) != CHANNEL_MODES:
        raise ValueError(f"channel_modes must be {sorted(CHANNEL_MODES)}")
    if modes["native64"].get("allowed_target_channels") != "all":
        raise ValueError("native64 must allow all original valid slots")
    for mode in ("matched18", "sparse18"):
        if modes[mode].get("allowed_target_channels") != "shared_target_channels":
            raise ValueError(f"{mode} must reference shared_target_channels")
    for profile_name, profile in profiles.items():
        datasets = profile.get("datasets")
        filters = profile.get("dataset_filters")
        if not isinstance(datasets, dict) or not datasets:
            raise ValueError(f"{profile_name} must define datasets")
        if not isinstance(filters, dict) or set(filters) != set(datasets):
            raise ValueError(f"{profile_name} dataset_filters must match datasets")
        for dataset_id, channel_mode in datasets.items():
            if dataset_id not in DATASET_NAMES:
                raise ValueError(f"{profile_name} has unknown dataset: {dataset_id}")
            if channel_mode not in CHANNEL_MODES:
                raise ValueError(f"{profile_name}/{dataset_id} has unknown channel mode: {channel_mode}")
            if dataset_id == "mipdb" and channel_mode != "sparse18":
                raise ValueError(f"{profile_name}: MIPDB must use sparse18")
            age_filter = filters[dataset_id]
            if set(age_filter) != {"min_age", "max_age"}:
                raise ValueError(f"{profile_name}/{dataset_id} must define min_age and max_age")
            if float(age_filter["min_age"]) > float(age_filter["max_age"]):
                raise ValueError(f"{profile_name}/{dataset_id} has reversed age range")
    return payload


def resolve_training_profile(
    profiles: Mapping[str, Any], profile_name: str, dataset_id: str,
) -> dict[str, Any]:
    dataset = str(dataset_id).casefold()
    if dataset not in DATASET_NAMES:
        raise ValueError(f"unknown dataset_id: {dataset_id}")
    try:
        profile = profiles["profiles"][profile_name]
    except KeyError as error:
        raise ValueError(f"unknown training profile: {profile_name}") from error
    if dataset not in profile["datasets"]:
        raise ValueError(f"training profile {profile_name} does not include {dataset}")
    mode_name = profile["datasets"][dataset]
    value = profiles["channel_modes"][mode_name]["allowed_target_channels"]
    allowed = list(STANDARD_64_CHANNELS) if value == "all" else list(profiles[value])
    age_filter = profile["dataset_filters"][dataset]
    return {
        "profile_name": profile_name,
        "dataset_id": dataset,
        "channel_mode": mode_name,
        "allowed_target_channels": allowed,
        "min_age": float(age_filter["min_age"]),
        "max_age": float(age_filter["max_age"]),
    }


def allowed_channel_mask(profile: Mapping[str, Any]) -> np.ndarray:
    allowed = set(profile["allowed_target_channels"])
    return np.asarray([name in allowed for name in STANDARD_64_CHANNELS], dtype=bool)


def apply_training_profile(
    token_inputs: np.ndarray,
    token_valid_mask: Sequence[bool],
    dataset_id: str,
    profile: Mapping[str, Any],
    *,
    targets: np.ndarray | None = None,
    channel_valid_mask: Sequence[bool] | None = None,
) -> dict[str, np.ndarray]:
    """Apply an additional allow-mask while preserving the fixed [640,100] contract."""
    if str(dataset_id).casefold() != str(profile.get("dataset_id", "")).casefold():
        raise ValueError("dataset_id does not match resolved training profile")
    num_channels = len(STANDARD_64_CHANNELS)
    num_tokens = num_channels * NUM_TIME_PATCHES
    inputs = np.asarray(token_inputs, dtype=np.float32).reshape(num_tokens, PATCH_LEN).copy()
    source_token_valid = np.asarray(token_valid_mask, dtype=bool).reshape(num_tokens)
    channel_allowed = allowed_channel_mask(profile)
    token_allowed = np.repeat(channel_allowed, NUM_TIME_PATCHES)
    token_valid = source_token_valid & token_allowed
    inputs[~token_valid] = 0.0

    if channel_valid_mask is None:
        source_channel_valid = source_token_valid.reshape(num_channels, NUM_TIME_PATCHES).any(axis=1)
    else:
        source_channel_valid = np.asarray(channel_valid_mask, dtype=bool)
        if source_channel_valid.shape != (num_channels,):
            raise ValueError("channel_valid_mask must have 64 entries")
    channel_valid = source_channel_valid & channel_allowed
    result = {
        "token_inputs": inputs,
        "channel_valid_mask": channel_valid.astype(np.float32),
        "token_valid_mask": token_valid.astype(np.float32),
    }
    if targets is not None:
        target_array = np.asarray(targets, dtype=np.float32).copy()
        if target_array.shape[0] != num_tokens:
            target_array = target_array.reshape(num_tokens, -1)
        target_array[~token_valid] = 0.0
        result["targets"] = target_array
    return result


class ProfiledTokenDataset(Dataset):
    """Read-time view applying one resolved profile to an existing token cache."""

    def __init__(self, dataset: Dataset, dataset_id: str, profile: Mapping[str, Any]):
        self.dataset = dataset
        self.dataset_id = str(dataset_id).casefold()
        self.profile = dict(profile)

    def __len__(self) -> int:
        return len(self.dataset)

    def _apply(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        output = dict(sample)
        transformed = apply_training_profile(
            sample["token_inputs"].detach().cpu().numpy(),
            sample["token_valid_mask"].detach().cpu().numpy(),
            self.dataset_id,
            self.profile,
            targets=sample.get("targets").detach().cpu().numpy() if "targets" in sample else None,
            channel_valid_mask=(
                sample.get("channel_valid_mask").detach().cpu().numpy()
                if "channel_valid_mask" in sample else None
            ),
        )
        for key, value in transformed.items():
            output[key] = torch.from_numpy(value)
        return output

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._apply(self.dataset[index])

    def __getitems__(self, indices: Sequence[int]) -> list[dict[str, Any]]:
        if hasattr(self.dataset, "__getitems__"):
            samples = self.dataset.__getitems__(indices)
        else:
            samples = [self.dataset[index] for index in indices]
        return [self._apply(sample) for sample in samples]

    def get_sampling_rows(self) -> list[dict[str, Any]]:
        return self.dataset.get_sampling_rows()
