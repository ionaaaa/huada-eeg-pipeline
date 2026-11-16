#!/usr/bin/env python3
"""Authoritative FBMAE channel-profile and checkpoint deployment contracts."""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from build_huada_model_tokens import NUM_TIME_PATCHES, PATCH_LEN, STANDARD_64_CHANNELS


TOKEN_MASK_SEMANTICS_VERSION = "token_valid_mask_true_is_valid_v1"
CHECKPOINT_PROFILE_CONTRACT_VERSION = "fbmae-deployment-profile-v1"
NATIVE64_PROFILE_ID = "standard64_native_v1"
SPARSE18_PROFILE_ID = "standard64_sparse18_v1"
SPARSE18_CHANNELS = [
    "Fp1", "Fp2", "Fz", "F3", "F7", "C3", "T7", "P3", "P7",
    "Pz", "O1", "O2", "P4", "P8", "C4", "T8", "F8", "F4",
]
LEGACY_DEPLOYMENT_PROFILE_IDS = {
    "dortmund_native64": NATIVE64_PROFILE_ID,
    "dortmund_matched18": SPARSE18_PROFILE_ID,
    "joint_matched18": SPARSE18_PROFILE_ID,
    "joint_native64_vs_sparse18": NATIVE64_PROFILE_ID,
}
DATASET_NAMES = {"dortmund", "mipdb"}


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _build_channel_profile(profile_id: str, allowed_channels: Sequence[str]) -> dict[str, Any]:
    allowed = [str(value) for value in allowed_channels]
    if not profile_id:
        raise ValueError("channel profile ID must not be empty")
    if not allowed or len(allowed) != len(set(allowed)):
        raise ValueError(f"{profile_id} must contain unique allowed channels")
    unknown = sorted(set(allowed) - set(STANDARD_64_CHANNELS))
    if unknown:
        raise ValueError(f"{profile_id} has channels outside STANDARD_64_CHANNELS: {unknown}")
    payload = {
        "profile_id": profile_id,
        "base_axis": list(STANDARD_64_CHANNELS),
        "allowed_channels": allowed,
        "num_model_slots": len(STANDARD_64_CHANNELS),
        "num_allowed_slots": len(allowed),
        "num_allowed_tokens": len(allowed) * NUM_TIME_PATCHES,
        "token_mask_semantics_version": TOKEN_MASK_SEMANTICS_VERSION,
        "effective_valid_token_policy": "source_token_valid_and_profile_allowed",
    }
    payload["canonical_sha256"] = _canonical_sha256(payload)
    return payload


CHANNEL_PROFILE_REGISTRY = {
    NATIVE64_PROFILE_ID: _build_channel_profile(NATIVE64_PROFILE_ID, STANDARD_64_CHANNELS),
    SPARSE18_PROFILE_ID: _build_channel_profile(SPARSE18_PROFILE_ID, SPARSE18_CHANNELS),
}


def get_channel_profile(profile_id: str) -> dict[str, Any]:
    try:
        profile = dict(CHANNEL_PROFILE_REGISTRY[str(profile_id)])
    except KeyError as error:
        raise ValueError(f"unknown channel profile: {profile_id}") from error
    supplied_sha = profile.pop("canonical_sha256")
    actual_sha = _canonical_sha256(profile)
    if supplied_sha != actual_sha:
        raise ValueError(f"registry SHA mismatch for {profile_id}")
    profile["canonical_sha256"] = actual_sha
    return profile


def load_training_profiles(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("checkpoint_profile_contract_version") != CHECKPOINT_PROFILE_CONTRACT_VERSION:
        raise ValueError("training profile contract version mismatch")
    profiles = payload.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("training profile catalog must define at least one profile")
    missing_frozen = sorted(set(LEGACY_DEPLOYMENT_PROFILE_IDS) - set(profiles))
    if missing_frozen:
        raise ValueError(f"training profile catalog is missing frozen profile IDs: {missing_frozen}")
    for profile_name, profile in profiles.items():
        datasets = profile.get("datasets")
        filters = profile.get("dataset_filters")
        deployments = profile.get("deployment_profiles")
        if not isinstance(datasets, dict) or not datasets:
            raise ValueError(f"{profile_name} must define datasets")
        if not isinstance(filters, dict) or set(filters) != set(datasets):
            raise ValueError(f"{profile_name} dataset_filters must match datasets")
        if not isinstance(deployments, dict) or set(deployments) != {"huada"}:
            raise ValueError(f"{profile_name} must define deployment_profiles.huada")
        for dataset_id, channel_profile_id in datasets.items():
            if dataset_id not in DATASET_NAMES:
                raise ValueError(f"{profile_name} has unknown dataset: {dataset_id}")
            get_channel_profile(channel_profile_id)
            age_filter = filters[dataset_id]
            if not {"min_age", "max_age"}.issubset(age_filter) or (
                set(age_filter) - {"min_age", "max_age", "block_id"}
            ):
                raise ValueError(f"{profile_name}/{dataset_id} must define min_age and max_age")
            if "block_id" in age_filter and (
                dataset_id != "mipdb" or not str(age_filter["block_id"]).strip()
            ):
                raise ValueError(f"{profile_name}/{dataset_id} has invalid block_id filter")
            if float(age_filter["min_age"]) > float(age_filter["max_age"]):
                raise ValueError(f"{profile_name}/{dataset_id} has reversed age range")
        deployment_id = deployments["huada"]
        get_channel_profile(deployment_id)
        if (
            profile_name in LEGACY_DEPLOYMENT_PROFILE_IDS
            and deployment_id != LEGACY_DEPLOYMENT_PROFILE_IDS[profile_name]
        ):
            raise ValueError(f"{profile_name} has an invalid frozen Huada deployment mapping")
        if profile_name.startswith("joint_") and set(datasets) != {"dortmund", "mipdb"}:
            raise ValueError(f"{profile_name} must contain Dortmund and MIPDB")
        if "mipdb" in datasets and datasets["mipdb"] != SPARSE18_PROFILE_ID:
            raise ValueError(f"{profile_name}: MIPDB must use {SPARSE18_PROFILE_ID}")
    return payload


def resolve_training_profile(
    profiles: Mapping[str, Any], profile_name: str, dataset_id: str,
) -> dict[str, Any]:
    dataset = str(dataset_id).casefold()
    if dataset not in DATASET_NAMES:
        raise ValueError(f"unknown dataset_id: {dataset_id}")
    try:
        training = profiles["profiles"][profile_name]
    except KeyError as error:
        raise ValueError(f"unknown training profile: {profile_name}") from error
    if dataset not in training["datasets"]:
        raise ValueError(f"training profile {profile_name} does not include {dataset}")
    channel_profile = get_channel_profile(training["datasets"][dataset])
    age_filter = training["dataset_filters"][dataset]
    return {
        "profile_id": profile_name,
        "profile_name": profile_name,
        "dataset_id": dataset,
        "channel_profile_id": channel_profile["profile_id"],
        "channel_profile_sha256": channel_profile["canonical_sha256"],
        "allowed_target_channels": channel_profile["allowed_channels"],
        "min_age": float(age_filter["min_age"]),
        "max_age": float(age_filter["max_age"]),
        "dataset_filter": dict(age_filter),
    }


def build_checkpoint_profile_contract(
    profiles: Mapping[str, Any], profile_name: str,
) -> dict[str, Any]:
    try:
        training = profiles["profiles"][profile_name]
    except KeyError as error:
        raise ValueError(f"unknown training profile: {profile_name}") from error
    deployments = {}
    for target_dataset, channel_profile_id in training["deployment_profiles"].items():
        channel = get_channel_profile(channel_profile_id)
        deployments[target_dataset] = {
            "channel_profile_id": channel["profile_id"],
            "channel_profile_sha256": channel["canonical_sha256"],
            "num_model_slots": channel["num_model_slots"],
            "num_allowed_slots": channel["num_allowed_slots"],
            "num_allowed_tokens": channel["num_allowed_tokens"],
            "token_mask_semantics_version": channel["token_mask_semantics_version"],
            "effective_valid_token_policy": channel["effective_valid_token_policy"],
        }
    return {
        "checkpoint_profile_contract_version": CHECKPOINT_PROFILE_CONTRACT_VERSION,
        "training_profile": {
            "profile_id": profile_name,
            "datasets": dict(training["datasets"]),
            "dataset_filters": {
                key: dict(value) for key, value in training["dataset_filters"].items()
            },
        },
        "deployment_profiles": deployments,
    }


def resolve_checkpoint_deployment_profile(
    checkpoint_payload: Mapping[str, Any],
    target_dataset: str,
    requested_profile: str = "auto",
    allow_profile_override: bool = False,
) -> dict[str, Any]:
    target = str(target_dataset).casefold()
    deployments = checkpoint_payload.get("deployment_profiles")
    inferred = False
    inference_warning = None
    if isinstance(deployments, Mapping):
        if (
            checkpoint_payload.get("checkpoint_profile_contract_version")
            != CHECKPOINT_PROFILE_CONTRACT_VERSION
        ):
            raise ValueError("checkpoint deployment profile contract version mismatch")
        contract = deployments.get(target)
        if not isinstance(contract, Mapping):
            raise ValueError(f"checkpoint has no deployment profile for {target}")
        checkpoint_profile_id = str(contract.get("channel_profile_id", ""))
        checkpoint_profile = get_channel_profile(checkpoint_profile_id)
        expected = {
            "channel_profile_sha256": checkpoint_profile["canonical_sha256"],
            "num_model_slots": checkpoint_profile["num_model_slots"],
            "num_allowed_slots": checkpoint_profile["num_allowed_slots"],
            "num_allowed_tokens": checkpoint_profile["num_allowed_tokens"],
            "token_mask_semantics_version": checkpoint_profile["token_mask_semantics_version"],
            "effective_valid_token_policy": checkpoint_profile["effective_valid_token_policy"],
        }
        mismatches = {
            key: {"checkpoint": contract.get(key), "registry": value}
            for key, value in expected.items() if contract.get(key) != value
        }
        if mismatches:
            raise ValueError(f"checkpoint deployment profile contract mismatch: {mismatches}")
    else:
        if checkpoint_payload.get("checkpoint_profile_contract_version") is not None:
            raise ValueError("new checkpoint contract is missing deployment_profiles")
        training = checkpoint_payload.get("training_profile")
        if not isinstance(training, Mapping):
            raise ValueError("checkpoint has neither deployment_profiles nor a legacy training_profile")
        legacy_id = training.get("profile_id", training.get("profile_name"))
        if legacy_id not in LEGACY_DEPLOYMENT_PROFILE_IDS:
            raise ValueError(f"unknown legacy training profile: {legacy_id!r}")
        checkpoint_profile_id = LEGACY_DEPLOYMENT_PROFILE_IDS[str(legacy_id)]
        checkpoint_profile = get_channel_profile(checkpoint_profile_id)
        inferred = True
        inference_warning = (
            f"deployment profile inferred from legacy training profile {legacy_id}; "
            "regenerate the checkpoint with an explicit deployment_profiles contract"
        )
        warnings.warn(inference_warning, RuntimeWarning, stacklevel=2)

    requested = str(requested_profile)
    if requested == "auto":
        effective_profile_id = checkpoint_profile_id
        override = False
    else:
        get_channel_profile(requested)
        if requested != checkpoint_profile_id and not allow_profile_override:
            raise ValueError("requested profile does not match checkpoint deployment contract")
        effective_profile_id = requested
        override = requested != checkpoint_profile_id
    effective = get_channel_profile(effective_profile_id)
    return {
        **effective,
        "target_dataset": target,
        "checkpoint_profile_id": checkpoint_profile_id,
        "requested_profile": requested,
        "effective_profile_id": effective_profile_id,
        "profile_override": override,
        "not_primary_experiment": override,
        "deployment_profile_inferred": inferred,
        "deployment_profile_warning": inference_warning,
    }


def apply_input_channel_profile(
    token_inputs: np.ndarray,
    token_valid_mask: Sequence[bool],
    channel_mask: Sequence[bool] | None,
    bad_channel_mask: Sequence[bool] | None,
    profile: Mapping[str, Any],
    *,
    targets: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    num_channels = len(STANDARD_64_CHANNELS)
    num_tokens = num_channels * NUM_TIME_PATCHES
    inputs = np.asarray(token_inputs, dtype=np.float32).reshape(num_tokens, PATCH_LEN).copy()
    source_token_valid = np.asarray(token_valid_mask, dtype=bool).reshape(num_tokens)
    profile_allowed = np.asarray(
        [name in set(profile["allowed_channels"]) for name in STANDARD_64_CHANNELS],
        dtype=bool,
    )
    if channel_mask is None:
        present = source_token_valid.reshape(num_channels, NUM_TIME_PATCHES).any(axis=1)
    else:
        present = np.asarray(channel_mask, dtype=bool).reshape(num_channels)
    bad = (
        np.zeros(num_channels, dtype=bool)
        if bad_channel_mask is None else np.asarray(bad_channel_mask, dtype=bool).reshape(num_channels)
    )
    final_channel_valid = present & ~bad & profile_allowed
    final_token_valid = source_token_valid & np.repeat(final_channel_valid, NUM_TIME_PATCHES)
    inputs[~final_token_valid] = 0.0
    result = {
        "token_inputs": inputs,
        "channel_valid_mask": final_channel_valid.astype(np.float32),
        "token_valid_mask": final_token_valid.astype(np.float32),
    }
    if targets is not None:
        target_array = np.asarray(targets, dtype=np.float32).copy()
        if target_array.shape[0] != num_tokens:
            target_array = target_array.reshape(num_tokens, -1)
        target_array[~final_token_valid] = 0.0
        result["targets"] = target_array
    return result


def apply_training_profile(
    token_inputs: np.ndarray,
    token_valid_mask: Sequence[bool],
    dataset_id: str,
    profile: Mapping[str, Any],
    *,
    targets: np.ndarray | None = None,
    channel_valid_mask: Sequence[bool] | None = None,
) -> dict[str, np.ndarray]:
    if str(dataset_id).casefold() != str(profile.get("dataset_id", "")).casefold():
        raise ValueError("dataset_id does not match resolved training profile")
    channel_profile = get_channel_profile(profile["channel_profile_id"])
    return apply_input_channel_profile(
        token_inputs, token_valid_mask, channel_valid_mask, None, channel_profile,
        targets=targets,
    )


class ProfiledTokenDataset(Dataset):
    """Read-time view applying one resolved profile to an existing token dataset."""

    def __init__(self, dataset: Dataset, dataset_id: str, profile: Mapping[str, Any]):
        self.dataset = dataset
        self.dataset_id = str(dataset_id).casefold()
        self.profile = dict(profile)

    def __len__(self) -> int:
        return len(self.dataset)

    @property
    def metadata(self):
        return self.dataset.metadata

    def _apply(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        output = dict(sample)
        if "channel_profile_id" in self.profile:
            channel_profile = get_channel_profile(self.profile["channel_profile_id"])
        else:
            channel_profile = self.profile
        channel_mask_value = sample.get("channel_mask", sample.get("channel_valid_mask"))
        bad_mask_value = sample.get("bad_channel_mask")
        transformed = apply_input_channel_profile(
            sample["token_inputs"].detach().cpu().numpy(),
            sample["token_valid_mask"].detach().cpu().numpy(),
            channel_mask_value.detach().cpu().numpy() if channel_mask_value is not None else None,
            bad_mask_value.detach().cpu().numpy() if bad_mask_value is not None else None,
            channel_profile,
            targets=sample.get("targets").detach().cpu().numpy() if "targets" in sample else None,
        )
        for key, value in transformed.items():
            output[key] = torch.from_numpy(value)
        return output

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._apply(self.dataset[index])

    def __getitems__(self, indices: Sequence[int]) -> list[dict[str, Any]]:
        samples = (
            self.dataset.__getitems__(indices) if hasattr(self.dataset, "__getitems__")
            else [self.dataset[index] for index in indices]
        )
        return [self._apply(sample) for sample in samples]

    def get_sampling_rows(self) -> list[dict[str, Any]]:
        return self.dataset.get_sampling_rows()

    def get_subject_label_rows(self, *args, **kwargs):
        return self.dataset.get_subject_label_rows(*args, **kwargs)
