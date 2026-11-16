from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
DOWNSTREAM = SCRIPTS / "downstream_50m"
for path in (SCRIPTS, DOWNSTREAM):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import dortmund_huada_zero_shot as zero_core  # noqa: E402
import run_huada_dortmund_age_probe_transfer_cv7 as transfer  # noqa: E402
from fbmae_training_profiles import (  # noqa: E402
    NATIVE64_PROFILE_ID,
    SPARSE18_PROFILE_ID,
    ProfiledTokenDataset,
    apply_input_channel_profile,
    build_checkpoint_profile_contract,
    get_channel_profile,
    load_training_profiles,
    resolve_checkpoint_deployment_profile,
)


CATALOG = ROOT / "configs" / "mipdb_dortmund_fbmae_training_profiles_v1.json"


def contract(profile_id: str) -> dict:
    return build_checkpoint_profile_contract(load_training_profiles(CATALOG), profile_id)


def arrays():
    return np.ones((640, 100), np.float32), np.ones(640, bool), np.ones(64, bool)


def test_native64_and_sparse18_keep_checkpoint_geometry_and_expected_counts():
    inputs, tokens, channels = arrays()
    native = apply_input_channel_profile(
        inputs, tokens, channels, np.zeros(64, bool), get_channel_profile(NATIVE64_PROFILE_ID)
    )
    sparse = apply_input_channel_profile(
        inputs, tokens, channels, np.zeros(64, bool), get_channel_profile(SPARSE18_PROFILE_ID)
    )
    assert native["token_inputs"].shape == sparse["token_inputs"].shape == (640, 100)
    assert native["channel_valid_mask"].sum() == 64
    assert native["token_valid_mask"].sum() == 640
    assert sparse["channel_valid_mask"].sum() == 18
    assert sparse["token_valid_mask"].sum() == 180
    assert np.all(sparse["token_inputs"][sparse["token_valid_mask"] == 0] == 0)


def test_original_bad_channel_stays_invalid_under_native_and_sparse_profiles():
    inputs, tokens, channels = arrays()
    bad = np.zeros(64, bool)
    bad[0] = True  # Fp1 is one of sparse18, so both profiles must preserve this bad channel.
    native = apply_input_channel_profile(
        inputs, tokens, channels, bad, get_channel_profile(NATIVE64_PROFILE_ID)
    )
    sparse = apply_input_channel_profile(
        inputs, tokens, channels, bad, get_channel_profile(SPARSE18_PROFILE_ID)
    )
    assert native["token_valid_mask"][:10].sum() == 0
    assert sparse["token_valid_mask"][:10].sum() == 0
    assert native["token_valid_mask"].sum() == 630
    assert sparse["token_valid_mask"].sum() == 170
    assert np.all(native["token_inputs"][:10] == 0)
    assert np.all(sparse["token_inputs"][:10] == 0)


@pytest.mark.parametrize(
    "training_profile,expected_profile,expected_tokens",
    [
        ("dortmund_native64", NATIVE64_PROFILE_ID, 640),
        ("dortmund_matched18", SPARSE18_PROFILE_ID, 180),
        ("joint_matched18", SPARSE18_PROFILE_ID, 180),
        ("joint_native64_vs_sparse18", NATIVE64_PROFILE_ID, 640),
    ],
)
def test_checkpoint_contract_resolves_frozen_huada_mapping(
    training_profile, expected_profile, expected_tokens,
):
    resolved = resolve_checkpoint_deployment_profile(
        contract(training_profile), "huada", "auto"
    )
    assert resolved["effective_profile_id"] == expected_profile
    assert resolved["num_allowed_tokens"] == expected_tokens
    assert not resolved["deployment_profile_inferred"]


def test_catalog_accepts_future_explicit_profile_without_downstream_branch(tmp_path):
    payload = json.loads(CATALOG.read_text(encoding="utf-8"))
    payload["profiles"]["future_joint_control"] = {
        "role": "future_control",
        "datasets": {
            "dortmund": NATIVE64_PROFILE_ID,
            "mipdb": SPARSE18_PROFILE_ID,
        },
        "deployment_profiles": {"huada": NATIVE64_PROFILE_ID},
        "dataset_filters": {
            "dortmund": {"min_age": 20, "max_age": 35},
            "mipdb": {"min_age": 20, "max_age": 35},
        },
    }
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_training_profiles(path)
    resolved = resolve_checkpoint_deployment_profile(
        build_checkpoint_profile_contract(loaded, "future_joint_control"), "huada"
    )
    assert resolved["effective_profile_id"] == NATIVE64_PROFILE_ID


def test_legacy_exact_profile_inference_warns_and_unknown_or_missing_fails_closed():
    with pytest.warns(RuntimeWarning, match="inferred"):
        resolved = resolve_checkpoint_deployment_profile(
            {"training_profile": {"profile_name": "joint_matched18"}}, "huada"
        )
    assert resolved["effective_profile_id"] == SPARSE18_PROFILE_ID
    assert resolved["deployment_profile_inferred"]
    with pytest.raises(ValueError, match="unknown legacy"):
        resolve_checkpoint_deployment_profile(
            {"training_profile": {"profile_id": "matched-ish"}}, "huada"
        )
    with pytest.raises(ValueError, match="neither deployment_profiles"):
        resolve_checkpoint_deployment_profile({}, "huada")


def test_deployment_sha_tamper_and_unapproved_override_fail_closed():
    payload = contract("dortmund_native64")
    tampered = copy.deepcopy(payload)
    tampered["deployment_profiles"]["huada"]["channel_profile_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="contract mismatch"):
        resolve_checkpoint_deployment_profile(tampered, "huada")
    wrong_version = copy.deepcopy(payload)
    wrong_version["checkpoint_profile_contract_version"] = "future-unknown-version"
    with pytest.raises(ValueError, match="contract version mismatch"):
        resolve_checkpoint_deployment_profile(wrong_version, "huada")
    with pytest.raises(ValueError, match="does not match"):
        resolve_checkpoint_deployment_profile(
            payload, "huada", SPARSE18_PROFILE_ID, allow_profile_override=False
        )
    diagnostic = resolve_checkpoint_deployment_profile(
        payload, "huada", SPARSE18_PROFILE_ID, allow_profile_override=True
    )
    assert diagnostic["profile_override"] and diagnostic["not_primary_experiment"]


def checkpoint_report(profile_id: str) -> dict:
    deployment = resolve_checkpoint_deployment_profile(
        contract("dortmund_native64"), "huada", profile_id,
        allow_profile_override=profile_id != NATIVE64_PROFILE_ID,
    )
    return {
        "sha256": "a" * 64,
        "electrode_vocabulary_sha256": "vocab",
        "token_mask_semantics_version": zero_core.MASK_SEMANTICS_VERSION,
        "best_epoch": 1,
        "deployment_profile": deployment,
    }


def test_feature_cache_key_binds_effective_profile_and_uses_shared_resolver():
    info = {"row_count": 2, "sample_id_fingerprint": "samples"}
    _, native_key = transfer.build_dortmund_cache_identity(
        input_fingerprint="input", input_info=info,
        checkpoint_report=checkpoint_report(NATIVE64_PROFILE_ID),
        code_sha256="code", git_commit="git",
    )
    sparse_report = checkpoint_report(SPARSE18_PROFILE_ID)
    sparse_identity, sparse_key = transfer.build_dortmund_cache_identity(
        input_fingerprint="input", input_info=info, checkpoint_report=sparse_report,
        code_sha256="code", git_commit="git",
    )
    assert native_key != sparse_key
    assert sparse_identity["target_dataset"] == "huada"
    assert sparse_identity["input_profile_id"] == SPARSE18_PROFILE_ID
    assert sparse_identity["input_profile_sha256"] == sparse_report["deployment_profile"]["canonical_sha256"]
    assert zero_core.resolve_checkpoint_deployment_profile is transfer.resolve_checkpoint_deployment_profile


class TinyTokenDataset(Dataset):
    def __init__(self):
        self.metadata = [{"sample_id": "x"}]
        self.sample = {
            "sample_id": "x", "subject_id": "s", "recording_id": "r",
            "token_inputs": torch.ones(640, 100),
            "token_valid_mask": torch.ones(640),
            "channel_valid_mask": torch.ones(64),
        }

    def __len__(self):
        return 1

    def __getitem__(self, index):
        assert index == 0
        return self.sample


def test_online_and_cache_views_apply_identical_profile_before_forward():
    profile = resolve_checkpoint_deployment_profile(
        contract("joint_matched18"), "huada"
    )
    online = ProfiledTokenDataset(TinyTokenDataset(), "huada", profile)[0]
    cache_view = ProfiledTokenDataset(TinyTokenDataset(), "huada", profile)[0]
    assert torch.equal(online["token_inputs"], cache_view["token_inputs"])
    assert torch.equal(online["token_valid_mask"], cache_view["token_valid_mask"])
    embeddings = torch.arange(640, dtype=torch.float32).view(1, 640, 1)
    mask = online["token_valid_mask"].view(1, 640, 1)
    online_mean = (embeddings * mask).sum(1) / mask.sum(1)
    cache_mask = cache_view["token_valid_mask"].view(1, 640, 1)
    cache_mean = (embeddings * cache_mask).sum(1) / cache_mask.sum(1)
    assert torch.equal(online_mean, cache_mean)
