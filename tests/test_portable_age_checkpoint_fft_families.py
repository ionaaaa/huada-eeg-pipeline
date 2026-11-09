from __future__ import annotations

import copy
import importlib
from pathlib import Path
import sys

import pytest
import torch

from scripts.checkpoints import portable_age_checkpoint as huada


SOURCE_MODULE = (
    Path(__file__).resolve().parents[2]
    / "neuroai-repro"
    / "neuralbench-repo"
    / "neuralbench"
    / "checkpoints"
    / "portable_age_export.py"
)


def _source_module():
    monorepo_root = SOURCE_MODULE.parents[3]
    for package_root in (
        "neuralbench-repo", "neuraltrain-repo", "neuralset-repo", "neuralfetch-repo"
    ):
        source_root = str(monorepo_root / package_root)
        if source_root not in sys.path:
            sys.path.insert(0, source_root)
    return importlib.import_module("neuralbench.checkpoints.portable_age_export")


@pytest.fixture(scope="module")
def exporter():
    return _source_module()


def _rehash(payload: dict) -> None:
    payload["integrity"] = {
        "backbone_state_fingerprint": huada.tensor_state_fingerprint(payload["backbone_state_dict"]),
        "age_probe_state_fingerprint": huada.tensor_state_fingerprint(payload["age_probe_state_dict"]),
        "model_config_sha256": huada.metadata_sha256(payload["model"]["model_config"]),
        "input_contract_sha256": huada.metadata_sha256(payload["input_contract"]),
        "representation_contract_sha256": huada.metadata_sha256(payload["representation_contract"]),
        "target_transform_sha256": huada.metadata_sha256(payload["target_transform"]),
    }


@pytest.mark.parametrize("family,dim", [("labram", 200), ("reve", 512)])
def test_neuralbench_fixture_round_trips_through_huada(exporter, tmp_path, family, dim):
    payload, _ = exporter.build_fixture_payload(family)
    validated = huada.validate_portable_age_checkpoint(payload, allow_fixture=True)
    assert validated.age_probe_state_dict["weight"].shape == (1, dim)
    path = tmp_path / f"{family}.pt"
    saved = huada.save_portable_age_checkpoint(payload, path, allow_fixture=True)
    loaded = huada.load_portable_age_checkpoint(
        path,
        expected_file_sha256=saved.file_sha256,
        expected_model_family=family,
        allow_fixture=True,
    )
    assert loaded.model_family == family
    raw = torch.load(path, map_location="cpu", weights_only=True)
    assert raw["validation_mode"] == "fixture"


def test_cross_repo_schema_drift_sentinels_match(exporter):
    assert exporter.SCHEMA_SPEC_SHA256 == huada.SCHEMA_SPEC_SHA256
    assert exporter.CANONICAL_FIXTURE_FINGERPRINT == huada.CANONICAL_FIXTURE_FINGERPRINT


@pytest.mark.parametrize("family", ["labram", "reve"])
def test_feature_dim_nonfinite_contract_and_split_fail_closed(exporter, family):
    base, _ = exporter.build_fixture_payload(family)
    mutations = []
    value = copy.deepcopy(base)
    value["representation_contract"]["feature_dim"] += 1
    value["age_probe_state_dict"]["weight"] = torch.zeros(
        1, value["representation_contract"]["feature_dim"]
    )
    _rehash(value)
    mutations.append((value, "feature_dim"))
    value = copy.deepcopy(base)
    first = next(t for t in value["backbone_state_dict"].values() if t.is_floating_point())
    first.reshape(-1)[0] = float("nan")
    mutations.append((value, "NaN or Inf"))
    value = copy.deepcopy(base)
    value["input_contract"]["status"] = "provisional"
    _rehash(value)
    mutations.append((value, "status"))
    value = copy.deepcopy(base)
    value["input_contract"]["source_contract_sha256"] = "0" * 64
    _rehash(value)
    mutations.append((value, "source_contract_sha256"))
    value = copy.deepcopy(base)
    value["source_training"]["split_manifest_sha256"] = "0" * 64
    mutations.append((value, "split_manifest_sha256"))
    for payload, match in mutations:
        with pytest.raises(huada.PortableCheckpointError, match=match):
            huada.validate_portable_age_checkpoint(payload, allow_fixture=True)


def test_labram_duplicate_scaling_and_reve_labram_scaling_fail_closed(exporter):
    labram, _ = exporter.build_fixture_payload("labram")
    labram["input_contract"]["wrapper_scale"] = 10000.0
    _rehash(labram)
    with pytest.raises(huada.PortableCheckpointError, match="wrapper_scale"):
        huada.validate_portable_age_checkpoint(labram, allow_fixture=True)

    reve, _ = exporter.build_fixture_payload("reve")
    reve["input_contract"]["extractor_scale"] = 10000.0
    _rehash(reve)
    with pytest.raises(huada.PortableCheckpointError, match="extractor_scale|x10000"):
        huada.validate_portable_age_checkpoint(reve, allow_fixture=True)


@pytest.mark.parametrize("family", ["labram", "reve"])
def test_scaler_truncation_rejected(exporter, family):
    payload, _ = exporter.build_fixture_payload(family)
    payload["target_transform"]["std"] = 4.218591
    _rehash(payload)
    with pytest.raises(huada.PortableCheckpointError, match="audited precision"):
        huada.validate_portable_age_checkpoint(payload, allow_fixture=True)


def test_reve_final_layer_is_retained_and_declared_outside_age_path(exporter):
    payload, _ = exporter.build_fixture_payload("reve")
    assert any(key.startswith("final_layer.1.") for key in payload["backbone_state_dict"])
    assert any(key.startswith("final_layer.2.") for key in payload["backbone_state_dict"])
    modules = payload["model"]["model_config"]["source_only_frozen_modules"]
    assert {item["module"] for item in modules} == {"final_layer.1", "final_layer.2"}
    assert all(item["in_age_forward_path"] is False for item in modules)
