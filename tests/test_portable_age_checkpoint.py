from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from scripts.checkpoints import portable_age_checkpoint as portable
from scripts.checkpoints.portable_age_checkpoint_demo import build_demo_fixture


def fixture_payload() -> dict:
    return copy.deepcopy(build_demo_fixture())


def rehash(payload: dict) -> None:
    payload["integrity"].update({
        "backbone_state_fingerprint": portable.tensor_state_fingerprint(
            payload["backbone_state_dict"]
        ),
        "age_probe_state_fingerprint": portable.tensor_state_fingerprint(
            payload["age_probe_state_dict"]
        ),
        "model_config_sha256": portable.metadata_sha256(payload["model"]["model_config"]),
        "input_contract_sha256": portable.metadata_sha256(payload["input_contract"]),
        "representation_contract_sha256": portable.metadata_sha256(
            payload["representation_contract"]
        ),
        "target_transform_sha256": portable.metadata_sha256(payload["target_transform"]),
    })


def save_fixture(tmp_path: Path):
    path = tmp_path / "fixture.pt"
    loaded = portable.save_portable_age_checkpoint(
        fixture_payload(), path, allow_fixture=True
    )
    return path, loaded


def test_valid_fixture_round_trip_and_read_only_view(tmp_path):
    path, saved = save_fixture(tmp_path)
    loaded = portable.load_portable_age_checkpoint(
        path,
        expected_file_sha256=saved.file_sha256,
        expected_model_name="fbmae_50m",
        expected_model_family="fbmae",
        expected_source_checkpoint_sha256="5" * 64,
        allow_fixture=True,
    )
    assert loaded.path == path.resolve()
    assert loaded.payload["target_transform"]["std"] == 4.218591189948011
    assert loaded.age_probe_state_dict["weight"].shape == (1, 512)
    with pytest.raises(TypeError):
        loaded.payload["schema_name"] = "changed"


def test_loader_uses_weights_only_cpu(monkeypatch, tmp_path):
    path, _ = save_fixture(tmp_path)
    original = portable.torch.load
    calls = []

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(portable.torch, "load", spy)
    portable.load_portable_age_checkpoint(path, allow_fixture=True)
    assert calls == [{"map_location": "cpu", "weights_only": True}]


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("schema_name", "wrong", "schema_name"),
        ("schema_version", 2, "schema_version"),
        ("checkpoint_kind", "wrong", "checkpoint_kind"),
    ],
)
def test_schema_identity_fail_closed(field, value, match):
    payload = fixture_payload()
    payload[field] = value
    with pytest.raises(portable.PortableCheckpointError, match=match):
        portable.validate_portable_age_checkpoint(payload, allow_fixture=True)


def test_loader_does_not_claim_support_for_future_schema_version(tmp_path):
    path, _ = save_fixture(tmp_path)
    with pytest.raises(portable.PortableCheckpointError, match="implements schema_version 1 only"):
        portable.load_portable_age_checkpoint(
            path, expected_schema_version=2, allow_fixture=True
        )


def test_missing_required_and_unknown_top_level_fail_closed():
    payload = fixture_payload()
    payload.pop("task")
    with pytest.raises(portable.PortableCheckpointError, match="missing"):
        portable.validate_portable_age_checkpoint(payload, allow_fixture=True)
    payload = fixture_payload()
    payload["optimizer_state_dict"] = {"unsafe": object()}
    with pytest.raises(portable.PortableCheckpointError, match="unexpected"):
        portable.validate_portable_age_checkpoint(payload, allow_fixture=True)


def test_metadata_non_json_type_and_type_subclass_fail_closed():
    class StringSubclass(str):
        pass

    for value in (Path("bad"), StringSubclass("bad")):
        payload = fixture_payload()
        payload["provenance"]["exporter_name"] = value
        with pytest.raises(portable.PortableCheckpointError, match="forbidden non-JSON"):
            portable.validate_portable_age_checkpoint(payload, allow_fixture=True)


def test_metadata_non_string_mapping_key_is_not_silently_coerced():
    payload = fixture_payload()
    payload["model"]["model_config"][1] = "bad"
    with pytest.raises(portable.PortableCheckpointError, match="non-string mapping key"):
        portable.validate_portable_age_checkpoint(payload, allow_fixture=True)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_metadata_non_finite_float_fail_closed(value):
    payload = fixture_payload()
    payload["source_training"]["validation_metrics"] = {"mae": value}
    with pytest.raises(portable.PortableCheckpointError, match="non-finite"):
        portable.validate_portable_age_checkpoint(payload, allow_fixture=True)


def test_empty_backbone_and_non_tensor_value_fail_closed():
    payload = fixture_payload()
    payload["backbone_state_dict"] = {}
    with pytest.raises(portable.PortableCheckpointError, match="must not be empty"):
        portable.validate_portable_age_checkpoint(payload, allow_fixture=True)
    payload = fixture_payload()
    payload["backbone_state_dict"]["tokenizer.bad"] = [1.0]
    with pytest.raises(portable.PortableCheckpointError, match="must be a tensor"):
        portable.validate_portable_age_checkpoint(payload, allow_fixture=True)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_backbone_non_finite_fail_closed(value):
    payload = fixture_payload()
    payload["backbone_state_dict"]["tokenizer.proj.0.weight"][0, 0] = value
    with pytest.raises(portable.PortableCheckpointError, match="NaN or Inf"):
        portable.validate_portable_age_checkpoint(payload, allow_fixture=True)


@pytest.mark.parametrize(
    "prefix", ["module.", "model.", "model.wrapped_model.", "_lightning_module."]
)
def test_training_wrapper_prefixes_are_rejected(prefix):
    payload = fixture_payload()
    tensor = payload["backbone_state_dict"].pop("tokenizer.proj.0.weight")
    payload["backbone_state_dict"][prefix + "tokenizer.proj.0.weight"] = tensor
    with pytest.raises(portable.PortableCheckpointError, match="wrapper prefix"):
        portable.validate_portable_age_checkpoint(payload, allow_fixture=True)


@pytest.mark.parametrize("missing", ["weight", "bias"])
def test_probe_missing_key_fail_closed(missing):
    payload = fixture_payload()
    payload["age_probe_state_dict"].pop(missing)
    with pytest.raises(portable.PortableCheckpointError, match="keys mismatch"):
        portable.validate_portable_age_checkpoint(payload, allow_fixture=True)


def test_probe_extra_key_fail_closed():
    payload = fixture_payload()
    payload["age_probe_state_dict"]["extra"] = torch.zeros(1)
    with pytest.raises(portable.PortableCheckpointError, match="unexpected"):
        portable.validate_portable_age_checkpoint(payload, allow_fixture=True)


@pytest.mark.parametrize(
    "weight,bias,match",
    [
        (torch.zeros(512), torch.zeros(1), "weight must have shape"),
        (torch.zeros(1, 511), torch.zeros(1), "weight must have shape"),
        (torch.zeros(1, 512), torch.zeros(2), "bias must have shape"),
    ],
)
def test_probe_shape_and_feature_dim_fail_closed(weight, bias, match):
    payload = fixture_payload()
    payload["age_probe_state_dict"] = {"weight": weight, "bias": bias}
    with pytest.raises(portable.PortableCheckpointError, match=match):
        portable.validate_portable_age_checkpoint(payload, allow_fixture=True)


@pytest.mark.parametrize("key,index", [("weight", (0, 0)), ("bias", (0,))])
def test_probe_non_finite_fail_closed(key, index):
    payload = fixture_payload()
    payload["age_probe_state_dict"][key][index] = float("nan")
    with pytest.raises(portable.PortableCheckpointError, match="NaN or Inf"):
        portable.validate_portable_age_checkpoint(payload, allow_fixture=True)


@pytest.mark.parametrize("std", [0.0, -1.0])
def test_scaler_nonpositive_std_fail_closed(std):
    payload = fixture_payload()
    payload["target_transform"]["std"] = std
    with pytest.raises(portable.PortableCheckpointError, match="std must be positive"):
        portable.validate_portable_age_checkpoint(payload, allow_fixture=True)


@pytest.mark.parametrize("field", ["mean", "std"])
def test_scaler_nonfinite_fail_closed(field):
    payload = fixture_payload()
    payload["target_transform"][field] = float("nan")
    with pytest.raises(portable.PortableCheckpointError, match="non-finite|must be finite"):
        portable.validate_portable_age_checkpoint(payload, allow_fixture=True)


def test_target_unit_fail_closed():
    payload = fixture_payload()
    payload["target_transform"]["target_unit"] = "month"
    with pytest.raises(portable.PortableCheckpointError, match="target_unit"):
        portable.validate_portable_age_checkpoint(payload, allow_fixture=True)


@pytest.mark.parametrize(
    "field",
    ["backbone_state_fingerprint", "age_probe_state_fingerprint"],
)
def test_tensor_fingerprint_mismatch_fail_closed(field):
    payload = fixture_payload()
    payload["integrity"][field] = "f" * 64
    with pytest.raises(portable.PortableCheckpointError, match=field):
        portable.validate_portable_age_checkpoint(payload, allow_fixture=True)


@pytest.mark.parametrize(
    "field",
    [
        "model_config_sha256", "input_contract_sha256",
        "representation_contract_sha256", "target_transform_sha256",
    ],
)
def test_contract_hash_mismatch_fail_closed(field):
    payload = fixture_payload()
    payload["integrity"][field] = "f" * 64
    with pytest.raises(portable.PortableCheckpointError, match=field):
        portable.validate_portable_age_checkpoint(payload, allow_fixture=True)


def test_source_sha_format_and_selection_rules_fail_closed():
    payload = fixture_payload()
    payload["source_checkpoint"]["source_checkpoint_sha256"] = "not-a-sha"
    with pytest.raises(portable.PortableCheckpointError, match="source_checkpoint_sha256"):
        portable.validate_portable_age_checkpoint(payload, allow_fixture=True)
    payload = fixture_payload()
    payload["source_training"]["selection_metric"] = "test_mae"
    with pytest.raises(portable.PortableCheckpointError, match="test metrics"):
        portable.validate_portable_age_checkpoint(payload, allow_fixture=True)
    payload = fixture_payload()
    payload["source_training"]["train_subject_count"] = 0
    with pytest.raises(portable.PortableCheckpointError, match=">= 1"):
        portable.validate_portable_age_checkpoint(payload, allow_fixture=True)


def test_expected_file_sha_mismatch_precedes_deserialization(monkeypatch, tmp_path):
    path, _ = save_fixture(tmp_path)
    monkeypatch.setattr(
        portable.torch, "load", lambda *a, **k: pytest.fail("deserialization must not run")
    )
    with pytest.raises(portable.PortableCheckpointError, match="file SHA256 mismatch"):
        portable.load_portable_age_checkpoint(
            path, expected_file_sha256="0" * 64, allow_fixture=True
        )


def test_expected_model_and_source_mismatches(tmp_path):
    path, _ = save_fixture(tmp_path)
    for kwargs, match in (
        ({"expected_model_name": "other"}, "model_name mismatch"),
        ({"expected_model_family": "other"}, "model_family mismatch"),
        ({"expected_source_checkpoint_sha256": "6" * 64}, "source checkpoint SHA256"),
    ):
        with pytest.raises(portable.PortableCheckpointError, match=match):
            portable.load_portable_age_checkpoint(
                path, allow_fixture=True, **kwargs
            )


def test_unknown_model_family_rejected():
    payload = fixture_payload()
    payload["model"]["model_family"] = "unknown-family"
    with pytest.raises(portable.PortableCheckpointError, match="no portable model validator"):
        portable.validate_portable_age_checkpoint(payload, allow_fixture=True)


def test_fbmae_family_validator_passes_and_rejects_contract_mismatch():
    portable.validate_portable_age_checkpoint(fixture_payload(), allow_fixture=True)
    payload = fixture_payload()
    payload["representation_contract"]["pooling"] = "mean"
    rehash(payload)
    with pytest.raises(portable.PortableCheckpointError, match="masked_mean"):
        portable.validate_portable_age_checkpoint(payload, allow_fixture=True)
    payload = fixture_payload()
    payload["model"]["model_config"]["depth"] = 11
    rehash(payload)
    with pytest.raises(portable.PortableCheckpointError, match="model_config mismatch"):
        portable.validate_portable_age_checkpoint(payload, allow_fixture=True)


@pytest.mark.parametrize("manifest_field", ["portable_file_size_bytes", "portable_file_sha256"])
def test_sidecar_size_and_sha_mismatch_fail_closed(tmp_path, manifest_field):
    path, _ = save_fixture(tmp_path)
    manifest_path = Path(str(path) + portable.MANIFEST_SUFFIX)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[manifest_field] = 0 if manifest_field.endswith("bytes") else "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(portable.PortableCheckpointError, match="manifest mismatch"):
        portable.load_portable_age_checkpoint(path, allow_fixture=True)


def test_default_rejects_overwrite_and_missing_directory(tmp_path):
    path, _ = save_fixture(tmp_path)
    with pytest.raises(FileExistsError):
        portable.save_portable_age_checkpoint(
            fixture_payload(), path, allow_fixture=True
        )
    with pytest.raises(FileNotFoundError, match="Target directory"):
        portable.save_portable_age_checkpoint(
            fixture_payload(), tmp_path / "missing" / "x.pt", allow_fixture=True
        )


def test_atomic_publish_reloads_and_leaves_no_temporary_files(tmp_path):
    path, result = save_fixture(tmp_path)
    assert path.is_file()
    assert Path(str(path) + portable.MANIFEST_SUFFIX).is_file()
    assert result.file_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert not [item for item in tmp_path.iterdir() if item.name.endswith(".tmp")]
    portable.load_portable_age_checkpoint(path, allow_fixture=True)


def test_save_failure_does_not_publish_destination(monkeypatch, tmp_path):
    destination = tmp_path / "never-valid.pt"
    monkeypatch.setattr(
        portable, "load_portable_age_checkpoint",
        lambda *a, **k: (_ for _ in ()).throw(portable.PortableCheckpointError("injected")),
    )
    with pytest.raises(portable.PortableCheckpointError, match="injected"):
        portable.save_portable_age_checkpoint(
            fixture_payload(), destination, allow_fixture=True
        )
    assert not destination.exists()
    assert not Path(str(destination) + portable.MANIFEST_SUFFIX).exists()


def test_fixture_is_rejected_by_default_and_formal_mode_checks_completeness(tmp_path):
    path, _ = save_fixture(tmp_path)
    with pytest.raises(portable.PortableCheckpointError, match="fixture checkpoint"):
        portable.load_portable_age_checkpoint(path)
    payload = fixture_payload()
    payload["validation_mode"] = "formal"
    with pytest.raises(portable.PortableCheckpointError, match="complete audited state"):
        portable.validate_portable_age_checkpoint(payload)


def test_formal_provenance_requires_git_and_critical_hashes():
    for field in ("source_git_commit", "critical_code_sha256"):
        payload = fixture_payload()
        payload["validation_mode"] = "formal"
        payload["provenance"][field] = "" if field == "source_git_commit" else {}
        with pytest.raises(portable.PortableCheckpointError, match=field):
            portable.validate_portable_age_checkpoint(payload)


def test_fixture_mode_may_omit_formal_only_provenance():
    payload = fixture_payload()
    payload["provenance"].pop("source_git_commit")
    payload["provenance"].pop("critical_code_sha256")
    payload["source_checkpoint"].pop("source_checkpoint_sha256")
    portable.validate_portable_age_checkpoint(payload, allow_fixture=True)


def test_formal_mode_requires_source_checkpoint_sha256():
    payload = fixture_payload()
    payload["validation_mode"] = "formal"
    payload["source_checkpoint"].pop("source_checkpoint_sha256")
    with pytest.raises(portable.PortableCheckpointError, match="source_checkpoint_sha256"):
        portable.validate_portable_age_checkpoint(payload)


def test_loader_source_has_no_training_framework_or_model_construction_imports():
    source_path = Path(portable.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported = []
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
        elif isinstance(node, ast.Call):
            calls.append(ast.unparse(node.func))
    assert not any("lightning" in name.lower() for name in imported)
    assert not any(name.endswith(".model") for name in imported)
    assert not any(call.startswith("torch.nn") for call in calls)
    assert not any("Optimizer" in call or "Scheduler" in call for call in calls)


def test_common_schema_can_host_future_labram_and_reve_contracts():
    for family in ("labram_test", "reve_test"):
        portable.register_portable_model_validator(family, lambda payload, mode: None)
        payload = fixture_payload()
        payload["model"]["model_name"] = family + "_model"
        payload["model"]["model_family"] = family
        payload["model"]["model_config"] = {"architecture": family, "width": 16}
        payload["input_contract"]["input_kind"] = family + "_native_input"
        payload["representation_contract"].update({
            "representation_name": family + "_representation",
            "representation_layer": "family-defined",
            "pooling": "family-defined",
            "uses_token_valid_mask": False,
        })
        payload["backbone_state_dict"] = {"canonical.weight": torch.ones(1)}
        rehash(payload)
        result = portable.validate_portable_age_checkpoint(payload, allow_fixture=True)
        assert result.model_family == family


def test_tensor_fingerprint_stable_across_order_and_does_not_mutate():
    first_tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3).t()
    original_stride = first_tensor.stride()
    a = {"b": torch.ones(1), "a": first_tensor}
    b = {"a": first_tensor.clone(), "b": torch.ones(1)}
    assert portable.tensor_state_fingerprint(a) == portable.tensor_state_fingerprint(b)
    assert first_tensor.stride() == original_stride


def test_metadata_hash_is_order_independent_and_nan_forbidden():
    assert portable.metadata_sha256({"b": 2, "a": 1}) == portable.metadata_sha256(
        {"a": 1, "b": 2}
    )
    with pytest.raises(ValueError):
        portable.metadata_sha256({"bad": float("nan")})


def test_manifest_required_by_default_but_can_be_explicitly_disabled(tmp_path):
    path, _ = save_fixture(tmp_path)
    Path(str(path) + portable.MANIFEST_SUFFIX).unlink()
    with pytest.raises(portable.PortableCheckpointError, match="manifest is required"):
        portable.load_portable_age_checkpoint(path, allow_fixture=True)
    portable.load_portable_age_checkpoint(
        path, allow_fixture=True, require_manifest=False
    )


def test_non_cpu_map_location_rejected_before_load(tmp_path):
    path, _ = save_fixture(tmp_path)
    with pytest.raises(portable.PortableCheckpointError, match="load on CPU"):
        portable.load_portable_age_checkpoint(
            path, map_location="cuda", allow_fixture=True
        )


def test_symlink_checkpoint_is_not_treated_as_regular_file(tmp_path):
    path, _ = save_fixture(tmp_path)
    link = tmp_path / "link.pt"
    try:
        link.symlink_to(path)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(FileNotFoundError, match="not a regular file"):
        portable.load_portable_age_checkpoint(link, allow_fixture=True)
