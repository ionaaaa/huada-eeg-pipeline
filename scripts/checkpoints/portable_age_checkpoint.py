"""Portable age full-checkpoint v1: construction, validation, and safe I/O.

This module is deliberately model-construction-free.  A portable file contains
canonical model tensors plus JSON-compatible reconstruction contracts; it never
contains framework, optimizer, scheduler, callback, loop, or RNG objects.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch

from scripts.downstream_50m.neuralbench_checkpoint import (
    sha256_file,
    tensor_state_fingerprint,
)


SCHEMA_NAME = "portable_age_full_checkpoint"
SCHEMA_VERSION = 1
CHECKPOINT_KIND = "portable_age_full_finetuned"
MANIFEST_SUFFIX = ".manifest.json"
SCHEMA_SPEC_SHA256 = "b7a77fcb5b359bb9621d9a800f7090fedef1f247a926d4d93ece8764c03c076f"
CANONICAL_FIXTURE_FINGERPRINT = "d0b559872ff8bd1258da9bec52ce53cd0dfc1b0c2fd99f447fb6f159f3a48325"
_DORTMUND_SPLIT_SEMANTIC_SHA256 = (
    "a4e732f40958aae88cc68756f826a4747cdb10ed0d9d85d41a9b9d1c16c2d29a"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_WRAPPER_PREFIXES = ("module.", "model.", "model.wrapped_model.", "_lightning_module.")
_FLOAT_DTYPES = frozenset(
    {torch.float16, torch.float32, torch.float64, torch.bfloat16}
)
_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_name", "schema_version", "checkpoint_kind", "validation_mode",
        "task", "model", "backbone_state_dict", "age_probe_state_dict",
        "input_contract", "representation_contract", "target_transform",
        "source_training", "source_checkpoint", "provenance", "integrity",
    }
)
_REQUIRED_TOP_LEVEL_FIELDS = _TOP_LEVEL_FIELDS.difference({"validation_mode"})
_METADATA_FIELDS = tuple(
    field for field in _TOP_LEVEL_FIELDS
    if field not in {"backbone_state_dict", "age_probe_state_dict", "validation_mode"}
)


class PortableCheckpointError(RuntimeError):
    """A portable checkpoint is unsafe, malformed, or contract-incompatible."""


ModelValidator = Callable[[Mapping[str, Any], str], None]
_MODEL_VALIDATORS: dict[str, ModelValidator] = {}


@dataclass(frozen=True)
class PortableAgeCheckpoint:
    """Validated checkpoint view; never the unvalidated object returned by torch."""

    payload: Mapping[str, Any]
    path: Path | None = None
    file_sha256: str | None = None
    file_size_bytes: int | None = None

    @property
    def model_name(self) -> str:
        return str(self.payload["model"]["model_name"])

    @property
    def model_family(self) -> str:
        return str(self.payload["model"]["model_family"])

    @property
    def backbone_state_dict(self) -> Mapping[str, torch.Tensor]:
        return self.payload["backbone_state_dict"]

    @property
    def age_probe_state_dict(self) -> Mapping[str, torch.Tensor]:
        return self.payload["age_probe_state_dict"]


def register_portable_model_validator(
    model_family: str,
    validator: ModelValidator,
    *,
    replace: bool = False,
) -> None:
    """Register one explicit family contract; unknown families remain rejected."""

    family = str(model_family).strip().lower()
    if not family or not callable(validator):
        raise ValueError("model_family must be non-empty and validator must be callable")
    if family in _MODEL_VALIDATORS and not replace:
        raise ValueError(f"A validator is already registered for model family {family!r}")
    _MODEL_VALIDATORS[family] = validator


def canonical_json(value: Any) -> str:
    """The v1 metadata canonicalization, independent of insertion order/platform."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def metadata_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _fail(message: str) -> None:
    raise PortableCheckpointError(message)


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{name} must be a mapping")
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        _fail(
            f"{name} keys mismatch: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )


def _require_nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(f"{name} must be a non-empty string")
    return value


def _require_sha256(value: Any, name: str) -> str:
    text = _require_nonempty_string(value, name).lower()
    if not _SHA256_RE.fullmatch(text):
        _fail(f"{name} must be exactly 64 lowercase hexadecimal characters")
    return text


def _require_int(value: Any, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        _fail(f"{name} must be >= {minimum}")
    return value


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{name} must be finite")
    return result


def _validate_json_value(value: Any, path: str = "metadata") -> None:
    if value is None or type(value) in (str, bool):
        return
    if type(value) is int:
        return
    if type(value) is float:
        if not math.isfinite(value):
            _fail(f"{path} contains non-finite float")
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _validate_json_value(item, f"{path}[{index}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            if not isinstance(key, str):
                _fail(f"{path} contains a non-string mapping key")
            _validate_json_value(item, f"{path}.{key}")
        return
    _fail(f"{path} contains forbidden non-JSON type {type(value).__name__}")


def _clone_tree(value: Any, *, readonly: bool) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        copied = {key: _clone_tree(item, readonly=readonly) for key, item in value.items()}
        return MappingProxyType(copied) if readonly else copied
    if isinstance(value, list):
        copied = [_clone_tree(item, readonly=readonly) for item in value]
        return tuple(copied) if readonly else copied
    # Read-only views encode validated JSON lists as tuples.  Convert them
    # back to plain lists before torch.save so nested MappingProxyType items
    # never leak into the serialized portable object.
    if type(value) is tuple:
        copied = [_clone_tree(item, readonly=readonly) for item in value]
        return tuple(copied) if readonly else copied
    return value


def _validate_state_dict(
    value: Any,
    name: str,
    *,
    allow_scalar: bool = True,
) -> Mapping[str, torch.Tensor]:
    state = _require_mapping(value, name)
    if not state:
        _fail(f"{name} must not be empty")
    for key, tensor in state.items():
        if not isinstance(key, str) or not key:
            _fail(f"{name} keys must be non-empty strings")
        if key.startswith(_WRAPPER_PREFIXES):
            _fail(f"{name} key retains forbidden training wrapper prefix: {key}")
        if type(tensor) is not torch.Tensor:
            _fail(f"{name}[{key!r}] must be a tensor")
        if tensor.layout != torch.strided or tensor.device.type == "meta" or tensor.is_quantized:
            _fail(f"{name}[{key!r}] must be a materialized strided tensor")
        if tensor.ndim == 0 and not allow_scalar:
            _fail(f"{name}[{key!r}] may not be a scalar tensor")
        if tensor.is_floating_point() or tensor.is_complex():
            if not bool(torch.isfinite(tensor).all().item()):
                _fail(f"{name}[{key!r}] contains NaN or Inf")
    return state


def _validate_task(payload: Mapping[str, Any]) -> None:
    task = _require_mapping(payload["task"], "task")
    expected = {
        "name": "age_regression", "output_dim": 1,
        "target_name": "age", "target_unit": "year",
    }
    if dict(task) != expected:
        _fail(f"task must equal the portable age v1 contract: {expected}")


def _validate_model(payload: Mapping[str, Any]) -> tuple[str, str]:
    model = _require_mapping(payload["model"], "model")
    _require_exact_keys(model, {"model_name", "model_family", "model_config"}, "model")
    name = _require_nonempty_string(model["model_name"], "model.model_name")
    family = _require_nonempty_string(model["model_family"], "model.model_family").lower()
    _require_mapping(model["model_config"], "model.model_config")
    return name, family


def _validate_contracts(payload: Mapping[str, Any]) -> int:
    input_contract = _require_mapping(payload["input_contract"], "input_contract")
    input_required = {
        "contract_version", "input_kind", "sampling_rate_hz", "window_seconds",
        "input_dtype", "shape_semantics", "channel_semantics",
        "electrode_vocabulary_name", "electrode_vocabulary_sha256",
        "channel_canonicalization_version", "channel_canonicalization_sha256",
        "mask_semantics_version", "mask_semantics_sha256", "preprocessing_profile",
        "preprocessing_config_sha256",
    }
    missing = input_required.difference(input_contract)
    if missing:
        _fail(f"input_contract is missing required fields: {sorted(missing)}")
    if type(input_contract["contract_version"]) is not int or input_contract["contract_version"] != 1:
        _fail("input_contract.contract_version must be 1")
    for field in (
        "input_kind", "input_dtype", "shape_semantics", "channel_semantics",
        "electrode_vocabulary_name", "channel_canonicalization_version",
        "mask_semantics_version", "preprocessing_profile",
    ):
        _require_nonempty_string(input_contract[field], f"input_contract.{field}")
    if _finite_float(input_contract["sampling_rate_hz"], "input_contract.sampling_rate_hz") <= 0:
        _fail("input_contract.sampling_rate_hz must be positive")
    if _finite_float(input_contract["window_seconds"], "input_contract.window_seconds") <= 0:
        _fail("input_contract.window_seconds must be positive")
    for field in (
        "electrode_vocabulary_sha256", "channel_canonicalization_sha256",
        "mask_semantics_sha256", "preprocessing_config_sha256",
    ):
        _require_sha256(input_contract[field], f"input_contract.{field}")

    representation = _require_mapping(
        payload["representation_contract"], "representation_contract"
    )
    representation_required = {
        "contract_version", "representation_name", "representation_layer", "pooling",
        "feature_dim", "representation_dtype", "uses_token_valid_mask",
    }
    missing = representation_required.difference(representation)
    if missing:
        _fail(f"representation_contract is missing required fields: {sorted(missing)}")
    if type(representation["contract_version"]) is not int or representation["contract_version"] != 1:
        _fail("representation_contract.contract_version must be 1")
    _require_nonempty_string(representation["representation_name"], "representation_contract.representation_name")
    _require_nonempty_string(representation["pooling"], "representation_contract.pooling")
    feature_dim = _require_int(
        representation["feature_dim"], "representation_contract.feature_dim", minimum=1
    )
    if representation["representation_dtype"] != "float32":
        _fail("representation_contract.representation_dtype must be 'float32' in v1")
    if not isinstance(representation["uses_token_valid_mask"], bool):
        _fail("representation_contract.uses_token_valid_mask must be boolean")
    layer = representation["representation_layer"]
    if layer is not None and (isinstance(layer, bool) or not isinstance(layer, (int, str))):
        _fail("representation_contract.representation_layer must be int, string, or null")
    return feature_dim


def _validate_probe(payload: Mapping[str, Any], feature_dim: int) -> None:
    probe = _require_mapping(payload["age_probe_state_dict"], "age_probe_state_dict")
    _require_exact_keys(probe, {"weight", "bias"}, "age_probe_state_dict")
    _validate_state_dict(probe, "age_probe_state_dict", allow_scalar=False)
    weight, bias = probe["weight"], probe["bias"]
    if weight.dtype not in _FLOAT_DTYPES or bias.dtype not in _FLOAT_DTYPES:
        _fail("age probe weight and bias must use a supported floating dtype")
    if weight.dtype != bias.dtype:
        _fail("age probe weight and bias dtypes must match")
    if weight.ndim != 2 or tuple(weight.shape) != (1, feature_dim):
        _fail(f"age probe weight must have shape [1,{feature_dim}]")
    if tuple(bias.shape) != (1,):
        _fail("age probe bias must have shape [1]")


def _validate_target_transform(payload: Mapping[str, Any]) -> None:
    transform = _require_mapping(payload["target_transform"], "target_transform")
    required = {"type", "mean", "std", "ddof", "equal_subject_weight", "target_unit"}
    missing = required.difference(transform)
    if missing:
        _fail(f"target_transform is missing required fields: {sorted(missing)}")
    if transform["type"] != "zscore":
        _fail("target_transform.type must be 'zscore' in v1")
    _finite_float(transform["mean"], "target_transform.mean")
    if _finite_float(transform["std"], "target_transform.std") <= 0:
        _fail("target_transform.std must be positive")
    if transform["ddof"] != 0 or transform["equal_subject_weight"] is not True:
        _fail("target_transform must use ddof=0 and equal_subject_weight=true")
    if transform["target_unit"] != "year":
        _fail("target_transform.target_unit must be 'year'")


def _validate_source(payload: Mapping[str, Any], validation_mode: str) -> None:
    source = _require_mapping(payload["source_training"], "source_training")
    required = {
        "dataset_name", "dataset_fingerprint", "split_manifest_sha256",
        "train_subject_count", "validation_subject_count", "test_subject_count",
        "best_epoch", "selection_metric", "validation_metrics", "test_metrics",
    }
    missing = required.difference(source)
    if missing:
        _fail(f"source_training is missing required fields: {sorted(missing)}")
    _require_nonempty_string(source["dataset_name"], "source_training.dataset_name")
    _require_nonempty_string(source["dataset_fingerprint"], "source_training.dataset_fingerprint")
    _require_sha256(source["split_manifest_sha256"], "source_training.split_manifest_sha256")
    for field in ("train_subject_count", "validation_subject_count", "test_subject_count"):
        _require_int(source[field], f"source_training.{field}", minimum=1)
    _require_int(source["best_epoch"], "source_training.best_epoch", minimum=0)
    metric = _require_nonempty_string(source["selection_metric"], "source_training.selection_metric")
    if "test" in metric.lower():
        _fail("source_training.selection_metric may not select on test metrics")
    _validate_json_value(source["validation_metrics"], "source_training.validation_metrics")
    _validate_json_value(source["test_metrics"], "source_training.test_metrics")

    source_checkpoint = _require_mapping(payload["source_checkpoint"], "source_checkpoint")
    required_checkpoint = {"checkpoint_format", "source_checkpoint_size_bytes"}
    if validation_mode == "formal":
        required_checkpoint.add("source_checkpoint_sha256")
    missing = required_checkpoint.difference(source_checkpoint)
    if missing:
        _fail(f"source_checkpoint is missing required fields: {sorted(missing)}")
    _require_nonempty_string(source_checkpoint["checkpoint_format"], "source_checkpoint.checkpoint_format")
    if "source_checkpoint_sha256" in source_checkpoint:
        _require_sha256(
            source_checkpoint["source_checkpoint_sha256"],
            "source_checkpoint.source_checkpoint_sha256",
        )
    _require_int(
        source_checkpoint["source_checkpoint_size_bytes"],
        "source_checkpoint.source_checkpoint_size_bytes",
        minimum=1,
    )

    provenance = _require_mapping(payload["provenance"], "provenance")
    required_provenance = {
        "exporter_name", "exporter_version", "source_repository", "source_git_dirty",
        "created_at_utc", "python_version", "pytorch_version",
    }
    if validation_mode == "formal":
        required_provenance.update({"source_git_commit", "critical_code_sha256"})
    missing = required_provenance.difference(provenance)
    if missing:
        _fail(f"provenance is missing required fields: {sorted(missing)}")
    for field in (
        "exporter_name", "exporter_version", "source_repository", "created_at_utc",
        "python_version", "pytorch_version",
    ):
        _require_nonempty_string(provenance[field], f"provenance.{field}")
    if type(provenance["source_git_dirty"]) is not bool:
        _fail("provenance.source_git_dirty must be boolean")
    critical = _require_mapping(
        provenance.get("critical_code_sha256", {}), "provenance.critical_code_sha256"
    )
    if validation_mode == "formal":
        _require_nonempty_string(provenance["source_git_commit"], "provenance.source_git_commit")
        if not critical:
            _fail("formal provenance.critical_code_sha256 must not be empty")
        for path, digest in critical.items():
            _require_nonempty_string(path, "provenance.critical_code_sha256 path")
            _require_sha256(digest, f"provenance.critical_code_sha256[{path!r}]")


def _validate_integrity(payload: Mapping[str, Any]) -> None:
    integrity = _require_mapping(payload["integrity"], "integrity")
    expected_fields = {
        "backbone_state_fingerprint": tensor_state_fingerprint(payload["backbone_state_dict"]),
        "age_probe_state_fingerprint": tensor_state_fingerprint(payload["age_probe_state_dict"]),
        "model_config_sha256": metadata_sha256(payload["model"]["model_config"]),
        "input_contract_sha256": metadata_sha256(payload["input_contract"]),
        "representation_contract_sha256": metadata_sha256(payload["representation_contract"]),
        "target_transform_sha256": metadata_sha256(payload["target_transform"]),
    }
    _require_exact_keys(integrity, set(expected_fields), "integrity")
    for field, expected in expected_fields.items():
        actual = _require_sha256(integrity[field], f"integrity.{field}")
        if actual != expected:
            _fail(f"integrity.{field} mismatch")


def _validate_fbmae(payload: Mapping[str, Any], validation_mode: str) -> None:
    config = payload["model"]["model_config"]
    required_config = {
        "class", "input_dim", "d_model", "n_heads", "depth", "mlp_ratio",
        "dropout", "n_channels", "n_time_patches", "n_bands",
        "frames_per_patch", "objective",
    }
    missing = required_config.difference(config)
    if missing:
        _fail(f"FBMAE model_config is missing required fields: {sorted(missing)}")
    expected = {
        "class": "EEGPretrainModel", "input_dim": 100, "d_model": 512,
        "n_heads": 8, "depth": 12, "mlp_ratio": 4.0, "dropout": 0.1,
        "n_channels": 64, "n_time_patches": 10, "n_bands": 5,
        "frames_per_patch": 5, "objective": "timefreq",
    }
    mismatches = {key: {"actual": config.get(key), "expected": value}
                  for key, value in expected.items() if config.get(key) != value}
    if mismatches:
        _fail(f"FBMAE model_config mismatch: {mismatches}")
    representation = payload["representation_contract"]
    if representation["feature_dim"] != config["d_model"]:
        _fail("FBMAE feature_dim must equal model_config.d_model")
    if representation["representation_layer"] != 8:
        _fail("FBMAE representation_layer must be 8 for the current age contract")
    if representation["pooling"] != "masked_mean":
        _fail("FBMAE pooling must be 'masked_mean'")
    if representation["uses_token_valid_mask"] is not True:
        _fail("FBMAE representation must use token_valid_mask")
    input_contract = payload["input_contract"]
    if input_contract["input_kind"] != "fbmae_timefreq_tokens":
        _fail("FBMAE input_kind must be 'fbmae_timefreq_tokens'")
    if input_contract["mask_semantics_version"] != "token_valid_mask_true_is_valid_v1":
        _fail("FBMAE mask semantics version mismatch")
    if input_contract["input_dtype"] != "float32":
        _fail("FBMAE input dtype must be float32")
    state = payload["backbone_state_dict"]
    roots = ("tokenizer.", "channel_embed.", "time_embed.", "encoder.encoder.layers.", "head.head.")
    if any(not key.startswith(roots) for key in state):
        _fail("FBMAE backbone contains a non-canonical key root")
    if not any(key.startswith("tokenizer.") for key in state):
        _fail("FBMAE backbone must contain tokenizer tensors")
    if not any(key.startswith("encoder.encoder.layers.") for key in state):
        _fail("FBMAE backbone must contain encoder layer tensors")
    if validation_mode == "formal":
        tensor_count = len(state)
        parameter_count = sum(int(tensor.numel()) for tensor in state.values())
        if tensor_count != 154 or parameter_count != 38_194_713:
            _fail(
                "formal FBMAE backbone must contain the complete audited state "
                f"(154 tensors/38194713 parameters), got {tensor_count}/{parameter_count}"
            )
        for root in roots:
            if not any(key.startswith(root) for key in state):
                _fail(f"formal FBMAE backbone is missing canonical key root {root}")


register_portable_model_validator("fbmae", _validate_fbmae)


def _require_family_value(value: Any, expected: Any, name: str) -> None:
    if type(value) is not type(expected) or value != expected:
        _fail(f"{name} must equal {expected!r}, got {value!r}")


def _validate_dortmund_fft_common(
    payload: Mapping[str, Any],
    validation_mode: str,
    *,
    family: str,
    feature_dim: int,
    token_count: int,
    source_contract_sha256: str,
) -> None:
    config = payload["model"]["model_config"]
    required_config = {
        "architecture", "canonical_state_layout", "feature_dim",
        "pretrained_snapshot_revision", "pretrained_snapshot_sha256",
        "expected_backbone_tensor_count", "expected_backbone_parameter_count",
        "schema_spec_sha256", "canonical_fixture_fingerprint",
    }
    missing = required_config.difference(config)
    if missing:
        _fail(f"{family} model_config is missing required fields: {sorted(missing)}")
    _require_family_value(config["architecture"], family, f"{family}.architecture")
    _require_family_value(config["feature_dim"], feature_dim, f"{family}.feature_dim")
    _require_family_value(
        config["schema_spec_sha256"], SCHEMA_SPEC_SHA256,
        f"{family}.schema_spec_sha256",
    )
    _require_family_value(
        config["canonical_fixture_fingerprint"], CANONICAL_FIXTURE_FINGERPRINT,
        f"{family}.canonical_fixture_fingerprint",
    )
    _require_sha256(config["pretrained_snapshot_sha256"], f"{family}.pretrained_snapshot_sha256")
    _require_nonempty_string(
        config["pretrained_snapshot_revision"], f"{family}.pretrained_snapshot_revision"
    )
    _require_int(
        config["expected_backbone_tensor_count"],
        f"{family}.expected_backbone_tensor_count", minimum=1,
    )
    _require_int(
        config["expected_backbone_parameter_count"],
        f"{family}.expected_backbone_parameter_count", minimum=1,
    )

    input_contract = payload["input_contract"]
    common_input = {
        "status": "frozen", "release_status": "released",
        "unverified_fields": [], "failed_fields": [],
        "source_contract_schema_version": 1,
        "source_contract_sha256": source_contract_sha256,
        "sampling_rate_hz": 200.0, "window_seconds": 2.0,
        "window_samples": 400, "input_dtype": "float32",
        "input_shape": ["B", 64, 400],
    }
    for key, expected in common_input.items():
        if key not in input_contract:
            _fail(f"{family} input_contract is missing {key!r}")
        _require_family_value(input_contract[key], expected, f"{family}.input_contract.{key}")

    representation = payload["representation_contract"]
    expected_representation = {
        "feature_dim": feature_dim,
        "backbone_output_shape": ["B", token_count, feature_dim],
        "pooling": "mean",
        "pooling_axis": 1,
        "pooling_token_count": token_count,
        "uses_token_valid_mask": False,
        "mask_true_semantics": None,
        "all_invalid_behavior": "not_applicable_no_mask",
        "representation_dtype": "float32",
    }
    for key, expected in expected_representation.items():
        if key not in representation:
            _fail(f"{family} representation_contract is missing {key!r}")
        _require_family_value(
            representation[key], expected, f"{family}.representation_contract.{key}"
        )

    transform = payload["target_transform"]
    for key, expected in {
        "mean": 27.5,
        "std": 4.218591213226318,
        "recomputed_std": 4.218591189948011,
        "inverse_formula": "predicted_z * std + mean",
    }.items():
        if key not in transform:
            _fail(f"{family} target_transform is missing {key!r}")
        actual = transform[key]
        if isinstance(expected, float):
            if not isinstance(actual, (int, float)) or abs(float(actual) - expected) > 1e-12:
                _fail(f"{family} target_transform.{key} does not preserve audited precision")
        else:
            _require_family_value(actual, expected, f"{family}.target_transform.{key}")

    source_training = payload["source_training"]
    _require_family_value(
        source_training["split_manifest_sha256"],
        _DORTMUND_SPLIT_SEMANTIC_SHA256,
        f"{family}.source_training.split_manifest_sha256",
    )
    for key, expected in {
        "dataset_name": "Dortmund Vital Study",
        "train_subject_count": 172,
        "validation_subject_count": 21,
        "test_subject_count": 21,
        "adaptation": "full_finetune",
        "selection_mode": "min",
    }.items():
        if key not in source_training:
            _fail(f"{family} source_training is missing {key!r}")
        _require_family_value(source_training[key], expected, f"{family}.source_training.{key}")

    source_checkpoint = payload["source_checkpoint"]
    for key, expected in {
        "source_checkpoint_role": "best",
        "safe_load_method": "torch_weights_only_with_minimal_numpy_allowlist",
    }.items():
        if key not in source_checkpoint:
            _fail(f"{family} source_checkpoint is missing {key!r}")
        _require_family_value(source_checkpoint[key], expected, f"{family}.source_checkpoint.{key}")

    if validation_mode == "formal":
        state = payload["backbone_state_dict"]
        if len(state) != config["expected_backbone_tensor_count"]:
            _fail(
                f"formal {family} backbone tensor count mismatch: "
                f"expected={config['expected_backbone_tensor_count']}, actual={len(state)}"
            )
        parameter_count = sum(
            int(tensor.numel()) for key, tensor in state.items()
            if not key.startswith("channel_wrapper.")
        )
        if parameter_count != config["expected_backbone_parameter_count"]:
            _fail(
                f"formal {family} backbone parameter count mismatch: "
                f"expected={config['expected_backbone_parameter_count']}, actual={parameter_count}"
            )


def _validate_labram(payload: Mapping[str, Any], validation_mode: str) -> None:
    _validate_dortmund_fft_common(
        payload, validation_mode,
        family="labram", feature_dim=200, token_count=129,
        source_contract_sha256="d6c6aa3c745dfaac83c0b931c6327f16b493672ebe52902c96e16fac24cca579",
    )
    config = payload["model"]["model_config"]
    _require_family_value(
        config["canonical_state_layout"],
        "braindecode_labram_plus_channel_wrapper_buffers_v1",
        "labram.canonical_state_layout",
    )
    input_contract = payload["input_contract"]
    expected_input = {
        "input_kind": "labram_fft_time_series",
        "cache_payload_semantics": "mne_volts_times_10000",
        "extractor_scale": 10000.0,
        "extractor_scale_stage": "before_signal_cache",
        "extractor_scale_application_count": 1,
        "cache_contains_scaled_signal": True,
        "wrapper_scale": 1.0,
        "model_input_scale_relative_to_cache": 1.0,
        "bandpass_hz": [0.1, 75.0],
        "notch_hz": [50.0],
        "channel_axis_count": 64,
    }
    for key, expected in expected_input.items():
        _require_family_value(input_contract.get(key), expected, f"labram.input_contract.{key}")
    representation = payload["representation_contract"]
    for key, expected in {
        "representation_layer": "final_backbone_tokens",
        "pooling_includes_cls_token": True,
        "pooling_token_scope": "all_129_tokens_including_cls",
        "patch_length_samples": 200,
        "temporal_patch_count": 2,
    }.items():
        _require_family_value(representation.get(key), expected, f"labram.representation_contract.{key}")
    state = payload["backbone_state_dict"]
    allowed_roots = (
        "channel_wrapper.", "patch_embed.", "temporal_embedding", "cls_token",
        "mask_token", "position", "blocks.", "norm.", "fc_norm.", "final_layer.",
    )
    if any(not key.startswith(allowed_roots) for key in state):
        _fail("LaBraM backbone contains a non-canonical key root")
    for key in ("channel_wrapper._positionless_mask", "channel_wrapper._known_mask"):
        if key not in state or state[key].dtype is not torch.bool or tuple(state[key].shape) != (64,):
            _fail(f"LaBraM canonical wrapper buffer is missing or invalid: {key}")
    if validation_mode == "formal":
        if config["expected_backbone_tensor_count"] != 223:
            _fail("formal LaBraM expected backbone tensor count must be 223")
        if config["expected_backbone_parameter_count"] != 5_817_336:
            _fail("formal LaBraM expected backbone parameter count must be 5817336")
        for root in ("patch_embed.", "blocks.", "temporal_embedding"):
            if not any(key.startswith(root) for key in state):
                _fail(f"formal LaBraM backbone is missing canonical root {root}")


def _validate_reve(payload: Mapping[str, Any], validation_mode: str) -> None:
    _validate_dortmund_fft_common(
        payload, validation_mode,
        family="reve", feature_dim=512, token_count=128,
        source_contract_sha256="f72f1e8ff93a9f51ed9d76a499019dd092a892bd308bbfc8e974d406b300c092",
    )
    config = payload["model"]["model_config"]
    _require_family_value(
        config["canonical_state_layout"], "braindecode_reve_encoder_wrapper_v1",
        "reve.canonical_state_layout",
    )
    source_only = config.get("source_only_frozen_modules")
    expected_modules = [
        {
            "module": "final_layer.1", "in_age_forward_path": False,
            "required_for_exact_source_model_reconstruction": True,
            "excluded_from_age_representation": True,
        },
        {
            "module": "final_layer.2", "in_age_forward_path": False,
            "required_for_exact_source_model_reconstruction": True,
            "excluded_from_age_representation": True,
        },
    ]
    _require_family_value(source_only, expected_modules, "reve.source_only_frozen_modules")
    input_contract = payload["input_contract"]
    expected_input = {
        "input_kind": "reve_fft_time_series",
        "cache_payload_semantics": "per_recording_per_channel_standardized_then_clipped",
        "amplitude_scale_factor": 1.0,
        "cache_contains_scaled_signal": False,
        "wrapper_scale": 1.0,
        "bandpass_hz": [0.5, 99.5],
        "notch_hz": None,
        "normalization_method": "StandardScaler",
        "normalization_scope": "per_recording_per_channel",
        "normalization_statistics_source": "entire_preprocessed_recording",
        "clip_range": [-15.0, 15.0],
        "position_bank_shape": [64, 3],
        "position_mask": None,
    }
    for key, expected in expected_input.items():
        _require_family_value(input_contract.get(key), expected, f"reve.input_contract.{key}")
    if input_contract.get("extractor_scale") == 10000.0 or input_contract.get("cache_payload_semantics") == "mne_volts_times_10000":
        _fail("REVE input contract contains forbidden LaBraM x10000 semantics")
    _require_sha256(input_contract.get("position_bank_fingerprint"), "reve.position_bank_fingerprint")
    representation = payload["representation_contract"]
    for key, expected in {
        "representation_layer": "final_encoder_layer",
        "pooling_includes_cls_token": False,
        "pooling_token_scope": "all_128_encoder_tokens_no_cls",
        "patch_length_samples": 200,
        "patch_overlap_samples": 20,
        "patch_step_samples": 180,
        "temporal_patch_count": 2,
    }.items():
        _require_family_value(representation.get(key), expected, f"reve.representation_contract.{key}")
    state = payload["backbone_state_dict"]
    allowed_roots = (
        "to_patch_embedding", "fourier4d", "mlp4d", "ln", "transformer",
        "final_layer", "channel_wrapper", "cls_query_token", "default_pos",
    )
    invalid_keys = sorted(
        key for key in state
        if not any(
            key == root or key.startswith(f"{root}.") for root in allowed_roots
        )
    )
    if invalid_keys:
        _fail(
            "REVE backbone contains non-canonical keys: "
            + ", ".join(invalid_keys[:20])
        )
    for prefix in ("final_layer.1.", "final_layer.2."):
        if not any(key.startswith(prefix) for key in state):
            _fail(f"REVE complete backbone must retain {prefix}")
    if validation_mode == "formal":
        if config["expected_backbone_tensor_count"] != 143:
            _fail("formal REVE expected backbone tensor count must be 143")
        if config["expected_backbone_parameter_count"] != 69_451_266:
            _fail("formal REVE expected backbone parameter count must be 69451266")
        for root in (
            "to_patch_embedding", "mlp4d", "ln", "transformer",
            "final_layer.1", "final_layer.2",
        ):
            if not any(
                key == root or key.startswith(f"{root}.") for key in state
            ):
                _fail(f"formal REVE backbone is missing canonical root {root}")


register_portable_model_validator("labram", _validate_labram)
register_portable_model_validator("reve", _validate_reve)


def validate_portable_age_checkpoint(
    payload: Any,
    *,
    expected_model_name: str | None = None,
    expected_model_family: str | None = None,
    expected_schema_version: int = SCHEMA_VERSION,
    expected_source_checkpoint_sha256: str | None = None,
    allow_fixture: bool = False,
) -> PortableAgeCheckpoint:
    """Validate every v1 field and return an isolated, read-only checkpoint view."""

    if expected_schema_version != SCHEMA_VERSION:
        _fail(f"this loader implements schema_version {SCHEMA_VERSION} only")
    root = _require_mapping(payload, "checkpoint root")
    unknown = set(root).difference(_TOP_LEVEL_FIELDS)
    missing = _REQUIRED_TOP_LEVEL_FIELDS.difference(root)
    if missing or unknown:
        _fail(f"top-level fields mismatch: missing={sorted(missing)}, unexpected={sorted(unknown)}")
    if root["schema_name"] != SCHEMA_NAME:
        _fail(f"schema_name must be {SCHEMA_NAME!r}")
    if isinstance(root["schema_version"], bool) or root["schema_version"] != expected_schema_version:
        _fail(f"schema_version must be {expected_schema_version}")
    if root["checkpoint_kind"] != CHECKPOINT_KIND:
        _fail(f"checkpoint_kind must be {CHECKPOINT_KIND!r}")
    validation_mode = root.get("validation_mode", "formal")
    if validation_mode not in {"formal", "fixture"}:
        _fail("validation_mode must be 'formal' or 'fixture'")
    if validation_mode == "fixture" and not allow_fixture:
        _fail("fixture checkpoint is rejected unless allow_fixture=True")
    for field in _METADATA_FIELDS:
        _validate_json_value(root[field], field)
    _validate_task(root)
    model_name, model_family = _validate_model(root)
    if expected_model_name is not None and model_name != expected_model_name:
        _fail(f"model_name mismatch: expected {expected_model_name!r}, got {model_name!r}")
    if expected_model_family is not None and model_family != expected_model_family.lower():
        _fail(
            f"model_family mismatch: expected {expected_model_family!r}, got {model_family!r}"
        )
    _validate_state_dict(root["backbone_state_dict"], "backbone_state_dict")
    feature_dim = _validate_contracts(root)
    _validate_probe(root, feature_dim)
    _validate_target_transform(root)
    _validate_source(root, validation_mode)
    if expected_source_checkpoint_sha256 is not None:
        expected_source = _require_sha256(
            expected_source_checkpoint_sha256, "expected_source_checkpoint_sha256"
        )
        actual_source_value = root["source_checkpoint"].get("source_checkpoint_sha256")
        if actual_source_value is None:
            _fail("source checkpoint SHA256 is missing")
        actual_source = actual_source_value.lower()
        if actual_source != expected_source:
            _fail("source checkpoint SHA256 mismatch")
    _validate_integrity(root)
    validator = _MODEL_VALIDATORS.get(model_family)
    if validator is None:
        _fail(f"no portable model validator registered for family {model_family!r}")
    validator(root, validation_mode)
    return PortableAgeCheckpoint(payload=_clone_tree(root, readonly=True))


def _manifest_path(path: str | Path) -> Path:
    return Path(str(Path(path)) + MANIFEST_SUFFIX)


def _manifest_payload(path: Path, checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "portable_file_sha256": sha256_file(path),
        "portable_file_size_bytes": path.stat().st_size,
        "schema_name": checkpoint["schema_name"],
        "schema_version": checkpoint["schema_version"],
        "checkpoint_kind": checkpoint["checkpoint_kind"],
        "model_name": checkpoint["model"]["model_name"],
        "backbone_state_fingerprint": checkpoint["integrity"]["backbone_state_fingerprint"],
        "age_probe_state_fingerprint": checkpoint["integrity"]["age_probe_state_fingerprint"],
    }


def write_portable_manifest(
    checkpoint_path: str | Path,
    checkpoint: Mapping[str, Any] | PortableAgeCheckpoint,
    *,
    manifest_path: str | Path | None = None,
    overwrite: bool = False,
    allow_fixture: bool = False,
) -> Path:
    """Atomically write the JSON sidecar for an already validated checkpoint file."""

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    candidate = checkpoint.payload if isinstance(checkpoint, PortableAgeCheckpoint) else checkpoint
    root = validate_portable_age_checkpoint(
        _clone_tree(candidate, readonly=False), allow_fixture=allow_fixture
    ).payload
    destination = Path(manifest_path) if manifest_path is not None else _manifest_path(path)
    if not destination.parent.is_dir():
        raise FileNotFoundError(f"Manifest directory does not exist: {destination.parent}")
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(_manifest_payload(path, root), ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def _validate_manifest(path: Path, checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
    manifest_path = _manifest_path(path)
    if manifest_path.is_symlink() or not manifest_path.is_file():
        _fail(f"portable manifest is required: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PortableCheckpointError(f"portable manifest is unreadable: {manifest_path}") from exc
    _validate_json_value(manifest, "manifest")
    expected = _manifest_payload(path, checkpoint)
    if not isinstance(manifest, dict):
        _fail("portable manifest root must be an object")
    if manifest.keys() != expected.keys() or any(
        type(manifest[key]) is not type(expected[key]) or manifest[key] != expected[key]
        for key in expected.keys() & manifest.keys()
    ):
        differences = sorted(
            key for key in set(manifest).union(expected) if manifest.get(key) != expected.get(key)
        )
        _fail(f"portable manifest mismatch: {differences}")
    return manifest


def load_portable_age_checkpoint(
    path: str | Path,
    *,
    expected_file_sha256: str | None = None,
    expected_model_name: str | None = None,
    expected_model_family: str | None = None,
    expected_schema_version: int = SCHEMA_VERSION,
    expected_source_checkpoint_sha256: str | None = None,
    map_location: str | torch.device = "cpu",
    require_manifest: bool = True,
    allow_fixture: bool = False,
) -> PortableAgeCheckpoint:
    """SHA-first, weights-only, fail-closed portable checkpoint loader."""

    checkpoint_path = Path(path)
    if checkpoint_path.is_symlink() or not checkpoint_path.is_file():
        raise FileNotFoundError(f"Portable checkpoint is not a regular file: {checkpoint_path}")
    if str(map_location) != "cpu":
        _fail("portable checkpoint validation must load on CPU")
    actual_sha = sha256_file(checkpoint_path)
    if expected_file_sha256 is not None:
        expected = _require_sha256(expected_file_sha256, "expected_file_sha256")
        if actual_sha != expected:
            _fail(f"portable file SHA256 mismatch: expected={expected}, actual={actual_sha}")
    try:
        raw = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise PortableCheckpointError("PyTorch weights_only=True support is required") from exc
    except Exception as exc:
        raise PortableCheckpointError("safe weights-only checkpoint load failed") from exc
    if sha256_file(checkpoint_path) != actual_sha:
        _fail("portable checkpoint changed during safe load")
    validated = validate_portable_age_checkpoint(
        raw,
        expected_model_name=expected_model_name,
        expected_model_family=expected_model_family,
        expected_schema_version=expected_schema_version,
        expected_source_checkpoint_sha256=expected_source_checkpoint_sha256,
        allow_fixture=allow_fixture,
    )
    if require_manifest:
        _validate_manifest(checkpoint_path, validated.payload)
    return PortableAgeCheckpoint(
        payload=validated.payload,
        path=checkpoint_path.resolve(),
        file_sha256=actual_sha,
        file_size_bytes=checkpoint_path.stat().st_size,
    )


def build_portable_age_checkpoint(
    *,
    model_name: str,
    model_family: str,
    model_config: Mapping[str, Any],
    backbone_state_dict: Mapping[str, torch.Tensor],
    age_probe_state_dict: Mapping[str, torch.Tensor],
    input_contract: Mapping[str, Any],
    representation_contract: Mapping[str, Any],
    target_transform: Mapping[str, Any],
    source_training: Mapping[str, Any],
    source_checkpoint: Mapping[str, Any],
    provenance: Mapping[str, Any],
    validation_mode: str = "formal",
) -> dict[str, Any]:
    """Build and validate a plain-tensor/plain-Python portable payload."""

    payload: dict[str, Any] = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "checkpoint_kind": CHECKPOINT_KIND,
        "validation_mode": validation_mode,
        "task": {
            "name": "age_regression", "output_dim": 1,
            "target_name": "age", "target_unit": "year",
        },
        "model": {
            "model_name": model_name,
            "model_family": model_family,
            "model_config": _clone_tree(model_config, readonly=False),
        },
        "backbone_state_dict": _clone_tree(backbone_state_dict, readonly=False),
        "age_probe_state_dict": _clone_tree(age_probe_state_dict, readonly=False),
        "input_contract": _clone_tree(input_contract, readonly=False),
        "representation_contract": _clone_tree(representation_contract, readonly=False),
        "target_transform": _clone_tree(target_transform, readonly=False),
        "source_training": _clone_tree(source_training, readonly=False),
        "source_checkpoint": _clone_tree(source_checkpoint, readonly=False),
        "provenance": _clone_tree(provenance, readonly=False),
    }
    _validate_state_dict(payload["backbone_state_dict"], "backbone_state_dict")
    _validate_state_dict(
        payload["age_probe_state_dict"], "age_probe_state_dict", allow_scalar=False
    )
    payload["integrity"] = {
        "backbone_state_fingerprint": tensor_state_fingerprint(payload["backbone_state_dict"]),
        "age_probe_state_fingerprint": tensor_state_fingerprint(payload["age_probe_state_dict"]),
        "model_config_sha256": metadata_sha256(payload["model"]["model_config"]),
        "input_contract_sha256": metadata_sha256(payload["input_contract"]),
        "representation_contract_sha256": metadata_sha256(payload["representation_contract"]),
        "target_transform_sha256": metadata_sha256(payload["target_transform"]),
    }
    validate_portable_age_checkpoint(payload, allow_fixture=validation_mode == "fixture")
    return payload


def save_portable_age_checkpoint(
    payload: Mapping[str, Any],
    path: str | Path,
    *,
    overwrite: bool = False,
    allow_fixture: bool = False,
) -> PortableAgeCheckpoint:
    """Validate, stage, safe-reload, then atomically publish file and sidecar."""

    destination = Path(path)
    manifest_destination = _manifest_path(destination)
    if not destination.parent.is_dir():
        raise FileNotFoundError(f"Target directory does not exist: {destination.parent}")
    if (destination.exists() or manifest_destination.exists()) and not overwrite:
        raise FileExistsError(destination if destination.exists() else manifest_destination)
    validated = validate_portable_age_checkpoint(payload, allow_fixture=allow_fixture)
    token = uuid.uuid4().hex
    staged_checkpoint = destination.with_name(f".{destination.name}.{token}.tmp")
    staged_manifest = destination.with_name(f".{manifest_destination.name}.{token}.tmp")
    published_checkpoint = False
    try:
        torch.save(_clone_tree(validated.payload, readonly=False), staged_checkpoint)
        staged_sha = sha256_file(staged_checkpoint)
        reloaded = load_portable_age_checkpoint(
            staged_checkpoint,
            expected_file_sha256=staged_sha,
            require_manifest=False,
            allow_fixture=allow_fixture,
        )
        staged_manifest.write_text(
            json.dumps(
                _manifest_payload(staged_checkpoint, reloaded.payload),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )
        if sha256_file(staged_checkpoint) != staged_sha:
            _fail("staged checkpoint changed after validation")
        # The manifest is the validity marker and is published last.
        os.replace(staged_checkpoint, destination)
        published_checkpoint = True
        final_manifest = json.loads(staged_manifest.read_text(encoding="utf-8"))
        final_manifest["portable_file_size_bytes"] = destination.stat().st_size
        final_manifest["portable_file_sha256"] = sha256_file(destination)
        staged_manifest.write_text(
            json.dumps(final_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(staged_manifest, manifest_destination)
        result = load_portable_age_checkpoint(
            destination,
            expected_file_sha256=staged_sha,
            require_manifest=True,
            allow_fixture=allow_fixture,
        )
        return result
    except Exception:
        if published_checkpoint and destination.exists() and not manifest_destination.exists():
            destination.unlink()
        raise
    finally:
        for temporary in (staged_checkpoint, staged_manifest):
            if temporary.exists():
                temporary.unlink()
