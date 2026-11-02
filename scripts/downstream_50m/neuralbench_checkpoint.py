"""Fail-closed loaders for authoritative NeuralBench full checkpoints.

The source checkpoint is a Lightning checkpoint whose model tensors live below
``state_dict``.  Mapping is intentionally exact: suffix matching and partial
``strict=False`` loads are not permitted here.
"""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


DEFAULT_NEURALBENCH_SEX_SHA256 = (
    "56284992271d8cf0245bb6cc28f6098daf569d1188923196bfdd3d090a60342f"
)
DEFAULT_NEURALBENCH_HBN_AGE_SHA256 = (
    "a3438f5b13af4bb3a146e1c84f6ebdba22477d74046170825b98b2ef1be98b1d"
)
SOURCE_BACKBONE_PREFIX = "model.wrapped_model.backbone."
SOURCE_PROBE_WEIGHT = "model.probe.weight"
SOURCE_PROBE_BIAS = "model.probe.bias"
ALLOWED_IGNORED_STATE_KEYS = frozenset({"loss.weight"})
KEY_REMAP_VERSION = "neuralbench-sex-explicit-v1"
AGE_KEY_REMAP_VERSION = "neuralbench-hbn-age-explicit-v1"


class NeuralBenchCheckpointError(RuntimeError):
    """Checkpoint integrity, structure, or compatibility failure."""


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint_sha256(path: str | Path, expected_sha256: str) -> dict[str, Any]:
    """Check existence and SHA before any deserialization is allowed."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"NeuralBench checkpoint does not exist: {path}")
    expected = str(expected_sha256).strip().lower()
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise ValueError("Expected checkpoint SHA256 must be exactly 64 hexadecimal characters")
    actual = sha256_file(path)
    if actual != expected:
        raise NeuralBenchCheckpointError(
            f"Checkpoint SHA256 mismatch for {path}: expected={expected}, actual={actual}"
        )
    return {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": actual}


def safe_load_neuralbench_checkpoint(
    path: str | Path,
    expected_sha256: str,
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    """Verify bytes first, then use PyTorch's restricted weights-only loader."""

    file_info = verify_checkpoint_sha256(path, expected_sha256)
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise NeuralBenchCheckpointError(
            f"Checkpoint root must be a mapping, got {type(checkpoint).__name__}"
        )
    if "state_dict" not in checkpoint:
        raise NeuralBenchCheckpointError("Checkpoint root is missing required 'state_dict'")
    state = checkpoint["state_dict"]
    if not isinstance(state, Mapping):
        raise NeuralBenchCheckpointError("Checkpoint 'state_dict' must be a mapping")
    if not state:
        raise NeuralBenchCheckpointError("Checkpoint 'state_dict' is empty")
    non_tensors = sorted(str(key) for key, value in state.items() if not torch.is_tensor(value))
    if non_tensors:
        raise NeuralBenchCheckpointError(
            "Checkpoint state_dict contains non-tensor entries: " + ", ".join(non_tensors)
        )
    return checkpoint, file_info


def _component(source_key: str) -> str:
    return _component_for_probe(source_key, "downstream_sex_probe")


def _component_for_probe(source_key: str, probe_component: str) -> str:
    if source_key in (SOURCE_PROBE_WEIGHT, SOURCE_PROBE_BIAS):
        return probe_component
    if source_key == "loss.weight":
        return "loss_class_weight_buffer"
    if source_key.startswith(SOURCE_BACKBONE_PREFIX):
        relative = source_key[len(SOURCE_BACKBONE_PREFIX) :]
        if relative.startswith("head."):
            return "backbone_upstream_head"
        if relative.startswith("tokenizer."):
            return "tokenizer"
        if relative.startswith("channel_embed."):
            return "channel_embedding"
        if relative.startswith("time_embed."):
            return "time_embedding"
        if relative.startswith("encoder.encoder.layers."):
            return "transformer_encoder"
        return "backbone_other"
    return "unexpected_source"


def _target_key(source_key: str) -> str | None:
    if source_key.startswith(SOURCE_BACKBONE_PREFIX):
        return "backbone." + source_key[len(SOURCE_BACKBONE_PREFIX) :]
    if source_key == SOURCE_PROBE_WEIGHT:
        return "head.weight"
    if source_key == SOURCE_PROBE_BIAS:
        return "head.bias"
    return None


def _shape_text(tensor: torch.Tensor | None) -> str:
    return "" if tensor is None else "[" + ",".join(str(v) for v in tensor.shape) + "]"


def remap_neuralbench_sex_checkpoint(
    source_state: Mapping[str, torch.Tensor],
    target_state: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], dict[str, Any]]:
    """Map every source tensor by exact prefix and require complete target coverage.

    ``model.wrapped_model.backbone.head.*`` remains part of the upstream
    pretraining backbone.  Only the two exact ``model.probe.*`` keys map to the
    downstream sex head.
    """

    mapped, mapping_rows, summary = _remap_neuralbench_checkpoint(
        source_state,
        target_state,
        probe_weight_shape=(2, 512),
        probe_bias_shape=(2,),
        probe_name="sex",
        probe_component="downstream_sex_probe",
        key_remap_version=KEY_REMAP_VERSION,
        allowed_ignored_state_keys=ALLOWED_IGNORED_STATE_KEYS,
        expected_backbone_tensor_count=None,
        require_all_float_tensors_finite=False,
    )
    # Preserve the established sex zero-shot summary schema exactly.
    for field in (
        "source_backbone_tensor_count",
        "source_probe_tensor_count",
        "all_float_tensors_finite",
    ):
        summary.pop(field, None)
    return mapped, mapping_rows, summary


def remap_neuralbench_hbn_age_checkpoint(
    source_state: Mapping[str, torch.Tensor],
    target_state: Mapping[str, torch.Tensor],
    *,
    expected_backbone_tensor_count: int = 154,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], dict[str, Any]]:
    """Strictly map the audited HBN age full-finetune checkpoint.

    The age source has exactly 154 backbone tensors and two exact regression
    probe tensors.  Unlike the sex source, it has no allowed class-weight
    buffer or other ignored state.
    """

    return _remap_neuralbench_checkpoint(
        source_state,
        target_state,
        probe_weight_shape=(1, 512),
        probe_bias_shape=(1,),
        probe_name="age",
        probe_component="downstream_age_probe",
        key_remap_version=AGE_KEY_REMAP_VERSION,
        allowed_ignored_state_keys=frozenset(),
        expected_backbone_tensor_count=int(expected_backbone_tensor_count),
        require_all_float_tensors_finite=True,
    )


def _remap_neuralbench_checkpoint(
    source_state: Mapping[str, torch.Tensor],
    target_state: Mapping[str, torch.Tensor],
    *,
    probe_weight_shape: tuple[int, ...],
    probe_bias_shape: tuple[int, ...],
    probe_name: str,
    probe_component: str,
    key_remap_version: str,
    allowed_ignored_state_keys: frozenset[str],
    expected_backbone_tensor_count: int | None,
    require_all_float_tensors_finite: bool,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], dict[str, Any]]:
    """Shared exact remap engine; task wrappers define the probe contract."""

    source = {str(key): value for key, value in source_state.items()}
    target = {str(key): value for key, value in target_state.items()}
    required_probe = {SOURCE_PROBE_WEIGHT, SOURCE_PROBE_BIAS}
    missing_probe = sorted(required_probe.difference(source))
    if missing_probe:
        raise NeuralBenchCheckpointError(
            f"Checkpoint is missing required {probe_name} probe keys: " + ", ".join(missing_probe)
        )
    if tuple(source[SOURCE_PROBE_WEIGHT].shape) != probe_weight_shape:
        shape = "[" + ",".join(str(value) for value in probe_weight_shape) + "]"
        raise NeuralBenchCheckpointError(f"model.probe.weight must have shape {shape}")
    if tuple(source[SOURCE_PROBE_BIAS].shape) != probe_bias_shape:
        shape = "[" + ",".join(str(value) for value in probe_bias_shape) + "]"
        raise NeuralBenchCheckpointError(f"model.probe.bias must have shape {shape}")
    backbone_keys = [key for key in source if key.startswith(SOURCE_BACKBONE_PREFIX)]
    if not backbone_keys:
        raise NeuralBenchCheckpointError("Checkpoint contains no NeuralBench backbone tensors")
    if (
        expected_backbone_tensor_count is not None
        and len(backbone_keys) != expected_backbone_tensor_count
    ):
        raise NeuralBenchCheckpointError(
            "NeuralBench backbone tensor count mismatch: "
            f"expected={expected_backbone_tensor_count}, actual={len(backbone_keys)}"
        )

    mapped: dict[str, torch.Tensor] = {}
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    matched_components: dict[str, list[dict[str, Any]]] = {
        "core_backbone": [], "full_backbone": [], "probe": []
    }
    for source_key in sorted(source):
        source_tensor = source[source_key]
        component = _component_for_probe(source_key, probe_component)
        target_key = _target_key(source_key)
        target_tensor = target.get(target_key) if target_key is not None else None
        status = "matched"
        reason = "explicit key remap; exact shape and dtype"
        if (
            require_all_float_tensors_finite
            and (source_tensor.is_floating_point() or source_tensor.is_complex())
            and not torch.isfinite(source_tensor).all().item()
        ):
            status, reason = "non_finite", "source floating tensor contains non-finite values"
            failures.append(f"{source_key}: {reason}")
        elif source_key in allowed_ignored_state_keys:
            if tuple(source_tensor.shape) != (2,) or not source_tensor.dtype.is_floating_point:
                status, reason = "invalid_ignored", "loss.weight must be floating point with shape [2]"
                failures.append(f"{source_key}: {reason}")
            else:
                status, reason = "ignored_allowed", "training-only class-weight buffer"
        elif target_key is None:
            status, reason = "unexpected_source", "no explicit source-to-target mapping"
            failures.append(f"{source_key}: {reason}")
        elif target_tensor is None:
            status, reason = "missing_target", "mapped target key does not exist"
            failures.append(f"{source_key}: {reason}")
        elif tuple(source_tensor.shape) != tuple(target_tensor.shape):
            status, reason = "shape_mismatch", "source and target shapes differ"
            failures.append(f"{source_key}: {reason}")
        elif source_tensor.dtype != target_tensor.dtype:
            status, reason = "dtype_mismatch", "source and target dtypes differ"
            failures.append(f"{source_key}: {reason}")
        else:
            if target_key in mapped:
                status, reason = "duplicate_target", "multiple source tensors map to one target"
                failures.append(f"{source_key}: {reason}")
            else:
                mapped[target_key] = source_tensor
                item = {"source_key": source_key, "target_key": target_key, "numel": source_tensor.numel()}
                if source_key.startswith(SOURCE_BACKBONE_PREFIX):
                    matched_components["full_backbone"].append(item)
                    if component != "backbone_upstream_head":
                        matched_components["core_backbone"].append(item)
                elif source_key in required_probe:
                    matched_components["probe"].append(item)
        rows.append({
            "source_key": source_key,
            "target_key": target_key or "",
            "component": component,
            "source_shape": _shape_text(source_tensor),
            "target_shape": _shape_text(target_tensor),
            "source_dtype": str(source_tensor.dtype),
            "target_dtype": "" if target_tensor is None else str(target_tensor.dtype),
            "source_numel": int(source_tensor.numel()),
            "status": status,
            "reason": reason,
        })

    missing_targets = sorted(set(target).difference(mapped))
    for target_key in missing_targets:
        target_tensor = target[target_key]
        rows.append({
            "source_key": "", "target_key": target_key, "component": "target_only",
            "source_shape": "", "target_shape": _shape_text(target_tensor),
            "source_dtype": "", "target_dtype": str(target_tensor.dtype),
            "source_numel": "", "status": "missing_source",
            "reason": "model tensor has no checkpoint source",
        })
        failures.append(f"{target_key}: model tensor has no checkpoint source")

    def stats(name: str, expected_target_keys: list[str]) -> dict[str, Any]:
        matched = matched_components[name]
        expected_numel = sum(int(target[key].numel()) for key in expected_target_keys)
        matched_numel = sum(int(item["numel"]) for item in matched)
        return {
            "expected_keys": len(expected_target_keys),
            "matched_keys": len(matched),
            "expected_parameters": expected_numel,
            "matched_parameters": matched_numel,
            "key_match_fraction": len(matched) / len(expected_target_keys) if expected_target_keys else 0.0,
            "parameter_match_fraction": matched_numel / expected_numel if expected_numel else 0.0,
        }

    full_target_keys = sorted(key for key in target if key.startswith("backbone."))
    core_target_keys = sorted(
        key for key in full_target_keys if not key.startswith("backbone.head.")
    )
    probe_target_keys = ["head.weight", "head.bias"]
    core = stats("core_backbone", core_target_keys)
    full = stats("full_backbone", full_target_keys)
    probe = stats("probe", probe_target_keys)
    summary = {
        "key_remap_version": key_remap_version,
        "source_backbone_tensor_count": len(backbone_keys),
        "source_probe_tensor_count": sum(key in source for key in required_probe),
        "all_float_tensors_finite": not any(row["status"] == "non_finite" for row in rows),
        "core_backbone": core,
        "full_backbone": full,
        "probe": probe,
        "core_backbone_key_match_fraction": core["key_match_fraction"],
        "core_backbone_parameter_match_fraction": core["parameter_match_fraction"],
        "full_backbone_key_match_fraction": full["key_match_fraction"],
        "full_backbone_parameter_match_fraction": full["parameter_match_fraction"],
        "probe_key_match_fraction": probe["key_match_fraction"],
        "probe_parameter_match_fraction": probe["parameter_match_fraction"],
        "ignored_source_keys": sorted(key for key in source if key in allowed_ignored_state_keys),
        "unexplained_source_keys": sorted(
            key for key in source if _target_key(key) is None and key not in allowed_ignored_state_keys
        ),
        "missing_target_keys": missing_targets,
    }
    required_fractions = (
        core["key_match_fraction"], core["parameter_match_fraction"],
        full["key_match_fraction"], full["parameter_match_fraction"],
        probe["key_match_fraction"], probe["parameter_match_fraction"],
    )
    if failures or any(value != 1.0 for value in required_fractions):
        detail = "; ".join(failures[:20])
        if len(failures) > 20:
            detail += f"; ... {len(failures) - 20} additional failures"
        raise NeuralBenchCheckpointError(
            "NeuralBench checkpoint/model mapping is not exact (tensor values omitted): " + detail
        )
    return mapped, rows, summary


def load_neuralbench_sex_model_state(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    expected_sha256: str,
) -> tuple[Mapping[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Safely load, explicitly remap, and strictly populate a meta-created model."""

    checkpoint, file_info = safe_load_neuralbench_checkpoint(checkpoint_path, expected_sha256)
    mapped, rows, summary = remap_neuralbench_sex_checkpoint(
        checkpoint["state_dict"], model.state_dict()
    )
    try:
        incompatible = model.load_state_dict(mapped, strict=True, assign=True)
    except TypeError as exc:  # pragma: no cover - guards unsupported old PyTorch
        raise NeuralBenchCheckpointError("PyTorch with load_state_dict(assign=True) is required") from exc
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise NeuralBenchCheckpointError("Strict model load reported incompatible keys")
    loaded = model.state_dict()
    for key, source_tensor in mapped.items():
        value = loaded[key].detach().cpu()
        if value.dtype != source_tensor.dtype or value.shape != source_tensor.shape or not torch.equal(value, source_tensor):
            raise NeuralBenchCheckpointError(f"Loaded model tensor differs from checkpoint for key {key}")
    summary["all_model_state_tensors_from_checkpoint"] = True
    summary["random_materialized_parameter_count"] = 0
    return checkpoint, file_info, rows, summary


def load_neuralbench_hbn_age_model_state(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    expected_sha256: str,
    *,
    expected_backbone_tensor_count: int = 154,
) -> tuple[Mapping[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Safely load and strictly populate an HBN age zero-shot model."""

    checkpoint, file_info = safe_load_neuralbench_checkpoint(checkpoint_path, expected_sha256)
    mapped, rows, summary = remap_neuralbench_hbn_age_checkpoint(
        checkpoint["state_dict"],
        model.state_dict(),
        expected_backbone_tensor_count=expected_backbone_tensor_count,
    )
    try:
        incompatible = model.load_state_dict(mapped, strict=True, assign=True)
    except TypeError as exc:  # pragma: no cover - guards unsupported old PyTorch
        raise NeuralBenchCheckpointError("PyTorch with load_state_dict(assign=True) is required") from exc
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise NeuralBenchCheckpointError("Strict model load reported incompatible keys")
    loaded = model.state_dict()
    for key, source_tensor in mapped.items():
        value = loaded[key].detach().cpu()
        if (
            value.dtype != source_tensor.dtype
            or value.shape != source_tensor.shape
            or not torch.equal(value, source_tensor)
        ):
            raise NeuralBenchCheckpointError(
                f"Loaded model tensor differs from checkpoint for key {key}"
            )
    summary["all_model_state_tensors_from_checkpoint"] = True
    summary["random_materialized_parameter_count"] = 0
    summary["age_probe_parameter_count"] = int(
        mapped["head.weight"].numel() + mapped["head.bias"].numel()
    )
    return checkpoint, file_info, rows, summary


def write_checkpoint_key_mapping(path: str | Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "source_key", "target_key", "component", "source_shape", "target_shape",
        "source_dtype", "target_dtype", "source_numel", "status", "reason",
    ]
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
