"""Fail-closed Huada target-input contracts for native LaBraM and REVE.

These contracts describe the *target* dataset adapter.  They are deliberately
separate from the Dortmund source contract embedded in a portable checkpoint.
Matching channel counts are never treated as evidence of matching channel
semantics.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np


HUADA_CHANNEL_AXIS = (
    "Fp1", "Fp2", "AF3", "Fpz", "AF4", "F7", "F3", "F4", "F8",
    "FC5", "FC1", "FC2", "FC6", "T7", "C3", "Cz", "C4", "T8",
    "CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4", "P8",
    "PO3", "PO4", "O1", "Oz", "O2", "F5", "F1", "Fz", "F2",
    "F6", "FT7", "FT8", "FC3", "FC4", "FCz", "C1", "C2", "C5",
    "C6", "CPz", "CP3", "CP4", "TP7", "TP8", "P1", "P2", "P5",
    "P6", "POz", "PO5", "PO6", "PO7", "PO8", "CB1", "CB2", "AF7",
    "AF8",
)

CONTRACT_VERSION = 1
REVE_POSITION_SOURCE_SHA256 = (
    "f62ca8d7dd4b88e1422ea17348440a41ed0b73b9ff7d12c3b1229492c9b448a5"
)
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_HASH_EXCLUDED_FIELDS = frozenset({"contract_sha256", "path_hint"})


class NativeContractError(RuntimeError):
    """A Huada target input contract is unsafe, incomplete, or incompatible."""


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _json_tree(value: Any, path: str = "contract") -> None:
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise NativeContractError(f"non-finite metadata at {path}")
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _json_tree(item, f"{path}[{index}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise NativeContractError(f"non-string mapping key at {path}")
            _json_tree(item, f"{path}.{key}")
        return
    raise NativeContractError(f"non-JSON metadata type at {path}: {type(value).__name__}")


def contract_hash_payload(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Return the logical contract, excluding self-reference/path hints."""

    return {
        key: value
        for key, value in contract.items()
        if key not in _HASH_EXCLUDED_FIELDS
    }


def contract_sha256(contract: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        canonical_json(contract_hash_payload(contract)).encode("utf-8")
    ).hexdigest()


def channel_axis_fingerprint(channels: Sequence[str]) -> str:
    if not channels or any(type(name) is not str or not name for name in channels):
        raise NativeContractError("channel axis must contain non-empty strings")
    if len(set(channels)) != len(channels):
        raise NativeContractError("channel axis contains duplicate names")
    return hashlib.sha256(canonical_json(list(channels)).encode("utf-8")).hexdigest()


def position_bank_fingerprint(
    channels: Sequence[str], positions: Sequence[Sequence[float]] | np.ndarray
) -> str:
    """Hash exact channel order plus canonical little-endian float32 positions."""

    channel_axis_fingerprint(channels)
    array = np.asarray(positions)
    if array.shape != (len(channels), 3):
        raise NativeContractError(
            f"position bank must have shape {(len(channels), 3)}, got {array.shape}"
        )
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise NativeContractError("position bank must contain finite numeric values")
    canonical = np.ascontiguousarray(array, dtype="<f4")
    digest = hashlib.sha256()
    digest.update(b"huada_position_bank_v1\0")
    digest.update(canonical_json(list(channels)).encode("utf-8"))
    digest.update(b"\0float32_le\0")
    digest.update(canonical_json(list(canonical.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _require_equal(mapping: Mapping[str, Any], key: str, expected: Any, path: str) -> None:
    actual = mapping.get(key)
    if type(actual) is not type(expected) or actual != expected:
        raise NativeContractError(
            f"{path}.{key} must equal {expected!r}, got {actual!r}"
        )


def _validate_common(contract: Mapping[str, Any], family: str) -> None:
    _json_tree(dict(contract))
    _require_equal(
        contract, "schema_name", f"huada_{family}_target_input_contract", "contract"
    )
    _require_equal(contract, "schema_version", CONTRACT_VERSION, "contract")
    _require_equal(contract, "model_family", family, "contract")
    _require_equal(contract, "status", "frozen", "contract")
    _require_equal(contract, "release_status", "released", "contract")
    _require_equal(contract, "validation_mode", "formal", "contract")
    _require_equal(contract, "source_format", "BDF", "contract")
    _require_equal(contract, "source_data_unit", "MNE_volts", "contract")
    _require_equal(contract, "source_sampling_rate_hz", 500.0, "contract")
    _require_equal(contract, "target_sampling_rate_hz", 200.0, "contract")
    _require_equal(contract, "window_seconds", 2.0, "contract")
    _require_equal(contract, "window_samples", 400, "contract")
    _require_equal(contract, "window_hop_samples", 400, "contract")
    _require_equal(contract, "cache_scope", "full_recording", "contract")
    _require_equal(contract, "cache_dtype", "float32", "contract")
    _require_equal(contract, "fbmae_token_lance_reused", False, "contract")
    axis = contract.get("channel_axis")
    if axis != list(HUADA_CHANNEL_AXIS):
        raise NativeContractError("contract channel_axis is not the audited Huada BDF order")
    expected_axis_sha = channel_axis_fingerprint(HUADA_CHANNEL_AXIS)
    _require_equal(contract, "channel_axis_fingerprint", expected_axis_sha, "contract")
    _require_equal(contract, "channel_semantics", "huada_bdf_header_order_v1", "contract")
    if contract.get("dortmund_channel_axis_reused") is not False:
        raise NativeContractError("Dortmund channel axis reuse is forbidden")
    value = contract.get("contract_sha256")
    if type(value) is not str or not _SHA_RE.fullmatch(value):
        raise NativeContractError("contract_sha256 must be a complete lowercase SHA256")
    observed = contract_sha256(contract)
    if value != observed:
        raise NativeContractError(
            f"target input contract SHA mismatch: expected={value}, actual={observed}"
        )


def validate_huada_target_contract(
    contract: Mapping[str, Any], *, expected_family: str | None = None
) -> Mapping[str, Any]:
    if type(contract) is not dict:
        raise NativeContractError("target input contract root must be a dict")
    family = contract.get("model_family")
    if family not in {"labram", "reve"}:
        raise NativeContractError(f"unsupported target model_family: {family!r}")
    if expected_family is not None and family != expected_family:
        raise NativeContractError(
            f"target model family mismatch: expected={expected_family}, actual={family}"
        )
    _validate_common(contract, family)
    preprocessing = contract.get("preprocessing")
    if type(preprocessing) is not dict:
        raise NativeContractError("preprocessing must be a dict")
    if family == "labram":
        expected = {
            "processing_order": ["bandpass", "notch", "resample", "extractor_scale"],
            "bandpass_hz": [0.1, 75.0],
            "notch_hz": [50.0],
            "notch_harmonic_policy": "multiples_below_bandpass_high_and_source_nyquist",
            "extractor_scale": 10000.0,
            "extractor_scale_stage": "after_resample_before_full_recording_cache",
            "extractor_scale_application_count": 1,
            "cache_payload_semantics": "mne_volts_times_10000",
            "wrapper_scale": 1.0,
        }
        for key, value in expected.items():
            _require_equal(preprocessing, key, value, "preprocessing")
        mapping = contract.get("channel_mapping")
        if type(mapping) is not dict:
            raise NativeContractError("LaBraM channel_mapping must be a dict")
        _require_equal(mapping, "source_channel_axis", list(HUADA_CHANNEL_AXIS), "channel_mapping")
        mapped = mapping.get("mapped_channel_names")
        if type(mapped) is not list or len(mapped) != 64 or len(set(mapped)) != 64:
            raise NativeContractError("LaBraM mapped_channel_names must contain 64 unique names")
        _require_equal(mapping, "position_semantics", "validity_only_not_geometry", "channel_mapping")
        _require_equal(mapping, "invalid_position_value", -0.1, "channel_mapping")
        _require_equal(mapping, "all_channels_valid", True, "channel_mapping")
        _require_equal(mapping, "vocabulary_size", 128, "channel_mapping")
        _require_equal(
            mapping,
            "vocabulary_source_version",
            "braindecode-1.8.1",
            "channel_mapping",
        )
        _require_equal(mapping, "all_names_verified_in_vocabulary", True, "channel_mapping")
    else:
        expected = {
            "processing_order": ["bandpass", "resample", "standardize", "clip"],
            "bandpass_hz": [0.5, 99.5],
            "notch_hz": None,
            "normalization_method": "StandardScaler",
            "normalization_scope": "per_recording_per_channel",
            "normalization_statistics_source": "entire_preprocessed_recording",
            "normalization_ddof": 0,
            "constant_channel_scale": 1.0,
            "clip_range": [-15.0, 15.0],
            "amplitude_scale_factor": 1.0,
            "cache_payload_semantics": "per_recording_per_channel_standardized_then_clipped",
        }
        for key, value in expected.items():
            _require_equal(preprocessing, key, value, "preprocessing")
        if preprocessing.get("extractor_scale") == 10000.0:
            raise NativeContractError("REVE contract contains forbidden LaBraM x10000 scaling")
        bank = contract.get("position_bank")
        if type(bank) is not dict:
            raise NativeContractError("REVE position_bank must be an embedded dict")
        _require_equal(bank, "channel_axis", list(HUADA_CHANNEL_AXIS), "position_bank")
        _require_equal(bank, "shape", [64, 3], "position_bank")
        _require_equal(bank, "position_mask", None, "position_bank")
        _require_equal(
            bank,
            "position_semantics",
            "pretrained_model_channel_name_embedding_not_measured_huada_digitization",
            "position_bank",
        )
        _require_equal(bank, "dortmund_position_bank_reused", False, "position_bank")
        positions = bank.get("positions")
        observed = position_bank_fingerprint(HUADA_CHANNEL_AXIS, positions)
        if bank.get("fingerprint") != observed:
            raise NativeContractError("REVE Huada position bank fingerprint mismatch")
        source_sha = bank.get("source_asset_sha256")
        if source_sha != REVE_POSITION_SOURCE_SHA256:
            raise NativeContractError("REVE position-bank source asset SHA is not released")
    return MappingProxyType(dict(contract))


def load_huada_target_contract(
    path: str | Path,
    *,
    expected_family: str | None = None,
    expected_sha256: str | None = None,
) -> Mapping[str, Any]:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise FileNotFoundError(f"target input contract is not a regular file: {candidate}")
    file_sha = hashlib.sha256(candidate.read_bytes()).hexdigest()
    if expected_sha256 is not None:
        expected = expected_sha256.strip().lower()
        if not _SHA_RE.fullmatch(expected) or expected != file_sha:
            raise NativeContractError(
                f"target contract file SHA mismatch: expected={expected}, actual={file_sha}"
            )
    try:
        raw = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise NativeContractError("cannot read target input contract JSON") from exc
    return validate_huada_target_contract(raw, expected_family=expected_family)


def build_reve_position_bank(
    source: Mapping[str, Sequence[float]],
    *,
    source_asset_sha256: str,
    channel_name_mapping: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Select a Huada-ordered 64x3 bank from the released REVE asset."""

    if source_asset_sha256 != REVE_POSITION_SOURCE_SHA256:
        raise NativeContractError("source REVE position asset SHA mismatch")
    mapping = dict(channel_name_mapping or {})
    unknown = sorted(set(mapping).difference(HUADA_CHANNEL_AXIS))
    if unknown:
        raise NativeContractError(f"position mapping has unknown Huada channels: {unknown}")
    resolved_names = [mapping.get(name, name) for name in HUADA_CHANNEL_AXIS]
    if len(set(resolved_names)) != len(resolved_names):
        raise NativeContractError("position mapping is not one-to-one")
    missing = [name for name in resolved_names if name not in source]
    if missing:
        raise NativeContractError(f"REVE position asset lacks mapped channels: {missing}")
    positions = np.asarray([source[name] for name in resolved_names], dtype=np.float32)
    fingerprint = position_bank_fingerprint(HUADA_CHANNEL_AXIS, positions)
    return {
        "channel_axis": list(HUADA_CHANNEL_AXIS),
        "resolved_position_names": resolved_names,
        "shape": [64, 3],
        "dtype": "float32",
        "positions": positions.tolist(),
        "fingerprint": fingerprint,
        "position_mask": None,
        "source_asset_sha256": source_asset_sha256,
        "source_asset": "brain-bzh/reve-positions positions.json",
        "position_semantics": "pretrained_model_channel_name_embedding_not_measured_huada_digitization",
        "dortmund_position_bank_reused": False,
    }


def make_huada_target_contract(
    family: str,
    *,
    reve_position_bank: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the frozen v1 target contract; REVE requires a released bank."""

    if family not in {"labram", "reve"}:
        raise NativeContractError(f"unsupported target family: {family!r}")
    value: dict[str, Any] = {
        "schema_name": f"huada_{family}_target_input_contract",
        "schema_version": 1,
        "model_family": family,
        "status": "frozen",
        "release_status": "released",
        "validation_mode": "formal",
        "source_format": "BDF",
        "source_data_unit": "MNE_volts",
        "source_sampling_rate_hz": 500.0,
        "target_sampling_rate_hz": 200.0,
        "window_seconds": 2.0,
        "window_samples": 400,
        "window_hop_samples": 400,
        "cache_scope": "full_recording",
        "cache_dtype": "float32",
        "channel_axis": list(HUADA_CHANNEL_AXIS),
        "channel_axis_fingerprint": channel_axis_fingerprint(HUADA_CHANNEL_AXIS),
        "channel_semantics": "huada_bdf_header_order_v1",
        "channel_axis_evidence": (
            "preprocessing_audit/20261030_105520/channel_header_inventory.csv"
        ),
        "source_reference": "unknown",
        "source_montage": "unknown",
        "known_unknowns": ["source_reference", "source_montage"],
        "dortmund_channel_axis_reused": False,
        "fbmae_token_lance_reused": False,
    }
    if family == "labram":
        value["preprocessing"] = {
            "processing_order": ["bandpass", "notch", "resample", "extractor_scale"],
            "bandpass_hz": [0.1, 75.0],
            "notch_hz": [50.0],
            "notch_harmonic_policy": "multiples_below_bandpass_high_and_source_nyquist",
            "extractor_scale": 10000.0,
            "extractor_scale_stage": "after_resample_before_full_recording_cache",
            "extractor_scale_application_count": 1,
            "cache_payload_semantics": "mne_volts_times_10000",
            "wrapper_scale": 1.0,
        }
        value["channel_mapping"] = {
            "source_channel_axis": list(HUADA_CHANNEL_AXIS),
            "mapped_channel_names": list(HUADA_CHANNEL_AXIS),
            "mapping_method": "explicit_case_insensitive_labram_vocabulary_match",
            "position_semantics": "validity_only_not_geometry",
            "invalid_position_value": -0.1,
            "valid_position_marker": [0.0, 0.0, 0.0],
            "all_channels_valid": True,
            "vocabulary_name": "braindecode_LABRAM_CHANNEL_ORDER",
            "vocabulary_size": 128,
            "vocabulary_source_version": "braindecode-1.8.1",
            "all_names_verified_in_vocabulary": True,
            "dortmund_mapping_reused": False,
        }
    else:
        if reve_position_bank is None:
            raise NativeContractError(
                "REVE target contract cannot be released without a Huada-ordered position bank"
            )
        value["preprocessing"] = {
            "processing_order": ["bandpass", "resample", "standardize", "clip"],
            "bandpass_hz": [0.5, 99.5],
            "notch_hz": None,
            "normalization_method": "StandardScaler",
            "normalization_scope": "per_recording_per_channel",
            "normalization_statistics_source": "entire_preprocessed_recording",
            "normalization_ddof": 0,
            "constant_channel_scale": 1.0,
            "clip_range": [-15.0, 15.0],
            "amplitude_scale_factor": 1.0,
            "extractor_scale": None,
            "cache_payload_semantics": "per_recording_per_channel_standardized_then_clipped",
        }
        value["position_bank"] = dict(reve_position_bank)
    value["contract_sha256"] = contract_sha256(value)
    validate_huada_target_contract(value, expected_family=family)
    return value


def write_huada_target_contract(
    path: str | Path, contract: Mapping[str, Any]
) -> Path:
    validated = validate_huada_target_contract(
        dict(contract), expected_family=str(contract.get("model_family"))
    )
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite target contract: {target}")
    if not target.parent.is_dir():
        raise FileNotFoundError(f"target contract parent does not exist: {target.parent}")
    target.write_text(
        json.dumps(dict(validated), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    # Re-read and validate published bytes.
    load_huada_target_contract(target, expected_family=str(contract["model_family"]))
    return target
