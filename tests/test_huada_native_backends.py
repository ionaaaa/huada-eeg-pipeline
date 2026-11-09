from __future__ import annotations

import copy
import json

import numpy as np
import pytest
import torch
import torch.nn as nn

from scripts.checkpoints.portable_age_checkpoint import PortableAgeCheckpoint
from scripts.native_backends import contracts
from scripts.native_backends.portable_age_backend import (
    HuadaLaBraMAgeBackend,
    HuadaREVEAgeBackend,
)
from scripts.native_backends.preprocessing import (
    NativePreprocessingError,
    NativeRecordingCache,
    iter_native_windows,
    preprocess_native_recording,
    valid_notch_harmonics,
)


def _reve_contract() -> dict:
    source = {
        name: [index / 100.0, index / 200.0, index / 300.0]
        for index, name in enumerate(contracts.HUADA_CHANNEL_AXIS)
    }
    bank = contracts.build_reve_position_bank(
        source,
        source_asset_sha256=contracts.REVE_POSITION_SOURCE_SHA256,
    )
    return contracts.make_huada_target_contract("reve", reve_position_bank=bank)


def test_frozen_labram_contract_and_channel_axis_are_exact():
    path = "configs/huada_labram_target_input_contract_v1.json"
    loaded = contracts.load_huada_target_contract(path, expected_family="labram")
    assert loaded["channel_axis"] == list(contracts.HUADA_CHANNEL_AXIS)
    assert loaded["fbmae_token_lance_reused"] is False
    assert loaded["channel_mapping"]["position_semantics"] == "validity_only_not_geometry"


def test_channel_slot_equality_never_substitutes_for_semantics():
    value = contracts.make_huada_target_contract("labram")
    value["channel_axis"][0], value["channel_axis"][1] = value["channel_axis"][1], value["channel_axis"][0]
    value["contract_sha256"] = contracts.contract_sha256(value)
    with pytest.raises(contracts.NativeContractError, match="audited Huada BDF order"):
        contracts.validate_huada_target_contract(value)


def test_reve_contract_embeds_huada_ordered_bank_not_dortmund_bank():
    value = _reve_contract()
    loaded = contracts.validate_huada_target_contract(value, expected_family="reve")
    assert loaded["position_bank"]["channel_axis"] == list(contracts.HUADA_CHANNEL_AXIS)
    assert loaded["position_bank"]["dortmund_position_bank_reused"] is False
    assert loaded["position_bank"]["position_mask"] is None


def test_reve_position_bank_fingerprint_and_source_sha_fail_closed():
    value = _reve_contract()
    value["position_bank"]["positions"][0][0] += 1
    value["contract_sha256"] = contracts.contract_sha256(value)
    with pytest.raises(contracts.NativeContractError, match="fingerprint"):
        contracts.validate_huada_target_contract(value)
    value = _reve_contract()
    value["position_bank"]["source_asset_sha256"] = "0" * 64
    value["position_bank"]["fingerprint"] = contracts.position_bank_fingerprint(
        contracts.HUADA_CHANNEL_AXIS, value["position_bank"]["positions"]
    )
    value["contract_sha256"] = contracts.contract_sha256(value)
    with pytest.raises(contracts.NativeContractError, match="source asset SHA"):
        contracts.validate_huada_target_contract(value)


def _identity_resample(calls: list[dict]):
    def apply(data, **kwargs):
        calls.append(kwargs)
        target = int(round(data.shape[1] * kwargs["target_sfreq"] / kwargs["source_sfreq"]))
        return data[:, :target]

    return apply


def test_labram_native_chain_scales_once_after_full_recording_filtering():
    calls: list[dict] = []
    raw = np.full((64, 1000), 2e-6, dtype=np.float64)
    result = preprocess_native_recording(
        raw, family="labram", filter_resample=_identity_resample(calls)
    )
    assert calls == [{
        "source_sfreq": 500.0, "low_hz": 0.1, "high_hz": 75.0,
        "notch_hz": [50.0], "target_sfreq": 200.0,
    }]
    assert result.data.shape == (64, 400)
    assert result.data.dtype == np.float32
    assert np.all(result.data == np.float32(0.02))
    assert result.metadata["processing_order"] == ["bandpass", "notch", "resample", "extractor_scale"]
    assert valid_notch_harmonics(50, source_sfreq=500, bandpass_high_hz=75) == [50.0]


def test_reve_native_chain_uses_entire_recording_scaler_clip_and_no_notch():
    calls: list[dict] = []
    raw = np.tile(np.arange(1000, dtype=np.float64), (64, 1))
    raw[0] = 4.0  # StandardScaler constant-feature semantics: scale=1, output zero.
    result = preprocess_native_recording(
        raw, family="reve", filter_resample=_identity_resample(calls)
    )
    assert calls[0]["notch_hz"] is None
    assert calls[0]["low_hz"] == 0.5 and calls[0]["high_hz"] == 99.5
    assert result.metadata["scaler"]["statistics_sample_count"] == 400
    assert result.metadata["scaler"]["constant_channel_indices"] == [0]
    assert np.all(result.data[0] == 0)
    assert np.max(np.abs(result.data)) <= 15
    assert result.metadata["extractor_scale"] is None


def test_preprocessing_rejects_wrong_shape_rate_and_nonfinite():
    with pytest.raises(NativePreprocessingError, match=r"\[64,T\]"):
        preprocess_native_recording(np.zeros((63, 100)), family="labram")
    with pytest.raises(NativePreprocessingError, match="500 Hz"):
        preprocess_native_recording(np.zeros((64, 100)), family="labram", source_sfreq=200)
    value = np.zeros((64, 100)); value[0, 0] = np.nan
    with pytest.raises(NativePreprocessingError, match="finite"):
        preprocess_native_recording(value, family="labram")


def test_full_recording_cache_roundtrip_windows_and_no_overwrite(tmp_path):
    calls: list[dict] = []
    processed = preprocess_native_recording(
        np.tile(np.arange(2000, dtype=np.float64), (64, 1)),
        family="labram",
        filter_resample=_identity_resample(calls),
    )
    contract = contracts.make_huada_target_contract("labram")
    root = tmp_path / "SUB1"
    NativeRecordingCache.write(root, processed, contract=contract, recording_id="SUB1")
    cached, metadata = NativeRecordingCache.load(
        root,
        expected_family="labram",
        expected_contract_sha256=contract["contract_sha256"],
    )
    assert cached.shape == (64, 800)
    assert len(list(iter_native_windows(cached))) == 2
    assert metadata["fbmae_token_lance_reused"] is False
    with pytest.raises(FileExistsError):
        NativeRecordingCache.write(root, processed, contract=contract, recording_id="SUB1")


class _FakeLabram(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([2.0]))

    def forward(self, x, *, ch_names, return_all_tokens):
        assert len(ch_names) == 64 and return_all_tokens is True
        base = x.mean(dim=(1, 2), keepdim=False)[:, None, None] * self.weight
        return base.expand(-1, 129, 200)


class _FakeReve(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([3.0]))
        self.last_pos = None

    def forward(self, eeg, *, pos, return_output):
        assert return_output is True
        self.last_pos = pos.detach().clone()
        base = eeg.mean(dim=(1, 2), keepdim=False)[:, None, None] * self.weight
        return [base.expand(-1, 128, 512)]


def _checkpoint(family: str, inner: nn.Module) -> PortableAgeCheckpoint:
    state = dict(inner.state_dict())
    if family == "labram":
        state.update({
            "channel_wrapper._positionless_mask": torch.ones(64, dtype=torch.bool),
            # Deliberately false: target backend must not reuse this source adapter.
            "channel_wrapper._known_mask": torch.zeros(64, dtype=torch.bool),
        })
        feature = 200
    else:
        feature = 512
    payload = {
        "model": {"model_name": f"fake_{family}", "model_family": family},
        "backbone_state_dict": state,
        "age_probe_state_dict": {
            "weight": torch.ones(1, feature), "bias": torch.tensor([0.5])
        },
        "target_transform": {"mean": 27.5, "std": 4.218591213226318},
    }
    return PortableAgeCheckpoint(payload=payload)


def test_labram_backend_rebuilds_huada_adapter_and_outputs_age():
    source = _FakeLabram()
    checkpoint = _checkpoint("labram", source)
    backend = HuadaLaBraMAgeBackend.build(
        checkpoint,
        contracts.make_huada_target_contract("labram"),
        model_factory=_FakeLabram,
    )
    assert backend.wrapped_model._known_mask.all()
    result = backend(torch.ones(2, 64, 400))
    assert result["backbone_tokens"].shape == (2, 129, 200)
    assert result["representation"].shape == (2, 200)
    assert result["predicted_z"].shape == (2, 1)
    assert result["predicted_age_years"].dtype == torch.float32


def test_reve_backend_injects_huada_position_bank():
    source = _FakeReve()
    checkpoint = _checkpoint("reve", source)
    contract = _reve_contract()
    backend = HuadaREVEAgeBackend.build(
        checkpoint, contract, model_factory=_FakeReve
    )
    result = backend(torch.ones(2, 64, 400))
    assert result["backbone_tokens"].shape == (2, 128, 512)
    assert backend.wrapped_model.model.last_pos.shape == (2, 64, 3)
    expected = torch.tensor(contract["position_bank"]["positions"])
    assert torch.equal(backend.wrapped_model.model.last_pos[0], expected)


def test_backend_rejects_fbmae_window_shape_and_nonfinite():
    backend = HuadaLaBraMAgeBackend.build(
        _checkpoint("labram", _FakeLabram()),
        contracts.make_huada_target_contract("labram"),
        model_factory=_FakeLabram,
    )
    with pytest.raises(Exception, match=r"\[B,64,400\]"):
        backend(torch.zeros(1, 64, 2000))
    value = torch.zeros(1, 64, 400); value[0, 0, 0] = torch.nan
    with pytest.raises(Exception, match="finite float32"):
        backend(value)
