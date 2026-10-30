from __future__ import annotations

import inspect
import sys
from pathlib import Path

import lance
import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import build_huada_model_tokens as tokens  # noqa: E402


def make_row(subject_id: str = "SUB1", bad_channel: str | None = None) -> dict:
    time = np.arange(2000, dtype=np.float32) / 200.0
    data = np.stack([
        0.1 * np.sin(2 * np.pi * (2.0 + (index % 20)) * time) + 0.01 * np.cos(2 * np.pi * 0.5 * time)
        for index in range(64)
    ]).astype(np.float32)
    bad_mask = np.array([name == bad_channel for name in tokens.STANDARD_64_CHANNELS], dtype=bool)
    data[bad_mask] = 0.0
    return {
        "sample_id": f"{subject_id}__recording01__w00000",
        "subject_id": subject_id,
        "source_path": f"{subject_id}/recording01.bdf",
        "recording_id": "recording01",
        "window_index": 0,
        "start_time": 0.0,
        "data": data.reshape(-1),
        "shape": [64, 2000],
        "channel_names": list(tokens.STANDARD_64_CHANNELS),
        "channel_mask": np.ones(64, dtype=bool),
        "bad_channel_mask": bad_mask,
        "qc_pass": True,
        "age": None if subject_id == "SUB5" else 31,
        "sex": "male",
        "source_sampling_rate": 500.0,
        "target_sampling_rate": 200.0,
        "preprocessing_config_hash": "abc123",
    }


def test_two_repository_data_contracts_are_identical():
    result = tokens.validate_reference_contracts()
    assert result["data_contract_equal"] is True
    assert tokens.REFERENCE_AUDIT["50m"]["data_config"] == tokens.REFERENCE_AUDIT["1b"]["data_config"]


def test_derived_geometry_and_patch_shapes():
    assert tokens.TOTAL_POINTS == 1000
    assert tokens.PATCH_LEN == tokens.PATCH_STRIDE == 100
    assert tokens.NUM_TIME_PATCHES == 10
    signal = np.zeros((64, 1000), dtype=np.float32)
    patches = tokens.cut_patches(signal)
    assert patches.shape == (64, 10, 100)
    assert patches.reshape(640, 100).shape == (640, 100)


def test_resample_200_to_100_produces_1000_points():
    signal = np.arange(64 * 2000, dtype=np.float32).reshape(64, 2000)
    assert tokens.resample_signal(signal).shape == (64, 1000)


def test_aliases_and_public_repository_standard_order():
    assert tokens.canonical_channel_name("EEG FP1-Ref") == "Fp1"
    assert tokens.canonical_channel_name("T3") == "T7"
    assert tokens.canonical_channel_name("Cz.") == "Cz"
    assert tokens.canonical_channel_name("A1") == "A1"
    assert len(tokens.STANDARD_64_CHANNELS) == 64
    assert tokens.STANDARD_64_CHANNELS[-3:] == ["Iz", "F9", "F10"]


def test_channel_mask_and_bad_mask_define_validity():
    row = make_row("SUB5", "FC1")
    signal = np.asarray(row["data"], np.float32).reshape(64, 2000) * 200.0
    aligned, valid, audit = tokens.align_to_standard_channels(
        signal, row["channel_names"], row["channel_mask"], row["bad_channel_mask"]
    )
    fc1 = tokens.STANDARD_64_CHANNELS.index("FC1")
    assert audit["source_order_matches_standard"] is True
    assert valid.sum() == 63
    assert valid[fc1] == 0
    assert np.all(aligned[fc1] == 0)


def test_unknown_missing_and_duplicates_are_recorded_not_guessed():
    row = make_row()
    names = list(row["channel_names"])
    names[0] = "UNKNOWN"
    names[1] = "Fp2"  # duplicate Fp2
    signal = np.asarray(row["data"], np.float32).reshape(64, 2000)
    _, valid, audit = tokens.align_to_standard_channels(signal, names, row["channel_mask"], row["bad_channel_mask"])
    assert "UNKNOWN" in audit["unknown_channels"]
    assert "Fp1" in audit["missing_channels"]
    assert "Fp2" in audit["duplicates"]
    assert valid[tokens.STANDARD_64_CHANNELS.index("Fp2")] == 0


@pytest.fixture(scope="module")
def processed_sub5() -> tuple[dict, dict]:
    return tokens.process_row(make_row("SUB5", "FC1"))


def test_sub5_fc1_produces_ten_invalid_zero_tokens(processed_sub5):
    processed, audit = processed_sub5
    inputs = processed["token_inputs"].reshape(640, 100)
    targets = processed["targets"].reshape(640, 5, 5)
    token_mask = processed["token_valid_mask"]
    fc1 = tokens.STANDARD_64_CHANNELS.index("FC1")
    fc1_slice = slice(fc1 * 10, (fc1 + 1) * 10)
    assert audit["valid_channel_count"] == 63
    assert token_mask.sum() == 630
    assert np.all(token_mask[fc1_slice] == 0)
    assert np.all(inputs[fc1_slice] == 0)
    assert np.all(targets[fc1_slice] == 0)


def test_processed_shapes_channel_major_order_and_finiteness(processed_sub5):
    processed, _ = processed_sub5
    assert processed["token_inputs"].reshape(640, 100).shape == (640, 100)
    assert processed["targets"].reshape(640, 5, 5).shape == (640, 5, 5)
    np.testing.assert_array_equal(processed["token_channel_indices"], np.repeat(np.arange(64), 10))
    np.testing.assert_array_equal(processed["token_time_indices"], np.tile(np.arange(10), 64))
    assert np.isfinite(processed["token_inputs"]).all()
    assert np.isfinite(processed["targets"]).all()


def test_valid_channel_zscore_is_zero_mean_unit_std():
    rng = np.random.default_rng(42)
    signal = rng.normal(size=(64, 1000)).astype(np.float32)
    valid = np.ones(64, dtype=np.float32)
    valid[7] = 0
    standardized = tokens.standardize_signal(signal, valid)
    np.testing.assert_allclose(standardized[valid.astype(bool)].mean(axis=1), 0.0, atol=2e-6)
    np.testing.assert_allclose(standardized[valid.astype(bool)].std(axis=1), 1.0, atol=2e-6)
    assert np.all(standardized[7] == 0)


def test_normalized_input_is_multiplied_by_200_not_one_million(monkeypatch):
    row = make_row()
    row["data"] = np.full(64 * 2000, 0.25, dtype=np.float32)
    captured = {}
    original = tokens.align_to_standard_channels

    def capture(signal, *args, **kwargs):
        captured["maximum"] = float(np.max(signal))
        return original(signal, *args, **kwargs)

    monkeypatch.setattr(tokens, "align_to_standard_channels", capture)
    tokens.process_row(row)
    assert captured["maximum"] == pytest.approx(50.0)


def test_process_row_does_not_call_common_cache_operations():
    source = inspect.getsource(tokens.process_row)
    assert "resample_signal" in source
    assert "bandpass" not in source
    assert "notch" not in source
    assert "window_slices" not in source
    assert "1e6" not in source.lower()


def test_schema_preserves_ids_nullable_age_and_forbids_supervised_columns():
    schema = tokens.output_schema()
    assert schema.field("sample_id").type == tokens.pa.string()
    assert schema.field("age").nullable is True
    assert not {"label", "label_name", "split"}.intersection(schema.names)
    assert schema.field("token_inputs").type.list_size == 64000
    assert schema.field("targets").type.list_size == 16000


def test_lance_roundtrip_preserves_shape_dtype_and_mask(tmp_path, processed_sub5):
    processed, _ = processed_sub5
    output = tmp_path / "roundtrip.lance"
    lance.write_dataset(tokens.rows_to_table([processed]), output, mode="create", data_storage_version="stable")
    result = tokens.validate_roundtrip(output, 1)
    assert result["rows"] == 1
    row = lance.dataset(output).to_table().to_pylist()[0]
    assert np.asarray(row["token_inputs"], dtype=np.float32).shape == (64000,)
    assert np.asarray(row["targets"], dtype=np.float32).shape == (16000,)
    np.testing.assert_array_equal(row["token_valid_mask"], processed["token_valid_mask"])


def test_same_processed_row_satisfies_both_model_geometry_contracts(processed_sub5):
    processed, _ = processed_sub5
    for key in ("50m", "1b"):
        contract = tokens._contract(key)
        assert contract["input_dim"] == processed["patch_len"]
        assert contract["n_channels"] == processed["n_channels"]
        assert contract["num_tokens"] == processed["num_tokens"]
    assert tokens._contract("50m")["d_model"] != tokens._contract("1b")["d_model"]


def test_public_repo_montage_does_not_alias_huada_extra_channels():
    assert not {"PO5", "PO6", "CB1", "CB2"}.intersection(tokens.STANDARD_64_CHANNELS)
    assert {"AFz", "Iz", "F9", "F10"}.issubset(tokens.STANDARD_64_CHANNELS)


def test_source_snapshot_detects_no_read_only_change(tmp_path):
    source = tmp_path / "source.lance"
    source.mkdir()
    marker = source / "marker"
    marker.write_text("unchanged", encoding="utf-8")
    before = tokens.source_snapshot(source)
    after = tokens.source_snapshot(source)
    assert before == after
