from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

import lance
import numpy as np
import pytest
import tomllib


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import adapt_dortmund_fbmae as adapter  # noqa: E402
import build_huada_model_tokens as huada_tokens  # noqa: E402
from preprocess_dortmund_age_windows import DORTMUND_CHANNELS, dortmund_lance_schema  # noqa: E402
from preprocess_huada_common_windows import _write_lance_batch  # noqa: E402


PROFILE = ROOT / "configs" / "dortmund_fbmae_channel_profile.toml"
VOCAB = ROOT / "configs" / "fbmae_electrode_vocab_snapshot.json"


def common_row(subject="sub-020", age=20.0, condition="eyes_open", window=0, bad_name=None):
    time = np.arange(2000, dtype=np.float32) / 200.0
    data = np.stack([
        0.08 * np.sin(2 * np.pi * (2.0 + index % 25) * time)
        + 0.01 * np.cos(2 * np.pi * 0.5 * time)
        for index in range(64)
    ]).astype(np.float32)
    bad = np.asarray([name == bad_name for name in DORTMUND_CHANNELS], dtype=bool)
    data[bad] = 0
    recording = f"{subject}_{condition}_pre"
    return {
        "sample_id": f"ds005385__{subject}__ses-1__{condition}__acq-pre__{recording}__w{window:05d}__s{window * 10000:09d}",
        "subject_id": subject, "session_id": "ses-1", "run_id": "run-01",
        "recording_id": recording, "modality": "eeg", "data": data.reshape(-1),
        "shape": [64, 2000], "original_shape": [64, 2000], "valid_length": 128000,
        "qc_pass": True, "label": np.float32(age), "age": np.float32(age),
        "task_type": "regression", "target_name": "age", "target_unit": "year",
        "eyes_condition": condition, "pre_or_post": "pre_cognitive",
        "source_path": f"{subject}/ses-1/{recording}.edf", "start_time": float(window * 10),
        "duration": 10.0, "source_start_sample": window * 10000,
        "target_start_sample": window * 2000, "channel_names": list(DORTMUND_CHANNELS),
        "channel_mask": np.ones(64, dtype=bool), "bad_channel_mask": bad,
        "channel_status": ["bad" if value else "good" for value in bad],
        "channel_layout": "dataset_level", "bad_channel_count": int(bad.sum()),
        "clip_fraction": np.float32(0), "positive_clip_fraction": np.float32(0),
        "negative_clip_fraction": np.float32(0), "pre_normalization_abs_max_uv": np.float32(16),
        "source_sampling_rate": np.float32(1000), "target_sampling_rate": np.float32(200),
        "preprocessing_config_hash": "fixture-config-hash",
    }


def write_input_fixture(root: Path, windows=2) -> tuple[Path, list[dict]]:
    root.mkdir()
    rows = []
    subjects = [("sub-020", 20.0, "female"), ("sub-270", 27.0, "male"), ("sub-350", 35.0, "female")]
    for subject, age, _ in subjects:
        for condition in ("eyes_open", "eyes_closed"):
            for window in range(windows):
                bad = "FC1" if subject == "sub-270" and condition == "eyes_closed" and window == 0 else None
                rows.append(common_row(subject, age, condition, window, bad))
    lance_path = root / "dataset.lance"
    _write_lance_batch(lance_path, rows, True, dortmund_lance_schema())
    with (root / "subject_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["subject_id", "age", "sex"])
        writer.writeheader()
        for subject, age, sex in subjects:
            writer.writerow({"subject_id": subject, "age": age, "sex": sex})
    return lance_path, rows


def resolved_profile():
    return adapter.load_channel_profile(PROFILE, VOCAB)


def test_profile_parses_and_all_64_native_channels_have_unique_authoritative_ids():
    profile, mapping, summary, model_channels = resolved_profile()
    assert profile["source"]["channel_names"] == DORTMUND_CHANNELS
    assert len(mapping) == len({row["electrode_id"] for row in mapping}) == 64
    assert summary["channel_mapping_complete"] and summary["unknown_channel_count"] == 0
    assert model_channels == huada_tokens.STANDARD_64_CHANNELS
    assert model_channels[-3:] == ["Iz", "F9", "F10"]


def test_unknown_channel_fails_closed(tmp_path):
    text = PROFILE.read_text(encoding="utf-8").replace('"TP9", "CP5"', '"NOT_REAL", "CP5"')
    path = tmp_path / "unknown.toml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="unknown=.*NOT_REAL"):
        adapter.load_channel_profile(path, VOCAB)


def test_duplicate_electrode_id_fails_closed(tmp_path):
    profile = tomllib.loads(PROFILE.read_text(encoding="utf-8"))
    profile["source"]["channel_names"][17] = "TP9"
    profile["source"]["electrode_ids"][17] = profile["source"]["electrode_ids"][16]
    # Small, explicit TOML mutation is easier by replacing the relevant adjacent source entries/IDs.
    text = PROFILE.read_text(encoding="utf-8").replace('"TP9", "CP5"', '"TP9", "TP9"', 1)
    text = text.replace("49, 452, 423", "49, 49, 423", 1)
    path = tmp_path / "duplicate.toml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate_ids"):
        adapter.load_channel_profile(path, VOCAB)


def test_mapping_is_name_based_not_huada_position_and_reference_is_not_fabricated():
    _, mapping, summary, model_channels = resolved_profile()
    by_name = {row["source_channel_name"]: row for row in mapping}
    assert by_name["Fp2"]["source_index"] == 1
    assert by_name["Fp2"]["fbmae_standard_slot"] == model_channels.index("Fp2") == 2
    assert by_name["TP9"]["fbmae_standard_slot"] == ""
    assert "FCz" not in DORTMUND_CHANNELS and "FCz" in summary["missing_fixed_fbmae_slots"]
    assert summary["positional_huada_mapping_used"] is False


def test_status_excluded_and_reference_policy_unchanged():
    profile, _, summary, _ = resolved_profile()
    assert "Status" not in profile["source"]["channel_names"]
    reference = summary["reference"]
    assert reference["source_reference"] == "FCz" and reference["ground"] == "AFz"
    assert reference["rereference_applied"] is False
    assert reference["processing_reference"] == "unchanged"
    assert reference["reference_channel_present_in_data"] is False


def test_token_row_preserves_metadata_age_and_masks_bad_channel_to_zero():
    profile, _, _, model_channels = resolved_profile()
    source_ids = profile["resolved_source_electrode_ids"]
    model_ids = profile["resolved_model_electrode_ids"]
    source = common_row("sub-270", 27, "eyes_closed", 1, "FC1")
    row, audit = adapter.build_token_row(source, model_channels, source_ids, model_ids, "male")
    fc1 = model_channels.index("FC1")
    token_slice = slice(fc1 * 10, (fc1 + 1) * 10)
    assert row["sample_id"] == source["sample_id"] and row["recording_id"] == source["recording_id"]
    assert row["age"] == row["label"] == 27 and row["eyes_condition"] == "eyes_closed"
    assert row["session_id"] == "ses-1" and row["run_id"] == "run-01" and row["acquisition"] == "pre"
    assert row["task_type"] == "regression" and row["target_unit"] == "year"
    assert row["source_bad_channel_mask"][DORTMUND_CHANNELS.index("FC1")]
    assert np.all(row["token_valid_mask"][token_slice] == 0)
    assert np.all(row["token_inputs"].reshape(640, 100)[token_slice] == 0)
    assert np.all(row["targets"].reshape(640, 5, 5)[token_slice] == 0)
    assert audit["valid_channel_count"] == 57  # six fixed slots absent + bad FC1
    assert np.array_equal(row["token_electrode_ids"], np.repeat(row["electrode_ids"], 10))


def test_eo_ec_remain_separate_and_smoke_selection_limits_each_recording():
    rows = [common_row("sub-020", 20, condition, window) for condition in ("eyes_open", "eyes_closed") for window in range(3)]
    indices, detail = adapter.choose_smoke_indices(rows, ["sub-020"], 2)
    selected = [rows[index] for index in indices]
    assert len(selected) == 4 and {row["eyes_condition"] for row in selected} == {"eyes_open", "eyes_closed"}
    assert set(detail["recording_counts"].values()) == {2}
    with pytest.raises(ValueError, match="absent"):
        adapter.choose_smoke_indices(rows, ["sub-does-not-exist"], 2)


def test_huada_default_path_has_no_regression_from_channel_parameterization():
    source = {
        "sample_id": "SUB1__r__w000", "subject_id": "SUB1", "source_path": "x.bdf",
        "recording_id": "r", "window_index": 0, "start_time": 0.0,
        "data": common_row()["data"], "shape": [64, 2000],
        "channel_names": list(huada_tokens.STANDARD_64_CHANNELS),
        "channel_mask": np.ones(64, bool), "bad_channel_mask": np.zeros(64, bool),
        "qc_pass": True, "age": 30, "sex": "female", "target_sampling_rate": 200.0,
        "preprocessing_config_hash": "x",
    }
    default, _ = huada_tokens.process_row(source)
    explicit, _ = huada_tokens.process_row(source, standard_channels=huada_tokens.STANDARD_64_CHANNELS)
    np.testing.assert_array_equal(default["token_inputs"], explicit["token_inputs"])
    np.testing.assert_array_equal(default["targets"], explicit["targets"])


def test_end_to_end_fixture_smoke_audits_roundtrip_loader_and_source_protection(tmp_path):
    input_dir = tmp_path / "input"
    lance_path, source_rows = write_input_fixture(input_dir, windows=2)
    before = adapter.snapshot_tree(input_dir)
    output = tmp_path / "output"
    code = adapter.main([
        "--input-lance", str(lance_path), "--input-dir", str(input_dir),
        "--output-dir", str(output), "--source-uri", "cos://fixture/dortmund/",
        "--channel-profile", str(PROFILE), "--electrode-vocab", str(VOCAB),
        "--smoke", "--smoke-subjects", "sub-020,sub-270,sub-350",
        "--smoke-max-windows-per-recording", "1", "--batch-size", "2",
        "--expected-subjects", "3", "--expected-recordings", "6",
    ])
    assert code == 0 and (output / "_SUCCESS").exists()
    assert adapter.snapshot_tree(input_dir) == before
    token_ds = lance.dataset(output / "dortmund_fbmae_tokens.lance")
    assert token_ds.count_rows() == 6
    assert "split" not in token_ds.schema.names
    rows = token_ds.to_table().to_pylist()
    expected_ids = {
        row["sample_id"] for row in source_rows if row["target_start_sample"] == 0
    }
    assert {row["sample_id"] for row in rows} == expected_ids
    assert all(len(row["token_inputs"]) == 64000 and len(row["targets"]) == 16000 for row in rows)
    summary = json.loads((output / "adapter_summary.json").read_text(encoding="utf-8"))
    assert summary["adapter_ready"] is False and summary["smoke_passed"] is True
    assert summary["stages"]["token_smoke"] == "passed"
    assert summary["stages"]["full_conversion"] == "not_performed"
    assert summary["token_validation"]["loader_batch_validation"] == "passed"
    assert summary["source_data_unchanged"] is True
    assert adapter.REQUIRED_OUTPUTS.issubset({item.name for item in output.iterdir()})


def test_batch_size_does_not_change_selected_token_rows(tmp_path):
    input_dir = tmp_path / "input"
    lance_path, _ = write_input_fixture(input_dir, windows=1)
    common = [
        "--input-lance", str(lance_path), "--input-dir", str(input_dir),
        "--source-uri", "cos://fixture/dortmund/", "--channel-profile", str(PROFILE),
        "--electrode-vocab", str(VOCAB), "--smoke", "--smoke-subjects", "sub-020",
        "--smoke-max-windows-per-recording", "1", "--expected-subjects", "3",
        "--expected-recordings", "6",
    ]
    out1, out2 = tmp_path / "batch1", tmp_path / "batch4"
    assert adapter.main([*common, "--output-dir", str(out1), "--batch-size", "1"]) == 0
    assert adapter.main([*common, "--output-dir", str(out2), "--batch-size", "4", "--num-workers", "2"]) == 0
    a = lance.dataset(out1 / "dortmund_fbmae_tokens.lance").to_table().to_pylist()
    b = lance.dataset(out2 / "dortmund_fbmae_tokens.lance").to_table().to_pylist()
    assert [row["sample_id"] for row in a] == [row["sample_id"] for row in b]
    for left, right in zip(a, b):
        np.testing.assert_array_equal(left["token_inputs"], right["token_inputs"])
        np.testing.assert_array_equal(left["targets"], right["targets"])


def test_input_and_output_directory_guards(tmp_path):
    input_dir = tmp_path / "input"
    lance_path, _ = write_input_fixture(input_dir, windows=1)
    existing = tmp_path / "existing"
    existing.mkdir()
    base = ["--input-lance", str(lance_path), "--input-dir", str(input_dir), "--source-uri", "cos://fixture/"]
    with pytest.raises(FileExistsError):
        adapter.main([*base, "--output-dir", str(existing)])
    with pytest.raises(ValueError, match="input directory or its child"):
        adapter.main([*base, "--output-dir", str(input_dir / "forbidden")])


def test_auto_discovery_fails_closed_on_multiple_formal_lance_datasets(tmp_path):
    first, _ = write_input_fixture(tmp_path / "first", windows=1)
    second = tmp_path / "second" / "dataset.lance"
    second.parent.mkdir()
    shutil.copytree(first, second)
    with pytest.raises(RuntimeError, match="exactly one"):
        adapter.discover_input_lance(tmp_path)
