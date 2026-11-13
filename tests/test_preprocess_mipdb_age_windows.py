from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import preprocess_mipdb_age_windows as mipdb
from preprocess_huada_common_windows import _prepare_output_dir, _write_lance_batch


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "mipdb_age_preprocessing.toml"


def cache_row(sample_id: str = "sample-1") -> dict:
    return {
        "sample_id": sample_id,
        "subject_id": "sub-A00000001",
        "recording_id": "sub-A00000001_task-block01_eeg",
        "block_id": "block01",
        "source_path": "sub-A00000001/eeg/a.vhdr",
        "age": np.float32(25),
        "label": np.float32(25),
        "task_type": "regression",
        "task_semantics": "unknown",
        "resting_state_confirmed": False,
        "block01_candidate": True,
        "header_signature": "abcdef0123456789",
        "window_index": 0,
        "start_sample": 0,
        "start_seconds": 0.0,
        "target_start_sample": 0,
        "duration": 10.0,
        "modality": "eeg",
        "data": np.zeros(mipdb.DATA_LENGTH, dtype=np.float32),
        "shape": [128, 2000],
        "original_shape": [128, 2000],
        "valid_length": mipdb.DATA_LENGTH,
        "channel_names": mipdb.CORE_CHANNELS,
        "channel_mask": np.ones(128, dtype=bool),
        "bad_channel_mask": np.zeros(128, dtype=bool),
        "channel_status": ["good"] * 128,
        "bad_channel_count": 0,
        "qc_pass": True,
        "clip_fraction": np.float32(0),
        "positive_clip_fraction": np.float32(0),
        "negative_clip_fraction": np.float32(0),
        "pre_normalization_abs_max_uv": np.float32(0),
        "source_sampling_rate": np.float32(500),
        "target_sampling_rate": np.float32(200),
        "source_reference": "unknown",
        "processing_reference": "unchanged",
        "preprocessing_config_hash": "abc",
    }


def write_discovery_recording(data_dir: Path, subject: str, block: str) -> None:
    eeg = data_dir / subject / "eeg"
    eeg.mkdir(parents=True, exist_ok=True)
    prefix = f"{subject}_task-{block}"
    (eeg / f"{prefix}_eeg.vhdr").write_text("header", encoding="utf-8")
    (eeg / f"{prefix}_eeg.eeg").write_bytes(b"binary")
    (eeg / f"{prefix}_eeg.vmrk").write_text("marker", encoding="utf-8")
    (eeg / f"{prefix}_eeg.json").write_text(
        json.dumps({"PowerLineFrequency": "n/a"}), encoding="utf-8"
    )
    (eeg / f"{prefix}_channels.tsv").write_text(
        "name\ttype\tunits\nE1\tEEG\tµV\n", encoding="utf-8"
    )


def test_config_records_explicit_sampled_psd_60_hz_decision():
    config = mipdb.load_mipdb_config(CONFIG)
    assert config["preprocessing"]["notch_hz"] == 60.0
    assert config["line_frequency"]["selected_hz"] == 60.0
    assert config["line_frequency"]["decision"] == "sampled_psd"
    assert config["line_frequency"]["aggregate_60hz_median_ratio"] > 10


def test_config_rejects_unresolved_notch(tmp_path):
    text = CONFIG.read_text(encoding="utf-8").replace("notch_hz = 60.0", "notch_hz = 0.0")
    path = tmp_path / "bad.toml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="50 or 60"):
        mipdb.load_mipdb_config(path)


def test_core_channels_are_exact_native_e1_to_e128_and_aux_excluded():
    names = [*mipdb.CORE_CHANNELS, "E129", "Status"]
    types = ["eeg"] * 128 + ["eeg", "stim"]
    core, aux = mipdb.resolve_core_channels(names, types)
    assert core == list(range(128))
    assert aux == [128, 129]
    assert [names[index] for index in core] == mipdb.CORE_CHANNELS


def test_core_channel_missing_or_reordered_fails_closed():
    with pytest.raises(RuntimeError, match="missing core"):
        mipdb.resolve_core_channels(mipdb.CORE_CHANNELS[:-1], ["eeg"] * 127)
    names = mipdb.CORE_CHANNELS.copy()
    names[0], names[1] = names[1], names[0]
    with pytest.raises(RuntimeError, match="not canonical"):
        mipdb.resolve_core_channels(names, ["eeg"] * 128)


@pytest.mark.parametrize("value", [None, "n/a", "unknown", ""])
def test_unknown_sidecar_power_line_is_not_guessed(value):
    assert mipdb.parse_power_line_frequency(value) is None


def test_exact_age_metadata_required(tmp_path):
    metadata = tmp_path / "participants.csv"
    metadata.write_text("participant_id,age\nA1,20-25\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exact numeric age"):
        mipdb.load_age_metadata(metadata)


def test_dynamic_discovery_has_no_subject_or_recording_count_gate(tmp_path):
    data = tmp_path / "data"
    write_discovery_recording(data, "sub-A00000001", "block01")
    write_discovery_recording(data, "sub-A00000001", "block02")
    write_discovery_recording(data, "sub-A00000002", "block01")
    metadata = tmp_path / "ages.csv"
    metadata.write_text(
        "participant_id,age,sex\nA00000001,25,M\nA00000002,30,F\n", encoding="utf-8"
    )
    allowlist = tmp_path / "allowlist.txt"
    allowlist.write_text("A00000001\nA00000002\n", encoding="utf-8")
    args = mipdb.parse_args([
        "--data-dir", str(data), "--config", str(CONFIG),
        "--output-dir", str(tmp_path / "out"), "--age-metadata", str(metadata),
        "--subject-allowlist", str(allowlist),
    ])
    pipeline = mipdb.MIPDBPipeline(args)
    pipeline.build_manifests()
    assert len(pipeline.subjects) == 2
    assert len(pipeline.recordings) == 3
    assert pipeline.subjects["sub-A00000001"]["recording_count"] == 2


def test_smoke_requires_two_or_three_subjects(tmp_path):
    base = [
        "--data-dir", str(tmp_path), "--config", str(CONFIG),
        "--output-dir", str(tmp_path / "out"),
        "--age-metadata", str(tmp_path / "ages.csv"), "--smoke",
    ]
    with pytest.raises(SystemExit):
        mipdb.parse_args([*base, "--smoke-subjects", "sub-A1"])


def test_schema_and_lance_roundtrip_are_128_by_2000(tmp_path):
    schema = mipdb.mipdb_lance_schema()
    assert schema.field("data").type.value_type == mipdb.pa.float32()
    assert schema.field("bad_channel_mask").type.list_size == 128
    path = tmp_path / "dataset.lance"
    _write_lance_batch(path, [cache_row()], True, schema)
    result = mipdb.validate_mipdb_lance(path, 1)
    assert result["all_rows_valid"] is True


def test_output_directory_cannot_overwrite_without_explicit_flag(tmp_path):
    data = tmp_path / "raw"
    data.mkdir()
    output = tmp_path / "out"
    output.mkdir()
    (output / "prior.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        _prepare_output_dir(output, False, data)


def test_full_entrypoint_is_dynamic_and_does_not_use_dortmund_count_gates():
    text = (ROOT / "commands" / "run_mipdb_age_preprocessing_full.sh").read_text(encoding="utf-8")
    assert "preprocess_mipdb_age_windows.py" in text
    assert "mipdb_age_preprocessing.toml" in text
    assert "--age-metadata" in text and "--subject-allowlist" in text
    assert "EXPECTED_SUBJECTS" not in text
    assert "EXPECTED_RECORDINGS" not in text
    assert "--overwrite" not in text
    assert "coscli" not in text.lower()


def test_window_identity_and_profile_semantics_are_explicit():
    sample_id = mipdb.make_sample_id(
        "sub-A00000001", "sub-A00000001_task-block01_eeg", 3, 15000
    )
    assert sample_id.endswith("__w00003__s000015000")
    row = cache_row(sample_id)
    assert row["task_semantics"] == "unknown"
    assert row["resting_state_confirmed"] is False
    assert row["block01_candidate"] is True
