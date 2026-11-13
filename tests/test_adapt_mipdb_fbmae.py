from __future__ import annotations

import json
import sys
from pathlib import Path

import lance
import numpy as np
import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import adapt_mipdb_fbmae as adapter
import build_huada_model_tokens as token_core
import fbmae_training_profiles as training_profiles
import preprocess_mipdb_age_windows as common
from preprocess_huada_common_windows import _write_lance_batch


ROOT = Path(__file__).resolve().parents[1]
MAPPING = ROOT / "configs" / "mipdb_egi128_to_standard64_sparse18_v1.json"
PROFILES = ROOT / "configs" / "mipdb_dortmund_fbmae_training_profiles_v1.json"
VOCAB = ROOT / "configs" / "fbmae_electrode_vocab_snapshot.json"


def common_row(*, bad_channel: str | None = None, clip_fraction: float = 0.25) -> dict:
    rng = np.random.default_rng(7)
    data = rng.normal(0, 0.1, size=(128, 2000)).astype(np.float32)
    bad = np.asarray([name == bad_channel for name in adapter.SOURCE_CHANNELS], dtype=bool)
    data[bad] = 0.0
    return {
        "sample_id": "nm000153__sub-A1__sub-A1_task-block08_eeg__w00000__s000000000",
        "subject_id": "sub-A1", "recording_id": "sub-A1_task-block08_eeg",
        "block_id": "block08", "source_path": "sub-A1/eeg/a.vhdr",
        "age": np.float32(25), "label": np.float32(25), "task_type": "regression",
        "task_semantics": "unknown", "resting_state_confirmed": False,
        "block01_candidate": False, "header_signature": "abcdef0123456789",
        "window_index": 0, "start_sample": 0, "start_seconds": 0.0,
        "target_start_sample": 0, "duration": 10.0, "modality": "eeg",
        "data": data.reshape(-1), "shape": [128, 2000], "original_shape": [128, 2000],
        "valid_length": 256000, "channel_names": list(adapter.SOURCE_CHANNELS),
        "channel_mask": np.ones(128, dtype=bool), "bad_channel_mask": bad,
        "channel_status": ["bad" if value else "good" for value in bad],
        "bad_channel_count": int(bad.sum()), "qc_pass": True,
        "clip_fraction": np.float32(clip_fraction), "positive_clip_fraction": np.float32(0),
        "negative_clip_fraction": np.float32(0), "pre_normalization_abs_max_uv": np.float32(100),
        "source_sampling_rate": np.float32(500), "target_sampling_rate": np.float32(200),
        "source_reference": "unknown", "processing_reference": "unchanged",
        "preprocessing_config_hash": "config-hash",
    }


def resolved():
    mapping = adapter.load_mapping_profile(MAPPING)
    mapping_hash = adapter.canonical_json_sha256(mapping)
    source_ids, target_ids = adapter.load_electrode_ids(VOCAB)
    return mapping, mapping_hash, source_ids, target_ids


def test_frozen_sparse18_mapping_and_fingerprints():
    mapping, _, source_ids, target_ids = resolved()
    assert mapping["profile_name"] == "mipdb_egi128_to_standard64_sparse18_v1"
    assert len(mapping["mappings"]) == 18
    assert mapping["mappings"][0] == {"source": "E22", "target": "Fp1"}
    assert mapping["mappings"][-1] == {"source": "E124", "target": "F4"}
    assert len(source_ids) == len(set(source_ids)) == 128
    assert len(target_ids) == len(set(target_ids)) == 64
    assert adapter.sequence_fingerprint(adapter.SOURCE_CHANNELS) == "f6faffa66d984463392dca61ec8c2b5a05ff06a8a16f449f0927ca0a582708c4"
    assert adapter.sequence_fingerprint(token_core.STANDARD_64_CHANNELS) == "d2dd8d3c5247572011f280244754eed6f1b621ab0857122f355cb55a4bb7b74a"


def test_unmapped_real_bad_channels_are_excluded_without_target_bad_propagation():
    mapping, mapping_hash, source_ids, target_ids = resolved()
    for bad_channel in ("E119", "E56"):
        processed, audit = adapter.build_token_row(
            common_row(bad_channel=bad_channel), mapping, mapping_hash, source_ids, target_ids
        )
        assert audit["source_bad_channels_excluded_by_mapping"] == bad_channel
        assert audit["source_mapped_bad_channels"] == ""
        assert audit["target_bad_count"] == 0
        assert np.asarray(processed["target_bad_channel_mask"], bool).sum() == 0
        assert np.asarray(processed["channel_valid_mask"], bool).sum() == 18
        assert np.asarray(processed["token_valid_mask"], bool).sum() == 180


def test_synthetic_mapped_bad_channel_invalidates_exactly_ten_tokens():
    mapping, mapping_hash, source_ids, target_ids = resolved()
    processed, audit = adapter.build_token_row(
        common_row(bad_channel="E22"), mapping, mapping_hash, source_ids, target_ids
    )
    fp1 = token_core.STANDARD_64_CHANNELS.index("Fp1")
    token_slice = slice(fp1 * 10, (fp1 + 1) * 10)
    assert audit["source_mapped_bad_channels"] == "E22"
    assert audit["target_bad_count"] == 1 and audit["valid_channel_count"] == 17
    assert np.asarray(processed["target_channel_mask"], bool)[fp1]
    assert np.asarray(processed["target_bad_channel_mask"], bool)[fp1]
    assert np.all(np.asarray(processed["token_valid_mask"])[token_slice] == 0)
    assert np.all(np.asarray(processed["token_inputs"]).reshape(640, 100)[token_slice] == 0)
    assert np.all(np.asarray(processed["targets"]).reshape(640, 5, 5)[token_slice] == 0)
    assert np.asarray(processed["token_valid_mask"], bool).sum() == 170


def test_missing_46_slots_are_zero_present_false_bad_false():
    mapping = adapter.load_mapping_profile(MAPPING)
    aligned, audit = adapter.align_sparse18(common_row(), mapping)
    present = np.asarray(aligned["channel_mask"], bool)
    bad = np.asarray(aligned["bad_channel_mask"], bool)
    data = np.asarray(aligned["data"])
    assert present.sum() == 18 and (~present).sum() == 46
    assert bad.sum() == 0
    assert np.all(data[~present] == 0)
    assert audit["target_valid_count"] == 18


def test_output_contract_is_checkpoint_compatible_and_metadata_preserved():
    mapping, mapping_hash, source_ids, target_ids = resolved()
    source = common_row(clip_fraction=0.91914845)
    processed, _ = adapter.build_token_row(source, mapping, mapping_hash, source_ids, target_ids)
    assert processed["num_tokens"] == 640 and processed["patch_len"] == 100
    assert np.asarray(processed["token_inputs"]).shape == (64000,)
    assert np.asarray(processed["targets"]).shape == (16000,)
    for key in ("sample_id", "subject_id", "recording_id", "block_id", "source_path", "window_index"):
        assert processed[key] == source[key]
    assert processed["window_start_sample"] == source["start_sample"]
    assert processed["window_start_seconds"] == source["start_seconds"]
    assert processed["task_semantics"] == "unknown"
    assert processed["resting_state_confirmed"] is False
    assert float(processed["clip_fraction"]) == pytest.approx(0.91914845)
    assert float(processed["source_128_clip_fraction"]) == pytest.approx(0.91914845)
    assert "split" not in adapter.output_schema().names and "fold" not in adapter.output_schema().names


def test_mapped_clip_fraction_uses_only_valid_sparse18_sources():
    mapping = adapter.load_mapping_profile(MAPPING)
    source = common_row(clip_fraction=0.9)
    data = np.asarray(source["data"], np.float32).reshape(128, 2000)
    data[:] = 0.25
    data[adapter.SOURCE_CHANNELS.index("E22")] = 1.0
    source["data"] = data.reshape(-1)
    _, audit = adapter.align_sparse18(source, mapping)
    assert audit["mapped_18_clip_fraction"] == pytest.approx(1 / 18)
    assert source["clip_fraction"] == pytest.approx(0.9)


def test_training_profiles_mask_dortmund_and_mipdb_without_new_geometry():
    profiles = training_profiles.load_training_profiles(PROFILES)
    inputs = np.ones((640, 100), np.float32)
    targets = np.ones((640, 5, 5), np.float32)
    native64 = np.ones(64, bool)
    matched = training_profiles.apply_training_profile(
        token_inputs=inputs, targets=targets, channel_valid_mask=native64,
        profiles=profiles, profile_name="matched18", dataset_name="dortmund",
    )
    assert matched["channel_valid_mask"].sum() == 18
    assert matched["token_valid_mask"].sum() == 180
    assert np.all(matched["token_inputs"][matched["token_valid_mask"] == 0] == 0)
    sensitivity = training_profiles.apply_training_profile(
        token_inputs=inputs, targets=targets, channel_valid_mask=native64,
        profiles=profiles, profile_name="native64_vs_sparse18", dataset_name="dortmund",
    )
    assert sensitivity["channel_valid_mask"].sum() == 64
    assert sensitivity["token_valid_mask"].sum() == 640
    mipdb = training_profiles.apply_training_profile(
        token_inputs=inputs, targets=targets, channel_valid_mask=native64,
        profiles=profiles, profile_name="native64_vs_sparse18", dataset_name="mipdb",
    )
    assert mipdb["channel_valid_mask"].sum() == 18
    assert mipdb["token_valid_mask"].sum() == 180


def test_end_to_end_fixture_preserves_source_and_writes_one_cache(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    input_lance = input_dir / "dataset.lance"
    source = common_row(bad_channel="E119")
    _write_lance_batch(input_lance, [source], True, common.mipdb_lance_schema())
    before = adapter.snapshot_tree(input_dir)
    output = tmp_path / "output"
    assert adapter.main([
        "--input-dir", str(input_dir), "--input-lance", str(input_lance),
        "--output-dir", str(output), "--mapping-config", str(MAPPING),
        "--training-profiles", str(PROFILES), "--electrode-vocab", str(VOCAB),
        "--batch-size", "1", "--num-workers", "0",
    ]) == 0
    assert adapter.snapshot_tree(input_dir) == before
    assert (output / "_SUCCESS").exists()
    token_path = output / adapter.TOKEN_LANCE_NAME
    assert lance.dataset(token_path).count_rows() == 1
    validation = adapter.validate_output(token_path, 1)
    assert validation["checkpoint_input_shape"] == [640, 100]
    assert validation["valid_token_count_distribution"] == {180: 1}
    assert len(list(output.glob("*.lance"))) == 1
    summary = json.loads((output / "adapter_summary.json").read_text(encoding="utf-8"))
    assert summary["one_physical_token_cache"] is True
    assert summary["training_profiles"] == ["matched18", "native64_vs_sparse18"]


def test_full_entrypoint_has_no_hardcoded_cohort_counts_or_overwrite():
    text = (ROOT / "commands" / "run_mipdb_fbmae_adapter_full.sh").read_text(encoding="utf-8")
    assert "adapt_mipdb_fbmae.py" in text
    assert "EXPECTED_SUBJECTS" not in text and "EXPECTED_RECORDINGS" not in text
    assert "--overwrite" not in text and "[640,100]" in text
