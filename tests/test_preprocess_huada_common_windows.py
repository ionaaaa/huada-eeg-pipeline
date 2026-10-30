from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import lance
import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import preprocess_huada_common_windows as preprocessing  # noqa: E402
from preprocess_huada_common_windows import (  # noqa: E402
    common_window_schema,
    _prepare_output_dir,
    _snapshot_raw_files,
    build_full_run_command,
    convert_mne_v_to_uv,
    detect_hard_bad_channel,
    dropped_tail,
    evaluate_window_qc,
    identify_window_bad_channels,
    limit_window_slices,
    make_sample_id,
    merge_bad_channel_masks,
    normalize_clip_divide_200,
    parse_args,
    parse_smoke_subjects,
    Pipeline,
    REQUIRED_OUTPUT_NAMES,
    rows_to_arrow,
    select_recordings,
    validate_lance_roundtrip,
    validate_smoke_output_dir,
    window_slices,
    zero_bad_channels,
)


def make_cache_row(subject_id: str = "SUB5", bad_index: int = 3) -> dict:
    data = np.linspace(-1, 1, 128000, dtype=np.float32).reshape(64, 2000)
    data[bad_index] = 0.0
    channel_names = [f"C{i}" for i in range(64)]
    channel_names[bad_index] = "FC1"
    return {
        "sample_id": f"{subject_id}__recording01__w00000", "subject_id": subject_id,
        "modality": "eeg", "source_path": f"{subject_id}/a.bdf",
        "recording_id": "recording01", "window_index": 0,
        "start_time": 0.0, "duration": 10.0, "source_start_sample": 0,
        "source_stop_sample": 5000, "data": data.reshape(-1), "shape": [64, 2000],
        "original_shape": [64, 2000], "valid_length": 128000,
        "channel_names": channel_names, "channel_mask": np.ones(64, bool),
        "bad_channel_mask": np.array([i == bad_index for i in range(64)]),
        "channel_status": ["bad" if i == bad_index else "good" for i in range(64)],
        "qc_pass": True, "bad_channel_count": 1, "age": None, "sex": "male",
        "original_directory_name": "SUB5_M", "clip_fraction": np.float32(0.0),
        "positive_clip_fraction": np.float32(0.0), "negative_clip_fraction": np.float32(0.0),
        "pre_normalization_abs_max_uv": np.float32(200.0),
        "source_sampling_rate": np.float32(500.0), "target_sampling_rate": np.float32(200.0),
        "preprocessing_config_hash": "abc",
    }


def test_mne_v_to_uv_conversion():
    actual = convert_mne_v_to_uv(np.array([[1e-6, -2e-6]], dtype=np.float64))
    np.testing.assert_allclose(actual, [[1.0, -2.0]])


def test_all_zero_channel_detected():
    reasons = detect_hard_bad_channel(np.zeros(100))
    assert "all_zero" in reasons and "strict_constant" in reasons


def test_strict_constant_channel_detected():
    reasons = detect_hard_bad_channel(np.full(100, 3.25))
    assert "strict_constant" in reasons and "exact_zero_variance" in reasons


def test_physical_full_scale_constant_detected():
    reasons = detect_hard_bad_channel(np.full(100, -2250000.0), physical_min_uv=-2250000.0)
    assert "fixed_at_physical_minimum" in reasons


def test_normal_signal_not_misclassified():
    x = np.array([0.0, 1.0, -1.0, 2.0])
    assert detect_hard_bad_channel(x, -2250000.0, 2249999.0) == []


def test_recording_bad_propagates_to_all_windows():
    recording_bad = np.array([True, False, False])
    for window_bad in [np.zeros(3, bool), np.array([False, False, True])]:
        merged = merge_bad_channel_masks(recording_bad, window_bad)
        assert merged[0]


def test_window_new_bad_channel_is_merged():
    window = np.vstack([np.arange(10), np.zeros(10), np.arange(10) + 1.0])
    new_bad, reasons, finite = identify_window_bad_channels(window, [False, False, False])
    assert finite and new_bad.tolist() == [False, True, False]
    assert reasons[1] == ["window_all_zero"]
    assert merge_bad_channel_masks([True, False, False], new_bad).tolist() == [True, True, False]


def test_ten_second_nonoverlapping_windows_and_tail():
    assert window_slices(4500, 2000, 2000) == [(0, 2000), (2000, 4000)]
    assert dropped_tail(4500, 2000, 2000, 200.0) == (500, 2.5)


def test_bad_channel_zeroed_but_channel_mask_true():
    data = np.ones((3, 5))
    zeroed, channel_mask = zero_bad_channels(data, [False, True, False])
    assert channel_mask.tolist() == [True, True, True]
    assert np.all(zeroed[1] == 0.0)


def test_clip_divide_200_formula_and_range():
    x = np.array([[-400.0, -200.0, -100.0, 0.0, 100.0, 200.0, 400.0]])
    y = normalize_clip_divide_200(x)
    np.testing.assert_array_equal(y, np.array([[-1, -1, -0.5, 0, 0.5, 1, 1]], dtype=np.float32))
    assert y.dtype == np.float32 and y.min() >= -1 and y.max() <= 1


def test_qc_pass_rules():
    good = np.random.default_rng(1).normal(size=(64, 2000))
    passed, reasons = evaluate_window_qc(good, np.zeros(64, bool))
    assert passed and reasons == []
    failed, reasons = evaluate_window_qc(np.zeros((64, 2000)), np.zeros(64, bool))
    assert not failed and "entire_window_zero" in reasons
    failed, reasons = evaluate_window_qc(good, np.ones(64, bool))
    assert not failed and "all_channels_bad" in reasons


def test_sample_id_stable_and_unique():
    ids = {make_sample_id("SUB1", "recording01", i) for i in range(20)}
    assert make_sample_id("SUB1", "recording01", 0) == "SUB1__recording01__w00000"
    assert len(ids) == 20
    assert make_sample_id("SUB1", "recording02", 0) not in ids


def test_lance_roundtrip_preserves_shape_dtype_masks_and_data(tmp_path: Path):
    row = make_cache_row()
    data = np.asarray(row["data"], dtype=np.float32).reshape(64, 2000)
    path = tmp_path / "tiny.lance"
    table = rows_to_arrow([row], common_window_schema())
    lance.write_dataset(table, path, mode="create", data_storage_version="stable")
    result = validate_lance_roundtrip(path, expected_rows=1)
    assert result["schema_matches"] and result["bad_slots_zero"] and result["sample_id_unique"]
    read = lance.dataset(path).to_table()
    restored = np.asarray(read["data"][0].values, dtype=np.float32).reshape(64, 2000)
    np.testing.assert_array_equal(restored, data)
    names = read["channel_names"][0].as_py()
    fc1 = names.index("FC1")
    assert read["channel_mask"][0].values[fc1].as_py() is True
    assert read["bad_channel_mask"][0].values[fc1].as_py() is True
    assert read["channel_status"][0].as_py()[fc1] == "bad"
    assert np.all(restored[fc1] == 0.0)


def test_smoke_subject_parser_and_selection():
    subjects = parse_smoke_subjects(["SUB1,SUB5"])
    recordings = [
        {"subject_id": "SUB1", "source_path": "SUB1/a.bdf"},
        {"subject_id": "SUB2", "source_path": "SUB2/a.bdf"},
        {"subject_id": "SUB5", "source_path": "SUB5/a.bdf"},
    ]
    assert subjects == ["SUB1", "SUB5"]
    assert [row["subject_id"] for row in select_recordings(recordings, True, subjects)] == ["SUB1", "SUB5"]
    assert select_recordings(recordings, False, subjects) == recordings


def test_unselected_recordings_are_not_dispatched_for_opening(monkeypatch, tmp_path: Path):
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.recordings = select_recordings([
        {"subject_id": "SUB1", "source_path": "SUB1/a.bdf"},
        {"subject_id": "SUB2", "source_path": "SUB2/a.bdf"},
        {"subject_id": "SUB5", "source_path": "SUB5/a.bdf"},
    ], True, ["SUB1", "SUB5"])
    pipeline.window_rows = []
    pipeline._first_lance_write = True
    pipeline.lance_path = tmp_path / "unused.lance"
    pipeline.validate_inputs = lambda: None
    dispatched: list[str] = []

    def fake_process(rec, recording_id):
        dispatched.append(rec["subject_id"])
        pipeline.window_rows.append({})
        pipeline._first_lance_write = False

    pipeline._process_one = fake_process
    monkeypatch.setattr(preprocessing, "validate_lance_roundtrip", lambda path, expected: {"row_count": expected})
    pipeline.run()
    assert dispatched == ["SUB1", "SUB5"]
    assert "SUB2" not in dispatched


def test_smoke_window_limit_does_not_change_real_tail():
    possible = window_slices(121252, 2000, 2000)
    emitted, truncated = limit_window_slices(possible, True, 2)
    assert len(possible) == 60 and len(emitted) == 2 and truncated
    assert dropped_tail(121252, 2000, 2000, 200.0) == (1252, 6.26)
    full, full_truncated = limit_window_slices(possible, False, None)
    assert full == possible and not full_truncated
    assert [make_sample_id("SUB1", "recording01", i) for i in range(2)] == [
        "SUB1__recording01__w00000", "SUB1__recording01__w00001",
    ]


def test_parse_args_records_smoke_mode_and_full_command(tmp_path: Path):
    args = parse_args([
        "--data-dir", str(tmp_path / "data"), "--config", str(PROJECT_ROOT / "configs" / "huada_resting_preprocessing.toml"),
        "--subject-manifest", str(tmp_path / "manifest.csv"),
        "--recording-validation", str(tmp_path / "recordings.csv"),
        "--channel-header-inventory", str(tmp_path / "channels.csv"),
        "--output-dir", str(tmp_path / "smoke"), "--smoke",
        "--smoke-subjects", "SUB1,SUB5", "--smoke-max-windows-per-recording", "2",
    ])
    assert args.run_mode == "smoke"
    assert args.smoke_subjects == ["SUB1", "SUB5"]
    assert args.smoke_max_windows_per_recording == 2
    assert "--smoke" not in args.full_run_command
    assert "--overwrite" not in args.full_run_command
    assert not args.full_run_output_dir.exists()


def test_existing_output_requires_explicit_overwrite(tmp_path: Path):
    output = tmp_path / "existing"
    output.mkdir()
    (output / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        _prepare_output_dir(output, False, tmp_path / "raw")


def test_smoke_output_rejects_formal_full_tree(tmp_path: Path):
    project = tmp_path / "project"
    with pytest.raises(ValueError):
        validate_smoke_output_dir(project / "preprocessed_common" / "formal", project)
    validate_smoke_output_dir(project / "preprocessed_common_smoke" / "run", project)


def test_source_snapshot_size_and_nanosecond_mtime_unchanged(tmp_path: Path):
    data_dir = tmp_path / "data"
    source = data_dir / "SUB1" / "a.bdf"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"read-only-source")
    before = _snapshot_raw_files(data_dir, ["SUB1/a.bdf"])
    assert source.read_bytes() == b"read-only-source"
    after = _snapshot_raw_files(data_dir, ["SUB1/a.bdf"])
    assert before == after


def test_additional_outputs_and_internal_statistics(tmp_path: Path):
    output = tmp_path / "smoke_output"
    output.mkdir()
    row = make_cache_row()
    lance_path = output / "common_windows.lance"
    lance.write_dataset(rows_to_arrow([row]), lance_path, mode="create", data_storage_version="stable")
    roundtrip = validate_lance_roundtrip(lance_path, 1)

    pipeline = Pipeline.__new__(Pipeline)
    pipeline.output_dir = output
    pipeline.lance_path = lance_path
    pipeline.data_dir = tmp_path / "data"
    pipeline.data_dir.mkdir()
    pipeline.pre = {
        "normalization": "clip_divide_200", "normalization_detail": "x_uV / 200, then clip to [-1, 1]",
        "notch_hz": 50.0, "bandpass_low": 0.3, "bandpass_high": 75.0,
        "target_sfreq": 200.0, "window_seconds": 10.0, "hop_seconds": 10.0, "segment_samples": 2000,
    }
    pipeline.exp = {"sampling_rate_hz": 500.0}
    pipeline.windowing = {"drop_incomplete_tail": True, "padding": False}
    pipeline.config_hash = "abc"
    pipeline.manifest = {"SUB5": {"subject_id": "SUB5"}}
    pipeline.selected_subjects = ["SUB5"]
    pipeline.recordings = [{"subject_id": "SUB5", "source_path": "SUB5/a.bdf"}]
    pipeline.issues = []
    pipeline.errors = []
    pipeline.bad_rows = [{
        "subject_id": "SUB5", "source_path": "SUB5/a.bdf", "channel_index": 3,
        "channel_name": "FC1", "scope": "recording", "bad_reason": "strict_constant",
        "affected_emitted_windows": 1, "data_zeroed": True, "mask_verified": True, "is_bad": True,
    }]
    pipeline.recording_rows = [{
        "subject_id": "SUB5", "source_path": "SUB5/a.bdf", "run_mode": "smoke",
        "recording_id": "recording01", "source_samples": 300000, "source_duration_seconds": 600.0,
        "resampled_samples": 120000, "possible_complete_windows": 60, "emitted_windows": 1,
        "smoke_truncated": True, "dropped_tail_samples": 0, "dropped_tail_seconds": 0.0,
        "recording_bad_channels": "FC1", "qc_pass_windows": 1, "qc_fail_windows": 0,
        "mean_clip_fraction": 0.0, "max_clip_fraction": 0.0, "processing_success": True,
    }]
    pipeline.window_rows = [{
        "sample_id": row["sample_id"], "subject_id": "SUB5", "source_path": "SUB5/a.bdf",
        "recording_id": "recording01", "window_index": 0, "start_time": 0.0, "duration": 10.0,
        "shape": "[64,2000]", "qc_pass": True, "bad_channel_count": 1,
        "bad_channel_names": "FC1", "clip_fraction": 0.0, "positive_clip_fraction": 0.0,
        "negative_clip_fraction": 0.0, "pre_normalization_abs_max_uv": 200.0,
    }]
    pipeline.tail_rows = [{
        "subject_id": "SUB5", "source_path": "SUB5/a.bdf", "recording_id": "recording01",
        "raw_samples": 300000, "resampled_samples": 120000, "possible_complete_windows": 60,
        "emitted_windows": 1, "smoke_truncated": True, "smoke_omitted_complete_windows": 59,
        "complete_windows": 60, "dropped_resampled_samples": 0, "dropped_source_samples": 0,
        "dropped_tail_seconds": 0.0,
    }]
    pipeline.subject_stats = defaultdict(dict, {
        "SUB5": {"window_count": 1, "dropped_tail_seconds": 0.0, "qc_pass": 1, "qc_fail": 0,
                 "recording_bad_channels": {"FC1"}, "window_new_bad_channels": set(),
                 "clip_fractions": [0.0]},
    })
    pipeline.args = SimpleNamespace(
        config=PROJECT_ROOT / "configs" / "huada_resting_preprocessing.toml",
        subject_manifest=tmp_path / "subject_manifest.csv", recording_validation=tmp_path / "recordings.csv",
        channel_header_inventory=tmp_path / "channels.csv", run_mode="smoke", smoke=True,
        smoke_max_windows_per_recording=1, run_started="2026-10-30T00:00:00+08:00",
        actual_command="python preprocess.py --smoke", full_run_command="python preprocess.py --output-dir new-full\n",
        full_run_command_path=tmp_path / "full_run_command.ps1",
    )
    snapshot = {"SUB5/a.bdf": {"size_bytes": 10, "mtime_ns": 20}}
    summary = pipeline.write_outputs(roundtrip, snapshot, snapshot)

    assert all((output / name).exists() for name in REQUIRED_OUTPUT_NAMES)
    assert (output / "full_run_command.ps1").exists()
    assert (tmp_path / "full_run_command.ps1").exists()
    assert (output / "preprocessing_errors.log").read_text(encoding="utf-8").startswith("timestamp\tstage")
    with (output / "preprocessing_issues.csv").open(encoding="utf-8-sig", newline="") as handle:
        assert next(csv.reader(handle)) == [
            "severity", "stage", "subject_id", "source_path", "sample_id", "channel_name",
            "issue_type", "description", "action",
        ]
    provenance = json.loads((output / "processing_provenance.json").read_text(encoding="utf-8"))
    saved_summary = json.loads((output / "preprocessing_summary.json").read_text(encoding="utf-8"))
    assert provenance["run_mode"] == "smoke"
    assert saved_summary["emitted_window_count"] == lance.dataset(lance_path).count_rows() == 1
    assert saved_summary["unique_sample_id_count"] == 1
    assert saved_summary["contains_label"] is False
    assert summary["source_files_unchanged"] is True
    with (output / "bad_channel_inventory.csv").open(encoding="utf-8-sig", newline="") as handle:
        bad = next(csv.DictReader(handle))
    assert bad["channel_name"] == "FC1" and bad["scope"] == "recording"
    assert bad["data_zeroed"] == "True" and bad["mask_verified"] == "True"
