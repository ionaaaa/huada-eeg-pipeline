from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import preprocess_dortmund_age_windows as dortmund
from preprocess_huada_common_windows import (
    _prepare_output_dir,
    _snapshot_raw_files,
    _write_lance_batch,
    apply_common_signal_processing,
    dropped_tail,
    evaluate_window_qc,
    limit_window_slices,
    merge_bad_channel_masks,
    normalize_clip_divide_200,
    window_slices,
    zero_bad_channels,
)


def rec(subject="sub-001", session="ses-1", task="EyesOpen", acq="pre"):
    return {
        "subject_id": subject, "session_id": session, "task": task,
        "acquisition": acq,
        "eyes_condition": "eyes_open" if task == "EyesOpen" else "eyes_closed",
        "recording_id": f"{subject}_{session}_task-{task}_acq-{acq}_eeg",
    }


def cache_row(sample_id="x", age=25.0):
    return {
        "sample_id": sample_id, "subject_id": "sub-001", "session_id": "ses-1",
        "run_id": "run-01", "recording_id": "r1", "modality": "eeg",
        "data": np.zeros(128000, np.float32), "shape": [64, 2000],
        "original_shape": [64, 2000], "valid_length": 128000, "qc_pass": True,
        "label": np.float32(age), "age": np.float32(age), "task_type": "regression",
        "target_name": "age", "target_unit": "year", "eyes_condition": "eyes_open",
        "pre_or_post": "pre_cognitive", "source_path": "sub-001/a.edf", "start_time": 0.0,
        "duration": 10.0, "source_start_sample": 0, "target_start_sample": 0,
        "channel_names": dortmund.DORTMUND_CHANNELS, "channel_mask": np.ones(64, bool),
        "bad_channel_mask": np.zeros(64, bool), "channel_status": ["good"] * 64,
        "channel_layout": "dataset_level", "bad_channel_count": 0,
        "clip_fraction": np.float32(0), "positive_clip_fraction": np.float32(0),
        "negative_clip_fraction": np.float32(0), "pre_normalization_abs_max_uv": np.float32(0),
        "source_sampling_rate": np.float32(1000), "target_sampling_rate": np.float32(200),
        "preprocessing_config_hash": "abc",
    }


def test_profile_a_selects_only_ses1_pre():
    assert dortmund.profile_a_decision({"session": "ses-1", "acquisition": "pre", "task": "EyesOpen"}, {"TaskName": "RestingState"}) == (True, "selected_profile_a")


def test_profile_a_accepts_eyes_open_and_closed():
    for task in ("EyesOpen", "EyesClosed"):
        assert dortmund.profile_a_decision({"session": "ses-1", "acquisition": "pre", "task": task}, {"TaskName": "RestingState"})[0]


def test_ses2_excluded_for_age_semantics():
    assert dortmund.profile_a_decision({"session": "ses-2", "acquisition": "pre", "task": "EyesOpen"}, {"TaskName": "RestingState"})[1] == "excluded_age_semantics_unsafe"


def test_post_excluded_for_profile_a():
    assert dortmund.profile_a_decision({"session": "ses-1", "acquisition": "post", "task": "EyesOpen"}, {"TaskName": "RestingState"})[1] == "excluded_profile_a"


def test_profile_a_formal_214_428_gate():
    rows = []
    for index in range(214):
        sid = f"sub-{index:03d}"
        rows.extend([rec(sid, task="EyesOpen"), rec(sid, task="EyesClosed")])
    dortmund.validate_profile_a_counts(rows)


def test_profile_a_gate_fails_closed_on_missing_recording():
    with pytest.raises(ValueError, match="count gate failed"):
        dortmund.validate_profile_a_counts([rec()], 1, 2)


@pytest.mark.parametrize("value, expected", [("20", 20), (35, 35), ("24.0", 24)])
def test_numeric_age_reading(value, expected):
    assert dortmund.parse_numeric_age(value) == expected


@pytest.mark.parametrize("value", ["20-25", "young", "n/a", 19, 36])
def test_invalid_age_rejected(value):
    with pytest.raises(ValueError):
        dortmund.parse_numeric_age(value)


def test_age_and_label_equal_in_schema_row():
    row = cache_row(age=27)
    assert row["age"] == row["label"]


def test_status_is_not_native_eeg_channel():
    assert len(dortmund.DORTMUND_CHANNELS) == 64
    assert "Status" not in dortmund.DORTMUND_CHANNELS


@pytest.mark.parametrize("status_type", ["stim", "eeg", "misc"])
def test_status_exclusion_does_not_depend_on_mne_inferred_type(status_type):
    names = [*dortmund.DORTMUND_CHANNELS, "Status"]
    types = ["eeg"] * 64 + [status_type]
    eeg_indices, status_indices = dortmund.resolve_dortmund_channel_indices(names, types)
    assert eeg_indices == list(range(64))
    assert status_indices == [64]


def test_status_must_exist_exactly_once():
    with pytest.raises(RuntimeError, match="one exact-name Status"):
        dortmund.resolve_dortmund_channel_indices(
            [*dortmund.DORTMUND_CHANNELS, "Trigger"],
            ["eeg"] * 64 + ["stim"],
        )


def test_native_channel_order_is_stable():
    assert dortmund.DORTMUND_CHANNELS[:5] == ["Fp1", "Fp2", "F7", "F3", "Fz"]
    assert dortmund.DORTMUND_CHANNELS[-1] == "PO8"


def test_no_huada_slot_mapping_fields():
    names = set(dortmund.dortmund_lance_schema().names)
    assert "huada_channel_index" not in names and "electrode_ids" not in names


def test_reference_policy_constants():
    assert dortmund.SOURCE_REFERENCE == "FCz" and dortmund.SOURCE_GROUND == "AFz"


def test_mne_v_to_uv_reused():
    from preprocess_huada_common_windows import convert_mne_v_to_uv
    np.testing.assert_allclose(convert_mne_v_to_uv(np.array([1e-6])), [1.0])


def test_common_processing_resamples_1000_to_200():
    t = np.arange(0, 12, 0.001)
    x = np.sin(2 * np.pi * 10 * t)[None, :]
    y = apply_common_signal_processing(x, 1000, 200, 50, 0.3, 75)
    assert y.shape == (1, 2400)


def test_ten_second_window_is_2000_points():
    assert window_slices(4000, 2000, 2000) == [(0, 2000), (2000, 4000)]


def test_incomplete_tail_is_dropped_without_padding():
    assert dropped_tail(4500, 2000, 2000, 200) == (500, 2.5)


def test_sixty_second_recording_generates_six_windows():
    assert len(window_slices(60 * 200, 2000, 2000)) == 6


def test_duration_conflict_uses_edf_samples():
    actual, matches = dortmund.resolve_duration(184000, 1000, 192)
    assert actual == 184 and not matches


def test_incomplete_full_read_fails_closed():
    with pytest.raises(RuntimeError, match="Actual samples"):
        dortmund.validate_full_read(999, 1000, False, False)


@pytest.mark.parametrize("has_nan,has_inf", [(True, False), (False, True)])
def test_nan_or_inf_fails_closed(has_nan, has_inf):
    with pytest.raises(RuntimeError, match="NaN/Inf"):
        dortmund.validate_full_read(1000, 1000, has_nan, has_inf)


def test_constant_bad_channel_mask_and_zero():
    data = np.ones((64, 2000)); bad = np.zeros(64, bool); bad[3] = True
    zeroed, channel_mask = zero_bad_channels(data, bad)
    assert channel_mask.all() and np.all(zeroed[3] == 0)


def test_masks_and_window_shape():
    passed, reasons = evaluate_window_qc(np.ones((64, 2000)), np.zeros(64, bool))
    assert passed and not reasons


def test_normalization_is_float32_in_range():
    y = normalize_clip_divide_200(np.array([-1000, 0, 1000]))
    assert y.dtype == np.float32 and y.tolist() == [-1.0, 0.0, 1.0]


def test_clip_fraction_is_not_a_hard_qc_rule():
    passed, reasons = evaluate_window_qc(np.full((64, 2000), 10000.0), np.zeros(64, bool))
    assert passed and not reasons


def test_deterministic_sample_id_and_condition_separation():
    a = dortmund.make_dortmund_sample_id("sub-1", "ses-1", "eyes_open", "pre", "r", 0, 0)
    b = dortmund.make_dortmund_sample_id("sub-1", "ses-1", "eyes_closed", "pre", "r", 0, 0)
    assert a == dortmund.make_dortmund_sample_id("sub-1", "ses-1", "eyes_open", "pre", "r", 0, 0) and a != b


def test_lance_roundtrip(tmp_path):
    path = tmp_path / "dataset.lance"
    _write_lance_batch(path, [cache_row()], True, dortmund.dortmund_lance_schema())
    result = dortmund.validate_dortmund_lance(path, 1)
    assert result["all_rows_valid"] and result["sample_id_unique"]


def test_regression_label_schema_and_no_split_or_tokens():
    schema = dortmund.dortmund_lance_schema()
    assert schema.field("label").type == pa.float32() and schema.field("age").type == pa.float32()
    assert not ({"split", "fold", "model_tokens", "electrode_ids"} & set(schema.names))


def test_source_snapshot_before_after_equal(tmp_path):
    data = tmp_path / "data"; data.mkdir(); (data / "a.edf").write_bytes(b"abc")
    assert _snapshot_raw_files(data, ["a.edf"]) == _snapshot_raw_files(data, ["a.edf"])


def test_smoke_subject_selection_happens_on_manifest():
    rows = [rec("sub-001"), rec("sub-230"), rec("sub-600")]
    selected = dortmund.select_smoke_recordings(rows, True, ["sub-230", "sub-600"])
    assert [x["subject_id"] for x in selected] == ["sub-230", "sub-600"]


def test_smoke_window_limit_is_two():
    selected, truncated = limit_window_slices(window_slices(10000, 2000, 2000), True, 2)
    assert len(selected) == 2 and truncated


def test_output_directory_protection(tmp_path):
    data = tmp_path / "raw"; data.mkdir()
    with pytest.raises(ValueError):
        _prepare_output_dir(data / "output", False, data)


def test_bids_entity_parser_is_exact():
    parsed = dortmund.parse_bids_edf("sub-230_ses-1_task-EyesClosed_acq-pre_eeg.edf")
    assert parsed == {"subject": "sub-230", "session": "ses-1", "task": "EyesClosed", "acquisition": "pre", "run": ""}
    assert dortmund.parse_bids_edf("notes_EyesClosed_pre.edf") == {}


def test_profile_task_semantics_must_be_documented():
    ok, reason = dortmund.profile_a_decision({"session": "ses-1", "acquisition": "pre", "task": "EyesOpen"}, {"TaskName": "Task"})
    assert not ok and reason == "excluded_task_semantics_unconfirmed"


def test_merge_recording_and_window_bad_masks():
    assert merge_bad_channel_masks([True, False], [False, True]).tolist() == [True, True]


def test_schema_data_and_mask_fixed_sizes():
    schema = dortmund.dortmund_lance_schema()
    assert schema.field("data").type.list_size == 128000
    assert schema.field("channel_mask").type.list_size == schema.field("bad_channel_mask").type.list_size == 64


@pytest.mark.parametrize("name, smoke_expected", [
    ("run_dortmund_age_preprocessing_smoke.sh", True),
    ("run_dortmund_age_preprocessing_full.sh", False),
])
def test_tione_shell_entry_is_read_only_and_scoped(name, smoke_expected):
    text = (Path(__file__).resolve().parents[1] / "commands" / name).read_text(encoding="utf-8")
    assert "set -euo pipefail" in text
    assert "DORTMUND_SOURCE_URI" in text and "DORTMUND_DATA_DIR" in text
    assert "preprocess_dortmund_age_windows.py" in text
    assert "coscli" not in text and " rm " not in text and "--overwrite" not in text
    assert ("--smoke" in text) is smoke_expected
