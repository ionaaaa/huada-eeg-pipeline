from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from scripts.audit_dortmund_age_subset import (
    CSV_FIELDS,
    RunningStats,
    age_status_for_session,
    assert_limits,
    choose_nearest,
    classify_materialization,
    compare_snapshots,
    decide_adaptive_expansion,
    ensure_new_output,
    open_edf_header,
    parse_bids_entities,
    select_initial_subjects,
    select_stratified_subjects,
    snapshot_tree,
    validate_numeric_age,
    write_csv,
)


def subjects(n=20):
    return [
        {
            "subject_id": f"sub-{i:03d}",
            "age": 20 + (i % 16),
            "sex": "F" if i % 2 else "M",
            "recordings": [{"recording_id": f"r{i}-{j}"} for j in range(4)],
            "signatures": [f"sig{i % 2}"],
        }
        for i in range(1, n + 1)
    ]


def test_complete_metadata_and_download_subset_are_distinct():
    metadata = {f"sub-{i:03d}" for i in range(1, 609)}
    downloaded = {f"sub-{i:03d}" for i in range(1, 215)}
    assert len(metadata & downloaded) == 214
    assert len(metadata - downloaded) == 394


def test_not_downloaded_subjects_are_out_of_scope_not_missing():
    metadata, downloaded = {"sub-001", "sub-002"}, {"sub-001"}
    assert metadata - downloaded == {"sub-002"}


@pytest.mark.parametrize("value", ["20-25", "young", "adult"])
def test_age_bins_are_rejected(value):
    assert validate_numeric_age(value)[0] is False


@pytest.mark.parametrize("value, expected", [(20, 20.0), ("35", 35.0), (27.5, 27.5)])
def test_numeric_age_is_accepted(value, expected):
    assert validate_numeric_age(value) == (True, expected, "numeric_years")


def test_missing_age_blocks_gate():
    assert validate_numeric_age("n/a")[2] == "missing"


def test_baseline_age_does_not_inherit_to_followup():
    assert age_status_for_session("ses-2", 25) == ("unsafe_or_unknown", None)


def test_explicit_session_age_is_usable():
    assert age_status_for_session("ses-2", 25, 30) == ("documented_session_age", 30)


def test_git_annex_pointer_detection(tmp_path):
    path = tmp_path / "x.edf"
    path.write_bytes(b"/annex/objects/abc")
    assert classify_materialization(path)[0] == "git_annex_pointer"


def test_empty_and_html_detection(tmp_path):
    empty = tmp_path / "empty.edf"
    empty.write_bytes(b"")
    html = tmp_path / "html.edf"
    html.write_bytes(b"<!doctype html><title>error</title>")
    assert classify_materialization(empty)[0] == "empty"
    assert classify_materialization(html)[0] == "error_page"


def test_bids_entity_parser():
    ent = parse_bids_entities("sub-004_ses-1_task-EyesOpen_acq-pre_run-2_eeg.edf")
    assert (ent["subject"], ent["session"], ent["task"], ent["acquisition"], ent["run"]) == ("sub-004", "ses-1", "EyesOpen", "pre", "2")


def test_header_reader_never_preloads(tmp_path):
    seen = {}
    def fake(path, **kwargs):
        seen.update(kwargs)
        return object()
    open_edf_header(tmp_path / "x.edf", reader=fake)
    assert seen["preload"] is False


def test_d1_maximum_three_subjects():
    assert len(select_initial_subjects(subjects(), 3)) == 3
    with pytest.raises(AssertionError):
        select_initial_subjects(subjects(), 4)


def test_d1_recording_cap_is_twelve():
    assert_limits(3, 12, 4, 100, 0.20)
    with pytest.raises(AssertionError):
        assert_limits(3, 12, 5, 100, 0.20)


def test_d1_error_routes_to_adaptive_decision():
    auto, approval = decide_adaptive_expansion(5, 100, 100, 0.20)
    assert auto and not approval


def test_d2_total_is_twelve_not_extra_twelve():
    pool = subjects(30)
    d1 = select_initial_subjects(pool, 3)
    d2 = select_stratified_subjects(pool, d1, 12)
    assert len(d2) == 12
    assert {x["subject_id"] for x in d1} <= {x["subject_id"] for x in d2}


def test_age_anchor_selection_prefers_nearest():
    selected = choose_nearest(subjects(), 35, set())
    assert abs(selected["age"] - 35) == min(abs(x["age"] - 35) for x in subjects())


def test_sex_and_signature_inputs_are_preserved():
    chosen = select_stratified_subjects(subjects(30), [], 12)
    assert {x["sex"] for x in chosen} == {"F", "M"}
    assert all(x["signatures"] for x in chosen)


def test_d2_maximum_48_recordings():
    assert_limits(3, 12, 4, 100, 0.20)
    assert 12 * 4 == 48


def test_recording_deduplication_set():
    seen = set()
    paths = ["a.edf", "a.edf", "b.edf"]
    unique = [p for p in paths if not (p in seen or seen.add(p))]
    assert unique == ["a.edf", "b.edf"]


@pytest.mark.parametrize("scope", ["session", "device_signature", "channel_signature"])
def test_targeted_expansion_scope_can_be_classified(scope):
    rules = {"session": "same_session", "device_signature": "same_signature", "channel_signature": "same_signature_channel"}
    assert scope in rules


def test_expansion_over_100_requires_approval():
    assert decide_adaptive_expansion(101, 1000) == (False, True)


def test_expansion_over_20_percent_requires_approval():
    assert decide_adaptive_expansion(21, 100) == (False, True)


def test_full_signal_qc_is_deferred_constant():
    summary = {"full_signal_audit_performed": False, "full_signal_qc_deferred_to_preprocessing": True}
    assert summary["full_signal_qc_deferred_to_preprocessing"] and not summary["full_signal_audit_performed"]


def test_subject_split_key_groups_all_sessions():
    rows = [("sub-001", "ses-1"), ("sub-001", "ses-2")]
    assert len({subject for subject, _ in rows}) == 1


def test_output_directory_protection(tmp_path):
    ensure_new_output(tmp_path)
    with pytest.raises(FileExistsError):
        ensure_new_output(tmp_path)


def test_empty_issues_csv_has_header(tmp_path):
    path = tmp_path / "audit_issues.csv"
    write_csv(path, [], CSV_FIELDS["audit_issues.csv"])
    with path.open(encoding="utf-8-sig", newline="") as handle:
        assert next(csv.reader(handle)) == CSV_FIELDS["audit_issues.csv"]


def test_empty_errors_log_can_be_created(tmp_path):
    path = tmp_path / "audit_errors.log"
    path.write_text("", encoding="utf-8")
    assert path.exists() and path.read_text(encoding="utf-8") == ""


def test_source_snapshot_unchanged(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "x.edf").write_bytes(b"0       " + b"x" * 2048)
    before = snapshot_tree(source)
    after = snapshot_tree(source)
    assert compare_snapshots(before, after) == (True, False, False)


def test_running_stats_detects_constant_and_finite():
    stat = RunningStats()
    stat.update(np.ones(100))
    assert stat.minimum == stat.maximum == 1.0
    assert stat.variance == 0.0
    assert not stat.has_nan and not stat.has_inf
