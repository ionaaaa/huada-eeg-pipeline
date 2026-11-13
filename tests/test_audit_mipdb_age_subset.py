from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.audit_dortmund_age_subset import RunningStats, compare_snapshots, ensure_new_output, snapshot_tree
from scripts.audit_mipdb_age_subset import (
    CSV_FIELDS,
    PRIORITY_RECORDINGS,
    Audit,
    binary_geometry,
    build_sampling_plan,
    channel_flags,
    load_allowlist,
    parse_vhdr,
    preflight_dataset,
    recommend_expansion,
    resolve_age_labels,
)


def write_root(root: Path, participants: list[tuple[str, str, str]] | None = None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "synthetic MIPDB", "Version": "1.0.2", "BIDSVersion": "1.9.0", "License": "test"}),
        encoding="utf-8",
    )
    (root / "participants.json").write_text(
        json.dumps({"age": {"Units": "years"}, "sex": {"Levels": {"F": "female", "M": "male"}}}),
        encoding="utf-8",
    )
    rows = participants or []
    text = "participant_id\tage\tsex\thand\tweight\theight\n"
    text += "".join(f"{sid}\t{age}\t{sex}\tn/a\tn/a\tn/a\n" for sid, age, sex in rows)
    (root / "participants.tsv").write_text(text, encoding="utf-8")


def write_triplet(root: Path, sid: str, task: str = "block01", samples: int = 10, channels: int = 2, remainder: int = 0, marker: bool = True) -> Path:
    eeg_dir = root / sid / "eeg"
    eeg_dir.mkdir(parents=True, exist_ok=True)
    base = f"{sid}_task-{task}_eeg"
    vhdr = eeg_dir / f"{base}.vhdr"
    channel_lines = "\n".join(f"Ch{i}=E{i},,0.1,µV" for i in range(1, channels + 1))
    vhdr.write_text(
        "Brain Vision Data Exchange Header File Version 1.0\n"
        "[Common Infos]\n"
        f"DataFile={base}.eeg\n"
        f"MarkerFile={base}.vmrk\n"
        "DataFormat=BINARY\nDataOrientation=MULTIPLEXED\n"
        f"NumberOfChannels={channels}\nSamplingInterval=2000.0\n"
        "[Binary Infos]\nBinaryFormat=IEEE_FLOAT_32\n"
        f"[Channel Infos]\n{channel_lines}\n",
        encoding="utf-8",
    )
    (eeg_dir / f"{base}.eeg").write_bytes(b"\0" * (samples * channels * 4 + remainder))
    if marker:
        (eeg_dir / f"{base}.vmrk").write_text(
            f"Brain Vision Data Exchange Marker File, Version 1.0\n[Common Infos]\nDataFile={base}.eeg\n[Marker Infos]\n",
            encoding="utf-8",
        )
    return vhdr


def make_dataset(tmp_path: Path, subjects: list[str]) -> Path:
    root = tmp_path / "MIPDB" / "mipdb_download" / "nm000153_py"
    write_root(root, [(sid, "25", "F") for sid in subjects])
    for sid in subjects:
        write_triplet(root, sid)
    return root


def fake_recording(subject: str, task: str, path: str, duration: float, signature: str = "sig1", age: float = 25, sex: str = "F") -> dict:
    return {
        "subject_id": subject,
        "task": task,
        "recording": Path(path).stem,
        "source_path": path,
        "duration_seconds": duration,
        "sampling_frequency_hz": 500.0,
        "header_signature": signature,
        "header_read_success": True,
        "exact_age": age,
        "sex": sex,
    }


def test_versioned_allowlist_is_exact_authoritative_23():
    path = Path(__file__).resolve().parents[1] / "configs" / "mipdb_age_20_35_subjects.txt"
    values = load_allowlist(path)
    assert len(values) == 23
    assert "sub-A00054122" not in values
    assert "sub-A00056640" not in values


def test_allowlist_filter_excludes_outside_subject(tmp_path):
    root = make_dataset(tmp_path, ["sub-A", "sub-B"])
    result = preflight_dataset(root, ["sub-A"])
    assert result["passed"]
    assert result["extra_subjects"] == ["sub-B"]
    assert {row["subject_id"] for row in result["subjects"]} == {"sub-A"}


def test_missing_allowlisted_subject_fails_closed(tmp_path):
    root = make_dataset(tmp_path, ["sub-A"])
    result = preflight_dataset(root, ["sub-A", "sub-B"])
    assert not result["passed"]
    assert result["missing_subjects"] == ["sub-B"]
    assert any(row["issue_type"] == "allowlisted_subject_missing" for row in result["issues"])


def test_exact_age_prefers_participants_tsv(tmp_path):
    root = make_dataset(tmp_path, ["sub-A"])
    resolved = resolve_age_labels(root, ["sub-A"])
    assert resolved["exact_age_labels_ready"]
    assert resolved["labels"]["sub-A"]["age"] == 25
    assert resolved["labels"]["sub-A"]["age_source"].endswith("participants.tsv")


def test_exact_age_local_public_metadata_fallback_is_explicit(tmp_path):
    root = make_dataset(tmp_path, ["sub-A"])
    write_root(root, [("sub-A", "n/a", "n/a")])
    fallback = root.parents[1] / "MIPDB_PublicFile.csv"
    fallback.write_text("ID,Age,Sex\nA,31,1\n", encoding="utf-8")
    resolved = resolve_age_labels(root, ["sub-A"])
    assert resolved["exact_age_labels_ready"]
    assert resolved["labels"]["sub-A"]["age"] == 31
    assert resolved["labels"]["sub-A"]["sex_status"] == "raw_numeric_code_unmapped_no_local_schema"
    assert resolved["labels"]["sub-A"]["age_source"] == str(fallback)


def test_missing_exact_age_fails_label_validation(tmp_path):
    root = make_dataset(tmp_path, ["sub-A"])
    write_root(root, [("sub-A", "n/a", "n/a")])
    resolved = resolve_age_labels(root, ["sub-A"])
    assert not resolved["exact_age_labels_ready"]
    assert resolved["exact_age_label_validation"] == "failed"


def test_vhdr_binary_geometry_and_size_mismatch(tmp_path):
    root = make_dataset(tmp_path, [])
    vhdr = write_triplet(root, "sub-A", samples=10, channels=2, remainder=1)
    parsed = parse_vhdr(vhdr)
    binary = vhdr.with_suffix(".eeg")
    assert binary_geometry(parsed, binary.stat().st_size) == (10, 1)
    result = preflight_dataset(root, ["sub-A"])
    assert not result["passed"]
    assert any(row["issue_type"] == "binary_frame_size_mismatch" for row in result["issues"])


def test_missing_companion_fails_preflight(tmp_path):
    root = make_dataset(tmp_path, [])
    write_triplet(root, "sub-A", marker=False)
    result = preflight_dataset(root, ["sub-A"])
    assert not result["passed"]
    assert any(row["issue_type"] == "brainvision_companion_missing" for row in result["issues"])


def test_empty_and_temp_files_fail_preflight(tmp_path):
    root = make_dataset(tmp_path, ["sub-A"])
    eeg = next((root / "sub-A").rglob("*.eeg"))
    eeg.write_bytes(b"")
    (root / "sub-A" / "eeg" / "download.part").write_bytes(b"partial")
    result = preflight_dataset(root, ["sub-A"])
    kinds = {row["issue_type"] for row in result["issues"]}
    assert not result["passed"]
    assert "file_empty" in kinds
    assert "partial_or_temp_file" in kinds


def test_manifest_size_mismatch_fails_preflight(tmp_path):
    root = make_dataset(tmp_path, ["sub-A"])
    eeg = next((root / "sub-A").rglob("*.eeg"))
    manifest_dir = root / ".nemar"
    manifest_dir.mkdir()
    manifest_dir.joinpath("availability-report.json").write_text(
        json.dumps({"files": [{"path": eeg.relative_to(root).as_posix(), "size": eeg.stat().st_size + 4}]}), encoding="utf-8"
    )
    result = preflight_dataset(root, ["sub-A"])
    assert not result["passed"]
    assert any(row["issue_type"] == "manifest_size_mismatch" for row in result["issues"])


def test_release_manifest_excludes_unpublished_allowlist_subject_without_failing(tmp_path):
    root = make_dataset(tmp_path, ["sub-A"])
    eeg = next((root / "sub-A").rglob("*.eeg"))
    manifest = root.parents[1] / "manifest_v1.0.2.json"
    manifest.write_text(
        json.dumps([
            {"path": eeg.relative_to(root).as_posix(), "size": eeg.stat().st_size},
            {"path": "sub-A/eeg/sub-A_task-block01_events.tsv", "size": 10},
        ]),
        encoding="utf-8",
    )
    result = preflight_dataset(root, ["sub-A", "sub-B"], manifest)
    assert result["passed"]
    assert result["audit_subjects"] == ["sub-A"]
    assert result["not_published_subjects"] == ["sub-B"]
    assert result["optional_manifest_missing_count"] == 1
    by_subject = {row["subject_id"]: row for row in result["subjects"]}
    assert by_subject["sub-B"]["download_status"] == "not_published_in_release_manifest"


def test_release_manifest_missing_published_subject_still_fails_closed(tmp_path):
    root = make_dataset(tmp_path, ["sub-A"])
    manifest = root.parents[1] / "manifest_v1.0.2.json"
    manifest.write_text(
        json.dumps([{"path": "sub-B/eeg/sub-B_task-block01_eeg.eeg", "size": 80}]),
        encoding="utf-8",
    )
    result = preflight_dataset(root, ["sub-A", "sub-B"], manifest)
    assert not result["passed"]
    assert result["missing_published_subjects"] == ["sub-B"]
    assert any(row["issue_type"] == "published_allowlisted_subject_missing" for row in result["issues"])


def test_sampling_is_deterministic_capped_and_forces_priority_recordings():
    p1 = PRIORITY_RECORDINGS[0][:-4] + ".vhdr"
    p2 = PRIORITY_RECORDINGS[1][:-4] + ".vhdr"
    rows = [
        fake_recording("sub-A00058775", "block01", p1, 100, "sig1", 23, "M"),
        fake_recording("sub-A00062842", "block03", p2, 200, "sig2", 25, "F"),
    ]
    for index in range(20):
        sid = f"sub-X{index:02d}"
        rows.append(fake_recording(sid, f"block{index % 6 + 1:02d}", f"{sid}/eeg/{sid}_task-block{index % 6 + 1:02d}_eeg.vhdr", 50 + index, f"sig{index % 3}", 20 + index % 16, "F" if index % 2 else "M"))
    first = build_sampling_plan(rows, max_recordings=12, smoke_subjects=5, exact_age_ready=True, sex_ready=True)
    second = build_sampling_plan(rows, max_recordings=12, smoke_subjects=5, exact_age_ready=True, sex_ready=True)
    assert first == second
    assert len(first) <= 12
    assert {p1, p2} <= {row["source_path"] for row in first}
    assert len({row["header_signature"] for row in first}) >= 2
    assert len({row["subject_id"] for row in first}) == 5
    sampled_ages = {float(row["exact_age"]) for row in first}
    assert min(sampled_ages) == 20
    assert max(sampled_ages) == 35


def test_sampling_without_age_sex_records_non_stratified():
    rows = [fake_recording(f"sub-{i}", f"block{i}", f"sub-{i}/eeg/r{i}.vhdr", i + 10) for i in range(5)]
    plan = build_sampling_plan(rows, max_recordings=4, smoke_subjects=3)
    assert len(plan) <= 4
    assert all(row["age_sex_stratified"] is False for row in plan)


def test_running_stats_flags_nan_inf_zero_and_constant():
    zero = RunningStats()
    zero.update(np.zeros(8))
    constant = RunningStats()
    constant.update(np.ones(8))
    nonfinite = RunningStats()
    nonfinite.update(np.array([0.0, np.nan, np.inf]))
    assert channel_flags(zero)["all_zero"]
    assert channel_flags(constant)["strict_constant"]
    assert channel_flags(nonfinite)["has_nan"] and channel_flags(nonfinite)["has_inf"]


def test_source_snapshot_unchanged(tmp_path):
    root = make_dataset(tmp_path, ["sub-A"])
    before = snapshot_tree(root)
    after = snapshot_tree(root)
    assert compare_snapshots(before, after) == (True, False, False)


def test_output_directory_cannot_overwrite(tmp_path):
    ensure_new_output(tmp_path)
    with pytest.raises(FileExistsError):
        ensure_new_output(tmp_path)


def test_signal_smoke_limit_validation():
    rows = [fake_recording(f"sub-{i}", f"block{i}", f"sub-{i}/eeg/r{i}.vhdr", i + 10) for i in range(20)]
    with pytest.raises(ValueError):
        build_sampling_plan(rows, max_recordings=13)


def test_adaptive_expansion_is_never_auto_executed():
    rows = [{"recording": "r1", "fully_decoded": False, "samples_match_header": False}]
    decision = recommend_expansion(rows, [])[0]
    assert decision["adaptive_expansion_triggered"]
    assert decision["approval_required"]
    assert not decision["auto_executed"]


def test_fail_closed_run_writes_all_required_outputs_without_signal(tmp_path):
    root = make_dataset(tmp_path, ["sub-A"])
    allowlist = tmp_path / "allow.txt"
    allowlist.write_text("sub-A\nsub-B\n", encoding="utf-8")
    args = argparse.Namespace(
        data_dir=str(root), subject_allowlist=str(allowlist), output_dir=str(tmp_path / "out"),
        chunk_seconds=30.0, max_signal_recordings=12, smoke_subjects=5, skip_signal_smoke=False,
    )
    audit = Audit(args)
    output = audit.run()
    assert not audit.preflight_passed
    required = {
        "audit_report.md", "audit_summary.json", "processing_provenance.json", "joint_training_readiness.json",
        "audit_errors.log", *CSV_FIELDS.keys(),
    }
    assert required <= {path.name for path in output.iterdir()}
    assert list(csv.reader((output / "signal_smoke_recordings.csv").open(encoding="utf-8-sig")))[0] == CSV_FIELDS["signal_smoke_recordings.csv"]
    summary = json.loads((output / "audit_summary.json").read_text(encoding="utf-8"))
    assert summary["signal_smoke_status"] == "not_run_blocked_by_preflight"
    assert summary["source_data_unchanged"]
