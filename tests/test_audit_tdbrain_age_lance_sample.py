import csv
import importlib.util
import json
from pathlib import Path

import pyarrow as pa
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "audit_tdbrain_age_lance_sample.py"
SPEC = importlib.util.spec_from_file_location("tdbrain_age_audit", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(audit)


def make_args(tmp_path, *extra):
    return audit.parse_args([
        "--source-uri", "cos://bucket/path", "--output-dir", str(tmp_path / "out"),
        "--metadata-only", "--stop-if-age-unconfirmed", *extra,
    ])


@pytest.mark.parametrize("flag,value", [
    ("--max-head-rows", "11"), ("--max-subjects", "4"), ("--max-rows-per-subject", "4"),
])
def test_hard_limits_reject_expansion(tmp_path, flag, value):
    with pytest.raises(ValueError):
        audit.enforce_limits(make_args(tmp_path, flag, value))


def test_metadata_only_and_stop_flags_are_mandatory(tmp_path):
    args = audit.parse_args(["--source-uri", "cos://bucket/path", "--output-dir", str(tmp_path / "out")])
    with pytest.raises(ValueError):
        audit.enforce_limits(args)


def test_age_field_detection_and_opaque_label_gate():
    schema = pa.schema([pa.field("subject_id", pa.string()), pa.field("age", pa.int32())])
    candidates, source = audit.detect_age_candidates(schema, [])
    assert candidates[0]["semantics_documented"] is True
    assert source == "field_name"
    opaque = pa.schema([pa.field("label", pa.int32())])
    candidates, source = audit.detect_age_candidates(opaque, [])
    assert candidates[0]["semantics_documented"] is False
    assert source is None


def test_binned_age_is_detected_from_field_metadata():
    field = pa.field("age", pa.string(), metadata={b"description": b"age bin such as 20-25"})
    candidates, _ = audit.detect_age_candidates(pa.schema([field]), [])
    assert candidates[0]["is_binned"] is True


def test_documented_depression_label_is_not_an_age_candidate():
    schema = pa.schema([pa.field("label", pa.int32())])
    sidecars = [{
        "uri": "metadata.toml",
        "text": 'task = "Depression detection"\ndescription = "Classification label"\nname = "control"\nname = "depressed"',
    }]
    candidates, source = audit.detect_age_candidates(schema, sidecars)
    assert candidates == []
    assert source is None


def test_projection_excludes_high_dimensional_columns():
    schema = pa.schema([
        pa.field("sample_id", pa.string()), pa.field("subject_id", pa.string()),
        pa.field("age", pa.int32()), pa.field("data", pa.list_(pa.float32())),
        pa.field("feature_tensor", pa.list_(pa.float32())), pa.field("channel_mask", pa.list_(pa.bool_())),
    ])
    projection = audit.projection_for(schema, "age")
    assert projection == ["sample_id", "subject_id", "age"]
    assert all(not audit.FORBIDDEN_COLUMNS.search(name) for name in projection)


@pytest.mark.parametrize("value,error", [
    (None, "null"), (-1, "placeholder"), (0, "placeholder"), (999, "placeholder"),
    ("20-25", "bin"), ("unknown", "placeholder"), (42, None), (42.5, None),
])
def test_age_value_classification(value, error):
    _, actual = audit.normalize_age(value)
    assert actual == error


def test_limited_consistency_never_expands_beyond_head_subjects():
    rows = [
        {"subject_id": "S1", "session_id": "A", "age": 20},
        {"subject_id": "S1", "session_id": "A", "age": 20},
        {"subject_id": "S2", "session_id": "A", "age": 30},
        {"subject_id": "S3", "session_id": "A", "age": 40},
    ]
    output, subjects = audit.limited_consistency(rows, "age", 2, 1)
    assert subjects == 2
    assert sum(int(row["rows_checked"]) for row in output) == 2


def test_empty_csv_outputs_have_stable_headers(tmp_path):
    out = tmp_path / "audit"
    audit.initialize_outputs(out)
    with (out / "sampled_metadata_rows.csv").open(encoding="utf-8-sig", newline="") as handle:
        assert next(csv.reader(handle)) == audit.SAMPLED_HEADERS
    with (out / "age_consistency_sample.csv").open(encoding="utf-8-sig", newline="") as handle:
        assert next(csv.reader(handle)) == audit.CONSISTENCY_HEADERS


def test_cos_operations_are_read_only(monkeypatch):
    calls = []
    def fake_run(command, **kwargs):
        calls.append(command)
        class Result:
            returncode = 0
            stdout = ""
            stderr = ""
        return Result()
    monkeypatch.setattr(audit.subprocess, "run", fake_run)
    audit.run_coscli("coscli", ["ls", "cos://bucket/path/", "--limit", "10"], [])
    with pytest.raises(AssertionError):
        audit.run_coscli("coscli", ["rm", "cos://bucket/path/x"], [])
    with pytest.raises(AssertionError):
        audit.run_coscli("coscli", ["cp", "local", "cos://bucket/path/x"], [])
    assert calls[0][1] == "ls"


def test_explicit_cos_endpoint_is_passed_to_coscli(monkeypatch):
    calls = []
    def fake_run(command, **kwargs):
        calls.append(command)
        class Result:
            returncode = 0
            stdout = ""
            stderr = ""
        return Result()
    monkeypatch.setattr(audit.subprocess, "run", fake_run)
    audit.run_coscli(
        "coscli",
        ["ls", "cos://bucket/path/", "--limit", "10"],
        [],
        "cos.ap-beijing.myqcloud.com",
    )
    assert calls == [[
        "coscli", "ls", "cos://bucket/path/", "--limit", "10",
        "-e", "cos.ap-beijing.myqcloud.com", "--disable-log",
    ]]


def test_coscli_table_output_is_parsed():
    text = """KEY | TYPE | LAST MODIFIED | ETAG | SIZE | RESTORESTATUS
-------------------------------------------------------------+----------
  omni-downstream-dataset/lance_2.0/TDBRAIN/dataset.lance/ | DIR | | | |
  omni-downstream-dataset/lance_2.0/TDBRAIN/metadata.toml | STANDARD | x | y | 3.91 KB |
TOTAL OBJECTS: | 2
"""
    rows = audit.parse_ls_output(
        text,
        "cos://omni-eeg-bench-1442740494/omni-downstream-dataset/lance_2.0/TDBRAIN",
    )
    assert rows == [
        {
            "uri": "cos://omni-eeg-bench-1442740494/omni-downstream-dataset/lance_2.0/TDBRAIN/dataset.lance/",
            "is_dir": True,
            "size_bytes": None,
        },
        {
            "uri": "cos://omni-eeg-bench-1442740494/omni-downstream-dataset/lance_2.0/TDBRAIN/metadata.toml",
            "is_dir": False,
            "size_bytes": int(3.91 * 1024),
        },
    ]


def test_base_summary_proves_no_signal_or_full_scan(tmp_path):
    summary = audit.base_summary(make_args(tmp_path))
    assert summary["signal_column_materialized"] is False
    assert summary["feature_column_materialized"] is False
    assert summary["data_tensor_materialized"] is False
    assert summary["full_fragment_scan_performed"] is False
    assert summary["full_dataset_downloaded"] is False
    assert summary["cos_objects_modified"] is False
