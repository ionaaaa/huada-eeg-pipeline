from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from scripts.downstream_50m import audit_neuralbench_sex_checkpoint as audit


def _source_state():
    return {
        "model.wrapped_model.backbone.tokenizer.proj.0.weight": torch.zeros(4, 2),
        "model.wrapped_model.backbone.encoder.encoder.layers.0.norm1.weight": torch.ones(4),
        "model.wrapped_model.backbone.head.head.0.weight": torch.zeros(4, 4),
        "model.probe.weight": torch.zeros(2, 4),
        "model.probe.bias": torch.zeros(2),
        "loss.weight": torch.ones(2),
    }


def _target_states():
    backbone = {
        "tokenizer.proj.0.weight": torch.zeros(4, 2),
        "encoder.encoder.layers.0.norm1.weight": torch.ones(4),
        "head.head.0.weight": torch.zeros(4, 4),
    }
    head = {"weight": torch.zeros(2, 4), "bias": torch.zeros(2)}
    return backbone, head


def test_explicit_key_mapping_and_probe_upstream_head_separation():
    backbone, head = _target_states()
    rows, summary = audit.build_key_mapping(_source_state(), backbone, head)
    indexed = {row["source_key"]: row for row in rows if row["source_key"]}
    assert indexed["model.wrapped_model.backbone.head.head.0.weight"]["component"] == "backbone_upstream_head"
    assert indexed["model.probe.weight"]["component"] == "downstream_sex_probe"
    assert indexed["model.probe.weight"]["target_key"] == "head.weight"
    assert indexed["loss.weight"]["status"] == "ignored"
    assert summary["core_backbone_key_match_fraction"] == 1.0
    assert summary["full_backbone_parameter_match_fraction"] == 1.0


def test_mapping_reports_missing_unexpected_and_shape_mismatch():
    source = _source_state()
    source["model.wrapped_model.backbone.unknown.weight"] = torch.zeros(3)
    source["other.weight"] = torch.zeros(1)
    backbone, head = _target_states()
    backbone["tokenizer.proj.0.weight"] = torch.zeros(5, 2)
    backbone["target_only.weight"] = torch.zeros(7)
    rows, summary = audit.build_key_mapping(source, backbone, head)
    statuses = {(row["source_key"], row["target_key"]): row["status"] for row in rows}
    assert statuses[("model.wrapped_model.backbone.tokenizer.proj.0.weight", "backbone.tokenizer.proj.0.weight")] == "shape_mismatch"
    assert statuses[("model.wrapped_model.backbone.unknown.weight", "backbone.unknown.weight")] == "missing_target"
    assert statuses[("other.weight", "")] == "unexpected"
    assert "backbone.target_only.weight" in summary["missing_source"]


def test_label_mapping_fails_closed_without_metadata():
    result = audit.resolve_label_mapping(
        predefined_mapping=None, observed_labels=None, label_meanings=None, evidence=[]
    )
    assert result["resolved"] is False
    assert result["dataset_metadata_required"] is True
    assert result["probe_row_swap_required"] is None


def test_label_mapping_and_class_row_swap():
    result = audit.resolve_label_mapping(
        predefined_mapping=None,
        observed_labels=["M", "F"],
        label_meanings={"F": "male", "M": "female"},
        evidence=[],
    )
    assert result["source_mapping"] == {"male": 0, "female": 1}
    assert result["target_row_from_source_row"] == [1, 0]
    assert result["probe_row_swap_required"] is True
    weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    bias = torch.tensor([5.0, 6.0])
    new_weight, new_bias = audit.normalize_probe_rows(weight, bias, [1, 0])
    assert torch.equal(new_weight, weight[[1, 0]])
    assert torch.equal(new_bias, bias[[1, 0]])


def test_raw_and_normalized_logits_are_distinct_when_rows_swap():
    raw = torch.tensor([[0.25, 0.75]])
    normalized = audit.normalize_logits(raw, [1, 0])
    assert torch.equal(normalized, torch.tensor([[0.75, 0.25]]))
    assert not torch.equal(raw, normalized)


def test_parity_stats_reports_values_without_timing_dependency():
    left = torch.tensor([[1.0, 2.0]])
    right = torch.tensor([[1.0, 2.0 + 1e-7]])
    result = audit.parity_stats(left, right, atol=1e-5, rtol=1e-5)
    assert result["shape_equal"] is True
    assert result["dtype_equal"] is True
    assert result["finite"] is True
    assert result["allclose"] is True
    assert result["max_abs_diff"] >= 0.0


def test_safe_load_checkpoint_uses_weights_only_and_cpu(monkeypatch, tmp_path):
    called = {}

    def fake_load(path, **kwargs):
        called.update(kwargs)
        return {"state_dict": {"x": torch.ones(1)}}

    monkeypatch.setattr(audit.torch, "load", fake_load)
    result = audit.safe_load_checkpoint(tmp_path / "fake.ckpt")
    assert "state_dict" in result
    assert called == {"map_location": "cpu", "weights_only": True}


def test_safe_read_does_not_modify_checkpoint(tmp_path):
    path = tmp_path / "checkpoint.pt"
    torch.save({"state_dict": {"x": torch.arange(4)}}, path)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    checkpoint = audit.safe_load_checkpoint(path)
    assert torch.equal(checkpoint["state_dict"]["x"], torch.arange(4))
    after = hashlib.sha256(path.read_bytes()).hexdigest()
    assert before == after


def test_run_directory_resolution_requires_unique_match(tmp_path):
    base = tmp_path / "runs"
    base.mkdir()
    prefix = "seed=33,task_name=sex,brain_model_output_size=2"
    suffix = "823-56ab9fd6"
    run = base / f"{prefix},anything...{suffix}"
    run.mkdir()
    (run / "best.ckpt").write_bytes(b"x")
    (run / "config.yaml").write_text("task_name: sex\n", encoding="utf-8")
    assert audit.resolve_source_run_dir(base, prefix, suffix) == run.resolve()
    other = base / f"{prefix},other...{suffix}"
    other.mkdir()
    (other / "best.ckpt").write_bytes(b"x")
    (other / "config.yaml").write_text("task_name: sex\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="exactly one"):
        audit.resolve_source_run_dir(base, prefix, suffix)
