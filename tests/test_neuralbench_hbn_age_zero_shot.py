from __future__ import annotations

import hashlib
import json
import logging
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

PROJECT = Path(__file__).resolve().parents[1]
MODULE_DIR = PROJECT / "scripts" / "downstream_50m"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from scripts.downstream_50m import neuralbench_checkpoint as checkpointing
from scripts.downstream_50m import run_neuralbench_hbn_age_zero_shot as age_zero_shot
from scripts.downstream_50m.dataset import collate_fn
from scripts.downstream_50m.largeScale_fineTune import EEGDownstreamClassifier
from scripts.downstream_50m.neuralbench_zero_shot_common import model_state_fingerprint


class TinyBackbone(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.proj = nn.Linear(2, 512, bias=True, device=device)

    def extract_token_embeddings(
        self, token_inputs, token_channel_indices, token_time_indices, return_layer_idx=None
    ):
        del token_channel_indices, token_time_indices, return_layer_idx
        return self.proj(token_inputs)


def _meta_age_model():
    with torch.device("meta"):
        return EEGDownstreamClassifier(
            TinyBackbone(), 512, 1, probe_layer_idx=8, pooling="mean"
        )


def _source_state():
    return {
        "model.wrapped_model.backbone.proj.weight": (
            torch.arange(1024, dtype=torch.float32).reshape(512, 2) / 1024
        ),
        "model.wrapped_model.backbone.proj.bias": torch.linspace(-0.1, 0.1, 512),
        "model.probe.weight": torch.arange(512, dtype=torch.float32).reshape(1, 512) / 512,
        "model.probe.bias": torch.tensor([-0.25], dtype=torch.float32),
    }


def _write_checkpoint(path: Path, state=None):
    torch.save({"state_dict": _source_state() if state is None else state}, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_age_safe_loader_uses_sha_before_weights_only_load(monkeypatch, tmp_path):
    path = tmp_path / "fake.ckpt"
    path.write_bytes(b"safe")
    digest = hashlib.sha256(b"safe").hexdigest()
    calls = []

    def fake_load(actual, **kwargs):
        calls.append((actual, kwargs))
        return {"state_dict": {"x": torch.ones(1)}}

    monkeypatch.setattr(checkpointing.torch, "load", fake_load)
    checkpointing.safe_load_neuralbench_checkpoint(path, digest)
    assert calls[0][1] == {"map_location": "cpu", "weights_only": True}

    calls.clear()
    with pytest.raises(checkpointing.NeuralBenchCheckpointError, match="SHA256 mismatch"):
        checkpointing.safe_load_neuralbench_checkpoint(path, "0" * 64)
    assert calls == []


def test_age_exact_154_backbone_remap_and_probe_contract():
    source = {
        f"model.wrapped_model.backbone.tensor_{index}": torch.tensor([float(index)])
        for index in range(154)
    }
    target = {
        f"backbone.tensor_{index}": torch.empty(1, dtype=torch.float32)
        for index in range(154)
    }
    source.update({
        "model.probe.weight": torch.ones(1, 512),
        "model.probe.bias": torch.ones(1),
    })
    target.update({"head.weight": torch.empty(1, 512), "head.bias": torch.empty(1)})
    mapped, rows, summary = checkpointing.remap_neuralbench_hbn_age_checkpoint(
        source, target
    )
    assert len(mapped) == 156
    assert summary["source_backbone_tensor_count"] == 154
    assert summary["source_probe_tensor_count"] == 2
    assert summary["full_backbone_key_match_fraction"] == 1.0
    assert summary["probe_parameter_match_fraction"] == 1.0
    assert summary["all_float_tensors_finite"] is True
    assert next(row for row in rows if row["source_key"] == "model.probe.weight")[
        "component"
    ] == "downstream_age_probe"


def test_age_probe_does_not_confuse_sex_or_upstream_head():
    target = {
        "backbone.tokenizer.weight": torch.empty(1),
        "backbone.head.weight": torch.empty(3, 3),
        "head.weight": torch.empty(1, 512),
        "head.bias": torch.empty(1),
    }
    source = {
        "model.wrapped_model.backbone.tokenizer.weight": torch.ones(1),
        "model.wrapped_model.backbone.head.weight": torch.ones(3, 3),
        "model.probe.weight": torch.ones(1, 512),
        "model.probe.bias": torch.ones(1),
    }
    mapped, rows, _ = checkpointing.remap_neuralbench_hbn_age_checkpoint(
        source, target, expected_backbone_tensor_count=2
    )
    assert torch.equal(mapped["backbone.head.weight"], source["model.wrapped_model.backbone.head.weight"])
    assert torch.equal(mapped["head.weight"], source["model.probe.weight"])
    assert next(row for row in rows if row["source_key"].endswith("backbone.head.weight"))[
        "component"
    ] == "backbone_upstream_head"

    sex_source = dict(source)
    sex_source["model.probe.weight"] = torch.ones(2, 512)
    sex_source["model.probe.bias"] = torch.ones(2)
    with pytest.raises(checkpointing.NeuralBenchCheckpointError, match=r"\[1,512\]"):
        checkpointing.remap_neuralbench_hbn_age_checkpoint(
            sex_source, target, expected_backbone_tensor_count=2
        )


@pytest.mark.parametrize("failure", ["missing", "unexpected", "shape", "dtype", "nonfinite"])
def test_age_mapping_fails_closed(failure):
    source = _source_state()
    target = _meta_age_model().state_dict()
    if failure == "missing":
        source.pop("model.wrapped_model.backbone.proj.bias")
    elif failure == "unexpected":
        source["some.suffix.probe.weight"] = torch.zeros(1, 512)
    elif failure == "shape":
        source["model.wrapped_model.backbone.proj.bias"] = torch.zeros(511)
    elif failure == "dtype":
        source["model.wrapped_model.backbone.proj.bias"] = torch.zeros(512, dtype=torch.float64)
    else:
        source["model.wrapped_model.backbone.proj.bias"][0] = float("nan")
    with pytest.raises(checkpointing.NeuralBenchCheckpointError):
        checkpointing.remap_neuralbench_hbn_age_checkpoint(
            source, target, expected_backbone_tensor_count=2
        )


def test_age_full_model_loaded_only_from_checkpoint_and_frozen(tmp_path):
    path = tmp_path / "fake.ckpt"
    digest = _write_checkpoint(path)
    model = _meta_age_model()
    _, _, _, summary = checkpointing.load_neuralbench_hbn_age_model_state(
        model, path, digest, expected_backbone_tensor_count=2
    )
    assert summary["all_model_state_tensors_from_checkpoint"] is True
    assert summary["random_materialized_parameter_count"] == 0
    assert summary["age_probe_parameter_count"] == 513
    assert not any(value.is_meta for value in model.state_dict().values())
    assert torch.equal(model.head.weight.detach(), _source_state()["model.probe.weight"])
    model = age_zero_shot.prepare_model_for_inference(model, torch.device("cpu"))
    assert not any(parameter.requires_grad for parameter in model.parameters())
    assert model.training is False


def test_age_fake_checkpoint_forward_parity_and_masked_mean(tmp_path):
    path = tmp_path / "fake.ckpt"
    source = _source_state()
    digest = _write_checkpoint(path, source)
    model = _meta_age_model()
    checkpointing.load_neuralbench_hbn_age_model_state(
        model, path, digest, expected_backbone_tensor_count=2
    )
    model.eval()
    tokens = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]]])
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    indices = torch.zeros(1, 3, dtype=torch.long)
    with torch.inference_mode():
        actual = model(tokens, indices, indices, mask)
        embedded = torch.nn.functional.linear(
            tokens,
            source["model.wrapped_model.backbone.proj.weight"],
            source["model.wrapped_model.backbone.proj.bias"],
        )
        pooled = (embedded * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        expected = torch.nn.functional.linear(
            pooled, source["model.probe.weight"], source["model.probe.bias"]
        )
    assert actual.shape == (1, 1)
    assert torch.equal(actual, expected)


def test_age_eligibility_excludes_sub5_and_keeps_seven_subjects():
    dataset = object.__new__(age_zero_shot.NeuralBenchHbnAgeZeroShotDataset)
    dataset.excluded_samples = []
    ages = {
        "SUB1": 31.0, "SUB2": 29.0, "SUB3": 26.0, "SUB4": 26.0,
        "SUB5": None, "SUB6": 28.0, "SUB7": 28.0, "SUB8": 22.0,
    }
    dataset._metadata = [
        {"sample_id": f"{subject}_{window}", "subject_id": subject, "age": age}
        for subject, age in ages.items()
        for window in range(2)
    ]
    selected = dataset._select_indices()
    dataset.indices = selected
    dataset.length = len(selected)
    assert len(selected) == 14
    age_zero_shot._validate_authoritative_eligibility(dataset)
    sub5 = next(row for row in dataset.subject_eligibility if row["subject_id"] == "SUB5")
    assert sub5["age"] == ""
    assert sub5["exclusion_reason"] == "missing_age"
    assert len(dataset.excluded_samples) == 2


class FixtureDataset(Dataset):
    def __init__(self):
        self.rows = []
        specs = [
            ("s0", "SUB1", 31.0, -2.0),
            ("s1", "SUB1", 31.0, -1.0),
            ("s2", "SUB8", 22.0, 1.0),
            ("s3", "SUB8", 22.0, 2.0),
        ]
        for index, (sample, subject, age, signal) in enumerate(specs):
            tokens = torch.zeros(3, 512)
            tokens[:2, 0] = signal
            tokens[2, 0] = 9999
            self.rows.append({
                "sample_id": sample,
                "subject_id": subject,
                "recording_id": "R1",
                "window_index": torch.tensor(index),
                "label": torch.tensor(age),
                "label_name": "",
                "token_inputs": tokens,
                "targets": torch.zeros(3, 1, 1),
                "token_channel_indices": torch.zeros(3, dtype=torch.long),
                "token_time_indices": torch.zeros(3, dtype=torch.long),
                "token_valid_mask": torch.tensor([1.0, 1.0, 0.0]),
                "channel_valid_mask": torch.ones(1),
            })

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class IdentityBackbone(nn.Module):
    def extract_token_embeddings(self, token_inputs, **kwargs):
        return token_inputs


class CountingAgeModel(EEGDownstreamClassifier):
    def __init__(self):
        super().__init__(IdentityBackbone(), 512, 1, probe_layer_idx=8, pooling="mean")
        self.forward_calls = 0
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.bias.zero_()
            self.head.weight[0, 0] = 1

    def forward(self, *args, **kwargs):
        self.forward_calls += 1
        return super().forward(*args, **kwargs)


def test_fixture_single_pass_raw_outputs_metrics_and_diagnostics(tmp_path):
    dataset = FixtureDataset()
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
    model = CountingAgeModel().eval()
    before = model_state_fingerprint(model)
    logger = logging.getLogger("fixture-age-zero-shot")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    rows, forward_count = age_zero_shot.run_zero_shot_inference(
        model, loader, torch.device("cpu"), logger, 1
    )
    after = model_state_fingerprint(model)
    assert forward_count == model.forward_calls == 2
    assert before == after
    assert [row["predicted_age"] for row in rows] == [-2.0, -1.0, 1.0, 2.0]
    assert all(row["true_age_outside_source_range"] for row in rows)
    result = age_zero_shot.write_zero_shot_artifacts(
        tmp_path, rows, ["s0", "s1", "s2", "s3"]
    )
    window, subject_mean, subject_median, subjects, diagnostics = result
    assert window["sample_count"] == 4 and window["metric_role"] == "auxiliary"
    assert subject_mean["sample_count"] == subject_median["sample_count"] == 2
    assert len(subjects) == 2
    assert subjects[0]["mean_predicted_age"] == -1.5
    assert subjects[1]["median_predicted_age"] == 1.5
    assert diagnostics["diagnostic_only"] is True
    assert diagnostics["calibration_applied"] is False
    assert diagnostics["linear_fit_applied_to_predictions"] is False
    assert {path.name for path in tmp_path.iterdir()} == {
        "window_predictions.csv", "subject_predictions.csv", "window_metrics.json",
        "subject_mean_metrics.json", "subject_median_metrics.json",
        "extrapolation_diagnostics.json",
    }


def test_regression_metrics_pearson_spearman_and_no_clipping():
    truth = [22.0, 26.0, 28.0, 31.0]
    prediction = [-100.0, -50.0, 0.0, 50.0]
    metrics = age_zero_shot.regression_metrics(truth, prediction)
    rho, defined = age_zero_shot.spearman_rho(truth, prediction)
    assert metrics["pearson_r"] > 0.9
    assert metrics["mean_signed_error"] < 0
    assert rho == pytest.approx(1.0) and defined is True
    assert min(prediction) == -100.0


def test_dry_run_does_not_scan_rows_load_checkpoint_or_create_output(monkeypatch, tmp_path):
    checkpoint_path = tmp_path / "fake.ckpt"
    checkpoint_path.write_bytes(b"x")
    digest = hashlib.sha256(b"x").hexdigest()
    monkeypatch.setattr(
        age_zero_shot, "_inspect_lance", lambda path: {"version": 1, "row_count": 4}
    )
    monkeypatch.setattr(
        age_zero_shot,
        "NeuralBenchHbnAgeZeroShotDataset",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not scan rows")),
    )
    monkeypatch.setattr(
        checkpointing.torch,
        "load",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not load")),
    )
    output = tmp_path / "formal-output"
    args = Namespace(
        input_lance=str(tmp_path / "fixture.lance"),
        checkpoint=str(checkpoint_path),
        expected_checkpoint_sha256=digest,
        output_dir=str(output),
        device="cpu",
        batch_size=2,
        num_workers=0,
        log_level="INFO",
        log_every_batches=1,
        dry_run=True,
    )
    assert age_zero_shot.run(args) == 0
    assert not output.exists()


def test_age_source_has_no_update_calibration_split_or_cache_path():
    source = Path(age_zero_shot.__file__).read_text(encoding="utf-8")
    assert "torch.optim" not in source
    assert ".backward(" not in source
    assert ".train(" not in source
    assert "run_age_cv7" not in source
    assert "run_neuralbench_sex" not in source
    assert "calibrated_prediction" not in source
    assert "feature_cache" not in source.replace('"feature_cache_created": False', "")
    assert "softmax(" not in source and "sigmoid(" not in source


def test_age_cloud_script_is_isolated_and_safe():
    script = (
        Path(__file__).parents[1]
        / "commands/run_50m_neuralbench_hbn_age_zero_shot_full.sh"
    ).read_text(encoding="utf-8")
    lower = script.lower()
    assert "set -euo pipefail" in script
    assert "PYTHONUNBUFFERED=1" in script
    assert '"$PYTHON_BIN" -u' in script
    assert "run_neuralbench_hbn_age_zero_shot.py" in script
    for forbidden in (
        "age_cv7", "legacy", "sex_zero_shot", "probe_transfer", "feature_cache",
        "torch.optim", "backward", "calibration",
    ):
        assert forbidden not in lower


def test_required_provenance_constants_and_source_metrics_are_not_target_results():
    assert checkpointing.DEFAULT_NEURALBENCH_HBN_AGE_SHA256 == (
        "a3438f5b13af4bb3a146e1c84f6ebdba22477d74046170825b98b2ef1be98b1d"
    )
    assert age_zero_shot.RUN_MODE == "external_checkpoint_zero_shot"
    assert age_zero_shot.SOURCE_AGE_MIN == 5.17
    assert age_zero_shot.SOURCE_AGE_MAX == 20.88
    assert age_zero_shot.PROBE_LAYER_IDX == 8
    assert age_zero_shot.POOLING == "mean"
