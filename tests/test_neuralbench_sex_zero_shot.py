from __future__ import annotations

import hashlib
import logging
import sys
from argparse import Namespace
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

PROJECT = Path(__file__).resolve().parents[1]
MODULE_DIR = PROJECT / "scripts" / "downstream_50m"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from scripts.downstream_50m.dataset import collate_fn
from scripts.downstream_50m.largeScale_fineTune import EEGDownstreamClassifier
from scripts.downstream_50m import neuralbench_checkpoint as checkpointing
from scripts.downstream_50m import run_neuralbench_sex_zero_shot as zero_shot
from scripts.downstream_50m.sex_cv4 import aggregate_subject_predictions


class TinyBackbone(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.proj = nn.Linear(2, 512, bias=True, device=device)

    def extract_token_embeddings(
        self, token_inputs, token_channel_indices, token_time_indices, return_layer_idx=None
    ):
        del token_channel_indices, token_time_indices, return_layer_idx
        return self.proj(token_inputs)


def _meta_model():
    with torch.device("meta"):
        return EEGDownstreamClassifier(
            TinyBackbone(), 512, 2, probe_layer_idx=8, pooling="mean"
        )


def _source_state(include_loss=True):
    state = {
        "model.wrapped_model.backbone.proj.weight": torch.arange(1024, dtype=torch.float32).reshape(512, 2) / 1024,
        "model.wrapped_model.backbone.proj.bias": torch.linspace(-0.1, 0.1, 512),
        "model.probe.weight": torch.arange(1024, dtype=torch.float32).reshape(2, 512) / 2048,
        "model.probe.bias": torch.tensor([-0.25, 0.25], dtype=torch.float32),
    }
    if include_loss:
        state["loss.weight"] = torch.tensor([1.0, 1.0])
    return state


def _write_checkpoint(path: Path, state=None):
    torch.save({"state_dict": _source_state() if state is None else state}, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_safe_loader_uses_weights_only_and_cpu(monkeypatch, tmp_path):
    path = tmp_path / "fake.ckpt"
    path.write_bytes(b"safe-bytes")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    called = {}

    def fake_load(actual_path, **kwargs):
        called["path"] = actual_path
        called.update(kwargs)
        return {"state_dict": {"x": torch.ones(1)}}

    monkeypatch.setattr(checkpointing.torch, "load", fake_load)
    checkpoint, info = checkpointing.safe_load_neuralbench_checkpoint(path, digest)
    assert checkpoint["state_dict"]["x"].item() == 1
    assert info["sha256"] == digest
    assert called["map_location"] == "cpu"
    assert called["weights_only"] is True


def test_sha_mismatch_fails_before_torch_load(monkeypatch, tmp_path):
    path = tmp_path / "fake.ckpt"
    path.write_bytes(b"not-authoritative")
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("torch.load must not run")

    monkeypatch.setattr(checkpointing.torch, "load", forbidden)
    with pytest.raises(checkpointing.NeuralBenchCheckpointError, match="SHA256 mismatch"):
        checkpointing.safe_load_neuralbench_checkpoint(path, "0" * 64)
    assert called is False


def test_explicit_backbone_remap_and_upstream_head_probe_separation():
    target = {
        "backbone.tokenizer.weight": torch.empty(1),
        "backbone.head.weight": torch.empty(3, 3),
        "head.weight": torch.empty(2, 512),
        "head.bias": torch.empty(2),
    }
    source = {
        "model.wrapped_model.backbone.tokenizer.weight": torch.ones(1),
        "model.wrapped_model.backbone.head.weight": torch.ones(3, 3),
        "model.probe.weight": torch.ones(2, 512),
        "model.probe.bias": torch.ones(2),
    }
    mapped, rows, summary = checkpointing.remap_neuralbench_sex_checkpoint(source, target)
    assert torch.equal(mapped["backbone.head.weight"], source["model.wrapped_model.backbone.head.weight"])
    assert torch.equal(mapped["head.weight"], source["model.probe.weight"])
    components = {row["source_key"]: row["component"] for row in rows}
    assert components["model.wrapped_model.backbone.head.weight"] == "backbone_upstream_head"
    assert components["model.probe.weight"] == "downstream_sex_probe"
    assert summary["probe_key_match_fraction"] == 1.0


def test_probe_shape_and_loss_weight_allowed_ignore():
    mapped, rows, summary = checkpointing.remap_neuralbench_sex_checkpoint(
        _source_state(), _meta_model().state_dict()
    )
    assert mapped["head.weight"].shape == (2, 512)
    loss_row = next(row for row in rows if row["source_key"] == "loss.weight")
    assert loss_row["status"] == "ignored_allowed"
    assert summary["ignored_source_keys"] == ["loss.weight"]


@pytest.mark.parametrize("failure", ["missing", "shape", "dtype", "unexpected"])
def test_mapping_fails_closed_for_all_incompatibilities(failure):
    source = _source_state()
    target = _meta_model().state_dict()
    if failure == "missing":
        source.pop("model.wrapped_model.backbone.proj.bias")
    elif failure == "shape":
        source["model.wrapped_model.backbone.proj.bias"] = torch.zeros(511)
    elif failure == "dtype":
        source["model.wrapped_model.backbone.proj.bias"] = torch.zeros(512, dtype=torch.float64)
    else:
        source["unexplained.trainable_tensor"] = torch.zeros(1)
    with pytest.raises(checkpointing.NeuralBenchCheckpointError, match="not exact"):
        checkpointing.remap_neuralbench_sex_checkpoint(source, target)


def test_full_model_is_populated_only_from_checkpoint_and_probe_equal(tmp_path):
    path = tmp_path / "fake.ckpt"
    digest = _write_checkpoint(path)
    model = _meta_model()
    _, _, _, summary = checkpointing.load_neuralbench_sex_model_state(model, path, digest)
    assert summary["all_model_state_tensors_from_checkpoint"] is True
    assert summary["random_materialized_parameter_count"] == 0
    assert not any(value.is_meta for value in model.state_dict().values())
    assert torch.equal(model.head.weight.detach(), _source_state()["model.probe.weight"])
    assert torch.equal(model.head.bias.detach(), _source_state()["model.probe.bias"])


def test_label_mapping_and_masked_mean_contract():
    assert zero_shot.LABEL_MAPPING == {"female": 0, "male": 1}
    assert zero_shot.normalize_sex("F") == (0, "female")
    assert zero_shot.normalize_sex("M") == (1, "male")
    model = EEGDownstreamClassifier(TinyBackbone(), 512, 2, probe_layer_idx=8, pooling="mean")
    embeddings = torch.zeros(1, 3, 512)
    embeddings[0, 0, 0] = 2
    embeddings[0, 1, 0] = 4
    embeddings[0, 2, 0] = 1000
    pooled = model.pool_valid_tokens(embeddings, torch.tensor([[1, 1, 0]]))
    assert pooled[0, 0].item() == 3


def test_fake_checkpoint_forward_parity(tmp_path):
    path = tmp_path / "fake.ckpt"
    source = _source_state()
    digest = _write_checkpoint(path, source)
    model = _meta_model()
    checkpointing.load_neuralbench_sex_model_state(model, path, digest)
    model.eval()
    tokens = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]]])
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    indices = torch.zeros(1, 3, dtype=torch.long)
    with torch.inference_mode():
        actual = model(tokens, indices, indices, mask)
        embedded = torch.nn.functional.linear(
            tokens, source["model.wrapped_model.backbone.proj.weight"],
            source["model.wrapped_model.backbone.proj.bias"],
        )
        pooled = (embedded * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        expected = torch.nn.functional.linear(
            pooled, source["model.probe.weight"], source["model.probe.bias"]
        )
    assert torch.equal(actual, expected)


class FixtureDataset(Dataset):
    def __init__(self):
        self.rows = []
        specs = [
            ("s0", "SUB_F", 0, -2.0),
            ("s1", "SUB_F", 0, -1.0),
            ("s2", "SUB_M", 1, 1.0),
            ("s3", "SUB_M", 1, 2.0),
        ]
        for index, (sample, subject, label, signal) in enumerate(specs):
            tokens = torch.zeros(3, 512)
            tokens[:2, 0] = signal
            tokens[2, 0] = 9999
            self.rows.append({
                "sample_id": sample, "subject_id": subject, "recording_id": "R1",
                "window_index": torch.tensor(index), "label": torch.tensor(label),
                "label_name": zero_shot.LABEL_TO_SEX[label], "token_inputs": tokens,
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


class CountingModel(EEGDownstreamClassifier):
    def __init__(self):
        super().__init__(IdentityBackbone(), 512, 2, probe_layer_idx=8, pooling="mean")
        self.forward_calls = 0
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.bias.zero_()
            self.head.weight[0, 0] = -1
            self.head.weight[1, 0] = 1

    def forward(self, *args, **kwargs):
        self.forward_calls += 1
        return super().forward(*args, **kwargs)


def test_fixture_single_pass_probabilities_outputs_and_metrics(tmp_path):
    dataset = FixtureDataset()
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
    model = CountingModel().eval()
    logger = logging.getLogger("fixture-zero-shot")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    rows, forward_count = zero_shot.run_zero_shot_inference(
        model, loader, torch.device("cpu"), logger, 1
    )
    assert forward_count == model.forward_calls == 2
    assert len(rows) == 4
    assert [row["sample_id"] for row in rows] == ["s0", "s1", "s2", "s3"]
    assert all(row["correct"] for row in rows)
    window, probability, majority, subjects = zero_shot.write_zero_shot_artifacts(
        tmp_path, rows, ["s0", "s1", "s2", "s3"]
    )
    assert window["accuracy"] == probability["accuracy"] == majority["accuracy"] == 1.0
    assert window["sample_count"] == 4
    assert len(subjects) == 2
    assert {path.name for path in tmp_path.iterdir()} == {
        "window_predictions.csv", "subject_predictions.csv", "window_metrics.json",
        "subject_probability_mean_metrics.json", "subject_majority_vote_metrics.json",
    }


def test_subject_majority_tie_uses_probability_mean_then_class_zero():
    rows = [
        {"subject_id": "S", "true_label": 0, "prob_female": 0.9, "prob_male": 0.1, "predicted_label": 0},
        {"subject_id": "S", "true_label": 0, "prob_female": 0.4, "prob_male": 0.6, "predicted_label": 1},
    ]
    subject = aggregate_subject_predictions(rows)[0]
    assert subject["tie_break_used"] is True
    assert subject["majority_vote_pred_label"] == 0
    exact = [dict(rows[0], prob_female=0.6, prob_male=0.4), dict(rows[1], prob_female=0.4, prob_male=0.6)]
    assert aggregate_subject_predictions(exact)[0]["majority_vote_pred_label"] == 0


def test_window_round_trip_accepts_saved_false_correctness():
    row = {
        "sample_id": "wrong", "subject_id": "S", "recording_id": "R", "window_index": "0",
        "true_label": "0", "true_sex": "female", "raw_logit_female": "-1",
        "raw_logit_male": "1", "prob_female": "0.1", "prob_male": "0.9",
        "predicted_label": "1", "predicted_sex": "male", "correct": "False",
    }
    zero_shot.validate_window_rows([row], ["wrong"])


def test_dry_run_does_not_scan_rows_or_create_output(monkeypatch, tmp_path):
    checkpoint_path = tmp_path / "fake.ckpt"
    checkpoint_path.write_bytes(b"x")
    digest = hashlib.sha256(b"x").hexdigest()
    monkeypatch.setattr(zero_shot, "_inspect_lance", lambda path: {"version": 1, "row_count": 4})
    monkeypatch.setattr(
        zero_shot, "NeuralBenchSexZeroShotDataset",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not scan rows")),
    )
    output = tmp_path / "formal-output"
    args = Namespace(
        input_lance=str(tmp_path / "fixture.lance"), checkpoint=str(checkpoint_path),
        expected_checkpoint_sha256=digest, output_dir=str(output), device="cpu",
        batch_size=2, num_workers=0, log_level="INFO", log_every_batches=1,
        dry_run=True,
    )
    assert zero_shot.run(args) == 0
    assert not output.exists()


def test_zero_shot_source_has_no_execution_path_for_update_mechanisms():
    source = Path(zero_shot.__file__).read_text(encoding="utf-8")
    assert "torch.optim" not in source
    assert ".backward(" not in source
    assert ".train(" not in source
    assert "run_sex_cv4_sweep" not in source
    assert "feature_cache" not in source.replace('"feature_cache_created": False', "")


def test_cloud_script_is_isolated_and_has_required_safety_controls():
    script = (Path(__file__).parents[1] / "commands" / "run_50m_neuralbench_sex_zero_shot_full.sh").read_text(encoding="utf-8")
    lower = script.lower()
    assert "set -euo pipefail" in script
    assert "PYTHONUNBUFFERED=1" in script
    assert '"$PYTHON_BIN" -u' in script
    assert "run_sex_cv4" not in lower
    assert "sweep" not in lower
    assert "coscli" not in lower and "cos://" not in lower
    assert "torch.optim" not in lower and "backward" not in lower
