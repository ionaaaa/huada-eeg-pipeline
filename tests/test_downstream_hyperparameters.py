from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn


PROJECT = Path(__file__).resolve().parents[1]
MODULE_DIR = PROJECT / "scripts" / "downstream_50m"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import largeScale_fineTune as training
from configs import Config


def _args(*extra):
    return training.parse_args([
        "--input-lance", "fixture.lance",
        "--task", "sex_classification",
        "--output-dir", "output",
        "--smoke",
        *extra,
    ])


@pytest.mark.parametrize("pooling", ["mean", "flatten"])
def test_cli_pooling_choices_and_overrides(pooling):
    cfg = Config()
    args = _args(
        "--pooling", pooling,
        "--head-lr", "0.0025",
        "--weight-decay", "0.007",
    )
    training.apply_cli_overrides(cfg, args)
    assert cfg.finetune.pooling == pooling
    assert cfg.finetune.head_lr == pytest.approx(0.0025)
    assert cfg.finetune.weight_decay == pytest.approx(0.007)


def test_cli_omission_preserves_config_defaults():
    cfg = Config()
    cfg.finetune.pooling = "flatten"
    cfg.finetune.head_lr = 0.004
    cfg.finetune.weight_decay = 0.006
    training.apply_cli_overrides(cfg, _args())
    assert cfg.finetune.pooling == "flatten"
    assert cfg.finetune.head_lr == pytest.approx(0.004)
    assert cfg.finetune.weight_decay == pytest.approx(0.006)


def test_invalid_cli_pooling_and_numeric_values_fail_before_training():
    with pytest.raises(SystemExit):
        _args("--pooling", "attention")
    for option, value, message in (
        ("--head-lr", "0", "head_lr"),
        ("--head-lr", "-0.1", "head_lr"),
        ("--weight-decay", "-0.1", "weight_decay"),
    ):
        with pytest.raises(ValueError, match=message):
            training.apply_cli_overrides(Config(), _args(option, value))


class IdentityBackbone(nn.Module):
    def __init__(self, d_model=512):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d_model))

    def extract_token_embeddings(self, token_inputs, **kwargs):
        return token_inputs * self.scale


def _model(pooling, num_tokens=640, freeze_backbone=False):
    return training.EEGDownstreamClassifier(
        IdentityBackbone(), 512, 2,
        pooling=pooling,
        num_tokens=num_tokens,
        freeze_backbone=freeze_backbone,
    )


def _batch(batch_size=2, num_tokens=640):
    mask = torch.ones(batch_size, num_tokens, dtype=torch.bool)
    mask[:, -10:] = False
    return {
        "token_inputs": torch.randn(batch_size, num_tokens, 512),
        "token_channel_indices": torch.zeros(batch_size, num_tokens, dtype=torch.long),
        "token_time_indices": torch.zeros(batch_size, num_tokens, dtype=torch.long),
        "token_valid_mask": mask,
    }


def test_mean_and_flatten_shapes_masking_head_sizes_and_backwards():
    mean = _model("mean")
    flatten = _model("flatten")
    assert set(mean.backbone.state_dict()) == set(flatten.backbone.state_dict())
    assert mean.head_input_dim == 512
    assert flatten.head_input_dim == 640 * 512
    assert training.head_parameter_count(mean) == 1026
    assert training.head_parameter_count(flatten) == 655362

    embeddings = torch.ones(1, 640, 512)
    embeddings[:, -10:] = 1e20
    mask = torch.ones(1, 640, dtype=torch.bool)
    mask[:, -10:] = False
    mean_features = mean.extract_features(embeddings, mask)
    flat_features = flatten.extract_features(embeddings, mask)
    assert mean_features.shape == (1, 512)
    assert flat_features.shape == (1, 640 * 512)
    assert torch.allclose(mean_features, torch.ones_like(mean_features))
    assert torch.equal(flat_features.reshape(1, 640, 512)[:, -10:], torch.zeros(1, 10, 512))

    batch = _batch()
    for model in (mean, flatten):
        output = model(**batch)
        assert output.shape == (2, 2)
        nn.CrossEntropyLoss()(output, torch.tensor([0, 1])).backward()
        assert model.head.weight.grad is not None


def test_optimizer_uses_resolved_lr_weight_decay_and_keeps_types():
    cfg = Config()
    cfg.finetune.head_lr = 0.002
    cfg.finetune.weight_decay = 0.003
    cfg.finetune.mode = "linear_probe"
    linear_model = _model("mean", freeze_backbone=True)
    linear_optimizer = training.make_optimizer(cfg, linear_model)
    linear_report = training.optimizer_configuration(linear_optimizer)
    assert isinstance(linear_optimizer, torch.optim.SGD)
    assert linear_report["param_groups"] == [{
        "param_group_name": "head",
        "learning_rate": pytest.approx(0.002),
        "weight_decay": pytest.approx(0.003),
        "parameter_count": 1026,
    }]

    cfg.finetune.mode = "finetune"
    cfg.finetune.backbone_lr = 0.0004
    finetune_model = _model("mean")
    finetune_optimizer = training.make_optimizer(cfg, finetune_model)
    report = training.optimizer_configuration(finetune_optimizer)
    assert isinstance(finetune_optimizer, torch.optim.AdamW)
    groups = {group["param_group_name"]: group for group in report["param_groups"]}
    assert groups["backbone"]["learning_rate"] == pytest.approx(0.0004)
    assert groups["head"]["learning_rate"] == pytest.approx(0.002)
    assert groups["backbone"]["weight_decay"] == pytest.approx(0.003)
    assert groups["head"]["weight_decay"] == pytest.approx(0.003)


def _save_checkpoint(path, model, cfg, include_pooling=True):
    optimizer = training.make_optimizer(cfg, model)
    report = training.optimizer_configuration(optimizer)
    training.save_downstream_checkpoint(
        path, model, cfg,
        {"num_tokens": 640, "patch_len": 100},
        8, 1, {"validation": {}}, {}, report,
    )
    if not include_pooling:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload.pop("pooling")
        payload.pop("head_input_dim")
        torch.save(payload, path)


@pytest.mark.parametrize("pooling,expected_dim", [("mean", 512), ("flatten", 640 * 512)])
def test_checkpoint_saves_hyperparameters_and_legacy_shape_inference(tmp_path, pooling, expected_dim):
    cfg = Config()
    cfg.finetune.pooling = pooling
    cfg.finetune.head_lr = 0.002
    cfg.finetune.weight_decay = 0.003
    model = _model(pooling)
    path = tmp_path / f"{pooling}.pt"
    _save_checkpoint(path, model, cfg)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["pooling"] == pooling
    assert payload["head_input_dim"] == expected_dim
    assert payload["head_lr"] == pytest.approx(0.002)
    assert payload["weight_decay"] == pytest.approx(0.003)

    _save_checkpoint(path, model, cfg, include_pooling=False)
    loaded = training.load_downstream_checkpoint_strict(_model(pooling), path, torch.device("cpu"))
    assert loaded["_load_report"] == {
        "pooling": pooling, "pooling_inferred": True, "head_input_dim": expected_dim,
    }


def test_checkpoint_pooling_mismatch_and_ambiguous_legacy_fail(tmp_path):
    cfg = Config()
    cfg.finetune.pooling = "mean"
    mean_path = tmp_path / "mean.pt"
    _save_checkpoint(mean_path, _model("mean"), cfg)
    with pytest.raises(RuntimeError, match="does not match requested pooling"):
        training.load_downstream_checkpoint_strict(_model("flatten"), mean_path, torch.device("cpu"))

    cfg.finetune.pooling = "flatten"
    flatten_path = tmp_path / "flatten.pt"
    _save_checkpoint(flatten_path, _model("flatten"), cfg)
    with pytest.raises(RuntimeError, match="does not match requested pooling"):
        training.load_downstream_checkpoint_strict(_model("mean"), flatten_path, torch.device("cpu"))

    ambiguous_model = _model("mean", num_tokens=1)
    ambiguous_path = tmp_path / "ambiguous.pt"
    torch.save({"model_state_dict": ambiguous_model.state_dict()}, ambiguous_path)
    with pytest.raises(RuntimeError, match="Cannot uniquely infer"):
        training.load_downstream_checkpoint_strict(ambiguous_model, ambiguous_path, torch.device("cpu"))


def test_cloud_script_is_single_cv4_entry_with_explicit_hyperparameters():
    script = (PROJECT / "commands" / "run_50m_sex_cv4_full.sh").read_text(encoding="utf-8")
    assert "sex_stratified_subject_cv4" in script
    assert "legacy_single_split" not in script
    assert "create_downstream_manifest.py" not in script
    assert script.count("scripts/downstream_50m/largeScale_fineTune.py") == 1
    for option in ("--pooling", "--head-lr", "--weight-decay"):
        assert option in script
    assert "50m_sex_cv4_${POOLING_TAG}_lr${LR_TAG}_wd${WD_TAG}_${STAMP}" in script
    assert "for " not in script
    for credential_marker in ("SECRET_KEY", "ACCESS_KEY", "github_pat", "ghp_"):
        assert credential_marker not in script
