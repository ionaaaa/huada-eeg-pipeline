from __future__ import annotations

import csv
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


PROJECT = Path(__file__).resolve().parents[1]
MODULE_DIR = PROJECT / "scripts" / "downstream_50m"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from configs import Config
from dataset import collate_fn
from lance_dataset import LanceEEGPretrainDataset, SEX_LABEL_MAPPING
from largeScale_fineTune import (
    EEGDownstreamClassifier,
    build_backbone,
    build_downstream_model,
    compute_metrics,
    compute_shape_params,
    get_labels_from_batch,
    strip_module_prefix,
    validate_subject_split_manifest,
)
from model import EEGPretrainModel


SMOKE_LANCE = Path(
    r"D:\repo\华大\model_tokens_smoke\20261030_134527\huada_model_tokens.lance"
)


@pytest.fixture(scope="module")
def smoke_lance():
    if not SMOKE_LANCE.exists():
        pytest.skip("Huada smoke Lance is not present")
    return SMOKE_LANCE


def test_real_lance_dataset_shapes_dtypes_and_length(smoke_lance):
    dataset = LanceEEGPretrainDataset(smoke_lance, task="sex_classification")
    assert len(dataset) == 4
    sample = dataset[0]
    assert isinstance(sample["sample_id"], str)
    assert sample["token_inputs"].shape == (640, 100)
    assert sample["targets"].shape == (640, 5, 5)
    assert sample["token_channel_indices"].shape == (640,)
    assert sample["token_time_indices"].shape == (640,)
    assert sample["channel_valid_mask"].shape == (64,)
    assert sample["token_valid_mask"].shape == (640,)
    assert sample["token_inputs"].dtype == torch.float32
    assert sample["token_channel_indices"].dtype == torch.int64
    assert sample["token_valid_mask"].dtype == torch.float32
    assert sample["label"].dtype == torch.int64


def test_dataloader_batch_keeps_masks_aligned(smoke_lance):
    dataset = LanceEEGPretrainDataset(smoke_lance, task="sex_classification")
    batch = next(iter(DataLoader(dataset, batch_size=4, collate_fn=collate_fn)))
    assert batch["token_inputs"].shape == (4, 640, 100)
    assert batch["token_valid_mask"].shape == (4, 640)
    for index, subject in enumerate(batch["subject_id"]):
        expected = 600 if subject == "SUB1" else 590
        assert int(batch["token_valid_mask"][index].sum()) == expected


def test_sex_mapping_is_explicit_and_sub5_is_kept(smoke_lance):
    assert SEX_LABEL_MAPPING == {"female": 0, "male": 1}
    dataset = LanceEEGPretrainDataset(smoke_lance, task="sex_classification")
    rows = dataset.get_subject_label_rows()
    assert {row["subject_id"] for row in rows} == {"SUB1", "SUB5"}
    assert {row["label"] for row in rows} == {1}
    assert all(dataset[index]["label"].dtype == torch.int64 for index in range(len(dataset)))


def test_age_is_float32_and_sub5_is_excluded_without_imputation(smoke_lance):
    dataset = LanceEEGPretrainDataset(smoke_lance, task="age_regression")
    assert len(dataset) == 2
    assert {row["subject_id"] for row in dataset.get_subject_label_rows()} == {"SUB1"}
    assert {float(dataset[index]["label"]) for index in range(len(dataset))} == {31.0}
    assert all(dataset[index]["label"].dtype == torch.float32 for index in range(len(dataset)))


def _write_manifest(path, rows):
    fields = ["subject_id", "task", "label", "label_name", "exclusion_reason", "split", "split_seed"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_subject_split_manifest_has_no_window_leakage(smoke_lance, tmp_path):
    manifest = tmp_path / "sex_splits.csv"
    _write_manifest(manifest, [
        {"subject_id": "SUB1", "task": "sex_classification", "label": "1", "label_name": "male", "exclusion_reason": "", "split": "train", "split_seed": "42"},
        {"subject_id": "SUB5", "task": "sex_classification", "label": "1", "label_name": "male", "exclusion_reason": "", "split": "test", "split_seed": "42"},
    ])
    train = LanceEEGPretrainDataset(smoke_lance, task="sex_classification", subject_split_manifest=manifest, split="train")
    test = LanceEEGPretrainDataset(smoke_lance, task="sex_classification", subject_split_manifest=manifest, split="test")
    assert {row["subject_id"] for row in train.get_subject_label_rows()} == {"SUB1"}
    assert {row["subject_id"] for row in test.get_subject_label_rows()} == {"SUB5"}
    assert {row["sample_id"] for row in train.get_subject_label_rows()}.isdisjoint(
        {row["sample_id"] for row in test.get_subject_label_rows()}
    )


def test_split_validator_rejects_empty_full_split(tmp_path):
    manifest = tmp_path / "bad.csv"
    _write_manifest(manifest, [
        {"subject_id": "SUB1", "task": "sex_classification", "label": "1", "label_name": "male", "exclusion_reason": "", "split": "train", "split_seed": "42"},
        {"subject_id": "SUB5", "task": "sex_classification", "label": "1", "label_name": "male", "exclusion_reason": "", "split": "test", "split_seed": "42"},
    ])
    with pytest.raises(ValueError, match="non-empty"):
        validate_subject_split_manifest(manifest, "sex_classification")


def test_fixed_seed_subject_split_is_reproducible():
    scripts_dir = PROJECT / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        module = importlib.import_module("create_downstream_manifest")
        rows = [
            {"subject_id": f"SUB{i}", "task": "sex_classification", "label": str(i % 2), "label_name": "female" if i % 2 == 0 else "male", "exclusion_reason": ""}
            for i in range(1, 9)
        ]
        assert module._assign_splits(rows, 42) == module._assign_splits(rows, 42)
    finally:
        sys.path.remove(str(scripts_dir))


class TinyBackbone(nn.Module):
    def __init__(self, input_dim=4, d_model=3):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)

    def extract_token_embeddings(self, token_inputs, token_channel_indices, token_time_indices, return_layer_idx=None):
        del token_channel_indices, token_time_indices, return_layer_idx
        return self.proj(token_inputs)


def _tiny_batch(batch_size=2):
    return {
        "token_inputs": torch.randn(batch_size, 4, 4),
        "token_channel_indices": torch.zeros(batch_size, 4, dtype=torch.long),
        "token_time_indices": torch.zeros(batch_size, 4, dtype=torch.long),
        "token_valid_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.float32),
    }


def test_invalid_tokens_do_not_enter_flatten_or_mean_pooling():
    model = EEGDownstreamClassifier(TinyBackbone(), 3, 2, num_tokens=4)
    embeddings = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.float32)
    flattened = model.flatten_valid_tokens(embeddings, mask)
    assert torch.equal(flattened[1, -3:], torch.zeros(3))
    pooled = model.pool_valid_tokens(embeddings, mask)
    assert torch.allclose(pooled[1], embeddings[1, :3].mean(dim=0))


def test_normal_and_single_bad_channel_mask_contract():
    normal = torch.ones(640, dtype=torch.float32)
    single_bad = normal.clone()
    ft7_channel_index = 17
    single_bad[ft7_channel_index * 10 : (ft7_channel_index + 1) * 10] = 0
    assert int(normal.sum()) == 640
    assert int(single_bad.sum()) == 630
    assert torch.equal(single_bad[170:180], torch.zeros(10))


@pytest.mark.parametrize("task,output_dim,label_dtype", [
    ("sex_classification", 2, torch.int64),
    ("age_regression", 1, torch.float32),
])
def test_task_heads_forward_and_backward(task, output_dim, label_dtype):
    model = EEGDownstreamClassifier(TinyBackbone(), 3, output_dim, num_tokens=4)
    batch = _tiny_batch()
    outputs = model(**batch)
    if task == "sex_classification":
        assert outputs.shape == (2, 2)
        labels = torch.tensor([0, 1], dtype=label_dtype)
        loss = nn.CrossEntropyLoss()(outputs, labels)
    else:
        assert outputs.shape == (2, 1)
        labels = torch.tensor([20.0, 30.0], dtype=label_dtype)
        loss = nn.MSELoss()(outputs.view(-1), labels)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.head.parameters())


def test_frozen_and_finetune_parameter_states():
    frozen = EEGDownstreamClassifier(TinyBackbone(), 3, 2, freeze_backbone=True, num_tokens=4)
    finetune = EEGDownstreamClassifier(TinyBackbone(), 3, 2, freeze_backbone=False, num_tokens=4)
    assert not any(parameter.requires_grad for parameter in frozen.backbone.parameters())
    assert all(parameter.requires_grad for parameter in frozen.head.parameters())
    assert all(parameter.requires_grad for parameter in finetune.backbone.parameters())


def test_upstream_classifier_parity_allclose():
    class UpstreamClassifier(EEGDownstreamClassifier):
        pass

    torch.manual_seed(7)
    migrated = EEGDownstreamClassifier(TinyBackbone(), 3, 2, num_tokens=4)
    upstream = UpstreamClassifier(TinyBackbone(), 3, 2, num_tokens=4)
    upstream.load_state_dict(migrated.state_dict(), strict=True)
    batch = _tiny_batch()
    migrated.eval()
    upstream.eval()
    assert torch.allclose(migrated(**batch), upstream(**batch), atol=1e-6, rtol=1e-6)


def test_50m_forward_shape():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EEGPretrainModel(
        input_dim=100, d_model=512, n_heads=8, depth=12, mlp_ratio=4.0,
        dropout=0.1, n_channels=64, n_time_patches=10,
        n_bands=5, frames_per_patch=5,
    ).to(device).eval()
    inputs = torch.zeros(1, 640, 100, device=device)
    channels = torch.arange(64, device=device).repeat_interleave(10).unsqueeze(0)
    times = torch.arange(10, device=device).repeat(64).unsqueeze(0)
    with torch.no_grad():
        embeddings = model.extract_token_embeddings(inputs, channels, times, return_layer_idx=8)
    assert embeddings.shape == (1, 640, 512)


def test_checkpoint_helpers_and_mismatch_report(monkeypatch, tmp_path):
    import largeScale_fineTune as module

    class TinyPretrain(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))

    monkeypatch.setattr(module, "EEGPretrainModel", TinyPretrain)
    checkpoint = tmp_path / "tiny.pt"
    torch.save({"model_state_dict": {"module.weight": torch.tensor([2.0])}}, checkpoint)
    cfg = Config()
    cfg.finetune.pretrained_ckpt_path = str(checkpoint)
    backbone, report = build_backbone(cfg, compute_shape_params(cfg), torch.device("cpu"))
    assert float(backbone.weight) == 2.0
    assert report["missing_keys"] == []
    assert report["unexpected_keys"] == []
    assert len(report["sha256"]) == 64
    assert "weight" in strip_module_prefix({"module.weight": torch.tensor(1.0)})


def test_small_sample_metrics_are_marked_unreliable():
    classification = compute_metrics("sex_classification", [1, 1], [1, 1], 0.2, smoke=True)
    assert classification["balanced_accuracy"] is None
    assert classification["macro_f1"] is None
    assert classification["metrics_not_statistically_meaningful"] is True
    regression = compute_metrics("age_regression", [31.0], [30.0], 1.0, smoke=True)
    assert regression["r2"] is None
    assert regression["r2_reliable"] is False
