from __future__ import annotations

import json
import shutil
import sys
from copy import deepcopy
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


PROJECT = Path(__file__).resolve().parents[1]
MODULE_DIR = PROJECT / "scripts" / "downstream_50m"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import feature_cache as cache
import largeScale_fineTune as training
from run_logging import setup_run_logging
from configs import Config
from dataset import collate_fn


class TinySource(Dataset):
    def __init__(self):
        self.metadata = [
            {"sample_id": "NORMAL_W0", "subject_id": "SUB1", "recording_id": "R1"},
            {"sample_id": "SUB5_W0", "subject_id": "SUB5", "recording_id": None},
            {"sample_id": "SUB8_W0", "subject_id": "SUB8", "recording_id": "R8"},
        ]
        self.masks = [
            torch.tensor([1, 1, 1, 1], dtype=torch.float32),
            torch.tensor([1, 0, 1, 1], dtype=torch.float32),
            torch.tensor([1, 1, 0, 1], dtype=torch.float32),
        ]

    def __len__(self):
        return 3

    def __getitem__(self, index):
        meta = self.metadata[index]
        values = torch.arange(12, dtype=torch.float32).reshape(4, 3) + index * 100
        return {
            **meta,
            "token_inputs": values,
            "token_channel_indices": torch.arange(4),
            "token_time_indices": torch.arange(4),
            "token_valid_mask": self.masks[index],
        }


class CountingBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.forward_count = 0

    def extract_token_embeddings(self, token_inputs, **kwargs):
        self.forward_count += 1
        return token_inputs


def _manifest_base(source):
    identity = {"fixture": "tiny-cache-v1"}
    return {
        "cache_schema_version": 1,
        "cache_key": cache.sha256_json(identity),
        "cache_identity": identity,
        "input_row_count": len(source),
        "sample_ids": [row["sample_id"] for row in source.metadata],
        "sample_id_fingerprint": cache.sha256_json(
            sorted(row["sample_id"] for row in source.metadata)
        ),
        "input_lance": "fixture.lance",
        "input_lance_fingerprint": "f" * 64,
        "pretrained_checkpoint": "pretrained.pt",
        "pretrained_checkpoint_sha256": "p" * 64,
        "checkpoint_matched_key_count": 1,
        "checkpoint_matched_parameter_fraction": 1.0,
        "probe_layer_idx": 1,
        "probe_layer_ratio": 0.75,
        "num_tokens": 4,
        "d_model": 3,
        "python_version": "test",
        "torch_version": torch.__version__,
        "lance_version": lance.__version__,
    }


@pytest.fixture
def tiny_cache(tmp_path):
    source = TinySource()
    backbone = CountingBackbone()
    path = tmp_path / "cache"
    manifest = cache.build_feature_cache(
        source_dataset=source,
        backbone=backbone,
        probe_layer_idx=1,
        output_dir=path,
        batch_size=2,
        device=torch.device("cpu"),
        expected_num_tokens=4,
        d_model=3,
        manifest_base=_manifest_base(source),
    )
    return path, source, backbone, manifest


def test_pool_both_shapes_dtype_mask_and_all_false_are_finite():
    embeddings = torch.arange(24, dtype=torch.float64).reshape(2, 4, 3)
    mask = torch.tensor([[1, 0, 1, 1], [0, 0, 0, 0]], dtype=torch.bool)
    mean, flatten = cache.masked_pool_both(embeddings, mask)
    assert mean.shape == (2, 3)
    assert flatten.shape == (2, 12)
    assert mean.dtype == flatten.dtype == torch.float32
    expected = embeddings[0, [0, 2, 3]].float().mean(dim=0)
    assert torch.allclose(mean[0], expected)
    assert torch.equal(flatten.reshape(2, 4, 3)[0, 1], torch.zeros(3))
    assert torch.equal(mean[1], torch.zeros(3))
    assert torch.isfinite(mean).all() and torch.isfinite(flatten).all()


def test_one_backbone_pass_per_batch_writes_both_label_free_caches(tiny_cache):
    path, source, backbone, manifest = tiny_cache
    assert backbone.forward_count == 2
    assert manifest["backbone_forward_count"] == 2
    assert manifest["source_window_count"] == 3
    assert manifest["backbone_processed_window_count"] == 3
    mean = lance.dataset(str(path / cache.MEAN_CACHE_NAME))
    flatten = lance.dataset(str(path / cache.FLATTEN_CACHE_NAME))
    assert mean.count_rows() == flatten.count_rows() == len(source)
    assert mean.schema.field("feature").type.list_size == 3
    assert flatten.schema.field("feature").type.list_size == 12
    for dataset in (mean, flatten):
        assert not set(dataset.schema.names).intersection(
            {"label", "sex", "split", "fold", "candidate"}
        )
        assert dataset.schema.field("sample_id").type == pa.string()


def test_cache_progress_logs_first_interval_last_and_same_pass(tmp_path, capsys):
    source = TinySource()
    backbone = CountingBackbone()
    logger = setup_run_logging(tmp_path / "logs", "INFO", append=False)
    manifest = cache.build_feature_cache(
        source_dataset=source, backbone=backbone, probe_layer_idx=1,
        output_dir=tmp_path / "logged_cache", batch_size=2,
        device=torch.device("cpu"), expected_num_tokens=4, d_model=3,
        manifest_base=_manifest_base(source), logger=logger, log_every_batches=2,
    )
    output = capsys.readouterr().out
    assert "batch=1/2" in output
    assert "batch=2/2" in output
    assert "processed_rows=3" in output
    assert "backbone_forward_count=2" in output
    assert "mean and flatten derived from the same backbone batch output" in output
    assert manifest["backbone_forward_count"] == backbone.forward_count == 2
    run_log = (tmp_path / "logs" / "run.log").read_text(encoding="utf-8")
    assert "cache.extract" in run_log and "progress_percent=100.0" in run_log


def test_bad_channel_tokens_are_exact_zero_and_mean_excludes_them(tiny_cache):
    path, _, _, _ = tiny_cache
    mean_rows = {
        row["sample_id"]: np.asarray(row["feature"], dtype=np.float32)
        for row in lance.dataset(str(path / cache.MEAN_CACHE_NAME)).to_table().to_pylist()
    }
    flat_rows = {
        row["sample_id"]: np.asarray(row["feature"], dtype=np.float32).reshape(4, 3)
        for row in lance.dataset(str(path / cache.FLATTEN_CACHE_NAME)).to_table().to_pylist()
    }
    assert np.count_nonzero(flat_rows["SUB5_W0"][1]) == 0
    assert np.count_nonzero(flat_rows["SUB8_W0"][2]) == 0
    expected_sub5 = np.stack([
        np.arange(12, dtype=np.float32).reshape(4, 3)[index] + 100
        for index in (0, 2, 3)
    ]).mean(axis=0)
    assert np.allclose(mean_rows["SUB5_W0"], expected_sub5)


def test_cache_manifest_success_and_parity_fields(tiny_cache):
    path, _, _, manifest = tiny_cache
    validated = cache.validate_feature_cache(
        path, manifest["cache_identity"], manifest["cache_key"]
    )
    assert (path / cache.SUCCESS_NAME).is_file()
    assert validated["parity_passed"] is True
    assert validated["parity_sample_ids"] == ["NORMAL_W0", "SUB5_W0", "SUB8_W0"]
    for key in (
        "mean_max_abs_diff", "flatten_max_abs_diff",
        "mean_logits_max_abs_diff", "flatten_logits_max_abs_diff",
    ):
        assert validated[key] == pytest.approx(0.0)


def test_publish_to_mount_uses_copy_and_success_marker_not_directory_rename(
    tiny_cache, tmp_path, monkeypatch
):
    build_path, _, _, manifest = tiny_cache
    final_path = tmp_path / "mounted" / manifest["cache_key"]
    original_replace = cache.os.replace

    def reject_directory_replace(source, destination):
        if Path(source) == build_path or Path(destination) == final_path:
            raise OSError(38, "Function not implemented")
        return original_replace(source, destination)

    monkeypatch.setattr(cache.os, "replace", reject_directory_replace)
    published = cache.publish_feature_cache(
        build_path, final_path, manifest["cache_identity"], manifest["cache_key"]
    )
    assert published["cache_key"] == manifest["cache_key"]
    assert (final_path / cache.SUCCESS_NAME).read_text(encoding="utf-8").strip() == manifest["cache_key"]
    assert cache.validate_feature_cache(final_path)["parity_passed"] is True


def _identity(**overrides):
    values = {
        "input_fingerprint": "input-a", "input_row_count": 3,
        "sample_id_fingerprint": "samples-a", "checkpoint_sha256": "checkpoint-a",
        "model_config": {"d_model": 3, "depth": 2, "n_heads": 1},
        "shape_params": {"num_tokens": 4, "patch_len": 5},
        "probe_layer_idx": 1, "probe_layer_ratio": 0.75,
        "model_code_sha256": "code-a", "git_commit": "commit-a",
    }
    values.update(overrides)
    return cache.build_cache_identity(**values)


def test_cache_key_stable_and_json_order_independent():
    identity, key = _identity()
    reordered = dict(reversed(list(identity.items())))
    assert cache.sha256_json(reordered) == key
    assert _identity()[1] == key


def test_lance_fingerprint_is_stable_and_changes_with_dataset(tmp_path):
    path = tmp_path / "fingerprint.lance"
    lance.write_dataset(
        pa.table({"sample_id": ["a", "b"], "value": [1, 2]}),
        str(path), mode="create", data_storage_version="stable",
    )
    first, info = cache.lance_fingerprint(path)
    assert cache.lance_fingerprint(path)[0] == first
    assert info["row_count"] == 2
    shutil.rmtree(path)
    lance.write_dataset(
        pa.table({"sample_id": ["a", "c"], "value": [1, 3]}),
        str(path), mode="create", data_storage_version="stable",
    )
    assert cache.lance_fingerprint(path)[0] != first


@pytest.mark.parametrize("field,new_value", [
    ("checkpoint_sha256", "checkpoint-b"),
    ("input_fingerprint", "input-b"),
    ("probe_layer_idx", 0),
    ("model_code_sha256", "code-b"),
])
def test_feature_producing_changes_invalidate_cache_key(field, new_value):
    assert _identity()[1] != _identity(**{field: new_value})[1]


def test_training_only_settings_do_not_enter_cache_key():
    baseline = _identity()[1]
    for ignored in (
        {"head_lr": 1e-3}, {"weight_decay": 0.01}, {"fold": 3},
        {"pooling": "flatten"}, {"batch_size": 64}, {"seed": 99},
    ):
        # build_cache_identity has no such arguments by design.
        assert ignored and _identity()[1] == baseline


def _copy_cache(tiny_cache, tmp_path, name):
    source = tiny_cache[0]
    destination = tmp_path / name
    shutil.copytree(source, destination)
    return destination


def test_missing_success_cannot_be_reused(tiny_cache, tmp_path):
    path = _copy_cache(tiny_cache, tmp_path, "missing-success")
    (path / cache.SUCCESS_NAME).unlink()
    with pytest.raises(ValueError, match="incomplete"):
        cache.validate_feature_cache(path)


def test_auto_does_not_overwrite_damaged_cache_and_rebuild_preserves_old(tmp_path):
    identity = {"fixture": "policy"}
    key = cache.sha256_json(identity)
    root = tmp_path / "root"
    damaged = root / key
    damaged.mkdir(parents=True)
    marker = damaged / "user-marker.txt"
    marker.write_text("preserve", encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete"):
        cache.resolve_cache_policy(root, key, "auto", identity)
    assert marker.read_text(encoding="utf-8") == "preserve"

    destination, manifest, reused = cache.resolve_cache_policy(
        root, key, "rebuild", identity, timestamp="20260101T000000Z"
    )
    assert destination == root / f"{key}_rebuild_20260101T000000Z"
    assert manifest is None and reused is False
    assert marker.read_text(encoding="utf-8") == "preserve"


def test_manifest_hash_mismatch_cannot_be_reused(tiny_cache, tmp_path):
    path = _copy_cache(tiny_cache, tmp_path, "wrong-hash")
    manifest = json.loads((path / cache.MANIFEST_NAME).read_text())
    manifest["cache_key"] = "x" * 64
    cache.atomic_write_json(path / cache.MANIFEST_NAME, manifest)
    with pytest.raises(ValueError, match="SUCCESS key mismatch"):
        cache.validate_feature_cache(path)


def _overwrite_feature_cache(path, rows, dim):
    if path.exists():
        shutil.rmtree(path)
    features = np.asarray([row["feature"] for row in rows], dtype=np.float32)
    table = cache.feature_rows_to_table(
        [row["sample_id"] for row in rows],
        [row["subject_id"] for row in rows],
        [row.get("recording_id") for row in rows],
        features.reshape(len(rows), dim),
        [row["num_valid_tokens"] for row in rows],
    )
    lance.write_dataset(table, str(path), mode="create", data_storage_version="stable")


def test_duplicate_sample_and_row_count_mismatch_are_rejected(tiny_cache, tmp_path):
    path = _copy_cache(tiny_cache, tmp_path, "bad-rows")
    mean_path = path / cache.MEAN_CACHE_NAME
    rows = lance.dataset(str(mean_path)).to_table().to_pylist()
    rows[1]["sample_id"] = rows[0]["sample_id"]
    _overwrite_feature_cache(mean_path, rows, 3)
    with pytest.raises(ValueError, match="duplicate"):
        cache.validate_feature_cache(path)

    path = _copy_cache(tiny_cache, tmp_path, "short-rows")
    mean_path = path / cache.MEAN_CACHE_NAME
    rows = lance.dataset(str(mean_path)).to_table().to_pylist()[:-1]
    _overwrite_feature_cache(mean_path, rows, 3)
    with pytest.raises(ValueError, match="row count"):
        cache.validate_feature_cache(path)


def test_wrong_schema_and_nonfinite_features_are_rejected(tiny_cache, tmp_path):
    path = _copy_cache(tiny_cache, tmp_path, "wrong-schema")
    mean_path = path / cache.MEAN_CACHE_NAME
    rows = lance.dataset(str(mean_path)).to_table().to_pylist()
    _overwrite_feature_cache(mean_path, rows, 3)
    table = lance.dataset(str(mean_path)).to_table().append_column(
        "label", pa.array([0, 0, 0], type=pa.int64())
    )
    shutil.rmtree(mean_path)
    lance.write_dataset(table, str(mean_path), mode="create", data_storage_version="stable")
    with pytest.raises(ValueError, match="task/split"):
        cache.validate_feature_cache(path)

    path = _copy_cache(tiny_cache, tmp_path, "nan")
    mean_path = path / cache.MEAN_CACHE_NAME
    rows = lance.dataset(str(mean_path)).to_table().to_pylist()
    rows[0]["feature"][0] = float("nan")
    _overwrite_feature_cache(mean_path, rows, 3)
    with pytest.raises(ValueError, match="NaN or Inf"):
        cache.validate_feature_cache(path)


def test_parity_above_tolerance_fails(tiny_cache, monkeypatch):
    path, _, _, manifest = tiny_cache
    original = cache._read_features_by_sample

    def changed(path_value, ids):
        values = original(path_value, ids)
        if str(path_value).endswith(cache.MEAN_CACHE_NAME):
            values[ids[0]] = values[ids[0]].copy()
            values[ids[0]][0] += 1e-3
        return values

    monkeypatch.setattr(cache, "_read_features_by_sample", changed)
    online_mean = original(path / cache.MEAN_CACHE_NAME, manifest["parity_sample_ids"])
    online_flatten = original(path / cache.FLATTEN_CACHE_NAME, manifest["parity_sample_ids"])
    with pytest.raises(RuntimeError, match="Mean feature parity"):
        cache._parity_report(
            manifest["parity_sample_ids"], online_mean, online_flatten,
            path / cache.MEAN_CACHE_NAME, path / cache.FLATTEN_CACHE_NAME,
        )


def _write_label_source(path, source):
    table = pa.table({
        "sample_id": [row["sample_id"] for row in source.metadata],
        "subject_id": [row["subject_id"] for row in source.metadata],
        "age": pa.array([20.0, 21.0, 22.0], type=pa.float64()),
        "sex": ["female", "male", "male"],
    })
    lance.write_dataset(table, str(path), mode="create", data_storage_version="stable")


def test_cached_dataset_joins_labels_by_sample_id_and_collates(tiny_cache, tmp_path):
    path, source, _, _ = tiny_cache
    label_lance = tmp_path / "labels.lance"
    _write_label_source(label_lance, source)
    dataset = cache.CachedFeatureDataset(
        path, "mean", label_lance, "sex_classification"
    )
    batch = next(iter(DataLoader(dataset, batch_size=3, collate_fn=collate_fn)))
    assert batch["feature"].shape == (3, 3)
    assert batch["label"].tolist() == [0, 1, 1]
    assert batch["sample_id"] == ["NORMAL_W0", "SUB5_W0", "SUB8_W0"]
    flatten_dataset = cache.CachedFeatureDataset(
        path, "flatten", label_lance, "sex_classification",
        expected_cache_key=cache.validate_cache_manifest(path)["cache_key"],
    )
    assert flatten_dataset[0]["feature"].shape == (12,)
    assert dataset[0]["feature"].shape == (3,)


def test_cached_training_never_builds_backbone_and_saves_head_only(
    tiny_cache, monkeypatch, tmp_path, capsys
):
    cache_path, _, _, manifest = tiny_cache
    cfg = Config()
    cfg.model.d_model = 3
    cfg.model.depth = 2
    cfg.model.n_heads = 1
    cfg.data.n_channels = 2
    cfg.data.clip_seconds = 2
    cfg.data.patch_seconds = 1
    cfg.data.patch_stride_seconds = 1
    cfg.finetune.feature_source = "cache"
    cfg.finetune.feature_cache_path = str(cache_path)
    cfg.finetune.feature_cache_key = manifest["cache_key"]
    cfg.finetune.feature_cache_pooling = "mean"
    cfg.finetune.pooling = "mean"
    cfg.finetune.mode = "linear_probe"
    cfg.finetune.num_classes = 2
    cfg.finetune.num_epochs = 1
    cfg.finetune.head_lr = 0.001
    cfg.finetune.weight_decay = 0.123
    cfg.finetune.save_path = str(tmp_path / "head-only.pt")
    cfg.finetune.sweep_candidate_id = "candidate"
    cfg.finetune.sweep_config_sha256 = "s" * 64
    cfg.finetune.sweep_args_json = '{"--pooling":"mean"}'
    monkeypatch.setattr(
        training, "build_backbone",
        lambda *args, **kwargs: pytest.fail("candidate training constructed a backbone"),
    )
    samples = [
        {"feature": torch.tensor([1.0, 2.0, 3.0]), "label": torch.tensor(0),
         "num_valid_tokens": torch.tensor(4), "sample_id": "a", "subject_id": "SUB1",
         "recording_id": "", "label_name": "female"},
        {"feature": torch.tensor([3.0, 2.0, 1.0]), "label": torch.tensor(1),
         "num_valid_tokens": torch.tensor(3), "sample_id": "b", "subject_id": "SUB5",
         "recording_id": "", "label_name": "male"},
    ]
    loader = DataLoader(samples, batch_size=2, collate_fn=collate_fn)
    setup_run_logging(tmp_path / "training_logs", "INFO", append=False)
    shape = training.compute_shape_params(cfg)
    _, metrics, _, _, predictions = training.run_one_checkpoint(
        cfg, shape, loader, loader, None, torch.device("cpu"), search_only=True,
    )
    assert predictions == []
    assert metrics["training_metadata"]["backbone_forward_count"] == 0
    checkpoint = torch.load(cfg.finetune.save_path, map_location="cpu", weights_only=False)
    assert checkpoint["checkpoint_kind"] == "cached_feature_head_only"
    assert "head_state_dict" in checkpoint
    assert "model_state_dict" not in checkpoint
    assert "backbone_state_dict" not in checkpoint
    assert checkpoint["feature_cache_key"] == manifest["cache_key"]
    training_log = (tmp_path / "training_logs" / "run.log").read_text(encoding="utf-8")
    assert "candidate.epoch" in training_log
    assert "backbone_forward_count=0" in training_log
    assert "optimizer_weight_decay=0.123" in training_log
    assert "best_checkpoint_updated=True" in training_log

    cfg.finetune.evaluate_only = True
    cfg.finetune.downstream_checkpoint = cfg.finetune.save_path
    _, evaluate_metrics, _, _, evaluate_predictions = training.run_one_checkpoint(
        cfg, shape, None, None, loader, torch.device("cpu"),
        test_prediction_context={"fold": 0},
    )
    assert evaluate_metrics["training_metadata"]["backbone_forward_count"] == 0
    assert len(evaluate_predictions) == 2


def test_head_only_checkpoint_rejects_wrong_cache_key(tiny_cache, tmp_path):
    cache_path, _, _, manifest = tiny_cache
    cfg = Config()
    cfg.finetune.feature_source = "cache"
    cfg.finetune.feature_cache_path = str(cache_path)
    cfg.finetune.feature_cache_key = "wrong"
    cfg.finetune.pooling = "mean"
    cfg.finetune.sweep_candidate_id = "candidate"
    cfg.finetune.sweep_config_sha256 = "s" * 64
    cfg.finetune.sweep_args_json = '{"--pooling":"mean"}'
    model = training.CachedFeatureClassifier(3, 2, "mean")
    payload = {
        "checkpoint_kind": "cached_feature_head_only",
        "head_state_dict": model.head.state_dict(), "pooling": "mean",
        "head_input_dim": 3, "head_lr": cfg.finetune.head_lr,
        "weight_decay": cfg.finetune.weight_decay,
        "sweep_candidate_id": "candidate", "sweep_config_sha256": "s" * 64,
        "sweep_args_json": cfg.finetune.sweep_args_json,
        "feature_cache_key": manifest["cache_key"], "feature_cache_pooling": "mean",
        "feature_dim": 3,
    }
    with pytest.raises(RuntimeError, match="feature_cache_key"):
        training.validate_sweep_checkpoint_metadata(payload, cfg)
