from __future__ import annotations

import csv
import hashlib
import json
import sys
from argparse import Namespace
from pathlib import Path

import lance
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset


PROJECT = Path(__file__).resolve().parents[1]
MODULE_DIR = PROJECT / "scripts" / "downstream_50m"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import feature_cache as cache
import run_neuralbench_sex_probe_transfer_cv4 as transfer
from run_logging import setup_run_logging
from sex_cv4 import aggregate_subject_predictions, build_sex_stratified_subject_cv4


def _probe_state():
    return {"weight": torch.zeros(2, 512), "bias": torch.zeros(2)}


def _write_probe_checkpoint(path: Path):
    torch.save({"state_dict": {
        "model.probe.weight": _probe_state()["weight"],
        "model.probe.bias": _probe_state()["bias"],
    }}, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_source_probe_safe_load_sha_shape_and_fingerprint(monkeypatch, tmp_path):
    path = tmp_path / "source.ckpt"
    digest = _write_probe_checkpoint(path)
    real_load = transfer.torch.load
    calls = []

    def wrapped(*args, **kwargs):
        calls.append(kwargs)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(transfer.torch, "load", wrapped)
    state, fingerprint, info = transfer.load_source_probe_state(path, digest)
    assert state["weight"].shape == (2, 512) and state["bias"].shape == (2,)
    assert len(fingerprint) == 64 and info["sha256"] == digest
    assert calls == [{"map_location": "cpu", "weights_only": True}]
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        transfer.load_source_probe_state(path, "0" * 64)
    assert len(calls) == 1


class MeanSource(Dataset):
    def __init__(self):
        self.metadata = [
            {"sample_id": "NORMAL_W0", "subject_id": "SUB1", "recording_id": "R1"},
            {"sample_id": "SUB5_W0", "subject_id": "SUB5", "recording_id": "R5"},
            {"sample_id": "SUB8_W0", "subject_id": "SUB8", "recording_id": "R8"},
        ]

    def __len__(self):
        return 3

    def __getitem__(self, index):
        feature = torch.arange(4 * 512, dtype=torch.float32).reshape(4, 512) + index
        mask = torch.tensor([1, 1, 0, 1], dtype=torch.float32)
        return {
            **self.metadata[index], "token_inputs": feature,
            "token_channel_indices": torch.arange(4), "token_time_indices": torch.arange(4),
            "token_valid_mask": mask,
        }


class IdentityBackbone(nn.Module):
    def __init__(self):
        super().__init__(); self.forward_count = 0

    def extract_token_embeddings(self, token_inputs, **kwargs):
        self.forward_count += 1
        return token_inputs


def test_mean_only_cache_schema_label_free_and_source_probe_parity(tmp_path):
    source = MeanSource(); backbone = IdentityBackbone(); head = nn.Linear(512, 2)
    identity = {"fixture": "neuralbench-mean-v1"}
    manifest = cache.build_feature_cache(
        source_dataset=source, backbone=backbone, probe_layer_idx=8,
        output_dir=tmp_path / "cache", batch_size=2, device=torch.device("cpu"),
        expected_num_tokens=4, d_model=512,
        manifest_base={
            "cache_key": cache.sha256_json(identity), "cache_identity": identity,
            "input_row_count": 3, "sample_ids": [row["sample_id"] for row in source.metadata],
        }, requested_poolings=("mean",), parity_head=head,
    )
    cache_dir = tmp_path / "cache"
    assert (cache_dir / cache.MEAN_CACHE_NAME).is_dir()
    assert not (cache_dir / cache.FLATTEN_CACHE_NAME).exists()
    ds = lance.dataset(str(cache_dir / cache.MEAN_CACHE_NAME))
    assert ds.schema.field("feature").type.list_size == 512
    assert not set(ds.schema.names).intersection({"label", "sex", "split", "fold"})
    assert manifest["requested_poolings"] == ["mean"]
    assert manifest["parity_passed"] is True
    assert manifest["mean_max_abs_diff"] == manifest["mean_logits_max_abs_diff"] == 0.0
    assert [row["sample_id"] for row in manifest["parity_records"]] == ["NORMAL_W0", "SUB5_W0", "SUB8_W0"]
    assert all(set(row) >= {"online_feature", "cached_feature", "online_logits", "cached_logits", "parity_passed"} for row in manifest["parity_records"])
    assert backbone.forward_count == 2
    cache.validate_feature_cache(cache_dir, identity, manifest["cache_key"])


def test_neuralbench_cache_identity_contains_representation_not_training_settings():
    identity, key = cache.build_cache_identity(
        input_fingerprint="input", input_row_count=480, sample_id_fingerprint="samples",
        checkpoint_sha256="checkpoint", model_config={"d_model": 512, "depth": 12, "n_heads": 8},
        shape_params={"num_tokens": 640, "patch_len": 100}, probe_layer_idx=8,
        probe_layer_ratio=0.75, model_code_sha256="model-code", git_commit="commit",
        requested_poolings=("mean",), checkpoint_remap_version="remap-v1",
        feature_dim=512, feature_extraction_code_sha256="feature-code",
        source_checkpoint_kind="neuralbench",
    )
    assert identity["requested_poolings"] == ["mean"]
    assert identity["pretrained_checkpoint_sha256"] == "checkpoint"
    assert identity["checkpoint_remap_version"] == "remap-v1"
    assert identity["feature_dim"] == 512 and identity["cache_dtype"] == "float32"
    assert identity["probe_layer_idx"] == 8 and "token_valid_mask_semantics_version" in identity
    assert not set(identity).intersection({"fold", "head_lr", "weight_decay", "epochs", "patience", "batch_size", "seed"})
    changed = dict(identity, pretrained_checkpoint_sha256="other")
    assert cache.sha256_json(changed) != key


def test_source_initialized_probe_has_no_random_values_and_exactly_1026_trainable():
    state = {"weight": torch.arange(1024).reshape(2, 512).float(), "bias": torch.tensor([1.0, -1.0])}
    source_before = {key: value.clone() for key, value in state.items()}
    first = transfer.build_source_initialized_probe(state, torch.device("cpu"))
    second = transfer.build_source_initialized_probe(state, torch.device("cpu"))
    assert torch.equal(first.head.weight, state["weight"])
    assert torch.equal(first.head.bias, state["bias"])
    assert transfer.tensor_state_fingerprint(transfer.probe_state_cpu(first)) == transfer.tensor_state_fingerprint(transfer.probe_state_cpu(second))
    assert sum(parameter.numel() for parameter in first.parameters() if parameter.requires_grad) == 1026
    assert not hasattr(first, "backbone")
    with torch.no_grad():
        first.head.weight.add_(1.0)
    assert all(torch.equal(state[key], source_before[key]) for key in state)


def test_optimizer_contains_only_probe_and_resolved_adamw_values():
    model = transfer.build_source_initialized_probe(_probe_state(), torch.device("cpu"))
    optimizer = transfer.make_probe_optimizer(model, 1e-5, 0.0)
    assert optimizer.__class__.__name__ == "AdamW"
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["name"] == "probe"
    assert sum(p.numel() for p in optimizer.param_groups[0]["params"]) == 1026
    assert optimizer.param_groups[0]["lr"] == 1e-5
    assert optimizer.param_groups[0]["weight_decay"] == 0.0


@pytest.mark.parametrize(
    "candidate,best,expected",
    [
        ({"epoch": 1, "balanced_accuracy": .6, "loss": 2., "macro_f1": .1}, {"epoch": 0, "balanced_accuracy": .5, "loss": 1., "macro_f1": .9}, True),
        ({"epoch": 1, "balanced_accuracy": .5, "loss": .9, "macro_f1": .1}, {"epoch": 0, "balanced_accuracy": .5, "loss": 1., "macro_f1": .9}, True),
        ({"epoch": 1, "balanced_accuracy": .5, "loss": 1., "macro_f1": .8}, {"epoch": 0, "balanced_accuracy": .5, "loss": 1., "macro_f1": .7}, True),
        ({"epoch": 1, "balanced_accuracy": .5 + 1e-12, "loss": 1. - 1e-12, "macro_f1": .7 + 1e-12}, {"epoch": 0, "balanced_accuracy": .5, "loss": 1., "macro_f1": .7}, False),
    ],
)
def test_epoch_selection_rule_and_numeric_tolerance(candidate, best, expected):
    assert transfer.candidate_is_better(candidate, best) is expected


SUBJECT_LABELS = {"SUB1": 1, "SUB2": 0, "SUB3": 0, "SUB4": 0, "SUB5": 1, "SUB6": 1, "SUB7": 0, "SUB8": 1}


class ProbeFixtureDataset(Dataset):
    def __init__(self, windows_per_subject=2):
        self.rows = []
        for subject, label in sorted(SUBJECT_LABELS.items()):
            for window in range(windows_per_subject):
                feature = torch.zeros(512)
                feature[0] = 1.0 if label == 1 else -1.0
                self.rows.append({
                    "sample_id": f"{subject}_W{window}", "subject_id": subject,
                    "recording_id": "R", "feature": feature,
                    "num_valid_tokens": torch.tensor(4), "label_name": "male" if label else "female",
                    "label": torch.tensor(label),
                })

    def __len__(self): return len(self.rows)
    def __getitem__(self, index): return self.rows[index]
    def __getitems__(self, indices): return [self.rows[int(i)] for i in indices]
    def get_subject_label_rows(self):
        return [{"index": i, "sample_id": r["sample_id"], "subject_id": r["subject_id"], "label": int(r["label"])} for i, r in enumerate(self.rows)]


def _cache_info(tmp_path):
    path = tmp_path / "cache"; path.mkdir(exist_ok=True)
    return {"feature_cache_key": "c" * 64, "feature_cache_path": str(path), "manifest": {}}


def test_fold_epoch0_early_stop_test_isolation_and_head_only_checkpoint(monkeypatch, tmp_path):
    dataset = ProbeFixtureDataset()
    split = build_sex_stratified_subject_cv4(SUBJECT_LABELS, 42)["folds"][0]
    calls = []

    def fake_source(*args):
        calls.append(1)
        return _probe_state(), transfer.tensor_state_fingerprint(_probe_state()), {"sha256": "a" * 64}

    monkeypatch.setattr(transfer, "load_source_probe_state", fake_source)
    logger = setup_run_logging(tmp_path / "logs", "INFO", append=False)
    summary, rows = transfer.run_fold(
        fold_info=split, dataset=dataset, checkpoint_path="fake.ckpt", expected_sha256="a" * 64,
        cache_info=_cache_info(tmp_path), output_dir=tmp_path, device=torch.device("cpu"),
        batch_size=4, num_workers=0, seed=42, head_lr=1e-20, weight_decay=0.0,
        epochs=10, patience=2, logger=logger,
    )
    assert calls == [1]
    assert summary["best_epoch"] == 0 and summary["adaptation_applied"] is False
    assert summary["stopped_early"] is True and summary["epochs_executed"] == 2
    assert summary["test_accessed_during_selection"] is False
    assert summary["test_evaluation_count"] == 1 and len(rows) == 4
    assert summary["backbone_constructed"] is False and summary["backbone_forward_count"] == 0
    checkpoint = torch.load(tmp_path / "fold_0" / "checkpoints" / "best_probe.pt", weights_only=True)
    assert checkpoint["checkpoint_kind"] == transfer.CHECKPOINT_KIND
    assert set(checkpoint["probe_state_dict"]) == {"weight", "bias"}
    assert not set(checkpoint).intersection({"backbone", "model_state_dict", "feature_cache"})
    assert checkpoint["best_epoch"] == 0
    history = list(csv.DictReader((tmp_path / "fold_0" / "epoch_history.csv").open()))
    assert history[0]["epoch"] == "0" and history[0]["is_best"] == "True"


def test_fold_adaptation_improves_and_restores_best(monkeypatch, tmp_path):
    dataset = ProbeFixtureDataset()
    split = build_sex_stratified_subject_cv4(SUBJECT_LABELS, 42)["folds"][1]
    fingerprint = transfer.tensor_state_fingerprint(_probe_state())
    monkeypatch.setattr(transfer, "load_source_probe_state", lambda *args: (_probe_state(), fingerprint, {"sha256": "a" * 64}))
    summary, _ = transfer.run_fold(
        fold_info=split, dataset=dataset, checkpoint_path="fake", expected_sha256="a" * 64,
        cache_info=_cache_info(tmp_path), output_dir=tmp_path, device=torch.device("cpu"),
        batch_size=4, num_workers=0, seed=42, head_lr=0.1, weight_decay=0.0,
        epochs=3, patience=2, logger=setup_run_logging(tmp_path / "logs", "INFO", append=False),
    )
    assert summary["best_epoch"] > 0 and summary["adaptation_applied"] is True
    assert summary["probe_parameter_delta"]["changed_parameter_count"] > 0
    assert summary["best_validation"]["balanced_accuracy"] >= summary["epoch0_validation"]["balanced_accuracy"]
    payload = torch.load(tmp_path / "fold_1" / "checkpoints" / "best_probe.pt", weights_only=True)
    assert payload["best_epoch"] == summary["best_epoch"]


def test_four_folds_restart_same_source_and_oof_is_complete(monkeypatch, tmp_path):
    dataset = ProbeFixtureDataset(windows_per_subject=1)
    payload = build_sex_stratified_subject_cv4(SUBJECT_LABELS, 42)
    fingerprint = transfer.tensor_state_fingerprint(_probe_state())
    calls = []

    def fake_source(*args):
        calls.append(fingerprint)
        return _probe_state(), fingerprint, {"sha256": "a" * 64}

    monkeypatch.setattr(transfer, "load_source_probe_state", fake_source)
    summaries=[]; rows=[]
    logger=setup_run_logging(tmp_path / "logs", "INFO", append=False)
    info=_cache_info(tmp_path)
    for fold in payload["folds"]:
        summary, predictions = transfer.run_fold(
            fold_info=fold, dataset=dataset, checkpoint_path="fake", expected_sha256="a" * 64,
            cache_info=info, output_dir=tmp_path, device=torch.device("cpu"), batch_size=4,
            num_workers=0, seed=42, head_lr=1e-20, weight_decay=0.0, epochs=1,
            patience=1, logger=logger,
        )
        summaries.append(summary); rows.extend(predictions)
    assert calls == [fingerprint] * 4
    result = transfer.aggregate_outputs(tmp_path, payload, dataset, summaries, rows)
    assert len(rows) == len({row["sample_id"] for row in rows}) == 8
    assert result["source_probe_fingerprint"] == fingerprint
    assert all(summary["test_evaluation_count"] == 1 for summary in summaries)
    assert result["window_metrics"]["sample_count"] == 8
    assert result["subject_probability_mean_metrics"]["sample_count"] == 8


def test_480_oof_coverage_validator_without_formal_model_execution():
    eligible={}; rows=[]; subject_folds={}
    for subject_index in range(8):
        subject=f"SUB{subject_index+1}"; fold=subject_index//2; label=subject_index%2
        subject_folds[subject]=fold
        for window in range(60):
            sample=f"{subject}_W{window}"; eligible[sample]={"subject_id":subject,"label":label}
            rows.append({"fold":fold,"sample_id":sample,"subject_id":subject,"true_label":label,"prob_female":1-label,"prob_male":label,"predicted_label":label})
    from sex_cv4 import validate_oof_window_predictions
    assert validate_oof_window_predictions(rows, eligible, subject_folds)
    assert len(rows)==len({row["sample_id"] for row in rows})==480


def test_zero_shot_comparison_validates_inputs_and_is_post_selection(tmp_path):
    zero=tmp_path/"zero"; zero.mkdir()
    _json=lambda name,value:(zero/name).write_text(json.dumps(value),encoding="utf-8")
    metrics={"accuracy":1.0,"balanced_accuracy":1.0,"macro_f1":1.0,"confusion_matrix":[[1,0],[0,1]],"sample_count":2}
    _json("zero_shot_summary.json",{"evaluation_kind":"external_checkpoint_zero_shot","window_metrics":metrics,"subject_probability_mean_metrics":metrics,"subject_majority_vote_metrics":metrics})
    _json("zero_shot_provenance.json",{"checkpoint":{"sha256":"a"*64},"input_lance":{"content_fingerprint_before_sha256":"b"*64}})
    with (zero/"window_predictions.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["sample_id"]); w.writeheader(); w.writerows([{"sample_id":"s0"},{"sample_id":"s1"}])
    with (zero/"subject_predictions.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["subject_id","probability_mean_pred_label"]); w.writeheader(); w.writerows([{"subject_id":"A","probability_mean_pred_label":0},{"subject_id":"B","probability_mean_pred_label":1}])
    subject_rows=[{"subject_id":"A","probability_mean_pred_label":0},{"subject_id":"B","probability_mean_pred_label":0}]
    result=transfer.compare_zero_shot(zero,{"window_metrics":metrics,"subject_probability_mean_metrics":metrics,"subject_majority_vote_metrics":metrics,"subject_rows":subject_rows},[{"sample_id":"s0"},{"sample_id":"s1"}],"a"*64,"b"*64)
    assert result["used_for_selection"] is False and result["comparison_performed_after_all_fold_tests"] is True
    assert sum(row["prediction_changed"] for row in result["subject_prediction_changes"])==1
    with pytest.raises(ValueError,match="fingerprint"):
        transfer.compare_zero_shot(zero,{"window_metrics":metrics,"subject_probability_mean_metrics":metrics,"subject_majority_vote_metrics":metrics,"subject_rows":subject_rows},[{"sample_id":"s0"},{"sample_id":"s1"}],"a"*64,"wrong")


def test_dry_run_does_not_create_output_cache_or_train(monkeypatch, tmp_path):
    monkeypatch.setattr(transfer,"build_cache_identity",lambda *args: ({"pooling":"mean"},"c"*64,"i"*64,{"row_count":4},"a"*64,{"commit":"x","dirty":False}))
    monkeypatch.setattr(transfer,"ensure_neuralbench_mean_cache",lambda *args: (_ for _ in ()).throw(AssertionError("must not build")))
    args=Namespace(input_lance="fixture",checkpoint="fake",expected_checkpoint_sha256="a"*64,output_dir=str(tmp_path/"out"),feature_cache_root=str(tmp_path/"cache"),feature_cache_policy="auto",feature_cache_batch_size=2,cache_log_every_batches=1,zero_shot_results_dir=None,head_lr=1e-5,weight_decay=0.,epochs=2,early_stopping_patience=1,batch_size=2,seed=42,split_seed=42,device="cpu",num_workers=0,log_level="INFO",dry_run=True)
    assert transfer.run(args)==0
    assert not (tmp_path/"out").exists() and not (tmp_path/"cache").exists()


def test_cloud_script_isolated_from_forbidden_entrypoints():
    text=(PROJECT/"commands"/"run_50m_neuralbench_sex_probe_transfer_cv4_full.sh").read_text(encoding="utf-8")
    lower=text.lower()
    assert "set -euo pipefail" in text and "PYTHONUNBUFFERED=1" in text
    assert '"$PYTHON_BIN" -u' in text
    for forbidden in ("run_50m_sex_cv4_sweep_full.sh","run_50m_sex_cv4_full.sh","run_neuralbench_sex_zero_shot.py","legacy single split","flatten search","full finetune"):
        assert forbidden not in lower
