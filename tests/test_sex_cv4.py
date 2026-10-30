from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn


PROJECT = Path(__file__).resolve().parents[1]
MODULE_DIR = PROJECT / "scripts" / "downstream_50m"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import largeScale_fineTune as training
from configs import Config
from sex_cv4 import (
    PROTOCOL,
    aggregate_subject_predictions,
    binary_classification_metrics,
    build_sex_stratified_subject_cv4,
    split_rows,
    validate_oof_window_predictions,
    write_oof_artifacts,
)


SUBJECT_LABELS = {
    "F01": 0, "F02": 0, "F03": 0, "F04": 0,
    "M01": 1, "M02": 1, "M03": 1, "M04": 1,
}


def _payload(seed=42):
    return build_sex_stratified_subject_cv4(SUBJECT_LABELS, seed)


def test_cv4_fold_count_sizes_balance_no_overlap_and_test_once():
    payload = _payload()
    assert payload["protocol"] == PROTOCOL
    assert len(payload["folds"]) == 4
    test_counts = Counter()
    for fold in payload["folds"]:
        train = set(fold["train_subjects"])
        validation = set(fold["validation_subjects"])
        test = set(fold["test_subjects"])
        assert (len(train), len(validation), len(test)) == (4, 2, 2)
        assert not (train & validation or train & test or validation & test)
        assert Counter(SUBJECT_LABELS[s] for s in train) == {0: 2, 1: 2}
        assert Counter(SUBJECT_LABELS[s] for s in validation) == {0: 1, 1: 1}
        assert Counter(SUBJECT_LABELS[s] for s in test) == {0: 1, 1: 1}
        test_counts.update(test)
    assert test_counts == Counter({subject: 1 for subject in SUBJECT_LABELS})


def test_cv4_seed_reproducibility_and_other_seed_remains_legal():
    assert _payload(42) == _payload(42)
    baseline = _payload(42)["pairs"]
    alternatives = [_payload(seed) for seed in range(43, 60)]
    assert all(len(value["folds"]) == 4 for value in alternatives)
    assert any(value["pairs"] != baseline for value in alternatives)


def test_manifest_rows_assign_every_subject_once_per_fold():
    rows = split_rows(_payload())
    assert len(rows) == 32
    for fold in range(4):
        fold_rows = [row for row in rows if row["fold"] == fold]
        assert {row["subject_id"] for row in fold_rows} == set(SUBJECT_LABELS)
        assert Counter(row["split"] for row in fold_rows) == {
            "train": 4, "validation": 2, "test": 2,
        }


def _oof_rows(payload, windows_per_subject=2):
    test_fold = {
        subject: fold["fold"]
        for fold in payload["folds"]
        for subject in fold["test_subjects"]
    }
    rows = []
    eligible = {}
    for subject, label in sorted(SUBJECT_LABELS.items()):
        for window in range(windows_per_subject):
            sample = f"{subject}_W{window}"
            eligible[sample] = {"subject_id": subject, "label": label}
            probability = 0.8 if label == 1 else 0.2
            rows.append({
                "fold": test_fold[subject],
                "sample_id": sample,
                "subject_id": subject,
                "sex": "male" if label else "female",
                "true_label": label,
                "prob_female": 1.0 - probability,
                "prob_male": probability,
                "predicted_label": label,
                "predicted_sex": "male" if label else "female",
                "num_valid_tokens": 640,
                "checkpoint_path": f"fold_{test_fold[subject]}/best.pt",
            })
    return rows, eligible, test_fold


def test_oof_exact_coverage_probabilities_and_subject_test_fold():
    payload = _payload()
    rows, eligible, test_fold = _oof_rows(payload)
    assert validate_oof_window_predictions(rows, eligible, test_fold)
    duplicated = rows + [dict(rows[0])]
    with pytest.raises(ValueError, match="Duplicate"):
        validate_oof_window_predictions(duplicated, eligible, test_fold)
    wrong_fold = [dict(row) for row in rows]
    wrong_fold[0]["fold"] = (wrong_fold[0]["fold"] + 1) % 4
    with pytest.raises(ValueError, match="fold mismatch"):
        validate_oof_window_predictions(wrong_fold, eligible, test_fold)
    wrong_probability = [dict(row) for row in rows]
    wrong_probability[0]["prob_male"] = 0.9
    with pytest.raises(ValueError, match="sum to one"):
        validate_oof_window_predictions(wrong_probability, eligible, test_fold)


def test_subject_probability_mean_majority_vote_and_tie_break():
    rows = [
        {"subject_id": "F01", "true_label": 0, "fold": 0, "prob_female": 0.9, "prob_male": 0.1, "predicted_label": 0},
        {"subject_id": "F01", "true_label": 0, "fold": 0, "prob_female": 0.4, "prob_male": 0.6, "predicted_label": 1},
        {"subject_id": "M01", "true_label": 1, "fold": 1, "prob_female": 0.5, "prob_male": 0.5, "predicted_label": 0},
        {"subject_id": "M01", "true_label": 1, "fold": 1, "prob_female": 0.5, "prob_male": 0.5, "predicted_label": 1},
        {"subject_id": "M01", "true_label": 1, "fold": 1, "prob_female": 0.5, "prob_male": 0.5, "predicted_label": 0},
        {"subject_id": "M01", "true_label": 1, "fold": 1, "prob_female": 0.5, "prob_male": 0.5, "predicted_label": 1},
    ]
    subjects = {row["subject_id"]: row for row in aggregate_subject_predictions(rows)}
    assert subjects["F01"]["probability_mean_pred_label"] == 0
    assert subjects["F01"]["majority_vote_pred_label"] == 0
    assert subjects["F01"]["tie_break_used"] is True
    assert subjects["M01"]["majority_vote_pred_label"] == 0
    assert subjects["M01"]["tie_break_used"] is True


def test_binary_metrics_known_confusion_accuracy_balanced_accuracy_macro_f1():
    metrics = binary_classification_metrics([0, 0, 1, 1], [0, 1, 1, 1])
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["balanced_accuracy"] == pytest.approx(0.75)
    assert metrics["macro_f1"] == pytest.approx((2 / 3 + 0.8) / 2)


def test_write_oof_artifacts_has_all_required_files(tmp_path):
    payload = _payload()
    rows, eligible, _ = _oof_rows(payload)
    per_fold = []
    for fold in range(4):
        fold_rows = [row for row in rows if row["fold"] == fold]
        per_fold.append({
            "fold": fold, "fold_seed": 42 + fold, "num_windows": len(fold_rows),
            "num_subjects": 2, "loss": 0.1, "accuracy": 1.0,
            "balanced_accuracy": 1.0, "macro_f1": 1.0,
            "confusion_matrix": "[[1,0],[0,1]]", "checkpoint_path": f"fold_{fold}/best.pt",
            "checkpoint_sha256": "a" * 64, "matched_key_count": 1,
            "matched_parameter_fraction": 1.0,
        })
    summary = write_oof_artifacts(
        tmp_path, payload, rows, per_fold, eligible,
        {str(index): {"matched_parameter_fraction": 1.0} for index in range(4)},
        512, 1026,
        {
            "pooling": "mean", "head_lr": 1e-3, "weight_decay": 1e-3,
            "optimizer_name": "SGD", "num_tokens": 640, "d_model": 512,
        },
    )
    required = {
        "sex_cv4_oof_window_predictions.csv", "sex_cv4_oof_subject_predictions.csv",
        "sex_cv4_per_fold_metrics.csv", "sex_cv4_window_metrics.json",
        "sex_cv4_subject_metrics.json", "sex_cv4_summary.json", "sex_cv4_report.md",
    }
    assert required.issubset({path.name for path in tmp_path.iterdir()})
    assert summary["num_windows"] == len(eligible)
    assert summary["num_subjects"] == 8
    assert summary["head_input_dim"] == 512
    assert summary["head_parameter_count"] == 1026


def test_one_bad_channel_tokens_are_excluded_even_with_huge_values():
    class IdentityBackbone(nn.Module):
        def extract_token_embeddings(self, token_inputs, **kwargs):
            return token_inputs

    model = training.EEGDownstreamClassifier(IdentityBackbone(), 512, 2)
    embeddings = torch.ones(1, 640, 512)
    embeddings[:, 170:180] = 1e20
    mask = torch.ones(1, 640, dtype=torch.bool)
    mask[:, 170:180] = False
    pooled = model.pool_valid_tokens(embeddings, mask)
    assert mask.sum().item() == 630
    assert pooled.shape == (1, 512)
    assert torch.allclose(pooled, torch.ones_like(pooled))


def test_fresh_heads_do_not_share_parameters():
    class Backbone(nn.Module):
        def extract_token_embeddings(self, token_inputs, **kwargs):
            return token_inputs

    first = training.EEGDownstreamClassifier(Backbone(), 512, 2)
    second = training.EEGDownstreamClassifier(Backbone(), 512, 2)
    assert first.head is not second.head
    assert first.head.weight.data_ptr() != second.head.weight.data_ptr()


def test_low_backbone_checkpoint_match_fails(monkeypatch, tmp_path):
    class TinyPretrain(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.large = nn.Parameter(torch.zeros(100))
            self.small = nn.Parameter(torch.zeros(1))

    monkeypatch.setattr(training, "EEGPretrainModel", TinyPretrain)
    checkpoint = tmp_path / "partial.pt"
    torch.save({"model_state_dict": {"small": torch.ones(1)}}, checkpoint)
    cfg = Config()
    cfg.finetune.pretrained_ckpt_path = str(checkpoint)
    with pytest.raises(RuntimeError, match="matched only"):
        training.build_backbone(cfg, training.compute_shape_params(cfg), torch.device("cpu"))


class _MetadataDataset:
    def __init__(self):
        self.rows = []
        for subject, label in sorted(SUBJECT_LABELS.items()):
            self.rows.append({
                "index": len(self.rows), "sample_id": f"{subject}_W0",
                "subject_id": subject, "label": label,
            })

    def __len__(self):
        return len(self.rows)

    def get_subject_label_rows(self):
        return list(self.rows)


def test_cv4_orchestration_smoke_writes_fold_checkpoints_and_oof(monkeypatch, tmp_path):
    dataset = _MetadataDataset()
    monkeypatch.setattr(training, "load_downstream_dataset", lambda cfg: dataset)
    monkeypatch.setattr(training, "build_loaders", lambda train, val, test, cfg, device: (train, val, test))
    calls = []

    def fake_run(cfg, shape, train, validation, test, device, **kwargs):
        fold = kwargs["test_prediction_context"]["fold"]
        checkpoint = Path(cfg.finetune.save_path)
        checkpoint.write_bytes(f"fold-{fold}".encode())
        calls.append((fold, id(cfg), checkpoint))
        predictions = []
        for local_index in test.indices:
            row = dataset.rows[local_index]
            label = int(row["label"])
            predictions.append({
                "fold": fold, "sample_id": row["sample_id"], "subject_id": row["subject_id"],
                "sex": "male" if label else "female", "true_label": label,
                "prob_female": 0.9 if label == 0 else 0.1,
                "prob_male": 0.1 if label == 0 else 0.9,
                "predicted_label": label, "predicted_sex": "male" if label else "female",
                "num_valid_tokens": 630, "checkpoint_path": str(checkpoint),
            })
        metrics = {
            "loss": 0.1, "num_samples": len(predictions), "accuracy": 1.0,
            "balanced_accuracy": 1.0, "macro_f1": 1.0,
            "confusion_matrix": [[1, 0], [0, 1]],
        }
        report = {
            "checkpoint_path": "pretrained.pt", "checkpoint_sha256": "b" * 64,
            "matched_key_count": 100, "missing_keys": [], "unexpected_keys": [],
            "shape_mismatch_keys": [], "matched_parameter_fraction": 1.0,
        }
        training_metadata = {
            "pooling": "mean", "head_lr": 1e-3, "weight_decay": 1e-3,
            "optimizer_name": "SGD", "optimizer_param_groups": [],
            "head_input_dim": 512, "head_parameter_count": 1026,
            "num_tokens": 640, "d_model": 512,
        }
        return object(), {
            "epochs": [], "test": metrics, "training_metadata": training_metadata,
        }, 8, report, predictions

    monkeypatch.setattr(training, "run_one_checkpoint", fake_run)
    cfg = Config()
    cfg.finetune.task = "sex_classification"
    cfg.finetune.protocol = PROTOCOL
    cfg.finetune.folds = 4
    cfg.finetune.pretrained_ckpt_path = "pretrained.pt"
    cfg.finetune.num_epochs = 1
    result, reports = training.run_sex_cv4(
        cfg, training.compute_shape_params(cfg), torch.device("cpu"), tmp_path,
    )
    assert len(calls) == 4
    assert len({value[1] for value in calls}) == 4
    assert len({value[2] for value in calls}) == 4
    assert all(path.exists() for _, _, path in calls)
    assert len(reports) == 4
    assert result["sample_count"] == 8
    with (tmp_path / "sex_cv4_oof_window_predictions.csv").open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 8
    assert len({row["sample_id"] for row in rows}) == 8
    assert json.loads((tmp_path / "sex_cv4_summary.json").read_text(encoding="utf-8"))["pooling"] == "mean"
