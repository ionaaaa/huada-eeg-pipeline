from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


PROJECT = Path(__file__).resolve().parents[1]
MODULE_DIR = PROJECT / "scripts" / "downstream_50m"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import age_cv7
import largeScale_fineTune as training
from configs import Config


AGES = {
    "SUB1": 31.0, "SUB2": 29.0, "SUB3": 26.0, "SUB4": 26.0,
    "SUB6": 28.0, "SUB7": 28.0, "SUB8": 22.0,
}


def test_cv7_is_deterministic_complete_and_subject_isolated():
    left = age_cv7.build_age_subject_cv7(AGES, 42)
    right = age_cv7.build_age_subject_cv7(AGES, 42)
    assert left == right
    assert len(left["folds"]) == 7
    tests = []
    validations = []
    train_counts = {subject: 0 for subject in AGES}
    for fold in left["folds"]:
        train = set(fold["train_subjects"])
        validation = set(fold["validation_subjects"])
        test = set(fold["test_subjects"])
        assert (len(train), len(validation), len(test)) == (5, 1, 1)
        assert not (train & validation or train & test or validation & test)
        assert train | validation | test == set(AGES)
        tests.extend(test)
        validations.extend(validation)
        for subject in train:
            train_counts[subject] += 1
    assert sorted(tests) == sorted(AGES)
    assert sorted(validations) == sorted(AGES)
    assert set(train_counts.values()) == {5}


def test_different_seed_changes_permutation_but_remains_valid():
    first = age_cv7.build_age_subject_cv7(AGES, 42)
    second = age_cv7.build_age_subject_cv7(AGES, 43)
    assert first["permutation"] != second["permutation"]
    assert age_cv7.validate_age_cv7_payload(second)


def test_cv7_requires_exactly_seven_finite_eligible_subjects():
    with pytest.raises(ValueError, match="exactly 7"):
        age_cv7.build_age_subject_cv7({**AGES, "SUB5": 30.0}, 42)
    invalid = dict(AGES)
    invalid["SUB1"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        age_cv7.build_age_subject_cv7(invalid, 42)


def test_outside_train_age_range_is_protocol_metadata_not_exclusion():
    payload = age_cv7.build_age_subject_cv7(AGES, 42)
    test_subjects = {fold["test_subject"] for fold in payload["folds"]}
    assert test_subjects == set(AGES)
    for fold in payload["folds"]:
        expected = not (
            fold["train_age_min"] <= fold["test_age"] <= fold["train_age_max"]
        )
        assert fold["test_is_outside_train_age_range"] is expected


def test_single_subject_r2_is_null_and_oof_r2_is_defined():
    single = age_cv7.regression_metrics([28, 28, 28], [27, 28, 29])
    assert single["r2"] is None
    assert single["r2_defined"] is False
    oof = age_cv7.regression_metrics([22, 26, 28, 31], [23, 25, 28, 30])
    assert oof["r2"] is not None
    assert oof["r2_defined"] is True
    assert oof["mae"] == pytest.approx(0.75)
    assert oof["rmse"] == pytest.approx((3 / 4) ** 0.5)


def test_best_epoch_uses_mae_rmse_loss_then_earlier_with_tolerance():
    incumbent = {"mae": 1.0, "rmse": 1.2, "loss": 1.44, "epoch": 4}
    assert age_cv7.validation_result_is_better(
        {"mae": .9, "rmse": 9, "loss": 9, "epoch": 9}, incumbent
    )
    assert age_cv7.validation_result_is_better(
        {"mae": 1, "rmse": 1.1, "loss": 9, "epoch": 9}, incumbent
    )
    assert age_cv7.validation_result_is_better(
        {"mae": 1, "rmse": 1.2, "loss": 1.4, "epoch": 9}, incumbent
    )
    assert age_cv7.validation_result_is_better(
        {"mae": 1 + 1e-12, "rmse": 1.2, "loss": 1.44, "epoch": 3}, incumbent
    )
    assert not age_cv7.validation_result_is_better(
        {"mae": 1, "rmse": 1.2, "loss": 1.44, "epoch": 5}, incumbent
    )


def test_candidate_ranking_never_uses_r2_or_test_metrics():
    rows = [
        {"candidate_id": "b", "best_validation_mae": 1, "best_validation_rmse": 2,
         "best_validation_loss": 4, "test_mae": 0, "r2": 999},
        {"candidate_id": "a", "best_validation_mae": 1, "best_validation_rmse": 2,
         "best_validation_loss": 4, "test_mae": 999, "r2": -999},
    ]
    assert age_cv7.rank_candidates(rows)[0]["candidate_id"] == "a"


def test_subject_mean_median_aggregation_and_oof_coverage_rejects_sub5():
    rows = [
        {"sample_id": "A", "subject_id": "SUB1", "true_age": 31,
         "predicted_age": 30, "fold": 0, "test_is_outside_train_age_range": True},
        {"sample_id": "B", "subject_id": "SUB1", "true_age": 31,
         "predicted_age": 34, "fold": 0, "test_is_outside_train_age_range": True},
        {"sample_id": "C", "subject_id": "SUB2", "true_age": 29,
         "predicted_age": 28, "fold": 1, "test_is_outside_train_age_range": False},
    ]
    subjects = age_cv7.aggregate_subject_predictions(rows)
    assert subjects[0]["mean_predicted_age"] == pytest.approx(32)
    assert subjects[0]["median_predicted_age"] == pytest.approx(32)
    assert subjects[0]["prediction_std"] == pytest.approx(2)
    metrics = age_cv7.subject_aggregation_metrics(subjects)
    assert metrics["mean"]["sample_count"] == 2
    age_cv7.validate_oof_coverage(
        rows, {row["sample_id"]: {} for row in rows}, {"SUB1": 0, "SUB2": 1}
    )
    bad = [dict(rows[0], sample_id="X", subject_id="SUB5")]
    with pytest.raises(ValueError, match="SUB5"):
        age_cv7.validate_oof_coverage(bad, {"X": {}}, {"SUB5": 0})


def test_train_mean_baseline_uses_only_fold_train_subjects():
    fold = age_cv7.build_age_subject_cv7(AGES, 42)["folds"][0]
    expected = sum(AGES[subject] for subject in fold["train_subjects"]) / 5
    assert age_cv7.train_mean_baseline(fold, AGES) == pytest.approx(expected)
    changed = dict(AGES)
    changed[fold["test_subject"]] += 1000
    changed[fold["validation_subject"]] += 1000
    assert age_cv7.train_mean_baseline(fold, changed) == pytest.approx(expected)


def test_cached_age_head_shape_parameter_count_optimizer_and_fingerprint():
    cfg = Config()
    cfg.finetune.task = "age_regression"
    cfg.finetune.num_classes = 1
    cfg.finetune.feature_source = "cache"
    cfg.finetune.pooling = "mean"
    cfg.finetune.head_lr = 0.0003
    cfg.finetune.weight_decay = 0.001
    shape = training.compute_shape_params(cfg)
    torch.manual_seed(42)
    model, _ = training.build_downstream_model(cfg, None, torch.device("cpu"), shape)
    assert tuple(model.head.weight.shape) == (1, 512)
    assert tuple(model.head.bias.shape) == (1,)
    assert training.head_parameter_count(model) == 513
    first = training.head_state_fingerprint(model)
    torch.manual_seed(42)
    same, _ = training.build_downstream_model(cfg, None, torch.device("cpu"), shape)
    assert training.head_state_fingerprint(same) == first
    torch.manual_seed(43)
    other, _ = training.build_downstream_model(cfg, None, torch.device("cpu"), shape)
    assert training.head_state_fingerprint(other) != first
    optimizer = training.make_optimizer(cfg, model)
    report = training.optimizer_configuration(optimizer)["param_groups"][0]
    assert report["learning_rate"] == pytest.approx(0.0003)
    assert report["weight_decay"] == pytest.approx(0.001)
    assert report["parameter_count"] == 513


def test_age_head_only_checkpoint_has_strict_metadata_and_no_backbone(tmp_path):
    cfg = Config()
    cfg.finetune.task = "age_regression"
    cfg.finetune.num_classes = 1
    cfg.finetune.feature_source = "cache"
    cfg.finetune.pooling = "mean"
    cfg.finetune.feature_cache_key = "k" * 64
    cfg.finetune.feature_cache_path = str(tmp_path / "cache")
    cfg.finetune.fold_index = 2
    cfg.finetune.seed = 44
    cfg.finetune.sweep_candidate_id = "candidate"
    cfg.finetune.sweep_config_sha256 = "c" * 64
    cfg.finetune.sweep_args_json = "{}"
    cfg.finetune.initial_head_fingerprint = "h" * 64
    shape = training.compute_shape_params(cfg)
    model, layer = training.build_downstream_model(cfg, None, torch.device("cpu"), shape)
    optimizer = training.make_optimizer(cfg, model)
    path = tmp_path / "head.pt"
    training.save_downstream_checkpoint(
        path, model, cfg, shape, layer, 3,
        {"validation": {"mae": 1, "rmse": 1, "loss": 1}},
        {"checkpoint_sha256": "p" * 64}, training.optimizer_configuration(optimizer),
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["checkpoint_kind"] == "cached_feature_age_regression_head_only"
    assert payload["label_unit"] == "years"
    assert payload["age_target_normalization"] == "none"
    assert payload["age_prediction_clipping"] == "none"
    assert "head_state_dict" in payload
    assert "model_state_dict" not in payload
    assert "backbone_state_dict" not in payload
    loaded = training.load_downstream_checkpoint_strict(model, path, torch.device("cpu"))
    assert loaded["fold"] == 2

