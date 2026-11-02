from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import lance
import numpy as np
import pyarrow as pa
import pytest
import torch


PROJECT = Path(__file__).resolve().parents[1]
MODULE_DIR = PROJECT / "scripts" / "downstream_50m"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import age_cv7
import feature_cache
import largeScale_fineTune as training
import run_age_cv7_sweep as sweep
from configs import Config
from run_logging import setup_run_logging


AGES = {
    "SUB1": 31.0, "SUB2": 29.0, "SUB3": 26.0, "SUB4": 26.0,
    "SUB5": None, "SUB6": 28.0, "SUB7": 28.0, "SUB8": 22.0,
}


def _write_input_lance(path):
    rows = []
    for subject, age in AGES.items():
        for window in range(2):
            rows.append({
                "sample_id": f"{subject}_W{window}", "subject_id": subject, "age": age,
            })
    table = pa.table({
        "sample_id": [row["sample_id"] for row in rows],
        "subject_id": [row["subject_id"] for row in rows],
        "age": pa.array([row["age"] for row in rows], type=pa.float64()),
        "sex": ["male" if int(row["subject_id"][3:]) % 2 else "female" for row in rows],
    })
    lance.write_dataset(table, str(path), mode="create", data_storage_version="stable")
    return rows


def _options(tmp_path, dry_run=False):
    checkpoint = tmp_path / "pretrained.pt"
    checkpoint.write_bytes(b"fixture checkpoint")
    return SimpleNamespace(
        input_lance=str(tmp_path / "input.lance"), checkpoint=checkpoint,
        sweep_config=PROJECT / "configs" / "50m_age_cv7_sweep.json",
        output_dir=tmp_path / "output", storage_options=None, device="cpu",
        epochs=2, batch_size=4, num_workers=0, seed=42, split_seed=42,
        feature_cache_root=tmp_path / "cache_root", feature_cache_build_root=None,
        feature_cache_policy="auto", feature_cache_batch_size=2,
        cache_log_every_batches=1, train_log_every_epochs=1, log_level="INFO",
        dry_run=dry_run, trainer_script=PROJECT / "scripts" / "downstream_50m" / "largeScale_fineTune.py",
    )


def _cache_info(tmp_path, row_count):
    cache_path = tmp_path / "fake_cache" / ("c" * 64)
    (cache_path / "mean_features.lance").mkdir(parents=True)
    return {
        "feature_cache_key": "c" * 64,
        "feature_cache_path": str(cache_path),
        "feature_cache_policy": "auto",
        "feature_cache_reused": True,
        "cache_build_seconds": 0.0,
        "manifest": {
            "requested_poolings": ["mean"], "mean_row_count": row_count,
            "mean_feature_dim": 512, "dtype": "float32", "parity_passed": True,
            "backbone_forward_count": 0, "backbone_processed_window_count": row_count,
            "source_window_count": row_count,
        },
    }


def _value(command, flag):
    return command[command.index(flag) + 1]


def _fake_runner():
    calls = []

    def runner(command, **kwargs):
        del kwargs
        calls.append(command)
        stage = _value(command, "--stage")
        fold = int(_value(command, "--fold-index"))
        candidate = _value(command, "--sweep-candidate-id")
        output = Path(_value(command, "--output-dir"))
        output.mkdir(parents=True)
        if stage == "search":
            checkpoint_dir = output / "checkpoints"
            checkpoint_dir.mkdir()
            checkpoint = checkpoint_dir / "downstream_finetune.pt"
            checkpoint.write_bytes(f"{fold}:{candidate}".encode())
            lr = float(_value(command, "--head-lr"))
            wd = float(_value(command, "--weight-decay"))
            mae = 1.0 + lr + wd
            summary = {
                "fold": fold, "candidate_id": candidate, "feature_source": "cache",
                "feature_cache_key": "c" * 64, "backbone_forward_count": 0,
                "checkpoint_kind": "cached_feature_age_regression_head_only",
                "test_accessed_during_selection": False, "test_accessed": False,
                "best_epoch": 1, "best_validation_mae": mae,
                "best_validation_rmse": mae, "best_validation_loss": mae * mae,
                "best_validation_r2": None, "best_validation_r2_defined": False,
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": sweep.sha256_file(checkpoint),
                "initial_head_fingerprint": f"fold-{fold}-same-init",
            }
            sweep.atomic_write_json(output / "validation_summary.json", summary)
        else:
            test_subject = age_cv7.build_age_subject_cv7(
                {key: value for key, value in AGES.items() if value is not None}, 42
            )["folds"][fold]["test_subject"]
            true_age = AGES[test_subject]
            predictions = []
            for window in range(2):
                prediction = true_age + 1 + window * 0.1
                predictions.append({
                    "fold": fold, "split": "test", "sample_id": f"{test_subject}_W{window}",
                    "subject_id": test_subject, "recording_id": f"R{test_subject}",
                    "window_index": window, "true_age": true_age,
                    "predicted_age": prediction, "signed_error": prediction - true_age,
                    "absolute_error": abs(prediction - true_age), "num_valid_tokens": 640,
                    "candidate_id": candidate, "checkpoint_path": _value(command, "--downstream-checkpoint"),
                    "test_is_outside_train_age_range": str(
                        age_cv7.build_age_subject_cv7(
                            {key: value for key, value in AGES.items() if value is not None}, 42
                        )["folds"][fold]["test_is_outside_train_age_range"]
                    ).lower(),
                })
            sweep._write_csv(output / "test_predictions.csv", predictions)
            metrics = age_cv7.regression_metrics(
                [row["true_age"] for row in predictions],
                [row["predicted_age"] for row in predictions],
            )
            metrics.update({
                "loss": sum(row["signed_error"] ** 2 for row in predictions) / 2,
                "candidate_id": candidate, "evaluation_call_count": 1,
                "r2": None, "r2_defined": False,
                "checkpoint_kind": "cached_feature_age_regression_head_only",
            })
            sweep.atomic_write_json(output / "test_metrics.json", metrics)
        return SimpleNamespace(stdout="fixture ok\n", stderr="")

    return runner, calls


def test_formal_config_is_mean_only_four_candidates():
    _, candidates = sweep.load_age_sweep_config(
        PROJECT / "configs" / "50m_age_cv7_sweep.json"
    )
    assert len(candidates) == 4
    assert {row["pooling"] for row in candidates} == {"mean"}
    assert {row["head_lr"] for row in candidates} == {0.0003, 0.001}
    assert {row["weight_decay"] for row in candidates} == {0.0, 0.001}


def test_age_eligibility_excludes_only_sub5_with_required_reason(tmp_path):
    options = _options(tmp_path)
    _write_input_lance(Path(options.input_lance))
    rows, ages, eligible_samples, counts, total = sweep.read_age_eligibility(options)
    sub5 = next(row for row in rows if row["subject_id"] == "SUB5")
    assert sub5 == {
        "subject_id": "SUB5", "age": "", "eligible_for_age": "false",
        "exclusion_reason": "missing_age", "window_count": 2,
    }
    assert set(ages) == sweep.ELIGIBLE_SUBJECTS
    assert len(eligible_samples) == 14 and total == 16
    assert set(counts.values()) == {2}


def test_dry_run_builds_28_search_commands_without_data_cache_or_training(tmp_path):
    options = _options(tmp_path, dry_run=True)
    called = {"cache": False}

    def forbidden_cache(_):
        called["cache"] = True
        raise AssertionError("dry-run must not touch cache")

    result = sweep.run_sweep(options, cache_ensurer=forbidden_cache)
    assert result == {"dry_run": True, "candidate_count": 4, "fold_count": 7, "command_count": 28}
    assert called["cache"] is False
    commands = json.loads((Path(options.output_dir) / "dry_run_commands.json").read_text())
    assert len(commands) == 28
    assert all("age_regression" in command and age_cv7.PROTOCOL in command for command in commands)
    assert all("flatten" not in command for command in commands)
    assert all("sex_classification" not in command for command in commands)


def test_fixture_sweep_searches_validation_only_tests_once_and_aggregates_oof(tmp_path):
    options = _options(tmp_path)
    source = _write_input_lance(Path(options.input_lance))
    cache_info = _cache_info(tmp_path, len(source))
    runner, calls = _fake_runner()
    summary = sweep.run_sweep(
        options, runner=runner, cache_ensurer=lambda _: cache_info
    )
    search_calls = [call for call in calls if _value(call, "--stage") == "search"]
    evaluate_calls = [call for call in calls if _value(call, "--stage") == "evaluate"]
    assert len(search_calls) == 28
    assert len(evaluate_calls) == 7
    assert all(_value(call, "--stage") == "search" for call in calls[:28])
    assert all(_value(call, "--stage") == "evaluate" for call in calls[28:])
    assert all("--downstream-checkpoint" not in call for call in search_calls)
    assert all("--downstream-checkpoint" in call for call in evaluate_calls)
    assert summary["test_evaluation_calls"] == {str(fold): 1 for fold in range(7)}
    assert summary["checkpoint_kind"] == "cached_feature_age_regression_head_only"
    assert summary["candidate_training_backbone_forward_count"] == 0
    assert summary["eligible_window_count"] == 14
    assert summary["subject_mean_metrics"]["sample_count"] == 7
    assert summary["window_metrics"]["sample_count"] == 14
    assert summary["age_target_normalization"] == "none"
    assert summary["age_prediction_clipping"] == "none"
    with (Path(options.output_dir) / "oof_window_predictions.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        oof = list(csv.DictReader(handle))
    assert len(oof) == 14
    assert len({row["sample_id"] for row in oof}) == 14
    assert all(row["subject_id"] != "SUB5" for row in oof)
    with (Path(options.output_dir) / "oof_subject_predictions.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        subjects = list(csv.DictReader(handle))
    assert len(subjects) == 7
    baseline = json.loads((Path(options.output_dir) / "oof_baseline_metrics.json").read_text())
    assert baseline["predictor"] == "fold_train_subject_mean_age"
    assert (Path(options.output_dir) / "age_cv7_report.md").is_file()
    assert (Path(options.output_dir) / "processing_provenance.json").is_file()


def test_cloud_script_is_age_only_mean_cached_cv7_and_not_legacy_or_sex():
    script = (PROJECT / "commands" / "run_50m_age_cv7_sweep_full.sh").read_text()
    assert script.count("scripts/downstream_50m/run_age_cv7_sweep.py") == 1
    assert "50m_age_cv7_sweep_${STAMP}" in script
    assert "EPOCHS=\"${EPOCHS:-100}\"" in script
    for forbidden in (
        "legacy_single_split", "run_sex_cv4", "sex_classification",
        "NeuralBench", "zero_shot", "probe_transfer", "flatten", "finetune",
    ):
        assert forbidden not in script
    for required in (
        "50m_age_cv7_sweep.json", "pretrain_checkpoint_4.pt", "--feature-cache-root",
        "--feature-cache-policy", "--feature-cache-batch-size", "--epochs",
        "--split-seed", "--train-log-every-epochs",
    ):
        assert required in script


def test_actual_cached_feature_age_stage_search_and_one_time_evaluate(tmp_path, monkeypatch):
    label_lance = tmp_path / "labels.lance"
    source_rows = _write_input_lance(label_lance)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    features = np.stack([
        np.full(512, index / 100.0, dtype=np.float32)
        for index in range(len(source_rows))
    ])
    table = feature_cache.feature_rows_to_table(
        [row["sample_id"] for row in source_rows],
        [row["subject_id"] for row in source_rows],
        ["fixture"] * len(source_rows), features, [640] * len(source_rows),
    )
    lance.write_dataset(
        table, str(cache_dir / feature_cache.MEAN_CACHE_NAME),
        mode="create", data_storage_version="stable",
    )
    identity = {"fixture": "age-stage"}
    cache_key = feature_cache.sha256_json(identity)
    feature_cache.atomic_write_json(cache_dir / feature_cache.MANIFEST_NAME, {
        "status": "completed", "cache_key": cache_key, "cache_identity": identity,
        "parity_passed": True, "requested_poolings": ["mean"],
        "input_row_count": len(source_rows), "mean_feature_dim": 512,
        "mean_row_count": len(source_rows), "dtype": "float32",
        "pretrained_checkpoint": "fixture.pt",
        "pretrained_checkpoint_sha256": "p" * 64,
        "checkpoint_matched_key_count": 1,
        "checkpoint_matched_parameter_fraction": 1.0,
    })
    (cache_dir / feature_cache.SUCCESS_NAME).write_text(cache_key + "\n")

    def config(stage, run_dir):
        cfg = Config()
        cfg.data.lance_path = str(label_lance)
        cfg.finetune.task = "age_regression"
        cfg.finetune.num_classes = 1
        cfg.finetune.protocol = age_cv7.PROTOCOL
        cfg.finetune.folds = 7
        cfg.finetune.fold_index = 0
        cfg.finetune.stage = stage
        cfg.finetune.feature_source = "cache"
        cfg.finetune.feature_cache_path = str(cache_dir)
        cfg.finetune.feature_cache_key = cache_key
        cfg.finetune.pooling = "mean"
        cfg.finetune.device = "cpu"
        cfg.finetune.num_epochs = 2
        cfg.finetune.batch_size = 4
        cfg.finetune.use_prefetch_loader = False
        cfg.finetune.sweep_candidate_id = "fixture-candidate"
        cfg.finetune.sweep_config_sha256 = "s" * 64
        cfg.finetune.sweep_args_json = json.dumps({
            "--pooling": "mean", "--head-lr": 0.001, "--weight-decay": 0.0,
        }, sort_keys=True, separators=(",", ":"))
        cfg.finetune.run_dir = str(run_dir)
        cfg.finetune.checkpoint_dir = str(run_dir / "checkpoints")
        cfg.finetune.metrics_dir = str(run_dir / "metrics")
        cfg.finetune.save_path = str(run_dir / "checkpoints" / "downstream_finetune.pt")
        return cfg

    search_dir = tmp_path / "search"
    (search_dir / "checkpoints").mkdir(parents=True)
    (search_dir / "metrics").mkdir()
    setup_run_logging(search_dir, "ERROR", append=False)
    search_cfg = config("search", search_dir)
    accessed_subjects = []
    original_getitems = feature_cache.CachedFeatureDataset.__getitems__

    def tracking_getitems(self, indices):
        accessed_subjects.extend(self.metadata[int(index)]["subject_id"] for index in indices)
        return original_getitems(self, indices)

    monkeypatch.setattr(feature_cache.CachedFeatureDataset, "__getitems__", tracking_getitems)
    result, _ = training.run_age_cv7_stage(
        search_cfg, training.compute_shape_params(search_cfg), torch.device("cpu"), search_dir
    )
    fold_zero = age_cv7.build_age_subject_cv7(
        {key: value for key, value in AGES.items() if value is not None}, 42
    )["folds"][0]
    assert fold_zero["test_subject"] not in accessed_subjects
    checkpoint = Path(search_cfg.finetune.save_path)
    assert checkpoint.is_file()
    assert result["stage_summary"]["best_validation_r2"] is None
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["checkpoint_kind"] == "cached_feature_age_regression_head_only"

    evaluate_dir = tmp_path / "evaluate"
    (evaluate_dir / "metrics").mkdir(parents=True)
    setup_run_logging(evaluate_dir, "ERROR", append=False)
    evaluate_cfg = config("evaluate", evaluate_dir)
    evaluate_cfg.finetune.evaluate_only = True
    evaluate_cfg.finetune.downstream_checkpoint = str(checkpoint)
    accessed_subjects.clear()
    evaluated, _ = training.run_age_cv7_stage(
        evaluate_cfg, training.compute_shape_params(evaluate_cfg),
        torch.device("cpu"), evaluate_dir,
    )
    assert evaluated["stage_summary"]["evaluation_call_count"] == 1
    assert evaluated["stage_summary"]["r2"] is None
    assert evaluated["stage_summary"]["r2_defined"] is False
    assert set(accessed_subjects) == {fold_zero["test_subject"]}
    with (evaluate_dir / "test_predictions.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert rows and {row["subject_id"] for row in rows} == {
        fold_zero["test_subject"]
    }
