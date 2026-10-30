from __future__ import annotations

import csv
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest


PROJECT = Path(__file__).resolve().parents[1]
MODULE_DIR = PROJECT / "scripts" / "downstream_50m"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import largeScale_fineTune as training
import run_sex_cv4_sweep as sweep
from configs import Config


CONFIG_PATH = PROJECT / "configs" / "50m_sex_cv4_sweep.json"


def _config():
    return sweep.load_sweep_config(CONFIG_PATH)


def _options(tmp_path, *extra):
    return sweep.parse_args([
        "--input-lance", "fixture.lance",
        "--checkpoint", str(tmp_path / "pretrained.pt"),
        "--sweep-config", str(CONFIG_PATH),
        "--output-dir", str(tmp_path / "sweep"),
        "--device", "cpu",
        *extra,
    ])


def _fake_cache_info(tmp_path):
    cache_path = tmp_path / "fake_feature_cache" / ("c" * 64)
    for name in ("mean_features.lance", "flatten_features.lance"):
        path = cache_path / name
        path.mkdir(parents=True, exist_ok=True)
        (path / "data.bin").write_bytes(b"cache")
    return {
        "feature_cache_key": "c" * 64,
        "feature_cache_path": str(cache_path),
        "feature_cache_policy": "auto",
        "feature_cache_reused": True,
        "cache_build_seconds": 0.0,
        "manifest": {
            "mean_row_count": 8, "dtype": "float32", "parity_passed": True,
            "backbone_forward_count": 1, "backbone_processed_window_count": 8,
            "source_window_count": 8,
        },
    }


def test_current_config_expands_eight_unique_candidates_four_per_space():
    candidates = sweep.expand_candidates(_config())
    assert len(candidates) == 8
    assert len({row["candidate_id"] for row in candidates}) == 8
    assert len({row["config_sha256"] for row in candidates}) == 8
    assert [row["search_space"] for row in candidates].count("mean") == 4
    assert [row["search_space"] for row in candidates].count("flatten") == 4


def test_candidate_id_is_stable_when_argument_key_order_changes():
    payload = _config()
    reordered = deepcopy(payload)
    for space in reordered["search_spaces"]:
        space["grid"] = dict(reversed(list(space["grid"].items())))
        space["fixed_args"] = dict(reversed(list(space["fixed_args"].items())))
    left = {row["config_sha256"]: row["candidate_id"] for row in sweep.expand_candidates(payload)}
    right = {row["config_sha256"]: row["candidate_id"] for row in sweep.expand_candidates(reordered)}
    assert left == right


def test_fictional_future_cli_parameter_expands_without_searcher_changes():
    payload = _config()
    payload["search_spaces"] = [deepcopy(payload["search_spaces"][0])]
    payload["search_spaces"][0]["grid"]["--head-dropout"] = [0.0, 0.2]
    candidates = sweep.expand_candidates(payload)
    assert len(candidates) == 8
    assert {row["head_dropout"] for row in candidates} == {0.0, 0.2}
    assert all("--head-dropout" in row["normalized_cli_args"] for row in candidates)


def test_generic_cli_value_conversion_string_numeric_bool_and_null():
    values = {
        "--a-bool": True, "--b-false": False, "--c-null": None,
        "--d-string": "x", "--e-int": 3, "--f-float": 0.5,
    }
    assert sweep.cli_args_to_list(values) == [
        "--a-bool", "--d-string", "x", "--e-int", "3", "--f-float", "0.5",
    ]


@pytest.mark.parametrize("bad", [[1], {"nested": 1}, object()])
def test_illegal_cli_value_types_fail_before_training(bad):
    with pytest.raises(TypeError):
        sweep.cli_args_to_list({"--future": bad})


@pytest.mark.parametrize("flag", sorted(sweep.PROTECTED_ARGS))
def test_protected_arguments_are_rejected(flag):
    payload = _config()
    payload["search_spaces"] = [{"name": "bad", "fixed_args": {flag: "x"}, "grid": {}}]
    with pytest.raises(ValueError, match="protected"):
        sweep.expand_candidates(payload)


def test_schema_version_and_duplicate_candidates_are_rejected(tmp_path):
    payload = _config()
    payload["schema_version"] = 2
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        sweep.load_sweep_config(path)

    duplicate = _config()
    duplicate["search_spaces"].append(deepcopy(duplicate["search_spaces"][0]))
    with pytest.raises(ValueError, match="Duplicate"):
        sweep.expand_candidates(duplicate)


def test_selection_uses_validation_only_and_deterministic_tie_breakers():
    rows = [
        {"candidate_id": "b", "best_validation_balanced_accuracy": 0.8,
         "best_validation_loss": 0.4, "best_validation_macro_f1": 0.7},
        {"candidate_id": "a", "best_validation_balanced_accuracy": 0.8,
         "best_validation_loss": 0.4, "best_validation_macro_f1": 0.7},
        {"candidate_id": "z", "best_validation_balanced_accuracy": 0.7,
         "best_validation_loss": 0.1, "best_validation_macro_f1": 1.0,
         "test_accuracy": 1.0},
    ]
    assert sweep.select_candidate(rows, _config()["selection"])["candidate_id"] == "a"
    bad = deepcopy(_config()["selection"])
    bad["primary_metric"] = "test_accuracy"
    with pytest.raises(ValueError, match="validation-only"):
        sweep.validate_selection(bad)


def test_stage_command_has_same_fold_seed_inputs_and_generic_candidate_args(tmp_path):
    options = _options(tmp_path)
    candidate = sweep.expand_candidates(_config())[0]
    first = sweep.build_stage_command(options, candidate, 2, "search", tmp_path / "one")
    second = sweep.build_stage_command(options, candidate, 2, "search", tmp_path / "two")
    assert first[first.index("--seed") + 1] == second[second.index("--seed") + 1] == "42"
    assert first[first.index("--fold-index") + 1] == "2"
    for flag, value in candidate["normalized_cli_args"].items():
        assert flag in first
        assert first[first.index(flag) + 1] == str(value)


class _MetadataDataset:
    def __init__(self):
        self.rows = []
        for sex, label in (("F", 0), ("M", 1)):
            for number in range(4):
                self.rows.append({
                    "index": len(self.rows), "sample_id": f"{sex}{number}",
                    "subject_id": f"{sex}{number}", "label": label,
                })

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]

    def __getitems__(self, indices):
        return [self.rows[index] for index in indices]

    def get_subject_label_rows(self):
        return list(self.rows)


def test_search_stage_builds_only_train_validation_and_never_test(monkeypatch, tmp_path):
    dataset = _MetadataDataset()
    monkeypatch.setattr(training, "load_downstream_dataset", lambda cfg: dataset)
    loader_splits = []

    def fake_loader(subset, cfg, device, shuffle):
        loader_splits.append({dataset.rows[index]["subject_id"] for index in subset.indices})
        return subset

    monkeypatch.setattr(training, "_loader", fake_loader)

    def fake_run(cfg, shape, train, validation, test, device, **kwargs):
        assert test is None
        assert kwargs["search_only"] is True
        checkpoint = Path(cfg.finetune.save_path)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"checkpoint")
        metrics = {"loss": 0.2, "accuracy": 0.7, "balanced_accuracy": 0.75,
                   "macro_f1": 0.72}
        return object(), {
            "best_epoch": 2, "best_validation": metrics, "epochs": [],
            "training_metadata": {"pooling": "mean"},
        }, 0, {}, []

    monkeypatch.setattr(training, "run_one_checkpoint", fake_run)
    cfg = Config()
    cfg.finetune.task = "sex_classification"
    cfg.finetune.stage = "search"
    cfg.finetune.fold_index = 0
    cfg.finetune.split_seed = 42
    cfg.finetune.seed = 42
    cfg.finetune.sweep_candidate_id = "candidate"
    cfg.finetune.sweep_config_sha256 = "a" * 64
    cfg.finetune.sweep_args_json = '{"--pooling":"mean"}'
    cfg.finetune.save_path = str(tmp_path / "checkpoints" / "best.pt")
    result, _ = training.run_sex_cv4_stage(
        cfg, training.compute_shape_params(cfg), "cpu", tmp_path,
    )
    assert len(loader_splits) == 2
    assert set(result["sample_count"]) == {"train", "validation"}
    summary = json.loads((tmp_path / "validation_summary.json").read_text())
    assert summary["test_accessed"] is False
    assert not any(key.startswith("test_") and key != "test_accessed" for key in summary)


def test_evaluate_stage_builds_only_test_and_does_not_train(monkeypatch, tmp_path):
    dataset = _MetadataDataset()
    monkeypatch.setattr(training, "load_downstream_dataset", lambda cfg: dataset)
    loaded_subjects = []

    def fake_loader(subset, cfg, device, shuffle):
        assert shuffle is False
        loaded_subjects.extend(dataset.rows[index]["subject_id"] for index in subset.indices)
        return subset

    monkeypatch.setattr(training, "_loader", fake_loader)

    def fake_run(cfg, shape, train, validation, test, device, **kwargs):
        assert train is None and validation is None
        assert kwargs["test_prediction_context"] == {"fold": 1}
        rows = []
        for index in test.indices:
            source = dataset.rows[index]
            label = source["label"]
            rows.append({
                "fold": 1, "sample_id": source["sample_id"],
                "subject_id": source["subject_id"], "sex": "female" if label == 0 else "male",
                "true_label": label, "prob_female": 0.9 if label == 0 else 0.1,
                "prob_male": 0.1 if label == 0 else 0.9, "predicted_label": label,
                "predicted_sex": "female" if label == 0 else "male", "num_valid_tokens": 630,
                "checkpoint_path": cfg.finetune.downstream_checkpoint,
            })
        return object(), {"test": {
            "loss": 0.1, "accuracy": 1.0, "balanced_accuracy": 1.0,
            "macro_f1": 1.0, "confusion_matrix": [[1, 0], [0, 1]],
        }}, 0, {}, rows

    monkeypatch.setattr(training, "run_one_checkpoint", fake_run)
    checkpoint = tmp_path / "selected.pt"
    checkpoint.write_bytes(b"unchanged")
    before = sweep.sha256_file(checkpoint)
    cfg = Config()
    cfg.finetune.task = "sex_classification"
    cfg.finetune.stage = "evaluate"
    cfg.finetune.evaluate_only = True
    cfg.finetune.fold_index = 1
    cfg.finetune.split_seed = 42
    cfg.finetune.seed = 42
    cfg.finetune.sweep_candidate_id = "candidate"
    cfg.finetune.sweep_config_sha256 = "a" * 64
    cfg.finetune.sweep_args_json = '{"--pooling":"mean"}'
    cfg.finetune.downstream_checkpoint = str(checkpoint)
    result, _ = training.run_sex_cv4_stage(
        cfg, training.compute_shape_params(cfg), "cpu", tmp_path,
    )
    assert len(loaded_subjects) == 2
    assert set(result["sample_count"]) == {"test"}
    assert sweep.sha256_file(checkpoint) == before
    assert json.loads((tmp_path / "test_metrics.json").read_text())["evaluation_call_count"] == 1


def _fake_runner_factory(fail_search_once=False):
    calls = []
    failed = {"done": False}

    def runner(command, check, cwd, capture_output, text):
        assert check and capture_output and text
        calls.append(list(command))
        value = lambda flag: command[command.index(flag) + 1]
        stage = value("--stage")
        fold = int(value("--fold-index"))
        candidate_id = value("--sweep-candidate-id")
        config_hash = value("--sweep-config-sha256")
        args_json = value("--sweep-args-json")
        output = Path(value("--output-dir"))
        assert not output.exists(), "orchestrator must not pre-create trainer output dirs"
        if fail_search_once and stage == "search" and not failed["done"]:
            failed["done"] = True
            raise subprocess.CalledProcessError(2, command, "", "mock failure")
        output.mkdir(parents=True, exist_ok=True)
        if stage == "search":
            checkpoint = output / "checkpoints" / "downstream_finetune.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(f"{fold}-{candidate_id}".encode())
            # All ties exercise candidate_id as the final deterministic tie-breaker.
            summary = {
                "status": "completed", "stage": "search", "fold": fold,
                "fold_seed": 42 + fold, "candidate_id": candidate_id,
                "candidate_config_sha256": config_hash,
                "normalized_cli_args": json.loads(args_json), "best_epoch": 1,
                "best_validation_loss": 0.5,
                "best_validation_accuracy": 0.5,
                "best_validation_balanced_accuracy": 0.5,
                "best_validation_macro_f1": 0.5,
                "checkpoint_path": str(checkpoint.resolve()),
                "checkpoint_sha256": sweep.sha256_file(checkpoint),
                "feature_source": "cache", "feature_cache_key": "c" * 64,
                "feature_cache_path": str(output.parent),
                "feature_cache_pooling": json.loads(args_json)["--pooling"],
                "feature_cache_reused": True, "feature_dim": 512,
                "backbone_forward_count": 0,
                "checkpoint_kind": "cached_feature_head_only",
                "test_accessed": False,
            }
            sweep.atomic_write_json(output / "validation_summary.json", summary)
        else:
            checkpoint = Path(value("--downstream-checkpoint"))
            subjects = [(f"F{fold}", 0), (f"M{fold}", 1)]
            rows = []
            for subject, label in subjects:
                rows.append({
                    "fold": fold, "split": "test", "sample_id": f"{subject}_W0",
                    "subject_id": subject, "sex": "female" if label == 0 else "male",
                    "true_label": label, "prob_female": 0.9 if label == 0 else 0.1,
                    "prob_male": 0.1 if label == 0 else 0.9,
                    "predicted_label": label, "predicted_sex": "female" if label == 0 else "male",
                    "num_valid_tokens": 630, "checkpoint_path": str(checkpoint),
                })
            sweep._write_csv(output / "test_predictions.csv", rows)
            sweep.atomic_write_json(output / "test_metrics.json", {
                "status": "evaluated", "fold": fold, "candidate_id": candidate_id,
                "candidate_config_sha256": config_hash,
                "test_prediction_count": 2, "evaluation_call_count": 1,
                "feature_source": "cache", "feature_cache_key": "c" * 64,
                "backbone_forward_count": 0,
                "checkpoint_kind": "cached_feature_head_only",
            })
        return SimpleNamespace(stdout="mock ok\n", stderr="")

    return runner, calls


def test_end_to_end_mock_trainer_search_evaluate_once_resume_and_oof(tmp_path):
    options = _options(tmp_path)
    runner, calls = _fake_runner_factory()
    cache_info = _fake_cache_info(tmp_path)
    summary = sweep.run_sweep(
        options, runner=runner, cache_ensurer=lambda options: cache_info
    )
    assert len([call for call in calls if call[call.index("--stage") + 1] == "search"]) == 32
    assert len([call for call in calls if call[call.index("--stage") + 1] == "evaluate"]) == 4
    assert {
        call[call.index("--feature-cache-key") + 1] for call in calls
    } == {"c" * 64}
    assert {
        call[call.index("--feature-cache-path") + 1] for call in calls
    } == {cache_info["feature_cache_path"]}
    search_calls = [call for call in calls if call[call.index("--stage") + 1] == "search"]
    assert sum(call[call.index("--pooling") + 1] == "mean" for call in search_calls) == 16
    assert sum(call[call.index("--pooling") + 1] == "flatten" for call in search_calls) == 16
    assert summary["test_evaluation_calls"] == {str(fold): 1 for fold in range(4)}
    assert summary["backbone_forward_count_during_candidate_training"] == 0
    assert summary["backbone_forward_count_during_test_evaluation"] == 0
    assert summary["candidate_checkpoint_kind"] == "cached_feature_head_only"
    assert summary["window_metrics"]["accuracy"] == pytest.approx(1.0)
    assert summary["subject_metrics"]["probability_mean"]["accuracy"] == pytest.approx(1.0)
    with (Path(options.output_dir) / "oof_window_predictions.csv").open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 8
    assert len({row["sample_id"] for row in rows}) == 8
    assert all(row["split"] == "test" for row in rows)
    assert all(row["selected_candidate_args_json"] for row in rows)
    run_log = (Path(options.output_dir) / "run.log").read_text(encoding="utf-8")
    assert "feature_source=cache" in run_log
    assert "backbone_forward_count=0" in run_log
    assert "search.ranking" in run_log
    assert "starting one-time test evaluation" in run_log
    assert "oof.complete" in run_log
    assert "subject_probability_mean=" in run_log
    assert "subject_majority_vote=" in run_log

    resume = _options(tmp_path, "--resume")
    no_call_runner = lambda *args, **kwargs: pytest.fail("completed resume should not run commands")
    resumed = sweep.run_sweep(
        resume, runner=no_call_runner, cache_ensurer=lambda options: cache_info
    )
    assert resumed["test_evaluation_calls"] == {str(fold): 1 for fold in range(4)}


def test_candidate_completion_requires_hash_status_and_checkpoint(tmp_path):
    candidate = sweep.expand_candidates(_config())[0]
    directory = tmp_path / "candidate"
    checkpoint = directory / "checkpoints" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"ok")
    sweep.atomic_write_json(directory / "validation_summary.json", {
        "status": "completed", "candidate_config_sha256": candidate["config_sha256"],
        "checkpoint_path": str(checkpoint), "checkpoint_sha256": sweep.sha256_file(checkpoint),
    })
    assert sweep._candidate_complete(directory, candidate, {"status": "completed"})
    checkpoint.unlink()
    assert not sweep._candidate_complete(directory, candidate, {"status": "completed"})
    checkpoint.write_bytes(b"ok")
    changed = dict(candidate, config_sha256="b" * 64)
    assert not sweep._candidate_complete(directory, changed, {"status": "completed"})
    assert not sweep._candidate_complete(directory, candidate, {"status": "failed"})


def test_evaluate_checkpoint_metadata_must_match_selected_candidate():
    cfg = Config()
    cfg.finetune.pooling = "mean"
    cfg.finetune.head_lr = 0.001
    cfg.finetune.weight_decay = 0.0
    cfg.finetune.sweep_candidate_id = "chosen"
    cfg.finetune.sweep_config_sha256 = "a" * 64
    cfg.finetune.sweep_args_json = '{"--pooling":"mean"}'
    payload = {
        "pooling": "mean", "head_lr": 0.001, "weight_decay": 0.0,
        "sweep_candidate_id": "chosen", "sweep_config_sha256": "a" * 64,
        "sweep_args_json": '{"--pooling":"mean"}',
    }
    assert training.validate_sweep_checkpoint_metadata(payload, cfg)
    payload["head_lr"] = 0.1
    with pytest.raises(RuntimeError, match="head_lr mismatch"):
        training.validate_sweep_checkpoint_metadata(payload, cfg)


def test_failed_candidate_stops_by_default_and_continue_on_error_is_explicit(tmp_path):
    options = _options(tmp_path)
    runner, _ = _fake_runner_factory(fail_search_once=True)
    with pytest.raises(subprocess.CalledProcessError):
        sweep.run_sweep(
            options, runner=runner,
            cache_ensurer=lambda options: _fake_cache_info(tmp_path),
        )
    state = json.loads((Path(options.output_dir) / "sweep_state.json").read_text())
    statuses = [entry["status"] for entry in state["folds"]["0"]["candidates"].values()]
    assert statuses.count("failed") == 1

    other = tmp_path / "other"
    options = _options(other, "--continue-on-error")
    runner, _ = _fake_runner_factory(fail_search_once=True)
    summary = sweep.run_sweep(
        options, runner=runner,
        cache_ensurer=lambda options: _fake_cache_info(other),
    )
    assert summary["search_complete"] is False


def test_dry_run_writes_manifest_and_commands_without_starting_training(tmp_path):
    options = _options(tmp_path, "--dry-run")
    result = sweep.run_sweep(
        options, runner=lambda *args, **kwargs: pytest.fail("dry-run launched trainer")
    )
    assert result == {"dry_run": True, "candidate_count": 8, "command_count": 32}
    output = Path(options.output_dir)
    assert (output / "candidate_manifest.csv").is_file()
    assert len(json.loads((output / "dry_run_commands.json").read_text())) == 32
    plan = json.loads((output / "dry_run_plan.json").read_text())
    assert len(plan["evaluation_templates"]) == 4
    assert plan["feature_cache"]["build_performed"] is False
    assert all("--feature-cache-key" in command for command in plan["search_commands"])
    assert not list(output.rglob("*.pt"))


def test_atomic_state_writer_uses_os_replace(monkeypatch, tmp_path):
    calls = []
    original = sweep.os.replace

    def recording_replace(source, target):
        calls.append((Path(source), Path(target)))
        return original(source, target)

    monkeypatch.setattr(sweep.os, "replace", recording_replace)
    target = tmp_path / "state.json"
    sweep.atomic_write_json(target, {"status": "running"})
    assert calls and calls[0][1] == target
    assert json.loads(target.read_text()) == {"status": "running"}


def test_cloud_sweep_script_is_sequential_new_timestamp_and_has_no_legacy_flow():
    script = (PROJECT / "commands" / "run_50m_sex_cv4_sweep_full.sh").read_text()
    assert script.count("scripts/downstream_50m/run_sex_cv4_sweep.py") == 1
    assert "50m_sex_cv4_sweep_${STAMP}" in script
    assert "test ! -e \"$SWEEP_OUTPUT\"" in script
    for marker in (
        "legacy_single_split", "create_downstream_manifest.py", "SEX_SPLIT_MANIFEST",
        "SEX_DOWNSTREAM_CKPT", "SECRET_KEY", "ACCESS_KEY", "parallel", "&",
    ):
        assert marker not in script
    for path in (
        "/opt/ml/input", "huada_model_tokens_8sub.lance", "pretrain_checkpoint_4.pt",
        "50m_sex_cv4_sweep.json", "downstream_outputs", "downstream_feature_cache",
    ):
        assert path in script
    for option in (
        "--feature-cache-root", "--feature-cache-policy", "--feature-cache-batch-size",
        "--feature-cache-build-root",
        "--cache-log-every-batches", "--train-log-every-epochs", "--log-level",
    ):
        assert option in script
    assert "export PYTHONUNBUFFERED=1" in script
    assert '"$PYTHON_BIN" -u scripts/downstream_50m/run_sex_cv4_sweep.py' in script
