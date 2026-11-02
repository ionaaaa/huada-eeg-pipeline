"""Sequential validation-only parameter search for cached-feature age CV7."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import lance

from age_cv7 import (
    AGE_UNIT,
    PREDICTION_CLIPPING,
    PROTOCOL,
    TARGET_NORMALIZATION,
    aggregate_subject_predictions,
    build_age_subject_cv7,
    rank_candidates,
    regression_metrics,
    subject_aggregation_metrics,
    train_mean_baseline,
    validate_oof_coverage,
    write_fold_artifacts,
)
from feature_cache import MEAN_CACHE_NAME, directory_size, ensure_feature_cache
from lance_dataset import load_storage_options
from run_logging import log_event, log_exception, setup_run_logging
from run_sex_cv4_sweep import (
    _run_command,
    _write_csv,
    atomic_write_json,
    canonical_json,
    cli_args_to_list,
    expand_candidates,
    sha256_file,
    sha256_json,
    write_candidate_manifest,
)


DEFAULT_TRAINER = Path(__file__).with_name("largeScale_fineTune.py")
EXPECTED_SUBJECTS = {"SUB1", "SUB2", "SUB3", "SUB4", "SUB5", "SUB6", "SUB7", "SUB8"}
ELIGIBLE_SUBJECTS = EXPECTED_SUBJECTS - {"SUB5"}
ALLOWED_CANDIDATE_ARGS = {"--pooling", "--head-lr", "--weight-decay"}
SELECTION_SPEC = [
    {"metric": "best_validation_mae", "mode": "min"},
    {"metric": "best_validation_rmse", "mode": "min"},
    {"metric": "best_validation_loss", "mode": "min"},
    {"metric": "candidate_id", "mode": "asc"},
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-lance", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sweep-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--storage-options")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--feature-cache-root", type=Path, required=True)
    parser.add_argument("--feature-cache-build-root", type=Path)
    parser.add_argument(
        "--feature-cache-policy", choices=["auto", "require", "rebuild"], default="auto"
    )
    parser.add_argument("--feature-cache-batch-size", type=int, default=8)
    parser.add_argument("--cache-log-every-batches", type=int, default=5)
    parser.add_argument("--train-log-every-epochs", type=int, default=5)
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--trainer-script", type=Path, default=DEFAULT_TRAINER, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def load_age_sweep_config(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("Age sweep requires schema_version=1")
    selection = [
        {
            "metric": payload.get("selection", {}).get("primary_metric"),
            "mode": payload.get("selection", {}).get("primary_mode"),
        },
        *payload.get("selection", {}).get("tie_breakers", []),
    ]
    if selection != SELECTION_SPEC:
        raise ValueError("Age selection must be validation MAE/RMSE/loss then candidate_id")
    candidates = expand_candidates(payload)
    for candidate in candidates:
        args = candidate["normalized_cli_args"]
        if set(args) != ALLOWED_CANDIDATE_ARGS:
            raise ValueError(f"Age v1 candidate has unsupported arguments: {sorted(args)}")
        if args["--pooling"] != "mean":
            raise ValueError("Age v1 sweep is mean-only")
    if len(candidates) != 4:
        raise ValueError(f"Formal age v1 config must expand to four candidates; got {len(candidates)}")
    return payload, candidates


def build_stage_command(options, candidate, fold, stage, output_dir, cache_info, checkpoint=None):
    command = [
        sys.executable, "-u", str(Path(options.trainer_script).resolve()),
        "--input-lance", str(options.input_lance),
        "--task", "age_regression",
        "--protocol", PROTOCOL,
        "--folds", "7",
        "--fold-index", str(fold),
        "--stage", stage,
        "--output-dir", str(Path(output_dir).resolve()),
        "--checkpoint", str(Path(options.checkpoint).resolve()),
        "--device", options.device,
        "--mode", "linear_probe",
        "--epochs", str(options.epochs),
        "--batch-size", str(options.batch_size),
        "--num-workers", str(options.num_workers),
        "--seed", str(options.seed),
        "--split-seed", str(options.split_seed),
        "--feature-cache-path", str(cache_info["feature_cache_path"]),
        "--feature-cache-key", str(cache_info["feature_cache_key"]),
        "--sweep-candidate-id", candidate["candidate_id"],
        "--sweep-config-sha256", candidate["config_sha256"],
        "--sweep-args-json", canonical_json(candidate["normalized_cli_args"]),
        "--train-log-every-epochs", str(options.train_log_every_epochs),
        "--log-level", options.log_level,
    ]
    if cache_info.get("feature_cache_reused"):
        command.append("--feature-cache-reused")
    if options.storage_options:
        command.extend(["--storage-options", str(options.storage_options)])
    if stage == "evaluate":
        if checkpoint is None:
            raise ValueError("Age evaluate command requires a selected checkpoint")
        command.extend(["--downstream-checkpoint", str(Path(checkpoint).resolve())])
    command.extend(cli_args_to_list(candidate["normalized_cli_args"]))
    return command


def read_age_eligibility(options):
    storage = load_storage_options(options.storage_options)
    kwargs = {} if storage is None else {"storage_options": storage}
    dataset = lance.dataset(str(options.input_lance), **kwargs)
    required = {"sample_id", "subject_id", "age"}
    if not required.issubset(dataset.schema.names):
        raise ValueError(f"Input Lance lacks age eligibility fields: {sorted(required - set(dataset.schema.names))}")
    rows = dataset.to_table(columns=sorted(required)).to_pylist()
    per_subject = defaultdict(list)
    sample_ids = set()
    for row in rows:
        sample_id = str(row["sample_id"])
        if sample_id in sample_ids:
            raise ValueError(f"Duplicate sample_id in input Lance: {sample_id}")
        sample_ids.add(sample_id)
        per_subject[str(row["subject_id"])].append(row)
    if set(per_subject) != EXPECTED_SUBJECTS:
        raise ValueError(f"Formal age cohort mismatch: {sorted(per_subject)}")
    eligibility = []
    subject_ages = {}
    eligible_samples = {}
    window_counts = {}
    for subject in sorted(per_subject, key=lambda value: int(value[3:])):
        subject_rows = per_subject[subject]
        ages = {None if row["age"] is None else float(row["age"]) for row in subject_rows}
        if len(ages) != 1:
            raise ValueError(f"Inconsistent age values within {subject}: {ages}")
        age = ages.pop()
        eligible = age is not None
        if subject == "SUB5" and eligible:
            raise ValueError("SUB5 must remain age=null")
        if subject != "SUB5" and not eligible:
            raise ValueError(f"Expected eligible age is missing for {subject}")
        eligibility.append({
            "subject_id": subject,
            "age": "" if age is None else age,
            "eligible_for_age": str(eligible).lower(),
            "exclusion_reason": "" if eligible else "missing_age",
            "window_count": len(subject_rows),
        })
        if eligible:
            subject_ages[subject] = age
            window_counts[subject] = len(subject_rows)
            for row in subject_rows:
                eligible_samples[str(row["sample_id"])] = {
                    "subject_id": subject, "true_age": age,
                }
    if set(subject_ages) != ELIGIBLE_SUBJECTS:
        raise ValueError(f"Age eligible cohort mismatch: {sorted(subject_ages)}")
    return eligibility, subject_ages, eligible_samples, window_counts, len(rows)


def _candidate_row(fold, candidate, summary):
    row = {
        "fold": fold,
        "candidate_id": candidate["candidate_id"],
        "search_space": candidate["search_space"],
        "config_sha256": candidate["config_sha256"],
        "normalized_cli_args": canonical_json(candidate["normalized_cli_args"]),
        "best_epoch": summary["best_epoch"],
        "best_validation_mae": summary["best_validation_mae"],
        "best_validation_rmse": summary["best_validation_rmse"],
        "best_validation_loss": summary["best_validation_loss"],
        "best_validation_r2": None,
        "best_validation_r2_defined": False,
        "checkpoint_path": summary["checkpoint_path"],
        "checkpoint_sha256": summary["checkpoint_sha256"],
        "initial_head_fingerprint": summary["initial_head_fingerprint"],
        "test_accessed_during_selection": False,
    }
    row.update({key: value for key, value in candidate.items() if key not in row})
    return row


def _read_validation(candidate_dir, candidate, fold, cache_key):
    summary = json.loads((Path(candidate_dir) / "validation_summary.json").read_text(encoding="utf-8"))
    required = {
        "fold": fold,
        "candidate_id": candidate["candidate_id"],
        "feature_source": "cache",
        "feature_cache_key": cache_key,
        "backbone_forward_count": 0,
        "checkpoint_kind": "cached_feature_age_regression_head_only",
        "test_accessed_during_selection": False,
    }
    for key, expected in required.items():
        if summary.get(key) != expected:
            raise RuntimeError(f"Age validation summary {key} mismatch")
    if summary.get("best_validation_r2") is not None or summary.get("best_validation_r2_defined") is not False:
        raise RuntimeError("Single-subject validation R² must be null/undefined")
    if sha256_file(summary["checkpoint_path"]) != summary["checkpoint_sha256"]:
        raise RuntimeError("Age candidate checkpoint hash mismatch")
    return summary


def _global_validation_rows(fold_rows, candidates, selected_by_fold):
    grouped = defaultdict(list)
    for row in fold_rows:
        grouped[row["candidate_id"]].append(row)
    result = []
    by_id = {row["candidate_id"]: row for row in candidates}
    for candidate_id, rows in grouped.items():
        candidate = by_id[candidate_id]
        result.append({
            "candidate_id": candidate_id,
            "normalized_cli_args": canonical_json(candidate["normalized_cli_args"]),
            "fold_count": len(rows),
            "mean_validation_mae": statistics.mean(float(row["best_validation_mae"]) for row in rows),
            "mean_validation_rmse": statistics.mean(float(row["best_validation_rmse"]) for row in rows),
            "mean_validation_loss": statistics.mean(float(row["best_validation_loss"]) for row in rows),
            "selected_fold_count": sum(value["candidate_id"] == candidate_id for value in selected_by_fold.values()),
            "descriptive_only": True,
            "used_to_reselect_oof_configuration": False,
        })
    return sorted(result, key=lambda row: (
        row["mean_validation_mae"], row["mean_validation_rmse"],
        row["mean_validation_loss"], row["candidate_id"],
    ))


def _aggregate_outputs(output_dir, fold_payload, eligible_samples, selected_by_fold):
    window_rows = []
    baseline_fold_rows = []
    baseline_window_rows = []
    subject_test_folds = {}
    subject_ages = fold_payload["subject_ages"]
    for fold_info in fold_payload["folds"]:
        fold = int(fold_info["fold"])
        selected = selected_by_fold[fold]
        path = Path(output_dir) / f"fold_{fold}" / "test" / "test_predictions.csv"
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        baseline_prediction = train_mean_baseline(fold_info, subject_ages)
        validation_baseline = regression_metrics(
            [fold_info["validation_age"]], [baseline_prediction]
        )
        test_baseline = regression_metrics([fold_info["test_age"]], [baseline_prediction])
        baseline_fold_rows.append({
            "fold": fold,
            "train_subjects": ";".join(fold_info["train_subjects"]),
            "validation_subject": fold_info["validation_subject"],
            "test_subject": fold_info["test_subject"],
            "baseline_prediction": baseline_prediction,
            "validation_mae": validation_baseline["mae"],
            "validation_rmse": validation_baseline["rmse"],
            "validation_r2": None,
            "validation_r2_defined": False,
            "test_mae": test_baseline["mae"],
            "test_rmse": test_baseline["rmse"],
            "test_r2": None,
            "test_r2_defined": False,
            "test_is_outside_train_age_range": fold_info["test_is_outside_train_age_range"],
        })
        subject_test_folds[fold_info["test_subject"]] = fold
        for row in rows:
            converted = dict(row)
            for key in ("fold", "window_index", "num_valid_tokens"):
                converted[key] = int(converted[key])
            for key in ("true_age", "predicted_age", "signed_error", "absolute_error"):
                converted[key] = float(converted[key])
            converted["test_is_outside_train_age_range"] = str(
                converted["test_is_outside_train_age_range"]
            ).lower() == "true"
            converted["candidate_id"] = selected["candidate_id"]
            window_rows.append(converted)
            baseline_window_rows.append({
                "fold": fold,
                "sample_id": converted["sample_id"],
                "subject_id": converted["subject_id"],
                "true_age": converted["true_age"],
                "predicted_age": baseline_prediction,
                "test_is_outside_train_age_range": fold_info[
                    "test_is_outside_train_age_range"
                ],
            })
    validate_oof_coverage(window_rows, eligible_samples, subject_test_folds)
    window_rows.sort(key=lambda row: (row["fold"], row["sample_id"]))
    _write_csv(Path(output_dir) / "oof_window_predictions.csv", window_rows)
    subject_rows = aggregate_subject_predictions(window_rows)
    if len(subject_rows) != 7:
        raise RuntimeError("Age OOF aggregation must contain exactly seven subjects")
    _write_csv(Path(output_dir) / "oof_subject_predictions.csv", subject_rows)
    _write_csv(Path(output_dir) / "baseline_fold_predictions.csv", baseline_fold_rows)
    window_metrics = regression_metrics(
        [row["true_age"] for row in window_rows],
        [row["predicted_age"] for row in window_rows],
    )
    window_metrics.update({
        "window_count": len(window_rows), "subject_count": 7,
        "windows_are_independent_subjects": False,
    })
    subject_metrics = subject_aggregation_metrics(subject_rows)
    baseline_window_metrics = regression_metrics(
        [row["true_age"] for row in baseline_window_rows],
        [row["predicted_age"] for row in baseline_window_rows],
    )
    baseline_subject_rows = aggregate_subject_predictions(baseline_window_rows)
    baseline_subject_metrics = subject_aggregation_metrics(baseline_subject_rows)
    baseline_metrics = {
        "predictor": "fold_train_subject_mean_age",
        "window": baseline_window_metrics,
        "subject_mean": baseline_subject_metrics["mean"],
        "subject_median": baseline_subject_metrics["median"],
        "model_absolute_improvement": {
            "window_mae": baseline_window_metrics["mae"] - window_metrics["mae"],
            "window_rmse": baseline_window_metrics["rmse"] - window_metrics["rmse"],
            "subject_mean_mae": baseline_subject_metrics["mean"]["mae"] - subject_metrics["mean"]["mae"],
            "subject_mean_rmse": baseline_subject_metrics["mean"]["rmse"] - subject_metrics["mean"]["rmse"],
            "subject_mean_r2": subject_metrics["mean"]["r2"] - baseline_subject_metrics["mean"]["r2"],
        },
    }
    atomic_write_json(Path(output_dir) / "oof_window_metrics.json", window_metrics)
    atomic_write_json(Path(output_dir) / "oof_subject_mean_metrics.json", subject_metrics["mean"])
    atomic_write_json(Path(output_dir) / "oof_subject_median_metrics.json", subject_metrics["median"])
    atomic_write_json(Path(output_dir) / "oof_baseline_metrics.json", baseline_metrics)
    return window_metrics, subject_metrics, baseline_metrics, subject_rows


def run_sweep(options, runner=subprocess.run, cache_ensurer=ensure_feature_cache):
    started = time.monotonic()
    config, candidates = load_age_sweep_config(options.sweep_config)
    config_hash = sha256_json(config)
    output_dir = Path(options.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to reuse non-empty age output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_run_logging(output_dir, options.log_level, append=False)
    options.logger = logger
    options.requested_poolings = ("mean",)
    log_event(
        logger, "INFO", "run.start", "starting cached-feature age CV7 sweep",
        input_lance=str(options.input_lance), checkpoint=str(options.checkpoint),
        folds=7, candidates_per_fold=len(candidates), epochs=options.epochs,
        batch_size=options.batch_size, train_seed=options.seed,
        split_seed=options.split_seed, target_unit=AGE_UNIT,
        age_target_normalization=TARGET_NORMALIZATION,
        age_prediction_clipping=PREDICTION_CLIPPING,
    )
    resolved = {**config, "sweep_config_sha256": config_hash, "candidate_count": len(candidates), "candidates": candidates}
    atomic_write_json(output_dir / "resolved_sweep_config.json", resolved)
    write_candidate_manifest(output_dir / "candidate_manifest.csv", candidates)

    placeholder_cache = {
        "feature_cache_path": str(Path(options.feature_cache_root).resolve() / "<cache-key>"),
        "feature_cache_key": "<cache-key>", "feature_cache_reused": False,
    }
    if options.dry_run:
        commands = [
            build_stage_command(
                options, candidate, fold, "search",
                output_dir / f"fold_{fold}" / "candidates" / candidate["candidate_id"],
                placeholder_cache,
            )
            for fold in range(7) for candidate in candidates
        ]
        evaluation_templates = [
            {"fold": fold, "stage": "evaluate", "test_evaluation_count": 1,
             "selected_by": "validation_mae_rmse_loss", "output_dir": str(output_dir / f"fold_{fold}" / "test")}
            for fold in range(7)
        ]
        atomic_write_json(output_dir / "dry_run_commands.json", commands)
        atomic_write_json(output_dir / "dry_run_plan.json", {
            "formal_training_performed": False,
            "cache_build_performed": False,
            "requested_poolings": ["mean"],
            "search_command_count": len(commands),
            "evaluation_templates": evaluation_templates,
        })
        log_event(logger, "INFO", "run.dry_run", "dry-run completed without data/cache/training access",
                  search_command_count=len(commands))
        return {"dry_run": True, "candidate_count": 4, "fold_count": 7, "command_count": 28}

    eligibility, subject_ages, eligible_samples, window_counts, source_rows = read_age_eligibility(options)
    _write_csv(output_dir / "age_eligibility.csv", eligibility)
    log_event(logger, "INFO", "age.eligibility", "age eligibility validated",
              eligible_subjects=sorted(subject_ages), excluded_subject="SUB5",
              exclusion_reason="missing_age", eligible_windows=len(eligible_samples),
              source_windows=source_rows)
    fold_payload = build_age_subject_cv7(subject_ages, options.split_seed)
    write_fold_artifacts(output_dir, fold_payload, window_counts)

    cache_info = cache_ensurer(options)
    options.feature_cache_info = cache_info
    cache_manifest = cache_info["manifest"]
    if cache_manifest.get("requested_poolings") not in (["mean"], ["mean", "flatten"]):
        raise RuntimeError("Feature cache does not contain an audited mean representation")
    if int(cache_manifest["mean_row_count"]) != source_rows:
        raise RuntimeError("Feature cache row count must equal all source windows")
    log_event(logger, "INFO", "cache.ready", "label-free cache ready for age CV7",
              cache_rows=cache_manifest["mean_row_count"], eligible_age_rows=len(eligible_samples),
              feature_dim=cache_manifest["mean_feature_dim"], dtype=cache_manifest["dtype"],
              cache_reused=cache_info["feature_cache_reused"], backbone_forward_count=0)

    project_root = Path(__file__).resolve().parents[2]
    fold_rows = []
    selected_by_fold = {}
    test_call_counts = {}
    for fold in range(7):
        completed = []
        fingerprints = set()
        for candidate in candidates:
            candidate_dir = output_dir / f"fold_{fold}" / "candidates" / candidate["candidate_id"]
            runner_log = output_dir / f"fold_{fold}" / "runner_logs" / f"{candidate['candidate_id']}.train.log"
            command = build_stage_command(options, candidate, fold, "search", candidate_dir, cache_info)
            log_event(logger, "INFO", "search.candidate", "starting age candidate",
                      fold=fold, candidate_id=candidate["candidate_id"],
                      **{key[2:].replace('-', '_'): value for key, value in candidate["normalized_cli_args"].items()},
                      test_accessed_during_selection=False)
            _run_command(command, project_root, runner_log, runner, output_dir / "run.log")
            summary = _read_validation(candidate_dir, candidate, fold, cache_info["feature_cache_key"])
            fingerprints.add(summary["initial_head_fingerprint"])
            row = _candidate_row(fold, candidate, summary)
            completed.append(row)
            fold_rows.append(row)
        if len(fingerprints) != 1:
            raise RuntimeError(f"Fold {fold} candidates did not share identical head initialization")
        ranked = rank_candidates(completed)
        for row in ranked:
            log_event(
                logger, "INFO", "search.ranking", "age validation-only candidate ranking",
                fold=fold, rank=row["rank"], candidate_id=row["candidate_id"],
                validation_mae=row["best_validation_mae"],
                validation_rmse=row["best_validation_rmse"],
                validation_loss=row["best_validation_loss"],
            )
        selected = ranked[0]
        selected_candidate = next(row for row in candidates if row["candidate_id"] == selected["candidate_id"])
        selected_payload = {
            "fold": fold, "candidate_id": selected["candidate_id"],
            "normalized_cli_args": selected_candidate["normalized_cli_args"],
            "checkpoint_path": selected["checkpoint_path"],
            "checkpoint_sha256": selected["checkpoint_sha256"],
            "selection_metrics": {key: selected[key] for key in (
                "best_validation_mae", "best_validation_rmse", "best_validation_loss")},
            "selection_reason": "validation MAE min; RMSE min; loss min; candidate_id asc",
            "initial_head_fingerprint": selected["initial_head_fingerprint"],
            "test_accessed_during_selection": False,
        }
        selected_by_fold[fold] = selected_payload
        selected_dir = output_dir / f"fold_{fold}" / "selected"
        selected_dir.mkdir(parents=True, exist_ok=False)
        atomic_write_json(selected_dir / "selection.json", selected_payload)
        _write_csv(output_dir / f"fold_{fold}" / "candidate_ranking.csv", ranked)
        log_event(logger, "INFO", "search.selected", "age fold candidate selected",
                  fold=fold, candidate_id=selected["candidate_id"],
                  selection_reason=selected_payload["selection_reason"])

    _write_csv(output_dir / "fold_search_results.csv", fold_rows)
    _write_csv(output_dir / "fold_selected_configs.csv", [
        {"fold": fold, **payload} for fold, payload in selected_by_fold.items()
    ])
    _write_csv(output_dir / "global_validation_ranking.csv",
               _global_validation_rows(fold_rows, candidates, selected_by_fold))

    # All validation-only searches and fold selections are locked before any
    # test dataset is opened.  Test results therefore cannot affect later folds.
    for fold in range(7):
        selected = selected_by_fold[fold]
        selected_candidate = next(
            row for row in candidates if row["candidate_id"] == selected["candidate_id"]
        )
        test_dir = output_dir / f"fold_{fold}" / "test"
        test_log = output_dir / f"fold_{fold}" / "runner_logs" / f"{selected['candidate_id']}.evaluate.log"
        if sha256_file(selected["checkpoint_path"]) != selected["checkpoint_sha256"]:
            raise RuntimeError("Selected age checkpoint changed before test")
        command = build_stage_command(
            options, selected_candidate, fold, "evaluate", test_dir, cache_info,
            selected["checkpoint_path"],
        )
        test_call_counts[fold] = 1
        _run_command(command, project_root, test_log, runner, output_dir / "run.log")
        test_metrics = json.loads((test_dir / "test_metrics.json").read_text(encoding="utf-8"))
        if (
            test_metrics.get("candidate_id") != selected["candidate_id"]
            or test_metrics.get("evaluation_call_count") != 1
            or test_metrics.get("r2") is not None
            or test_metrics.get("r2_defined") is not False
            or test_metrics.get("checkpoint_kind") != "cached_feature_age_regression_head_only"
        ):
            raise RuntimeError("Age one-time test artifact validation failed")
        atomic_write_json(output_dir / f"fold_{fold}" / "fold_summary.json", {
            "fold": fold, "selected": selected, "test": test_metrics,
            "test_evaluation_count": 1, "test_accessed_during_selection": False,
        })
        log_event(logger, "INFO", "test.complete", "one-time age test completed",
                  fold=fold, candidate_id=selected["candidate_id"],
                  test_mae=test_metrics["mae"], test_rmse=test_metrics["rmse"],
                  test_r2=None, evaluation_call_count=1)

    window_metrics, subject_metrics, baseline_metrics, subject_rows = _aggregate_outputs(
        output_dir, fold_payload, eligible_samples, selected_by_fold
    )
    log_event(logger, "INFO", "oof.complete", "age OOF coverage and subject aggregation completed",
              oof_windows=window_metrics["window_count"], oof_subjects=7,
              subject_mean_metrics=subject_metrics["mean"], baseline=baseline_metrics)

    elapsed = time.monotonic() - started
    summary = {
        "status": "completed", "protocol": PROTOCOL, "fold_count": 7,
        "candidate_count_per_fold": 4, "test_evaluation_calls": {str(k): v for k, v in test_call_counts.items()},
        "eligible_subjects": sorted(subject_ages), "excluded": [{"subject_id": "SUB5", "reason": "missing_age"}],
        "eligible_window_count": len(eligible_samples), "source_window_count": source_rows,
        "window_metrics": window_metrics, "subject_mean_metrics": subject_metrics["mean"],
        "subject_median_metrics": subject_metrics["median"], "baseline_metrics": baseline_metrics,
        "primary_result": "subject_mean_metrics", "windows_are_independent_subjects": False,
        "feature_cache_key": cache_info["feature_cache_key"],
        "feature_cache_path": cache_info["feature_cache_path"],
        "feature_cache_reused": cache_info["feature_cache_reused"],
        "feature_dim": cache_manifest["mean_feature_dim"], "cache_dtype": cache_manifest["dtype"],
        "candidate_training_backbone_forward_count": 0,
        "checkpoint_kind": "cached_feature_age_regression_head_only",
        "age_target_normalization": TARGET_NORMALIZATION,
        "age_prediction_clipping": PREDICTION_CLIPPING, "label_unit": AGE_UNIT,
        "elapsed_seconds": elapsed,
    }
    atomic_write_json(output_dir / "age_cv7_summary.json", summary)
    provenance = {
        **summary,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_lance": str(options.input_lance),
        "pretrained_checkpoint": str(Path(options.checkpoint).resolve()),
        "pretrained_checkpoint_sha256": sha256_file(options.checkpoint),
        "sweep_config": str(Path(options.sweep_config).resolve()),
        "sweep_config_sha256": config_hash,
        "fold_algorithm_version": fold_payload["algorithm_version"],
        "fold_permutation": fold_payload["permutation"],
        "test_accessed_during_selection": False,
        "global_validation_ranking_is_descriptive_only": True,
    }
    atomic_write_json(output_dir / "processing_provenance.json", provenance)
    _write_csv(output_dir / "age_cv7_issues.csv", [], ["severity", "stage", "message"])
    (output_dir / "age_cv7_errors.log").write_text("", encoding="utf-8")
    report = f"""# 50M age regression subject-level CV7

- eligible subjects: 7; SUB5 excluded because `age=null` (`missing_age`)
- folds: 7; each fold uses 5 train / 1 validation / 1 one-time test subject
- candidates per fold: 4, mean pooling only
- target unit: years; normalization: none; prediction clipping: none
- OOF windows: {window_metrics['window_count']}; OOF subjects: 7
- primary subject-mean MAE/RMSE/R2: {subject_metrics['mean']['mae']:.6f} / {subject_metrics['mean']['rmse']:.6f} / {subject_metrics['mean']['r2']:.6f}
- train-subject-mean baseline subject MAE: {baseline_metrics['subject_mean']['mae']:.6f}

Window-level metrics are auxiliary because windows from the same subject are highly correlated and are not independent subjects. Subject-mean metrics are the primary result. Per-fold validation/test R² is undefined and stored as null; only aggregated OOF R² is interpreted.
"""
    (output_dir / "age_cv7_report.md").write_text(report, encoding="utf-8")
    log_event(logger, "INFO", "run.complete", "age CV7 sweep completed",
              output_dir=str(output_dir), elapsed_seconds=round(elapsed, 1))
    return summary


def main(argv=None):
    options = parse_args(argv)
    if min(options.epochs, options.batch_size, options.feature_cache_batch_size,
           options.cache_log_every_batches, options.train_log_every_epochs) <= 0:
        raise ValueError("Epoch, batch, cache and log intervals must be positive")
    if options.num_workers < 0:
        raise ValueError("num-workers must be non-negative")
    try:
        result = run_sweep(options)
    except Exception as exc:
        output_dir = Path(options.output_dir)
        if output_dir.exists():
            with (output_dir / "age_cv7_errors.log").open("a", encoding="utf-8") as handle:
                handle.write(f"{datetime.now(timezone.utc).isoformat()}\t{type(exc).__name__}\t{exc}\n")
        raise
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
