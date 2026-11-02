"""Configuration-driven sequential grid search for 50M sex CV4 downstream runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from feature_cache import (
    FLATTEN_CACHE_NAME,
    MEAN_CACHE_NAME,
    directory_size,
    ensure_feature_cache,
)
from sex_cv4 import (
    aggregate_subject_predictions, binary_classification_metrics,
    build_sex_stratified_subject_cv4,
)
from run_logging import log_event, log_exception, setup_run_logging


SCHEMA_VERSION = 1
PROTOCOL = "sex_stratified_subject_cv4"
PROTECTED_ARGS = {
    "--input-lance", "--checkpoint", "--output-dir", "--protocol", "--task",
    "--fold-index", "--stage", "--subject-split-manifest",
    "--downstream-checkpoint", "--evaluate", "--sweep-candidate-id",
    "--sweep-config-sha256", "--sweep-args-json",
    "--feature-cache-path", "--feature-cache-key", "--feature-cache-reused",
    "--feature-cache-root", "--feature-cache-build-root", "--feature-cache-policy",
}
ALLOWED_SELECTION_METRICS = {
    "best_validation_balanced_accuracy", "best_validation_loss",
    "best_validation_macro_f1", "candidate_id",
}
DEFAULT_TRAINER = Path(__file__).with_name("largeScale_fineTune.py")


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _validate_scalar(value, location):
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{location} must be finite")
        return
    raise TypeError(f"{location} has unsupported CLI value type: {type(value).__name__}")


def cli_args_to_list(arguments):
    result = []
    for flag in sorted(arguments):
        if not isinstance(flag, str) or not flag.startswith("--"):
            raise ValueError(f"CLI argument key must start with --: {flag!r}")
        value = arguments[flag]
        _validate_scalar(value, flag)
        if value is True:
            result.append(flag)
        elif value is False or value is None:
            continue
        else:
            result.extend([flag, str(value)])
    return result


def _field_name(flag):
    return flag[2:].replace("-", "_")


def _tag(value):
    text = str(value).lower().replace("+", "").replace(".", "p")
    text = re.sub(r"[^a-z0-9_-]+", "-", text).strip("-")
    return text or "none"


def _candidate_id(space_name, normalized_args, digest):
    pieces = [_tag(space_name)]
    for flag, value in normalized_args.items():
        pieces.append(f"{_tag(flag[2:])}-{_tag(value)}")
    readable = "_".join(pieces)
    return f"{readable[:120].rstrip('_')}_{digest[:8]}"


def validate_selection(selection):
    if not isinstance(selection, dict):
        raise ValueError("selection must be an object")
    entries = [{
        "metric": selection.get("primary_metric"),
        "mode": selection.get("primary_mode"),
    }, *selection.get("tie_breakers", [])]
    if not entries or not all(isinstance(item, dict) for item in entries):
        raise ValueError("selection entries must be objects")
    expected = [
        {"metric": "best_validation_balanced_accuracy", "mode": "max"},
        {"metric": "best_validation_loss", "mode": "min"},
        {"metric": "best_validation_macro_f1", "mode": "max"},
        {"metric": "candidate_id", "mode": "asc"},
    ]
    for item in entries:
        metric, mode = item.get("metric"), item.get("mode")
        if metric not in ALLOWED_SELECTION_METRICS:
            raise ValueError(f"Selection metric is not validation-only: {metric!r}")
        allowed_modes = {"asc"} if metric == "candidate_id" else {"min", "max"}
        if mode not in allowed_modes:
            raise ValueError(f"Invalid mode {mode!r} for {metric}")
    if entries != expected:
        raise ValueError("Selection must use the fixed validation-only ordering")
    return entries


def load_sweep_config(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema_version: {payload.get('schema_version')!r}")
    validate_selection(payload.get("selection"))
    spaces = payload.get("search_spaces")
    if not isinstance(spaces, list) or not spaces:
        raise ValueError("search_spaces must be a non-empty list")
    return payload


def expand_candidates(payload):
    candidates = []
    seen_configs = set()
    seen_effective_cli = set()
    seen_ids = set()
    for space_index, space in enumerate(payload["search_spaces"]):
        if not isinstance(space, dict) or not isinstance(space.get("name"), str):
            raise ValueError(f"search_spaces[{space_index}] requires a string name")
        fixed = space.get("fixed_args", {})
        grid = space.get("grid", {})
        if not isinstance(fixed, dict) or not isinstance(grid, dict):
            raise ValueError("fixed_args and grid must be objects")
        overlap = set(fixed).intersection(grid)
        if overlap:
            raise ValueError(f"Arguments cannot be both fixed and grid: {sorted(overlap)}")
        for flag in [*fixed, *grid]:
            if not isinstance(flag, str) or not flag.startswith("--"):
                raise ValueError(f"CLI argument key must start with --: {flag!r}")
            if flag in PROTECTED_ARGS:
                raise ValueError(f"Search config cannot override protected argument {flag}")
        for flag, value in fixed.items():
            if isinstance(value, list):
                raise TypeError(f"fixed_args {flag} cannot be a list")
            _validate_scalar(value, f"fixed_args {flag}")
        grid_flags = sorted(grid)
        grid_values = []
        for flag in grid_flags:
            values = grid[flag]
            if not isinstance(values, list) or not values:
                raise ValueError(f"grid {flag} must be a non-empty list")
            for value in values:
                if isinstance(value, (list, dict)):
                    raise TypeError(f"grid {flag} contains an unsupported nested value")
                _validate_scalar(value, f"grid {flag}")
            grid_values.append(values)
        for combination in itertools.product(*grid_values):
            merged = dict(fixed)
            merged.update(dict(zip(grid_flags, combination)))
            normalized = {key: merged[key] for key in sorted(merged)}
            config_hash = sha256_json(normalized)
            if config_hash in seen_configs:
                raise ValueError(f"Duplicate candidate configuration: {normalized}")
            effective_cli = tuple(cli_args_to_list(normalized))
            if effective_cli in seen_effective_cli:
                raise ValueError(f"Duplicate effective candidate CLI: {normalized}")
            candidate_id = _candidate_id(space["name"], normalized, config_hash)
            if candidate_id in seen_ids:
                raise RuntimeError(f"Candidate id collision: {candidate_id}")
            seen_configs.add(config_hash)
            seen_effective_cli.add(effective_cli)
            seen_ids.add(candidate_id)
            metadata = {
                "candidate_id": candidate_id,
                "search_space": space["name"],
                "normalized_cli_args": normalized,
                "config_sha256": config_hash,
            }
            metadata.update({_field_name(flag): value for flag, value in normalized.items()})
            candidates.append(metadata)
    return candidates


def _write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for field in row:
                if field not in fields:
                    fields.append(field)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_candidate_manifest(path, candidates):
    rows = []
    dynamic = sorted({key for candidate in candidates for key in candidate} - {
        "candidate_id", "search_space", "normalized_cli_args", "config_sha256"
    })
    for candidate in candidates:
        row = dict(candidate)
        row["normalized_cli_args"] = canonical_json(candidate["normalized_cli_args"])
        rows.append(row)
    fields = [
        "candidate_id", "search_space", "normalized_cli_args", "config_sha256", *dynamic,
    ]
    _write_csv(path, rows, fields)


def selection_key(row, selection):
    key = []
    for item in validate_selection(selection):
        value = row[item["metric"]]
        if item["metric"] == "candidate_id":
            key.append(str(value))
        else:
            number = float(value)
            key.append(-number if item["mode"] == "max" else number)
    return tuple(key)


def select_candidate(rows, selection):
    if not rows:
        raise ValueError("No completed candidates are available for selection")
    return min(rows, key=lambda row: selection_key(row, selection))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-lance", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sweep-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--storage-options")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mode", choices=["linear_probe", "finetune"], default="linear_probe")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--feature-cache-root", type=Path, default=Path("downstream_feature_cache"))
    parser.add_argument("--feature-cache-build-root", type=Path)
    parser.add_argument(
        "--feature-cache-policy", choices=["auto", "require", "rebuild"], default="auto"
    )
    parser.add_argument("--feature-cache-batch-size", type=int, default=8)
    parser.add_argument("--cache-log-every-batches", type=int, default=5)
    parser.add_argument("--train-log-every-epochs", type=int, default=5)
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--trainer-script", type=Path, default=DEFAULT_TRAINER, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def build_stage_command(
    options, candidate, fold, stage, output_dir, checkpoint=None,
    feature_cache_info=None,
):
    command = [
        sys.executable, "-u", str(Path(options.trainer_script).resolve()),
        "--input-lance", str(options.input_lance),
        "--task", "sex_classification",
        "--protocol", PROTOCOL,
        "--folds", "4",
        "--fold-index", str(fold),
        "--stage", stage,
        "--output-dir", str(Path(output_dir).resolve()),
        "--checkpoint", str(Path(options.checkpoint).resolve()),
        "--device", options.device,
        "--mode", options.mode,
        "--epochs", str(options.epochs),
        "--batch-size", str(options.batch_size),
        "--num-workers", str(options.num_workers),
        "--seed", str(options.seed),
        "--split-seed", str(options.split_seed),
        "--sweep-candidate-id", candidate["candidate_id"],
        "--sweep-config-sha256", candidate["config_sha256"],
        "--sweep-args-json", canonical_json(candidate["normalized_cli_args"]),
        "--train-log-every-epochs", str(options.train_log_every_epochs),
        "--log-level", options.log_level,
    ]
    feature_cache_info = feature_cache_info or getattr(options, "feature_cache_info", None)
    if feature_cache_info:
        command.extend([
            "--feature-cache-path", feature_cache_info["feature_cache_path"],
            "--feature-cache-key", feature_cache_info["feature_cache_key"],
        ])
        if feature_cache_info.get("feature_cache_reused"):
            command.append("--feature-cache-reused")
    if options.storage_options:
        command.extend(["--storage-options", str(options.storage_options)])
    if stage == "evaluate":
        if checkpoint is None:
            raise ValueError("Evaluate command requires a selected checkpoint")
        command.extend(["--downstream-checkpoint", str(Path(checkpoint).resolve())])
    command.extend(cli_args_to_list(candidate["normalized_cli_args"]))
    return command


def _initial_state(config_hash, candidates):
    return {
        "schema_version": 1,
        "sweep_config_sha256": config_hash,
        "search_complete": False,
        "folds": {
            str(fold): {
                "candidates": {
                    candidate["candidate_id"]: {"status": "pending"}
                    for candidate in candidates
                },
                "selected_candidate_id": None,
                "evaluation": {"status": "pending", "call_count": 0},
            }
            for fold in range(4)
        },
    }


def _candidate_complete(candidate_dir, candidate, state_entry):
    summary_path = Path(candidate_dir) / "validation_summary.json"
    if state_entry.get("status") not in {"completed", "selected", "evaluated"}:
        return False
    if not summary_path.is_file():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        checkpoint = Path(summary["checkpoint_path"])
        return bool(
            summary.get("status") == "completed"
            and summary.get("candidate_config_sha256") == candidate["config_sha256"]
            and checkpoint.is_file()
            and sha256_file(checkpoint) == summary.get("checkpoint_sha256")
        )
    except (KeyError, OSError, ValueError, TypeError):
        return False


def _run_command(command, project_root, log_path, runner, root_log_path=None):
    log_path = Path(log_path)
    # The runner owns only its external log directory.  In particular, this
    # must never be the trainer's --output-dir (or a child of it), because the
    # trainer deliberately requires exclusive creation of that directory.
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if runner is subprocess.run:
        with log_path.open("w", encoding="utf-8") as child_log:
            process = subprocess.Popen(
                command, cwd=str(project_root), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                child_log.write(line)
                child_log.flush()
                if root_log_path is not None:
                    with Path(root_log_path).open("a", encoding="utf-8") as root_log:
                        root_log.write(line)
                        root_log.flush()
            return_code = process.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, command)
        return
    try:
        result = runner(
            command, check=True, cwd=str(project_root), capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as exc:
        log_path.write_text(
            (getattr(exc, "stdout", "") or "") + (getattr(exc, "stderr", "") or ""),
            encoding="utf-8",
        )
        raise
    else:
        log_path.write_text(
            (getattr(result, "stdout", "") or "")
            + (getattr(result, "stderr", "") or ""),
            encoding="utf-8",
        )


def _read_validation_summary(candidate_dir, candidate, fold, expected_cache_key=None):
    summary = json.loads(
        (Path(candidate_dir) / "validation_summary.json").read_text(encoding="utf-8")
    )
    if summary.get("fold") != fold or summary.get("candidate_id") != candidate["candidate_id"]:
        raise RuntimeError("Validation summary identity mismatch")
    if summary.get("test_accessed") is not False:
        raise RuntimeError("Candidate search did not attest test_accessed=false")
    if expected_cache_key is not None:
        if summary.get("feature_source") != "cache":
            raise RuntimeError("Candidate did not use cached features")
        if summary.get("feature_cache_key") != expected_cache_key:
            raise RuntimeError("Candidate feature cache key mismatch")
        if summary.get("backbone_forward_count") != 0:
            raise RuntimeError("Candidate unexpectedly executed the backbone")
        if summary.get("checkpoint_kind") != "cached_feature_head_only":
            raise RuntimeError("Candidate did not save a head-only checkpoint")
    forbidden = [key for key in summary if key.startswith("test_") and key != "test_accessed"]
    if forbidden:
        raise RuntimeError(f"Validation summary contains test fields: {forbidden}")
    return summary


def _fold_result_row(fold, candidate, summary):
    row = {
        "fold": fold,
        "candidate_id": candidate["candidate_id"],
        "search_space": candidate["search_space"],
        "config_sha256": candidate["config_sha256"],
        "normalized_cli_args": canonical_json(candidate["normalized_cli_args"]),
        "fold_seed": summary["fold_seed"],
        "best_epoch": summary["best_epoch"],
        "best_validation_loss": summary["best_validation_loss"],
        "best_validation_accuracy": summary["best_validation_accuracy"],
        "best_validation_balanced_accuracy": summary["best_validation_balanced_accuracy"],
        "best_validation_macro_f1": summary["best_validation_macro_f1"],
        "checkpoint_path": summary["checkpoint_path"],
        "checkpoint_sha256": summary["checkpoint_sha256"],
        "feature_source": summary.get("feature_source"),
        "feature_cache_key": summary.get("feature_cache_key"),
        "feature_cache_path": summary.get("feature_cache_path"),
        "feature_cache_pooling": summary.get("feature_cache_pooling"),
        "feature_cache_reused": summary.get("feature_cache_reused"),
        "feature_dim": summary.get("feature_dim"),
        "backbone_forward_count": summary.get("backbone_forward_count"),
        "checkpoint_kind": summary.get("checkpoint_kind"),
    }
    row.update({key: value for key, value in candidate.items() if key not in row})
    return row


def _write_rankings(output_dir, fold_rows, selected_by_fold, candidates):
    _write_csv(Path(output_dir) / "fold_search_results.csv", fold_rows)
    selected_rows = []
    for fold, selected in sorted(selected_by_fold.items()):
        row = {"fold": fold, **selected}
        row["normalized_cli_args"] = canonical_json(selected["normalized_cli_args"])
        selected_rows.append(row)
    _write_csv(Path(output_dir) / "fold_selected_configs.csv", selected_rows)

    grouped = defaultdict(list)
    for row in fold_rows:
        grouped[row["candidate_id"]].append(row)
    candidate_by_id = {value["candidate_id"]: value for value in candidates}
    global_rows = []
    for candidate_id in sorted(grouped):
        rows = grouped[candidate_id]
        balanced = [float(row["best_validation_balanced_accuracy"]) for row in rows]
        candidate = candidate_by_id[candidate_id]
        global_rows.append({
            "candidate_id": candidate_id,
            "search_space": candidate["search_space"],
            "normalized_cli_args": canonical_json(candidate["normalized_cli_args"]),
            "fold_count": len(rows),
            "mean_validation_balanced_accuracy": statistics.mean(balanced),
            "std_validation_balanced_accuracy": statistics.pstdev(balanced),
            "mean_validation_loss": statistics.mean(
                float(row["best_validation_loss"]) for row in rows
            ),
            "mean_validation_macro_f1": statistics.mean(
                float(row["best_validation_macro_f1"]) for row in rows
            ),
            "selected_fold_count": sum(
                selected["candidate_id"] == candidate_id
                for selected in selected_by_fold.values()
            ),
        })
    global_rows.sort(key=lambda row: (
        -row["mean_validation_balanced_accuracy"], row["mean_validation_loss"],
        -row["mean_validation_macro_f1"], row["candidate_id"],
    ))
    _write_csv(Path(output_dir) / "global_validation_ranking.csv", global_rows)


def aggregate_oof(output_dir, selected_by_fold):
    window_rows = []
    subject_folds = defaultdict(set)
    expected_subject_folds = {}
    for fold, selected in sorted(selected_by_fold.items()):
        test_dir = Path(output_dir) / f"fold_{fold}" / "test"
        metrics = json.loads((test_dir / "test_metrics.json").read_text(encoding="utf-8"))
        with (test_dir / "test_predictions.csv").open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if metrics.get("evaluation_call_count") != 1 or len(rows) != int(metrics["test_prediction_count"]):
            raise RuntimeError(f"Fold {fold} test evaluation count/coverage invariant failed")
        for subject in metrics.get("test_subjects", []):
            if subject in expected_subject_folds:
                raise RuntimeError(f"Subject {subject} was assigned to multiple test folds")
            expected_subject_folds[subject] = fold
        args_json = canonical_json(selected["normalized_cli_args"])
        for row in rows:
            if row.get("split") != "test" or int(row["fold"]) != fold:
                raise RuntimeError("Non-test prediction or wrong fold entered OOF")
            female, male = float(row["prob_female"]), float(row["prob_male"])
            if not math.isclose(female + male, 1.0, rel_tol=1e-5, abs_tol=1e-5):
                raise RuntimeError(f"Probability sum invariant failed for {row['sample_id']}")
            row.update({
                "fold": fold,
                "true_label": int(row["true_label"]),
                "predicted_label": int(row["predicted_label"]),
                "prob_female": female,
                "prob_male": male,
                "selected_candidate_id": selected["candidate_id"],
                "selected_candidate_args_json": args_json,
                "pooling": selected["normalized_cli_args"].get("--pooling"),
                "head_lr": selected["normalized_cli_args"].get("--head-lr"),
                "weight_decay": selected["normalized_cli_args"].get("--weight-decay"),
                "checkpoint_path": selected["checkpoint_path"],
            })
            subject_folds[row["subject_id"]].add(fold)
            window_rows.append(row)
    sample_counts = Counter(row["sample_id"] for row in window_rows)
    if not window_rows or any(count != 1 for count in sample_counts.values()):
        raise RuntimeError("Each OOF sample_id must appear exactly once")
    if any(len(folds) != 1 for folds in subject_folds.values()):
        raise RuntimeError("Each OOF subject must belong to exactly one test fold")
    if expected_subject_folds and {
        subject: next(iter(folds)) for subject, folds in subject_folds.items()
    } != expected_subject_folds:
        raise RuntimeError("OOF subjects do not exactly cover the declared test subjects")

    window_rows.sort(key=lambda row: (row["fold"], row["sample_id"]))
    window_fields = [
        "fold", "sample_id", "subject_id", "true_label", "prob_female", "prob_male",
        "predicted_label", "selected_candidate_id", "pooling", "head_lr",
        "weight_decay", "selected_candidate_args_json", "checkpoint_path", "split",
    ]
    _write_csv(Path(output_dir) / "oof_window_predictions.csv", window_rows, window_fields)
    window_metrics = binary_classification_metrics(
        [row["true_label"] for row in window_rows],
        [row["predicted_label"] for row in window_rows],
    )
    window_metrics.update({
        "num_windows": len(window_rows),
        "num_subjects": len(subject_folds),
        "coverage_complete": True,
    })
    atomic_write_json(Path(output_dir) / "oof_window_metrics.json", window_metrics)

    subject_rows = aggregate_subject_predictions(window_rows)
    _write_csv(Path(output_dir) / "oof_subject_predictions.csv", subject_rows)
    probability_metrics = binary_classification_metrics(
        [row["true_label"] for row in subject_rows],
        [row["probability_mean_pred_label"] for row in subject_rows],
    )
    majority_metrics = binary_classification_metrics(
        [row["true_label"] for row in subject_rows],
        [row["majority_vote_pred_label"] for row in subject_rows],
    )
    subject_metrics = {
        "num_subjects": len(subject_rows),
        "probability_mean": probability_metrics,
        "majority_vote": majority_metrics,
    }
    atomic_write_json(Path(output_dir) / "oof_subject_metrics.json", subject_metrics)
    return window_metrics, subject_metrics


def log_preflight_dataset(options, logger):
    """Read only label metadata and log the exact fixed CV4 plan before cache/training."""
    import lance
    from lance_dataset import SEX_LABEL_MAPPING, load_storage_options

    storage_options = load_storage_options(options.storage_options)
    kwargs = {} if storage_options is None else {"storage_options": storage_options}
    dataset = lance.dataset(str(options.input_lance), **kwargs)
    rows = dataset.to_table(columns=["sample_id", "subject_id", "sex"]).to_pylist()
    subject_labels = {}
    window_sex = Counter()
    subject_counts = Counter()
    for row in rows:
        subject = str(row["subject_id"])
        sex = str(row["sex"]).strip().lower()
        if sex not in SEX_LABEL_MAPPING:
            raise ValueError(f"Unsupported sex label during preflight: {sex!r}")
        label = SEX_LABEL_MAPPING[sex]
        if subject in subject_labels and subject_labels[subject] != label:
            raise ValueError(f"Inconsistent sex label for subject {subject}")
        subject_labels[subject] = label
        window_sex[sex] += 1
        subject_counts[subject] += 1
    folds = build_sex_stratified_subject_cv4(subject_labels, options.split_seed)["folds"]
    log_event(
        logger, "INFO", "preflight.dataset", "input cohort and labels validated",
        input_windows=len(rows), subject_count=len(subject_labels),
        window_sex_distribution=dict(window_sex),
        subject_sex_distribution={
            "female": sum(value == 0 for value in subject_labels.values()),
            "male": sum(value == 1 for value in subject_labels.values()),
        },
    )
    for fold_info in folds:
        split_fields = {}
        for split in ("train", "validation", "test"):
            subjects = list(fold_info[f"{split}_subjects"])
            split_fields[f"{split}_subjects"] = subjects
            split_fields[f"{split}_samples"] = sum(subject_counts[s] for s in subjects)
        log_event(logger, "INFO", "preflight.fold", "CV4 fold split validated",
                  fold=int(fold_info["fold"]), **split_fields)


def run_sweep(options, runner=subprocess.run, cache_ensurer=ensure_feature_cache):
    run_started = time.monotonic()
    config = load_sweep_config(options.sweep_config)
    candidates = expand_candidates(config)
    config_hash = sha256_json(config)
    output_dir = Path(options.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not options.resume:
        raise FileExistsError(f"Refusing to reuse non-empty output without --resume: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_run_logging(output_dir, options.log_level, append=options.resume)
    options.logger = logger
    poolings = sorted({c["normalized_cli_args"].get("--pooling") for c in candidates})
    head_lrs = sorted({c["normalized_cli_args"].get("--head-lr") for c in candidates})
    weight_decays = sorted({c["normalized_cli_args"].get("--weight-decay") for c in candidates})
    log_event(
        logger, "INFO", "run.start", "starting 50M cached-feature CV4 sweep",
        run_mode="dry-run" if options.dry_run else "cloud sweep",
        input_lance=str(options.input_lance), checkpoint=str(options.checkpoint),
        sweep_config=str(options.sweep_config), output_dir=str(output_dir),
        feature_cache_root=str(options.feature_cache_root),
        feature_cache_build_root=str(options.feature_cache_build_root),
        feature_cache_policy=options.feature_cache_policy, device=options.device,
        epochs=options.epochs, batch_size=options.batch_size, seed=options.seed,
        split_seed=options.split_seed, cache_batch_size=options.feature_cache_batch_size,
        candidate_count=len(candidates), pooling_values=poolings,
        head_lr_values=head_lrs, weight_decay_values=weight_decays,
    )
    state_path = output_dir / "sweep_state.json"
    if options.resume and state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("sweep_config_sha256") != config_hash:
            raise ValueError("Sweep config changed; refusing to resume incompatible state")
    else:
        state = _initial_state(config_hash, candidates)
    atomic_write_json(output_dir / "resolved_sweep_config.json", {
        **config,
        "sweep_config_sha256": config_hash,
        "candidate_count": len(candidates),
        "candidates": candidates,
    })
    write_candidate_manifest(output_dir / "candidate_manifest.csv", candidates)
    atomic_write_json(state_path, state)

    project_root = Path(__file__).resolve().parents[2]
    if options.dry_run:
        log_event(logger, "INFO", "run.dry_run", "validated configuration; training and cache build disabled",
                  candidate_count=len(candidates))
        cache_plan = {
            "feature_cache_root": str(Path(options.feature_cache_root).resolve()),
            "feature_cache_policy": options.feature_cache_policy,
            "feature_cache_build_root": (
                str(options.feature_cache_build_root)
                if options.feature_cache_build_root else "<system-local-temp>"
            ),
            "action": "validate/reuse or build once before all folds and candidates",
            "cache_key": "<resolved from Lance, checkpoint, backbone, probe and code fingerprints>",
            "build_performed": False,
        }
        print("# feature cache plan: " + canonical_json(cache_plan))
        dry_cache_info = {
            "feature_cache_path": str(Path(options.feature_cache_root).resolve() / "<cache-key>"),
            "feature_cache_key": "<cache-key>",
            "feature_cache_reused": False,
        }
        commands = []
        for fold in range(4):
            for candidate in candidates:
                candidate_dir = output_dir / f"fold_{fold}" / "candidates" / candidate["candidate_id"]
                command = build_stage_command(
                    options, candidate, fold, "search", candidate_dir,
                    feature_cache_info=dry_cache_info,
                )
                commands.append(command)
                print(subprocess.list2cmdline(command))
        evaluation_templates = [
            {
                "fold": fold,
                "stage": "evaluate",
                "candidate": "<selected-by-validation>",
                "checkpoint": "<selected-checkpoint>",
                "output_dir": str(output_dir / f"fold_{fold}" / "test"),
            }
            for fold in range(4)
        ]
        for template in evaluation_templates:
            print(
                f"# fold {template['fold']} evaluate selected validation candidate once: "
                f"{template['output_dir']}"
            )
        atomic_write_json(output_dir / "dry_run_commands.json", commands)
        atomic_write_json(output_dir / "dry_run_plan.json", {
            "feature_cache": cache_plan,
            "search_commands": commands,
            "evaluation_templates": evaluation_templates,
        })
        return {"dry_run": True, "candidate_count": len(candidates), "command_count": len(commands)}

    if runner is subprocess.run:
        for name, path in (
            ("input Lance", options.input_lance), ("checkpoint", options.checkpoint),
            ("sweep config", options.sweep_config),
        ):
            is_remote = "://" in str(path)
            exists = is_remote or Path(path).exists()
            log_event(logger, "INFO" if exists else "ERROR", "preflight.path",
                      f"{name} path check", path=str(path), exists=exists,
                      remote_uri=is_remote)
            if not exists:
                raise FileNotFoundError(path)
    else:
        log_event(logger, "INFO", "preflight.path", "path checks delegated to fixture runner")
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        log_event(logger, "INFO", "preflight.cuda", "CUDA availability checked",
                  requested_device=options.device, cuda_available=cuda_available)
        if str(options.device).startswith("cuda") and not cuda_available:
            raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
        if runner is subprocess.run:
            log_preflight_dataset(options, logger)
        feature_cache_info = cache_ensurer(options)
    except Exception:
        log_exception(logger, "run.failed", "preflight/cache stage failed",
                      cache_path=str(options.feature_cache_root))
        raise
    options.feature_cache_info = feature_cache_info
    cache_manifest = feature_cache_info["manifest"]
    state["feature_cache"] = {
        "key": feature_cache_info["feature_cache_key"],
        "path": feature_cache_info["feature_cache_path"],
        "policy": feature_cache_info["feature_cache_policy"],
        "reused": feature_cache_info["feature_cache_reused"],
    }
    atomic_write_json(state_path, state)

    fold_rows = []
    selected_by_fold = {}
    search_complete = True
    for fold in range(4):
        log_event(logger, "INFO", "search.fold", "starting fold candidate search; test is not used for selection",
                  fold=fold, candidate_count=len(candidates), fold_seed=options.seed + fold)
        fold_state = state["folds"][str(fold)]
        completed_rows = []
        for candidate in candidates:
            candidate_id = candidate["candidate_id"]
            entry = fold_state["candidates"][candidate_id]
            candidate_dir = output_dir / f"fold_{fold}" / "candidates" / candidate_id
            runner_log = (
                output_dir / f"fold_{fold}" / "runner_logs" / f"{candidate_id}.train.log"
            )
            if options.resume and _candidate_complete(candidate_dir, candidate, entry):
                log_event(
                    logger, "INFO", "search.candidate.resume",
                    "reusing completed candidate",
                    fold=fold, candidate_id=candidate_id,
                    candidate_dir=str(candidate_dir), candidate_dir_exists=True,
                    resume_decision="reuse_completed", runner_log_path=str(runner_log),
                )
                summary = _read_validation_summary(
                    candidate_dir, candidate, fold, feature_cache_info["feature_cache_key"]
                )
            else:
                if candidate_dir.exists():
                    log_event(
                        logger, "ERROR", "search.candidate.resume",
                        "candidate output directory exists but is not safely reusable",
                        fold=fold, candidate_id=candidate_id,
                        candidate_dir=str(candidate_dir), candidate_dir_exists=True,
                        resume_decision="refuse_incomplete_no_overwrite",
                        runner_log_path=str(runner_log),
                    )
                    raise FileExistsError(
                        "Refusing to overwrite incomplete or incompatible candidate "
                        f"output directory: {candidate_dir}"
                    )
                entry.update({"status": "running", "started_at": datetime.now(timezone.utc).isoformat()})
                atomic_write_json(state_path, state)
                command = build_stage_command(options, candidate, fold, "search", candidate_dir)
                args = candidate["normalized_cli_args"]
                log_event(
                    logger, "INFO", "search.candidate", "starting candidate",
                    fold=fold, candidate_id=candidate_id, pooling=args.get("--pooling"),
                    head_lr=args.get("--head-lr"), weight_decay=args.get("--weight-decay"),
                    feature_cache_path=feature_cache_info["feature_cache_path"],
                    feature_source="cache", backbone_constructed=False,
                    backbone_forward_count=0, candidate_dir=str(candidate_dir),
                    candidate_dir_exists=False, resume_decision="start_new",
                    runner_log_path=str(runner_log),
                )
                log_event(
                    logger, "INFO", "search.candidate",
                    "candidate output directory ownership delegated to trainer",
                    fold=fold, candidate_id=candidate_id,
                    candidate_dir=str(candidate_dir), candidate_dir_exists=False,
                    runner_log_path=str(runner_log),
                )
                log_event(
                    logger, "INFO", "train.start", "launching candidate trainer process",
                    fold=fold, candidate_id=candidate_id,
                    candidate_dir=str(candidate_dir), candidate_dir_exists=False,
                    runner_log_path=str(runner_log),
                )
                try:
                    _run_command(command, project_root, runner_log, runner,
                                 output_dir / "run.log")
                    summary = _read_validation_summary(
                        candidate_dir, candidate, fold, feature_cache_info["feature_cache_key"]
                    )
                    if not _candidate_complete(candidate_dir, candidate, {"status": "completed"}):
                        raise RuntimeError("Candidate completion artifacts failed validation")
                    entry.update({"status": "completed", "completed_at": datetime.now(timezone.utc).isoformat()})
                except Exception as exc:
                    entry.update({"status": "failed", "error": str(exc)})
                    search_complete = False
                    atomic_write_json(state_path, state)
                    log_exception(logger, "search.candidate.failed", "candidate training failed",
                                  fold=fold, candidate_id=candidate_id,
                                  cache_path=feature_cache_info["feature_cache_path"])
                    if not options.continue_on_error:
                        raise
                    continue
                atomic_write_json(state_path, state)
            row = _fold_result_row(fold, candidate, summary)
            fold_rows.append(row)
            completed_rows.append(row)
        if len(completed_rows) != len(candidates):
            search_complete = False
        selected = select_candidate(completed_rows, config["selection"])
        ranked = sorted(completed_rows, key=lambda row: selection_key(row, config["selection"]))
        for rank, row in enumerate(ranked, start=1):
            log_event(
                logger, "INFO", "search.ranking", "validation-only candidate ranking",
                fold=fold, candidate_id=row["candidate_id"], rank=rank,
                pooling=row.get("pooling"), head_lr=row.get("head_lr"),
                weight_decay=row.get("weight_decay"), best_epoch=row["best_epoch"],
                validation_balanced_accuracy=row["best_validation_balanced_accuracy"],
                validation_loss=row["best_validation_loss"],
                validation_macro_f1=row["best_validation_macro_f1"],
            )
        selected_candidate = next(
            candidate for candidate in candidates
            if candidate["candidate_id"] == selected["candidate_id"]
        )
        selected_payload = {
            "status": "selected",
            "fold": fold,
            "candidate_id": selected["candidate_id"],
            "search_space": selected_candidate["search_space"],
            "normalized_cli_args": selected_candidate["normalized_cli_args"],
            "selection_metrics": {
                key: selected[key] for key in (
                    "best_validation_balanced_accuracy", "best_validation_loss",
                    "best_validation_macro_f1",
                )
            },
            "checkpoint_path": selected["checkpoint_path"],
            "checkpoint_sha256": selected["checkpoint_sha256"],
            "selection_reason": (
                "validation balanced accuracy max; validation loss min; "
                "validation macro F1 max; candidate_id asc"
            ),
            "feature_source": "cache",
            "feature_cache_key": feature_cache_info["feature_cache_key"],
            "feature_cache_path": feature_cache_info["feature_cache_path"],
            "feature_cache_pooling": selected_candidate["normalized_cli_args"].get("--pooling"),
            "feature_cache_reused": feature_cache_info["feature_cache_reused"],
        }
        atomic_write_json(output_dir / f"fold_{fold}" / "selected_candidate.json", selected_payload)
        fold_state["selected_candidate_id"] = selected["candidate_id"]
        selected_by_fold[fold] = selected_payload
        log_event(
            logger, "INFO", "search.selected", "selected fold candidate using deterministic validation tie-breakers",
            fold=fold, candidate_id=selected["candidate_id"],
            pooling=selected_candidate["normalized_cli_args"].get("--pooling"),
            head_lr=selected_candidate["normalized_cli_args"].get("--head-lr"),
            weight_decay=selected_candidate["normalized_cli_args"].get("--weight-decay"),
            selection_reason=selected_payload["selection_reason"],
        )
        atomic_write_json(state_path, state)

    state["search_complete"] = search_complete
    _write_rankings(output_dir, fold_rows, selected_by_fold, candidates)
    atomic_write_json(state_path, state)

    for fold, selected in sorted(selected_by_fold.items()):
        evaluation = state["folds"][str(fold)]["evaluation"]
        test_dir = output_dir / f"fold_{fold}" / "test"
        evaluation_runner_log = (
            output_dir / f"fold_{fold}" / "runner_logs"
            / f"{selected['candidate_id']}.evaluate.log"
        )
        metrics_path = test_dir / "test_metrics.json"
        predictions_path = test_dir / "test_predictions.csv"
        valid_existing = False
        if options.resume and evaluation.get("status") == "evaluated":
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                valid_existing = bool(
                    predictions_path.is_file()
                    and metrics.get("candidate_id") == selected["candidate_id"]
                    and metrics.get("evaluation_call_count") == 1
                    and evaluation.get("call_count") == 1
                )
            except (OSError, ValueError):
                valid_existing = False
        if valid_existing:
            log_event(
                logger, "INFO", "test.resume", "reusing completed one-time test evaluation",
                fold=fold, candidate_id=selected["candidate_id"],
                test_dir=str(test_dir), test_dir_exists=True,
                resume_decision="reuse_evaluated",
                runner_log_path=str(evaluation_runner_log),
            )
            continue
        if int(evaluation.get("call_count", 0)) != 0:
            raise RuntimeError(
                f"Fold {fold} test was already invoked; refusing a second evaluation"
            )
        if test_dir.exists():
            log_event(
                logger, "ERROR", "test.resume",
                "test output directory exists but is not safely reusable",
                fold=fold, candidate_id=selected["candidate_id"],
                test_dir=str(test_dir), test_dir_exists=True,
                resume_decision="refuse_incomplete_no_overwrite",
                runner_log_path=str(evaluation_runner_log),
            )
            raise FileExistsError(
                f"Refusing to overwrite incomplete test output directory: {test_dir}"
            )
        evaluation.update({"status": "running", "call_count": 1})
        atomic_write_json(state_path, state)
        candidate = next(
            value for value in candidates if value["candidate_id"] == selected["candidate_id"]
        )
        if sha256_file(selected["checkpoint_path"]) != selected["checkpoint_sha256"]:
            raise RuntimeError(f"Fold {fold} selected checkpoint hash changed before evaluation")
        command = build_stage_command(
            options, candidate, fold, "evaluate", test_dir, selected["checkpoint_path"]
        )
        log_event(logger, "INFO", "test.start", "starting one-time test evaluation",
                  fold=fold, candidate_id=selected["candidate_id"], evaluation_call_count=1,
                  test_dir=str(test_dir), test_dir_exists=False,
                  resume_decision="start_new", runner_log_path=str(evaluation_runner_log))
        try:
            _run_command(command, project_root, evaluation_runner_log, runner,
                         output_dir / "run.log")
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            if (
                metrics.get("candidate_id") != selected["candidate_id"]
                or metrics.get("candidate_config_sha256") != candidate["config_sha256"]
                or metrics.get("evaluation_call_count") != 1
                or metrics.get("feature_source") != "cache"
                or metrics.get("feature_cache_key") != feature_cache_info["feature_cache_key"]
                or metrics.get("backbone_forward_count") != 0
                or metrics.get("checkpoint_kind") != "cached_feature_head_only"
                or not predictions_path.is_file()
            ):
                raise RuntimeError("Test evaluation artifacts failed identity validation")
            if sha256_file(selected["checkpoint_path"]) != selected["checkpoint_sha256"]:
                raise RuntimeError(f"Fold {fold} evaluation modified the selected checkpoint")
            evaluation.update({"status": "evaluated", "completed_at": datetime.now(timezone.utc).isoformat()})
            with predictions_path.open(encoding="utf-8-sig", newline="") as handle:
                fold_prediction_rows = list(csv.DictReader(handle))
            fold_subject_rows = aggregate_subject_predictions(fold_prediction_rows)
            fold_subject_probability = binary_classification_metrics(
                [int(row["true_label"]) for row in fold_subject_rows],
                [int(row["probability_mean_pred_label"]) for row in fold_subject_rows],
            )
            fold_subject_majority = binary_classification_metrics(
                [int(row["true_label"]) for row in fold_subject_rows],
                [int(row["majority_vote_pred_label"]) for row in fold_subject_rows],
            )
            log_event(logger, "INFO", "test.complete", "one-time test evaluation completed",
                      fold=fold, candidate_id=selected["candidate_id"],
                      window_accuracy=metrics.get("accuracy"),
                      window_balanced_accuracy=metrics.get("balanced_accuracy"),
                      window_macro_f1=metrics.get("macro_f1"),
                      subject_probability_mean=fold_subject_probability,
                      subject_majority_vote=fold_subject_majority)
        except Exception as exc:
            evaluation.update({"status": "failed", "error": str(exc)})
            atomic_write_json(state_path, state)
            log_exception(logger, "test.failed", "test evaluation failed",
                          fold=fold, candidate_id=selected["candidate_id"],
                          cache_path=feature_cache_info["feature_cache_path"])
            raise
        atomic_write_json(state_path, state)

    if any(
        state["folds"][str(fold)]["evaluation"].get("call_count") != 1
        for fold in range(4)
    ):
        raise RuntimeError("Every fold must have exactly one test evaluation call")
    window_metrics, subject_metrics = aggregate_oof(output_dir, selected_by_fold)
    log_event(
        logger, "INFO", "oof.complete", "OOF aggregation and integrity checks completed",
        selected_candidates={str(k): v["candidate_id"] for k, v in selected_by_fold.items()},
        oof_windows=window_metrics["num_windows"], oof_subjects=window_metrics["num_subjects"],
        sample_ids_unique=True, coverage_complete=window_metrics["coverage_complete"],
        window_accuracy=window_metrics["accuracy"],
        window_balanced_accuracy=window_metrics["balanced_accuracy"],
        window_macro_f1=window_metrics["macro_f1"],
        subject_probability_mean=subject_metrics["probability_mean"],
        subject_majority_vote=subject_metrics["majority_vote"],
        window_predictions_path=str(output_dir / "oof_window_predictions.csv"),
        subject_predictions_path=str(output_dir / "oof_subject_predictions.csv"),
        window_metrics_path=str(output_dir / "oof_window_metrics.json"),
        subject_metrics_path=str(output_dir / "oof_subject_metrics.json"),
    )
    summary = {
        "status": "completed",
        "local_code_validation": False,
        "cloud_full_run_pending": False,
        "search_complete": search_complete,
        "candidate_count": len(candidates),
        "fold_count": 4,
        "test_evaluation_calls": {
            str(fold): state["folds"][str(fold)]["evaluation"]["call_count"]
            for fold in range(4)
        },
        "window_metrics": window_metrics,
        "subject_metrics": subject_metrics,
        "selected_candidates": {
            str(fold): selected["candidate_id"] for fold, selected in selected_by_fold.items()
        },
        "per_fold_configs_may_differ": True,
        "feature_cache_key": feature_cache_info["feature_cache_key"],
        "feature_cache_path": feature_cache_info["feature_cache_path"],
        "feature_cache_policy": feature_cache_info["feature_cache_policy"],
        "feature_cache_reused": feature_cache_info["feature_cache_reused"],
        "feature_cache_build_root": feature_cache_info.get("feature_cache_build_root"),
        "cache_publish_mode": feature_cache_info.get("cache_publish_mode"),
        "cache_build_seconds": feature_cache_info["cache_build_seconds"],
        "cache_mean_size_bytes": directory_size(
            Path(feature_cache_info["feature_cache_path"]) / MEAN_CACHE_NAME
        ),
        "cache_flatten_size_bytes": directory_size(
            Path(feature_cache_info["feature_cache_path"]) / FLATTEN_CACHE_NAME
        ),
        "cache_row_count": cache_manifest["mean_row_count"],
        "cache_dtype": cache_manifest["dtype"],
        "cache_parity_passed": cache_manifest["parity_passed"],
        "backbone_forward_count_during_cache_build": cache_manifest["backbone_forward_count"],
        "backbone_processed_window_count": cache_manifest["backbone_processed_window_count"],
        "source_window_count": cache_manifest["source_window_count"],
        "backbone_forward_count_during_candidate_training": 0,
        "backbone_forward_count_during_test_evaluation": 0,
        "candidate_checkpoint_kind": "cached_feature_head_only",
    }
    atomic_write_json(output_dir / "sweep_summary.json", summary)
    report = [
        "# 50M sex CV4 parameter sweep", "",
        f"- candidates per fold: {len(candidates)}",
        f"- validation search complete: {search_complete}",
        "- test evaluations per fold: exactly 1",
        f"- OOF windows: {window_metrics['num_windows']}",
        "- each fold may select different pooling, head learning rate, and weight decay",
        "- feature cache is a label-free, parameter-free, per-sample deterministic transform",
        "- subject-level train/validation/test isolation remains at head training time",
        "- cached mode is invalid for backbone finetuning, LoRA/adapters, fitted feature scaling, probe search, or augmentation",
    ]
    (output_dir / "sweep_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    atomic_write_json(state_path, state)
    log_event(logger, "INFO", "run.complete", "sweep completed successfully",
              output_dir=str(output_dir), elapsed_seconds=round(time.monotonic() - run_started, 1),
              sweep_summary=str(output_dir / "sweep_summary.json"))
    return summary


def main(argv=None):
    options = parse_args(argv)
    if options.epochs <= 0 or options.batch_size <= 0 or options.num_workers < 0:
        raise ValueError("epochs/batch-size must be positive and num-workers non-negative")
    if min(options.feature_cache_batch_size, options.cache_log_every_batches,
           options.train_log_every_epochs) <= 0:
        raise ValueError("cache batch/log intervals and train log interval must be positive")
    if options.mode != "linear_probe":
        raise ValueError("Frozen feature cache sweep requires --mode linear_probe")
    if options.resume and options.feature_cache_policy == "rebuild":
        raise ValueError("--resume cannot be combined with feature-cache-policy rebuild")
    result = run_sweep(options)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
