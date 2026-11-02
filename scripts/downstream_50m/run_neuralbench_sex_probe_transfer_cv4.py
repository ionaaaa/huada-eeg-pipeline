#!/usr/bin/env python3
"""Stage-B NeuralBench source-probe transfer with subject-stratified CV4.

The NeuralBench full-finetuned backbone is used only to build/reuse a frozen,
label-free, mean-only feature cache.  Every fold then reloads the authoritative
source probe, evaluates epoch 0, trains only that probe on train subjects,
selects on validation subjects, and evaluates test subjects exactly once.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

try:
    from . import feature_cache as cache
    from .dataset import collate_fn
    from .largeScale_fineTune import CachedFeatureClassifier, train_one_epoch
    from .neuralbench_checkpoint import (
        DEFAULT_NEURALBENCH_SEX_SHA256,
        KEY_REMAP_VERSION,
        load_neuralbench_sex_model_state,
        safe_load_neuralbench_checkpoint,
    )
    from .run_logging import log_event, setup_run_logging
    from .run_neuralbench_sex_zero_shot import (
        FEATURE_DIM, LABEL_MAPPING, LABEL_TO_SEX, NUM_CLASSES, POOLING,
        PROBE_LAYER_IDX, PROBE_LAYER_RATIO, build_meta_zero_shot_model,
        path_content_fingerprint,
    )
    from .sex_cv4 import (
        aggregate_subject_predictions, binary_classification_metrics,
        build_sex_stratified_subject_cv4, split_rows,
        validate_oof_window_predictions,
    )
except ImportError:
    import feature_cache as cache
    from dataset import collate_fn
    from largeScale_fineTune import CachedFeatureClassifier, train_one_epoch
    from neuralbench_checkpoint import (
        DEFAULT_NEURALBENCH_SEX_SHA256,
        KEY_REMAP_VERSION,
        load_neuralbench_sex_model_state,
        safe_load_neuralbench_checkpoint,
    )
    from run_logging import log_event, setup_run_logging
    from run_neuralbench_sex_zero_shot import (
        FEATURE_DIM, LABEL_MAPPING, LABEL_TO_SEX, NUM_CLASSES, POOLING,
        PROBE_LAYER_IDX, PROBE_LAYER_RATIO, build_meta_zero_shot_model,
        path_content_fingerprint,
    )
    from sex_cv4 import (
        aggregate_subject_predictions, binary_classification_metrics,
        build_sex_stratified_subject_cv4, split_rows,
        validate_oof_window_predictions,
    )


RUN_MODE = "neuralbench_source_probe_transfer_cv4"
CHECKPOINT_KIND = "neuralbench_source_probe_transfer_head_only"
SOURCE_CHECKPOINT_KIND = "neuralbench_sex_full_finetune"
SELECTION_TOLERANCE = 1e-9
EXPECTED_SUBJECTS = 8
EXPECTED_WINDOWS = 480
PROBE_PARAMETER_COUNT = FEATURE_DIM * NUM_CLASSES + NUM_CLASSES
OOF_FIELDS = [
    "fold", "sample_id", "subject_id", "sex", "true_label", "prob_female",
    "prob_male", "predicted_label", "predicted_sex", "num_valid_tokens",
    "checkpoint_path",
]
SUBJECT_FIELDS = [
    "subject_id", "true_label", "true_sex", "num_windows", "mean_prob_female",
    "mean_prob_male", "probability_mean_pred_label", "probability_mean_pred_sex",
    "female_vote_count", "male_vote_count", "majority_vote_pred_label",
    "majority_vote_pred_sex", "tie_break_used", "test_fold",
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-lance", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-checkpoint-sha256", default=DEFAULT_NEURALBENCH_SEX_SHA256)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--feature-cache-root", required=True)
    parser.add_argument("--feature-cache-policy", choices=("auto", "require", "rebuild"), default="auto")
    parser.add_argument("--feature-cache-batch-size", type=int, default=8)
    parser.add_argument("--cache-log-every-batches", type=int, default=5)
    parser.add_argument("--zero-shot-results-dir")
    parser.add_argument("--head-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    positive = {
        "feature-cache-batch-size": args.feature_cache_batch_size,
        "cache-log-every-batches": args.cache_log_every_batches,
        "epochs": args.epochs,
        "early-stopping-patience": args.early_stopping_patience,
        "batch-size": args.batch_size,
    }
    for name, value in positive.items():
        if value <= 0:
            parser.error(f"--{name} must be positive")
    if args.num_workers < 0 or args.head_lr <= 0 or args.weight_decay < 0:
        parser.error("num-workers/learning-rate/weight-decay values are invalid")
    return args


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _git_state(project_root: Path) -> dict[str, Any]:
    def command(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=project_root, text=True, stderr=subprocess.DEVNULL).strip()
    try:
        return {"commit": command("rev-parse", "HEAD"), "dirty": bool(command("status", "--porcelain"))}
    except Exception:
        return {"commit": None, "dirty": None}


def _code_fingerprint(project_root: Path) -> str:
    relative_paths = [
        "scripts/downstream_50m/model.py",
        "scripts/downstream_50m/feature_cache.py",
        "scripts/downstream_50m/neuralbench_checkpoint.py",
        "scripts/downstream_50m/run_neuralbench_sex_zero_shot.py",
        "scripts/downstream_50m/run_neuralbench_sex_probe_transfer_cv4.py",
    ]
    return cache.sha256_json({path: cache.sha256_file(project_root / path) for path in relative_paths})


def tensor_state_fingerprint(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(str(key).encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def load_source_probe_state(checkpoint_path: str | Path, expected_sha256: str):
    """Re-read one fold's probe from the safe authoritative checkpoint."""
    checkpoint, info = safe_load_neuralbench_checkpoint(checkpoint_path, expected_sha256)
    state = checkpoint["state_dict"]
    weight = state.get("model.probe.weight")
    bias = state.get("model.probe.bias")
    if not torch.is_tensor(weight) or tuple(weight.shape) != (2, 512) or weight.dtype != torch.float32:
        raise RuntimeError("Source probe weight must be float32 [2,512]")
    if not torch.is_tensor(bias) or tuple(bias.shape) != (2,) or bias.dtype != torch.float32:
        raise RuntimeError("Source probe bias must be float32 [2]")
    probe_state = {"weight": weight.detach().cpu().clone(), "bias": bias.detach().cpu().clone()}
    return probe_state, tensor_state_fingerprint(probe_state), info


def build_source_initialized_probe(probe_state: Mapping[str, torch.Tensor], device: torch.device):
    """Materialize no random head values; assign checkpoint tensors to a meta module."""
    with torch.device("meta"):
        model = CachedFeatureClassifier(
            input_dim=FEATURE_DIM, num_classes=NUM_CLASSES, pooling=POOLING,
            use_mlp_head=False, head_dropout=0.0,
        )
    # ``assign=True`` preserves the incoming tensor storage.  Clone here so the
    # immutable source-probe baseline cannot be mutated by optimizer steps.
    mapped = {
        "head.weight": probe_state["weight"].detach().clone(),
        "head.bias": probe_state["bias"].detach().clone(),
    }
    model.load_state_dict(mapped, strict=True, assign=True)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    model = model.to(device)
    if any(value.is_meta for value in model.state_dict().values()):
        raise RuntimeError("Source probe still contains meta tensors")
    actual = {key.removeprefix("head."): value.detach().cpu() for key, value in model.state_dict().items()}
    if any(not torch.equal(actual[key], probe_state[key]) for key in ("weight", "bias")):
        raise RuntimeError("Fold initial head does not equal source probe")
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if trainable != PROBE_PARAMETER_COUNT:
        raise RuntimeError(f"Expected {PROBE_PARAMETER_COUNT} trainable probe parameters, got {trainable}")
    return model


def probe_state_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.head.state_dict().items()}


def probe_delta(source: Mapping[str, torch.Tensor], target: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    differences = [target[key].float() - source[key].float() for key in sorted(source)]
    flat = torch.cat([value.reshape(-1) for value in differences])
    return {
        "l2_norm": float(torch.linalg.vector_norm(flat).item()),
        "max_abs_diff": float(flat.abs().max().item()) if flat.numel() else 0.0,
        "changed_parameter_count": int(torch.count_nonzero(flat).item()),
        "total_parameter_count": int(flat.numel()),
    }


def candidate_is_better(candidate: Mapping[str, Any], best: Mapping[str, Any], tolerance=SELECTION_TOLERANCE) -> bool:
    """BA desc, loss asc, macro-F1 desc, then earlier epoch."""
    comparisons = (
        (float(candidate["balanced_accuracy"]), float(best["balanced_accuracy"]), 1),
        (float(candidate["loss"]), float(best["loss"]), -1),
        (float(candidate["macro_f1"]), float(best["macro_f1"]), 1),
    )
    for left, right, direction in comparisons:
        delta = (left - right) * direction
        if delta > tolerance:
            return True
        if delta < -tolerance:
            return False
    return int(candidate["epoch"]) < int(best["epoch"])


@torch.inference_mode()
def evaluate_probe(model, loader, device, fold=None, checkpoint_path=None, collect_rows=False):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_count = 0
    labels_out: list[int] = []
    predictions_out: list[int] = []
    rows: list[dict[str, Any]] = []
    for batch in loader:
        features = batch["feature"].to(device)
        labels = batch["label"].long().to(device)
        logits = model.classify_features(features)
        probabilities = torch.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)
        loss = criterion(logits, labels)
        count = int(labels.numel())
        total_loss += float(loss.item()) * count
        total_count += count
        labels_out.extend(labels.cpu().tolist())
        predictions_out.extend(predictions.cpu().tolist())
        if collect_rows:
            for index in range(count):
                truth = int(labels[index].item())
                predicted = int(predictions[index].item())
                rows.append({
                    "fold": int(fold),
                    "sample_id": str(batch["sample_id"][index]),
                    "subject_id": str(batch["subject_id"][index]),
                    "sex": LABEL_TO_SEX[truth],
                    "true_label": truth,
                    "prob_female": float(probabilities[index, 0].item()),
                    "prob_male": float(probabilities[index, 1].item()),
                    "predicted_label": predicted,
                    "predicted_sex": LABEL_TO_SEX[predicted],
                    "num_valid_tokens": int(batch["num_valid_tokens"][index].item()),
                    "checkpoint_path": "" if checkpoint_path is None else str(Path(checkpoint_path).resolve()),
                })
    if total_count == 0:
        raise RuntimeError("Evaluation loader is empty")
    metrics = binary_classification_metrics(labels_out, predictions_out)
    metrics.update({"loss": total_loss / total_count, "sample_count": total_count})
    return metrics, rows


def make_probe_optimizer(model: nn.Module, head_lr: float, weight_decay: float):
    parameters = [parameter for parameter in model.head.parameters() if parameter.requires_grad]
    if sum(parameter.numel() for parameter in parameters) != PROBE_PARAMETER_COUNT:
        raise RuntimeError("Optimizer parameter set is not exactly the source probe")
    optimizer = torch.optim.AdamW(
        [{"params": parameters, "name": "probe", "lr": head_lr, "weight_decay": weight_decay}],
        lr=head_lr, weight_decay=weight_decay,
    )
    group = optimizer.param_groups[0]
    if len(optimizer.param_groups) != 1 or not math.isclose(group["lr"], head_lr) or not math.isclose(group["weight_decay"], weight_decay):
        raise RuntimeError("AdamW probe parameter group does not match resolved configuration")
    return optimizer


def _save_head_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _load_best_head(model: nn.Module, path: Path, device: torch.device) -> Mapping[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("checkpoint_kind") != CHECKPOINT_KIND or set(payload) & {"backbone", "model_state_dict", "feature_cache"}:
        raise RuntimeError("Invalid head-only transfer checkpoint")
    state = payload.get("probe_state_dict")
    if not isinstance(state, Mapping) or set(state) != {"weight", "bias"}:
        raise RuntimeError("Head-only checkpoint has invalid probe state")
    model.head.load_state_dict(state, strict=True)
    model.to(device)
    return payload


def _data_loader(dataset, indices, batch_size, shuffle, seed, num_workers):
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        Subset(dataset, list(indices)), batch_size=int(batch_size), shuffle=bool(shuffle),
        num_workers=int(num_workers), collate_fn=collate_fn, drop_last=False,
        generator=generator if shuffle else None,
    )


def _indices_for_subjects(metadata, subjects):
    subject_set = set(subjects)
    return [index for index, row in enumerate(metadata) if str(row["subject_id"]) in subject_set]


def run_fold(
    *, fold_info, dataset, checkpoint_path, expected_sha256, cache_info,
    output_dir, device, batch_size, num_workers, seed, head_lr, weight_decay,
    epochs, patience, logger,
):
    fold = int(fold_info["fold"])
    fold_seed = int(seed) + fold
    fold_dir = output_dir / f"fold_{fold}"
    checkpoint_dir = fold_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    metadata = dataset.get_subject_label_rows()
    train_indices = _indices_for_subjects(metadata, fold_info["train_subjects"])
    validation_indices = _indices_for_subjects(metadata, fold_info["validation_subjects"])
    test_indices = _indices_for_subjects(metadata, fold_info["test_subjects"])
    if set(train_indices) & set(validation_indices) or set(train_indices) & set(test_indices) or set(validation_indices) & set(test_indices):
        raise RuntimeError("Fold window indices overlap")
    log_event(
        logger, "INFO", "fold.start", "starting source-probe transfer fold",
        fold=fold, fold_seed=fold_seed, feature_source="neuralbench_mean_cache",
        backbone_constructed=False, backbone_forward_count=0,
        head_initialization="neuralbench_source_probe",
        train_subjects=fold_info["train_subjects"],
        validation_subjects=fold_info["validation_subjects"], test_subjects=fold_info["test_subjects"],
    )
    # Required per-fold re-read; no previous fold state is accepted here.
    source_state, source_fingerprint, checkpoint_info = load_source_probe_state(checkpoint_path, expected_sha256)
    model = build_source_initialized_probe(source_state, device)
    initial_fingerprint = tensor_state_fingerprint(probe_state_cpu(model))
    if initial_fingerprint != source_fingerprint:
        raise RuntimeError("Fold initial head fingerprint does not match source")
    log_event(
        logger, "INFO", "fold.initialization", "source probe loaded for fold",
        fold=fold, head_initialization="neuralbench_source_probe",
        source_probe_fingerprint=source_fingerprint,
        source_checkpoint_sha256=checkpoint_info["sha256"],
        fold_initial_head_matches_source=True, trainable_parameter_count=PROBE_PARAMETER_COUNT,
        backbone_trainable_parameter_count=0,
    )
    train_loader = _data_loader(dataset, train_indices, batch_size, True, fold_seed, num_workers)
    validation_loader = _data_loader(dataset, validation_indices, batch_size, False, fold_seed, num_workers)
    criterion = nn.CrossEntropyLoss()
    optimizer_config = {
        "name": "AdamW", "learning_rate": float(head_lr),
        "weight_decay": float(weight_decay),
        "parameter_count": PROBE_PARAMETER_COUNT, "param_group_count": 1,
        "scheduler_created": False,
    }
    epoch0_metrics, epoch0_rows = evaluate_probe(model, validation_loader, device, collect_rows=True, fold=fold)
    _write_csv(fold_dir / "validation_epoch0_predictions.csv", epoch0_rows, OOF_FIELDS)
    best = {"epoch": 0, **epoch0_metrics}
    best_state = probe_state_cpu(model)
    best_checkpoint = checkpoint_dir / "best_probe.pt"
    history = [{"epoch": 0, "train_loss": "", **{f"validation_{key}": value for key, value in epoch0_metrics.items() if key != "confusion_matrix"}, "is_best": True}]

    def checkpoint_payload():
        delta = probe_delta(source_state, best_state)
        return {
            "checkpoint_kind": CHECKPOINT_KIND,
            "probe_state_dict": best_state,
            "source_checkpoint_sha256": checkpoint_info["sha256"],
            "source_probe_fingerprint": source_fingerprint,
            "feature_cache_key": cache_info["feature_cache_key"],
            "feature_cache_path": cache_info["feature_cache_path"],
            "pooling": POOLING, "feature_dim": FEATURE_DIM,
            "label_mapping": LABEL_MAPPING, "fold": fold,
            "best_epoch": int(best["epoch"]),
            "adaptation_applied": int(best["epoch"]) > 0,
            "optimizer_config": optimizer_config,
            "validation_metrics": {key: value for key, value in best.items() if key != "epoch"},
            "probe_parameter_delta": delta,
        }

    _save_head_checkpoint(best_checkpoint, checkpoint_payload())
    log_event(logger, "INFO", "fold.epoch0", "epoch 0 source probe is initial best candidate", fold=fold, **epoch0_metrics)
    # The optimizer starts from a fresh state only after epoch 0 is evaluated
    # and durably recorded as the initial best candidate.
    optimizer = make_probe_optimizer(model, head_lr, weight_decay)
    optimizer_config["learning_rate"] = float(optimizer.param_groups[0]["lr"])
    optimizer_config["weight_decay"] = float(optimizer.param_groups[0]["weight_decay"])
    non_improving = 0
    stopped_early = False
    for epoch in range(1, int(epochs) + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, "sex_classification")
        validation_metrics, _ = evaluate_probe(model, validation_loader, device)
        candidate = {"epoch": epoch, **validation_metrics}
        improved = candidate_is_better(candidate, best)
        if improved:
            best = candidate
            best_state = probe_state_cpu(model)
            _save_head_checkpoint(best_checkpoint, checkpoint_payload())
            non_improving = 0
        else:
            non_improving += 1
        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "validation_loss": validation_metrics["loss"],
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_balanced_accuracy": validation_metrics["balanced_accuracy"],
            "validation_macro_f1": validation_metrics["macro_f1"],
            "is_best": improved,
        })
        log_event(
            logger, "INFO", "train.epoch", "probe transfer epoch completed",
            fold=fold, epoch=f"{epoch}/{epochs}", train_loss=train_metrics["loss"],
            validation_loss=validation_metrics["loss"],
            validation_balanced_accuracy=validation_metrics["balanced_accuracy"],
            validation_macro_f1=validation_metrics["macro_f1"],
            best_epoch=best["epoch"], best_checkpoint_updated=improved,
            early_stopping_non_improving_epochs=non_improving,
            early_stopping_patience=patience,
            optimizer_learning_rate=optimizer.param_groups[0]["lr"],
            optimizer_weight_decay=optimizer.param_groups[0]["weight_decay"],
        )
        if non_improving >= int(patience):
            stopped_early = True
            log_event(logger, "INFO", "train.early_stop", "early stopping triggered", fold=fold, epoch=epoch, best_epoch=best["epoch"], patience=patience)
            break
    _write_csv(
        fold_dir / "epoch_history.csv", history,
        ["epoch", "train_loss", "train_accuracy", "validation_loss", "validation_accuracy", "validation_balanced_accuracy", "validation_macro_f1", "validation_sample_count", "is_best"],
    )
    del optimizer
    payload = _load_best_head(model, best_checkpoint, device)
    restored = probe_state_cpu(model)
    if any(not torch.equal(restored[key], payload["probe_state_dict"][key]) for key in restored):
        raise RuntimeError("Best head restoration failed")
    # Test loader is intentionally created only after selection is locked.
    test_loader = _data_loader(dataset, test_indices, batch_size, False, fold_seed, num_workers)
    test_metrics, test_rows = evaluate_probe(
        model, test_loader, device, fold=fold, checkpoint_path=best_checkpoint, collect_rows=True
    )
    _write_csv(fold_dir / "test_predictions.csv", test_rows, OOF_FIELDS)
    fold_summary = {
        "fold": fold, "fold_seed": fold_seed,
        "train_subjects": list(fold_info["train_subjects"]),
        "validation_subjects": list(fold_info["validation_subjects"]),
        "test_subjects": list(fold_info["test_subjects"]),
        "train_sample_count": len(train_indices), "validation_sample_count": len(validation_indices),
        "test_sample_count": len(test_indices),
        "source_probe_fingerprint": source_fingerprint,
        "fold_initial_head_matches_source": True,
        "epoch0_validation": epoch0_metrics,
        "best_validation": payload["validation_metrics"],
        "validation_epoch0_to_best_delta": {
            key: float(payload["validation_metrics"][key]) - float(epoch0_metrics[key])
            for key in ("accuracy", "balanced_accuracy", "macro_f1", "loss")
        },
        "best_epoch": int(payload["best_epoch"]),
        "adaptation_applied": bool(payload["adaptation_applied"]),
        "probe_parameter_delta": payload["probe_parameter_delta"],
        "epochs_executed": len(history) - 1, "stopped_early": stopped_early,
        "test_metrics": test_metrics,
        "test_accessed_during_selection": False,
        "test_evaluation_count": 1,
        "post_hoc_label_flip": False,
        "feature_source": "neuralbench_mean_cache",
        "backbone_constructed": False, "backbone_forward_count": 0,
        "backbone_trainable_parameter_count": 0,
        "trainable_parameter_count": PROBE_PARAMETER_COUNT,
        "optimizer_config": optimizer_config,
        "best_checkpoint": str(best_checkpoint.resolve()),
    }
    _write_json(fold_dir / "fold_summary.json", fold_summary)
    log_event(logger, "INFO", "test.evaluate", "locked best head evaluated on test exactly once", fold=fold, best_epoch=payload["best_epoch"], adaptation_applied=payload["adaptation_applied"], test_evaluation_count=1, backbone_forward_count=0, **test_metrics)
    return fold_summary, test_rows


def build_cache_identity(args, project_root: Path):
    lance_metadata_fingerprint, input_info = cache.lance_fingerprint(args.input_lance)
    input_info = {**input_info, "metadata_fingerprint": lance_metadata_fingerprint}
    input_fingerprint = path_content_fingerprint(args.input_lance)
    checkpoint_info = cache.sha256_file(args.checkpoint)
    if checkpoint_info != str(args.expected_checkpoint_sha256).lower():
        raise RuntimeError(f"Checkpoint SHA256 mismatch: expected={args.expected_checkpoint_sha256}, actual={checkpoint_info}")
    git = _git_state(project_root)
    identity, cache_key = cache.build_cache_identity(
        input_fingerprint=input_fingerprint,
        input_row_count=input_info["row_count"],
        sample_id_fingerprint=input_info["sample_id_fingerprint"],
        checkpoint_sha256=checkpoint_info,
        model_config={
            "input_dim": 100, "d_model": 512, "depth": 12, "n_heads": 8,
            "mlp_ratio": 4.0, "dropout": 0.1, "n_channels": 64,
            "n_time_patches": 10, "n_bands": 5, "frames_per_patch": 5,
            "objective": "timefreq",
        },
        shape_params={"num_tokens": 640, "patch_len": 100},
        probe_layer_idx=PROBE_LAYER_IDX, probe_layer_ratio=PROBE_LAYER_RATIO,
        model_code_sha256=cache.model_implementation_fingerprint(project_root),
        git_commit=git["commit"], requested_poolings=("mean",),
        checkpoint_remap_version=KEY_REMAP_VERSION, feature_dim=FEATURE_DIM,
        feature_extraction_code_sha256=_code_fingerprint(project_root),
        source_checkpoint_kind=SOURCE_CHECKPOINT_KIND,
    )
    return identity, cache_key, input_fingerprint, input_info, checkpoint_info, git


def ensure_neuralbench_mean_cache(args, logger):
    project_root = Path(__file__).resolve().parents[2]
    identity, cache_key, input_fingerprint, input_info, checkpoint_sha, git = build_cache_identity(args, project_root)
    log_event(
        logger, "INFO", "checkpoint.verify", "source checkpoint SHA256 verified",
        checkpoint=str(Path(args.checkpoint).resolve()), checkpoint_sha256=checkpoint_sha,
        expected_checkpoint_sha256=str(args.expected_checkpoint_sha256).lower(),
    )
    root = Path(args.feature_cache_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    final_dir, manifest, reused = cache.resolve_cache_policy(root, cache_key, args.feature_cache_policy, identity)
    log_event(logger, "INFO", "cache.lookup", "NeuralBench mean-only cache resolved", cache_key=cache_key, cache_policy=args.feature_cache_policy, cache_path=str(final_dir), cache_reused=reused, requested_poolings=["mean"])
    if reused:
        if manifest.get("requested_poolings") != ["mean"] or (final_dir / cache.FLATTEN_CACHE_NAME).exists():
            raise RuntimeError("Reused NeuralBench cache is not strictly mean-only")
        return {"feature_cache_key": cache_key, "feature_cache_path": str(final_dir), "feature_cache_policy": args.feature_cache_policy, "feature_cache_reused": True, "manifest": manifest}
    build_root = Path(tempfile.gettempdir()) / "huada_neuralbench_mean_cache_build"
    build_root.mkdir(parents=True, exist_ok=True)
    temporary_dir = build_root / f".{final_dir.name}.building.{uuid.uuid4().hex}"
    source = cache.TokenFeatureSourceDataset(args.input_lance)
    model = build_meta_zero_shot_model()
    _, checkpoint_file_info, _, mapping_summary = load_neuralbench_sex_model_state(
        model, args.checkpoint, args.expected_checkpoint_sha256
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    device = torch.device(args.device)
    model = model.to(device).eval()
    manifest_base = {
        "cache_schema_version": cache.CACHE_SCHEMA_VERSION,
        "cache_key": cache_key, "cache_identity": identity,
        "git_commit": git["commit"], "dirty_worktree": git["dirty"],
        "input_lance": str(Path(args.input_lance).resolve()),
        "input_lance_fingerprint": input_fingerprint,
        "input_lance_metadata_fingerprint": input_info["metadata_fingerprint"],
        "input_lance_version": input_info["version"],
        "input_row_count": input_info["row_count"],
        "sample_id_fingerprint": input_info["sample_id_fingerprint"],
        "sample_ids": [str(row["sample_id"]) for row in source.metadata],
        "pretrained_checkpoint": checkpoint_file_info["path"],
        "pretrained_checkpoint_size_bytes": checkpoint_file_info["size_bytes"],
        "pretrained_checkpoint_sha256": checkpoint_sha,
        "checkpoint_remap_version": KEY_REMAP_VERSION,
        "checkpoint_mapping": mapping_summary,
        "probe_layer_idx": PROBE_LAYER_IDX, "probe_layer_ratio": PROBE_LAYER_RATIO,
        "pooling": "mean", "num_tokens": 640, "d_model": FEATURE_DIM,
        "python_version": platform.python_version(), "torch_version": torch.__version__,
        "cache_publish_mode": "copy_then_success_marker",
    }
    try:
        manifest = cache.build_feature_cache(
            source_dataset=source, backbone=model.backbone,
            probe_layer_idx=PROBE_LAYER_IDX, output_dir=temporary_dir,
            batch_size=args.feature_cache_batch_size, device=device,
            expected_num_tokens=640, d_model=FEATURE_DIM,
            manifest_base=manifest_base, logger=logger,
            log_every_batches=args.cache_log_every_batches,
            requested_poolings=("mean",), parity_head=model.head,
        )
        manifest = cache.publish_feature_cache(temporary_dir, final_dir, identity, cache_key, logger=logger)
    finally:
        if temporary_dir.exists() and (temporary_dir / cache.SUCCESS_NAME).exists():
            shutil.rmtree(temporary_dir)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if (final_dir / cache.FLATTEN_CACHE_NAME).exists():
        raise RuntimeError("Mean-only workflow unexpectedly created flatten cache")
    return {"feature_cache_key": cache_key, "feature_cache_path": str(final_dir), "feature_cache_policy": args.feature_cache_policy, "feature_cache_reused": False, "manifest": manifest}


def aggregate_outputs(output_dir, split_payload, dataset, fold_summaries, oof_rows):
    metadata = dataset.get_subject_label_rows()
    eligible = {str(row["sample_id"]): {"subject_id": str(row["subject_id"]), "label": int(row["label"])} for row in metadata}
    subject_test_folds = {subject: int(fold["fold"]) for fold in split_payload["folds"] for subject in fold["test_subjects"]}
    validate_oof_window_predictions(oof_rows, eligible, subject_test_folds)
    oof_rows = sorted(oof_rows, key=lambda row: str(row["sample_id"]))
    _write_csv(output_dir / "oof_window_predictions.csv", oof_rows, OOF_FIELDS)
    window_metrics = binary_classification_metrics([row["true_label"] for row in oof_rows], [row["predicted_label"] for row in oof_rows])
    window_metrics["sample_count"] = len(oof_rows)
    subject_rows = aggregate_subject_predictions(oof_rows)
    _write_csv(output_dir / "oof_subject_predictions.csv", subject_rows, SUBJECT_FIELDS)
    probability_metrics = binary_classification_metrics([row["true_label"] for row in subject_rows], [row["probability_mean_pred_label"] for row in subject_rows])
    probability_metrics["sample_count"] = len(subject_rows)
    majority_metrics = binary_classification_metrics([row["true_label"] for row in subject_rows], [row["majority_vote_pred_label"] for row in subject_rows])
    majority_metrics["sample_count"] = len(subject_rows)
    _write_json(output_dir / "oof_window_metrics.json", window_metrics)
    _write_json(output_dir / "oof_subject_probability_mean_metrics.json", probability_metrics)
    _write_json(output_dir / "oof_subject_majority_vote_metrics.json", majority_metrics)
    fold_result_rows = []
    for row in fold_summaries:
        delta = row["validation_epoch0_to_best_delta"]
        change = row["probe_parameter_delta"]
        fold_result_rows.append({
            "fold": row["fold"], "fold_seed": row["fold_seed"],
            "train_subjects": ";".join(row["train_subjects"]),
            "validation_subjects": ";".join(row["validation_subjects"]),
            "test_subjects": ";".join(row["test_subjects"]),
            "best_epoch": row["best_epoch"], "adaptation_applied": row["adaptation_applied"],
            "epochs_executed": row["epochs_executed"], "stopped_early": row["stopped_early"],
            "epoch0_validation_balanced_accuracy": row["epoch0_validation"]["balanced_accuracy"],
            "best_validation_balanced_accuracy": row["best_validation"]["balanced_accuracy"],
            "validation_balanced_accuracy_delta": delta["balanced_accuracy"],
            "epoch0_validation_loss": row["epoch0_validation"]["loss"],
            "best_validation_loss": row["best_validation"]["loss"],
            "validation_loss_delta": delta["loss"],
            "epoch0_validation_macro_f1": row["epoch0_validation"]["macro_f1"],
            "best_validation_macro_f1": row["best_validation"]["macro_f1"],
            "validation_macro_f1_delta": delta["macro_f1"],
            "test_accuracy": row["test_metrics"]["accuracy"],
            "test_balanced_accuracy": row["test_metrics"]["balanced_accuracy"],
            "test_macro_f1": row["test_metrics"]["macro_f1"],
            "test_loss": row["test_metrics"]["loss"],
            "test_sample_count": row["test_sample_count"],
            "test_evaluation_count": row["test_evaluation_count"],
            "probe_delta_l2_norm": change["l2_norm"],
            "probe_delta_max_abs_diff": change["max_abs_diff"],
            "probe_changed_parameter_count": change["changed_parameter_count"],
            "source_probe_fingerprint": row["source_probe_fingerprint"],
        })
    fold_fields = list(fold_result_rows[0])
    _write_csv(output_dir / "fold_results.csv", fold_result_rows, fold_fields)
    initial_fingerprints = {row["source_probe_fingerprint"] for row in fold_summaries}
    if len(initial_fingerprints) != 1:
        raise RuntimeError("Fold source probe fingerprints differ")
    cm = window_metrics["confusion_matrix"]
    recalls = {
        "female": cm[0][0] / max(sum(cm[0]), 1),
        "male": cm[1][1] / max(sum(cm[1]), 1),
    }
    wrong_subjects = sorted(row["subject_id"] for row in subject_rows if int(row["probability_mean_pred_label"]) != int(row["true_label"]))
    return {
        "window_metrics": window_metrics,
        "subject_probability_mean_metrics": probability_metrics,
        "subject_majority_vote_metrics": majority_metrics,
        "subject_rows": subject_rows,
        "source_probe_fingerprint": next(iter(initial_fingerprints)),
        "best_epoch_zero_fold_count": sum(int(row["best_epoch"]) == 0 for row in fold_summaries),
        "adaptation_fold_count": sum(bool(row["adaptation_applied"]) for row in fold_summaries),
        "class_recall": recalls, "incorrect_subjects_probability_mean": wrong_subjects,
        "fold_result_rows": fold_result_rows,
    }


def compare_zero_shot(zero_dir: str | Path, transfer, oof_rows, checkpoint_sha, input_fingerprint):
    zero_dir = Path(zero_dir)
    summary = json.loads((zero_dir / "zero_shot_summary.json").read_text(encoding="utf-8"))
    provenance = json.loads((zero_dir / "zero_shot_provenance.json").read_text(encoding="utf-8"))
    if summary.get("evaluation_kind") != "external_checkpoint_zero_shot":
        raise ValueError("Zero-shot result kind is invalid")
    if provenance["checkpoint"]["sha256"] != checkpoint_sha:
        raise ValueError("Zero-shot source checkpoint SHA differs")
    if provenance["input_lance"]["content_fingerprint_before_sha256"] != input_fingerprint:
        raise ValueError("Zero-shot input Lance fingerprint differs")
    with (zero_dir / "window_predictions.csv").open(encoding="utf-8", newline="") as handle:
        zero_windows = list(csv.DictReader(handle))
    with (zero_dir / "subject_predictions.csv").open(encoding="utf-8", newline="") as handle:
        zero_subjects = list(csv.DictReader(handle))
    if {row["sample_id"] for row in zero_windows} != {row["sample_id"] for row in oof_rows}:
        raise ValueError("Zero-shot and transfer sample_id sets differ")
    transfer_subjects = {row["subject_id"]: row for row in transfer["subject_rows"]}
    subject_changes = []
    for row in zero_subjects:
        subject_id = row["subject_id"]
        new = transfer_subjects[subject_id]
        old_pred = int(row["probability_mean_pred_label"])
        new_pred = int(new["probability_mean_pred_label"])
        subject_changes.append({"subject_id": subject_id, "zero_shot_predicted_label": old_pred, "transfer_predicted_label": new_pred, "prediction_changed": old_pred != new_pred})
    zero_window = summary["window_metrics"]
    zero_probability = summary["subject_probability_mean_metrics"]
    zero_majority = summary["subject_majority_vote_metrics"]
    return {
        "comparison_performed_after_all_fold_tests": True,
        "used_for_selection": False,
        "input_lance_fingerprint_match": True,
        "source_checkpoint_sha_match": True,
        "sample_id_set_match": True,
        "zero_shot_window_metrics": zero_window,
        "probe_transfer_oof_window_metrics": transfer["window_metrics"],
        "window_accuracy_absolute_delta": transfer["window_metrics"]["accuracy"] - zero_window["accuracy"],
        "zero_shot_subject_probability_mean_metrics": zero_probability,
        "probe_transfer_subject_probability_mean_metrics": transfer["subject_probability_mean_metrics"],
        "subject_probability_mean_accuracy_absolute_delta": transfer["subject_probability_mean_metrics"]["accuracy"] - zero_probability["accuracy"],
        "zero_shot_subject_majority_vote_metrics": zero_majority,
        "probe_transfer_subject_majority_vote_metrics": transfer["subject_majority_vote_metrics"],
        "subject_majority_vote_accuracy_absolute_delta": transfer["subject_majority_vote_metrics"]["accuracy"] - zero_majority["accuracy"],
        "subject_prediction_changes": subject_changes,
    }


def dry_run(args):
    project_root = Path(__file__).resolve().parents[2]
    identity, cache_key, input_fingerprint, input_info, checkpoint_sha, git = build_cache_identity(args, project_root)
    print(json.dumps({
        "dry_run": True, "formal_output_created": False, "feature_cache_built": False,
        "training_performed": False, "run_mode": RUN_MODE,
        "input_row_count": input_info["row_count"], "input_lance_fingerprint": input_fingerprint,
        "checkpoint_sha256": checkpoint_sha, "feature_cache_key": cache_key,
        "cache_identity": identity, "git": git,
        "resolved_config": vars(args),
    }, ensure_ascii=False, indent=2, default=str, sort_keys=True))
    return 0


def run(args):
    if args.dry_run:
        return dry_run(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    error_path = output_dir / "transfer_errors.log"
    error_path.write_text("", encoding="utf-8")
    logger = setup_run_logging(output_dir, args.log_level, append=False)
    try:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        resolved = {
            "run_mode": RUN_MODE, "pooling": POOLING, "probe_layer_idx": PROBE_LAYER_IDX,
            "probe_layer_ratio": PROBE_LAYER_RATIO, "feature_dim": FEATURE_DIM,
            "label_mapping": LABEL_MAPPING, "optimizer": "AdamW", "scheduler": None,
            "head_lr": args.head_lr, "weight_decay": args.weight_decay,
            "max_epochs": args.epochs, "early_stopping_patience": args.early_stopping_patience,
            "batch_size": args.batch_size, "seed": args.seed, "split_seed": args.split_seed,
            "device": str(device), "post_hoc_label_flip": False,
        }
        _write_json(output_dir / "resolved_config.json", resolved)
        cache_info = ensure_neuralbench_mean_cache(args, logger)
        _write_json(output_dir / "feature_cache_reference.json", cache_info)
        if cache_info["manifest"].get("requested_poolings") != ["mean"]:
            raise RuntimeError("Stage-B requires a mean-only feature cache")
        dataset = cache.CachedFeatureDataset(
            cache_info["feature_cache_path"], "mean", args.input_lance,
            "sex_classification", expected_cache_key=cache_info["feature_cache_key"],
        )
        metadata = dataset.get_subject_label_rows()
        subject_labels = {}
        for row in metadata:
            subject = str(row["subject_id"]); label = int(row["label"])
            if subject in subject_labels and subject_labels[subject] != label:
                raise RuntimeError(f"Inconsistent subject label: {subject}")
            subject_labels[subject] = label
        split_payload = build_sex_stratified_subject_cv4(subject_labels, args.split_seed)
        _write_csv(output_dir / "fold_assignments.csv", split_rows(split_payload), ["fold", "subject_id", "sex", "label", "split", "split_seed"])
        log_event(logger, "INFO", "cv.split", "subject-stratified CV4 resolved", folds=split_payload["folds"])
        fold_summaries = []
        oof_rows = []
        for fold_info in split_payload["folds"]:
            summary, rows = run_fold(
                fold_info=fold_info, dataset=dataset, checkpoint_path=args.checkpoint,
                expected_sha256=args.expected_checkpoint_sha256, cache_info=cache_info,
                output_dir=output_dir, device=device, batch_size=args.batch_size,
                num_workers=args.num_workers, seed=args.seed, head_lr=args.head_lr,
                weight_decay=args.weight_decay, epochs=args.epochs,
                patience=args.early_stopping_patience, logger=logger,
            )
            fold_summaries.append(summary); oof_rows.extend(rows)
        transfer = aggregate_outputs(output_dir, split_payload, dataset, fold_summaries, oof_rows)
        identity = cache_info["manifest"]["cache_identity"]
        comparison = None
        if args.zero_shot_results_dir:
            comparison = compare_zero_shot(
                args.zero_shot_results_dir, transfer, oof_rows,
                cache_info["manifest"]["pretrained_checkpoint_sha256"],
                identity["input_lance_fingerprint"],
            )
            _write_json(output_dir / "zero_shot_vs_transfer_comparison.json", comparison)
            log_event(logger, "INFO", "comparison.zero_shot", "post-test zero-shot comparison completed", **{key: value for key, value in comparison.items() if key.endswith("delta")})
        issues = []
        if len(dataset) != EXPECTED_WINDOWS:
            issues.append({"severity": "warning", "code": "unexpected_window_count", "detail": f"expected {EXPECTED_WINDOWS}, observed {len(dataset)}"})
        if len(subject_labels) != EXPECTED_SUBJECTS:
            issues.append({"severity": "warning", "code": "unexpected_subject_count", "detail": f"expected {EXPECTED_SUBJECTS}, observed {len(subject_labels)}"})
        _write_csv(output_dir / "transfer_issues.csv", issues, ["severity", "code", "detail"])
        final_summary = {
            "evaluation_kind": "neuralbench_source_probe_transfer_subject_cv4",
            "run_mode": RUN_MODE, "fold_count": 4, "eligible_sample_count": len(dataset),
            "subject_count": len(subject_labels), "feature_source": "neuralbench_mean_cache",
            "feature_cache_key": cache_info["feature_cache_key"],
            "source_checkpoint_sha256": cache_info["manifest"]["pretrained_checkpoint_sha256"],
            "source_probe_fingerprint": transfer["source_probe_fingerprint"],
            "head_initialization": "neuralbench_source_probe",
            "trainable_parameter_count": PROBE_PARAMETER_COUNT,
            "backbone_trainable_parameter_count": 0,
            "backbone_forward_count_during_cv": 0,
            "best_epoch_zero_fold_count": transfer["best_epoch_zero_fold_count"],
            "adaptation_fold_count": transfer["adaptation_fold_count"],
            "window_metrics": transfer["window_metrics"],
            "subject_probability_mean_metrics": transfer["subject_probability_mean_metrics"],
            "subject_majority_vote_metrics": transfer["subject_majority_vote_metrics"],
            "class_recall": transfer["class_recall"],
            "incorrect_subjects_probability_mean": transfer["incorrect_subjects_probability_mean"],
            "test_accessed_during_selection": False, "test_evaluation_count_per_fold": 1,
            "post_hoc_label_flip": False, "zero_shot_comparison_performed": comparison is not None,
        }
        _write_json(output_dir / "transfer_summary.json", final_summary)
        provenance = {
            **final_summary, "resolved_config": resolved,
            "input_lance": {
                "path": str(Path(args.input_lance).resolve()),
                "content_fingerprint_sha256": identity["input_lance_fingerprint"],
                "metadata_fingerprint_sha256": cache_info["manifest"]["input_lance_metadata_fingerprint"],
                "version": cache_info["manifest"]["input_lance_version"],
                "row_count": identity["input_row_count"],
                "sample_id_fingerprint_sha256": identity["sample_id_fingerprint"],
            },
            "feature_cache": cache_info, "folds": fold_summaries,
            "git": _git_state(Path(__file__).resolve().parents[2]),
            "optimizer_reinitialized_per_fold": True, "scheduler_created": False,
            "source_checkpoint_optimizer_state_restored": False,
        }
        _write_json(output_dir / "processing_provenance.json", provenance)
        fold_report_rows = "\n".join(
            "| {fold} | {best_epoch} | {adaptation_applied} | {epoch0_validation_balanced_accuracy:.6f} | "
            "{best_validation_balanced_accuracy:.6f} | {test_balanced_accuracy:.6f} | "
            "{probe_delta_l2_norm:.6g} | {probe_delta_max_abs_diff:.6g} | {probe_changed_parameter_count} |".format(**row)
            for row in transfer["fold_result_rows"]
        )
        report = f"""# NeuralBench sex source-probe transfer CV4 report

- source checkpoint SHA256: `{final_summary['source_checkpoint_sha256']}`
- feature cache: mean-only `[N,512]`, key `{final_summary['feature_cache_key']}`
- folds/windows/subjects: `4` / `{len(dataset)}` / `{len(subject_labels)}`
- trainable parameters: `{PROBE_PARAMETER_COUNT}` probe parameters; backbone `0`
- folds retaining epoch 0: `{transfer['best_epoch_zero_fold_count']}`
- folds applying adaptation: `{transfer['adaptation_fold_count']}`
- window accuracy/balanced accuracy/macro-F1: `{transfer['window_metrics']['accuracy']:.6f}` / `{transfer['window_metrics']['balanced_accuracy']:.6f}` / `{transfer['window_metrics']['macro_f1']:.6f}`
- subject probability-mean accuracy: `{transfer['subject_probability_mean_metrics']['accuracy']:.6f}`
- subject majority-vote accuracy: `{transfer['subject_majority_vote_metrics']['accuracy']:.6f}`
- female/male recall: `{transfer['class_recall']['female']:.6f}` / `{transfer['class_recall']['male']:.6f}`
- incorrect subjects (probability mean): `{transfer['incorrect_subjects_probability_mean']}`
- test was accessed once per fold only after best-head lock; no post-hoc label flip was performed.

| fold | best epoch | adapted | epoch-0 val BA | best val BA | test BA | probe delta L2 | probe delta max abs | changed params |
|---:|---:|:---:|---:|---:|---:|---:|---:|---:|
{fold_report_rows}
"""
        (output_dir / "transfer_report.md").write_text(report, encoding="utf-8")
        log_event(logger, "INFO", "run.complete", "source-probe transfer CV4 completed", best_epoch_zero_fold_count=transfer["best_epoch_zero_fold_count"], adaptation_fold_count=transfer["adaptation_fold_count"], **transfer["window_metrics"])
        return 0
    except Exception:
        error_path.write_text(traceback.format_exc(), encoding="utf-8")
        logger.exception("probe transfer CV4 failed")
        raise


def main(argv: Sequence[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
