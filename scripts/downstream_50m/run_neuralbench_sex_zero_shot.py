#!/usr/bin/env python3
"""External NeuralBench full-checkpoint sex zero-shot evaluation.

This entry point performs one ordered online pass over eligible Lance windows.
It has no optimizer, scheduler, backward pass, split, cache, checkpoint search,
or target-label-driven calibration path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import lance
import numpy as np
import torch
from torch.utils.data import DataLoader

try:  # package import in tests
    from .dataset import collate_fn
    from .lance_dataset import LanceEEGPretrainDataset
    from .largeScale_fineTune import EEGDownstreamClassifier
    from .model import EEGPretrainModel
    from .neuralbench_checkpoint import (
        DEFAULT_NEURALBENCH_SEX_SHA256,
        KEY_REMAP_VERSION,
        load_neuralbench_sex_model_state,
        verify_checkpoint_sha256,
        write_checkpoint_key_mapping,
    )
    from .sex_cv4 import aggregate_subject_predictions, binary_classification_metrics
except ImportError:  # direct ``python scripts/...py`` execution
    from dataset import collate_fn
    from lance_dataset import LanceEEGPretrainDataset
    from largeScale_fineTune import EEGDownstreamClassifier
    from model import EEGPretrainModel
    from neuralbench_checkpoint import (
        DEFAULT_NEURALBENCH_SEX_SHA256,
        KEY_REMAP_VERSION,
        load_neuralbench_sex_model_state,
        verify_checkpoint_sha256,
        write_checkpoint_key_mapping,
    )
    from sex_cv4 import aggregate_subject_predictions, binary_classification_metrics


RUN_MODE = "zero_shot"
POOLING = "mean"
PROBE_LAYER_IDX = 8
PROBE_LAYER_RATIO = 0.75
FEATURE_DIM = 512
NUM_CLASSES = 2
LABEL_MAPPING = {"female": 0, "male": 1}
LABEL_TO_SEX = {0: "female", 1: "male"}
SOURCE_FORMAT = "pytorch_lightning_state_dict"
ZERO_SHOT_COLUMNS = list(LanceEEGPretrainDataset.TENSOR_COLUMNS) + ["window_index"]
WINDOW_FIELDS = [
    "sample_id", "subject_id", "recording_id", "window_index", "true_label",
    "true_sex", "raw_logit_female", "raw_logit_male", "prob_female", "prob_male",
    "predicted_label", "predicted_sex", "correct",
]
SUBJECT_FIELDS = [
    "subject_id", "true_label", "true_sex", "num_windows", "mean_prob_female",
    "mean_prob_male", "probability_mean_pred_label", "probability_mean_pred_sex",
    "female_vote_count", "male_vote_count", "majority_vote_pred_label",
    "majority_vote_pred_sex", "tie_break_used",
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-lance", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--expected-checkpoint-sha256", default=DEFAULT_NEURALBENCH_SEX_SHA256
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--log-every-batches", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    if args.log_every_batches <= 0:
        parser.error("--log-every-batches must be positive")
    if str(args.log_level).upper() not in logging._nameToLevel:
        parser.error("--log-level is invalid")
    return args


def normalize_sex(value: Any) -> tuple[int, str]:
    """Normalize only the audited female/male encodings."""

    if hasattr(value, "as_py"):
        value = value.as_py()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        key = str(int(value))
    elif isinstance(value, (float, np.floating)) and math.isfinite(float(value)) and float(value) in (0.0, 1.0):
        key = str(int(value))
    else:
        key = "" if value is None else str(value).strip().lower()
    aliases = {
        "female": (0, "female"), "f": (0, "female"), "0": (0, "female"),
        "male": (1, "male"), "m": (1, "male"), "1": (1, "male"),
    }
    if key not in aliases:
        raise ValueError("sex is empty or not an audited female/male encoding")
    return aliases[key]


class NeuralBenchSexZeroShotDataset(LanceEEGPretrainDataset):
    """Validated eligible-row view that preserves original Lance order."""

    def __init__(self, lance_path: str | Path):
        self.excluded_samples: list[dict[str, Any]] = []
        self.input_row_count = 0
        super().__init__(
            lance_path=str(lance_path),
            columns=ZERO_SHOT_COLUMNS,
            task="sex_classification",
            subject_split_manifest=None,
            split=None,
            max_samples=None,
        )

    def _label_from_row(self, row):
        return normalize_sex(row.get("sex"))

    def _select_indices(self, max_samples=None):
        del max_samples
        self.input_row_count = len(self._metadata)
        selected: list[int] = []
        seen_samples: set[str] = set()
        subject_labels: dict[str, int] = {}
        for index, row in enumerate(self._metadata):
            sample_id = str(row.get("sample_id", "")).strip()
            if not sample_id:
                raise ValueError(f"Input row {index} has an empty sample_id")
            if sample_id in seen_samples:
                raise ValueError(f"Duplicate sample_id in input Lance: {sample_id}")
            seen_samples.add(sample_id)
            subject_id = self._normalize_subject_id(row.get("subject_id")).strip()
            if not subject_id:
                raise ValueError(f"Input sample {sample_id} has an empty subject_id")
            try:
                label, label_name = normalize_sex(row.get("sex"))
            except ValueError:
                self.excluded_samples.append({
                    "sample_id": sample_id,
                    "subject_id": subject_id,
                    "reason": "sex_missing_or_invalid",
                })
                continue
            previous = subject_labels.setdefault(subject_id, label)
            if previous != label:
                raise ValueError(f"Inconsistent sex labels within subject {subject_id}")
            if LABEL_MAPPING[label_name] != label:
                raise AssertionError("Internal label normalization mismatch")
            selected.append(index)
        if not selected:
            raise ValueError("Input Lance has no eligible sex-labelled samples")
        return selected

    def _row_to_sample(self, row):
        sample = super()._row_to_sample(row)
        sample["window_index"] = torch.tensor(int(row["window_index"]), dtype=torch.int64)
        return sample


def assert_authoritative_contract(model: torch.nn.Module | None = None) -> None:
    assert POOLING == "mean"
    assert PROBE_LAYER_IDX == 8
    assert math.isclose(PROBE_LAYER_RATIO, 0.75)
    assert FEATURE_DIM == 512
    assert NUM_CLASSES == 2
    assert LABEL_MAPPING == {"female": 0, "male": 1}
    if model is not None:
        if model.pooling != POOLING or model.probe_layer_idx != PROBE_LAYER_IDX:
            raise RuntimeError("Constructed model violates pooling/layer contract")
        if model.head.in_features != FEATURE_DIM or model.head.out_features != NUM_CLASSES:
            raise RuntimeError("Constructed sex head violates [512,2] contract")


def build_meta_zero_shot_model() -> EEGDownstreamClassifier:
    """Create structure on ``meta`` so no random parameter value is materialized."""

    assert_authoritative_contract()
    with torch.device("meta"):
        backbone = EEGPretrainModel(
            input_dim=100,
            d_model=FEATURE_DIM,
            n_heads=8,
            depth=12,
            mlp_ratio=4.0,
            dropout=0.1,
            n_channels=64,
            n_time_patches=10,
            n_bands=5,
            frames_per_patch=5,
            objective="timefreq",
        )
        model = EEGDownstreamClassifier(
            backbone=backbone,
            d_model=FEATURE_DIM,
            num_classes=NUM_CLASSES,
            probe_layer_idx=PROBE_LAYER_IDX,
            freeze_backbone=False,
            use_mlp_head=False,
            head_dropout=0.0,
            pooling=POOLING,
            num_tokens=640,
        )
    assert_authoritative_contract(model)
    return model


def model_state_fingerprint(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def path_content_fingerprint(path: str | Path) -> str:
    """Hash a Lance file or directory without following external paths."""

    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"Input Lance does not exist: {root}")
    files = [root] if root.is_file() else sorted(item for item in root.rglob("*") if item.is_file())
    digest = hashlib.sha256()
    for item in files:
        relative = item.name if root.is_file() else item.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(str(item.stat().st_size).encode("ascii"))
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def sample_id_fingerprint(sample_ids: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for sample_id in sample_ids:
        digest.update(str(sample_id).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _git_info(root: Path) -> dict[str, Any]:
    def command(*parts: str) -> str:
        result = subprocess.run(
            ["git", *parts], cwd=root, check=True, capture_output=True, text=True
        )
        return result.stdout.strip()
    try:
        return {"commit": command("rev-parse", "HEAD"), "dirty": bool(command("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _cuda_allocated_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.memory_allocated(device) / (1024 * 1024))


def run_zero_shot_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    logger: logging.Logger,
    log_every_batches: int,
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    total_rows = len(loader.dataset)
    total_batches = len(loader)
    started = time.monotonic()
    forward_count = 0
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, start=1):
            model_inputs = {
                key: batch[key].to(device, non_blocking=device.type == "cuda")
                for key in (
                    "token_inputs", "token_channel_indices", "token_time_indices", "token_valid_mask"
                )
            }
            logits = model(**model_inputs)
            forward_count += 1
            if logits.ndim != 2 or logits.shape[1] != 2 or not torch.isfinite(logits).all():
                raise RuntimeError("Model logits must be finite with shape [B,2]")
            probabilities = torch.softmax(logits, dim=1)
            predictions = probabilities.argmax(dim=1)
            logits_cpu = logits.detach().cpu()
            probabilities_cpu = probabilities.detach().cpu()
            predictions_cpu = predictions.detach().cpu()
            labels = batch["label"].view(-1)
            for item_index, sample_id in enumerate(batch["sample_id"]):
                sample_id = str(sample_id)
                if sample_id in seen:
                    raise RuntimeError(f"Sample was forwarded more than once: {sample_id}")
                seen.add(sample_id)
                true_label = int(labels[item_index])
                predicted_label = int(predictions_cpu[item_index])
                rows.append({
                    "sample_id": sample_id,
                    "subject_id": str(batch["subject_id"][item_index]),
                    "recording_id": str(batch["recording_id"][item_index]),
                    "window_index": int(batch["window_index"][item_index]),
                    "true_label": true_label,
                    "true_sex": LABEL_TO_SEX[true_label],
                    "raw_logit_female": float(logits_cpu[item_index, 0]),
                    "raw_logit_male": float(logits_cpu[item_index, 1]),
                    "prob_female": float(probabilities_cpu[item_index, 0]),
                    "prob_male": float(probabilities_cpu[item_index, 1]),
                    "predicted_label": predicted_label,
                    "predicted_sex": LABEL_TO_SEX[predicted_label],
                    "correct": predicted_label == true_label,
                })
            processed = len(rows)
            if batch_index == 1 or batch_index == total_batches or batch_index % log_every_batches == 0:
                elapsed = max(time.monotonic() - started, 1e-9)
                throughput = processed / elapsed
                eta = (total_rows - processed) / throughput if throughput > 0 else None
                logger.info(
                    "batch=%d/%d processed_rows=%d total_rows=%d progress_percent=%.2f "
                    "elapsed_seconds=%.3f throughput_rows_per_second=%.3f eta_seconds=%s "
                    "forward_count=%d cuda_allocated_mb=%.3f",
                    batch_index, total_batches, processed, total_rows,
                    100.0 * processed / total_rows, elapsed, throughput,
                    "null" if eta is None else f"{eta:.3f}", forward_count,
                    _cuda_allocated_mb(device),
                )
    if len(rows) != total_rows or len(seen) != total_rows:
        raise RuntimeError("Inference coverage does not equal eligible dataset size")
    return rows, forward_count


def validate_window_rows(rows: Sequence[dict[str, Any]], expected_sample_ids: Sequence[str]) -> None:
    if len(rows) != len(expected_sample_ids):
        raise ValueError("Window prediction count differs from eligible sample count")
    row_ids = [str(row["sample_id"]) for row in rows]
    if len(row_ids) != len(set(row_ids)):
        raise ValueError("window_predictions contains duplicate sample_id values")
    if row_ids != list(expected_sample_ids):
        raise ValueError("Window prediction order/coverage differs from eligible Lance order")
    for row in rows:
        female, male = float(row["prob_female"]), float(row["prob_male"])
        if not math.isfinite(female) or not math.isfinite(male):
            raise ValueError(f"Non-finite probability for {row['sample_id']}")
        if not math.isclose(female + male, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"Probability sum mismatch for {row['sample_id']}")
        predicted = int(np.argmax([female, male]))
        if int(row["predicted_label"]) != predicted:
            raise ValueError(f"Probability argmax mismatch for {row['sample_id']}")
        if str(row["predicted_sex"]) != LABEL_TO_SEX[predicted]:
            raise ValueError(f"Predicted sex mapping mismatch for {row['sample_id']}")
        correct_value = row["correct"]
        if isinstance(correct_value, str):
            normalized_correct = correct_value.strip().lower()
            if normalized_correct not in {"true", "false"}:
                raise ValueError(f"Invalid correctness value for {row['sample_id']}")
            correct_value = normalized_correct == "true"
        if bool(correct_value) != (predicted == int(row["true_label"])):
            raise ValueError(f"Correctness mismatch for {row['sample_id']}")


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_zero_shot_artifacts(
    output_dir: Path,
    window_rows: list[dict[str, Any]],
    expected_sample_ids: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    validate_window_rows(window_rows, expected_sample_ids)
    window_path = output_dir / "window_predictions.csv"
    _write_csv(window_path, window_rows, WINDOW_FIELDS)
    with window_path.open("r", encoding="utf-8", newline="") as handle:
        round_trip = list(csv.DictReader(handle))
    validate_window_rows(round_trip, expected_sample_ids)

    window_metrics = binary_classification_metrics(
        [row["true_label"] for row in window_rows],
        [row["predicted_label"] for row in window_rows],
    )
    window_metrics["sample_count"] = len(window_rows)
    subject_rows = aggregate_subject_predictions(window_rows)
    _write_csv(output_dir / "subject_predictions.csv", subject_rows, SUBJECT_FIELDS)
    probability_metrics = binary_classification_metrics(
        [row["true_label"] for row in subject_rows],
        [row["probability_mean_pred_label"] for row in subject_rows],
    )
    probability_metrics["sample_count"] = len(subject_rows)
    majority_metrics = binary_classification_metrics(
        [row["true_label"] for row in subject_rows],
        [row["majority_vote_pred_label"] for row in subject_rows],
    )
    majority_metrics["sample_count"] = len(subject_rows)
    _write_json(output_dir / "window_metrics.json", window_metrics)
    _write_json(output_dir / "subject_probability_mean_metrics.json", probability_metrics)
    _write_json(output_dir / "subject_majority_vote_metrics.json", majority_metrics)
    return window_metrics, probability_metrics, majority_metrics, subject_rows


def _configure_logging(output_dir: Path, level: str) -> logging.Logger:
    logger = logging.getLogger("neuralbench_sex_zero_shot")
    logger.handlers.clear()
    logger.setLevel(str(level).upper())
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def _inspect_lance(path: str | Path) -> dict[str, Any]:
    if not Path(path).exists():
        raise FileNotFoundError(f"Input Lance does not exist: {path}")
    dataset = lance.dataset(str(path))
    required = set(ZERO_SHOT_COLUMNS)
    missing = sorted(required.difference(dataset.schema.names))
    if missing:
        raise ValueError(f"Input Lance is missing required columns: {missing}")
    return {"version": int(dataset.version), "row_count": int(dataset.count_rows())}


def dry_run(args: argparse.Namespace) -> int:
    """Validate paths/schema/SHA only; do not scan rows or create output."""

    assert_authoritative_contract()
    checkpoint = verify_checkpoint_sha256(args.checkpoint, args.expected_checkpoint_sha256)
    lance_info = _inspect_lance(args.input_lance)
    plan = {
        "dry_run": True,
        "formal_output_created": False,
        "full_lance_rows_read": False,
        "run_mode": RUN_MODE,
        "training_enabled": False,
        "checkpoint": checkpoint,
        "input_lance": {"path": str(Path(args.input_lance).resolve()), **lance_info},
        "device": args.device,
        "batch_size": args.batch_size,
        "pooling": POOLING,
        "probe_layer_idx": PROBE_LAYER_IDX,
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def run(args: argparse.Namespace) -> int:
    if args.dry_run:
        return dry_run(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    error_path = output_dir / "zero_shot_errors.log"
    error_path.write_text("", encoding="utf-8")
    logger = _configure_logging(output_dir, args.log_level)
    try:
        assert_authoritative_contract()
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        logger.info(
            "run_mode=zero_shot training_enabled=false optimizer_created=false "
            "scheduler_created=false backward_called=false checkpoint_selection=false "
            "pooling=mean probe_layer_idx=8 head_source=neuralbench_checkpoint"
        )
        input_before = path_content_fingerprint(args.input_lance)
        lance_info = _inspect_lance(args.input_lance)
        dataset = NeuralBenchSexZeroShotDataset(args.input_lance)
        eligible_metadata = dataset.get_subject_label_rows()
        expected_sample_ids = [str(row["sample_id"]) for row in eligible_metadata]
        all_input_sample_ids = [str(row["sample_id"]) for row in dataset._metadata]
        subject_ids = sorted({str(row["subject_id"]) for row in eligible_metadata})

        model = build_meta_zero_shot_model()
        checkpoint, checkpoint_info, mapping_rows, mapping_summary = load_neuralbench_sex_model_state(
            model, args.checkpoint, args.expected_checkpoint_sha256
        )
        del checkpoint
        write_checkpoint_key_mapping(output_dir / "checkpoint_key_mapping.csv", mapping_rows)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        if any(parameter.requires_grad for parameter in model.parameters()):
            raise RuntimeError("At least one model parameter remains trainable")
        model = model.to(device)
        model.eval()
        if model.training or model.backbone.training or model.head.training:
            raise RuntimeError("Model did not enter eval mode")
        before_forward_fingerprint = model_state_fingerprint(model)

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            drop_last=False,
            pin_memory=device.type == "cuda",
        )
        window_rows, forward_count = run_zero_shot_inference(
            model, loader, device, logger, args.log_every_batches
        )
        after_forward_fingerprint = model_state_fingerprint(model)
        if before_forward_fingerprint != after_forward_fingerprint:
            raise RuntimeError("Model parameter/state fingerprint changed during inference")
        window_metrics, probability_metrics, majority_metrics, subject_rows = write_zero_shot_artifacts(
            output_dir, window_rows, expected_sample_ids
        )
        input_after = path_content_fingerprint(args.input_lance)
        if input_before != input_after:
            raise RuntimeError("Input Lance content fingerprint changed during evaluation")

        issues: list[dict[str, Any]] = []
        if lance_info["row_count"] != 480:
            issues.append({"severity": "warning", "code": "unexpected_input_row_count", "detail": f"expected 480, observed {lance_info['row_count']}"})
        if len(subject_ids) != 8:
            issues.append({"severity": "warning", "code": "unexpected_subject_count", "detail": f"expected 8, observed {len(subject_ids)}"})
        _write_csv(output_dir / "zero_shot_issues.csv", issues, ["severity", "code", "detail"])

        exclusion_counts = dict(Counter(row["reason"] for row in dataset.excluded_samples))
        summary = {
            "evaluation_kind": "external_checkpoint_zero_shot",
            "run_mode": RUN_MODE,
            "target_training_samples": 0,
            "target_validation_samples": 0,
            "target_test_selection": False,
            "target_label_used_for_training": False,
            "target_label_used_for_calibration": False,
            "target_label_used_for_metrics_only": True,
            "head_source": "neuralbench_checkpoint",
            "backbone_source": "neuralbench_checkpoint",
            "eligible_sample_count": len(dataset),
            "excluded_sample_count": len(dataset.excluded_samples),
            "subject_count": len(subject_rows),
            "batch_count": len(loader),
            "forward_count": forward_count,
            "window_metrics": window_metrics,
            "subject_probability_mean_metrics": probability_metrics,
            "subject_majority_vote_metrics": majority_metrics,
            "post_hoc_label_flip_performed": False,
        }
        provenance = {
            **summary,
            "input_lance": {
                "path": str(Path(args.input_lance).resolve()),
                "version": lance_info["version"],
                "row_count": lance_info["row_count"],
                "sample_id_fingerprint_sha256": sample_id_fingerprint(all_input_sample_ids),
                "eligible_sample_id_fingerprint_sha256": sample_id_fingerprint(expected_sample_ids),
                "content_fingerprint_before_sha256": input_before,
                "content_fingerprint_after_sha256": input_after,
                "unmodified": input_before == input_after,
            },
            "checkpoint": {
                **checkpoint_info,
                "source_format": SOURCE_FORMAT,
                "expected_sha256": args.expected_checkpoint_sha256.lower(),
            },
            "key_remap_version": KEY_REMAP_VERSION,
            "checkpoint_mapping": mapping_summary,
            "pooling": POOLING,
            "probe_layer_idx": PROBE_LAYER_IDX,
            "probe_layer_ratio": PROBE_LAYER_RATIO,
            "feature_dim": FEATURE_DIM,
            "num_classes": NUM_CLASSES,
            "label_mapping": LABEL_MAPPING,
            "accepted_sex_encodings": {"female": ["female", "f", "0"], "male": ["male", "m", "1"]},
            "device": str(device),
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "excluded_samples_by_reason": exclusion_counts,
            "git": _git_info(Path(__file__).resolve().parents[2]),
            "model_state_fingerprint_before_forward_sha256": before_forward_fingerprint,
            "model_state_fingerprint_after_forward_sha256": after_forward_fingerprint,
            "model_state_unchanged": before_forward_fingerprint == after_forward_fingerprint,
            "all_parameters_requires_grad_false": True,
            "random_materialized_parameter_count": 0,
            "training_enabled": False,
            "optimizer_created": False,
            "scheduler_created": False,
            "backward_called": False,
            "checkpoint_selection": False,
            "feature_cache_created": False,
            "subject_majority_vote_tie_break": "probability_mean_argmax_then_female_on_exact_probability_tie",
        }
        _write_json(output_dir / "zero_shot_summary.json", summary)
        _write_json(output_dir / "zero_shot_provenance.json", provenance)
        report = f"""# NeuralBench sex full-checkpoint zero-shot report

- evaluation kind: `external_checkpoint_zero_shot`
- checkpoint SHA256: `{checkpoint_info['sha256']}` (strictly matched expected)
- eligible/excluded windows: `{len(dataset)}` / `{len(dataset.excluded_samples)}`
- subjects: `{len(subject_rows)}`
- batches/forward calls: `{len(loader)}` / `{forward_count}`
- pooling/layer: `mean` / zero-based `{PROBE_LAYER_IDX}`
- backbone/head source: `neuralbench_checkpoint` / `neuralbench_checkpoint`
- exact core backbone key/parameter match: `{mapping_summary['core_backbone_key_match_fraction']}` / `{mapping_summary['core_backbone_parameter_match_fraction']}`
- exact probe key/parameter match: `{mapping_summary['probe_key_match_fraction']}` / `{mapping_summary['probe_parameter_match_fraction']}`
- model fingerprint unchanged by inference: `{before_forward_fingerprint == after_forward_fingerprint}`
- target training, validation, calibration, checkpoint selection: `false`
- post-hoc output-column or label flip: not performed

Window accuracy is `{window_metrics['accuracy']:.6f}`. Subject probability-mean
accuracy is `{probability_metrics['accuracy']:.6f}` and subject majority-vote
accuracy is `{majority_metrics['accuracy']:.6f}`. Ground-truth sex labels were
used only after inference to calculate these metrics; they did not affect the
model, output order, threshold, epoch, or class mapping.
"""
        (output_dir / "zero_shot_report.md").write_text(report, encoding="utf-8")
        logger.info(
            "completed eligible_samples=%d excluded_samples=%d subjects=%d forward_count=%d",
            len(dataset), len(dataset.excluded_samples), len(subject_rows), forward_count,
        )
        return 0
    except Exception:
        error_path.write_text(traceback.format_exc(), encoding="utf-8")
        logger.exception("zero-shot evaluation failed")
        raise


def main(argv: Sequence[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
