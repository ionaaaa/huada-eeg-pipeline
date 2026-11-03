"""Shared Dortmund age full-finetune protocol utilities.

The model and Lance contracts remain in the existing 50M modules.  This file
contains only Dortmund-specific split-independent training policy: target
transform, balanced distributed sampling, strict intermediate-layer gradient
scope, metrics, selection, test access guard, and checkpoint validation.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import torch
import torch.distributed as dist
from scipy import stats
from torch.utils.data import Sampler

try:
    from .largeScale_fineTune import EEGDownstreamClassifier
except ImportError:
    from largeScale_fineTune import EEGDownstreamClassifier


SELECTION_ATOL = 1e-6
TARGET_TRANSFORM_VERSION = "train-subject-equal-zscore-ddof0-v1"
SAMPLER_VERSION = "global-subject-condition-balanced-sharded-v1"
MASK_SEMANTICS_VERSION = "token_valid_mask_true_is_valid_v1"
GRADIENT_SCOPE_VERSION = "probe-layer-inclusive-explicit-freeze-v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


@dataclass(frozen=True)
class TargetTransform:
    source_age_mean: float
    source_age_std: float
    train_subject_count: int
    train_subject_manifest_sha256: str
    ddof: int = 0
    target_transform: str = "zscore"
    target_unit: str = "year"
    equal_subject_weight: bool = True
    window_weighted: bool = False
    version: str = TARGET_TRANSFORM_VERSION

    def transform_tensor(self, years: torch.Tensor) -> torch.Tensor:
        return (years - float(self.source_age_mean)) / float(self.source_age_std)

    def inverse_tensor(self, z: torch.Tensor) -> torch.Tensor:
        return z * float(self.source_age_std) + float(self.source_age_mean)

    def transform_numpy(self, years: Sequence[float] | np.ndarray) -> np.ndarray:
        values = np.asarray(years, dtype=np.float64)
        return (values - self.source_age_mean) / self.source_age_std

    def inverse_numpy(self, z: Sequence[float] | np.ndarray) -> np.ndarray:
        values = np.asarray(z, dtype=np.float64)
        return values * self.source_age_std + self.source_age_mean


def build_target_transform(
    manifest_rows: Sequence[dict[str, Any]], manifest_sha256: str
) -> TargetTransform:
    train = [row for row in manifest_rows if str(row["split"]) == "train"]
    if not train:
        raise ValueError("Target transform requires train subjects")
    subject_ids = [str(row["subject_id"]) for row in train]
    if len(subject_ids) != len(set(subject_ids)):
        raise ValueError("Target transform input must contain one row per train subject")
    ages = np.asarray([float(row["age"]) for row in train], dtype=np.float64)
    if not np.isfinite(ages).all():
        raise ValueError("Train subject ages must be finite")
    mean = float(ages.mean())
    std = float(ages.std(ddof=0))
    if not math.isfinite(mean) or not math.isfinite(std) or std <= 1e-12:
        raise ValueError(f"Invalid train-only target statistics: mean={mean}, std={std}")
    return TargetTransform(
        source_age_mean=mean,
        source_age_std=std,
        train_subject_count=len(train),
        train_subject_manifest_sha256=str(manifest_sha256),
    )


class SubjectBalancedDistributedSampler(Sampler[int]):
    """Create one deterministic global sequence and shard positions by rank."""

    def __init__(
        self,
        rows: Sequence[dict[str, Any]],
        *,
        samples_per_epoch: int,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
        per_device_batch_size: int = 1,
    ):
        if not rows:
            raise ValueError("Training sampler requires rows")
        if world_size <= 0 or rank < 0 or rank >= world_size:
            raise ValueError("Invalid rank/world_size")
        if samples_per_epoch <= 0 or per_device_batch_size <= 0:
            raise ValueError("samples_per_epoch and batch size must be positive")
        self.rows = [dict(row) for row in rows]
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.per_device_batch_size = int(per_device_batch_size)
        multiple = self.world_size * self.per_device_batch_size
        self.requested_samples_per_epoch = int(samples_per_epoch)
        self.global_samples_per_epoch = int(math.ceil(samples_per_epoch / multiple) * multiple)
        self.local_samples_per_epoch = self.global_samples_per_epoch // self.world_size
        self.epoch = 0
        grouped: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        self.sample_ids = {}
        for row in self.rows:
            index = int(row["index"])
            subject = str(row["subject_id"])
            condition = (
                str(row.get("eyes_condition", "")).strip().lower()
                .replace("-", "_").replace(" ", "_")
            ) or "unknown"
            grouped[subject][condition].append(index)
            self.sample_ids[index] = str(row["sample_id"])
        self.grouped = {
            subject: {condition: tuple(sorted(indices)) for condition, indices in conditions.items()}
            for subject, conditions in grouped.items()
        }
        self.subjects = tuple(sorted(self.grouped))
        if not self.subjects:
            raise ValueError("No train subjects found")
        self._last_global: list[int] | None = None
        self._last_statistics: dict[str, Any] | None = None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._last_global = None
        self._last_statistics = None

    def _build_global(self) -> list[int]:
        rng = np.random.default_rng(self.seed + 1_000_003 * self.epoch)
        sequence = []
        subject_counts = Counter()
        condition_counts = Counter()
        missing_condition_subjects = []
        subject_order = list(self.subjects)
        while len(sequence) < self.global_samples_per_epoch:
            rng.shuffle(subject_order)
            for subject in subject_order:
                if len(sequence) >= self.global_samples_per_epoch:
                    break
                conditions = self.grouped[subject]
                preferred = [name for name in ("eyes_open", "eyes_closed") if name in conditions]
                if len(preferred) == 2:
                    minimum = min(condition_counts[(subject, name)] for name in preferred)
                    available = [name for name in preferred if condition_counts[(subject, name)] == minimum]
                    condition = available[int(rng.integers(len(available)))]
                elif preferred:
                    condition = preferred[0]
                    if subject not in missing_condition_subjects:
                        missing_condition_subjects.append(subject)
                else:
                    names = sorted(conditions)
                    condition = names[int(rng.integers(len(names)))]
                    if subject not in missing_condition_subjects:
                        missing_condition_subjects.append(subject)
                indices = conditions[condition]
                index = indices[int(rng.integers(len(indices)))]
                sequence.append(index)
                subject_counts[subject] += 1
                condition_counts[(subject, condition)] += 1
        sample_counts = Counter(sequence)
        rank_sequences = [sequence[rank::self.world_size] for rank in range(self.world_size)]
        position_overlap = sum(
            len(set(rank_sequences[left]).intersection(rank_sequences[right]))
            for left in range(self.world_size)
            for right in range(left + 1, self.world_size)
        )
        self._last_statistics = {
            "version": SAMPLER_VERSION,
            "epoch": self.epoch,
            "requested_samples": self.requested_samples_per_epoch,
            "global_samples": len(sequence),
            "local_samples_per_rank": self.local_samples_per_epoch,
            "subject_counts": dict(sorted(subject_counts.items())),
            "subject_condition_counts": {
                f"{subject}|{condition}": count
                for (subject, condition), count in sorted(condition_counts.items())
            },
            "missing_condition_subjects": sorted(missing_condition_subjects),
            "unique_window_count": len(sample_counts),
            "unique_window_fraction": len(sample_counts) / len(sequence),
            "repeated_draw_count": sum(count - 1 for count in sample_counts.values()),
            "cross_rank_window_identity_overlap": position_overlap,
            "ddp_padding_duplicates": 0,
            "rank_lengths": [len(values) for values in rank_sequences],
        }
        self._last_global = sequence
        return sequence

    def global_indices(self) -> list[int]:
        return list(self._last_global if self._last_global is not None else self._build_global())

    def statistics(self) -> dict[str, Any]:
        if self._last_statistics is None:
            self._build_global()
        return dict(self._last_statistics or {})

    def __iter__(self) -> Iterator[int]:
        global_sequence = self.global_indices()
        return iter(global_sequence[self.rank::self.world_size])

    def __len__(self) -> int:
        return self.local_samples_per_epoch


class NoPaddingDistributedEvalSampler(Sampler[int]):
    def __init__(self, size: int, rank: int = 0, world_size: int = 1):
        if size < 0 or world_size <= 0 or rank < 0 or rank >= world_size:
            raise ValueError("Invalid evaluation sampler arguments")
        self.size = int(size)
        self.rank = int(rank)
        self.world_size = int(world_size)

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.rank, self.size, self.world_size))

    def __len__(self) -> int:
        if self.rank >= self.size:
            return 0
        return (self.size - 1 - self.rank) // self.world_size + 1


def validate_prediction_coverage(
    gathered_rows: Sequence[dict[str, Any]], expected_sample_ids: Sequence[str]
) -> None:
    actual = [str(row["sample_id"]) for row in gathered_rows]
    duplicates = sorted(key for key, value in Counter(actual).items() if value > 1)
    expected = [str(value) for value in expected_sample_ids]
    if duplicates:
        raise ValueError(f"Distributed predictions contain duplicates: {duplicates[:10]}")
    if set(actual) != set(expected) or len(actual) != len(expected):
        missing = sorted(set(expected).difference(actual))
        extra = sorted(set(actual).difference(expected))
        raise ValueError(f"Prediction coverage mismatch: missing={missing[:10]}, extra={extra[:10]}")


class StrictDortmundAgeModel(EEGDownstreamClassifier):
    """Existing 50M downstream model with fail-closed all-invalid masking."""

    def pool_valid_tokens(self, token_embeddings, token_valid_mask):
        valid = token_valid_mask.to(device=token_embeddings.device, dtype=token_embeddings.dtype)
        counts = valid.sum(dim=1)
        if bool((counts <= 0).any().item()):
            raise ValueError("All-invalid token mask is not allowed")
        return (token_embeddings * valid.unsqueeze(-1)).sum(dim=1) / counts.unsqueeze(-1)


def configure_probe_gradient_scope(model: StrictDortmundAgeModel, probe_layer_idx: int) -> dict[str, Any]:
    probe = int(probe_layer_idx)
    depth = int(model.backbone.depth)
    if probe < 0 or probe >= depth:
        raise ValueError(f"probe_layer_idx must be in [0,{depth - 1}]")
    model.probe_layer_idx = probe
    for parameter in model.backbone.parameters():
        parameter.requires_grad = True
    frozen_reasons = {}
    for layer_index, layer in enumerate(model.backbone.encoder.encoder.layers):
        if layer_index > probe:
            for name, parameter in layer.named_parameters():
                parameter.requires_grad = False
                frozen_reasons[f"backbone.encoder.encoder.layers.{layer_index}.{name}"] = "after_probe_layer_not_in_forward_graph"
    for name, parameter in model.backbone.head.named_parameters():
        parameter.requires_grad = False
        frozen_reasons[f"backbone.head.{name}"] = "upstream_pretraining_head_not_in_downstream_loss"
    for parameter in model.head.parameters():
        parameter.requires_grad = True
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    frozen = [name for name, parameter in model.named_parameters() if not parameter.requires_grad]
    return {
        "version": GRADIENT_SCOPE_VERSION,
        "probe_layer_idx": probe,
        "executed_encoder_layers": list(range(probe + 1)),
        "explicitly_frozen_encoder_layers": list(range(probe + 1, depth)),
        "upstream_head_frozen": True,
        "trainable_parameter_names": trainable,
        "frozen_parameter_names": frozen,
        "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "frozen_parameter_count": sum(p.numel() for p in model.parameters() if not p.requires_grad),
        "frozen_reasons": frozen_reasons,
    }


def zero_initialize_age_head(model: StrictDortmundAgeModel) -> None:
    if not isinstance(model.head, torch.nn.Linear) or model.head.out_features != 1:
        raise ValueError("Dortmund age protocol requires Linear(feature_dim, 1)")
    torch.nn.init.zeros_(model.head.weight)
    torch.nn.init.zeros_(model.head.bias)


def audit_gradient_coverage(model: torch.nn.Module) -> dict[str, Any]:
    trainable = []
    frozen = []
    unexplained = []
    gradient_covered = []
    nonfinite = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            trainable.append(name)
            if parameter.grad is None:
                unexplained.append(name)
            else:
                gradient_covered.append(name)
                if not torch.isfinite(parameter.grad).all():
                    nonfinite.append(name)
        else:
            frozen.append(name)
    return {
        "trainable_parameter_names": trainable,
        "frozen_parameter_names": frozen,
        "gradient_covered_parameter_names": gradient_covered,
        "unexplained_trainable_without_gradient": unexplained,
        "nonfinite_gradient_parameters": nonfinite,
        "gradient_covered_parameter_count": sum(
            dict(model.named_parameters())[name].numel() for name in gradient_covered
        ),
    }


def _no_decay(name: str, parameter: torch.nn.Parameter) -> bool:
    lowered = name.lower()
    return parameter.ndim <= 1 or name.endswith(".bias") or "norm" in lowered


def build_optimizer(
    model: StrictDortmundAgeModel,
    *,
    backbone_lr: float,
    head_lr: float,
    weight_decay: float,
) -> tuple[torch.optim.AdamW, list[dict[str, Any]]]:
    if backbone_lr <= 0 or head_lr <= 0 or weight_decay < 0:
        raise ValueError("Invalid optimizer hyperparameters")
    buckets = {
        ("backbone", "decay"): [], ("backbone", "no_decay"): [],
        ("age_head", "decay"): [], ("age_head", "no_decay"): [],
    }
    names = {key: [] for key in buckets}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        owner = "age_head" if name.startswith("head.") else "backbone"
        decay = "no_decay" if _no_decay(name, parameter) else "decay"
        buckets[(owner, decay)].append(parameter)
        names[(owner, decay)].append(name)
    groups = []
    report = []
    for owner, decay in (("backbone", "decay"), ("backbone", "no_decay"), ("age_head", "decay"), ("age_head", "no_decay")):
        params = buckets[(owner, decay)]
        if not params:
            continue
        lr = float(head_lr if owner == "age_head" else backbone_lr)
        wd = float(0.0 if decay == "no_decay" else weight_decay)
        group_name = f"{owner}_{decay}"
        groups.append({"params": params, "lr": lr, "weight_decay": wd, "group_name": group_name})
        report.append({
            "group_name": group_name,
            "owner": owner,
            "decay_policy": decay,
            "lr": lr,
            "weight_decay": wd,
            "parameter_count": sum(p.numel() for p in params),
            "tensor_count": len(params),
            "parameter_names": names[(owner, decay)],
        })
    optimizer = torch.optim.AdamW(groups)
    return optimizer, report


def build_linear_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer, total_steps: int, warmup_ratio: float
) -> tuple[torch.optim.lr_scheduler.LambdaLR, dict[str, Any]]:
    if total_steps <= 0 or not 0.0 <= warmup_ratio < 1.0:
        raise ValueError("Invalid scheduler configuration")
    warmup_steps = int(round(total_steps * warmup_ratio))

    def factor(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor), {
        "name": "linear_warmup_cosine_decay",
        "total_steps": int(total_steps),
        "warmup_ratio": float(warmup_ratio),
        "warmup_steps": int(warmup_steps),
    }


def ensure_finite(name: str, value: torch.Tensor | float) -> None:
    tensor = value if torch.is_tensor(value) else torch.tensor(float(value))
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"Non-finite {name}")


def regression_metrics(truth: Sequence[float], prediction: Sequence[float]) -> dict[str, Any]:
    y = np.asarray(truth, dtype=np.float64).reshape(-1)
    p = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if y.shape != p.shape or y.size == 0 or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Regression inputs must be finite, non-empty, and equal length")
    error = p - y
    denominator = float(np.sum((y - y.mean()) ** 2))
    r2_defined = bool(y.size >= 2 and denominator > 0)

    def correlation(kind: str) -> dict[str, Any]:
        if y.size < 2:
            return {"value": None, "defined": False, "reason": "fewer_than_two_samples"}
        if float(y.std()) == 0.0 or float(p.std()) == 0.0:
            return {"value": None, "defined": False, "reason": "zero_variance"}
        result = stats.pearsonr(y, p) if kind == "pearson" else stats.spearmanr(y, p)
        value = float(result.statistic)
        if not math.isfinite(value):
            return {"value": None, "defined": False, "reason": "nonfinite_result"}
        return {"value": value, "defined": True, "reason": None}

    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "r2": float(1.0 - np.sum(error ** 2) / denominator) if r2_defined else None,
        "r2_defined": r2_defined,
        "r2_reason": None if r2_defined else "insufficient_samples_or_zero_truth_variance",
        "pearson_r": correlation("pearson"),
        "spearman_rho": correlation("spearman"),
        "bias": float(np.mean(error)),
        "sample_count": int(y.size),
    }


def aggregate_subject_predictions(window_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in window_rows:
        grouped[str(row["subject_id"])].append(row)
    result = []
    for subject_id in sorted(grouped):
        rows = grouped[subject_id]
        truth = {float(row["true_age"]) for row in rows}
        sexes = {str(row.get("sex", "")) for row in rows}
        if len(truth) != 1 or len(sexes) > 1:
            raise ValueError(f"Inconsistent subject metadata for {subject_id}")
        predictions = np.asarray([float(row["predicted_age"]) for row in rows])
        age = truth.pop()
        result.append({
            "subject_id": subject_id,
            "sex": next(iter(sexes), ""),
            "true_age": age,
            "mean_predicted_age": float(predictions.mean()),
            "median_predicted_age": float(np.median(predictions)),
            "prediction_std": float(predictions.std(ddof=0)),
            "signed_error": float(predictions.mean() - age),
            "absolute_error": float(abs(predictions.mean() - age)),
            "window_count": len(rows),
            "eyes_open_windows": sum(
                str(row.get("eyes_condition", "")).replace("-", "_") == "eyes_open"
                for row in rows
            ),
            "eyes_closed_windows": sum(
                str(row.get("eyes_condition", "")).replace("-", "_") == "eyes_closed"
                for row in rows
            ),
        })
    return result


def metric_bundle(window_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    subject_rows = aggregate_subject_predictions(window_rows)
    truth = [row["true_age"] for row in subject_rows]
    mean_metrics = regression_metrics(truth, [row["mean_predicted_age"] for row in subject_rows])
    median_metrics = regression_metrics(truth, [row["median_predicted_age"] for row in subject_rows])
    window_metrics = regression_metrics(
        [row["true_age"] for row in window_rows],
        [row["predicted_age"] for row in window_rows],
    )
    by_condition = {}
    for condition in ("eyes_open", "eyes_closed"):
        selected = [
            row for row in window_rows
            if str(row.get("eyes_condition", "")).replace("-", "_") == condition
        ]
        by_condition[condition] = None if not selected else regression_metrics(
            [row["true_age"] for row in selected], [row["predicted_age"] for row in selected]
        )
    by_sex = {}
    for sex in sorted({row["sex"] for row in subject_rows}):
        selected = [row for row in subject_rows if row["sex"] == sex]
        by_sex[sex] = regression_metrics(
            [row["true_age"] for row in selected],
            [row["mean_predicted_age"] for row in selected],
        )
    by_age = {}
    for age in sorted({float(row["true_age"]) for row in subject_rows}):
        selected = [row for row in subject_rows if float(row["true_age"]) == age]
        by_age[f"{age:g}"] = regression_metrics(
            [row["true_age"] for row in selected],
            [row["mean_predicted_age"] for row in selected],
        )
    return {
        "subject_mean": mean_metrics,
        "subject_median": median_metrics,
        "window": window_metrics,
        "by_condition": by_condition,
        "by_sex": by_sex,
        "by_age": by_age,
        "eo_ec_combined": window_metrics,
        "subject_rows": subject_rows,
    }


def train_mean_baseline(
    truth_subject_rows: Sequence[dict[str, Any]], train_mean: float
) -> dict[str, Any]:
    truth = [float(row["true_age"]) for row in truth_subject_rows]
    metrics = regression_metrics(truth, [float(train_mean)] * len(truth))
    metrics["baseline_prediction"] = float(train_mean)
    metrics["source"] = "train_subject_equal_mean"
    return metrics


def validation_is_better(
    candidate: dict[str, Any], incumbent: dict[str, Any] | None,
    tolerance: float = SELECTION_ATOL,
) -> bool:
    if tolerance < 0:
        raise ValueError("Selection tolerance must be non-negative")
    if incumbent is None:
        return True
    for key in ("subject_mae", "subject_rmse", "validation_loss"):
        left, right = float(candidate[key]), float(incumbent[key])
        if left < right - tolerance:
            return True
        if left > right + tolerance:
            return False
    return int(candidate["epoch"]) < int(incumbent["epoch"])


class TestAccessGuard:
    def __init__(self):
        self.selection_locked = False
        self.test_dataset_created = False
        self.test_dataloader_created = False
        self.test_rows_materialized = 0
        self.test_forward_count = 0
        self.test_metrics_computed = False

    def lock_selection(self) -> None:
        if self.selection_locked:
            raise RuntimeError("Selection is already locked")
        self.selection_locked = True

    def require_locked(self, action: str) -> None:
        if not self.selection_locked:
            raise RuntimeError(f"Test access before selection lock: {action}")

    def mark_dataset_created(self) -> None:
        self.require_locked("dataset")
        if self.test_dataset_created:
            raise RuntimeError("Test dataset may only be created once")
        self.test_dataset_created = True

    def mark_dataloader_created(self) -> None:
        self.require_locked("dataloader")
        self.test_dataloader_created = True

    def mark_rows(self, count: int) -> None:
        self.require_locked("row materialization")
        self.test_rows_materialized += int(count)

    def mark_forward(self) -> None:
        self.require_locked("forward")
        self.test_forward_count += 1

    def mark_metrics(self) -> None:
        self.require_locked("metrics")
        if self.test_metrics_computed:
            raise RuntimeError("Test metrics may only be computed once")
        self.test_metrics_computed = True

    def state(self) -> dict[str, Any]:
        return dict(self.__dict__)


def atomic_torch_save(payload: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    torch.save(payload, temporary)
    os.replace(temporary, path)


def validate_resume_payload(
    payload: dict[str, Any], *, token_fingerprint: str, manifest_sha256: str,
    world_size: int, resolved_config_sha256: str,
) -> None:
    expected = {
        "source_token_lance_fingerprint": token_fingerprint,
        "split_manifest_sha256": manifest_sha256,
        "world_size": int(world_size),
        "resolved_config_sha256": resolved_config_sha256,
    }
    mismatches = {
        key: {"checkpoint": payload.get(key), "expected": value}
        for key, value in expected.items() if payload.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Resume checkpoint compatibility mismatch: {mismatches}")
    if payload.get("resume_granularity") != "epoch_boundary":
        raise ValueError("Only epoch-boundary resume is supported")


def distributed_context() -> dict[str, Any]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return {
        "distributed_enabled": world_size > 1,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
    }


def initialize_distributed(device_arg: str | None = None) -> tuple[dict[str, Any], torch.device]:
    context = distributed_context()
    if context["distributed_enabled"]:
        use_cuda = torch.cuda.is_available() and device_arg != "cpu"
        backend = "nccl" if use_cuda else "gloo"
        if use_cuda:
            torch.cuda.set_device(context["local_rank"])
            device = torch.device("cuda", context["local_rank"])
        else:
            device = torch.device("cpu")
        dist.init_process_group(backend=backend, init_method="env://")
    else:
        backend = None
        if device_arg:
            device = torch.device(device_arg)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    context.update({"backend": backend, "device": str(device)})
    return context, device


def gather_prediction_rows(local_rows: list[dict[str, Any]], world_size: int) -> list[dict[str, Any]]:
    if world_size == 1:
        return list(local_rows)
    gathered: list[list[dict[str, Any]] | None] = [None] * world_size
    dist.all_gather_object(gathered, local_rows)
    return [row for part in gathered if part is not None for row in part]


def broadcast_object(value: Any, rank: int) -> Any:
    if not dist.is_available() or not dist.is_initialized():
        return value
    values = [value if rank == 0 else None]
    dist.broadcast_object_list(values, src=0)
    return values[0]


def capture_rng_state() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])
