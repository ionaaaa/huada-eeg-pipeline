"""Strict Dortmund-age -> Huada zero-shot shared logic.

The module intentionally contains no optimizer, scheduler, calibration, clipping,
or backward path.  The CLI entry is responsible for filesystem publication.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from scipy import stats

try:
    from .model import EEGPretrainModel
    from .neuralbench_checkpoint import tensor_state_fingerprint
except ImportError:
    from model import EEGPretrainModel
    from neuralbench_checkpoint import tensor_state_fingerprint


CHECKPOINT_KIND = "dortmund_age_50m_full_finetune"
EXPECTED_CHECKPOINT_SHA256 = "3e44d9e7942d20d3d004e191a077ac636eeea770a6d185149591b683c5b3f13e"
EXPECTED_BEST_EPOCH = 20
PROBE_LAYER_IDX = 8
POOLING = "masked_mean"
MODEL_POOLING = "mean"
FEATURE_DIM = 512
SOURCE_AGE_MEAN = 27.5
SOURCE_AGE_STD = 4.218591
TARGET_TRANSFORM = "zscore"
TARGET_UNIT = "year"
MASK_SEMANTICS_VERSION = "token_valid_mask_true_is_valid_v1"
EXPECTED_SUBJECTS = {f"SUB{i}" for i in range(1, 9)}
PROHIBITED_PREDICTION_FIELDS = {
    "prediction", "predicted_age", "predicted_age_years", "predicted_z",
    "calibrated_age", "split", "fold",
}


class ZeroShotValidationError(RuntimeError):
    pass


@dataclass(frozen=True)
class CheckpointContract:
    backbone_tensor_count: int = 154
    backbone_parameter_count: int = 38_194_713
    head_tensor_count: int = 2
    head_parameter_count: int = 513
    head_weight_shape: tuple[int, ...] = (1, 512)
    head_bias_shape: tuple[int, ...] = (1,)
    # ``None`` accepts any non-negative selected epoch.  The original formal
    # checkpoint contract keeps epoch 20 as its default, while callers that
    # consume a newly trained Dortmund checkpoint can opt into metadata-driven
    # validation without weakening the architecture/state-dict checks.
    best_epoch: int | None = EXPECTED_BEST_EPOCH
    probe_layer_idx: int = PROBE_LAYER_IDX
    feature_dim: int = FEATURE_DIM


FORMAL_CHECKPOINT_CONTRACT = CheckpointContract()


class StrictDortmundAgeInferenceModel(torch.nn.Module):
    """Inference-only form of the existing downstream wrapper.

    It preserves the authoritative backbone, intermediate-layer extraction and
    masked-mean semantics while omitting every training-only dependency.
    """

    def __init__(self, backbone: torch.nn.Module, feature_dim: int = FEATURE_DIM):
        super().__init__()
        self.backbone = backbone
        self.probe_layer_idx = PROBE_LAYER_IDX
        self.pooling = MODEL_POOLING
        self.head = torch.nn.Linear(int(feature_dim), 1)

    def forward(self, token_inputs, token_channel_indices, token_time_indices, token_valid_mask):
        embeddings = self.backbone.extract_token_embeddings(
            token_inputs=token_inputs,
            token_channel_indices=token_channel_indices,
            token_time_indices=token_time_indices,
            return_layer_idx=self.probe_layer_idx,
        )
        valid = token_valid_mask.to(device=embeddings.device, dtype=embeddings.dtype)
        counts = valid.sum(dim=1)
        if bool((counts <= 0).any().item()):
            raise ZeroShotValidationError("All-invalid token mask is not allowed")
        pooled = (embeddings * valid.unsqueeze(-1)).sum(dim=1) / counts.unsqueeze(-1)
        return self.head(pooled)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_state_fingerprint(model: torch.nn.Module) -> str:
    return tensor_state_fingerprint(model.state_dict())


def _finite_float(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ZeroShotValidationError(f"{name} must be finite")
    return result


def normalize_nullable_age(value: Any) -> float | None:
    if hasattr(value, "as_py"):
        value = value.as_py()
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return _finite_float(value, "age")


def _metadata_sha(value: Any) -> str | None:
    if isinstance(value, Mapping):
        direct = value.get("sha256") or value.get("vocabulary_sha256")
        if direct:
            return str(direct).lower()
        content = value.get("content")
        if isinstance(content, Mapping):
            direct = content.get("sha256") or content.get("vocabulary_sha256")
            if direct:
                return str(direct).lower()
    return None


def _electrode_vocabulary_sha(value: Any) -> str | None:
    """Prefer the embedded authoritative vocabulary hash over sidecar hash."""
    if isinstance(value, Mapping):
        content = value.get("content")
        if isinstance(content, Mapping) and content.get("sha256"):
            return str(content["sha256"]).lower()
    return _metadata_sha(value)


def _metadata_version(value: Any) -> str | None:
    if not isinstance(value, Mapping):
        return None
    for key in ("version", "profile_version", "name"):
        if value.get(key):
            return str(value[key])
    content = value.get("content")
    return _metadata_version(content) if isinstance(content, Mapping) else None


def build_formal_model() -> StrictDortmundAgeInferenceModel:
    """Build the exact 50M architecture on meta; checkpoint tensors materialize it."""

    with torch.device("meta"):
        backbone = EEGPretrainModel(
            input_dim=100, d_model=512, n_heads=8, depth=12,
            mlp_ratio=4.0, dropout=0.1, n_channels=64,
            n_time_patches=10, n_bands=5, frames_per_patch=5,
            objective="timefreq",
        )
        return StrictDortmundAgeInferenceModel(backbone=backbone, feature_dim=512)


def _audit_state_dict(
    state: Mapping[str, torch.Tensor], expected: Mapping[str, torch.Tensor], label: str,
) -> dict[str, Any]:
    missing = sorted(set(expected).difference(state))
    unexpected = sorted(set(state).difference(expected))
    shape_mismatch = []
    dtype_mismatch = []
    for key in sorted(set(state).intersection(expected)):
        if tuple(state[key].shape) != tuple(expected[key].shape):
            shape_mismatch.append({"key": key, "checkpoint": list(state[key].shape), "model": list(expected[key].shape)})
        if state[key].dtype != expected[key].dtype:
            dtype_mismatch.append({"key": key, "checkpoint": str(state[key].dtype), "model": str(expected[key].dtype)})
    report = {
        "label": label, "tensor_count": len(state),
        "parameter_count": sum(int(t.numel()) for t in state.values()),
        "missing_keys": missing, "unexpected_keys": unexpected,
        "shape_mismatch": shape_mismatch, "dtype_mismatch": dtype_mismatch,
    }
    if missing or unexpected or shape_mismatch or dtype_mismatch:
        raise ZeroShotValidationError(f"Strict {label} state_dict mismatch: {report}")
    return report


def validate_checkpoint_payload(
    payload: Any,
    *,
    model: torch.nn.Module,
    contract: CheckpointContract = FORMAL_CHECKPOINT_CONTRACT,
    expected_source_mean: float | None = SOURCE_AGE_MEAN,
    expected_source_std: float | None = SOURCE_AGE_STD,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ZeroShotValidationError("Checkpoint root must be a mapping")
    resume_keys = {"optimizer_state_dict", "scheduler_state_dict", "amp_scaler_state_dict", "global_step", "sampler_epoch"}
    if resume_keys.intersection(payload):
        raise ZeroShotValidationError("last/resume training checkpoint is forbidden for zero-shot")
    if "state_dict" in payload and "backbone_state_dict" not in payload:
        raise ZeroShotValidationError("NeuralBench/Lightning or head-only checkpoint is not accepted")
    if payload.get("checkpoint_kind") != CHECKPOINT_KIND:
        raise ZeroShotValidationError(f"checkpoint_kind mismatch: {payload.get('checkpoint_kind')!r}")
    best_epoch = int(payload.get("best_epoch", -1))
    if best_epoch < 0:
        raise ZeroShotValidationError("best_epoch must be non-negative")
    if contract.best_epoch is not None and best_epoch != int(contract.best_epoch):
        raise ZeroShotValidationError("best_epoch mismatch")
    architecture = payload.get("model_architecture_config")
    if not isinstance(architecture, Mapping):
        raise ZeroShotValidationError("model_architecture_config is missing")
    expected_arch = {
        "class": "EEGPretrainModel", "d_model": contract.feature_dim,
        "depth": 12, "n_heads": 8, "mlp_ratio": 4.0, "dropout": 0.1,
        "patch_len": 100, "num_tokens": 640, "num_time_patches": 10,
        "n_channels": 64, "probe_layer_idx": contract.probe_layer_idx,
        "pooling": POOLING, "feature_dim": contract.feature_dim,
    }
    mismatched_arch = {
        key: {"checkpoint": architecture.get(key), "expected": value}
        for key, value in expected_arch.items() if architecture.get(key) != value
    }
    if mismatched_arch:
        raise ZeroShotValidationError(f"model architecture mismatch: {mismatched_arch}")
    transform = payload.get("target_transform")
    if not isinstance(transform, Mapping):
        raise ZeroShotValidationError("target_transform is missing")
    if transform.get("target_transform") != TARGET_TRANSFORM or transform.get("target_unit") != TARGET_UNIT:
        raise ZeroShotValidationError("target transform kind/unit mismatch")
    if int(transform.get("ddof", -1)) != 0 or transform.get("equal_subject_weight") is not True:
        raise ZeroShotValidationError("target transform was not train-subject equal ddof=0")
    source_mean = _finite_float(transform.get("source_age_mean"), "source_age_mean")
    source_std = _finite_float(transform.get("source_age_std"), "source_age_std")
    if expected_source_mean is not None and not math.isclose(source_mean, expected_source_mean, rel_tol=0.0, abs_tol=1e-12):
        raise ZeroShotValidationError("source_age_mean mismatch")
    if expected_source_std is not None and not math.isclose(source_std, expected_source_std, rel_tol=0.0, abs_tol=5e-7):
        raise ZeroShotValidationError("source_age_std mismatch")
    if source_std <= 0:
        raise ZeroShotValidationError("source_age_std must be positive")
    if payload.get("token_mask_semantics_version") != MASK_SEMANTICS_VERSION:
        raise ZeroShotValidationError("checkpoint mask semantics mismatch")
    vocabulary_sha = _electrode_vocabulary_sha(payload.get("electrode_vocabulary"))
    canonicalization_sha = _metadata_sha(payload.get("channel_canonicalization"))
    if not vocabulary_sha or not canonicalization_sha:
        raise ZeroShotValidationError("checkpoint vocabulary/canonicalization hash is missing")
    backbone = payload.get("backbone_state_dict")
    head = payload.get("age_probe_state_dict")
    if not isinstance(backbone, Mapping) or not isinstance(head, Mapping):
        raise ZeroShotValidationError("full backbone and age probe state_dicts are required")
    backbone_report = _audit_state_dict(backbone, model.backbone.state_dict(), "backbone")
    head_report = _audit_state_dict(head, model.head.state_dict(), "age_head")
    expected_counts = {
        "backbone_tensor_count": (backbone_report["tensor_count"], contract.backbone_tensor_count),
        "backbone_parameter_count": (backbone_report["parameter_count"], contract.backbone_parameter_count),
        "head_tensor_count": (head_report["tensor_count"], contract.head_tensor_count),
        "head_parameter_count": (head_report["parameter_count"], contract.head_parameter_count),
    }
    bad_counts = {key: {"actual": actual, "expected": expected} for key, (actual, expected) in expected_counts.items() if actual != expected}
    if bad_counts:
        raise ZeroShotValidationError(f"checkpoint tensor/parameter counts mismatch: {bad_counts}")
    if tuple(head["weight"].shape) != contract.head_weight_shape or tuple(head["bias"].shape) != contract.head_bias_shape:
        raise ZeroShotValidationError("age head must be Linear(512,1)")
    return {
        "checkpoint_kind": CHECKPOINT_KIND, "best_epoch": best_epoch,
        "backbone": backbone_report, "age_head": head_report,
        "probe_layer_idx": int(architecture["probe_layer_idx"]),
        "pooling": architecture["pooling"], "feature_dim": int(architecture["feature_dim"]),
        "source_age_mean": source_mean, "source_age_std": source_std,
        "target_transform": TARGET_TRANSFORM, "target_unit": TARGET_UNIT,
        "electrode_vocabulary_sha256": vocabulary_sha,
        "electrode_vocabulary_version": _metadata_version(payload.get("electrode_vocabulary")) or f"sha256:{vocabulary_sha}",
        "electrode_vocabulary_sidecar_sha256": _metadata_sha(payload.get("electrode_vocabulary")),
        "channel_canonicalization_sha256": canonicalization_sha,
        "channel_canonicalization_version": _metadata_version(payload.get("channel_canonicalization")) or f"sha256:{canonicalization_sha}",
        "token_mask_semantics_version": MASK_SEMANTICS_VERSION,
        "missing_keys": 0, "unexpected_keys": 0, "shape_mismatch": 0, "dtype_mismatch": 0,
    }


def load_strict_checkpoint(
    checkpoint: str | Path,
    expected_sha256: str | None,
    device: torch.device,
    *,
    model_factory: Callable[[], torch.nn.Module] = build_formal_model,
    contract: CheckpointContract = FORMAL_CHECKPOINT_CONTRACT,
    expected_source_mean: float | None = SOURCE_AGE_MEAN,
    expected_source_std: float | None = SOURCE_AGE_STD,
) -> tuple[torch.nn.Module, dict[str, Any], Mapping[str, Any]]:
    path = Path(checkpoint)
    if path.name == "last_training_checkpoint.pt":
        raise ZeroShotValidationError("last_training_checkpoint.pt is forbidden")
    actual_sha = sha256_file(path)
    if expected_sha256 is not None and actual_sha.lower() != str(expected_sha256).lower():
        raise ZeroShotValidationError(f"checkpoint SHA256 mismatch: {actual_sha}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise ZeroShotValidationError("PyTorch weights_only=True support is required") from exc
    model = model_factory()
    report = validate_checkpoint_payload(
        payload,
        model=model,
        contract=contract,
        expected_source_mean=expected_source_mean,
        expected_source_std=expected_source_std,
    )
    model.backbone.load_state_dict(payload["backbone_state_dict"], strict=True, assign=True)
    model.head.load_state_dict(payload["age_probe_state_dict"], strict=True, assign=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.to(device).eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ZeroShotValidationError("Model still has trainable parameters")
    report.update({
        "checkpoint_path": str(path.resolve()), "checkpoint_sha256": actual_sha,
        "trainable_parameter_count": 0, "requires_grad_false": True,
        "optimizer_created": False, "scheduler_created": False, "backward_called": False,
    })
    return model, report, payload


def validate_input_compatibility(
    *, checkpoint_report: Mapping[str, Any], input_contract: Mapping[str, Any]
) -> dict[str, Any]:
    vocab = str(input_contract.get("electrode_vocabulary_sha256", "")).lower()
    canonical = str(input_contract.get("channel_canonicalization_sha256", "")).lower()
    mask = str(input_contract.get("token_mask_semantics_version", ""))
    if vocab != checkpoint_report["electrode_vocabulary_sha256"]:
        raise ZeroShotValidationError("Huada token vocabulary is incompatible with checkpoint")
    if canonical != checkpoint_report["channel_canonicalization_sha256"]:
        raise ZeroShotValidationError("Huada channel canonicalization is incompatible with checkpoint")
    if mask != checkpoint_report["token_mask_semantics_version"]:
        raise ZeroShotValidationError("Huada token mask semantics is incompatible with checkpoint")
    return {
        "electrode_vocabulary_compatible": True,
        "channel_canonicalization_compatible": True,
        "token_mask_semantics_compatible": True,
        "huada_slot_mapping_used": True,
        "dortmund_source_channel_order_used": False,
    }


def label_blind_collate(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("Empty batch")
    tensor_keys = ("token_inputs", "token_channel_indices", "token_time_indices", "token_valid_mask")
    result = {key: torch.stack([sample[key] for sample in samples]) for key in tensor_keys}
    for key in ("sample_id", "subject_id", "recording_id"):
        result[key] = [str(sample.get(key, "")) for sample in samples]
    result["window_index"] = [int(sample.get("window_index", 0)) for sample in samples]
    result["true_age_years"] = [normalize_nullable_age(sample.get("true_age_years")) for sample in samples]
    return result


def validate_sample_batch(batch: Mapping[str, Any]) -> None:
    tokens = batch["token_inputs"]
    channel = batch["token_channel_indices"]
    time_index = batch["token_time_indices"]
    mask = batch["token_valid_mask"]
    if tokens.ndim != 3 or tuple(tokens.shape[1:]) != (640, 100) or tokens.dtype != torch.float32:
        raise ZeroShotValidationError(f"token shape/dtype mismatch: {tokens.shape} {tokens.dtype}")
    if channel.shape != tokens.shape[:2] or channel.dtype != torch.int64:
        raise ZeroShotValidationError("token channel indices mismatch")
    if time_index.shape != tokens.shape[:2] or time_index.dtype != torch.int64:
        raise ZeroShotValidationError("token time indices mismatch")
    if mask.shape != tokens.shape[:2] or mask.dtype not in (torch.bool, torch.float32):
        raise ZeroShotValidationError("token_valid_mask mismatch")
    if not torch.isfinite(tokens).all() or not torch.isfinite(mask.float()).all():
        raise ZeroShotValidationError("Non-finite token input/mask")
    if bool((mask.float().sum(dim=1) <= 0).any().item()):
        raise ZeroShotValidationError("All-invalid input row")
    if channel.min().item() < 0 or channel.max().item() >= 64:
        raise ZeroShotValidationError("Invalid FBMAE model slot index")


def infer_label_blind(
    model: torch.nn.Module,
    batches: Sequence[Mapping[str, Any]] | Any,
    *, device: torch.device, source_mean: float, source_std: float,
    checkpoint_sha256: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    model.eval()
    with torch.inference_mode():
        for batch in batches:
            validate_sample_batch(batch)
            # Explicit allow-list: labels/ages can never enter model.forward.
            inputs = {
                "token_inputs": batch["token_inputs"].to(device),
                "token_channel_indices": batch["token_channel_indices"].to(device),
                "token_time_indices": batch["token_time_indices"].to(device),
                "token_valid_mask": batch["token_valid_mask"].to(device),
            }
            predicted_z = model(**inputs).reshape(-1).float().cpu()
            if not torch.isfinite(predicted_z).all():
                raise ZeroShotValidationError("Non-finite model prediction")
            predicted_age = predicted_z * float(source_std) + float(source_mean)
            for index in range(predicted_z.numel()):
                age = normalize_nullable_age(batch["true_age_years"][index])
                rows.append({
                    "sample_id": str(batch["sample_id"][index]),
                    "subject_id": str(batch["subject_id"][index]),
                    "recording_id": str(batch["recording_id"][index]),
                    "window_index": int(batch["window_index"][index]),
                    "label_available": age is not None,
                    "metrics_eligible": age is not None,
                    "true_age_years": age,
                    "predicted_z": float(predicted_z[index]),
                    "predicted_age_years": float(predicted_age[index]),
                    "num_valid_tokens": int(batch["token_valid_mask"][index].float().sum().item()),
                    "checkpoint_sha256": checkpoint_sha256,
                })
    return finalize_prediction_rows(rows, source_mean=source_mean, source_std=source_std)


def finalize_prediction_rows(
    rows: Sequence[Mapping[str, Any]], *, source_mean: float, source_std: float,
) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result = []
    for source in rows:
        row = dict(source)
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id or sample_id in seen:
            raise ZeroShotValidationError(f"Empty/duplicate sample_id: {sample_id!r}")
        seen.add(sample_id)
        z = _finite_float(row["predicted_z"], "predicted_z")
        expected = z * source_std + source_mean
        predicted = _finite_float(row["predicted_age_years"], "predicted_age_years")
        if not math.isclose(predicted, expected, rel_tol=1e-6, abs_tol=1e-5):
            raise ZeroShotValidationError("z -> years invariant failed")
        age = normalize_nullable_age(row.get("true_age_years"))
        row["label_available"] = age is not None
        row["metrics_eligible"] = age is not None
        if age is None:
            row.update({"true_age_years": None, "signed_error_years": None, "absolute_error_years": None, "squared_error_years": None})
        else:
            error = predicted - age
            row.update({"true_age_years": age, "signed_error_years": error, "absolute_error_years": abs(error), "squared_error_years": error * error})
        result.append(row)
    return result


def regression_metrics(truth: Sequence[float], prediction: Sequence[float]) -> dict[str, Any]:
    y = np.asarray(truth, dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    if y.size == 0 or y.shape != p.shape or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ZeroShotValidationError("Metric inputs must be finite, non-empty, equal shaped")
    error = p - y
    denominator = float(np.sum((y - y.mean()) ** 2))
    r2_defined = y.size >= 2 and denominator > 0
    def correlation(kind: str) -> dict[str, Any]:
        if y.size < 2:
            return {"value": None, "defined": False, "reason": "fewer_than_two_samples"}
        if float(y.std()) == 0.0 or float(p.std()) == 0.0:
            return {"value": None, "defined": False, "reason": "zero_variance"}
        value = float((stats.pearsonr if kind == "pearson" else stats.spearmanr)(y, p).statistic)
        return {"value": value, "defined": True, "reason": None} if math.isfinite(value) else {"value": None, "defined": False, "reason": "nonfinite_result"}
    return {
        "mae": float(np.mean(np.abs(error))), "rmse": float(np.sqrt(np.mean(error ** 2))),
        "r2": float(1 - np.sum(error ** 2) / denominator) if r2_defined else None,
        "r2_defined": bool(r2_defined), "r2_reason": None if r2_defined else "insufficient_samples_or_zero_truth_variance",
        "pearson_r": correlation("pearson"), "spearman_rho": correlation("spearman"),
        "bias": float(np.mean(error)), "sample_count": int(y.size),
    }


def aggregate_subjects(window_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in window_rows:
        groups[str(row["subject_id"])].append(row)
    output = []
    for subject_id in sorted(groups, key=lambda x: (int(x[3:]) if x.startswith("SUB") and x[3:].isdigit() else 10**9, x)):
        rows = groups[subject_id]
        ages = {normalize_nullable_age(row.get("true_age_years")) for row in rows}
        if len(ages) != 1:
            raise ZeroShotValidationError(f"Inconsistent age for {subject_id}")
        age = ages.pop()
        predictions = np.asarray([float(row["predicted_age_years"]) for row in rows])
        z = np.asarray([float(row["predicted_z"]) for row in rows])
        mean_prediction = float(predictions.mean())
        output.append({
            "subject_id": subject_id, "true_age": age, "window_count": len(rows),
            "mean_predicted_age": mean_prediction, "median_predicted_age": float(np.median(predictions)),
            "mean_predicted_z": float(z.mean()), "prediction_std": float(predictions.std(ddof=0)),
            "prediction_min": float(predictions.min()), "prediction_max": float(predictions.max()),
            "signed_error": None if age is None else mean_prediction - age,
            "absolute_error": None if age is None else abs(mean_prediction - age),
            "included_in_metrics": age is not None,
        })
    return output


def compute_metric_bundle(
    window_rows: Sequence[Mapping[str, Any]], *, source_mean: float = SOURCE_AGE_MEAN,
) -> dict[str, Any]:
    subject_all = aggregate_subjects(window_rows)
    subject_labeled = [row for row in subject_all if row["included_in_metrics"]]
    labeled_windows = [row for row in window_rows if row["metrics_eligible"]]
    truth = [float(row["true_age"]) for row in subject_labeled]
    subject_mean = regression_metrics(truth, [row["mean_predicted_age"] for row in subject_labeled])
    subject_median = regression_metrics(truth, [row["median_predicted_age"] for row in subject_labeled])
    window = regression_metrics(
        [float(row["true_age_years"]) for row in labeled_windows],
        [float(row["predicted_age_years"]) for row in labeled_windows],
    )
    window["level"] = "window_auxiliary_non_independent"
    source_mean = _finite_float(source_mean, "source_mean")
    baseline = regression_metrics(truth, [source_mean] * len(truth))
    baseline.update({"baseline_prediction": source_mean, "source": "dortmund_train_subject_mean"})
    delta_mae = subject_mean["mae"] - baseline["mae"]
    comparison = {
        "delta_MAE": delta_mae,
        "relative_MAE_improvement": None if baseline["mae"] == 0 else -delta_mae / baseline["mae"],
        "delta_RMSE": subject_mean["rmse"] - baseline["rmse"],
        "model_better_than_source_mean_baseline": subject_mean["mae"] < baseline["mae"],
    }
    return {
        "subject_rows_all": subject_all, "subject_rows_labeled": subject_labeled,
        "subject_mean_metrics": subject_mean, "subject_median_metrics": subject_median,
        "window_metrics": window, "source_mean_baseline_metrics": baseline,
        "baseline_comparison": comparison,
    }


def prediction_range_diagnostics(window_rows: Sequence[Mapping[str, Any]], subject_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    labeled_subjects = [row for row in subject_rows if row["included_in_metrics"]]
    truth = np.asarray([row["true_age"] for row in labeled_subjects], dtype=float)
    window_pred = np.asarray([row["predicted_age_years"] for row in window_rows], dtype=float)
    mean_pred = np.asarray([row["mean_predicted_age"] for row in labeled_subjects], dtype=float)
    median_pred = np.asarray([row["median_predicted_age"] for row in labeled_subjects], dtype=float)
    slope, intercept = (None, None)
    if truth.size >= 2 and float(truth.std()) > 0:
        slope, intercept = [float(v) for v in np.polyfit(truth, mean_pred, 1)]
    true_range = float(truth.max() - truth.min())
    return {
        "diagnostic_only": True, "used_for_calibration": False,
        "small_sample_warning": f"Only {truth.size} labeled subjects; slope/intercept are descriptive only.",
        "true_subject_age_range": [float(truth.min()), float(truth.max())],
        "predicted_window_age_range": [float(window_pred.min()), float(window_pred.max())],
        "predicted_subject_mean_age_range": [float(mean_pred.min()), float(mean_pred.max())],
        "predicted_subject_median_age_range": [float(median_pred.min()), float(median_pred.max())],
        "source_train_age_range": [20.0, 35.0], "huada_labeled_age_range": [22.0, 31.0],
        "fraction_predictions_within_source_range": float(np.mean((window_pred >= 20) & (window_pred <= 35))),
        "fraction_predictions_within_huada_range": float(np.mean((window_pred >= 22) & (window_pred <= 31))),
        "prediction_range_over_true_range": None if true_range == 0 else float(np.ptp(mean_pred) / true_range),
        "mean_bias": float(np.mean(mean_pred - truth)), "slope": slope, "intercept": intercept,
    }


def validate_coverage(rows: Sequence[Mapping[str, Any]], expected_sample_ids: Sequence[str]) -> dict[str, Any]:
    actual = [str(row["sample_id"]) for row in rows]
    duplicates = sorted(key for key, count in Counter(actual).items() if count > 1)
    missing = sorted(set(expected_sample_ids).difference(actual))
    extra = sorted(set(actual).difference(expected_sample_ids))
    subjects = {str(row["subject_id"]) for row in rows}
    if duplicates or missing or extra or subjects != EXPECTED_SUBJECTS:
        raise ZeroShotValidationError(f"Coverage failure duplicates={duplicates} missing={missing} extra={extra} subjects={sorted(subjects)}")
    return {"sample_id_unique": True, "missing": [], "extra": [], "subject_ids": sorted(subjects), "subject_count": 8, "row_count": len(rows)}


def recompute_from_csv(
    path: str | Path, source_mean: float, source_std: float,
    expected_sample_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        raw = list(csv.DictReader(handle))
    rows = []
    for row in raw:
        rows.append({
            **row,
            "window_index": int(row["window_index"]),
            "true_age_years": normalize_nullable_age(row.get("true_age_years")),
            "predicted_z": float(row["predicted_z"]),
            "predicted_age_years": float(row["predicted_age_years"]),
            "num_valid_tokens": int(row["num_valid_tokens"]),
        })
    finalized = finalize_prediction_rows(rows, source_mean=source_mean, source_std=source_std)
    bundle = compute_metric_bundle(finalized, source_mean=source_mean)
    coverage = None if expected_sample_ids is None else validate_coverage(finalized, expected_sample_ids)
    return {
        "row_count": len(finalized), "sample_ids": [row["sample_id"] for row in finalized],
        "coverage": coverage, "metrics": bundle, "z_to_years_recomputed": True,
        "sub5_excluded_from_metrics": all(row["subject_id"] != "SUB5" for row in bundle["subject_rows_labeled"]),
    }


def fingerprint_model(model: torch.nn.Module) -> dict[str, Any]:
    return {"sha256": model_state_fingerprint(model), "tensor_count": len(model.state_dict()), "parameter_count": sum(p.numel() for p in model.parameters())}


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            raise ZeroShotValidationError("NaN/Inf cannot be written to JSON")
        return float(value)
    if isinstance(value, (int, np.integer, bool)) or value is None or isinstance(value, str):
        return value
    return str(value)
