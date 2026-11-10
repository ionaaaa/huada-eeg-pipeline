"""Zero-initialized age-probe CV7 on frozen Huada native representations.

The portable LaBraM/REVE backbone is evaluated exactly once to build one
label-free mean-representation cache.  Every fold/candidate then trains only a
fresh Linear(feature_dim, 1) whose weight and bias are exactly zero.  Candidate
selection uses one validation subject; all seven selections are locked before
any test subject is evaluated once.
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
import traceback
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from scripts.checkpoints.portable_age_checkpoint import (
    load_portable_age_checkpoint,
    sha256_file,
    tensor_state_fingerprint,
)
from scripts.downstream_50m import age_cv7
from scripts.downstream_50m.dortmund_huada_zero_shot import regression_metrics

from .contracts import load_huada_target_contract
from .portable_age_backend import build_native_age_backend
from .preprocessing import iter_native_windows
from .run_huada_portable_age_zero_shot import (
    EXPECTED_SUBJECT_AGES,
    NativeZeroShotError,
    _cache_snapshot,
    _json_read,
    _json_write,
    _regular_file,
    _require_sha256,
    _validate_dynamic_parity,
    _validate_scaler,
    _validate_sidecar,
    load_native_cache_set,
    load_subject_age_manifest,
)


RUN_MODE = "huada_portable_age_zero_init_probe_cv7"
HEAD_CHECKPOINT_KIND = "huada_portable_age_zero_init_probe_head_only"
FEATURE_CACHE_SCHEMA = "huada_native_portable_mean_feature_cache"
FEATURE_CACHE_VERSION = 1
SELECTION_TOLERANCE = 1e-9
ELIGIBLE_SUBJECTS = frozenset(EXPECTED_SUBJECT_AGES).difference({"SUB5"})
OOF_FIELDS = [
    "fold", "sample_id", "subject_id", "recording_id", "window_index",
    "true_age", "true_z_source", "predicted_z", "predicted_age",
    "signed_error", "absolute_error", "squared_error", "num_valid_tokens",
    "test_is_outside_train_age_range", "checkpoint_path",
]
SUBJECT_FIELDS = [
    "subject_id", "true_age", "window_count", "mean_prediction",
    "median_prediction", "prediction_std", "prediction_min", "prediction_max",
    "signed_error", "absolute_error", "test_fold", "checkpoint_path",
]


class ZeroInitCV7Error(RuntimeError):
    pass


def _log(message: str) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{timestamp}] {message}", flush=True)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _csv_write(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})
    os.replace(temporary, path)


def _git_info(root: Path) -> dict[str, Any]:
    import subprocess

    def run(*args: str) -> str | None:
        result = subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else None

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": None if status is None else bool(status),
        "dirty_files": [] if not status else status.splitlines(),
    }


def _critical_code_sha256(project_root: Path) -> dict[str, str]:
    names = (
        "scripts/native_backends/run_huada_portable_age_zero_init_probe_cv7.py",
        "scripts/native_backends/portable_age_backend.py",
        "scripts/native_backends/preprocessing.py",
        "scripts/downstream_50m/age_cv7.py",
        "scripts/checkpoints/portable_age_checkpoint.py",
    )
    return {name: sha256_file(project_root / name) for name in names}


def fixed_probe_candidate(head_lr: float, weight_decay: float) -> dict[str, Any]:
    if not math.isfinite(head_lr) or head_lr <= 0:
        raise ZeroInitCV7Error("head_lr must be finite and positive")
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ZeroInitCV7Error("weight_decay must be finite and non-negative")
    normalized = {
        "head_lr": float(head_lr),
        "weight_decay": float(weight_decay),
        "pooling": "mean",
    }
    return {
        "candidate_id": f"zero_init_lr_{head_lr:g}_wd_{weight_decay:g}",
        **normalized,
        "candidate_sha256": _json_sha256(normalized),
    }


def build_zero_probe(feature_dim: int, device: torch.device) -> nn.Linear:
    if type(feature_dim) is not int or feature_dim <= 0:
        raise ZeroInitCV7Error("feature_dim must be a positive integer")
    probe = nn.Linear(feature_dim, 1, bias=True, device=device, dtype=torch.float32)
    with torch.no_grad():
        probe.weight.zero_()
        probe.bias.zero_()
    if torch.count_nonzero(probe.weight).item() or torch.count_nonzero(probe.bias).item():
        raise ZeroInitCV7Error("probe zero initialization failed")
    return probe


def probe_state(probe: nn.Linear) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in probe.state_dict().items()}


def probe_fingerprint(probe: nn.Linear) -> str:
    return tensor_state_fingerprint(probe_state(probe))


def candidate_is_better(candidate: Mapping[str, Any], incumbent: Mapping[str, Any] | None) -> bool:
    if incumbent is None:
        return True
    # Match the established FBMAE source-probe transfer CV7: validation
    # subject-mean MAE, validation-window MAE, z-MSE, then earlier epoch.
    for key in ("subject_mae", "window_mae", "loss"):
        left, right = float(candidate[key]), float(incumbent[key])
        if left < right - SELECTION_TOLERANCE:
            return True
        if left > right + SELECTION_TOLERANCE:
            return False
    return int(candidate["epoch"]) < int(incumbent["epoch"])


def _features_fingerprint(array: np.ndarray) -> str:
    value = np.asarray(array, dtype=np.float32, order="C")
    digest = hashlib.sha256()
    digest.update(b"torch.float32")
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _validate_zero_shot_reference(
    root: str | Path,
    *,
    family: str,
    portable_sha256: str,
    source_sha256: str,
) -> dict[str, Any]:
    directory = Path(root)
    if directory.is_symlink() or not directory.is_dir() or not (directory / "_SUCCESS").is_file():
        raise ZeroInitCV7Error(f"strict zero-shot reference is incomplete: {directory}")
    summary = _json_read(directory / "zero_shot_summary.json", "strict zero-shot summary")
    provenance = _json_read(directory / "zero_shot_provenance.json", "strict zero-shot provenance")
    if (
        summary.get("status") != "passed"
        or summary.get("model_family") != family
        or summary.get("training_performed") is not False
        or summary.get("calibration_performed") is not False
        or summary.get("prediction_clipping_performed") is not False
        or provenance.get("evaluation_kind") != "dortmund_portable_age_strict_zero_shot"
        or provenance.get("model_family") != family
        or provenance.get("portable_checkpoint_sha256") != portable_sha256
        or provenance.get("source_checkpoint_sha256") != source_sha256
    ):
        raise ZeroInitCV7Error("strict zero-shot reference provenance mismatch")
    metrics = _json_read(directory / "subject_mean_metrics.json", "strict zero-shot metrics")
    if metrics != summary.get("subject_mean_metrics"):
        raise ZeroInitCV7Error("strict zero-shot reference metric mismatch")
    return {"directory": str(directory.resolve()), "metrics": metrics}


def _feature_cache_identity(
    *, family: str, portable_sha256: str, backbone_fingerprint: str,
    contract: Mapping[str, Any], contract_file_sha256: str,
    cache_snapshot: Mapping[str, Any], feature_dim: int, token_count: int,
) -> dict[str, Any]:
    return {
        "schema_name": FEATURE_CACHE_SCHEMA,
        "schema_version": FEATURE_CACHE_VERSION,
        "model_family": family,
        "portable_checkpoint_sha256": portable_sha256,
        "backbone_state_fingerprint": backbone_fingerprint,
        "target_contract_sha256": contract["contract_sha256"],
        "target_contract_file_sha256": contract_file_sha256,
        "native_cache_snapshot_sha256": _json_sha256(cache_snapshot),
        "representation": {
            "pooling": "mean", "feature_dim": feature_dim,
            "token_count": token_count, "dtype": "float32",
        },
        "window": {"samples": 400, "seconds": 2.0, "overlap_samples": 0},
        "labels_in_cache": False,
    }


def _validate_feature_cache(path: Path, identity: Mapping[str, Any]) -> dict[str, Any]:
    if path.is_symlink() or not path.is_dir() or not (path / "_SUCCESS").is_file():
        raise ZeroInitCV7Error(f"feature cache is incomplete: {path}")
    manifest = _json_read(path / "manifest.json", "feature cache manifest")
    if manifest.get("identity") != dict(identity):
        raise ZeroInitCV7Error("feature cache identity mismatch")
    key = _json_sha256(identity)
    if manifest.get("feature_cache_key") != key or (path / "_SUCCESS").read_text().strip() != key:
        raise ZeroInitCV7Error("feature cache key/success marker mismatch")
    features_path = _regular_file(path / "features.float32.npy", "feature cache tensor")
    samples_path = _regular_file(path / "samples.json", "feature cache samples")
    if sha256_file(features_path) != manifest.get("features_file_sha256"):
        raise ZeroInitCV7Error("feature cache file SHA mismatch")
    if sha256_file(samples_path) != manifest.get("samples_file_sha256"):
        raise ZeroInitCV7Error("feature cache sample metadata SHA mismatch")
    samples = _json_read(samples_path, "feature cache samples")
    if type(samples) is not list or any(type(row) is not dict for row in samples):
        raise ZeroInitCV7Error("feature cache samples must be a JSON list of objects")
    features = np.load(features_path, mmap_mode="r", allow_pickle=False)
    expected_dim = int(identity["representation"]["feature_dim"])
    if (
        features.dtype != np.float32 or features.ndim != 2
        or features.shape != (len(samples), expected_dim) or not np.isfinite(features).all()
    ):
        raise ZeroInitCV7Error("feature cache tensor contract mismatch")
    if _features_fingerprint(features) != manifest.get("feature_fingerprint"):
        raise ZeroInitCV7Error("feature cache fingerprint mismatch")
    if (
        manifest.get("labels_in_cache") is not False
        or manifest.get("row_count") != len(samples)
        or manifest.get("feature_dim") != expected_dim
        or manifest.get("dtype") != "float32"
    ):
        raise ZeroInitCV7Error("feature cache manifest contract mismatch")
    expected_sample_keys = {
        "sample_id", "subject_id", "recording_id", "window_index", "num_valid_tokens"
    }
    forbidden = {"age", "label", "true_age", "sex"}
    expected_tokens = int(identity["representation"]["token_count"])
    for row in samples:
        if forbidden.intersection(row):
            raise ZeroInitCV7Error("feature cache contains labels")
        if set(row) != expected_sample_keys:
            raise ZeroInitCV7Error("feature cache sample metadata keys are not canonical")
        subject = row["subject_id"]
        if (
            type(subject) is not str
            or subject not in EXPECTED_SUBJECT_AGES
            or row["recording_id"] != subject
            or type(row["window_index"]) is not int
            or row["window_index"] < 0
            or row["sample_id"] != f"{subject}_W{row['window_index']:06d}"
            or row["num_valid_tokens"] != expected_tokens
        ):
            raise ZeroInitCV7Error("feature cache sample metadata contract mismatch")
    sample_ids = [row.get("sample_id") for row in samples]
    if len(sample_ids) != len(set(sample_ids)) or set(row.get("subject_id") for row in samples) != set(EXPECTED_SUBJECT_AGES):
        raise ZeroInitCV7Error("feature cache sample coverage mismatch")
    return {"path": str(path.resolve()), "manifest": manifest, "samples": samples, "features": features}


def ensure_feature_cache(
    *, root: Path, policy: str, family: str, checkpoint: Any,
    contract: Mapping[str, Any], contract_file_sha256: str,
    cache_snapshot: Mapping[str, Any], recordings: Mapping[str, np.ndarray],
    device: torch.device, batch_size: int,
) -> dict[str, Any]:
    feature_dim = int(checkpoint.payload["representation_contract"]["feature_dim"])
    token_count = 129 if family == "labram" else 128
    identity = _feature_cache_identity(
        family=family, portable_sha256=checkpoint.file_sha256,
        backbone_fingerprint=checkpoint.payload["integrity"]["backbone_state_fingerprint"],
        contract=contract, contract_file_sha256=contract_file_sha256,
        cache_snapshot=cache_snapshot, feature_dim=feature_dim, token_count=token_count,
    )
    key = _json_sha256(identity)
    target = root / key
    if target.exists():
        _log(f"validating and reusing feature cache key={key}")
        result = _validate_feature_cache(target, identity)
        result["reused"] = True
        return result
    if policy == "require":
        raise ZeroInitCV7Error(f"required feature cache is absent: {target}")
    if not root.is_dir():
        raise ZeroInitCV7Error(f"feature cache root must already exist: {root}")
    temporary = root / f".{key}.tmp.{uuid.uuid4().hex}"
    temporary.mkdir()
    _log(f"building label-free feature cache key={key} family={family}")
    model = build_native_age_backend(checkpoint, contract).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model_before = tensor_state_fingerprint(model.state_dict())
    rows: list[dict[str, Any]] = []
    feature_batches: list[np.ndarray] = []
    windows: list[torch.Tensor] = []
    identities: list[dict[str, Any]] = []

    def flush() -> None:
        if not windows:
            return
        value = torch.stack(windows).to(device=device, dtype=torch.float32)
        with torch.inference_mode():
            output = model(value)
        representation = output["representation"].detach().float().cpu()
        if representation.shape != (len(windows), feature_dim) or not torch.isfinite(representation).all():
            raise ZeroInitCV7Error("native backbone returned invalid representations")
        feature_batches.append(representation.numpy().copy())
        rows.extend(identities)
        windows.clear(); identities.clear()

    try:
        for subject_id in EXPECTED_SUBJECT_AGES:
            emitted = 0
            for window in iter_native_windows(recordings[subject_id]):
                emitted += 1
                windows.append(torch.from_numpy(window.data.copy()))
                identities.append({
                    "sample_id": f"{subject_id}_W{window.window_index:06d}",
                    "subject_id": subject_id, "recording_id": subject_id,
                    "window_index": int(window.window_index), "num_valid_tokens": token_count,
                })
                if len(windows) == batch_size:
                    flush()
            if emitted == 0:
                raise ZeroInitCV7Error(f"no complete native window for {subject_id}")
            _log(f"feature cache subject completed subject={subject_id} windows={emitted}")
        flush()
        features = np.concatenate(feature_batches, axis=0).astype(np.float32, copy=False)
        if features.shape != (len(rows), feature_dim) or not np.isfinite(features).all():
            raise ZeroInitCV7Error("assembled feature cache is invalid")
        np.save(temporary / "features.float32.npy", features, allow_pickle=False)
        _json_write(temporary / "samples.json", rows)
        manifest = {
            "identity": identity, "feature_cache_key": key,
            "row_count": len(rows), "feature_dim": feature_dim, "dtype": "float32",
            "feature_fingerprint": _features_fingerprint(features),
            "features_file_sha256": sha256_file(temporary / "features.float32.npy"),
            "samples_file_sha256": sha256_file(temporary / "samples.json"),
            "model_state_fingerprint_before": model_before,
            "model_state_fingerprint_after": tensor_state_fingerprint(model.state_dict()),
            "backbone_forward_only_during_cache_build": True,
            "backbone_forward_count_during_cv7": 0,
            "labels_in_cache": False,
        }
        if manifest["model_state_fingerprint_before"] != manifest["model_state_fingerprint_after"]:
            raise ZeroInitCV7Error("model changed during feature extraction")
        _json_write(temporary / "manifest.json", manifest)
        (temporary / "_SUCCESS").write_text(key + "\n", encoding="utf-8")
        os.replace(temporary, target)
        result = _validate_feature_cache(target, identity)
        result["reused"] = False
        _log(f"feature cache published key={key} rows={len(rows)}")
        return result
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _indices_for_subjects(samples: Sequence[Mapping[str, Any]], subjects: Sequence[str]) -> list[int]:
    selected = set(subjects)
    return [index for index, row in enumerate(samples) if row["subject_id"] in selected]


def _evaluate(
    probe: nn.Linear, features: torch.Tensor, samples: Sequence[Mapping[str, Any]],
    indices: Sequence[int], subject_ages: Mapping[str, float], *, device: torch.device,
    mean: float, std: float, batch_size: int, fold: int, checkpoint_path: str = "",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    probe.eval(); rows = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch_indices = list(indices[start:start + batch_size])
            value = features[batch_indices].to(device)
            prediction_z = probe(value).reshape(-1).float().cpu()
            if not torch.isfinite(prediction_z).all():
                raise ZeroInitCV7Error("probe returned non-finite predictions")
            prediction_age = prediction_z * std + mean
            for offset, index in enumerate(batch_indices):
                sample = samples[index]; subject = str(sample["subject_id"])
                truth = float(subject_ages[subject]); predicted = float(prediction_age[offset])
                error = predicted - truth
                rows.append({
                    "fold": fold, "sample_id": sample["sample_id"], "subject_id": subject,
                    "recording_id": sample["recording_id"], "window_index": sample["window_index"],
                    "true_age": truth, "true_z_source": (truth - mean) / std,
                    "predicted_z": float(prediction_z[offset]), "predicted_age": predicted,
                    "signed_error": error, "absolute_error": abs(error),
                    "squared_error": error * error, "num_valid_tokens": sample["num_valid_tokens"],
                    "checkpoint_path": checkpoint_path,
                })
    if not rows:
        raise ZeroInitCV7Error("probe evaluation received no rows")
    truth = [float(row["true_age"]) for row in rows]
    prediction = [float(row["predicted_age"]) for row in rows]
    subject_prediction = float(np.mean(prediction))
    subject_truth = float(np.mean(truth))
    window = regression_metrics(truth, prediction)
    loss = float(np.mean([
        (float(row["predicted_z"]) - float(row["true_z_source"])) ** 2 for row in rows
    ]))
    return {
        "loss": loss, "subject_mae": abs(subject_prediction - subject_truth),
        "subject_prediction": subject_prediction, "subject_truth": subject_truth,
        "window_mae": window["mae"], "window_rmse": window["rmse"],
        "sample_count": len(rows), "finite": True,
    }, rows


def _train_epoch(
    probe: nn.Linear, optimizer: torch.optim.Optimizer, features: torch.Tensor,
    samples: Sequence[Mapping[str, Any]], indices: Sequence[int],
    subject_ages: Mapping[str, float], *, device: torch.device, mean: float, std: float,
    batch_size: int, generator: torch.Generator,
) -> dict[str, float]:
    probe.train()
    order = torch.randperm(len(indices), generator=generator).tolist()
    total_loss = total_count = 0; max_gradient = 0.0
    for start in range(0, len(order), batch_size):
        batch_indices = [indices[order[index]] for index in range(start, min(start + batch_size, len(order)))]
        value = features[batch_indices].to(device)
        truth = torch.tensor(
            [(subject_ages[str(samples[index]["subject_id"])] - mean) / std for index in batch_indices],
            dtype=torch.float32, device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        predicted = probe(value).reshape(-1)
        loss = torch.mean((predicted - truth) ** 2)
        if not torch.isfinite(loss):
            raise ZeroInitCV7Error("non-finite probe training loss")
        loss.backward()
        gradients = [parameter.grad.reshape(-1) for parameter in probe.parameters() if parameter.grad is not None]
        gradient = float(torch.linalg.vector_norm(torch.cat(gradients))) if gradients else 0.0
        if not math.isfinite(gradient):
            raise ZeroInitCV7Error("non-finite probe gradient")
        optimizer.step()
        if any(not torch.isfinite(parameter).all() for parameter in probe.parameters()):
            raise ZeroInitCV7Error("non-finite probe parameter")
        total_loss += float(loss.detach()) * len(batch_indices); total_count += len(batch_indices)
        max_gradient = max(max_gradient, gradient)
    return {"train_z_mse": total_loss / total_count, "max_gradient_norm": max_gradient}


def _save_head(path: Path, payload: Mapping[str, Any]) -> str:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)
    return sha256_file(path)


def _load_head(
    path: Path, *, family: str, feature_dim: int, cache_key: str,
    fold: int, candidate_id: str, mean: float, std: float,
) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if type(payload) is not dict:
        raise ZeroInitCV7Error("head-only checkpoint root must be an exact mapping")
    expected = {
        "checkpoint_kind": HEAD_CHECKPOINT_KIND, "model_family": family,
        "feature_dim": feature_dim, "feature_cache_key": cache_key,
        "fold": fold, "candidate_id": candidate_id, "initialization": "all_zero",
        "source_mean": mean, "source_std": std,
        "age_target_normalization": "fixed_source_zscore",
        "target_scaler_fitted_on_huada": False,
        "prediction_clipping": "none",
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ZeroInitCV7Error("head-only checkpoint metadata mismatch")
    expected_initial = probe_fingerprint(build_zero_probe(feature_dim, torch.device("cpu")))
    if payload.get("initial_head_fingerprint") != expected_initial:
        raise ZeroInitCV7Error("head-only checkpoint initial state is not canonical all-zero")
    if set(payload) & {"backbone", "backbone_state_dict", "optimizer_state_dict", "scheduler_state_dict"}:
        raise ZeroInitCV7Error("head-only checkpoint contains forbidden training/backbone state")
    state = payload.get("head_state_dict")
    if type(state) is not dict or set(state) != {"weight", "bias"}:
        raise ZeroInitCV7Error("head-only checkpoint state is invalid")
    if tuple(state["weight"].shape) != (1, feature_dim) or tuple(state["bias"].shape) != (1,):
        raise ZeroInitCV7Error("head-only checkpoint shape mismatch")
    if any(
        not torch.is_tensor(value)
        or not value.dtype.is_floating_point
        or not torch.isfinite(value).all()
        for value in state.values()
    ):
        raise ZeroInitCV7Error("head-only checkpoint tensors are invalid")
    if payload.get("head_state_fingerprint") != tensor_state_fingerprint(state):
        raise ZeroInitCV7Error("head-only checkpoint fingerprint mismatch")
    return state


def _run_candidate(
    *, family: str, feature_dim: int, cache_key: str, features: torch.Tensor,
    samples: Sequence[Mapping[str, Any]], subject_ages: Mapping[str, float],
    fold_info: Mapping[str, Any], candidate: Mapping[str, Any], output_dir: Path,
    device: torch.device, mean: float, std: float, epochs: int, patience: int,
    batch_size: int, seed: int,
) -> dict[str, Any]:
    fold = int(fold_info["fold"]); candidate_id = str(candidate["candidate_id"])
    candidate_dir = output_dir / f"fold_{fold}" / "candidates" / candidate_id
    candidate_dir.mkdir(parents=True, exist_ok=False)
    probe = build_zero_probe(feature_dim, device)
    initial_fingerprint = probe_fingerprint(probe)
    optimizer = torch.optim.AdamW(
        probe.parameters(), lr=float(candidate["head_lr"]), weight_decay=float(candidate["weight_decay"])
    )
    train_indices = _indices_for_subjects(samples, fold_info["train_subjects"])
    validation_indices = _indices_for_subjects(samples, [fold_info["validation_subject"]])
    shuffle_generator = torch.Generator().manual_seed(seed + fold)
    best = None; history = []; best_path = candidate_dir / "best_head.pt"; stale = 0
    for epoch in range(0, epochs + 1):
        train_metrics = None
        if epoch > 0:
            train_metrics = _train_epoch(
                probe, optimizer, features, samples, train_indices, subject_ages,
                device=device, mean=mean, std=std, batch_size=batch_size,
                generator=shuffle_generator,
            )
        validation, _ = _evaluate(
            probe, features, samples, validation_indices, subject_ages,
            device=device, mean=mean, std=std, batch_size=batch_size, fold=fold,
        )
        row = {"fold": fold, "candidate_id": candidate_id, "epoch": epoch, **validation}
        if train_metrics:
            row.update(train_metrics)
        history.append(row)
        if candidate_is_better(row, best):
            best = dict(row); stale = 0
            state = probe_state(probe)
            payload = {
                "checkpoint_kind": HEAD_CHECKPOINT_KIND, "model_family": family,
                "feature_dim": feature_dim, "feature_cache_key": cache_key,
                "fold": fold, "candidate_id": candidate_id, "initialization": "all_zero",
                "initial_head_fingerprint": initial_fingerprint,
                "head_state_dict": state,
                "head_state_fingerprint": tensor_state_fingerprint(state),
                "best_epoch": epoch,
                "validation_metrics": dict(validation),
                "source_mean": mean, "source_std": std,
                "age_target_normalization": "fixed_source_zscore",
                "target_scaler_fitted_on_huada": False,
                "prediction_clipping": "none",
                "optimizer_config": {
                    "name": "AdamW", "head_lr": float(candidate["head_lr"]),
                    "weight_decay": float(candidate["weight_decay"]),
                    "scheduler": "none", "parameter_count": feature_dim + 1,
                },
            }
            _save_head(best_path, payload)
            _log(
                f"fold={fold} epoch={epoch} new_best validation_subject_mae="
                f"{validation['subject_mae']:.6f} validation_window_mae="
                f"{validation['window_mae']:.6f} validation_z_mse={validation['loss']:.6f}"
            )
        else:
            stale += 1
            if epoch == 1 or epoch % 10 == 0:
                _log(
                    f"fold={fold} epoch={epoch} stale={stale} validation_subject_mae="
                    f"{validation['subject_mae']:.6f}"
                )
        if epoch > 0 and stale >= patience:
            _log(f"fold={fold} early_stop epoch={epoch} patience={patience}")
            break
    if best is None:
        raise ZeroInitCV7Error("candidate did not produce a best checkpoint")
    _csv_write(candidate_dir / "epoch_metrics.csv", history, sorted({key for row in history for key in row}))
    return {
        "fold": fold, "candidate_id": candidate_id,
        "head_lr": candidate["head_lr"], "weight_decay": candidate["weight_decay"],
        "initial_head_fingerprint": initial_fingerprint,
        "best_epoch": int(best["epoch"]), "best_validation_mae": best["subject_mae"],
        "best_validation_rmse": best["window_rmse"], "best_validation_loss": best["loss"],
        "best_validation_window_mae": best["window_mae"],
        "checkpoint_path": str(best_path.resolve()), "checkpoint_sha256": sha256_file(best_path),
        "epochs_executed": len(history) - 1, "test_accessed_during_selection": False,
    }


def _aggregate(
    *, output_dir: Path, split_payload: Mapping[str, Any], subject_ages: Mapping[str, float],
    eligible_samples: Mapping[str, str], selected: Mapping[int, Mapping[str, Any]],
    oof_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    test_folds = {str(fold["test_subject"]): int(fold["fold"]) for fold in split_payload["folds"]}
    age_cv7.validate_oof_coverage(oof_rows, eligible_samples, test_folds)
    ordered = sorted(oof_rows, key=lambda row: (int(row["fold"]), str(row["sample_id"])))
    _csv_write(output_dir / "oof_window_predictions.csv", ordered, OOF_FIELDS)
    aggregated = age_cv7.aggregate_subject_predictions(ordered)
    subject_rows = []
    for row in aggregated:
        fold = int(row["test_fold"])
        subject_rows.append({
            "subject_id": row["subject_id"], "true_age": row["true_age"],
            "window_count": row["window_count"], "mean_prediction": row["mean_predicted_age"],
            "median_prediction": row["median_predicted_age"], "prediction_std": row["prediction_std"],
            "prediction_min": row["prediction_min"], "prediction_max": row["prediction_max"],
            "signed_error": row["signed_error"], "absolute_error": row["absolute_error"],
            "test_fold": fold, "checkpoint_path": selected[fold]["checkpoint_path"],
        })
    _csv_write(output_dir / "oof_subject_predictions.csv", subject_rows, SUBJECT_FIELDS)
    truth = [float(row["true_age"]) for row in subject_rows]
    mean_metrics = regression_metrics(truth, [float(row["mean_prediction"]) for row in subject_rows])
    median_metrics = regression_metrics(truth, [float(row["median_prediction"]) for row in subject_rows])
    window_metrics = regression_metrics(
        [float(row["true_age"]) for row in ordered], [float(row["predicted_age"]) for row in ordered]
    )
    window_metrics["level"] = "window_auxiliary_non_independent"
    source_baseline = regression_metrics(truth, [27.5] * len(truth))
    source_baseline.update({"baseline": "fixed_dortmund_source_mean", "prediction_years": 27.5})
    fold_baseline_rows = [{
        "fold": int(fold["fold"]), "test_subject": fold["test_subject"],
        "true_age": subject_ages[fold["test_subject"]],
        "prediction": age_cv7.train_mean_baseline(fold, subject_ages),
    } for fold in split_payload["folds"]]
    fold_baseline = regression_metrics(
        [row["true_age"] for row in fold_baseline_rows], [row["prediction"] for row in fold_baseline_rows]
    )
    fold_baseline.update({"baseline": "fold_train_subject_mean", "fold_predictions": fold_baseline_rows})
    for name, value in (
        ("oof_window_metrics.json", window_metrics),
        ("oof_subject_mean_metrics.json", mean_metrics),
        ("oof_subject_median_metrics.json", median_metrics),
        ("source_mean_baseline_metrics.json", source_baseline),
        ("fold_train_mean_baseline_metrics.json", fold_baseline),
    ):
        _json_write(output_dir / name, value)
    with (output_dir / "oof_subject_predictions.csv").open(encoding="utf-8-sig", newline="") as handle:
        reread = list(csv.DictReader(handle))
    recomputed = regression_metrics(
        [float(row["true_age"]) for row in reread], [float(row["mean_prediction"]) for row in reread]
    )
    for key in ("mae", "rmse", "r2", "bias"):
        if not math.isclose(float(recomputed[key]), float(mean_metrics[key]), rel_tol=0, abs_tol=1e-12):
            raise ZeroInitCV7Error("independent OOF metric recomputation failed")
    _json_write(output_dir / "metric_recomputation.json", {"passed": True, "subject_mean_metrics": recomputed})
    return {
        "oof_window_count": len(ordered), "oof_subject_count": len(subject_rows),
        "oof_subject_mean_metrics": mean_metrics, "oof_subject_median_metrics": median_metrics,
        "oof_window_metrics": window_metrics, "source_mean_baseline": source_baseline,
        "fold_train_mean_baseline": fold_baseline,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-family", choices=("labram", "reve"), required=True)
    parser.add_argument("--native-cache-run-root", required=True)
    parser.add_argument("--subject-manifest", required=True)
    parser.add_argument("--target-input-contract", required=True)
    parser.add_argument("--expected-target-contract-file-sha256", required=True)
    parser.add_argument("--portable-checkpoint", required=True)
    parser.add_argument("--portable-sidecar", required=True)
    parser.add_argument("--expected-portable-checkpoint-sha256", required=True)
    parser.add_argument("--expected-source-checkpoint-sha256", required=True)
    parser.add_argument("--dynamic-parity-dir", required=True)
    parser.add_argument("--strict-zero-shot-reference-dir")
    parser.add_argument("--feature-cache-root", required=True)
    parser.add_argument("--feature-cache-policy", choices=("auto", "require"), default="auto")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--feature-cache-batch-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--head-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--early-stopping-patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--expected-source-age-mean", type=float, default=27.5)
    parser.add_argument("--expected-source-age-std", type=float, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    for name in ("feature_cache_batch_size", "batch_size", "epochs", "early_stopping_patience"):
        if int(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    try:
        fixed_probe_candidate(args.head_lr, args.weight_decay)
    except ZeroInitCV7Error as exc:
        parser.error(str(exc))
    if not math.isclose(args.expected_source_age_mean, 27.5, rel_tol=0.0, abs_tol=1e-12):
        parser.error("Dortmund source age mean must remain 27.5")
    if not math.isfinite(args.expected_source_age_std) or args.expected_source_age_std <= 0:
        parser.error("--expected-source-age-std must be finite and positive")
    for value, label in (
        (args.expected_target_contract_file_sha256, "target contract SHA256"),
        (args.expected_portable_checkpoint_sha256, "portable checkpoint SHA256"),
        (args.expected_source_checkpoint_sha256, "source checkpoint SHA256"),
    ):
        try:
            _require_sha256(value, label)
        except NativeZeroShotError as exc:
            parser.error(str(exc))
    return args


def _dry_run(args: argparse.Namespace) -> int:
    print(json.dumps({
        "status": "dry_run_passed", "model_family": args.model_family,
        "fold_count": 7, "candidate_count_per_fold": 1,
        "candidate_search_count": 7, "test_evaluation_count": 7,
        "probe_initialization": "all_zero", "feature_cache_pooling": "mean",
        "would_read_native_cache": False, "would_read_portable_checkpoint": False,
        "would_construct_backbone": False, "would_execute_forward": False,
        "would_train_probe": False, "would_create_output": False,
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.dry_run:
        _dry_run(args)
        return {"status": "dry_run_passed"}
    project_root = Path(__file__).resolve().parents[2]
    output = Path(args.output_dir)
    if output.exists() or not output.parent.is_dir():
        raise ZeroInitCV7Error("output directory must be new and its parent must exist")
    output.mkdir()
    (output / "zero_init_cv7_errors.log").write_text("", encoding="utf-8")
    try:
        family = args.model_family
        _log(f"zero-init probe CV7 started family={family} device={args.device}")
        candidates = [fixed_probe_candidate(args.head_lr, args.weight_decay)]
        _log("[stage 1/7] validating subject manifest, target contract, and native cache")
        subject_manifest = _regular_file(args.subject_manifest, "subject manifest")
        subject_all = load_subject_age_manifest(subject_manifest)
        subject_ages = {key: float(value) for key, value in subject_all.items() if value is not None}
        if set(subject_ages) != ELIGIBLE_SUBJECTS:
            raise ZeroInitCV7Error("age eligible cohort mismatch")
        contract_path = _regular_file(args.target_input_contract, "target input contract")
        contract_file_sha = sha256_file(contract_path)
        if contract_file_sha != args.expected_target_contract_file_sha256.lower():
            raise ZeroInitCV7Error("target input contract file SHA mismatch")
        contract = load_huada_target_contract(contract_path, expected_family=family)
        _json_write(output / "resolved_config.json", vars(args))
        cache_root = Path(args.native_cache_run_root)
        cache_before = _cache_snapshot(cache_root)
        recordings, _ = load_native_cache_set(cache_root, family=family, target_contract=contract)
        _log("[stage 2/7] safe-loading portable checkpoint and validating dynamic parity")
        expected_portable_sha = _require_sha256(args.expected_portable_checkpoint_sha256, "portable SHA")
        expected_source_sha = _require_sha256(args.expected_source_checkpoint_sha256, "source SHA")
        checkpoint = load_portable_age_checkpoint(
            args.portable_checkpoint, expected_file_sha256=expected_portable_sha,
            expected_model_family=family, expected_source_checkpoint_sha256=expected_source_sha,
            map_location="cpu", require_manifest=True, allow_fixture=False,
        )
        _validate_sidecar(Path(args.portable_sidecar), checkpoint, family=family, expected_source_sha256=expected_source_sha)
        _validate_dynamic_parity(
            args.dynamic_parity_dir, family=family, portable_sha256=expected_portable_sha,
            source_sha256=expected_source_sha,
        )
        source_mean, source_std = _validate_scaler(
            checkpoint,
            expected_mean=args.expected_source_age_mean,
            expected_std=args.expected_source_age_std,
        )
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ZeroInitCV7Error("CUDA requested but unavailable")
        _log("[stage 3/7] building or reusing frozen mean-representation cache")
        feature_info = ensure_feature_cache(
            root=Path(args.feature_cache_root), policy=args.feature_cache_policy,
            family=family, checkpoint=checkpoint, contract=contract,
            contract_file_sha256=contract_file_sha, cache_snapshot=cache_before,
            recordings=recordings, device=device, batch_size=args.feature_cache_batch_size,
        )
        manifest = feature_info["manifest"]
        features = torch.from_numpy(np.array(feature_info["features"], copy=True))
        samples = feature_info["samples"]
        feature_dim = int(manifest["feature_dim"])
        cache_key = str(manifest["feature_cache_key"])
        split = age_cv7.build_age_subject_cv7(subject_ages, args.split_seed)
        age_cv7.write_fold_artifacts(
            output, split, Counter(row["subject_id"] for row in samples if row["subject_id"] != "SUB5")
        )
        _json_write(output / "resolved_probe_config.json", {
            "candidate": candidates[0],
            "protocol_reference": "dortmund_age_source_probe_huada_transfer_cv7",
            "only_intentional_change": "probe_initialization_all_zero",
        })
        _json_write(output / "feature_cache_manifest.json", manifest)
        zero_fingerprints = set(); candidate_rows = []; selected: dict[int, dict[str, Any]] = {}

        # Selection phase: every fold/candidate completes before any test access.
        _log("[stage 4/7] training zero-initialized probes with validation-only selection")
        for fold_info in split["folds"]:
            _log(
                f"fold={fold_info['fold']} selection start train={fold_info['train_subjects']} "
                f"validation={fold_info['validation_subject']} test_locked={fold_info['test_subject']}"
            )
            completed = []
            for candidate in candidates:
                row = _run_candidate(
                    family=family, feature_dim=feature_dim, cache_key=cache_key,
                    features=features, samples=samples, subject_ages=subject_ages,
                    fold_info=fold_info, candidate=candidate, output_dir=output,
                    device=device, mean=source_mean, std=source_std,
                    epochs=args.epochs, patience=args.early_stopping_patience,
                    batch_size=args.batch_size, seed=args.seed,
                )
                zero_fingerprints.add(row["initial_head_fingerprint"])
                completed.append(row)
            ranked = age_cv7.rank_candidates([dict(row) for row in completed])
            fold = int(fold_info["fold"])
            choice = dict(ranked[0])
            choice.update({
                "selection_reason": (
                    "single fixed FBMAE-transfer config; best epoch by validation "
                    "subject-mean MAE, window MAE, z-MSE, earlier epoch"
                ),
                "test_accessed_during_selection": False,
            })
            selected[fold] = choice
            candidate_rows.extend(ranked)
            ranking_dir = output / f"fold_{fold}"
            _csv_write(ranking_dir / "candidate_ranking.csv", ranked, list(ranked[0]))
            selected_dir = ranking_dir / "selected"
            selected_dir.mkdir(exist_ok=False)
            _json_write(selected_dir / "selection.json", choice)
            _log(
                f"fold={fold} selection locked best_epoch={choice['best_epoch']} "
                f"checkpoint_sha256={choice['checkpoint_sha256']}"
            )
        if len(zero_fingerprints) != 1:
            raise ZeroInitCV7Error("fold/candidate zero initialization fingerprints differ")
        _csv_write(output / "candidate_results.csv", candidate_rows, list(candidate_rows[0]))
        _csv_write(output / "selected_candidates.csv", list(selected.values()), list(next(iter(selected.values()))))
        _csv_write(output / "fold_search_results.csv", candidate_rows, list(candidate_rows[0]))
        _csv_write(output / "fold_selected_configs.csv", list(selected.values()), list(next(iter(selected.values()))))
        global_rows = []
        for candidate in candidates:
            rows = [row for row in candidate_rows if row["candidate_id"] == candidate["candidate_id"]]
            global_rows.append({
                "candidate_id": candidate["candidate_id"],
                "head_lr": candidate["head_lr"],
                "weight_decay": candidate["weight_decay"],
                "fold_count": len(rows),
                "mean_validation_mae": float(np.mean([row["best_validation_mae"] for row in rows])),
                "mean_validation_rmse": float(np.mean([row["best_validation_rmse"] for row in rows])),
                "mean_validation_loss": float(np.mean([row["best_validation_loss"] for row in rows])),
                "selected_fold_count": sum(
                    choice["candidate_id"] == candidate["candidate_id"] for choice in selected.values()
                ),
            })
        global_rows.sort(key=lambda row: (
            row["mean_validation_mae"], row["mean_validation_rmse"],
            row["mean_validation_loss"], row["candidate_id"],
        ))
        for rank, row in enumerate(global_rows, start=1):
            row["rank"] = rank
        _csv_write(output / "global_validation_ranking.csv", global_rows, list(global_rows[0]))

        # Test phase: exactly one selected head and one test evaluation per fold.
        _log("[stage 5/7] all selections locked; evaluating each test subject exactly once")
        oof_rows = []
        for fold_info in split["folds"]:
            fold = int(fold_info["fold"]); choice = selected[fold]
            checkpoint_path = Path(choice["checkpoint_path"])
            if sha256_file(checkpoint_path) != choice["checkpoint_sha256"]:
                raise ZeroInitCV7Error("selected head changed before test")
            state = _load_head(
                checkpoint_path, family=family, feature_dim=feature_dim,
                cache_key=cache_key, fold=fold, candidate_id=choice["candidate_id"],
                mean=source_mean, std=source_std,
            )
            probe = build_zero_probe(feature_dim, device)
            probe.load_state_dict(state, strict=True)
            test_indices = _indices_for_subjects(samples, [fold_info["test_subject"]])
            metrics, rows = _evaluate(
                probe, features, samples, test_indices, subject_ages,
                device=device, mean=source_mean, std=source_std,
                batch_size=args.batch_size, fold=fold, checkpoint_path=str(checkpoint_path.resolve()),
            )
            for row in rows:
                row["test_is_outside_train_age_range"] = bool(fold_info["test_is_outside_train_age_range"])
            oof_rows.extend(rows)
            _json_write(output / f"fold_{fold}" / "test_metrics.json", {
                **metrics, "evaluation_call_count": 1, "candidate_id": choice["candidate_id"],
                "test_subject": fold_info["test_subject"], "test_accessed_during_selection": False,
            })
            _log(
                f"fold={fold} test completed subject={fold_info['test_subject']} "
                f"subject_mae={metrics['subject_mae']:.6f} windows={metrics['sample_count']}"
            )
        eligible_samples = {
            row["sample_id"]: row["subject_id"] for row in samples if row["subject_id"] in ELIGIBLE_SUBJECTS
        }
        aggregate = _aggregate(
            output_dir=output, split_payload=split, subject_ages=subject_ages,
            eligible_samples=eligible_samples, selected=selected, oof_rows=oof_rows,
        )
        _log("[stage 6/7] OOF coverage and independent metric recomputation passed")
        zero_shot_comparison = {"status": "not_requested", "used_for_selection": False}
        if args.strict_zero_shot_reference_dir:
            reference = _validate_zero_shot_reference(
                args.strict_zero_shot_reference_dir,
                family=family,
                portable_sha256=expected_portable_sha,
                source_sha256=expected_source_sha,
            )
            zero_metrics = reference["metrics"]
            zero_shot_comparison = {
                "status": "compared_after_cv7_completion", "used_for_selection": False,
                "reference_dir": reference["directory"], "strict_zero_shot": zero_metrics,
                "zero_init_probe_cv7": aggregate["oof_subject_mean_metrics"],
                "delta_mae_cv7_minus_zero_shot": aggregate["oof_subject_mean_metrics"]["mae"] - zero_metrics["mae"],
            }
        _json_write(output / "strict_zero_shot_comparison.json", zero_shot_comparison)
        cache_after = _cache_snapshot(cache_root)
        if cache_before != cache_after:
            raise ZeroInitCV7Error("native recording cache changed during CV7")
        provenance = {
            "run_mode": RUN_MODE, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_family": family, "portable_checkpoint_sha256": expected_portable_sha,
            "source_checkpoint_sha256": expected_source_sha,
            "portable_backbone_state_fingerprint": checkpoint.payload["integrity"]["backbone_state_fingerprint"],
            "portable_source_age_probe_used_for_cv7": False,
            "source_scaler": {
                "mean": source_mean, "std": source_std, "ddof": 0,
                "target_unit": "year", "reused": True,
            },
            "age_target_normalization": "fixed_source_zscore",
            "portable_source_scaler_used_for_cv7": True,
            "huada_target_scaler_fitted": False,
            "native_cache_run_root": str(cache_root.resolve()),
            "native_cache_snapshot_sha256": _json_sha256(cache_before),
            "target_contract_file_sha256": contract_file_sha,
            "target_contract_sha256": contract["contract_sha256"],
            "feature_cache_key": cache_key, "feature_cache_path": feature_info["path"],
            "feature_cache_reused": feature_info["reused"], "feature_cache_labels": False,
            "fold_protocol": age_cv7.PROTOCOL, "fold_algorithm_version": age_cv7.ALGORITHM_VERSION,
            "split_seed": args.split_seed, "train_seed": args.seed,
            "optimizer": {
                "name": "AdamW", "head_lr": args.head_lr,
                "weight_decay": args.weight_decay, "scheduler": "none",
            },
            "probe_initialization": "all_zero", "initial_probe_fingerprint": next(iter(zero_fingerprints)),
            "backbone_frozen": True, "backbone_forward_count_during_cv7": 0,
            "test_accessed_during_selection": False, "test_evaluation_count": 7,
            "prediction_clipping": "none", "calibration": "none",
            "git": _git_info(project_root), "critical_code_sha256": _critical_code_sha256(project_root),
            "python_version": platform.python_version(), "pytorch_version": torch.__version__,
        }
        _json_write(output / "processing_provenance.json", provenance)
        _log("[stage 7/7] publishing summary and success marker")
        summary = {
            "status": "passed", "model_family": family, "fold_count": 7,
            "eligible_subject_count": 7, "excluded_subjects": ["SUB5"],
            "candidate_count_per_fold": 1, "probe_initialization": "all_zero",
            "initial_probe_fingerprint_identical": True,
            "portable_source_age_probe_used": False,
            "age_target_normalization": "fixed_source_zscore",
            "source_scaler_reused": True,
            "head_lr": args.head_lr, "weight_decay": args.weight_decay,
            "feature_cache_key": cache_key, "feature_cache_reused": feature_info["reused"],
            "feature_dim": feature_dim, "oof_window_count": aggregate["oof_window_count"],
            "oof_subject_count": aggregate["oof_subject_count"],
            "oof_subject_mean_metrics": aggregate["oof_subject_mean_metrics"],
            "source_mean_baseline": aggregate["source_mean_baseline"],
            "fold_train_mean_baseline": aggregate["fold_train_mean_baseline"],
            "backbone_trainable_parameter_count": 0,
            "probe_trainable_parameter_count": feature_dim + 1,
            "backbone_forward_count_during_cv7": 0,
            "test_accessed_during_selection": False, "test_evaluation_count": 7,
            "prediction_clipping_performed": False, "calibration_performed": False,
            "metric_recomputation_passed": True,
        }
        _json_write(output / "zero_init_probe_cv7_summary.json", summary)
        (output / "zero_init_probe_cv7_report.md").write_text(
            f"# Huada {family} zero-init probe CV7\n\n"
            f"- Status: passed\n- Probe initialization: all-zero\n"
            f"- Frozen mean feature cache: `{cache_key}`\n"
            f"- OOF subject-mean MAE/RMSE: {aggregate['oof_subject_mean_metrics']['mae']:.6f} / "
            f"{aggregate['oof_subject_mean_metrics']['rmse']:.6f}\n"
            "- No calibration or prediction clipping.\n",
            encoding="utf-8",
        )
        (output / "_SUCCESS").write_text("passed\n", encoding="utf-8")
        print(output)
        return summary
    except Exception:
        (output / "zero_init_cv7_errors.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
