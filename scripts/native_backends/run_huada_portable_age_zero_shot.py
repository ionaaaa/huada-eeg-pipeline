"""Strict Dortmund portable-age zero-shot evaluation on Huada native caches.

This entry point is intentionally inference-only.  It has no optimizer,
scheduler, calibration, prediction clipping, backward, or label-conditioned
model path.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from scripts.checkpoints.portable_age_checkpoint import (
    PortableAgeCheckpoint,
    load_portable_age_checkpoint,
    sha256_file,
    tensor_state_fingerprint,
)
from scripts.downstream_50m.dortmund_huada_zero_shot import (
    compute_metric_bundle,
    json_safe,
    prediction_range_diagnostics,
)

from .contracts import contract_sha256, load_huada_target_contract
from .portable_age_backend import build_native_age_backend
from .preprocessing import NativeRecordingCache, iter_native_windows


EXPECTED_SUBJECT_AGES: dict[str, float | None] = {
    "SUB1": 31.0,
    "SUB2": 29.0,
    "SUB3": 26.0,
    "SUB4": 26.0,
    "SUB5": None,
    "SUB6": 28.0,
    "SUB7": 28.0,
    "SUB8": 22.0,
}
BASELINE_AGE_YEARS = 27.5
_SHA256 = frozenset("0123456789abcdef")
WINDOW_FIELDS = [
    "sample_id", "subject_id", "recording_id", "window_index",
    "label_available", "metrics_eligible", "true_age_years", "predicted_z",
    "predicted_age_years", "signed_error_years", "absolute_error_years",
    "squared_error_years", "num_valid_tokens", "checkpoint_sha256",
]
SUBJECT_FIELDS = [
    "subject_id", "true_age", "window_count", "mean_predicted_age",
    "median_predicted_age", "mean_predicted_z", "prediction_std",
    "prediction_min", "prediction_max", "signed_error", "absolute_error",
    "included_in_metrics",
]


class NativeZeroShotError(RuntimeError):
    pass


def _require_sha256(value: str, label: str) -> str:
    text = str(value).lower()
    if len(text) != 64 or any(character not in _SHA256 for character in text):
        raise NativeZeroShotError(f"{label} must be exactly 64 hex characters")
    return text


def _regular_file(path: str | Path, label: str) -> Path:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise NativeZeroShotError(f"{label} is not a regular non-symlink file: {candidate}")
    return candidate


def _json_read(path: str | Path, label: str) -> Any:
    candidate = _regular_file(path, label)
    try:
        return json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise NativeZeroShotError(f"cannot read {label}: {candidate}") from exc


def _json_write(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(value), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _csv_write(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for source in rows:
            writer.writerow({key: "" if source.get(key) is None else source.get(key, "") for key in fields})
    os.replace(temporary, path)


def _validate_sidecar(
    path: Path,
    checkpoint: PortableAgeCheckpoint,
    *,
    family: str,
    expected_source_sha256: str,
) -> dict[str, Any]:
    sidecar = _json_read(path, "portable sidecar")
    if type(sidecar) is not dict:
        raise NativeZeroShotError("portable sidecar root must be an exact JSON object")
    payload = checkpoint.payload
    expected = {
        "portable_file_name": checkpoint.path.name if checkpoint.path is not None else None,
        "portable_file_sha256": checkpoint.file_sha256,
        "portable_file_size_bytes": checkpoint.file_size_bytes,
        "schema_name": payload["schema_name"],
        "schema_version": payload["schema_version"],
        "checkpoint_kind": payload["checkpoint_kind"],
        "model_name": payload["model"]["model_name"],
        "model_family": family,
        "source_checkpoint_sha256": expected_source_sha256,
        "backbone_state_fingerprint": payload["integrity"]["backbone_state_fingerprint"],
        "age_probe_state_fingerprint": payload["integrity"]["age_probe_state_fingerprint"],
    }
    mismatches = {
        key: {"expected": value, "actual": sidecar.get(key)}
        for key, value in expected.items()
        if sidecar.get(key) != value
    }
    if mismatches:
        raise NativeZeroShotError(f"portable sidecar mismatch: {mismatches}")
    if sidecar.get("dynamic_parity_status") not in {"pending", "passed"}:
        raise NativeZeroShotError("portable sidecar has invalid dynamic parity status")
    return sidecar


def _validate_dynamic_parity(
    root: str | Path,
    *,
    family: str,
    portable_sha256: str,
    source_sha256: str,
) -> dict[str, Any]:
    directory = Path(root)
    if directory.is_symlink() or not directory.is_dir() or not (directory / "_SUCCESS").is_file():
        raise NativeZeroShotError(f"dynamic parity directory is incomplete: {directory}")
    report = _json_read(directory / "dynamic_parity_report.json", "dynamic parity report")
    expected = {
        "status": "passed",
        "model_family": family,
        "source_checkpoint_sha256": source_sha256,
        "portable_checkpoint_sha256": portable_sha256,
        "dynamic_source_runtime_parity": "passed",
        "portable_transfer_readiness": "passed",
    }
    mismatches = {
        key: {"expected": value, "actual": report.get(key)}
        for key, value in expected.items()
        if report.get(key) != value
    }
    if mismatches:
        raise NativeZeroShotError(f"dynamic parity evidence mismatch: {mismatches}")
    return report


def _validate_scaler(
    checkpoint: PortableAgeCheckpoint,
    *,
    expected_mean: float,
    expected_std: float,
) -> tuple[float, float]:
    transform = checkpoint.payload["target_transform"]
    mean = float(transform["mean"])
    std = float(transform["std"])
    if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
        raise NativeZeroShotError("portable source scaler is invalid")
    if not math.isclose(mean, expected_mean, rel_tol=0.0, abs_tol=1e-12):
        raise NativeZeroShotError(f"portable source mean mismatch: expected={expected_mean}, actual={mean}")
    if not math.isclose(std, expected_std, rel_tol=0.0, abs_tol=1e-12):
        raise NativeZeroShotError(f"portable source std mismatch: expected={expected_std}, actual={std}")
    return mean, std


def _cache_snapshot(cache_root: Path) -> dict[str, Any]:
    rows = []
    for subject_id in EXPECTED_SUBJECT_AGES:
        root = cache_root / "recordings" / subject_id
        for name in ("recording.float32.npy", "metadata.json", "_SUCCESS"):
            path = root / name
            if path.is_symlink() or not path.is_file():
                raise NativeZeroShotError(f"cache artifact is missing or unsafe: {path}")
            stat = path.stat()
            rows.append(
                {
                    "subject_id": subject_id,
                    "name": name,
                    "size_bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "sha256": sha256_file(path),
                }
            )
    return {"cache_root": str(cache_root.resolve()), "files": rows}


def load_subject_age_manifest(
    path: str | Path, *, expected_file_sha256: str
) -> dict[str, float | None]:
    candidate = _regular_file(path, "Huada subject manifest")
    expected_sha = _require_sha256(expected_file_sha256, "subject manifest SHA256")
    actual_sha = sha256_file(candidate)
    if actual_sha != expected_sha:
        raise NativeZeroShotError(
            f"subject manifest SHA mismatch: expected={expected_sha}, actual={actual_sha}"
        )
    raw = _json_read(candidate, "Huada subject manifest")
    if type(raw) is not list or any(type(row) is not dict for row in raw):
        raise NativeZeroShotError("Huada subject manifest must be a JSON list of objects")
    ages: dict[str, float | None] = {}
    for row in raw:
        subject_id = row.get("subject_id")
        if subject_id in ages or subject_id not in EXPECTED_SUBJECT_AGES:
            raise NativeZeroShotError(f"unexpected or duplicate manifest subject: {subject_id!r}")
        age_value = row.get("age")
        age = None if age_value is None else float(age_value)
        if age is not None and not math.isfinite(age):
            raise NativeZeroShotError(f"non-finite age for {subject_id}")
        ages[str(subject_id)] = age
    if ages != EXPECTED_SUBJECT_AGES:
        raise NativeZeroShotError(
            f"subject manifest ages differ from the frozen Huada protocol: {ages}"
        )
    return ages


def load_native_cache_set(
    cache_run_root: str | Path,
    *,
    family: str,
    target_contract: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    root = Path(cache_run_root)
    recordings = root / "recordings"
    if root.is_symlink() or not root.is_dir() or not (root / "_SUCCESS").is_file():
        raise NativeZeroShotError(f"native cache run is incomplete: {root}")
    if recordings.is_symlink() or not recordings.is_dir():
        raise NativeZeroShotError("native cache recordings directory is missing or unsafe")
    actual_subjects = {item.name for item in recordings.iterdir() if item.is_dir()}
    if actual_subjects != set(EXPECTED_SUBJECT_AGES):
        raise NativeZeroShotError(
            f"native cache subject set mismatch: expected={sorted(EXPECTED_SUBJECT_AGES)}, actual={sorted(actual_subjects)}"
        )
    contract_hash = contract_sha256(target_contract)
    arrays: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {}
    for subject_id in EXPECTED_SUBJECT_AGES:
        data, record_metadata = NativeRecordingCache.load(
            recordings / subject_id,
            expected_family=family,
            expected_contract_sha256=contract_hash,
        )
        if record_metadata.get("recording_id") != subject_id:
            raise NativeZeroShotError(f"recording_id mismatch for {subject_id}")
        if not np.isfinite(data).all():
            raise NativeZeroShotError(f"native cache contains non-finite values: {subject_id}")
        if data.shape[1] < 400:
            raise NativeZeroShotError(f"native cache has no complete 2-second window: {subject_id}")
        arrays[subject_id] = data
        metadata[subject_id] = record_metadata
    return arrays, metadata


def _flush_batch(
    model: torch.nn.Module,
    windows: list[torch.Tensor],
    identities: list[dict[str, Any]],
    *,
    device: torch.device,
    source_mean: float,
    source_std: float,
    checkpoint_sha256: str,
    subject_ages: Mapping[str, float | None],
) -> list[dict[str, Any]]:
    if not windows:
        return []
    value = torch.stack(windows).to(device=device, dtype=torch.float32)
    output = model(value)
    z = output["predicted_z"].reshape(-1).float().cpu()
    backend_age = output["predicted_age_years"].reshape(-1).float().cpu()
    expected_age = z * float(source_std) + float(source_mean)
    if not torch.isfinite(z).all() or not torch.isfinite(backend_age).all():
        raise NativeZeroShotError("native backend returned non-finite predictions")
    if not torch.allclose(backend_age, expected_age, rtol=1e-6, atol=1e-5):
        raise NativeZeroShotError("backend age output differs from checkpoint scaler reconstruction")
    rows = []
    for index, identity in enumerate(identities):
        age = subject_ages[identity["subject_id"]]
        predicted_z = float(z[index])
        predicted_age = predicted_z * source_std + source_mean
        error = None if age is None else predicted_age - age
        rows.append(
            {
                **identity,
                "label_available": age is not None,
                "metrics_eligible": age is not None,
                "true_age_years": age,
                "predicted_z": predicted_z,
                "predicted_age_years": predicted_age,
                "num_valid_tokens": int(output["backbone_tokens"].shape[1]),
                "checkpoint_sha256": checkpoint_sha256,
                "signed_error_years": error,
                "absolute_error_years": None if error is None else abs(error),
                "squared_error_years": None if error is None else error * error,
            }
        )
    return rows


def infer_native_zero_shot(
    model: torch.nn.Module,
    recordings: Mapping[str, np.ndarray],
    *,
    device: torch.device,
    batch_size: int,
    source_mean: float,
    source_std: float,
    checkpoint_sha256: str,
    subject_ages: Mapping[str, float | None] = EXPECTED_SUBJECT_AGES,
) -> tuple[list[dict[str, Any]], int]:
    if dict(subject_ages) != EXPECTED_SUBJECT_AGES:
        raise NativeZeroShotError("subject ages differ from the frozen Huada protocol")
    rows: list[dict[str, Any]] = []
    windows: list[torch.Tensor] = []
    identities: list[dict[str, Any]] = []
    forward_count = 0
    model.eval()
    with torch.inference_mode():
        for subject_id in EXPECTED_SUBJECT_AGES:
            emitted = 0
            for window in iter_native_windows(recordings[subject_id]):
                emitted += 1
                windows.append(torch.from_numpy(window.data.copy()))
                identities.append(
                    {
                        "sample_id": f"{subject_id}_W{window.window_index:06d}",
                        "subject_id": subject_id,
                        "recording_id": subject_id,
                        "window_index": window.window_index,
                    }
                )
                if len(windows) == batch_size:
                    rows.extend(
                        _flush_batch(
                            model,
                            windows,
                            identities,
                            device=device,
                            source_mean=source_mean,
                            source_std=source_std,
                            checkpoint_sha256=checkpoint_sha256,
                            subject_ages=subject_ages,
                        )
                    )
                    forward_count += 1
                    windows, identities = [], []
            if emitted == 0:
                raise NativeZeroShotError(f"no complete windows emitted for {subject_id}")
        if windows:
            rows.extend(
                _flush_batch(
                    model,
                    windows,
                    identities,
                    device=device,
                    source_mean=source_mean,
                    source_std=source_std,
                    checkpoint_sha256=checkpoint_sha256,
                    subject_ages=subject_ages,
                )
            )
            forward_count += 1
    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise NativeZeroShotError("duplicate sample IDs were generated")
    if {row["subject_id"] for row in rows} != set(EXPECTED_SUBJECT_AGES):
        raise NativeZeroShotError("zero-shot prediction coverage is incomplete")
    return rows, forward_count


def _read_subject_result(root: Path, label: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if root.is_symlink() or not root.is_dir() or not (root / "_SUCCESS").is_file():
        raise NativeZeroShotError(f"comparison result is incomplete: {root}")
    metrics = _json_read(root / "subject_mean_metrics.json", f"{label} subject metrics")
    with _regular_file(root / "subject_predictions_all.csv", f"{label} subject predictions").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    if {row.get("subject_id") for row in rows} != set(EXPECTED_SUBJECT_AGES):
        raise NativeZeroShotError(f"{label} comparison subject coverage mismatch")
    if int(metrics.get("sample_count", -1)) != 7:
        raise NativeZeroShotError(f"{label} metrics must contain seven labeled subjects")
    return metrics, rows


def build_model_comparison(
    current_family: str,
    current_bundle: Mapping[str, Any],
    references: Sequence[str],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    models: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {
        current_family: (
            dict(current_bundle["subject_mean_metrics"]),
            [dict(row) for row in current_bundle["subject_rows_all"]],
        )
    }
    for specification in references:
        if "=" not in specification:
            raise NativeZeroShotError("--comparison-result must be MODEL_FAMILY=/absolute/path")
        label, raw_path = specification.split("=", 1)
        label = label.strip().lower()
        if label not in {"fbmae", "labram", "reve"} or label in models:
            raise NativeZeroShotError(f"invalid or duplicate comparison model: {label!r}")
        models[label] = _read_subject_result(Path(raw_path), label)
    metric_rows = []
    subject_maps: dict[str, dict[str, Mapping[str, Any]]] = {}
    for label, (metrics, rows) in sorted(models.items()):
        metric_rows.append(
            {
                "model_family": label,
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
                "pearson_r": metrics["pearson_r"].get("value"),
                "spearman_rho": metrics["spearman_rho"].get("value"),
                "bias": metrics["bias"],
                "subject_count": metrics["sample_count"],
            }
        )
        subject_maps[label] = {row["subject_id"]: row for row in rows}
    subject_rows = []
    for subject_id, age in EXPECTED_SUBJECT_AGES.items():
        row: dict[str, Any] = {"subject_id": subject_id, "true_age": age}
        for label, values in subject_maps.items():
            source = values[subject_id]
            prediction = source.get("mean_predicted_age", source.get("prediction"))
            if prediction in (None, ""):
                raise NativeZeroShotError(f"{label} has no subject-mean prediction for {subject_id}")
            row[f"{label}_mean_predicted_age"] = float(prediction)
        subject_rows.append(row)
    baseline = current_bundle["source_mean_baseline_metrics"]
    report = {
        "comparison_kind": "strict_zero_shot_subject_mean",
        "models": metric_rows,
        "fixed_baseline": baseline,
        "all_models_use_same_huada_subjects": True,
        "labeled_subject_count": 7,
        "sub5_prediction_retained_but_excluded_from_metrics": True,
    }
    return report, metric_rows, subject_rows


def _git_info(root: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        return subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    try:
        return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "dirty": None}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-family", required=True, choices=("labram", "reve"))
    parser.add_argument("--native-cache-run-root", required=True)
    parser.add_argument("--subject-manifest", required=True)
    parser.add_argument("--expected-subject-manifest-sha256", required=True)
    parser.add_argument("--target-input-contract", required=True)
    parser.add_argument("--expected-target-contract-file-sha256", required=True)
    parser.add_argument("--portable-checkpoint", required=True)
    parser.add_argument("--portable-sidecar", required=True)
    parser.add_argument("--expected-portable-checkpoint-sha256", required=True)
    parser.add_argument("--expected-source-checkpoint-sha256", required=True)
    parser.add_argument("--dynamic-parity-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--expected-source-age-mean", type=float, default=BASELINE_AGE_YEARS)
    parser.add_argument("--expected-source-age-std", type=float, required=True)
    parser.add_argument(
        "--comparison-result",
        action="append",
        default=[],
        help="Existing strict zero-shot result as MODEL_FAMILY=/absolute/path",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    for value, label in (
        (args.expected_target_contract_file_sha256, "target contract SHA256"),
        (args.expected_portable_checkpoint_sha256, "portable checkpoint SHA256"),
        (args.expected_source_checkpoint_sha256, "source checkpoint SHA256"),
        (args.expected_subject_manifest_sha256, "subject manifest SHA256"),
    ):
        try:
            _require_sha256(value, label)
        except NativeZeroShotError as exc:
            parser.error(str(exc))
    if not math.isfinite(args.expected_source_age_mean):
        parser.error("--expected-source-age-mean must be finite")
    if not math.isfinite(args.expected_source_age_std) or args.expected_source_age_std <= 0:
        parser.error("--expected-source-age-std must be finite and positive")
    return args


def _dry_run(args: argparse.Namespace) -> int:
    print(
        json.dumps(
            {
                "mode": "dry_run",
                "model_family": args.model_family,
                "native_cache_run_root": args.native_cache_run_root,
                "subject_manifest": args.subject_manifest,
                "portable_checkpoint": args.portable_checkpoint,
                "target_input_contract": args.target_input_contract,
                "dynamic_parity_dir": args.dynamic_parity_dir,
                "output_dir": args.output_dir,
                "will_read_cache": False,
                "will_read_checkpoint": False,
                "will_construct_model": False,
                "will_execute_forward": False,
                "training_enabled": False,
                "calibration_enabled": False,
                "prediction_clipping_enabled": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.dry_run:
        return _dry_run(args)

    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite zero-shot output: {output}")
    if not output.parent.is_dir():
        raise FileNotFoundError(f"output parent does not exist: {output.parent}")
    output.mkdir()
    log_path = output / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
        force=True,
    )
    error_path = output / "zero_shot_errors.log"
    error_path.write_text("", encoding="utf-8")
    try:
        family = args.model_family
        logging.info("[stage 1/8] validate target contract and native caches")
        subject_ages = load_subject_age_manifest(
            args.subject_manifest,
            expected_file_sha256=args.expected_subject_manifest_sha256,
        )
        contract_path = _regular_file(args.target_input_contract, "target input contract")
        expected_contract_file_sha = _require_sha256(
            args.expected_target_contract_file_sha256, "target contract SHA256"
        )
        if sha256_file(contract_path) != expected_contract_file_sha:
            raise NativeZeroShotError("target input contract file SHA mismatch")
        contract = load_huada_target_contract(contract_path, expected_family=family)
        cache_root = Path(args.native_cache_run_root)
        cache_before = _cache_snapshot(cache_root)
        recordings, cache_metadata = load_native_cache_set(
            cache_root, family=family, target_contract=contract
        )
        _json_write(output / "native_cache_validation.json", {
            "snapshot": cache_before,
            "recording_metadata": cache_metadata,
            "target_contract_file_sha256": expected_contract_file_sha,
            "target_contract_sha256": contract["contract_sha256"],
        })

        logging.info("[stage 2/8] SHA-first safe-load portable checkpoint")
        expected_portable_sha = _require_sha256(
            args.expected_portable_checkpoint_sha256, "portable checkpoint SHA256"
        )
        expected_source_sha = _require_sha256(
            args.expected_source_checkpoint_sha256, "source checkpoint SHA256"
        )
        checkpoint = load_portable_age_checkpoint(
            args.portable_checkpoint,
            expected_file_sha256=expected_portable_sha,
            expected_model_family=family,
            expected_source_checkpoint_sha256=expected_source_sha,
            map_location="cpu",
            require_manifest=True,
            allow_fixture=False,
        )
        sidecar = _validate_sidecar(
            Path(args.portable_sidecar),
            checkpoint,
            family=family,
            expected_source_sha256=expected_source_sha,
        )
        parity = _validate_dynamic_parity(
            args.dynamic_parity_dir,
            family=family,
            portable_sha256=expected_portable_sha,
            source_sha256=expected_source_sha,
        )
        source_mean, source_std = _validate_scaler(
            checkpoint,
            expected_mean=args.expected_source_age_mean,
            expected_std=args.expected_source_age_std,
        )
        _json_write(output / "portable_checkpoint_validation.json", {
            "portable_file_sha256": checkpoint.file_sha256,
            "portable_file_size_bytes": checkpoint.file_size_bytes,
            "source_checkpoint_sha256": expected_source_sha,
            "model_name": checkpoint.model_name,
            "model_family": checkpoint.model_family,
            "target_transform": checkpoint.payload["target_transform"],
            "integrity": checkpoint.payload["integrity"],
            "sidecar": sidecar,
            "dynamic_parity": parity,
        })

        logging.info("[stage 3/8] construct strict native backend")
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise NativeZeroShotError("CUDA device requested but CUDA is unavailable")
        model = build_native_age_backend(checkpoint, contract).to(device)
        fingerprint_before = {
            "sha256": tensor_state_fingerprint(model.state_dict()),
            "tensor_count": len(model.state_dict()),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        }
        _json_write(output / "model_fingerprint_before.json", fingerprint_before)

        logging.info("[stage 4/8] label-blind strict zero-shot forward")
        rows, forward_count = infer_native_zero_shot(
            model,
            recordings,
            device=device,
            batch_size=args.batch_size,
            source_mean=source_mean,
            source_std=source_std,
            checkpoint_sha256=expected_portable_sha,
            subject_ages=subject_ages,
        )
        _csv_write(output / "window_predictions.csv", rows, WINDOW_FIELDS)

        logging.info("[stage 5/8] aggregate subjects and compute metrics")
        bundle = compute_metric_bundle(rows)
        if len(bundle["subject_rows_labeled"]) != 7:
            raise NativeZeroShotError("exactly seven labeled subjects are required")
        if any(row["subject_id"] == "SUB5" for row in bundle["subject_rows_labeled"]):
            raise NativeZeroShotError("SUB5 entered zero-shot metrics")
        if not any(row["subject_id"] == "SUB5" for row in bundle["subject_rows_all"]):
            raise NativeZeroShotError("SUB5 unlabeled prediction is missing")
        if bundle["source_mean_baseline_metrics"]["baseline_prediction"] != BASELINE_AGE_YEARS:
            raise NativeZeroShotError("fixed 27.5-year baseline changed")
        _csv_write(output / "subject_predictions_all.csv", bundle["subject_rows_all"], SUBJECT_FIELDS)
        _csv_write(output / "subject_predictions_labeled.csv", bundle["subject_rows_labeled"], SUBJECT_FIELDS)
        for filename, key in (
            ("window_metrics.json", "window_metrics"),
            ("subject_mean_metrics.json", "subject_mean_metrics"),
            ("subject_median_metrics.json", "subject_median_metrics"),
            ("source_mean_baseline_metrics.json", "source_mean_baseline_metrics"),
            ("baseline_comparison.json", "baseline_comparison"),
        ):
            _json_write(output / filename, bundle[key])
        _json_write(
            output / "prediction_range_diagnostics.json",
            prediction_range_diagnostics(rows, bundle["subject_rows_all"]),
        )

        logging.info("[stage 6/8] build cross-model comparison")
        comparison, metric_rows, comparison_subject_rows = build_model_comparison(
            family, bundle, args.comparison_result
        )
        _json_write(output / "model_comparison.json", comparison)
        _csv_write(
            output / "model_comparison.csv",
            metric_rows,
            ["model_family", "mae", "rmse", "r2", "pearson_r", "spearman_rho", "bias", "subject_count"],
        )
        comparison_fields = ["subject_id", "true_age"] + [
            f"{label}_mean_predicted_age"
            for label in sorted({row["model_family"] for row in metric_rows})
        ]
        _csv_write(
            output / "model_subject_predictions_comparison.csv",
            comparison_subject_rows,
            comparison_fields,
        )

        logging.info("[stage 7/8] verify immutable model and cache")
        fingerprint_after = {
            "sha256": tensor_state_fingerprint(model.state_dict()),
            "tensor_count": len(model.state_dict()),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        }
        _json_write(output / "model_fingerprint_after.json", fingerprint_after)
        if fingerprint_before != fingerprint_after:
            raise NativeZeroShotError("model parameters changed during zero-shot")
        cache_after = _cache_snapshot(cache_root)
        if cache_before != cache_after:
            raise NativeZeroShotError("native recording cache changed during zero-shot")
        _json_write(output / "native_cache_snapshot_after.json", cache_after)

        logging.info("[stage 8/8] publish report")
        provenance = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "evaluation_kind": "dortmund_portable_age_strict_zero_shot",
            "model_family": family,
            "portable_checkpoint_sha256": expected_portable_sha,
            "source_checkpoint_sha256": expected_source_sha,
            "source_scaler": {"mean": source_mean, "std": source_std, "target_unit": "year"},
            "native_cache_run_root": str(cache_root.resolve()),
            "subject_manifest": str(Path(args.subject_manifest).resolve()),
            "subject_manifest_sha256": args.expected_subject_manifest_sha256,
            "target_contract_file_sha256": expected_contract_file_sha,
            "dynamic_parity_dir": str(Path(args.dynamic_parity_dir).resolve()),
            "device": str(device),
            "batch_size": args.batch_size,
            "forward_count": forward_count,
            "training_enabled": False,
            "optimizer_created": False,
            "scheduler_created": False,
            "backward_called": False,
            "calibration_used": False,
            "prediction_clipping_used": False,
            "labels_passed_to_model": False,
            "git": _git_info(Path(__file__).resolve().parents[2]),
        }
        _json_write(output / "resolved_config.json", vars(args))
        _json_write(output / "zero_shot_provenance.json", provenance)
        summary = {
            "status": "passed",
            "model_family": family,
            "window_count": len(rows),
            "subject_count": 8,
            "labeled_subject_count": 7,
            "sub5_predicted": True,
            "sub5_in_metrics": False,
            "subject_mean_metrics": bundle["subject_mean_metrics"],
            "fixed_27_5_baseline_metrics": bundle["source_mean_baseline_metrics"],
            "baseline_comparison": bundle["baseline_comparison"],
            "model_fingerprint_unchanged": True,
            "native_cache_unchanged": True,
            "training_performed": False,
            "calibration_performed": False,
            "prediction_clipping_performed": False,
        }
        _json_write(output / "zero_shot_summary.json", summary)
        report = f"""# Huada {family} Dortmund-age strict zero-shot

- Status: passed
- Portable checkpoint SHA256: `{expected_portable_sha}`
- Source FFT checkpoint SHA256: `{expected_source_sha}`
- Subjects: 8 total; 7 labeled; SUB5 prediction retained and excluded from metrics
- Windows: {len(rows)}; forward calls: {forward_count}
- Source scaler: predicted_z × {source_std!r} + {source_mean!r}
- Subject-mean MAE: {bundle['subject_mean_metrics']['mae']:.6f}
- Subject-mean RMSE: {bundle['subject_mean_metrics']['rmse']:.6f}
- Training/calibration/prediction clipping: none
"""
        (output / "zero_shot_report.md").write_text(report, encoding="utf-8")
        (output / "_SUCCESS").write_text("passed\n", encoding="utf-8")
        logging.info("strict zero-shot passed: %s", output)
        print(output)
        return 0
    except Exception:
        error_path.write_text(traceback.format_exc(), encoding="utf-8")
        logging.exception("strict native zero-shot failed")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
