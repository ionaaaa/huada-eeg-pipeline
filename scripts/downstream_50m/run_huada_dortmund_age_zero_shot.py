#!/usr/bin/env python3
"""Strict Dortmund full-finetuned age checkpoint zero-shot on Huada FBMAE tokens."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import subprocess
import sys
import traceback
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from .dortmund_huada_zero_shot import (
        EXPECTED_SUBJECTS, FORMAL_CHECKPOINT_CONTRACT, MASK_SEMANTICS_VERSION,
        ZeroShotValidationError,
        aggregate_subjects, compute_metric_bundle, fingerprint_model,
        infer_label_blind, json_safe, label_blind_collate, load_strict_checkpoint,
        prediction_range_diagnostics, recompute_from_csv, sha256_file,
        validate_coverage, validate_input_compatibility,
    )
except ImportError:
    from dortmund_huada_zero_shot import (
        EXPECTED_SUBJECTS, FORMAL_CHECKPOINT_CONTRACT, MASK_SEMANTICS_VERSION,
        ZeroShotValidationError,
        aggregate_subjects, compute_metric_bundle, fingerprint_model,
        infer_label_blind, json_safe, label_blind_collate, load_strict_checkpoint,
        prediction_range_diagnostics, recompute_from_csv, sha256_file,
        validate_coverage, validate_input_compatibility,
    )


STAGES = [
    "validate input", "validate checkpoint", "build frozen model",
    "zero-shot inference", "aggregate subject predictions",
    "compute metrics and baseline", "validate and publish artifacts",
]


def git_info(root: Path) -> dict[str, Any]:
    def command(*parts: str) -> str:
        return subprocess.run(["git", *parts], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
    try:
        return {"commit": command("rev-parse", "HEAD"), "dirty": bool(command("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-lance", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--expected-checkpoint-sha256",
        help="Optional 64-hex checkpoint identity lock. If omitted, the actual SHA256 is still computed and recorded.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--expected-source-age-mean", type=float,
        help="Optional checkpoint source-scaler lock; must be supplied together with --expected-source-age-std.",
    )
    parser.add_argument(
        "--expected-source-age-std", type=float,
        help="Optional checkpoint source-scaler lock; must be supplied together with --expected-source-age-mean.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    if args.expected_checkpoint_sha256 is not None and (
        len(args.expected_checkpoint_sha256) != 64
        or any(c not in "0123456789abcdefABCDEF" for c in args.expected_checkpoint_sha256)
    ):
        parser.error("--expected-checkpoint-sha256 must be 64 hex characters")
    expected_scaler = (args.expected_source_age_mean, args.expected_source_age_std)
    if (expected_scaler[0] is None) != (expected_scaler[1] is None):
        parser.error("--expected-source-age-mean and --expected-source-age-std must be supplied together")
    if expected_scaler[0] is not None and (
        not np.isfinite(expected_scaler[0])
        or not np.isfinite(expected_scaler[1])
        or expected_scaler[1] <= 0
    ):
        parser.error("Expected source age mean/std are invalid")
    return args


def _dry_run(args: argparse.Namespace) -> int:
    # Deliberately no Path.exists/open/mkdir, torch.load, Lance import, or model construction.
    print(json.dumps({
        "mode": "dry_run", "input_lance": args.input_lance,
        "checkpoint": args.checkpoint, "output_dir": args.output_dir,
        "device": args.device, "batch_size": args.batch_size,
        "num_workers": args.num_workers, "stages": STAGES,
        "will_open_lance": False, "will_read_checkpoint": False,
        "will_create_model": False, "will_create_output_dir": False,
        "will_execute_forward": False, "will_write_predictions": False,
    }, ensure_ascii=False, indent=2))
    return 0


def _json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for source in rows:
            row = {key: source.get(key, "") for key in fields}
            for key, value in list(row.items()):
                if value is None:
                    row[key] = ""
            writer.writerow(row)
    os.replace(temporary, path)


def _input_snapshot(path: Path, dataset: Any) -> dict[str, Any]:
    files = []
    for item in sorted(value for value in path.rglob("*") if value.is_file()):
        stat = item.stat()
        files.append({"relative_path": item.relative_to(path).as_posix(), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    digest = hashlib.sha256()
    for row in files:
        digest.update(json.dumps(row, sort_keys=True).encode())
    schema_text = str(dataset.schema)
    return {
        "absolute_path": str(path.resolve()), "lance_version": dataset.version,
        "schema": schema_text, "schema_sha256": hashlib.sha256(schema_text.encode()).hexdigest(),
        "row_count": dataset.count_rows(), "fragment_count": len(list(dataset.get_fragments())),
        "files": files, "metadata_fingerprint": digest.hexdigest(),
    }


def _read_json_if_present(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _extract_checkpoint_model_channels(payload: Mapping[str, Any]) -> list[str]:
    profile = payload.get("channel_canonicalization", {})
    content = profile.get("content", profile) if isinstance(profile, Mapping) else {}
    if isinstance(content, Mapping):
        resolved = content.get("resolved", content)
        model = resolved.get("model", {}) if isinstance(resolved, Mapping) else {}
        channels = model.get("standard_channels") if isinstance(model, Mapping) else None
        if channels:
            return [str(value) for value in channels]
    raise ZeroShotValidationError("Checkpoint has no embedded standard model-channel profile")


def _input_contract(root: Path, dataset: Any, checkpoint_report: Mapping[str, Any], payload: Mapping[str, Any]) -> dict[str, Any]:
    schema_names = set(dataset.schema.names)
    forbidden = sorted({"prediction", "predicted_age", "predicted_age_years", "predicted_z", "calibrated_age", "split", "fold"}.intersection(schema_names))
    if forbidden:
        raise ZeroShotValidationError(f"Input Lance contains prediction/training fields: {forbidden}")
    required = {"sample_id", "subject_id", "recording_id", "window_index", "age", "token_inputs", "token_channel_indices", "token_time_indices", "token_valid_mask", "num_tokens", "patch_len", "n_channels"}
    missing = sorted(required.difference(schema_names))
    if missing:
        raise ZeroShotValidationError(f"Input Lance missing fields: {missing}")

    repo_root = Path(__file__).resolve().parents[2]
    vocab_path = repo_root / "configs" / "fbmae_electrode_vocab_snapshot.json"
    vocab_sha = sha256_file(vocab_path)
    if vocab_sha != checkpoint_report["electrode_vocabulary_sha256"]:
        raise ZeroShotValidationError("Repository authoritative vocabulary hash differs from checkpoint")
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    name_to_id = {str(name).lower(): int(key) for key, name in vocab.items()}
    contract_sidecar = _read_json_if_present(root.parent / "model_contract_50m.json")
    if not isinstance(contract_sidecar, Mapping) or not contract_sidecar.get("standard_channels"):
        raise ZeroShotValidationError("Huada model_contract_50m.json with standard_channels is required")
    huada_channels = [str(value) for value in contract_sidecar["standard_channels"]]
    checkpoint_channels = _extract_checkpoint_model_channels(payload)
    if huada_channels != checkpoint_channels:
        raise ZeroShotValidationError("Huada model slots differ from checkpoint model slots")
    electrode_ids = [name_to_id.get(name.lower()) for name in huada_channels]
    if any(value is None for value in electrode_ids) or len(set(electrode_ids)) != 64:
        raise ZeroShotValidationError("Huada model channels do not map uniquely to authoritative vocabulary")
    return {
        "electrode_vocabulary_sha256": vocab_sha,
        # Compatibility is established by exact semantic channel/ID equality;
        # retain the checkpoint's source-profile hash as the governing record.
        "channel_canonicalization_sha256": checkpoint_report["channel_canonicalization_sha256"],
        "token_mask_semantics_version": MASK_SEMANTICS_VERSION,
        "model_channel_names": huada_channels, "model_electrode_ids": electrode_ids,
        "mapping_source": "Huada model_contract_50m + repository authoritative vocabulary",
        "dortmund_source_channel_order_used": False,
    }


class HuadaAllSubjectDataset(Dataset):
    """Ordered label-blind view; nullable age is metadata, never a model input."""

    COLUMNS = [
        "num_tokens", "patch_len", "n_channels", "token_inputs",
        "token_channel_indices", "token_time_indices", "token_valid_mask",
        "sample_id", "subject_id", "recording_id", "window_index", "age",
    ]

    def __init__(self, path: Path, lance_dataset: Any, metadata: Sequence[Mapping[str, Any]]):
        self.path = str(path)
        self.ds = lance_dataset
        self.metadata = list(metadata)

    def __len__(self) -> int:
        return len(self.metadata)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["ds"] = None
        return state

    def _open(self):
        if self.ds is None:
            import lance
            self.ds = lance.dataset(self.path)
        return self.ds

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self._open().take([int(index)], columns=self.COLUMNS).to_pylist()[0]
        num_tokens, patch_len = int(row["num_tokens"]), int(row["patch_len"])
        return {
            "token_inputs": torch.from_numpy(np.asarray(row["token_inputs"], dtype=np.float32).reshape(num_tokens, patch_len)),
            "token_channel_indices": torch.from_numpy(np.asarray(row["token_channel_indices"], dtype=np.int64)),
            "token_time_indices": torch.from_numpy(np.asarray(row["token_time_indices"], dtype=np.int64)),
            "token_valid_mask": torch.from_numpy(np.asarray(row["token_valid_mask"], dtype=np.float32)),
            "sample_id": str(row["sample_id"]), "subject_id": str(row["subject_id"]),
            "recording_id": str(row["recording_id"]), "window_index": int(row["window_index"]),
            "true_age_years": None if row["age"] is None else float(row["age"]),
        }


def _validate_metadata(dataset: Any) -> tuple[list[dict[str, Any]], list[str]]:
    rows = dataset.to_table(columns=["sample_id", "subject_id", "recording_id", "window_index", "age"]).to_pylist()
    sample_ids, subjects = [], set()
    ages: dict[str, float | None] = {}
    for index, row in enumerate(rows):
        sample_id = str(row.get("sample_id", "")).strip()
        subject_id = str(row.get("subject_id", "")).strip()
        if not sample_id or not subject_id:
            raise ZeroShotValidationError(f"Empty sample/subject ID at row {index}")
        if sample_id in set(sample_ids):
            raise ZeroShotValidationError(f"Duplicate sample_id: {sample_id}")
        age = None if row.get("age") is None else float(row["age"])
        if age is not None and not np.isfinite(age):
            raise ZeroShotValidationError("Non-finite age metadata")
        if subject_id in ages and ages[subject_id] != age:
            raise ZeroShotValidationError(f"Inconsistent age for {subject_id}")
        ages[subject_id] = age
        sample_ids.append(sample_id); subjects.add(subject_id)
    if subjects != EXPECTED_SUBJECTS:
        raise ZeroShotValidationError(f"Expected SUB1-SUB8, got {sorted(subjects)}")
    if ages.get("SUB5") is not None:
        raise ZeroShotValidationError("SUB5 must have nullable/missing age")
    return rows, sample_ids


def _metric_equal(left: Any, right: Any, atol: float = 1e-10) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(_metric_equal(left[key], right[key], atol) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_metric_equal(a, b, atol) for a, b in zip(left, right))
    if isinstance(left, (float, int)) and isinstance(right, (float, int)) and not isinstance(left, bool) and not isinstance(right, bool):
        return bool(np.isclose(float(left), float(right), rtol=0, atol=atol))
    return left == right


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.dry_run:
        return _dry_run(args)

    input_path = Path(args.input_lance)
    checkpoint_path = Path(args.checkpoint)
    output = Path(args.output_dir)
    if not input_path.is_dir():
        raise FileNotFoundError(input_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    output.mkdir(parents=True, exist_ok=False)
    log_path = output / "run.log"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)])
    errors = output / "zero_shot_errors.log"
    errors.write_text("", encoding="utf-8")
    try:
        logging.info("[stage 1/7] validate input")
        import lance
        lance_ds = lance.dataset(input_path)
        before = _input_snapshot(input_path, lance_ds)
        _json(output / "input_snapshot_before.json", before)
        metadata, expected_ids = _validate_metadata(lance_ds)

        logging.info("[stage 2/7] validate checkpoint")
        device = torch.device(args.device)
        model, checkpoint_report, payload = load_strict_checkpoint(
            checkpoint_path,
            args.expected_checkpoint_sha256,
            device,
            contract=replace(FORMAL_CHECKPOINT_CONTRACT, best_epoch=None),
            expected_source_mean=args.expected_source_age_mean,
            expected_source_std=args.expected_source_age_std,
        )
        checkpoint_report["sha256_expectation_enforced"] = args.expected_checkpoint_sha256 is not None
        checkpoint_report["source_scaler_expectation_enforced"] = args.expected_source_age_mean is not None
        contract = _input_contract(input_path, lance_ds, checkpoint_report, payload)
        compatibility = validate_input_compatibility(checkpoint_report=checkpoint_report, input_contract=contract)
        checkpoint_report["input_compatibility"] = compatibility
        _json(output / "checkpoint_validation.json", checkpoint_report)
        inventory = []
        for owner, state in (("backbone", payload["backbone_state_dict"]), ("age_head", payload["age_probe_state_dict"])):
            inventory.extend({"owner": owner, "key": key, "shape": list(value.shape), "dtype": str(value.dtype), "numel": value.numel()} for key, value in sorted(state.items()))
        _csv(output / "checkpoint_key_inventory.csv", inventory, ["owner", "key", "shape", "dtype", "numel"])

        logging.info("[stage 3/7] build frozen model")
        fingerprint_before = fingerprint_model(model)
        _json(output / "model_fingerprint_before.json", fingerprint_before)

        logging.info("[stage 4/7] zero-shot inference")
        data = HuadaAllSubjectDataset(input_path, lance_ds, metadata)
        loader_kwargs = {"dataset": data, "batch_size": args.batch_size, "shuffle": False, "num_workers": args.num_workers, "collate_fn": label_blind_collate}
        if args.num_workers > 0:
            loader_kwargs["multiprocessing_context"] = "spawn"
        loader = DataLoader(**loader_kwargs)
        rows = infer_label_blind(
            model, loader, device=device,
            source_mean=checkpoint_report["source_age_mean"],
            source_std=checkpoint_report["source_age_std"],
            checkpoint_sha256=checkpoint_report["checkpoint_sha256"],
        )
        coverage = validate_coverage(rows, expected_ids)
        _csv(output / "window_predictions.csv", rows, WINDOW_FIELDS)

        logging.info("[stage 5/7] aggregate subject predictions")
        bundle = compute_metric_bundle(rows, source_mean=checkpoint_report["source_age_mean"])
        _csv(output / "subject_predictions_all.csv", bundle["subject_rows_all"], SUBJECT_FIELDS)
        _csv(output / "subject_predictions_labeled.csv", bundle["subject_rows_labeled"], SUBJECT_FIELDS)

        logging.info("[stage 6/7] compute metrics and baseline")
        _json(output / "window_metrics.json", bundle["window_metrics"])
        _json(output / "subject_mean_metrics.json", bundle["subject_mean_metrics"])
        _json(output / "subject_median_metrics.json", bundle["subject_median_metrics"])
        _json(output / "source_mean_baseline_metrics.json", bundle["source_mean_baseline_metrics"])
        _json(output / "baseline_comparison.json", bundle["baseline_comparison"])
        diagnostics = prediction_range_diagnostics(rows, bundle["subject_rows_all"])
        _json(output / "prediction_range_diagnostics.json", diagnostics)
        _json(output / "coverage_validation.json", coverage)

        logging.info("[stage 7/7] validate and publish artifacts")
        recomputed = recompute_from_csv(
            output / "window_predictions.csv", checkpoint_report["source_age_mean"],
            checkpoint_report["source_age_std"], expected_sample_ids=expected_ids,
        )
        if not _metric_equal(bundle, recomputed["metrics"]):
            raise ZeroShotValidationError("CSV metric recomputation differs from in-memory metrics")
        recomputed["matches_in_memory"] = True
        _json(output / "metric_recomputation.json", recomputed)
        fingerprint_after = fingerprint_model(model)
        _json(output / "model_fingerprint_after.json", fingerprint_after)
        if fingerprint_before != fingerprint_after:
            raise ZeroShotValidationError("Model changed during inference")
        after = _input_snapshot(input_path, lance_ds)
        _json(output / "input_snapshot_after.json", after)
        if before != after:
            raise ZeroShotValidationError("Input Lance changed during zero-shot")
        resolved = vars(args).copy(); resolved["checkpoint_source_age_std_full_precision"] = checkpoint_report["source_age_std"]
        _json(output / "resolved_config.json", resolved)
        provenance = {
            "created_at": datetime.now().astimezone().isoformat(), "git": git_info(Path(__file__).resolve().parents[2]),
            "checkpoint": checkpoint_report, "input_contract": contract,
            "inference_semantics": "predicted_z * checkpoint_source_std + checkpoint_source_mean",
            "label_blind_forward": True, "calibration_used": False, "clipping_used": False,
            "optimizer_created": False, "scheduler_created": False, "backward_called": False,
        }
        _json(output / "zero_shot_provenance.json", provenance)
        summary = {
            "status": "passed", "input_rows": len(rows), "subject_count": 8,
            "labeled_subject_count": 7, "sub5_predicted": any(row["subject_id"] == "SUB5" for row in rows),
            "sub5_in_metrics": any(row["subject_id"] == "SUB5" for row in bundle["subject_rows_labeled"]),
            "checkpoint_sha256": checkpoint_report["checkpoint_sha256"],
            "source_age_mean": checkpoint_report["source_age_mean"], "source_age_std": checkpoint_report["source_age_std"],
            "model_fingerprint_unchanged": True, "input_lance_unchanged": True,
            "subject_mean_metrics": bundle["subject_mean_metrics"], "baseline_comparison": bundle["baseline_comparison"],
        }
        _json(output / "zero_shot_summary.json", summary)
        _csv(output / "zero_shot_issues.csv", [], ["severity", "stage", "issue", "detail"])
        report = "\n".join([
            "# Dortmund age -> Huada strict zero-shot", "",
            f"- Status: passed", f"- Windows: {len(rows)}", "- Subjects: 8 (7 labeled; SUB5 prediction retained)",
            f"- Checkpoint SHA256: `{checkpoint_report['checkpoint_sha256']}`",
            "- Transform: predicted_z × checkpoint source std + checkpoint source mean",
            "- Calibration/clipping/training: none", "- Window metrics are auxiliary and non-independent.",
        ]) + "\n"
        (output / "zero_shot_report.md").write_text(report, encoding="utf-8")
        (output / "_SUCCESS").write_text("passed\n", encoding="utf-8")
        return 0
    except Exception:
        errors.write_text(traceback.format_exc(), encoding="utf-8")
        logging.exception("Zero-shot failed")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
