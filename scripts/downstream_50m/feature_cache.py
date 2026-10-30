"""Frozen-backbone feature cache for 50M downstream linear probing."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from run_logging import log_event, log_exception


CACHE_SCHEMA_VERSION = 1
POOLING_IMPLEMENTATION_VERSION = "masked_mean_and_zero_masked_flatten_v1"
MASK_SEMANTICS_VERSION = "token_valid_mask_true_is_valid_v1"
CACHE_DTYPE = "float32"
MEAN_CACHE_NAME = "mean_features.lance"
FLATTEN_CACHE_NAME = "flatten_features.lance"
MANIFEST_NAME = "cache_manifest.json"
SUCCESS_NAME = "_SUCCESS"
PROHIBITED_CACHE_COLUMNS = {
    "label", "sex", "split", "fold", "candidate", "predicted_label",
}


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
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def get_git_state(project_root):
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_root,
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=project_root,
            stderr=subprocess.DEVNULL, text=True,
        ).strip())
        return commit, dirty
    except Exception:
        return None, None


def masked_pool_both(token_embeddings, token_valid_mask):
    """Compute mean and flatten from one backbone output, always as float32."""
    embeddings = token_embeddings.to(dtype=torch.float32)
    valid_mask = token_valid_mask.to(
        device=embeddings.device, dtype=embeddings.dtype,
    ).unsqueeze(-1)
    masked = embeddings * valid_mask
    denominator = valid_mask.sum(dim=1).clamp_min(1.0)
    mean_features = masked.sum(dim=1) / denominator
    flatten_features = masked.flatten(start_dim=1)
    if not torch.isfinite(mean_features).all() or not torch.isfinite(flatten_features).all():
        raise RuntimeError("Feature pooling produced NaN or Inf")
    return mean_features, flatten_features


class TokenFeatureSourceDataset(Dataset):
    """Label-free token reader used only while building the frozen cache."""

    COLUMNS = [
        "num_tokens", "patch_len", "token_inputs", "token_channel_indices",
        "token_time_indices", "token_valid_mask", "sample_id", "subject_id",
        "recording_id",
    ]

    def __init__(self, lance_path, storage_options=None):
        self.lance_path = str(lance_path)
        self.storage_options = storage_options
        kwargs = {} if storage_options is None else {"storage_options": storage_options}
        self.ds = lance.dataset(self.lance_path, **kwargs)
        missing = sorted(set(self.COLUMNS).difference(self.ds.schema.names))
        if missing:
            raise ValueError(f"Input Lance is missing cache source fields: {missing}")
        self.metadata = self.ds.to_table(
            columns=["sample_id", "subject_id", "recording_id"]
        ).to_pylist()
        sample_ids = [str(row["sample_id"]) for row in self.metadata]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("Input Lance sample_id values must be unique")

    def __len__(self):
        return len(self.metadata)

    @staticmethod
    def _convert(row):
        num_tokens = int(row["num_tokens"])
        patch_len = int(row["patch_len"])
        recording = row.get("recording_id")
        return {
            "sample_id": str(row["sample_id"]),
            "subject_id": str(row["subject_id"]),
            "recording_id": None if recording is None else str(recording),
            "token_inputs": torch.from_numpy(
                np.asarray(row["token_inputs"], dtype=np.float32).reshape(num_tokens, patch_len)
            ),
            "token_channel_indices": torch.from_numpy(
                np.asarray(row["token_channel_indices"], dtype=np.int64)
            ),
            "token_time_indices": torch.from_numpy(
                np.asarray(row["token_time_indices"], dtype=np.int64)
            ),
            "token_valid_mask": torch.from_numpy(
                np.asarray(row["token_valid_mask"], dtype=np.float32)
            ),
        }

    def __getitem__(self, index):
        row = self.ds.take([int(index)], columns=self.COLUMNS).to_pylist()[0]
        return self._convert(row)

    def __getitems__(self, indices):
        if not indices:
            return []
        rows = self.ds.take([int(index) for index in indices], columns=self.COLUMNS).to_pylist()
        return [self._convert(row) for row in rows]


def collate_cache_source(batch):
    return {
        "sample_id": [row["sample_id"] for row in batch],
        "subject_id": [row["subject_id"] for row in batch],
        "recording_id": [row["recording_id"] for row in batch],
        "token_inputs": torch.stack([row["token_inputs"] for row in batch]),
        "token_channel_indices": torch.stack([row["token_channel_indices"] for row in batch]),
        "token_time_indices": torch.stack([row["token_time_indices"] for row in batch]),
        "token_valid_mask": torch.stack([row["token_valid_mask"] for row in batch]),
    }


def _feature_schema(feature_dim):
    return pa.schema([
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("subject_id", pa.string(), nullable=False),
        pa.field("recording_id", pa.string(), nullable=True),
        pa.field("feature", pa.list_(pa.float32(), int(feature_dim)), nullable=False),
        pa.field("num_valid_tokens", pa.int32(), nullable=False),
    ])


def feature_rows_to_table(sample_ids, subject_ids, recording_ids, features, valid_counts):
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError("features must be rank 2")
    feature_array = pa.FixedSizeListArray.from_arrays(
        pa.array(features.reshape(-1), type=pa.float32()), features.shape[1]
    )
    return pa.Table.from_arrays([
        pa.array([str(value) for value in sample_ids], type=pa.string()),
        pa.array([str(value) for value in subject_ids], type=pa.string()),
        pa.array(recording_ids, type=pa.string()),
        feature_array,
        pa.array(np.asarray(valid_counts, dtype=np.int32), type=pa.int32()),
    ], schema=_feature_schema(features.shape[1]))


def _write_feature_batch(path, table, first):
    lance.write_dataset(
        table, str(path), mode="create" if first else "append",
        data_storage_version="stable",
    )


def lance_fingerprint(lance_path, storage_options=None):
    kwargs = {} if storage_options is None else {"storage_options": storage_options}
    dataset = lance.dataset(str(lance_path), **kwargs)
    rows = dataset.to_table(columns=["sample_id"]).to_pylist()
    sample_ids = [str(row["sample_id"]) for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Input Lance sample_id values must be unique")
    sample_fingerprint = sha256_json(sorted(sample_ids))
    fragments = []
    for fragment in dataset.get_fragments():
        fragments.append({
            "fragment_id": int(fragment.fragment_id),
            "physical_rows": int(fragment.physical_rows),
            "data_files": [
                {
                    "path": str(data_file.path),
                    "file_size_bytes": (
                        None if data_file.file_size_bytes is None
                        else int(data_file.file_size_bytes)
                    ),
                    "field_ids": list(data_file.fields),
                    "file_version": [
                        int(data_file.file_major_version),
                        int(data_file.file_minor_version),
                    ],
                }
                for data_file in fragment.data_files()
            ],
        })
    payload = {
        "path": str(lance_path),
        "version": int(dataset.version),
        "schema": str(dataset.schema),
        "row_count": int(dataset.count_rows()),
        "sample_id_fingerprint": sample_fingerprint,
        "fragments": fragments,
    }
    return sha256_json(payload), payload


def model_implementation_fingerprint(project_root):
    paths = [
        Path(project_root) / "scripts" / "downstream_50m" / "model.py",
        Path(project_root) / "scripts" / "downstream_50m" / "largeScale_fineTune.py",
        Path(project_root) / "scripts" / "downstream_50m" / "feature_cache.py",
    ]
    return sha256_json({str(path.relative_to(project_root)): sha256_file(path) for path in paths})


def build_cache_identity(
    *, input_fingerprint, input_row_count, sample_id_fingerprint,
    checkpoint_sha256, model_config, shape_params, probe_layer_idx,
    probe_layer_ratio, model_code_sha256, git_commit,
):
    """Only frozen feature-producing inputs belong in this identity."""
    identity = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "input_lance_fingerprint": input_fingerprint,
        "input_row_count": int(input_row_count),
        "sample_id_fingerprint": sample_id_fingerprint,
        "pretrained_checkpoint_sha256": checkpoint_sha256,
        "backbone_structure_config": model_config,
        "d_model": int(model_config["d_model"]),
        "depth": int(model_config["depth"]),
        "n_heads": int(model_config["n_heads"]),
        "num_tokens": int(shape_params["num_tokens"]),
        "token_input_length": int(shape_params["patch_len"]),
        "probe_layer_idx": int(probe_layer_idx),
        "probe_layer_ratio": float(probe_layer_ratio),
        "model_code_sha256": model_code_sha256,
        "git_commit": git_commit,
        "pooling_implementation_version": POOLING_IMPLEMENTATION_VERSION,
        "token_valid_mask_semantics_version": MASK_SEMANTICS_VERSION,
        "cache_dtype": CACHE_DTYPE,
    }
    return identity, sha256_json(identity)


def _cache_metadata_rows(path):
    dataset = lance.dataset(str(path))
    return dataset.to_table(
        columns=["sample_id", "subject_id", "recording_id", "num_valid_tokens"]
    ).to_pylist()


def _validate_feature_lance(path, expected_dim, expected_count, expected_sample_ids):
    dataset = lance.dataset(str(path))
    schema = dataset.schema
    if set(schema.names).intersection(PROHIBITED_CACHE_COLUMNS):
        raise ValueError("Feature cache contains task/split/prediction columns")
    if schema != _feature_schema(expected_dim):
        raise ValueError(f"Feature cache schema mismatch: {schema}")
    if dataset.count_rows() != expected_count:
        raise ValueError("Feature cache row count mismatch")
    rows = _cache_metadata_rows(path)
    sample_ids = [str(row["sample_id"]) for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Feature cache contains duplicate sample_id")
    if set(sample_ids) != set(expected_sample_ids):
        raise ValueError("Feature cache sample_id set mismatch")
    for batch in dataset.to_batches(columns=["feature"], batch_size=8):
        features = np.asarray(batch.column(0).to_pylist(), dtype=np.float32)
        if features.shape[1:] != (expected_dim,):
            raise ValueError("Feature cache feature shape mismatch")
        if not np.isfinite(features).all():
            raise ValueError("Feature cache contains NaN or Inf")
    return str(schema)


def _read_features_by_sample(path, sample_ids):
    dataset = lance.dataset(str(path))
    ids = [str(row["sample_id"]) for row in dataset.to_table(columns=["sample_id"]).to_pylist()]
    positions = {sample_id: index for index, sample_id in enumerate(ids)}
    if any(sample_id not in positions for sample_id in sample_ids):
        raise ValueError("Parity sample is missing from feature cache")
    rows = dataset.take(
        [positions[sample_id] for sample_id in sample_ids], columns=["sample_id", "feature"]
    ).to_pylist()
    return {str(row["sample_id"]): np.asarray(row["feature"], dtype=np.float32) for row in rows}


def _choose_parity_samples(metadata):
    by_subject = {}
    for row in metadata:
        by_subject.setdefault(str(row["subject_id"]), str(row["sample_id"]))
    required = ["SUB5", "SUB8"]
    missing = [subject for subject in required if subject not in by_subject]
    if missing:
        raise ValueError(f"Formal parity requires subjects absent from input: {missing}")
    normal = next((subject for subject in sorted(by_subject) if subject not in required), None)
    if normal is None:
        raise ValueError("Formal parity requires a normal subject")
    return [by_subject[normal], by_subject["SUB5"], by_subject["SUB8"]]


def _parity_report(parity_ids, online_mean, online_flatten, mean_path, flatten_path):
    cached_mean = _read_features_by_sample(mean_path, parity_ids)
    cached_flatten = _read_features_by_sample(flatten_path, parity_ids)
    mean_diffs, flatten_diffs = [], []
    for sample_id in parity_ids:
        left_mean = torch.from_numpy(online_mean[sample_id])
        right_mean = torch.from_numpy(cached_mean[sample_id])
        left_flatten = torch.from_numpy(online_flatten[sample_id])
        right_flatten = torch.from_numpy(cached_flatten[sample_id])
        if not torch.allclose(left_mean, right_mean, atol=1e-5, rtol=1e-5):
            raise RuntimeError(f"Mean feature parity failed for {sample_id}")
        if not torch.allclose(left_flatten, right_flatten, atol=1e-5, rtol=1e-5):
            raise RuntimeError(f"Flatten feature parity failed for {sample_id}")
        mean_diffs.append(float((left_mean - right_mean).abs().max().item()))
        flatten_diffs.append(float((left_flatten - right_flatten).abs().max().item()))

    generator_state = torch.random.get_rng_state()
    torch.manual_seed(1729)
    mean_head = nn.Linear(next(iter(online_mean.values())).shape[0], 2)
    flatten_head = nn.Linear(next(iter(online_flatten.values())).shape[0], 2)
    torch.random.set_rng_state(generator_state)
    mean_logit_diffs, flatten_logit_diffs = [], []
    with torch.inference_mode():
        for sample_id in parity_ids:
            mean_online_logits = mean_head(torch.from_numpy(online_mean[sample_id]))
            mean_cached_logits = mean_head(torch.from_numpy(cached_mean[sample_id]))
            flatten_online_logits = flatten_head(torch.from_numpy(online_flatten[sample_id]))
            flatten_cached_logits = flatten_head(torch.from_numpy(cached_flatten[sample_id]))
            if not torch.allclose(mean_online_logits, mean_cached_logits, atol=1e-5, rtol=1e-5):
                raise RuntimeError(f"Mean logit parity failed for {sample_id}")
            if not torch.allclose(flatten_online_logits, flatten_cached_logits, atol=1e-5, rtol=1e-5):
                raise RuntimeError(f"Flatten logit parity failed for {sample_id}")
            mean_logit_diffs.append(float((mean_online_logits - mean_cached_logits).abs().max()))
            flatten_logit_diffs.append(float((flatten_online_logits - flatten_cached_logits).abs().max()))
    return {
        "parity_sample_ids": parity_ids,
        "mean_max_abs_diff": max(mean_diffs, default=0.0),
        "flatten_max_abs_diff": max(flatten_diffs, default=0.0),
        "mean_logits_max_abs_diff": max(mean_logit_diffs, default=0.0),
        "flatten_logits_max_abs_diff": max(flatten_logit_diffs, default=0.0),
        "parity_passed": True,
    }


def build_feature_cache(
    *, source_dataset, backbone, probe_layer_idx, output_dir, batch_size,
    device, expected_num_tokens, d_model, manifest_base, logger=None,
    log_every_batches=5,
):
    """Stream source windows once and write mean+flatten from each same output batch."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    mean_path = output_dir / MEAN_CACHE_NAME
    flatten_path = output_dir / FLATTEN_CACHE_NAME
    parity_ids = _choose_parity_samples(source_dataset.metadata)
    online_mean, online_flatten = {}, {}
    loader = DataLoader(
        source_dataset, batch_size=int(batch_size), shuffle=False, num_workers=0,
        collate_fn=collate_cache_source, drop_last=False,
    )
    backbone.eval()
    total_rows = len(source_dataset)
    total_batches = math.ceil(total_rows / int(batch_size))
    log_event(
        logger, "INFO", "cache.extract", "starting single feature extraction pass",
        total_rows=total_rows, total_batches=total_batches, batch_size=int(batch_size),
        inference_mode=True, backbone_eval=True,
        detail="mean and flatten derived from the same backbone batch output",
    )
    processed = 0
    backbone_forward_count = 0
    first = True
    extraction_started = time.monotonic()
    with torch.inference_mode():
        for batch_number, batch in enumerate(loader, start=1):
            token_inputs = batch["token_inputs"].to(device)
            channel_indices = batch["token_channel_indices"].to(device)
            time_indices = batch["token_time_indices"].to(device)
            valid_mask = batch["token_valid_mask"].to(device)
            embeddings = backbone.extract_token_embeddings(
                token_inputs=token_inputs,
                token_channel_indices=channel_indices,
                token_time_indices=time_indices,
                return_layer_idx=probe_layer_idx,
            )
            backbone_forward_count += 1
            mean_features, flatten_features = masked_pool_both(embeddings, valid_mask)
            if mean_features.shape[1] != d_model:
                raise RuntimeError("Mean feature dimension mismatch")
            if flatten_features.shape[1] != expected_num_tokens * d_model:
                raise RuntimeError("Flatten feature dimension mismatch")
            invalid = ~valid_mask.bool()
            flattened_by_token = flatten_features.reshape(
                flatten_features.shape[0], expected_num_tokens, d_model
            )
            if torch.count_nonzero(flattened_by_token[invalid]).item() != 0:
                raise RuntimeError("Invalid token values are not exactly zero in flatten features")
            mean_np = mean_features.cpu().numpy().astype(np.float32, copy=False)
            flatten_np = flatten_features.cpu().numpy().astype(np.float32, copy=False)
            counts = valid_mask.sum(dim=1).cpu().numpy().astype(np.int32)
            _write_feature_batch(
                mean_path,
                feature_rows_to_table(
                    batch["sample_id"], batch["subject_id"], batch["recording_id"],
                    mean_np, counts,
                ),
                first,
            )
            _write_feature_batch(
                flatten_path,
                feature_rows_to_table(
                    batch["sample_id"], batch["subject_id"], batch["recording_id"],
                    flatten_np, counts,
                ),
                first,
            )
            for index, sample_id in enumerate(batch["sample_id"]):
                if sample_id in parity_ids:
                    online_mean[sample_id] = mean_np[index].copy()
                    online_flatten[sample_id] = flatten_np[index].copy()
            processed += len(batch["sample_id"])
            first = False
            if (
                batch_number == 1
                or batch_number == total_batches
                or batch_number % max(1, int(log_every_batches)) == 0
            ):
                elapsed = time.monotonic() - extraction_started
                throughput = processed / elapsed if elapsed > 0 else 0.0
                eta = (total_rows - processed) / throughput if throughput > 0 else None
                gpu = {}
                if torch.cuda.is_available() and str(device).startswith("cuda"):
                    gpu = {
                        "cuda_allocated_mb": round(torch.cuda.memory_allocated(device) / 1048576, 1),
                        "cuda_reserved_mb": round(torch.cuda.memory_reserved(device) / 1048576, 1),
                    }
                log_event(
                    logger, "INFO", "cache.extract", "feature extraction and Lance write progress",
                    batch=f"{batch_number}/{total_batches}", current_batch=batch_number,
                    total_batches=total_batches, processed_rows=processed,
                    total_rows=total_rows, progress_percent=round(100 * processed / total_rows, 1),
                    elapsed_seconds=round(elapsed, 1), throughput_rows_per_second=round(throughput, 2),
                    eta_seconds=None if eta is None else round(eta, 1),
                    backbone_forward_count=backbone_forward_count, **gpu,
                )
    if processed != len(source_dataset):
        raise RuntimeError("Backbone processed window count does not match source window count")
    expected_ids = [str(row["sample_id"]) for row in source_dataset.metadata]
    log_event(logger, "INFO", "cache.validate", "validating schema, row count, unique IDs and finite values")
    mean_schema = _validate_feature_lance(mean_path, d_model, processed, expected_ids)
    flatten_schema = _validate_feature_lance(
        flatten_path, expected_num_tokens * d_model, processed, expected_ids
    )
    log_event(
        logger, "INFO", "cache.parity", "validating online/cache feature and logits parity",
        parity_sample_ids=parity_ids, atol=1e-5, rtol=1e-5,
    )
    parity = _parity_report(
        parity_ids, online_mean, online_flatten, mean_path, flatten_path
    )
    manifest = {
        **manifest_base,
        "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_window_count": len(source_dataset),
        "backbone_processed_window_count": processed,
        "backbone_forward_count": backbone_forward_count,
        "mean_feature_dim": d_model,
        "flatten_feature_dim": expected_num_tokens * d_model,
        "dtype": CACHE_DTYPE,
        "mean_row_count": processed,
        "flatten_row_count": processed,
        "mean_schema": mean_schema,
        "flatten_schema": flatten_schema,
        "mask_validation_passed": True,
        **parity,
    }
    atomic_write_json(output_dir / MANIFEST_NAME, manifest)
    (output_dir / SUCCESS_NAME).write_text(manifest["cache_key"] + "\n", encoding="utf-8")
    log_event(
        logger, "INFO", "cache.parity", "parity validation passed", **parity,
    )
    log_event(
        logger, "INFO", "cache.validate", "cache validation passed",
        success_marker=str(output_dir / SUCCESS_NAME), mean_rows=processed,
        flatten_rows=processed, sample_ids_unique=True, finite_values=True,
        flatten_invalid_tokens_zero=True,
    )
    return manifest


def validate_cache_manifest(cache_dir, expected_identity=None, expected_key=None):
    cache_dir = Path(cache_dir)
    success = cache_dir / SUCCESS_NAME
    manifest_path = cache_dir / MANIFEST_NAME
    if not success.is_file() or not manifest_path.is_file():
        raise ValueError("Feature cache is incomplete: _SUCCESS or manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError("Feature cache manifest is not completed")
    if expected_key is not None and manifest.get("cache_key") != expected_key:
        raise ValueError("Feature cache key mismatch")
    if success.read_text(encoding="utf-8").strip() != manifest.get("cache_key"):
        raise ValueError("Feature cache _SUCCESS key mismatch")
    if expected_identity is not None:
        if manifest.get("cache_identity") != expected_identity:
            raise ValueError("Feature cache identity mismatch")
        if sha256_json(expected_identity) != manifest.get("cache_key"):
            raise ValueError("Feature cache manifest hash mismatch")
    if not manifest.get("parity_passed"):
        raise ValueError("Feature cache parity was not passed")
    return manifest


def validate_feature_cache(cache_dir, expected_identity=None, expected_key=None, logger=None):
    cache_dir = Path(cache_dir)
    log_event(logger, "INFO", "cache.validate", "validating existing cache",
              cache_path=str(cache_dir))
    manifest = validate_cache_manifest(cache_dir, expected_identity, expected_key)
    row_count = int(manifest["input_row_count"])
    expected_ids = manifest.get("sample_ids")
    if not isinstance(expected_ids, list):
        raise ValueError("Feature cache manifest has no sample_ids integrity list")
    _validate_feature_lance(
        cache_dir / MEAN_CACHE_NAME, int(manifest["mean_feature_dim"]), row_count, expected_ids
    )
    _validate_feature_lance(
        cache_dir / FLATTEN_CACHE_NAME,
        int(manifest["flatten_feature_dim"]), row_count, expected_ids,
    )
    log_event(
        logger, "INFO", "cache.validate", "existing cache integrity validation passed",
        success=True, manifest=True, schema=True, row_count=row_count,
        sample_ids_unique=True, finite_values=True,
        mean_feature_dim=manifest["mean_feature_dim"],
        flatten_feature_dim=manifest["flatten_feature_dim"],
    )
    return manifest


def directory_size(path):
    return sum(item.stat().st_size for item in Path(path).rglob("*") if item.is_file())


def resolve_cache_policy(root, cache_key, policy, identity, timestamp=None):
    """Choose reuse/build destination without ever deleting or overwriting a cache."""
    root = Path(root)
    canonical_dir = root / cache_key
    if policy in {"auto", "require"} and canonical_dir.exists():
        manifest = validate_feature_cache(canonical_dir, identity, cache_key)
        return canonical_dir, manifest, True
    if policy == "require":
        raise FileNotFoundError(f"Required feature cache does not exist: {canonical_dir}")
    if policy == "auto":
        return canonical_dir, None, False
    if policy == "rebuild":
        timestamp = timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        destination = root / f"{cache_key}_rebuild_{timestamp}"
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite feature cache: {destination}")
        return destination, None, False
    raise ValueError(f"Unsupported feature cache policy: {policy}")


def ensure_feature_cache(options):
    """Resolve, validate, or atomically build the one cache shared by the sweep."""
    from configs import Config
    import largeScale_fineTune as training
    from lance_dataset import load_storage_options

    started = time.monotonic()
    logger = getattr(options, "logger", None)
    project_root = Path(__file__).resolve().parents[2]
    storage_options = load_storage_options(options.storage_options)
    log_event(logger, "INFO", "cache.input", "loading input Lance metadata",
              input_lance=str(options.input_lance))
    input_fingerprint, input_info = lance_fingerprint(options.input_lance, storage_options)
    log_event(logger, "INFO", "cache.input", "input Lance metadata loaded",
              input_rows=input_info["row_count"])
    git_commit, dirty = get_git_state(project_root)
    cfg = Config()
    cfg.data.lance_path = str(options.input_lance)
    cfg.data.storage_options_path = options.storage_options
    cfg.finetune.pretrained_ckpt_path = str(Path(options.checkpoint).resolve())
    cfg.finetune.device = options.device
    cfg.finetune.probe_layer_ratio = 0.75
    shape_params = training.compute_shape_params(cfg)
    probe_layer_idx = training.compute_probe_layer_idx(
        cfg.model.depth, cfg.finetune.probe_layer_ratio
    )
    checkpoint_sha256 = sha256_file(options.checkpoint)
    log_event(logger, "INFO", "cache.checkpoint", "checkpoint SHA256 computed",
              checkpoint=str(options.checkpoint), checkpoint_sha256=checkpoint_sha256)
    identity, cache_key = build_cache_identity(
        input_fingerprint=input_fingerprint,
        input_row_count=input_info["row_count"],
        sample_id_fingerprint=input_info["sample_id_fingerprint"],
        checkpoint_sha256=checkpoint_sha256,
        model_config={
            "d_model": cfg.model.d_model, "depth": cfg.model.depth,
            "n_heads": cfg.model.n_heads, "mlp_ratio": cfg.model.mlp_ratio,
            "dropout": cfg.model.dropout,
        },
        shape_params=shape_params,
        probe_layer_idx=probe_layer_idx,
        probe_layer_ratio=cfg.finetune.probe_layer_ratio,
        model_code_sha256=model_implementation_fingerprint(project_root),
        git_commit=git_commit,
    )
    root = Path(options.feature_cache_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    policy = options.feature_cache_policy
    canonical_dir = root / cache_key
    log_event(
        logger, "INFO", "cache.lookup", "looking up feature cache",
        cache_key=cache_key, cache_key_prefix=cache_key[:12],
        cache_path=str(canonical_dir), cache_policy=policy,
    )
    try:
        final_dir, manifest, reused = resolve_cache_policy(
            root, cache_key, policy, identity
        )
    except Exception:
        log_exception(logger, "cache.lookup", "feature cache lookup/validation failed",
                      cache_path=str(canonical_dir), cache_policy=policy)
        raise
    if reused:
        log_event(
            logger, "INFO", "cache.lookup", "cache hit: feature extraction skipped",
            cache_path=str(final_dir), cache_policy=policy,
            row_count=manifest["mean_row_count"], schema_valid=True,
            integrity_valid=True,
        )
        return {
            "feature_cache_key": cache_key,
            "feature_cache_path": str(final_dir),
            "feature_cache_policy": policy,
            "feature_cache_reused": True,
            "cache_build_seconds": 0.0,
            "manifest": manifest,
        }
    log_event(
        logger, "INFO", "cache.lookup", "cache miss: building feature cache",
        cache_path=str(final_dir), cache_policy=policy,
        reason="matching complete cache not found" if policy == "auto" else "rebuild requested",
    )
    if final_dir.exists():
        raise FileExistsError(f"Refusing to overwrite feature cache: {final_dir}")
    temporary_dir = root / f".{final_dir.name}.building.{uuid.uuid4().hex}"
    source = TokenFeatureSourceDataset(options.input_lance, storage_options)
    manifest_base = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "cache_key": cache_key,
        "cache_identity": identity,
        "actual_command": subprocess.list2cmdline([sys.executable, *sys.argv]),
        "git_commit": git_commit,
        "dirty_worktree": dirty,
        "input_lance": str(options.input_lance),
        "input_lance_fingerprint": input_fingerprint,
        "input_row_count": input_info["row_count"],
        "sample_id_fingerprint": input_info["sample_id_fingerprint"],
        "sample_ids": [str(row["sample_id"]) for row in source.metadata],
        "pretrained_checkpoint": str(Path(options.checkpoint).resolve()),
        "pretrained_checkpoint_sha256": checkpoint_sha256,
        "probe_layer_idx": probe_layer_idx,
        "probe_layer_ratio": cfg.finetune.probe_layer_ratio,
        "num_tokens": shape_params["num_tokens"],
        "d_model": cfg.model.d_model,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "lance_version": lance.__version__,
    }
    try:
        device = torch.device(options.device)
        log_event(logger, "INFO", "cache.backbone", "constructing frozen 50M backbone",
                  device=str(device), probe_layer_idx=probe_layer_idx)
        backbone, checkpoint_report = training.build_backbone(
            cfg, shape_params, device, load_pretrained=True
        )
        manifest_base.update({
            "checkpoint_matched_key_count": checkpoint_report["matched_key_count"],
            "checkpoint_matched_parameter_fraction": checkpoint_report["matched_parameter_fraction"],
        })
        log_event(
            logger, "INFO", "cache.backbone", "pretrained checkpoint loaded",
            checkpoint_sha256=checkpoint_sha256,
            matched_key_count=checkpoint_report["matched_key_count"],
            matched_parameter_fraction=checkpoint_report["matched_parameter_fraction"],
            backbone_eval=True, inference_mode=True,
        )
        build_feature_cache(
            source_dataset=source,
            backbone=backbone,
            probe_layer_idx=probe_layer_idx,
            output_dir=temporary_dir,
            batch_size=options.feature_cache_batch_size,
            device=device,
            expected_num_tokens=shape_params["num_tokens"],
            d_model=cfg.model.d_model,
            manifest_base=manifest_base,
            logger=logger,
            log_every_batches=getattr(options, "cache_log_every_batches", 5),
        )
        del backbone
        manifest = validate_feature_cache(temporary_dir, identity, cache_key, logger=logger)
        log_event(logger, "INFO", "cache.publish", "atomically publishing completed cache",
                  temporary_path=str(temporary_dir), cache_path=str(final_dir))
        os.replace(temporary_dir, final_dir)
    except Exception as exc:
        temporary_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(temporary_dir / "BUILD_FAILED.json", {
            "status": "failed", "error": repr(exc),
            "failed_at": datetime.now(timezone.utc).isoformat(),
        })
        log_exception(logger, "cache.failed", "feature cache build failed",
                      cache_path=str(temporary_dir), error=repr(exc))
        raise
    elapsed = time.monotonic() - started
    log_event(
        logger, "INFO", "cache.complete", "feature cache build completed",
        processed_rows=manifest["backbone_processed_window_count"],
        backbone_forward_count=manifest["backbone_forward_count"],
        mean_shape=f"[{manifest['mean_row_count']},{manifest['mean_feature_dim']}]",
        flatten_shape=f"[{manifest['flatten_row_count']},{manifest['flatten_feature_dim']}]",
        mean_cache_path=str(final_dir / MEAN_CACHE_NAME),
        flatten_cache_path=str(final_dir / FLATTEN_CACHE_NAME),
        cache_size_bytes=directory_size(final_dir), elapsed_seconds=round(elapsed, 1),
        success_path=str(final_dir / SUCCESS_NAME),
    )
    return {
        "feature_cache_key": cache_key,
        "feature_cache_path": str(final_dir),
        "feature_cache_policy": policy,
        "feature_cache_reused": False,
        "cache_build_seconds": elapsed,
        "manifest": manifest,
    }


class CachedFeatureDataset(Dataset):
    """Join label-free cached features to source labels by sample_id, never row number."""

    def __init__(
        self, cache_dir, pooling, input_lance, task, storage_options=None,
        expected_cache_key=None,
    ):
        if pooling not in {"mean", "flatten"}:
            raise ValueError(f"Unsupported cached pooling: {pooling}")
        self.cache_dir = Path(cache_dir)
        self.pooling = pooling
        self.manifest = validate_cache_manifest(
            self.cache_dir, expected_key=expected_cache_key
        )
        cache_name = MEAN_CACHE_NAME if pooling == "mean" else FLATTEN_CACHE_NAME
        self.ds = lance.dataset(str(self.cache_dir / cache_name))
        expected_dim = int(
            self.manifest["mean_feature_dim"]
            if pooling == "mean" else self.manifest["flatten_feature_dim"]
        )
        if self.ds.schema != _feature_schema(expected_dim):
            raise ValueError("Selected feature cache schema mismatch")
        if self.ds.count_rows() != int(self.manifest["input_row_count"]):
            raise ValueError("Selected feature cache row count mismatch")
        kwargs = {} if storage_options is None else {"storage_options": storage_options}
        source = lance.dataset(str(input_lance), **kwargs)
        source_rows = source.to_table(
            columns=["sample_id", "subject_id", "age", "sex"]
        ).to_pylist()
        source_by_id = {str(row["sample_id"]): row for row in source_rows}
        cache_rows = self.ds.to_table(columns=["sample_id", "subject_id"]).to_pylist()
        self.metadata = []
        for index, cached in enumerate(cache_rows):
            sample_id = str(cached["sample_id"])
            if sample_id not in source_by_id:
                raise ValueError(f"Cached sample absent from source labels: {sample_id}")
            source_row = source_by_id[sample_id]
            if str(cached["subject_id"]) != str(source_row["subject_id"]):
                raise ValueError(f"Cached subject mismatch for {sample_id}")
            if task == "sex_classification":
                sex = str(source_row["sex"]).strip().lower()
                if sex not in {"female", "male"}:
                    raise ValueError(f"Unknown sex label for {sample_id}: {sex!r}")
                label = 0 if sex == "female" else 1
                label_name = sex
            else:
                if source_row["age"] is None:
                    continue
                label = float(source_row["age"])
                label_name = ""
            self.metadata.append({
                "index": index, "sample_id": sample_id,
                "subject_id": str(source_row["subject_id"]),
                "label": label, "label_name": label_name,
            })

    def __len__(self):
        return len(self.metadata)

    def _convert(self, row, metadata):
        feature = np.asarray(row["feature"], dtype=np.float32)
        label = metadata["label"]
        return {
            "sample_id": metadata["sample_id"],
            "subject_id": metadata["subject_id"],
            "recording_id": "" if row.get("recording_id") is None else str(row["recording_id"]),
            "feature": torch.from_numpy(feature),
            "num_valid_tokens": torch.tensor(int(row["num_valid_tokens"]), dtype=torch.int64),
            "label_name": metadata["label_name"],
            "label": torch.tensor(
                int(label) if isinstance(label, int) else float(label),
                dtype=torch.int64 if isinstance(label, int) else torch.float32,
            ),
        }

    def __getitem__(self, local_index):
        metadata = self.metadata[int(local_index)]
        row = self.ds.take(
            [metadata["index"]],
            columns=["sample_id", "recording_id", "feature", "num_valid_tokens"],
        ).to_pylist()[0]
        if str(row["sample_id"]) != metadata["sample_id"]:
            raise RuntimeError("Cached feature sample_id lookup mismatch")
        return self._convert(row, metadata)

    def __getitems__(self, local_indices):
        if not local_indices:
            return []
        metadata = [self.metadata[int(index)] for index in local_indices]
        rows = self.ds.take(
            [row["index"] for row in metadata],
            columns=["sample_id", "recording_id", "feature", "num_valid_tokens"],
        ).to_pylist()
        if [str(row["sample_id"]) for row in rows] != [row["sample_id"] for row in metadata]:
            raise RuntimeError("Cached feature batch sample_id lookup mismatch")
        return [self._convert(row, meta) for row, meta in zip(rows, metadata)]

    def get_subject_label_rows(self, indices=None, chunk_size=50000):
        del chunk_size
        selected = range(len(self.metadata)) if indices is None else [int(value) for value in indices]
        return [dict(self.metadata[index]) for index in selected]
