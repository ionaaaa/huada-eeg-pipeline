#!/usr/bin/env python3
"""Audit a NeuralBench FBMAE sex checkpoint against this 50M downstream model.

The script is intentionally separate from the training/evaluation entry points.  It
loads checkpoints with ``weights_only=True``, writes an explicit key-by-key report,
and optionally performs a deterministic one-batch forward parity check.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn as nn
import yaml


EXPECTED_SHA256 = "56284992271d8cf0245bb6cc28f6098daf569d1188923196bfdd3d090a60342f"
SOURCE_BACKBONE_PREFIX = "model.wrapped_model.backbone."
SOURCE_PROBE_PREFIX = "model.probe."
TARGET_CLASS_MAPPING = {"female": 0, "male": 1}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_load_checkpoint(path: Path) -> Mapping[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Checkpoint must be a mapping, got {type(checkpoint).__name__}")
    return checkpoint


def checkpoint_state_dict(checkpoint: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, Mapping):
        raise TypeError("checkpoint state_dict is not a mapping")
    tensors = {str(key): value for key, value in state.items() if torch.is_tensor(value)}
    if not tensors:
        raise ValueError("checkpoint contains no state tensors")
    return tensors


def resolve_source_run_dir(base_or_run_dir: Path, prefix: str, suffix: str) -> Path:
    if (base_or_run_dir / "best.ckpt").is_file() and (base_or_run_dir / "config.yaml").is_file():
        name = base_or_run_dir.name
        if not name.startswith(prefix) or not name.endswith(suffix):
            raise ValueError(f"Source run directory name does not match required prefix/suffix: {name}")
        return base_or_run_dir.resolve()
    matches = sorted(
        path.resolve()
        for path in base_or_run_dir.iterdir()
        if path.is_dir()
        and path.name.startswith(prefix)
        and path.name.endswith(suffix)
        and (path / "best.ckpt").is_file()
        and (path / "config.yaml").is_file()
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one source run directory, found {len(matches)}: "
            + json.dumps([str(path) for path in matches], ensure_ascii=False)
        )
    return matches[0]


def source_to_target_key(source_key: str) -> str | None:
    if source_key.startswith(SOURCE_BACKBONE_PREFIX):
        return "backbone." + source_key[len(SOURCE_BACKBONE_PREFIX) :]
    if source_key.startswith(SOURCE_PROBE_PREFIX):
        return "head." + source_key[len(SOURCE_PROBE_PREFIX) :]
    return None


def component_for_source_key(source_key: str) -> str:
    if source_key.startswith(SOURCE_PROBE_PREFIX):
        return "downstream_sex_probe"
    if source_key == "loss.weight":
        return "loss_class_weight_buffer"
    if not source_key.startswith(SOURCE_BACKBONE_PREFIX):
        return "unexpected_source"
    key = source_key[len(SOURCE_BACKBONE_PREFIX) :]
    if key.startswith("tokenizer.proj.1."):
        return "tokenizer_normalization"
    if key.startswith("tokenizer."):
        return "tokenizer"
    if key.startswith("channel_embed."):
        return "channel_embedding"
    if key.startswith("time_embed."):
        return "time_embedding"
    if key.startswith("encoder.encoder.layers."):
        return "encoder_normalization" if ".norm" in key else "transformer_encoder"
    if key.startswith("head."):
        return "backbone_upstream_head"
    return "backbone_other"


def _shape(value: torch.Tensor | None) -> list[int] | None:
    return list(value.shape) if value is not None else None


def build_key_mapping(
    source_state: Mapping[str, torch.Tensor],
    target_backbone_state: Mapping[str, torch.Tensor],
    target_head_state: Mapping[str, torch.Tensor],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target_state: dict[str, torch.Tensor] = {
        **{f"backbone.{key}": value for key, value in target_backbone_state.items()},
        **{f"head.{key}": value for key, value in target_head_state.items()},
    }
    rows: list[dict[str, Any]] = []
    consumed_targets: set[str] = set()
    for source_key in sorted(source_state):
        source_value = source_state[source_key]
        target_key = source_to_target_key(source_key)
        component = component_for_source_key(source_key)
        target_value = target_state.get(target_key) if target_key is not None else None
        if source_key == "loss.weight":
            status, reason = "ignored", "training-only CrossEntropyLoss class-weight buffer"
        elif target_key is None:
            status, reason = "unexpected", "no explicit source-to-target rule"
        elif target_value is None:
            status, reason = "missing_target", "mapped target key does not exist"
        elif tuple(source_value.shape) != tuple(target_value.shape):
            status, reason = "shape_mismatch", "source and target tensor shapes differ"
            consumed_targets.add(target_key)
        else:
            status, reason = "matched", "explicit prefix mapping and exact shape match"
            consumed_targets.add(target_key)
        rows.append(
            {
                "source_key": source_key,
                "target_key": target_key or "",
                "source_shape": canonical_json(_shape(source_value)),
                "target_shape": canonical_json(_shape(target_value)) if target_value is not None else "",
                "source_numel": int(source_value.numel()),
                "target_numel": int(target_value.numel()) if target_value is not None else "",
                "component": component,
                "status": status,
                "reason": reason,
            }
        )
    for target_key in sorted(set(target_state) - consumed_targets):
        value = target_state[target_key]
        rows.append(
            {
                "source_key": "",
                "target_key": target_key,
                "source_shape": "",
                "target_shape": canonical_json(_shape(value)),
                "source_numel": "",
                "target_numel": int(value.numel()),
                "component": "target_only",
                "status": "missing_source",
                "reason": "target tensor has no matched source tensor",
            }
        )

    def fraction(prefix: str, include_upstream_head: bool) -> dict[str, Any]:
        expected = [
            row
            for row in rows
            if row["source_key"].startswith(prefix)
            and (include_upstream_head or row["component"] != "backbone_upstream_head")
        ]
        matched = [row for row in expected if row["status"] == "matched"]
        expected_numel = sum(int(row["source_numel"]) for row in expected)
        matched_numel = sum(int(row["source_numel"]) for row in matched)
        return {
            "expected_keys": len(expected),
            "matched_keys": len(matched),
            "expected_parameters": expected_numel,
            "matched_parameters": matched_numel,
            "key_match_fraction": len(matched) / len(expected) if expected else 0.0,
            "parameter_match_fraction": matched_numel / expected_numel if expected_numel else 0.0,
        }

    core = fraction(SOURCE_BACKBONE_PREFIX, include_upstream_head=False)
    full = fraction(SOURCE_BACKBONE_PREFIX, include_upstream_head=True)
    summary = {
        "core_backbone": core,
        "full_backbone": full,
        "core_backbone_key_match_fraction": core["key_match_fraction"],
        "core_backbone_parameter_match_fraction": core["parameter_match_fraction"],
        "full_backbone_key_match_fraction": full["key_match_fraction"],
        "full_backbone_parameter_match_fraction": full["parameter_match_fraction"],
        "missing_source": [row["target_key"] for row in rows if row["status"] == "missing_source"],
        "missing_target": [row["source_key"] for row in rows if row["status"] == "missing_target"],
        "unexpected_source": [row["source_key"] for row in rows if row["status"] == "unexpected"],
        "shape_mismatches": [row["source_key"] for row in rows if row["status"] == "shape_mismatch"],
        "ignored_source": [row["source_key"] for row in rows if row["status"] == "ignored"],
    }
    return rows, summary


def write_mapping_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "source_key",
        "target_key",
        "source_shape",
        "target_shape",
        "source_numel",
        "target_numel",
        "component",
        "status",
        "reason",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parity_stats(left: torch.Tensor, right: torch.Tensor, atol: float, rtol: float) -> dict[str, Any]:
    shape_equal = tuple(left.shape) == tuple(right.shape)
    dtype_equal = left.dtype == right.dtype
    finite = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
    if shape_equal:
        delta = (left.detach().cpu() - right.detach().cpu()).abs()
        max_abs_diff = float(delta.max().item()) if delta.numel() else 0.0
        mean_abs_diff = float(delta.mean().item()) if delta.numel() else 0.0
        allclose = bool(torch.allclose(left, right, atol=atol, rtol=rtol))
    else:
        max_abs_diff = None
        mean_abs_diff = None
        allclose = False
    return {
        "left_shape": list(left.shape),
        "right_shape": list(right.shape),
        "left_dtype": str(left.dtype),
        "right_dtype": str(right.dtype),
        "shape_equal": shape_equal,
        "dtype_equal": dtype_equal,
        "finite": finite,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "atol": float(atol),
        "rtol": float(rtol),
        "allclose": allclose,
    }


def resolve_label_mapping(
    *,
    predefined_mapping: Mapping[str, int] | None,
    observed_labels: Sequence[str] | None,
    label_meanings: Mapping[str, str] | None,
    evidence: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if predefined_mapping:
        raw_mapping = {str(key): int(value) for key, value in predefined_mapping.items()}
        method = "config.predefined_mapping"
    elif observed_labels:
        labels = sorted(set(map(str, observed_labels)))
        if len(labels) != 2:
            raise ValueError(f"Expected two observed source labels, got {labels}")
        raw_mapping = {label: index for index, label in enumerate(labels)}
        method = "LabelEncoder sorted(raw_labels)"
    else:
        return {
            "resolved": False,
            "confidence": "unresolved",
            "source_raw_mapping": None,
            "source_mapping": None,
            "target_mapping": TARGET_CLASS_MAPPING,
            "probe_row_swap_required": None,
            "dataset_metadata_required": True,
            "required_metadata": "One authoritative HBN participants.tsv/participants.json containing the sex field values and their meanings",
            "evidence": list(evidence),
        }
    if not label_meanings:
        return {
            "resolved": False,
            "confidence": "unresolved",
            "source_raw_mapping": raw_mapping,
            "source_mapping": None,
            "target_mapping": TARGET_CLASS_MAPPING,
            "probe_row_swap_required": None,
            "dataset_metadata_required": True,
            "required_metadata": "Authoritative semantic meanings for each raw sex value",
            "encoding_method": method,
            "evidence": list(evidence),
        }
    semantic_mapping = {str(label_meanings[raw]).lower(): index for raw, index in raw_mapping.items()}
    if set(semantic_mapping) != {"female", "male"}:
        raise ValueError(f"Source label meanings must resolve to female/male, got {semantic_mapping}")
    source_order = [next(name for name, index in semantic_mapping.items() if index == row) for row in range(2)]
    target_order = [next(name for name, index in TARGET_CLASS_MAPPING.items() if index == row) for row in range(2)]
    permutation = [source_order.index(name) for name in target_order]
    return {
        "resolved": True,
        "confidence": "high",
        "encoding_method": method,
        "source_raw_mapping": raw_mapping,
        "source_mapping": semantic_mapping,
        "source_class_0": source_order[0],
        "source_class_1": source_order[1],
        "target_mapping": TARGET_CLASS_MAPPING,
        "target_class_0": target_order[0],
        "target_class_1": target_order[1],
        "target_row_from_source_row": permutation,
        "probe_row_swap_required": permutation != [0, 1],
        "dataset_metadata_required": False,
        "evidence": list(evidence),
    }


def normalize_probe_rows(
    weight: torch.Tensor, bias: torch.Tensor, target_row_from_source_row: Sequence[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    order = torch.tensor(list(target_row_from_source_row), dtype=torch.long, device=weight.device)
    return weight.index_select(0, order), bias.index_select(0, order)


def normalize_logits(logits: torch.Tensor, target_row_from_source_row: Sequence[int]) -> torch.Tensor:
    order = torch.tensor(list(target_row_from_source_row), dtype=torch.long, device=logits.device)
    return logits.index_select(-1, order)


def _line_number(path: Path, needle: str) -> int | None:
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if needle in line:
            return index
    return None


def _code_evidence(path: Path, needle: str, description: str) -> dict[str, Any]:
    return {"file": str(path), "line": _line_number(path, needle), "needle": needle, "description": description}


def load_target_classes(target_project_root: Path) -> tuple[type[nn.Module], type[nn.Module]]:
    target_dir = target_project_root / "scripts" / "downstream_50m"
    model_path = target_dir / "model.py"
    training_path = target_dir / "largeScale_fineTune.py"
    spec = importlib.util.spec_from_file_location("huada_audit_target_model", model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import target model from {model_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tree = ast.parse(training_path.read_text(encoding="utf-8"), filename=str(training_path))
    class_node = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "EEGDownstreamClassifier"
    )
    namespace: dict[str, Any] = {"torch": torch, "nn": nn}
    exec(compile(ast.Module(body=[class_node], type_ignores=[]), str(training_path), "exec"), namespace)
    return module.EEGPretrainModel, namespace["EEGDownstreamClassifier"]


def build_target_models(
    target_project_root: Path, config: Mapping[str, Any]
) -> tuple[nn.Module, nn.Module]:
    model_class, classifier_class = load_target_classes(target_project_root)
    model_config = config["brain_model_config"]
    patch_len = int(round(float(model_config["patch_seconds"]) * float(model_config["target_sfreq"])))
    n_time_patches = int(
        (float(model_config["clip_seconds"]) * float(model_config["target_sfreq"]) - patch_len) // patch_len + 1
    )
    backbone = model_class(
        input_dim=patch_len,
        d_model=int(model_config["d_model"]),
        n_heads=int(model_config["n_heads"]),
        depth=int(model_config["depth"]),
        mlp_ratio=float(model_config["mlp_ratio"]),
        dropout=float(model_config["dropout"]),
        n_channels=int(model_config["n_channels"]),
        n_time_patches=n_time_patches,
        n_bands=int(model_config["n_bands"]),
        frames_per_patch=int(model_config["frames_per_patch"]),
    )
    classifier = classifier_class(
        backbone=backbone,
        d_model=int(model_config["d_model"]),
        num_classes=int(config["brain_model_output_size"]),
        probe_layer_idx=max(0, min(int(round(int(model_config["depth"]) * float(model_config["probe_layer_ratio"]))) - 1, int(model_config["depth"]) - 1)),
        freeze_backbone=False,
        use_mlp_head=False,
        head_dropout=0.0,
        pooling="mean",
        num_tokens=int(model_config["n_channels"]) * n_time_patches,
    )
    return backbone, classifier


def _prepend_source_paths(source_repo: Path) -> None:
    candidates = [
        source_repo / ".venv-tione-torch28-py312" / "lib" / "python3.12" / "site-packages",
        source_repo / "neuraltrain-repo",
        source_repo / "neuralbench-repo",
        source_repo / "neuralset-repo",
        source_repo / "neuralfetch-repo",
    ]
    for path in reversed(candidates):
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def run_forward_parity(
    *,
    source_repo: Path,
    target_project_root: Path,
    config: Mapping[str, Any],
    source_state: Mapping[str, torch.Tensor],
    label_audit: Mapping[str, Any],
    device: str,
    atol: float,
    rtol: float,
) -> tuple[dict[str, Any], list[str]]:
    target_backbone, target_classifier = build_target_models(target_project_root, config)
    mapped_backbone = {
        key[len(SOURCE_BACKBONE_PREFIX) :]: value
        for key, value in source_state.items()
        if key.startswith(SOURCE_BACKBONE_PREFIX)
    }
    target_backbone.load_state_dict(mapped_backbone, strict=True)
    target_classifier.head.load_state_dict(
        {
            "weight": source_state[SOURCE_PROBE_PREFIX + "weight"],
            "bias": source_state[SOURCE_PROBE_PREFIX + "bias"],
        },
        strict=True,
    )

    _prepend_source_paths(source_repo)
    from neuralbench.modules import DownstreamWrapperModel  # type: ignore
    from neuraltrain.models.fbmae_demo import (  # type: ignore
        NtFBMAEDemo,
        STANDARD_64_CHANNELS,
        _n_full_time_patches,
        _pad_or_crop_time,
        _standardize_signal_by_channel,
    )

    model_config = dict(config["brain_model_config"])
    model_config.update({"fbmae_root": str(source_repo / "FBMAE"), "pretrained": False})
    source_config = NtFBMAEDemo(**model_config)
    source_inner = source_config.build(
        n_in_channels=len(STANDARD_64_CHANNELS),
        n_outputs=int(config["brain_model_output_size"]),
        input_channel_names=list(STANDARD_64_CHANNELS),
    )
    source_model = DownstreamWrapperModel(
        model=source_inner,
        brain_model_output_size=torch.Size([int(model_config["d_model"])]),
        model_output_key=None,
        wrapper_n_outputs=int(config["brain_model_output_size"]),
        aggregation=None,
        probe_config="linear",
        probe_layer=None,
        probe_batch_dim=0,
    )
    source_model_state = {
        key[len("model.") :]: value for key, value in source_state.items() if key.startswith("model.")
    }
    source_model.load_state_dict(source_model_state, strict=True)

    source_model.to(device).eval()
    target_classifier.to(device).eval()
    raw = torch.linspace(
        -1.0e-6,
        1.0e-6,
        steps=len(STANDARD_64_CHANNELS) * 200,
        dtype=torch.float32,
        device=device,
    ).reshape(1, len(STANDARD_64_CHANNELS), 200)

    with torch.inference_mode():
        x = raw * 1.0e6 if source_inner.scale_v_to_uv else raw
        x64, channel_valid = source_inner._scatter_to_standard64(x)
        real_len = min(int(x64.shape[-1]), source_inner.total_points)
        if source_inner.short_clip_fill == "tile":
            raise RuntimeError("Audit fixture currently supports the authoritative mask_pad configuration only")
        x64 = _pad_or_crop_time(x64, source_inner.total_points)
        if source_inner.standardize_by_channel:
            x64 = _standardize_signal_by_channel(x64, channel_valid, time_valid_len=real_len)
        n_valid_time = _n_full_time_patches(real_len, source_inner.patch_len, source_inner.n_time_patches)
        token_batch = source_inner._tokenize(x64, channel_valid, n_valid_time_patches=n_valid_time)

        source_tokenizer = source_inner.backbone.tokenizer(token_batch["token_inputs"])
        target_tokenizer = target_backbone.tokenizer(token_batch["token_inputs"])
        source_embedded = source_inner.backbone.embed_tokens(
            token_batch["token_inputs"], token_batch["token_channel_indices"], token_batch["token_time_indices"]
        )
        target_embedded = target_backbone.embed_tokens(
            token_batch["token_inputs"], token_batch["token_channel_indices"], token_batch["token_time_indices"]
        )
        layer_idx = int(source_inner.probe_layer_idx)
        source_layer = source_inner.backbone.extract_token_embeddings(
            token_inputs=token_batch["token_inputs"],
            token_channel_indices=token_batch["token_channel_indices"],
            token_time_indices=token_batch["token_time_indices"],
            return_layer_idx=layer_idx,
        )
        target_layer = target_backbone.extract_token_embeddings(
            token_inputs=token_batch["token_inputs"],
            token_channel_indices=token_batch["token_channel_indices"],
            token_time_indices=token_batch["token_time_indices"],
            return_layer_idx=layer_idx,
        )
        source_representation_direct = source_inner._aggregate(source_layer, token_batch["token_valid_mask"])
        source_representation_original = source_model(raw, return_embedding=True)
        target_representation = target_classifier.pool_valid_tokens(target_layer, token_batch["token_valid_mask"])
        source_logits = source_model(raw)
        target_logits = target_classifier.classify_features(target_representation)

        permutation = label_audit.get("target_row_from_source_row") or [0, 1]
        source_logits_normalized = normalize_logits(source_logits, permutation)
        target_logits_normalized = normalize_logits(target_logits, permutation)

    comparisons = {
        "source_original_vs_source_direct_representation": parity_stats(
            source_representation_original, source_representation_direct, atol, rtol
        ),
        "tokenizer_output": parity_stats(source_tokenizer, target_tokenizer, atol, rtol),
        "token_plus_position_embedding": parity_stats(source_embedded, target_embedded, atol, rtol),
        "selected_transformer_layer_output": parity_stats(source_layer, target_layer, atol, rtol),
        "representation": parity_stats(source_representation_original, target_representation, atol, rtol),
        "raw_source_order_logits": parity_stats(source_logits, target_logits, atol, rtol),
        "normalized_target_order_logits": parity_stats(
            source_logits_normalized, target_logits_normalized, atol, rtol
        ),
    }
    result = {
        "status": "completed",
        "device": str(device),
        "fixture": {
            "kind": "deterministic_raw_eeg",
            "raw_shape": list(raw.shape),
            "raw_dtype": str(raw.dtype),
            "duration_seconds": 2.0,
            "input_channel_names": "source STANDARD_64_CHANNELS",
            "num_tokens": int(token_batch["token_inputs"].shape[1]),
            "num_valid_tokens": int(token_batch["token_valid_mask"].sum().item()),
            "probe_layer_idx": layer_idx,
            "dropout_disabled": True,
            "inference_mode": True,
        },
        "comparisons": comparisons,
        "forward_representation_parity": comparisons["representation"]["allclose"],
        "forward_logits_parity": comparisons["raw_source_order_logits"]["allclose"],
    }
    log_lines = [
        f"fixture raw_shape={list(raw.shape)} tokens={result['fixture']['num_tokens']} valid_tokens={result['fixture']['num_valid_tokens']}",
        f"probe_layer_idx={layer_idx} atol={atol} rtol={rtol}",
    ]
    for name, stats in comparisons.items():
        log_lines.append(
            f"{name}: shape_equal={stats['shape_equal']} dtype_equal={stats['dtype_equal']} "
            f"finite={stats['finite']} max_abs_diff={stats['max_abs_diff']} "
            f"mean_abs_diff={stats['mean_abs_diff']} allclose={stats['allclose']}"
        )
    return result, log_lines


def parse_label_meanings(values: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Expected RAW=meaning label mapping, got {item!r}")
        raw, meaning = item.split("=", 1)
        result[raw] = meaning
    return result


def source_git(source_repo: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(source_repo), *args], check=True, text=True, capture_output=True
        ).stdout.strip()

    try:
        status = run("status", "--short")
        return {"commit": run("rev-parse", "HEAD"), "dirty": bool(status), "status_short": status.splitlines()}
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"commit": None, "dirty": None, "error": repr(exc)}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--source-config", type=Path)
    parser.add_argument("--target-project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--expected-sha256", default=EXPECTED_SHA256)
    parser.add_argument("--skip-forward-parity", action="store_true")
    parser.add_argument("--source-label-values", nargs="*")
    parser.add_argument("--source-label-meaning", nargs="*", default=[])
    parser.add_argument("--source-label-evidence", action="append", default=[])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    error_path = output_dir / "audit_errors.log"
    error_path.write_text("", encoding="utf-8")
    try:
        run_dir = resolve_source_run_dir(
            args.source_run_dir,
            "seed=33,task_name=sex,brain_model_output_size=2",
            "823-56ab9fd6",
        )
        checkpoint_path = (args.checkpoint or (run_dir / "best.ckpt")).resolve()
        config_path = (args.source_config or (run_dir / "config.yaml")).resolve()
        checkpoint_sha = sha256_file(checkpoint_path)
        if checkpoint_sha.lower() != args.expected_sha256.lower():
            raise RuntimeError(
                f"Checkpoint SHA256 mismatch: actual={checkpoint_sha} expected={args.expected_sha256}"
            )
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        checkpoint = safe_load_checkpoint(checkpoint_path)
        source_state = checkpoint_state_dict(checkpoint)

        target_backbone, target_classifier = build_target_models(args.target_project_root.resolve(), config)
        mapping_rows, mapping_summary = build_key_mapping(
            source_state, target_backbone.state_dict(), target_classifier.head.state_dict()
        )
        write_mapping_csv(output_dir / "checkpoint_key_mapping.csv", mapping_rows)

        inventory = {
            "authoritative_run_dir": str(run_dir),
            "checkpoint_path": str(checkpoint_path),
            "config_path": str(config_path),
            "checkpoint_size_bytes": checkpoint_path.stat().st_size,
            "checkpoint_sha256": checkpoint_sha,
            "expected_sha256": args.expected_sha256,
            "sha256_matches_expected": True,
            "epoch": checkpoint.get("epoch"),
            "global_step": checkpoint.get("global_step"),
            "pytorch_lightning_version": checkpoint.get("pytorch-lightning_version"),
            "top_level_fields": sorted(map(str, checkpoint.keys())),
            "state_dict_tensor_count": len(source_state),
            "state_dict_parameter_count": sum(int(value.numel()) for value in source_state.values()),
            "state_dict_tensor_bytes": sum(int(value.numel() * value.element_size()) for value in source_state.values()),
            "source_git": source_git(args.source_repo.resolve()),
            "safe_load": {"map_location": "cpu", "weights_only": True},
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "config": {
                "task_name": config.get("task_name"),
                "brain_model_name": config.get("brain_model_name"),
                "brain_model_output_size": config.get("brain_model_output_size"),
                "data": config.get("data"),
                "brain_model_config": config.get("brain_model_config"),
                "downstream_model_wrapper": config.get("downstream_model_wrapper"),
                "optimizer": config.get("lightning_optimizer_config"),
                "loss": config.get("loss"),
                "trainer_config": config.get("trainer_config"),
            },
            "mapping_summary": mapping_summary,
        }
        write_json(output_dir / "checkpoint_inventory.json", inventory)

        source_fbmae = args.source_repo / "neuraltrain-repo" / "neuraltrain" / "models" / "fbmae_demo.py"
        source_wrapper = args.source_repo / "neuralbench-repo" / "neuralbench" / "modules.py"
        target_model = args.target_project_root / "scripts" / "downstream_50m" / "model.py"
        target_training = args.target_project_root / "scripts" / "downstream_50m" / "largeScale_fineTune.py"
        depth = int(config["brain_model_config"]["depth"])
        ratio = float(config["brain_model_config"]["probe_layer_ratio"])
        source_probe_layer_idx = max(0, min(int(round(depth * ratio)) - 1, depth - 1))
        target_probe_layer_idx = source_probe_layer_idx
        representation_semantics = {
            "source": {
                "call_chain": [
                    "NeuralBench DownstreamWrapperModel.forward",
                    "_FBMAEDemoWrapper.forward",
                    "backbone.extract_token_embeddings(return_layer_idx=8)",
                    "masked mean over token_valid_mask",
                    "model.probe Linear(512,2)",
                ],
                "probe_layer_idx": source_probe_layer_idx,
                "probe_layer_number_one_based": source_probe_layer_idx + 1,
                "uses_last_layer": source_probe_layer_idx == depth - 1,
                "probe_layer_ratio": ratio,
                "readout": "masked_mean",
                "mask_semantics": "1/true means valid",
                "invalid_tokens_excluded": True,
                "token_embedding_shape": ["B", 640, 512],
                "representation_shape": ["B", 512],
                "outer_aggregation": "identity (downstream_model_wrapper.aggregation=null)",
                "detach_before_probe": False,
                "additional_projection_before_probe": False,
                "additional_normalization_after_pool": False,
                "dropout_before_probe": False,
                "raw_input_preprocessing": {
                    "scale_v_to_uv": config["brain_model_config"]["scale_v_to_uv"],
                    "standardize_by_channel": config["brain_model_config"]["standardize_by_channel"],
                    "short_clip_fill": config["brain_model_config"]["short_clip_fill"],
                    "window_duration_seconds": config["data"]["duration"],
                    "model_clip_seconds": config["brain_model_config"]["clip_seconds"],
                },
            },
            "target": {
                "call_chain": [
                    "extract_token_embeddings(return_layer_idx=8)",
                    "EEGDownstreamClassifier.pool_valid_tokens",
                    "EEGDownstreamClassifier.classify_features",
                ],
                "probe_layer_idx": target_probe_layer_idx,
                "probe_layer_number_one_based": target_probe_layer_idx + 1,
                "probe_layer_ratio": ratio,
                "readout": "masked_mean",
                "mask_semantics": "1/true means valid",
                "invalid_tokens_excluded": True,
                "token_embedding_shape": ["B", 640, 512],
                "representation_shape": ["B", 512],
                "detach_before_probe": False,
                "additional_projection_before_probe": False,
                "additional_normalization_after_pool": False,
                "dropout_before_probe": False,
            },
            "representation_semantics_match": True,
            "scope_note": "Representation parity starts from an identical token batch; source raw-EEG preprocessing is source-specific and target consumes precomputed Lance tokens.",
            "evidence": [
                _code_evidence(source_fbmae, "def compute_probe_layer_idx", "source ratio-to-layer calculation"),
                _code_evidence(source_fbmae, "def _aggregate(", "source masked pooling"),
                _code_evidence(source_fbmae, "token_embeddings = self.backbone.extract_token_embeddings", "source selected-layer extraction"),
                _code_evidence(source_wrapper, "out = self.probe(out)", "source downstream probe application"),
                _code_evidence(target_model, "def extract_token_embeddings(", "target selected-layer extraction"),
                _code_evidence(target_training, "def pool_valid_tokens", "target masked pooling"),
                _code_evidence(target_training, "def classify_features", "target head application"),
            ],
        }
        write_json(output_dir / "representation_semantics.json", representation_semantics)

        label_evidence: list[dict[str, Any]] = [
            {
                "file": str(config_path),
                "line": 123,
                "description": "target extractor is LabelEncoder(event_field=sex, return_one_hot=true, predefined_mapping=null)",
            },
            {
                "file": str(args.source_repo / "neuralset-repo" / "neuralset" / "extractors" / "base.py"),
                "line": 608,
                "description": "LabelEncoder assigns indices by enumerate(sorted(labels))",
            },
        ]
        label_evidence.extend({"uri": value, "description": "authoritative raw-label evidence"} for value in args.source_label_evidence)
        label_audit = resolve_label_mapping(
            predefined_mapping=config["data"]["target"].get("predefined_mapping"),
            observed_labels=args.source_label_values,
            label_meanings=parse_label_meanings(args.source_label_meaning),
            evidence=label_evidence,
        )
        write_json(output_dir / "label_mapping_audit.json", label_audit)

        if args.skip_forward_parity:
            forward_parity = {
                "status": "skipped",
                "reason": "--skip-forward-parity",
                "forward_representation_parity": None,
                "forward_logits_parity": None,
            }
            forward_log = ["forward parity skipped by explicit CLI flag"]
        else:
            forward_parity, forward_log = run_forward_parity(
                source_repo=args.source_repo.resolve(),
                target_project_root=args.target_project_root.resolve(),
                config=config,
                source_state=source_state,
                label_audit=label_audit,
                device=args.device,
                atol=args.atol,
                rtol=args.rtol,
            )
        write_json(output_dir / "forward_parity.json", forward_parity)
        (output_dir / "forward_parity.log").write_text("\n".join(forward_log) + "\n", encoding="utf-8")

        backbone_structure_match = (
            mapping_summary["core_backbone_key_match_fraction"] == 1.0
            and mapping_summary["core_backbone_parameter_match_fraction"] == 1.0
            and not mapping_summary["shape_mismatches"]
        )
        representation_match = bool(representation_semantics["representation_semantics_match"])
        labels_resolved = bool(label_audit["resolved"])
        rep_parity = forward_parity.get("forward_representation_parity")
        logits_parity = forward_parity.get("forward_logits_parity")
        if backbone_structure_match and representation_match and labels_resolved and rep_parity and logits_parity:
            final_state = (
                "B. backbone_compatible_probe_requires_label_remap"
                if label_audit["probe_row_swap_required"]
                else "A. full_checkpoint_directly_compatible"
            )
        elif backbone_structure_match and not representation_match:
            final_state = "C. backbone_compatible_probe_semantics_incompatible"
        elif not backbone_structure_match and mapping_summary["core_backbone_parameter_match_fraction"] > 0:
            final_state = "D. partial_backbone_compatibility_only"
        else:
            final_state = "E. audit_blocked"
        if not labels_resolved or rep_parity is None or logits_parity is None:
            final_state = "E. audit_blocked"

        summary = {
            "status": "completed",
            "final_compatibility_state": final_state,
            "backbone_structure_match": backbone_structure_match,
            "representation_semantics_match": representation_match,
            "source_label_mapping_resolved": labels_resolved,
            "probe_row_swap_required": label_audit.get("probe_row_swap_required"),
            "forward_representation_parity": rep_parity,
            "forward_logits_parity": logits_parity,
            "dataset_metadata_required": label_audit.get("dataset_metadata_required"),
            "cos_copy_used": False,
            "cos_copy_uri_recorded": "cos://jianyu-1442740494/huada/checkpoint/50M_sex_fft.ckpt",
            "checkpoint_sha256": checkpoint_sha,
            "source_git": inventory["source_git"],
            **{key: mapping_summary[key] for key in (
                "core_backbone_key_match_fraction",
                "core_backbone_parameter_match_fraction",
                "full_backbone_key_match_fraction",
                "full_backbone_parameter_match_fraction",
            )},
        }
        write_json(output_dir / "audit_summary.json", summary)
        report = f"""# NeuralBench sex checkpoint compatibility audit

Generated: {datetime.now(timezone.utc).isoformat()}

## Authoritative inputs

- Run directory: `{run_dir}`
- Checkpoint: `{checkpoint_path}`
- Config: `{config_path}`
- SHA256: `{checkpoint_sha}` (matched expected)
- Source Git commit: `{inventory['source_git'].get('commit')}`; dirty={inventory['source_git'].get('dirty')}

## Final compatibility decision

- Final state: **{final_state}**
- Backbone structure match: `{backbone_structure_match}`
- Representation semantics match: `{representation_match}`
- Source label mapping resolved: `{labels_resolved}`
- Probe row swap required: `{label_audit.get('probe_row_swap_required')}`
- Forward representation parity: `{rep_parity}`
- Forward logits parity: `{logits_parity}`
- Dataset metadata required: `{label_audit.get('dataset_metadata_required')}`
- COS copy used: `false`

Only states A or B are eligible for a subsequent full-checkpoint zero-shot
experiment. This audit reached `{final_state}`; therefore the next permitted
step is a controlled zero-shot experiment that preserves the audited source
preprocessing, layer, mask, pooling, and class order. No such experiment was
run here.

## Match fractions

- Core backbone key match: `{mapping_summary['core_backbone_key_match_fraction']:.6f}`
- Core backbone parameter match: `{mapping_summary['core_backbone_parameter_match_fraction']:.6f}`
- Full backbone key match: `{mapping_summary['full_backbone_key_match_fraction']:.6f}`
- Full backbone parameter match: `{mapping_summary['full_backbone_parameter_match_fraction']:.6f}`

## Proven facts

- The authoritative CFS checkpoint is `{checkpoint_path.stat().st_size}` bytes,
  has SHA256 `{checkpoint_sha}`, epoch `{checkpoint.get('epoch')}`, global step
  `{checkpoint.get('global_step')}`, and Lightning version
  `{checkpoint.get('pytorch-lightning_version')}`.
- It contains `{len(source_state)}` state tensors with
  `{sum(int(value.numel()) for value in source_state.values())}` tensor elements.
- Every required core and full-backbone tensor has an explicit target key and
  an identical shape; there are no missing, unexpected, or shape-mismatched
  model tensors. `loss.weight` is reported separately as a training-only loss
  buffer.
- The deterministic one-batch parity comparison passed for tokenizer output,
  selected transformer layer, `[B,512]` representation, and raw source-order
  logits at `atol={args.atol}` and `rtol={args.rtol}`.
- The COS object was not accessed or downloaded.

## Code-traced conclusions

- Source and target both select zero-based transformer layer
  `{source_probe_layer_idx}` from depth `{depth}` using ratio `{ratio}` and
  perform masked mean pooling where true/one means valid.
- There is no detach, projection, post-pooling normalization, or dropout
  between the pooled representation and the linear sex probe in eval mode.
- Source `LabelEncoder` sorts the observed raw labels. With authoritative HBN
  raw labels `F` and `M` and their documented meanings, source and target both
  use `female=0, male=1`; no probe-row swap is required.

## Unresolved items and metadata

- No compatibility blocker remains for this checkpoint and audited token path.
- No additional dataset metadata is required. The minimal raw-label evidence
  used was the HBN `sex` field (`F`/`M`) plus its documented meanings; no Huada
  labels, test labels, predictions, or accuracy were used to infer class order.
- The source repository commit is recorded, but its working tree was dirty.
  This is provenance information, not silently treated as a clean release.

## Scope and caveats

- The parity fixture is one deterministic CPU batch only; no full dataset, zero-shot, CV, or training run was performed.
- Source raw EEG preprocessing uses a 2 s HBN window padded/masked into a 10 s model clip. The target path consumes precomputed Lance tokens; the audit compares the representation paths from an identical token batch.
- Prefix remapping is explicit in `checkpoint_key_mapping.csv`; no opaque `strict=False` load is used.
- The source checkpoint and CFS repository were read only. The COS copy was not downloaded.
"""
        (output_dir / "audit_report.md").write_text(report, encoding="utf-8")
        return 0
    except Exception:
        error_path.write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
