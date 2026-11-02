"""Small task-neutral utilities for NeuralBench full-checkpoint zero-shot runs."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any, Sequence

import torch

try:  # package import
    from .largeScale_fineTune import EEGDownstreamClassifier
    from .model import EEGPretrainModel
except ImportError:  # direct script execution
    from largeScale_fineTune import EEGDownstreamClassifier
    from model import EEGPretrainModel


POOLING = "mean"
PROBE_LAYER_IDX = 8
PROBE_LAYER_RATIO = 0.75
FEATURE_DIM = 512


def build_meta_zero_shot_model(output_dim: int) -> EEGDownstreamClassifier:
    """Construct the audited architecture without materializing random values."""

    if int(output_dim) <= 0:
        raise ValueError("output_dim must be positive")
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
        return EEGDownstreamClassifier(
            backbone=backbone,
            d_model=FEATURE_DIM,
            num_classes=int(output_dim),
            probe_layer_idx=PROBE_LAYER_IDX,
            freeze_backbone=False,
            use_mlp_head=False,
            head_dropout=0.0,
            pooling=POOLING,
            num_tokens=640,
        )


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
    """Hash a Lance file/directory without following external paths."""

    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"Input Lance does not exist: {root}")
    files = [root] if root.is_file() else sorted(
        item for item in root.rglob("*") if item.is_file()
    )
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


def git_info(root: Path) -> dict[str, Any]:
    def command(*parts: str) -> str:
        result = subprocess.run(
            ["git", *parts], cwd=root, check=True, capture_output=True, text=True
        )
        return result.stdout.strip()

    try:
        return {
            "commit": command("rev-parse", "HEAD"),
            "dirty": bool(command("status", "--porcelain")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def cuda_allocated_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.memory_allocated(device) / (1024 * 1024))
