"""Create and verify a tiny FBMAE-shaped portable checkpoint fixture only."""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import torch

from .portable_age_checkpoint import (
    build_portable_age_checkpoint,
    load_portable_age_checkpoint,
    metadata_sha256,
    save_portable_age_checkpoint,
)


def build_demo_fixture() -> dict:
    """Return a deterministic, explicitly non-formal two-tensor FBMAE fixture."""

    zeros = "0" * 64
    mask_version = "token_valid_mask_true_is_valid_v1"
    return build_portable_age_checkpoint(
        model_name="fbmae_50m",
        model_family="fbmae",
        model_config={
            "class": "EEGPretrainModel", "input_dim": 100, "d_model": 512,
            "n_heads": 8, "depth": 12, "mlp_ratio": 4.0, "dropout": 0.1,
            "n_channels": 64, "n_time_patches": 10, "n_bands": 5,
            "frames_per_patch": 5, "objective": "timefreq",
        },
        backbone_state_dict={
            "tokenizer.proj.0.weight": torch.arange(512, dtype=torch.float32).reshape(512, 1),
            "encoder.encoder.layers.0.norm1.weight": torch.ones(512, dtype=torch.float32),
        },
        age_probe_state_dict={
            "weight": torch.zeros(1, 512, dtype=torch.float32),
            "bias": torch.zeros(1, dtype=torch.float32),
        },
        input_contract={
            "contract_version": 1,
            "input_kind": "fbmae_timefreq_tokens",
            "sampling_rate_hz": 200.0,
            "window_seconds": 10.0,
            "input_dtype": "float32",
            "shape_semantics": "batch,token,patch_feature",
            "channel_semantics": "canonical electrode id per token; invalid tokens masked",
            "electrode_vocabulary_name": "fixture_fbmae_64",
            "electrode_vocabulary_sha256": "1" * 64,
            "channel_canonicalization_version": "fixture-v1",
            "channel_canonicalization_sha256": "2" * 64,
            "mask_semantics_version": mask_version,
            "mask_semantics_sha256": metadata_sha256(mask_version),
            "preprocessing_profile": "fixture-only-no-data",
            "preprocessing_config_sha256": "3" * 64,
        },
        representation_contract={
            "contract_version": 1,
            "representation_name": "fbmae_encoder_token_embedding",
            "representation_layer": 8,
            "pooling": "masked_mean",
            "feature_dim": 512,
            "representation_dtype": "float32",
            "uses_token_valid_mask": True,
        },
        target_transform={
            "type": "zscore", "mean": 27.5, "std": 4.218591189948011,
            "ddof": 0, "equal_subject_weight": True, "target_unit": "year",
        },
        source_training={
            "dataset_name": "DortmundAge-fixture",
            "dataset_fingerprint": "fixture-no-data-access",
            "split_manifest_sha256": "4" * 64,
            "train_subject_count": 1,
            "validation_subject_count": 1,
            "test_subject_count": 1,
            "best_epoch": 0,
            "selection_metric": "validation_subject_mean_mae_years",
            "validation_metrics": {"mae_years": 0.0},
            "test_metrics": {"value": None, "defined": False, "reason": "fixture"},
        },
        source_checkpoint={
            "checkpoint_format": "fixture_synthetic",
            "source_checkpoint_sha256": "5" * 64,
            "source_checkpoint_size_bytes": 1,
        },
        provenance={
            "exporter_name": "huada-portable-fixture-demo",
            "exporter_version": "1",
            "source_repository": "huada-fixture",
            "source_git_commit": "fixture",
            "source_git_dirty": False,
            "critical_code_sha256": {"fixture": zeros},
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "python_version": platform.python_version(),
            "pytorch_version": str(torch.__version__),
        },
        validation_mode="fixture",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--filename", default="portable_age_fixture_v1.pt")
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / args.filename
    saved = save_portable_age_checkpoint(
        build_demo_fixture(), destination, allow_fixture=True
    )
    loaded = load_portable_age_checkpoint(
        destination,
        expected_file_sha256=saved.file_sha256,
        expected_model_name="fbmae_50m",
        expected_model_family="fbmae",
        allow_fixture=True,
    )
    print(json.dumps({
        "path": str(loaded.path),
        "manifest": str(destination) + ".manifest.json",
        "schema_name": loaded.payload["schema_name"],
        "schema_version": loaded.payload["schema_version"],
        "checkpoint_kind": loaded.payload["checkpoint_kind"],
        "validation_mode": loaded.payload["validation_mode"],
        "model_name": loaded.model_name,
        "model_family": loaded.model_family,
        "feature_dim": loaded.payload["representation_contract"]["feature_dim"],
        "file_sha256": loaded.file_sha256,
        "safe_load_verified": True,
        "formal_checkpoint_read": False,
        "formal_model_constructed": False,
        "formal_forward_count": 0,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
