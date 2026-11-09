from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.native_backends import run_huada_portable_age_zero_shot as zero


class _FakeNativeBackend(torch.nn.Module):
    def __init__(self, mean: float = 27.5, std: float = 4.0) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))
        self.mean = mean
        self.std = std

    def forward(self, value: torch.Tensor) -> dict[str, torch.Tensor]:
        z = value.mean(dim=(1, 2), keepdim=False).reshape(-1, 1) * self.weight
        tokens = z.reshape(-1, 1, 1).expand(-1, 2, 3)
        return {
            "backbone_tokens": tokens,
            "representation": tokens.mean(dim=1),
            "predicted_z": z,
            "predicted_age_years": z * self.std + self.mean,
        }


def _recordings() -> dict[str, np.ndarray]:
    return {
        subject: np.full((64, 800), index - 4, dtype=np.float32)
        for index, subject in enumerate(zero.EXPECTED_SUBJECT_AGES, start=1)
    }


def test_native_zero_shot_is_label_blind_retains_sub5_and_does_not_clip():
    model = _FakeNativeBackend()
    rows, forward_count = zero.infer_native_zero_shot(
        model,
        _recordings(),
        device=torch.device("cpu"),
        batch_size=3,
        source_mean=27.5,
        source_std=4.0,
        checkpoint_sha256="a" * 64,
    )
    assert forward_count == 6
    assert len(rows) == 16
    assert {row["subject_id"] for row in rows} == set(zero.EXPECTED_SUBJECT_AGES)
    sub5 = [row for row in rows if row["subject_id"] == "SUB5"]
    assert sub5 and all(row["true_age_years"] is None for row in sub5)
    assert all(row["metrics_eligible"] is False for row in sub5)
    # SUB1 z=-3 -> 15.5 years.  It is deliberately not clipped to source/Huada ranges.
    assert next(row for row in rows if row["subject_id"] == "SUB1")["predicted_age_years"] == pytest.approx(15.5)

    bundle = zero.compute_metric_bundle(rows)
    assert bundle["subject_mean_metrics"]["sample_count"] == 7
    assert bundle["source_mean_baseline_metrics"]["baseline_prediction"] == 27.5
    assert [row["subject_id"] for row in bundle["subject_rows_labeled"]] == [
        "SUB1", "SUB2", "SUB3", "SUB4", "SUB6", "SUB7", "SUB8"
    ]


def test_backend_age_must_equal_portable_scaler_reconstruction():
    model = _FakeNativeBackend(mean=0.0, std=1.0)
    with pytest.raises(zero.NativeZeroShotError, match="scaler reconstruction"):
        zero.infer_native_zero_shot(
            model,
            _recordings(),
            device=torch.device("cpu"),
            batch_size=8,
            source_mean=27.5,
            source_std=4.0,
            checkpoint_sha256="b" * 64,
        )


def _write_reference(root: Path, bundle: dict) -> None:
    root.mkdir()
    (root / "_SUCCESS").write_text("passed\n", encoding="utf-8")
    (root / "subject_mean_metrics.json").write_text(
        json.dumps(bundle["subject_mean_metrics"]), encoding="utf-8"
    )
    fields = [
        "subject_id", "true_age", "window_count", "mean_predicted_age",
        "median_predicted_age", "mean_predicted_z", "prediction_std",
        "prediction_min", "prediction_max", "signed_error", "absolute_error",
        "included_in_metrics",
    ]
    with (root / "subject_predictions_all.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(bundle["subject_rows_all"])


def test_three_model_comparison_has_same_seven_subject_metric_scope(tmp_path):
    rows, _ = zero.infer_native_zero_shot(
        _FakeNativeBackend(),
        _recordings(),
        device=torch.device("cpu"),
        batch_size=8,
        source_mean=27.5,
        source_std=4.0,
        checkpoint_sha256="c" * 64,
    )
    bundle = zero.compute_metric_bundle(rows)
    fbmae = tmp_path / "fbmae"
    reve = tmp_path / "reve"
    _write_reference(fbmae, bundle)
    _write_reference(reve, bundle)

    report, metric_rows, subject_rows = zero.build_model_comparison(
        "labram", bundle, [f"fbmae={fbmae}", f"reve={reve}"]
    )
    assert {row["model_family"] for row in metric_rows} == {"fbmae", "labram", "reve"}
    assert all(row["subject_count"] == 7 for row in metric_rows)
    assert len(subject_rows) == 8
    assert report["fixed_baseline"]["baseline_prediction"] == 27.5
    sub5 = next(row for row in subject_rows if row["subject_id"] == "SUB5")
    assert sub5["true_age"] is None
    assert set(sub5) == {
        "subject_id", "true_age", "fbmae_mean_predicted_age",
        "labram_mean_predicted_age", "reve_mean_predicted_age",
    }


def test_parse_args_requires_full_hashes_and_exact_positive_scaler():
    base = [
        "--model-family", "labram",
        "--native-cache-run-root", "cache",
        "--subject-manifest", "subjects.json",
        "--expected-subject-manifest-sha256", "0" * 64,
        "--target-input-contract", "contract.json",
        "--expected-target-contract-file-sha256", "1" * 64,
        "--portable-checkpoint", "portable.pt",
        "--portable-sidecar", "portable.sidecar.json",
        "--expected-portable-checkpoint-sha256", "2" * 64,
        "--expected-source-checkpoint-sha256", "3" * 64,
        "--dynamic-parity-dir", "parity",
        "--output-dir", "output",
        "--expected-source-age-std", "4.218591213226318",
    ]
    assert zero.parse_args(base).expected_source_age_std == 4.218591213226318
    with pytest.raises(SystemExit):
        zero.parse_args(base[:-1] + ["short"])


def test_dry_run_does_not_open_inputs_or_construct_model(capsys):
    result = zero.main([
        "--model-family", "reve",
        "--native-cache-run-root", "missing-cache",
        "--subject-manifest", "missing-subjects.json",
        "--expected-subject-manifest-sha256", "0" * 64,
        "--target-input-contract", "missing-contract",
        "--expected-target-contract-file-sha256", "1" * 64,
        "--portable-checkpoint", "missing.pt",
        "--portable-sidecar", "missing.sidecar.json",
        "--expected-portable-checkpoint-sha256", "2" * 64,
        "--expected-source-checkpoint-sha256", "3" * 64,
        "--dynamic-parity-dir", "missing-parity",
        "--output-dir", "missing-output",
        "--expected-source-age-std", "4.218591213226318",
        "--dry-run",
    ])
    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["will_read_cache"] is False
    assert payload["will_read_checkpoint"] is False
    assert payload["will_execute_forward"] is False
    assert payload["training_enabled"] is False
    assert payload["calibration_enabled"] is False
    assert payload["prediction_clipping_enabled"] is False


def test_subject_manifest_is_sha_bound_and_sub5_stays_null(tmp_path):
    source = Path("preprocessing_audit/20261030_105520/subject_manifest.json")
    copied = tmp_path / "subject_manifest.json"
    copied.write_bytes(source.read_bytes())
    sha = zero.sha256_file(copied)
    assert zero.load_subject_age_manifest(
        copied, expected_file_sha256=sha
    ) == zero.EXPECTED_SUBJECT_AGES
    with pytest.raises(zero.NativeZeroShotError, match="SHA mismatch"):
        zero.load_subject_age_manifest(copied, expected_file_sha256="f" * 64)


def test_entrypoint_has_no_training_or_prediction_clip_path():
    source = Path(zero.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "torch.optim",
        ".backward(",
        "optimizer.step(",
        "scheduler.step(",
        "np.clip(pred",
        "torch.clamp(pred",
    ):
        assert forbidden not in source
