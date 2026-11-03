from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parents[1]
MODULE_ROOT = ROOT / "scripts" / "downstream_50m"
sys.path.insert(0, str(MODULE_ROOT))

import age_cv7
import run_huada_dortmund_age_probe_transfer_cv7 as transfer


AGES = {"SUB1": 22.0, "SUB2": 24.0, "SUB3": 25.0, "SUB4": 27.0, "SUB6": 28.0, "SUB7": 30.0, "SUB8": 31.0}


class FakeFeatureDataset(Dataset):
    def __init__(self, windows_per_subject=2):
        self.rows = []
        for subject_index, (subject, age) in enumerate(AGES.items()):
            for window in range(windows_per_subject):
                feature = torch.zeros(512, dtype=torch.float32)
                feature[subject_index] = 1.0 + window * 0.01
                self.rows.append({
                    "feature": feature, "num_valid_tokens": torch.tensor(640),
                    "label": torch.tensor(age, dtype=torch.float32), "sample_id": f"{subject}_{window}",
                    "subject_id": subject, "recording_id": f"{subject}_EO",
                    "window_index": torch.tensor(window), "label_name": "",
                })

    def __len__(self): return len(self.rows)
    def __getitem__(self, index): return self.rows[index]
    def get_subject_label_rows(self):
        return [{"sample_id": row["sample_id"], "subject_id": row["subject_id"], "label": float(row["label"])} for row in self.rows]


def source_state():
    return {"weight": torch.zeros(1, 512), "bias": torch.zeros(1)}


def fake_source_loader(_path, _sha):
    state = source_state()
    scaler = transfer.FixedSourceScaler(27.5, 4.218591123456, 0)
    report = {"sha256": "a" * 64, "path": str(Path("fake_best_full_checkpoint.pt").resolve())}
    return state, transfer.tensor_state_fingerprint(state), report, scaler


def test_source_checkpoint_sha_mismatch_fails_before_torch_load(tmp_path, monkeypatch):
    path = tmp_path / "best_full_checkpoint.pt"
    path.write_bytes(b"small-fixture-not-a-real-checkpoint")
    monkeypatch.setattr(transfer.torch, "load", lambda *a, **k: (_ for _ in ()).throw(AssertionError("torch.load called")))
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        transfer.load_source_probe_state(path, "a" * 64)


def test_source_probe_strict_extraction_uses_full_precision_scaler(tmp_path, monkeypatch):
    path = tmp_path / "best_full_checkpoint.pt"
    payload = {"age_probe_state_dict": source_state(), "target_transform": {"ddof": 0}}
    torch.save(payload, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(transfer, "build_formal_model", lambda: object())
    monkeypatch.setattr(transfer, "validate_checkpoint_payload", lambda value, model: {
        "source_age_mean": 27.5, "source_age_std": 4.218591123456,
    })
    state, fingerprint, report, scaler = transfer.load_source_probe_state(path, digest)
    assert set(state) == {"weight", "bias"} and len(fingerprint) == 64
    assert report["sha256"] == digest and scaler.std == 4.218591123456


def test_fixed_source_scaler_round_trip_and_no_huada_fit():
    scaler = transfer.FixedSourceScaler(27.5, 4.218591123456)
    ages = torch.tensor([22.0, 27.5, 31.0])
    assert torch.allclose(scaler.inverse(scaler.transform(ages)), ages)
    assert scaler.metadata()["target_scaler_fitted_on_huada"] is False
    assert scaler.metadata()["target_transform"] == "fixed_source_zscore"


def test_cv7_excludes_sub5_and_assigns_each_subject_once(tmp_path):
    payload, digest, ages, samples, counts = transfer.build_split_artifacts(FakeFeatureDataset(), 42, tmp_path)
    assert len(digest) == 64 and set(ages) == set(AGES) and "SUB5" not in ages
    assert set(samples) == {f"{s}_{w}" for s in AGES for w in range(2)}
    assert set(counts.values()) == {2}
    assert sorted(f["test_subject"] for f in payload["folds"]) == sorted(AGES)
    assert sorted(f["validation_subject"] for f in payload["folds"]) == sorted(AGES)
    assert all(len(f["train_subjects"]) == 5 for f in payload["folds"])


def test_sub5_is_rejected_from_age_cv7(tmp_path):
    dataset = FakeFeatureDataset()
    dataset.rows.append({**dataset.rows[0], "sample_id": "SUB5_0", "subject_id": "SUB5"})
    with pytest.raises(RuntimeError, match="SUB5"):
        transfer.build_split_artifacts(dataset, 42, tmp_path)


def test_cache_key_binds_source_checkpoint_vocab_mask_and_input():
    info = {"row_count": 14, "sample_id_fingerprint": "samples"}
    base = {
        "sha256": "a" * 64, "electrode_vocabulary_sha256": "vocab-a",
        "token_mask_semantics_version": transfer.MASK_SEMANTICS_VERSION, "best_epoch": 20,
    }
    identity_a, key_a = transfer.build_dortmund_cache_identity(
        input_fingerprint="input-a", input_info=info, checkpoint_report=base,
        code_sha256="code", git_commit="commit",
    )
    identity_b, key_b = transfer.build_dortmund_cache_identity(
        input_fingerprint="input-a", input_info=info, checkpoint_report={**base, "sha256": "b" * 64},
        code_sha256="code", git_commit="commit",
    )
    assert key_a != key_b
    assert identity_a["requested_poolings"] == ["mean"]
    assert identity_a["pooling"] == "mean"
    assert identity_a["electrode_vocabulary_sha256"] == "vocab-a"


def test_source_probe_model_has_exactly_513_trainable_parameters():
    model = transfer.build_source_initialized_probe(source_state(), torch.device("cpu"))
    assert sum(p.numel() for p in model.parameters()) == 513
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 513
    assert not hasattr(model, "backbone")
    optimizer = transfer.make_probe_optimizer(model, 1e-5, 0.03)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.03)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-5)


def test_selection_tolerance_and_epoch_zero():
    epoch0 = {"subject_mae": 1.0, "window_mae": 1.0, "loss": 1.0, "epoch": 0}
    tiny_noise = {"subject_mae": 1.0 - 1e-12, "window_mae": 1.0 - 1e-12, "loss": 1.0 - 1e-12, "epoch": 1}
    assert transfer.candidate_is_better(epoch0, None)
    assert not transfer.candidate_is_better(tiny_noise, epoch0)


def test_validation_single_subject_metrics_are_undefined():
    dataset = FakeFeatureDataset()
    model = transfer.build_source_initialized_probe(source_state(), torch.device("cpu"))
    indices = transfer._indices_for_subjects(dataset.get_subject_label_rows(), ["SUB1"])
    loader = transfer._loader(dataset, indices, 8, False, 42, 0)
    metrics, rows = transfer.evaluate_probe(
        model, loader, torch.device("cpu"), transfer.FixedSourceScaler(27.5, 4.218591),
        fold=0, collect_rows=True, single_subject_validation=True,
    )
    assert rows and metrics["r2"] is None and metrics["r2_defined"] is False
    assert metrics["pearson_r"]["reason"] == "single validation subject"


def test_test_guard_fails_closed_and_allows_one_metric_computation():
    guard = transfer.FoldTestAccessGuard()
    with pytest.raises(RuntimeError, match="before selection"):
        guard.create_dataset()
    guard.lock(); guard.create_dataset(); guard.create_loader(); guard.forward(); guard.materialize(2); guard.metrics()
    with pytest.raises(RuntimeError, match="twice"):
        guard.metrics()
    assert guard.state()["test_forward_count"] == 1


def test_head_only_checkpoint_validation_rejects_backbone():
    scaler = transfer.FixedSourceScaler(27.5, 4.218591123456)
    fold = {"fold": 0, "train_subjects": ["SUB3", "SUB4", "SUB6", "SUB7", "SUB8"], "validation_subject": "SUB1", "test_subject": "SUB2"}
    payload = {
        "checkpoint_kind": transfer.HEAD_CHECKPOINT_KIND, "head_state_dict": source_state(),
        "best_head_fingerprint": transfer.tensor_state_fingerprint(source_state()),
        "source_full_checkpoint_sha256": "a" * 64, "feature_cache_key": "cache",
        "source_checkpoint_kind": transfer.SOURCE_CHECKPOINT_KIND,
        "pooling": "masked_mean", "feature_cache_pooling": "mean",
        "feature_dim": 512, "probe_layer_idx": 8, "fold": 0,
        "train_subjects": fold["train_subjects"],
        "validation_subject": "SUB1", "test_subject": "SUB2",
        "source_age_mean": scaler.mean, "source_age_std": scaler.std,
        "target_transform": "fixed_source_zscore", "target_unit": "year",
        "target_scaler_fitted_on_huada": False, "source_scaler_reused": True,
    }
    assert transfer.validate_head_checkpoint(payload, source_sha="a" * 64, cache_key="cache", fold_info=fold, scaler=scaler)
    with pytest.raises(RuntimeError, match="Invalid head-only"):
        transfer.validate_head_checkpoint({**payload, "backbone_state_dict": {}}, source_sha="a" * 64, cache_key="cache", fold_info=fold, scaler=scaler)


def test_fold_reloads_source_epoch0_can_win_and_test_runs_once(tmp_path, monkeypatch):
    calls = []
    def loader(path, sha):
        calls.append((path, sha)); return fake_source_loader(path, sha)
    monkeypatch.setattr(transfer, "load_source_probe_state", loader)
    def deliberately_worsen(model, loader, optimizer, device, scaler):
        with torch.no_grad(): model.head.bias.fill_(100.0)
        return {"train_z_mse": 1.0, "train_mae_years": 1.0, "gradient_norm": 1.0}
    monkeypatch.setattr(transfer, "train_epoch", deliberately_worsen)
    dataset = FakeFeatureDataset()
    payload = age_cv7.build_age_subject_cv7(AGES, 42)
    cache_info = {"feature_cache_key": "fixture-cache", "feature_cache_path": str(tmp_path / "cache")}
    summary, rows, history = transfer.run_fold(
        fold_info=payload["folds"][0], dataset=dataset, checkpoint_path="fixture.pt",
        expected_sha256="a" * 64, cache_info=cache_info, split_manifest_sha256="split",
        output_dir=tmp_path, device=torch.device("cpu"), batch_size=8, num_workers=0,
        seed=42, head_lr=1e-5, weight_decay=0.0, epochs=2, patience=1, logger=None,
    )
    assert len(calls) == 1 and summary["best_epoch"] == 0
    assert summary["test_evaluation_count"] == 1 and summary["test_access"]["test_metrics_computed"] is True
    assert summary["backbone_forward_count_during_fold_training"] == 0
    assert summary["best_probe_fingerprint"] == summary["initial_source_probe_fingerprint"]
    assert rows and history[0]["epoch"] == 0
    saved = torch.load(summary["best_checkpoint"], map_location="cpu", weights_only=True)
    assert "backbone_state_dict" not in saved and set(saved["head_state_dict"]) == {"weight", "bias"}


def test_oof_coverage_metrics_and_baselines_are_recomputed(tmp_path):
    dataset = FakeFeatureDataset()
    split, _, ages, samples, _ = transfer.build_split_artifacts(dataset, 42, tmp_path)
    fold_by_subject = {f["test_subject"]: f for f in split["folds"]}
    oof = []
    summaries = []
    for subject, age in ages.items():
        fold = fold_by_subject[subject]
        checkpoint = tmp_path / f"fold_{fold['fold']}" / "checkpoints" / "best_head.pt"
        summaries.append({
            "fold": fold["fold"], "train_subjects": fold["train_subjects"],
            "validation_subject": fold["validation_subject"], "test_subject": subject,
            "train_rows": 10, "validation_rows": 2, "test_rows": 2,
            "initial_source_probe_fingerprint": "same", "best_probe_fingerprint": "best",
            "final_probe_fingerprint": "final", "probe_parameter_delta_norm": 0.1,
            "best_epoch": 0, "epochs_executed": 0, "stopped_early": False,
            "best_checkpoint": str(checkpoint), "best_checkpoint_sha256": "a" * 64,
            "test_evaluation_count": 1, "backbone_forward_count_during_fold_training": 0,
        })
        for window in range(2):
            prediction = age + 0.25
            oof.append({
                "fold": fold["fold"], "sample_id": f"{subject}_{window}", "subject_id": subject,
                "recording_id": f"{subject}_EO", "window_index": window, "true_age": age,
                "true_z_source": (age - 27.5) / 4.218591, "predicted_z": (prediction - 27.5) / 4.218591,
                "predicted_age": prediction, "signed_error": 0.25, "absolute_error": 0.25,
                "squared_error": 0.0625, "num_valid_tokens": 640,
                "test_is_outside_train_age_range": fold["test_is_outside_train_age_range"],
                "checkpoint_path": str(checkpoint),
            })
    result = transfer.aggregate_oof_artifacts(
        output_dir=tmp_path, split_payload=split, subject_ages=ages, sample_subject=samples,
        fold_summaries=summaries, oof_rows=oof, epoch_rows=[{"fold": 0, "epoch": 0}],
    )
    assert result["oof_window_count"] == 14 and result["oof_subject_count"] == 7
    assert result["oof_subject_mean_metrics"]["mae"] == pytest.approx(0.25)
    assert result["source_mean_baseline"]["prediction_years"] == 27.5
    assert result["fold_train_mean_baseline"]["uses_validation_or_test_age"] is False
    assert result["metric_recomputation"]["passed"] is True


def test_nonfinite_fails_closed():
    dataset = FakeFeatureDataset()
    dataset.rows[0]["feature"][0] = float("nan")
    model = transfer.build_source_initialized_probe(source_state(), torch.device("cpu"))
    loader = transfer._loader(dataset, [0], 1, False, 42, 0)
    with pytest.raises(FloatingPointError, match="Non-finite"):
        transfer.evaluate_probe(model, loader, torch.device("cpu"), transfer.FixedSourceScaler(27.5, 4.218591), fold=0)


def test_dry_run_never_touches_paths_or_creates_output(tmp_path, monkeypatch):
    output = tmp_path / "must-not-exist"
    args = transfer.parse_args([
        "--input-lance", str(tmp_path / "absent.lance"), "--checkpoint", str(tmp_path / "absent.pt"),
        "--expected-checkpoint-sha256", "a" * 64, "--feature-cache-root", str(tmp_path / "cache"),
        "--output-dir", str(output), "--dry-run",
    ])
    monkeypatch.setattr(transfer, "ensure_dortmund_mean_cache", lambda *_: (_ for _ in ()).throw(AssertionError("I/O called")))
    result = transfer.run(args)
    assert result["checkpoint_read"] is False and result["input_lance_opened"] is False
    assert not output.exists() and not (tmp_path / "cache").exists()


def test_cloud_entry_is_formal_probe_transfer_only():
    text = (ROOT / "commands" / "run_huada_dortmund_age_probe_transfer_cv7.sh").read_text(encoding="utf-8")
    assert "set -euo pipefail" in text
    assert "run_huada_dortmund_age_probe_transfer_cv7.py" in text
    assert "sha256sum -c" in text and "best_full_checkpoint.pt" in text
    for forbidden in ("--overwrite", "last_training_checkpoint.pt", "full_finetune", "calibration", "clip"):
        assert forbidden not in text
