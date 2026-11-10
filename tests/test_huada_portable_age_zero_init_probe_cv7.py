from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.downstream_50m import age_cv7
from scripts.native_backends import run_huada_portable_age_zero_init_probe_cv7 as runner


AGES = {
    "SUB1": 31.0,
    "SUB2": 29.0,
    "SUB3": 26.0,
    "SUB4": 26.0,
    "SUB6": 28.0,
    "SUB7": 28.0,
    "SUB8": 22.0,
}


def _cache_fixture(tmp_path: Path, *, feature_dim: int = 3):
    identity = runner._feature_cache_identity(
        family="labram",
        portable_sha256="a" * 64,
        backbone_fingerprint="b" * 64,
        contract={"contract_sha256": "c" * 64},
        contract_file_sha256="d" * 64,
        cache_snapshot={"files": []},
        feature_dim=feature_dim,
        token_count=129,
    )
    key = runner._json_sha256(identity)
    root = tmp_path / key
    root.mkdir()
    subjects = list(runner.EXPECTED_SUBJECT_AGES)
    features = np.arange(len(subjects) * feature_dim, dtype=np.float32).reshape(
        len(subjects), feature_dim
    )
    samples = [
        {
            "sample_id": f"{subject}_W000000",
            "subject_id": subject,
            "recording_id": subject,
            "window_index": 0,
            "num_valid_tokens": 129,
        }
        for subject in subjects
    ]
    np.save(root / "features.float32.npy", features, allow_pickle=False)
    runner._json_write(root / "samples.json", samples)
    manifest = {
        "identity": identity,
        "feature_cache_key": key,
        "row_count": len(samples),
        "feature_dim": feature_dim,
        "dtype": "float32",
        "feature_fingerprint": runner._features_fingerprint(features),
        "features_file_sha256": runner.sha256_file(root / "features.float32.npy"),
        "samples_file_sha256": runner.sha256_file(root / "samples.json"),
        "labels_in_cache": False,
    }
    runner._json_write(root / "manifest.json", manifest)
    (root / "_SUCCESS").write_text(key + "\n", encoding="utf-8")
    return root, identity, features, samples


def test_fixed_probe_config_matches_fbmae_transfer_defaults():
    candidate = runner.fixed_probe_candidate(1e-5, 0.0)
    assert candidate["head_lr"] == 1e-5
    assert candidate["weight_decay"] == 0.0
    assert candidate["pooling"] == "mean"
    assert len(candidate["candidate_sha256"]) == 64
    with pytest.raises(runner.ZeroInitCV7Error):
        runner.fixed_probe_candidate(0.0, 0.0)


@pytest.mark.parametrize(("feature_dim", "parameter_count"), [(200, 201), (512, 513)])
def test_probe_is_exactly_zero_for_both_families(feature_dim, parameter_count):
    first = runner.build_zero_probe(feature_dim, torch.device("cpu"))
    second = runner.build_zero_probe(feature_dim, torch.device("cpu"))
    assert sum(value.numel() for value in first.parameters()) == parameter_count
    assert all(torch.count_nonzero(value).item() == 0 for value in first.parameters())
    assert runner.probe_fingerprint(first) == runner.probe_fingerprint(second)


def test_zero_probe_uses_full_source_zscore_and_starts_at_27_5_years():
    probe = runner.build_zero_probe(2, torch.device("cpu"))
    features = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    samples = [{
        "sample_id": "SUB1_W000000",
        "subject_id": "SUB1",
        "recording_id": "SUB1",
        "window_index": 0,
        "num_valid_tokens": 129,
    }]
    metrics, rows = runner._evaluate(
        probe,
        features,
        samples,
        [0],
        {"SUB1": 31.0},
        device=torch.device("cpu"),
        mean=27.5,
        std=4.218591213226318,
        batch_size=1,
        fold=0,
    )
    assert rows[0]["predicted_z"] == 0.0
    assert rows[0]["predicted_age"] == 27.5
    assert rows[0]["true_z_source"] == pytest.approx((31.0 - 27.5) / 4.218591213226318)
    assert metrics["loss"] == pytest.approx(rows[0]["true_z_source"] ** 2)


def test_cv7_is_exact_existing_protocol_and_excludes_sub5():
    payload = age_cv7.build_age_subject_cv7(AGES, 42)
    assert payload["protocol"] == age_cv7.PROTOCOL
    assert payload["algorithm_version"] == age_cv7.ALGORITHM_VERSION
    assert len(payload["folds"]) == 7
    assert {fold["test_subject"] for fold in payload["folds"]} == set(AGES)
    assert all(len(fold["train_subjects"]) == 5 for fold in payload["folds"])
    assert all("SUB5" not in json.dumps(fold) for fold in payload["folds"])


def test_validation_selection_matches_fbmae_transfer_metric_order():
    assert runner.candidate_is_better(
        {"subject_mae": 1, "window_mae": 2, "loss": 3, "epoch": 1},
        {"subject_mae": 1, "window_mae": 2, "loss": 3, "epoch": 2},
    )
    assert runner.candidate_is_better(
        {"subject_mae": 0.9, "window_mae": 999, "loss": 999, "epoch": 9},
        {"subject_mae": 1.0, "window_mae": 1, "loss": 1, "epoch": 1},
    )


def test_label_free_feature_cache_round_trip_and_tamper_rejection(tmp_path):
    root, identity, features, samples = _cache_fixture(tmp_path)
    result = runner._validate_feature_cache(root, identity)
    assert result["manifest"]["labels_in_cache"] is False
    assert np.array_equal(np.asarray(result["features"]), features)
    samples[0]["true_age"] = 31.0
    runner._json_write(root / "samples.json", samples)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["samples_file_sha256"] = runner.sha256_file(root / "samples.json")
    runner._json_write(root / "manifest.json", manifest)
    with pytest.raises(runner.ZeroInitCV7Error, match="contains labels"):
        runner._validate_feature_cache(root, identity)


def test_feature_cache_identity_binds_family_backbone_contract_and_native_cache():
    base = dict(
        family="labram",
        portable_sha256="a" * 64,
        backbone_fingerprint="b" * 64,
        contract={"contract_sha256": "c" * 64},
        contract_file_sha256="d" * 64,
        cache_snapshot={"files": []},
        feature_dim=200,
        token_count=129,
    )
    first = runner._feature_cache_identity(**base)
    for field, replacement in (
        ("family", "reve"),
        ("portable_sha256", "e" * 64),
        ("backbone_fingerprint", "f" * 64),
        ("cache_snapshot", {"files": ["changed"]}),
    ):
        changed = dict(base)
        changed[field] = replacement
        assert runner._json_sha256(runner._feature_cache_identity(**changed)) != runner._json_sha256(first)
    assert first["labels_in_cache"] is False


def test_strict_zero_shot_reference_is_bound_to_same_portable_source(tmp_path):
    portable_sha = "a" * 64
    source_sha = "b" * 64
    metrics = {"mae": 1.0, "rmse": 2.0}
    runner._json_write(tmp_path / "subject_mean_metrics.json", metrics)
    runner._json_write(tmp_path / "zero_shot_summary.json", {
        "status": "passed",
        "model_family": "labram",
        "subject_mean_metrics": metrics,
        "training_performed": False,
        "calibration_performed": False,
        "prediction_clipping_performed": False,
    })
    runner._json_write(tmp_path / "zero_shot_provenance.json", {
        "evaluation_kind": "dortmund_portable_age_strict_zero_shot",
        "model_family": "labram",
        "portable_checkpoint_sha256": portable_sha,
        "source_checkpoint_sha256": source_sha,
    })
    (tmp_path / "_SUCCESS").write_text("passed\n", encoding="utf-8")
    result = runner._validate_zero_shot_reference(
        tmp_path,
        family="labram",
        portable_sha256=portable_sha,
        source_sha256=source_sha,
    )
    assert result["metrics"] == metrics
    with pytest.raises(runner.ZeroInitCV7Error, match="provenance mismatch"):
        runner._validate_zero_shot_reference(
            tmp_path,
            family="reve",
            portable_sha256=portable_sha,
            source_sha256=source_sha,
        )


def test_head_only_checkpoint_round_trip_and_forbidden_backbone(tmp_path):
    probe = runner.build_zero_probe(3, torch.device("cpu"))
    payload = {
        "checkpoint_kind": runner.HEAD_CHECKPOINT_KIND,
        "model_family": "labram",
        "feature_dim": 3,
        "feature_cache_key": "cache",
        "fold": 0,
        "candidate_id": "candidate",
        "initialization": "all_zero",
        "initial_head_fingerprint": runner.probe_fingerprint(probe),
        "head_state_dict": runner.probe_state(probe),
        "source_mean": 27.5,
        "source_std": 4.218591213226318,
        "age_target_normalization": "fixed_source_zscore",
        "target_scaler_fitted_on_huada": False,
        "prediction_clipping": "none",
    }
    payload["head_state_fingerprint"] = runner.tensor_state_fingerprint(payload["head_state_dict"])
    path = tmp_path / "head.pt"
    runner._save_head(path, payload)
    state = runner._load_head(
        path,
        family="labram",
        feature_dim=3,
        cache_key="cache",
        fold=0,
        candidate_id="candidate",
        mean=27.5,
        std=4.218591213226318,
    )
    assert set(state) == {"weight", "bias"}
    payload["backbone_state_dict"] = {"x": torch.zeros(1)}
    runner._save_head(path, payload)
    with pytest.raises(runner.ZeroInitCV7Error, match="forbidden"):
        runner._load_head(
            path,
            family="labram",
            feature_dim=3,
            cache_key="cache",
            fold=0,
            candidate_id="candidate",
            mean=27.5,
            std=4.218591213226318,
        )


def test_candidate_search_starts_at_zero_and_writes_head_only_checkpoint(tmp_path):
    samples = []
    values = []
    for index, (subject, age) in enumerate(AGES.items()):
        samples.append({
            "sample_id": f"{subject}_W000000",
            "subject_id": subject,
            "recording_id": subject,
            "window_index": 0,
            "num_valid_tokens": 129,
        })
        values.append([float(index + 1), 1.0])
    split = age_cv7.build_age_subject_cv7(AGES, 42)
    row = runner._run_candidate(
        family="labram",
        feature_dim=2,
        cache_key="cache-key",
        features=torch.tensor(values, dtype=torch.float32),
        samples=samples,
        subject_ages=AGES,
        fold_info=split["folds"][0],
        candidate={"candidate_id": "fixture", "head_lr": 1e-5, "weight_decay": 0.0},
        output_dir=tmp_path,
        device=torch.device("cpu"),
        mean=27.5,
        std=4.218591213226318,
        epochs=1,
        patience=1,
        batch_size=16,
        seed=42,
    )
    expected = runner.probe_fingerprint(runner.build_zero_probe(2, torch.device("cpu")))
    assert row["initial_head_fingerprint"] == expected
    assert row["test_accessed_during_selection"] is False
    payload = torch.load(row["checkpoint_path"], map_location="cpu", weights_only=True)
    assert payload["checkpoint_kind"] == runner.HEAD_CHECKPOINT_KIND
    assert payload["age_target_normalization"] == "fixed_source_zscore"
    assert payload["source_mean"] == 27.5
    assert payload["source_std"] == 4.218591213226318
    assert payload["prediction_clipping"] == "none"
    assert "backbone_state_dict" not in payload
    assert "optimizer_state_dict" not in payload


def test_dry_run_does_not_touch_formal_artifacts_or_create_output(tmp_path, capsys):
    output = tmp_path / "must_not_exist"
    args = Namespace(
        dry_run=True,
        model_family="reve",
        head_lr=1e-5,
        weight_decay=0.0,
        output_dir=str(output),
        native_cache_run_root=str(tmp_path / "missing-cache"),
        portable_checkpoint=str(tmp_path / "missing-portable.pt"),
    )
    assert runner.run(args) == {"status": "dry_run_passed"}
    assert not output.exists()
    report = json.loads(capsys.readouterr().out)
    assert report["candidate_search_count"] == 7
    assert report["test_evaluation_count"] == 7
    assert report["would_read_native_cache"] is False
    assert report["would_read_portable_checkpoint"] is False


def test_synthetic_end_to_end_locks_all_selections_before_seven_tests(tmp_path, monkeypatch):
    subject_manifest = tmp_path / "subjects.json"
    subject_manifest.write_text(
        json.dumps([
            {"subject_id": subject, "age": age}
            for subject, age in runner.EXPECTED_SUBJECT_AGES.items()
        ]),
        encoding="utf-8",
    )
    contract = tmp_path / "contract.json"
    contract.write_text("{}", encoding="utf-8")
    contract_sha = runner.sha256_file(contract)
    samples = []
    feature_rows = []
    for index, subject in enumerate(runner.EXPECTED_SUBJECT_AGES):
        samples.append({
            "sample_id": f"{subject}_W000000",
            "subject_id": subject,
            "recording_id": subject,
            "window_index": 0,
            "num_valid_tokens": 129,
        })
        feature_rows.append([float(index + 1), 1.0])
    checkpoint = SimpleNamespace(
        file_sha256="a" * 64,
        payload={
            "representation_contract": {"feature_dim": 2},
            "integrity": {"backbone_state_fingerprint": "c" * 64},
            "target_transform": {
                "type": "zscore", "mean": 27.5, "std": 4.218591213226318,
                "ddof": 0, "equal_subject_weight": True, "target_unit": "year",
            },
        },
    )
    monkeypatch.setattr(runner, "load_huada_target_contract", lambda *_args, **_kwargs: {
        "contract_sha256": "d" * 64
    })
    monkeypatch.setattr(runner, "_cache_snapshot", lambda _root: {"files": []})
    monkeypatch.setattr(runner, "load_native_cache_set", lambda *_args, **_kwargs: ({}, {}))
    monkeypatch.setattr(runner, "load_portable_age_checkpoint", lambda *_args, **_kwargs: checkpoint)
    monkeypatch.setattr(runner, "_validate_sidecar", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(runner, "_validate_dynamic_parity", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(runner, "ensure_feature_cache", lambda **_kwargs: {
        "manifest": {"feature_dim": 2, "feature_cache_key": "fixture-cache"},
        "features": np.asarray(feature_rows, dtype=np.float32),
        "samples": samples,
        "path": str(tmp_path / "feature-cache"),
        "reused": True,
    })
    output = tmp_path / "output"
    args = Namespace(
        dry_run=False,
        model_family="labram",
        head_lr=1e-5,
        weight_decay=0.0,
        expected_source_age_mean=27.5,
        expected_source_age_std=4.218591213226318,
        subject_manifest=str(subject_manifest),
        target_input_contract=str(contract),
        expected_target_contract_file_sha256=contract_sha,
        native_cache_run_root=str(tmp_path / "native-cache"),
        portable_checkpoint=str(tmp_path / "portable.pt"),
        portable_sidecar=str(tmp_path / "sidecar.json"),
        expected_portable_checkpoint_sha256="a" * 64,
        expected_source_checkpoint_sha256="b" * 64,
        dynamic_parity_dir=str(tmp_path / "parity"),
        strict_zero_shot_reference_dir=None,
        feature_cache_root=str(tmp_path / "feature-cache-root"),
        feature_cache_policy="auto",
        output_dir=str(output),
        device="cpu",
        feature_cache_batch_size=8,
        batch_size=64,
        epochs=1,
        early_stopping_patience=1,
        seed=42,
        split_seed=42,
    )
    summary = runner.run(args)
    assert summary["status"] == "passed"
    assert summary["candidate_count_per_fold"] == 1
    assert summary["test_evaluation_count"] == 7
    assert summary["test_accessed_during_selection"] is False
    assert summary["portable_source_age_probe_used"] is False
    assert summary["age_target_normalization"] == "fixed_source_zscore"
    assert summary["source_scaler_reused"] is True
    assert (output / "_SUCCESS").is_file()
    assert len(list(output.glob("fold_*/test_metrics.json"))) == 7
    assert len(list(output.glob("fold_*/selected/selection.json"))) == 7
    assert len(list(output.glob("fold_*/candidates/*/best_head.pt"))) == 7
    oof = (output / "oof_subject_predictions.csv").read_text(encoding="utf-8-sig")
    assert "SUB5" not in oof


def test_thin_family_wrappers_share_one_implementation():
    root = Path(__file__).resolve().parents[1]
    generic = (root / "commands" / "run_huada_portable_age_zero_init_probe_cv7.sh").read_text(
        encoding="utf-8"
    )
    assert "run_huada_portable_age_zero_init_probe_cv7" in generic
    for family in ("labram", "reve"):
        text = (root / "commands" / f"run_huada_{family}_zero_init_probe_cv7.sh").read_text(
            encoding="utf-8"
        )
        assert f"MODEL_FAMILY={family}" in text
        assert "run_huada_portable_age_zero_init_probe_cv7.sh" in text
