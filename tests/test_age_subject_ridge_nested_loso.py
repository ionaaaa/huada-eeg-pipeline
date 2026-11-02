from __future__ import annotations

import csv
import json
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import lance
import numpy as np
import pyarrow as pa
import pytest
import torch


PROJECT = Path(__file__).resolve().parents[1]
MODULE_DIR = PROJECT / "scripts" / "downstream_50m"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import run_age_subject_ridge_nested_loso as ridge
from feature_cache import feature_rows_to_table
from largeScale_fineTune import load_downstream_checkpoint_strict


AGES = {
    "SUB1": 22.0, "SUB2": 24.0, "SUB3": 26.0, "SUB4": 28.0,
    "SUB5": None, "SUB6": 30.0, "SUB7": 28.0, "SUB8": 31.0,
}


def _subject_rows(*, signal=True, nonfinite=False):
    rows = []
    for index, (subject, age) in enumerate(AGES.items()):
        feature = np.zeros(512, dtype=np.float64)
        feature[0] = 0.0 if age is None else age
        if signal:
            feature[1] = 0.0 if age is None else 2.0 * age
        else:
            feature[index + 10] = 1.0
        if nonfinite and subject == "SUB3":
            feature[4] = np.nan
        rows.append(ridge.SubjectFeatureRow(
            subject_id=subject, age=age, eligibility=age is not None,
            exclusion_reason="missing_age" if age is None else "",
            window_count=index + 2, feature=feature,
            sample_ids=tuple(f"{subject}_W{window}" for window in range(index + 2)),
        ))
    return rows


def _window_rows():
    rows = []
    for index, (subject, age) in enumerate(AGES.items()):
        for window in range(index + 2):
            feature = np.zeros(512, dtype=np.float32)
            feature[0] = index + window
            rows.append(ridge.WindowFeatureRow(
                sample_id=f"{subject}_W{window}", subject_id=subject,
                recording_id="R1", window_index=window, age=age, feature=feature,
            ))
    return rows


def _write_fixture(root: Path, *, pooling="mean", dim=512, conflict_age=False, nonfinite=False):
    source_rows = []
    cache_ids, subjects, recordings, features, valid = [], [], [], [], []
    for subject_index, (subject, age) in enumerate(AGES.items()):
        window_count = subject_index + 2
        for window in range(window_count):
            sample_id = f"{subject}_W{window}"
            row_age = age
            if conflict_age and subject == "SUB2" and window == window_count - 1:
                row_age = float(age) + 1.0
            source_rows.append({
                "sample_id": sample_id, "subject_id": subject, "recording_id": "R1",
                "window_index": window, "age": row_age,
            })
            feature = np.zeros(dim, dtype=np.float32)
            if age is not None:
                feature[0] = float(age) + 0.01 * (window - (window_count - 1) / 2.0)
                feature[1] = 2.0 * float(age)
            feature[2 + subject_index] = 0.001
            if nonfinite and subject == "SUB3" and window == 0:
                feature[0] = np.nan
            cache_ids.append(sample_id)
            subjects.append(subject)
            recordings.append("R1")
            features.append(feature)
            valid.append(10)
    input_path = root / "input.lance"
    lance.write_dataset(pa.Table.from_pylist(source_rows), str(input_path), mode="create")
    cache_dir = root / "cache"
    cache_dir.mkdir()
    cache_path = cache_dir / "mean_features.lance"
    lance.write_dataset(
        feature_rows_to_table(cache_ids, subjects, recordings, np.stack(features), valid),
        str(cache_path), mode="create",
    )
    manifest = {
        "status": "completed", "cache_key": "fixture-key", "parity_passed": True,
        "requested_poolings": [pooling], "mean_feature_dim": dim,
        "dtype": "float32", "input_row_count": len(source_rows),
        "mean_row_count": len(source_rows), "sample_ids": cache_ids,
    }
    manifest_path = cache_dir / "cache_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (cache_dir / "_SUCCESS").write_text("fixture-key\n", encoding="utf-8")
    return input_path, cache_path, manifest_path


def _args(input_path, cache_path, manifest_path, output_path, **overrides):
    values = dict(
        input_lance=str(input_path), feature_cache_lance=str(cache_path),
        feature_cache_manifest=str(manifest_path), output_dir=str(output_path),
        alphas=list(ridge.DEFAULT_ALPHAS), aggregation="mean",
        feature_transform="train_center", target_transform="train_subject_zscore",
        selection_metric="mae", device="cpu", log_level="WARNING",
    )
    values.update(overrides)
    return Namespace(**values)


def _csv_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_window_features_are_averaged_per_subject_in_float64():
    aggregated = {row.subject_id: row for row in ridge.aggregate_subject_features(_window_rows())}
    sub3_windows = [row.feature.astype(np.float64) for row in _window_rows() if row.subject_id == "SUB3"]
    np.testing.assert_allclose(aggregated["SUB3"].feature, np.mean(sub3_windows, axis=0))
    assert aggregated["SUB3"].aggregation_dtype == "float64"
    assert aggregated["SUB5"].eligibility is False
    assert aggregated["SUB5"].exclusion_reason == "missing_age"


def test_window_count_does_not_weight_subject_regression():
    rows = _subject_rows()
    eligible = ridge.validate_huada_subject_inventory(rows)
    assert len(eligible) == 7
    x = np.stack([row.feature for row in eligible[:5]])
    y = np.asarray([row.age for row in eligible[:5]])
    prep = ridge.fit_train_preprocessing(x, y)
    assert prep["target_mean"] == pytest.approx(float(np.mean(y)))
    assert prep["target_mean"] != pytest.approx(np.average(y, weights=[row.window_count for row in eligible[:5]]))


def test_nested_loso_has_exact_expected_membership_and_counts():
    subjects = [row.subject_id for row in _subject_rows() if row.eligibility]
    splits = ridge.build_nested_loso_splits(reversed(subjects))
    assert len(splits) == 42
    assert {row["outer_fold"] for row in splits} == set(range(7))
    for outer_fold in range(7):
        fold = [row for row in splits if row["outer_fold"] == outer_fold]
        assert len(fold) == 6
        assert len({row["inner_validation_subject_id"] for row in fold}) == 6
        assert all(len(row["inner_train_subject_ids"]) == 5 for row in fold)
        assert all(row["outer_test_subject_id"] not in row["inner_train_subject_ids"] for row in fold)
        assert all(row["outer_test_subject_id"] != row["inner_validation_subject_id"] for row in fold)


def test_train_preprocessing_is_train_only_and_equal_subject_weight():
    rows = _subject_rows()
    train = rows[:5]
    x = np.stack([row.feature for row in train])
    y = np.asarray([row.age if row.age is not None else 23.0 for row in train])
    prep = ridge.fit_train_preprocessing(x, y)
    np.testing.assert_allclose(prep["feature_center"], x.mean(axis=0))
    assert prep["target_mean"] == pytest.approx(y.mean())
    assert prep["target_std"] == pytest.approx(y.std(ddof=0))
    altered_validation = rows[6].feature + 1e6
    assert not np.allclose(altered_validation, prep["feature_center"])
    with pytest.raises(ridge.RidgeValidationError, match="standard deviation"):
        ridge.fit_train_preprocessing(x, np.ones(5))


def test_dual_ridge_matches_primal_reference_and_uses_float64(monkeypatch):
    rng = np.random.default_rng(42)
    x = rng.normal(size=(5, 512)).astype(np.float64)
    x -= x.mean(axis=0)
    y = rng.normal(size=5).astype(np.float64)
    y -= y.mean()
    monkeypatch.setattr(torch, "inverse", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("inverse")))
    solved = ridge.solve_dual_ridge(x, y, 0.1)
    reference = np.linalg.solve(x.T @ x + 0.1 * np.eye(512), x.T @ y)
    np.testing.assert_allclose(solved["coefficient"], reference, atol=1e-10, rtol=1e-8)
    assert solved["compute_dtype"] == "float64"
    assert solved["compute_device"] == "cpu"
    assert solved["solver_form"] == "dual"
    assert solved["fit_intercept"] is False


def test_alpha_tie_break_uses_tolerance_then_stronger_regularization():
    rows = [
        {"alpha": 1.0, "mae": 2.0, "rmse": 3.0, "max_absolute_error": 4.0},
        {"alpha": 100.0, "mae": 2.0 + 1e-12, "rmse": 3.0, "max_absolute_error": 4.0},
    ]
    assert ridge.select_alpha(rows, tolerance=1e-9)["alpha"] == 100.0
    rows[1]["mae"] = 2.01
    assert ridge.select_alpha(rows, tolerance=1e-9)["alpha"] == 1.0


def test_outer_guard_denies_test_until_lock_and_allows_exactly_once():
    accessor = ridge.GuardedOuterAccessor(_subject_rows(), "SUB8")
    with pytest.raises(ridge.RidgeValidationError, match="before selection lock"):
        accessor.feature("SUB8")
    with pytest.raises(ridge.RidgeValidationError, match="before selection lock"):
        accessor.age("SUB8")
    accessor.feature("SUB1")
    accessor.age("SUB1")
    accessor.lock("fingerprint")
    accessor.feature("SUB8")
    accessor.age("SUB8")
    accessor.mark_test_prediction()
    record = accessor.isolation_record()
    assert record["selection_precedes_test_access"] is True
    assert record["test_evaluation_count"] == 1
    with pytest.raises(ridge.RidgeValidationError, match="more than once"):
        accessor.feature("SUB8")
    with pytest.raises(ridge.RidgeValidationError, match="more than once"):
        accessor.mark_test_prediction()


def test_nested_execution_produces_252_inner_and_7_unique_oof(tmp_path):
    result = ridge.run_nested_loso_from_subjects(
        _subject_rows(), ridge.DEFAULT_ALPHAS, tmp_path / "first",
        cache_key="fixture", cache_manifest_sha256="abc", logger=None,
    )
    assert result["inner_prediction_count"] == 252
    assert len(result["inner_results"]) == 42
    assert result["outer_prediction_count"] == 7
    assert len({row["subject_id"] for row in result["oof_rows"]}) == 7
    assert all(row["test_evaluation_count"] == 1 for row in result["isolation_rows"])
    assert all(row["selection_precedes_test_access"] for row in result["isolation_rows"])
    assert result["model_metrics"]["mae"] < result["baseline_metrics"]["mae"]
    truth = [row["true_age"] for row in result["oof_rows"]]
    prediction = [row["predicted_age"] for row in result["oof_rows"]]
    assert result["model_metrics"] == ridge.complete_subject_metrics(truth, prediction)
    age_by_subject = {row.subject_id: row.age for row in _subject_rows() if row.eligibility}
    for selected in result["selected_rows"]:
        development = selected["outer_development_subject_ids"].split(";")
        assert selected["outer_target_mean"] == pytest.approx(
            np.mean([age_by_subject[subject] for subject in development])
        )
    repeated = ridge.run_nested_loso_from_subjects(
        _subject_rows(), ridge.DEFAULT_ALPHAS, tmp_path / "second",
        cache_key="fixture", cache_manifest_sha256="abc", logger=None,
    )
    assert repeated["oof_rows"] == result["oof_rows"]
    assert [row["selected_alpha"] for row in repeated["selected_rows"]] == [
        row["selected_alpha"] for row in result["selected_rows"]
    ]


def test_baseline_is_outer_development_mean(tmp_path):
    rows = _subject_rows()
    result = ridge.run_nested_loso_from_subjects(
        rows, [0.01], tmp_path,
        cache_key="fixture", cache_manifest_sha256="abc", logger=None,
    )
    ages = {row.subject_id: row.age for row in rows if row.eligibility}
    for prediction in result["oof_rows"]:
        development = [age for subject, age in ages.items() if subject != prediction["subject_id"]]
        assert prediction["baseline_predicted_age"] == pytest.approx(np.mean(development))


@pytest.mark.parametrize("pooling,dim", [("flatten", 512), ("mean", 511)])
def test_cache_rejects_non_mean_only_or_wrong_dimension(tmp_path, pooling, dim):
    input_path, cache_path, manifest_path = _write_fixture(tmp_path, pooling=pooling, dim=dim)
    with pytest.raises(ridge.RidgeValidationError):
        ridge.ExistingMeanCacheSource(input_path, cache_path, manifest_path)


def test_conflicting_subject_age_and_nonfinite_feature_fail_closed(tmp_path):
    conflict_root = tmp_path / "conflict"
    conflict_root.mkdir()
    paths = _write_fixture(conflict_root, conflict_age=True)
    source = ridge.ExistingMeanCacheSource(*paths)
    with pytest.raises(ridge.RidgeValidationError, match="Conflicting ages"):
        ridge.aggregate_subject_features(source.load_and_join())
    nonfinite_root = tmp_path / "nonfinite"
    nonfinite_root.mkdir()
    paths = _write_fixture(nonfinite_root, nonfinite=True)
    source = ridge.ExistingMeanCacheSource(*paths)
    with pytest.raises(ridge.RidgeValidationError, match="Non-finite"):
        ridge.aggregate_subject_features(source.load_and_join())
    x = np.zeros((5, 512), dtype=np.float64)
    y = np.arange(5, dtype=np.float64)
    x[0, 0] = np.nan
    with pytest.raises(ridge.RidgeValidationError, match="Non-finite"):
        ridge.solve_dual_ridge(x, y, 1.0)


def test_end_to_end_synthetic_smoke_is_deterministic_and_beats_baseline(tmp_path):
    fixture_root = tmp_path / "fixture"
    fixture_root.mkdir()
    input_path, cache_path, manifest_path = _write_fixture(fixture_root)
    output = tmp_path / "output"
    assert ridge.run(_args(input_path, cache_path, manifest_path, output)) == 0
    inner = _csv_rows(output / "inner_cv_predictions.csv")
    oof = _csv_rows(output / "oof_subject_predictions.csv")
    inventory = _csv_rows(output / "subject_feature_inventory.csv")
    model = json.loads((output / "oof_subject_metrics.json").read_text(encoding="utf-8"))
    baseline = json.loads((output / "baseline_subject_metrics.json").read_text(encoding="utf-8"))
    provenance = json.loads((output / "processing_provenance.json").read_text(encoding="utf-8"))
    assert len(inner) == 252 and len(oof) == 7 and len(inventory) == 8
    assert model["mae"] < baseline["mae"]
    assert provenance["backbone_forward_count"] == 0
    assert provenance["optimizer_created"] is False
    assert provenance["selection_precedes_test_access_all_folds"] is True
    assert next(row for row in inventory if row["subject_id"] == "SUB5")["exclusion_reason"] == "missing_age"
    assert (output / "model_artifacts/outer_fold_0/selection.json").is_file()
    assert (output / "model_artifacts/outer_fold_0/model.npz").is_file()
    assert (output / "ridge_issues.csv").read_text(encoding="utf-8-sig").splitlines()[0]
    assert (output / "ridge_errors.log").read_text(encoding="utf-8").startswith("timestamp_utc")
    with pytest.raises(ridge.RidgeValidationError, match="already exists"):
        ridge.run(_args(input_path, cache_path, manifest_path, output))


def test_model_artifact_is_not_gradient_head_checkpoint(tmp_path):
    ridge.run_nested_loso_from_subjects(
        _subject_rows(), [1.0], tmp_path,
        cache_key="fixture", cache_manifest_sha256="abc", logger=None,
    )
    artifact = tmp_path / "model_artifacts/outer_fold_0/model.npz"
    manifest = json.loads((artifact.parent / "model_manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifact_kind"] == "age_subject_ridge_outer_fold"
    assert manifest["not_compatible_with_gradient_head_loader"] is True
    with pytest.raises(Exception):
        load_downstream_checkpoint_strict(
            SimpleNamespace(pooling="mean", head_input_dim=512), artifact, "cpu"
        )


def test_cloud_script_is_cpu_cache_require_only():
    text = (PROJECT / "commands/run_50m_age_subject_ridge_nested_loso_full.sh").read_text(encoding="utf-8")
    assert "--device cpu" in text
    assert "run_age_subject_ridge_nested_loso.py" in text
    for forbidden in ("build_feature_cache", "run_age_cv7_sweep", "cuda", "--overwrite", "optimizer"):
        assert forbidden not in text
