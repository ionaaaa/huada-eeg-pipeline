from __future__ import annotations

import csv
import json
import logging
import sys
from argparse import Namespace
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import pytest
import torch

PROJECT = Path(__file__).resolve().parents[1]
MODULE_DIR = PROJECT / "scripts" / "downstream_50m"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import diagnose_age_head_convergence as diagnostic
from feature_cache import feature_rows_to_table
from largeScale_fineTune import CachedFeatureClassifier, load_downstream_checkpoint_strict


def _fold_info():
    return {
        "fold": 0,
        "train_subjects": ["SUB1", "SUB2", "SUB3", "SUB4", "SUB6"],
        "validation_subject": "SUB7",
        "validation_subjects": ["SUB7"],
        "test_subject": "SUB8",
        "test_subjects": ["SUB8"],
    }


def _records(rows_per_subject=4, feature_scale=10.0):
    ages = {"SUB1": 22.0, "SUB2": 24.0, "SUB3": 26.0, "SUB4": 28.0, "SUB6": 30.0, "SUB7": 27.0}
    records = []
    for subject_index, (subject, age) in enumerate(ages.items()):
        for window in range(rows_per_subject):
            feature = np.zeros(512, dtype=np.float32)
            feature[subject_index] = feature_scale
            feature[20 + window] = 0.01 * window
            records.append(diagnostic.FeatureRecord(
                sample_id=f"{subject}_W{window}", subject_id=subject,
                recording_id="R1", window_index=window, age=age,
                feature=feature, num_valid_tokens=10,
            ))
    return records


def _args(**overrides):
    values = dict(
        fold=0, split_seed=42, train_seed=42, head_lr=0.1, weight_decay=0.0,
        batch_size=64, mini_samples_per_subject=2, mini_max_epochs=150,
        full_max_epochs=150, mini_convergence_mae=0.1, full_train_good_mae=1.0,
        plateau_window=5, plateau_slope_tolerance=1e-4, step_log_interval=1,
        gradient_clip_norm=None, device="cpu", log_level="INFO",
    )
    values.update(overrides)
    return Namespace(**values)


def _logger():
    logger = logging.getLogger("age-convergence-test")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    return logger


def _csv_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_canonical_fold_is_reproducible_and_label_free():
    subjects = ["SUB1", "SUB2", "SUB3", "SUB4", "SUB6", "SUB7", "SUB8"]
    first = diagnostic.resolve_canonical_fold(subjects, 42, 0)
    second = diagnostic.resolve_canonical_fold(reversed(subjects), 42, 0)
    assert first == second
    assert first["labels_used_to_resolve_assignment"] is False
    assert len(first["train_subjects"]) == 5
    assert len(first["validation_subjects"]) == len(first["test_subjects"]) == 1


def test_mini_sampling_is_stratified_and_reproducible():
    train = [row for row in _records() if row.subject_id in set(_fold_info()["train_subjects"])]
    left = diagnostic.stratified_mini_sample(train, 2, 42)
    right = diagnostic.stratified_mini_sample(train, 2, 42)
    assert [row.sample_id for row in left] == [row.sample_id for row in right]
    counts = {subject: 0 for subject in _fold_info()["train_subjects"]}
    for row in left:
        counts[row.subject_id] += 1
    assert set(counts.values()) == {2}
    assert len({row.age for row in left}) > 1


def test_test_sentinel_is_never_materialized():
    class GuardedSource:
        def __init__(self):
            self.calls = []

        def materialize_subjects(self, subjects, *, forbidden_subjects):
            self.calls.append((set(subjects), set(forbidden_subjects)))
            assert "SUB8" not in subjects
            assert set(forbidden_subjects) == {"SUB8"}
            return [row for row in _records() if row.subject_id in set(subjects)]

    source = GuardedSource()
    records = diagnostic.load_allowed_records(source, _fold_info())
    assert source.calls and all(row.subject_id != "SUB8" for row in records)


def test_join_rejects_test_duplicate_and_missing_samples():
    feature = np.zeros(512, dtype=np.float32)
    source = [{"sample_id": "A", "subject_id": "SUB1", "recording_id": "R", "window_index": 0, "age": 22.0}]
    cache = [{"sample_id": "A", "subject_id": "SUB1", "recording_id": "R", "feature": feature, "num_valid_tokens": 1}]
    assert len(diagnostic.join_selected_rows(source, cache, {"SUB1"}, {"SUB8"})) == 1
    with pytest.raises(diagnostic.DataValidationError):
        diagnostic.join_selected_rows(source, cache, {"SUB1"}, {"SUB1"})
    with pytest.raises(diagnostic.DataValidationError, match="Duplicate"):
        diagnostic.join_selected_rows(source + source, cache, {"SUB1"}, {"SUB8"})
    with pytest.raises(diagnostic.DataValidationError, match="mismatch"):
        diagnostic.join_selected_rows(source, [], {"SUB1"}, {"SUB8"})


def test_feature_diagnostics_fail_closed_on_nonfinite_and_wrong_dim():
    bad = _records()[:1]
    bad[0].feature[0] = np.nan
    with pytest.raises(diagnostic.NonFiniteError):
        diagnostic.feature_diagnostics(bad)
    wrong = [diagnostic.FeatureRecord("x", "SUB1", "R", 0, 22.0, np.zeros(3, np.float32))]
    with pytest.raises(diagnostic.DataValidationError):
        diagnostic.feature_diagnostics(wrong)


def test_gradient_statistics_are_available_before_optimizer_step():
    class GuardedAdamW(torch.optim.AdamW):
        def __init__(self, params, **kwargs):
            super().__init__(params, **kwargs)
            self.saw_gradients = False

        def step(self, closure=None):
            gradients = [parameter.grad for group in self.param_groups for parameter in group["params"]]
            assert gradients and all(value is not None and torch.isfinite(value).all() for value in gradients)
            self.saw_gradients = True
            return super().step(closure)

    model = diagnostic.create_age_head(42, torch.device("cpu"))
    optimizer = GuardedAdamW(model.parameters(), lr=0.01)
    rows = _records()[:3]
    batch = diagnostic.collate_records([diagnostic.FeatureRecordDataset(rows)[i] for i in range(3)])
    batch["forbidden_subjects"] = {"SUB8"}
    result = diagnostic.perform_training_step(
        model=model, optimizer=optimizer, criterion=torch.nn.MSELoss(), batch=batch,
        device=torch.device("cpu"), experiment="unit", epoch=1, global_step=1,
        batch_index=1, initial_parameters=diagnostic._head_vector(model).clone(),
        gradient_clip_norm=None, elapsed_seconds=0.0,
    )
    assert optimizer.saw_gradients
    assert result["gradient_finite"] is True
    assert result["gradient_total_l2_norm"] > 0
    assert result["gradient_norm_before_clip"] is None
    assert result["gradient_norm_after_clip"] is None
    assert result["gradient_clip_threshold"] is None
    assert result["optimizer_step_applied"] is True


def test_feature_prediction_gradient_and_parameter_nonfinite_fail_closed():
    rows = _records()[:3]

    def batch():
        value = diagnostic.collate_records([diagnostic.FeatureRecordDataset(rows)[i] for i in range(3)])
        value["forbidden_subjects"] = {"SUB8"}
        return value

    def invoke(model, optimizer, value):
        return diagnostic.perform_training_step(
            model=model, optimizer=optimizer, criterion=torch.nn.MSELoss(), batch=value,
            device=torch.device("cpu"), experiment="unit", epoch=1, global_step=1,
            batch_index=1, initial_parameters=diagnostic._head_vector(model).clone(),
            gradient_clip_norm=None, elapsed_seconds=0.0,
        )

    model = diagnostic.create_age_head(42, torch.device("cpu"))
    value = batch()
    value["feature"][0, 0] = float("nan")
    with pytest.raises(diagnostic.NonFiniteError, match="feature"):
        invoke(model, torch.optim.AdamW(model.parameters(), lr=0.01), value)

    model = diagnostic.create_age_head(42, torch.device("cpu"))
    with torch.no_grad():
        model.head.bias.fill_(float("inf"))
    with pytest.raises(diagnostic.NonFiniteError, match="Prediction"):
        invoke(model, torch.optim.AdamW(model.parameters(), lr=0.01), batch())

    model = diagnostic.create_age_head(42, torch.device("cpu"))
    hook = model.head.weight.register_hook(lambda grad: torch.full_like(grad, float("inf")))
    with pytest.raises(diagnostic.NonFiniteError, match="Gradient"):
        invoke(model, torch.optim.AdamW(model.parameters(), lr=0.01), batch())
    hook.remove()

    class CorruptingAdamW(torch.optim.AdamW):
        def step(self, closure=None):
            result = super().step(closure)
            with torch.no_grad():
                self.param_groups[0]["params"][0].fill_(float("inf"))
            return result

    model = diagnostic.create_age_head(42, torch.device("cpu"))
    with pytest.raises(diagnostic.NonFiniteError, match="parameters"):
        invoke(model, CorruptingAdamW(model.parameters(), lr=0.01), batch())


def test_train_validation_subject_and_sample_leakage_rejected(tmp_path):
    records = _records()
    train_sample = next(row for row in records if row.subject_id == "SUB1")
    validation_index = next(i for i, row in enumerate(records) if row.subject_id == "SUB7")
    validation = records[validation_index]
    records[validation_index] = diagnostic.FeatureRecord(
        sample_id=train_sample.sample_id, subject_id=validation.subject_id,
        recording_id=validation.recording_id, window_index=validation.window_index,
        age=validation.age, feature=validation.feature,
    )
    with pytest.raises(diagnostic.DataValidationError, match="leakage"):
        diagnostic.execute_from_records(
            records=records, fold_info=_fold_info(), output_dir=tmp_path,
            args=_args(), logger=_logger(), cache_key="fake",
        )


def test_synthetic_mini_overfit_full_reinit_and_trajectory_artifacts(tmp_path):
    result = diagnostic.execute_from_records(
        records=_records(), fold_info=_fold_info(), output_dir=tmp_path,
        args=_args(), logger=_logger(), cache_key="fake-cache",
    )
    mini, full = result["mini_overfit"], result["full_train"]
    assert mini["converged"] is True and mini["minimum_mae"] <= 0.1
    assert full["initial_head_fingerprint"] == mini["initial_head_fingerprint"]
    assert full["initial_head_fingerprint"] != mini["final_head_fingerprint"]
    assert result["test_dataset_created"] is False
    assert result["test_dataloader_created"] is False
    assert result["test_rows_materialized"] == result["test_forward_count"] == 0
    assert result["test_metrics_computed"] is False
    mini_steps = _csv_rows(tmp_path / "mini_overfit/step_trajectory.csv")
    mini_epochs = _csv_rows(tmp_path / "mini_overfit/epoch_trajectory.csv")
    full_steps = _csv_rows(tmp_path / "full_train/step_trajectory.csv")
    full_epochs = _csv_rows(tmp_path / "full_train/epoch_trajectory.csv")
    assert len(mini_steps) == mini["optimizer_steps"] == mini["step_trajectory_rows"]
    assert len(mini_epochs) == mini["epoch_trajectory_rows"]
    assert len(full_steps) == full["optimizer_steps"] == full["step_trajectory_rows"]
    assert len(full_epochs) == full["epoch_trajectory_rows"]
    assert [int(row["epoch"]) for row in mini_epochs] == list(range(1, len(mini_epochs) + 1))
    assert all(row["validation_mse"] == "" for row in mini_epochs)
    assert len(_csv_rows(tmp_path / "feature_dimension_statistics.csv")) == 3 * 512
    assert len(_csv_rows(tmp_path / "selected_mini_samples.csv")) == 10


def test_artifact_verifier_checks_summary_and_stable_files(tmp_path):
    result = diagnostic.execute_from_records(
        records=_records(), fold_info=_fold_info(), output_dir=tmp_path,
        args=_args(), logger=_logger(), cache_key="fake-cache",
    )
    diagnostic._write_csv(tmp_path / "split_subjects.csv", [], ["subject_id", "split", "materialized"])
    diagnostic._write_csv(tmp_path / "diagnostic_issues.csv", [], diagnostic.ISSUE_FIELDS)
    (tmp_path / "diagnostic_errors.log").write_text("", encoding="utf-8")
    assert diagnostic.verify_trajectory_artifacts(tmp_path, result) is True
    result["mini_overfit"]["step_trajectory_rows"] += 1
    with pytest.raises(RuntimeError, match="row-count mismatch"):
        diagnostic.verify_trajectory_artifacts(tmp_path, result)


def test_nonfinite_training_status_is_not_completed(tmp_path):
    records = _records(rows_per_subject=2, feature_scale=1e20)
    args = _args(mini_max_epochs=2, full_max_epochs=2)
    result = diagnostic.execute_from_records(
        records=records, fold_info=_fold_info(), output_dir=tmp_path,
        args=args, logger=_logger(), cache_key="fake",
    )
    assert result["status"] == "failed_non_finite"
    assert result["mini_overfit"]["status"] == "failed_non_finite"
    assert result["mini_overfit"]["status"] not in {"completed_converged", "completed_not_converged"}
    assert result["all_values_finite"] is False


def _write_lance_fixture(root: Path, *, pooling="mean", feature_dim=512, duplicate=False):
    subjects = ["SUB1", "SUB2", "SUB3", "SUB4", "SUB5", "SUB6", "SUB7", "SUB8"]
    ages = [22.0, 24.0, 26.0, 28.0, None, 30.0, 29.0, 27.0]
    sample_ids = [f"{subject}_W0" for subject in subjects]
    source = pa.table({
        "sample_id": sample_ids,
        "subject_id": subjects,
        "recording_id": ["R1"] * 8,
        "window_index": list(range(8)),
        "age": ages,
    })
    source_path = root / "input.lance"
    lance.write_dataset(source, str(source_path), mode="create")
    cache_dir = root / "cache"
    cache_dir.mkdir()
    features = np.zeros((8, feature_dim), dtype=np.float32)
    for index in range(8):
        features[index, index] = 10
    ids = sample_ids[:-1] + [sample_ids[-2]] if duplicate else sample_ids
    table = feature_rows_to_table(ids, subjects, ["R1"] * 8, features, np.ones(8, np.int32))
    lance.write_dataset(table, str(cache_dir / "mean_features.lance"), mode="create")
    manifest = {
        "status": "completed", "cache_key": "fixture-key", "parity_passed": True,
        "requested_poolings": [pooling], "mean_feature_dim": feature_dim,
        "dtype": "float32", "input_row_count": 8, "mean_row_count": 8,
        "sample_ids": sample_ids,
    }
    (cache_dir / "cache_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (cache_dir / "_SUCCESS").write_text("fixture-key\n", encoding="utf-8")
    return source_path, cache_dir / "mean_features.lance", cache_dir / "cache_manifest.json"


@pytest.mark.parametrize("pooling,dim", [("flatten", 512), ("mean", 511)])
def test_cache_rejects_non_mean_or_wrong_dimension(tmp_path, pooling, dim):
    source, cache, manifest = _write_lance_fixture(tmp_path, pooling=pooling, feature_dim=dim)
    with pytest.raises(diagnostic.DataValidationError):
        diagnostic.LanceDiagnosticSource(source, cache, manifest)


def test_cache_duplicate_sample_id_rejected_when_allowed_rows_loaded(tmp_path):
    source, cache, manifest = _write_lance_fixture(tmp_path, duplicate=True)
    reader = diagnostic.LanceDiagnosticSource(source, cache, manifest)
    fold = diagnostic.resolve_canonical_fold(reader.discover_subject_ids(), 42, 0)
    with pytest.raises(diagnostic.DataValidationError):
        diagnostic.load_allowed_records(reader, fold)


def test_sub5_null_excluded_and_no_backbone_or_sweep_path(tmp_path):
    source, cache, manifest = _write_lance_fixture(tmp_path)
    reader = diagnostic.LanceDiagnosticSource(source, cache, manifest)
    assert reader.discover_subject_ids() == ["SUB1", "SUB2", "SUB3", "SUB4", "SUB6", "SUB7", "SUB8"]
    text = Path(diagnostic.__file__).read_text(encoding="utf-8")
    assert "EEGPretrainModel" not in text
    assert "build_feature_cache(" not in text
    assert "run_age_cv7_sweep" not in text
    assert "evaluate_age_test_once" not in text


def test_end_to_end_tiny_lance_smoke_writes_summary_and_guards(tmp_path):
    source, cache, manifest = _write_lance_fixture(tmp_path)
    output = tmp_path / "output"
    args = _args(
        input_lance=str(source), feature_cache_lance=str(cache),
        feature_cache_manifest=str(manifest), output_dir=str(output),
        mini_samples_per_subject=1, mini_max_epochs=2, full_max_epochs=2,
    )
    assert diagnostic.run(args) == 0
    summary = json.loads((output / "convergence_summary.json").read_text(encoding="utf-8"))
    provenance = json.loads((output / "processing_provenance.json").read_text(encoding="utf-8"))
    assert summary["trajectory_artifact_validation_passed"] is True
    assert summary["test_rows_materialized"] == 0
    assert summary["test_metrics_computed"] is False
    assert provenance["backbone_constructed"] is False
    assert (output / "run.log").is_file()
    assert (output / "convergence_report.md").is_file()


def test_diagnostic_checkpoint_is_rejected_by_formal_loader(tmp_path):
    state = diagnostic._state_copy(diagnostic.create_age_head(42, torch.device("cpu")))
    path = tmp_path / "diagnostic.pt"
    diagnostic._diagnostic_checkpoint(path, state, "full_train", 1, _fold_info(), _args(), "fake")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["checkpoint_kind"] == diagnostic.CHECKPOINT_KIND
    assert payload["not_for_formal_evaluation"] is True
    assert "head_state_dict" not in payload and "model_state_dict" not in payload
    formal_model = CachedFeatureClassifier(512, 1, "mean")
    with pytest.raises(RuntimeError, match="no model_state_dict"):
        load_downstream_checkpoint_strict(formal_model, path, "cpu")


def test_empty_issue_and_error_artifacts_have_stable_schema(tmp_path):
    diagnostic._write_csv(tmp_path / "diagnostic_issues.csv", [], diagnostic.ISSUE_FIELDS)
    (tmp_path / "diagnostic_errors.log").write_text("", encoding="utf-8")
    assert (tmp_path / "diagnostic_issues.csv").read_text().strip().split(",") == diagnostic.ISSUE_FIELDS
    assert (tmp_path / "diagnostic_errors.log").read_text() == ""


def test_output_directory_existing_is_refused(tmp_path):
    args = _args()
    args.output_dir = str(tmp_path)
    with pytest.raises(FileExistsError):
        diagnostic.run(args)


def test_cloud_script_is_diagnostic_only_and_has_safe_defaults():
    script = (PROJECT / "commands/run_50m_age_convergence_diagnostic_full.sh").read_text()
    assert "set -euo pipefail" in script
    assert "FOLD=\"${FOLD:-0}\"" in script
    assert "HEAD_LR=\"${HEAD_LR:-0.0003}\"" in script
    assert "MINI_MAX_EPOCHS=\"${MINI_MAX_EPOCHS:-1000}\"" in script
    assert "FULL_MAX_EPOCHS=\"${FULL_MAX_EPOCHS:-300}\"" in script
    assert script.count("diagnose_age_head_convergence.py") == 2
    lower = script.lower()
    for forbidden in ("run_age_cv7_sweep", "zero_shot", "feature-cache-policy", "--overwrite"):
        assert forbidden not in lower
