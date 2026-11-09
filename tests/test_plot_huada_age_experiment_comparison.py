from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
import pytest

from scripts import plot_huada_age_experiment_comparison as plotter


AGES = {
    "SUB1": 31.0, "SUB2": 29.0, "SUB3": 26.0, "SUB4": 26.0,
    "SUB6": 28.0, "SUB7": 28.0, "SUB8": 22.0,
}


def write_csv(path: Path, *, subject_field="subject_id", age_field="true_age", prediction_field="mean_prediction", rows=None):
    rows = rows or [
        {subject_field: subject, age_field: age, prediction_field: age + index * 0.1}
        for index, (subject, age) in enumerate(AGES.items())
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def test_field_aliases_subject_set_and_sub5_exclusion(tmp_path):
    path = tmp_path / "subject_predictions.csv"
    rows = [
        {"participant_id": subject, "age": age, "mean_predicted_age_years": age + 0.5}
        for subject, age in AGES.items()
    ]
    rows.append({"participant_id": "SUB5", "age": "", "mean_predicted_age_years": 27.0})
    write_csv(path, rows=rows)
    parsed = plotter.read_subject_predictions(path)
    assert tuple(parsed) == plotter.EXPECTED_SUBJECTS
    assert "SUB5" not in parsed and parsed["SUB1"]["prediction"] == 31.5


def test_duplicate_subject_is_rejected(tmp_path):
    path = tmp_path / "predictions.csv"
    rows = [{"subject_id": s, "true_age": a, "prediction": a} for s, a in AGES.items()]
    rows.append(dict(rows[0]))
    write_csv(path, rows=rows)
    with pytest.raises(plotter.ComparisonValidationError, match="Duplicate"):
        plotter.read_subject_predictions(path)


def test_missing_subject_is_rejected(tmp_path):
    path = tmp_path / "predictions.csv"
    rows = [{"subject_id": s, "true_age": a, "prediction": a} for s, a in list(AGES.items())[:-1]]
    write_csv(path, rows=rows)
    with pytest.raises(plotter.ComparisonValidationError, match="Subject set mismatch"):
        plotter.read_subject_predictions(path)


@pytest.mark.parametrize("bad", ["nan", "inf", "-inf"])
def test_nonfinite_is_rejected(tmp_path, bad):
    path = tmp_path / "predictions.csv"
    rows = [{"subject_id": s, "true_age": a, "prediction": a} for s, a in AGES.items()]
    rows[0]["prediction"] = bad
    write_csv(path, rows=rows)
    with pytest.raises(plotter.ComparisonValidationError, match="Non-finite"):
        plotter.read_subject_predictions(path)


def test_true_age_mismatch_is_rejected():
    predictions = {
        experiment: {subject: {"true_age": age, "prediction": age} for subject, age in AGES.items()}
        for experiment in plotter.EXPERIMENTS
    }
    predictions["dortmund_zero_shot"]["SUB1"]["true_age"] = 30.0
    with pytest.raises(plotter.ComparisonValidationError, match="True age mismatch"):
        plotter.combine_experiments(predictions)


def test_metrics_independent_recomputation():
    truth = [22.0, 24.0, 28.0]
    prediction = [23.0, 23.0, 30.0]
    metrics = plotter.regression_metrics(truth, prediction)
    assert metrics["mae"] == pytest.approx(4 / 3)
    assert metrics["rmse"] == pytest.approx((6 / 3) ** 0.5)
    assert metrics["bias"] == pytest.approx(2 / 3)


def test_find_subject_csv_accepts_file_and_priority(tmp_path):
    direct = tmp_path / "direct.csv"; write_csv(direct)
    assert plotter.find_subject_prediction_csv(direct, "original_cv7") == direct.resolve()
    preferred = tmp_path / "oof_subject_predictions.csv"; write_csv(preferred)
    write_csv(tmp_path / "subject_predictions.csv")
    assert plotter.find_subject_prediction_csv(tmp_path, "original_cv7") == preferred.resolve()


def test_agg_backend_and_end_to_end_outputs(tmp_path, monkeypatch):
    assert matplotlib.get_backend().lower() == "agg"
    original = tmp_path / "original"; zero = tmp_path / "zero"; transfer = tmp_path / "transfer"
    predictions = {
        "original_cv7": [30.0, 28.0, 27.0, 26.5, 27.0, 28.5, 23.0],
        "dortmund_zero_shot": [31.5, 30.0, 29.0, 28.0, 29.0, 30.0, 27.0],
        "dortmund_probe_transfer": [30.5, 29.2, 26.5, 26.2, 27.8, 28.3, 23.5],
    }
    write_csv(original / "oof_subject_predictions.csv", rows=[
        {"subject_id": subject, "true_age": AGES[subject], "mean_predicted_age": value}
        for subject, value in zip(AGES, predictions["original_cv7"])
    ])
    write_csv(zero / "subject_predictions_labeled.csv", rows=[
        {"subject_id": subject, "true_age": AGES[subject], "mean_predicted_age": value}
        for subject, value in zip(AGES, predictions["dortmund_zero_shot"])
    ])
    write_csv(transfer / "oof_subject_predictions.csv", rows=[
        {"subject_id": subject, "true_age": AGES[subject], "mean_prediction": value}
        for subject, value in zip(AGES, predictions["dortmund_probe_transfer"])
    ])
    monkeypatch.setattr(plotter, "choose_font", lambda language: ("DejaVu Sans", "en", language == "zh"))
    fixture_expected = {}
    truth = list(AGES.values())
    for experiment, values in predictions.items():
        measured = plotter.regression_metrics(truth, values)
        fixture_expected[experiment] = {"mae": measured["mae"], "rmse": measured["rmse"]}
    monkeypatch.setattr(plotter, "EXPECTED_METRICS", fixture_expected)
    output = tmp_path / "output"
    args = plotter.parse_args([
        "--original-cv7-dir", str(original), "--dortmund-zero-shot-dir", str(zero),
        "--probe-transfer-dir", str(transfer), "--output-dir", str(output),
        "--language", "en", "--dpi", "80",
    ])
    result = plotter.run(args)
    assert result["subjects"] == 7 and result["experiments"] == 3
    for name in (
        "age_experiment_subject_comparison.csv", "age_experiment_metrics.csv",
        "age_experiment_comparison_summary.json", "age_predictions_by_subject.png",
        "age_predictions_by_subject.pdf", "age_predicted_vs_true_scatter.png",
        "age_predicted_vs_true_scatter.pdf", "plot_provenance.json", "plot_issues.csv",
    ):
        assert (output / name).is_file() and (output / name).stat().st_size > 0
    summary = json.loads((output / "age_experiment_comparison_summary.json").read_text(encoding="utf-8"))
    assert summary["all_true_ages_equal_across_experiments"] is True
    assert summary["metrics_match_formal_results"] is True
