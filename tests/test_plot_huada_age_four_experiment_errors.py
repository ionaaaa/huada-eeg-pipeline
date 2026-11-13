from __future__ import annotations

import csv
import io

import matplotlib
import pytest

from scripts.plot_huada_age_four_experiment_errors import (
    ComparisonError,
    EXPERIMENTS,
    SUBJECTS,
    metric_bundle,
    parse_subject_csv,
    plot_errors,
)


def fixture_csv(prediction_field="mean_prediction", delta=1.0):
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["subject_id", "true_age", prediction_field])
    writer.writeheader()
    for index, subject in enumerate(SUBJECTS):
        age = 22.0 + index
        writer.writerow({"subject_id": subject, "true_age": age, prediction_field: age + delta})
    writer.writerow({"subject_id": "SUB5", "true_age": "", prediction_field: 27.0})
    return output.getvalue().encode("utf-8")


def test_subject_parsing_sub5_exclusion_and_metrics():
    rows = parse_subject_csv(fixture_csv(delta=2.0), "mean_prediction", "fixture")
    assert tuple(rows) == SUBJECTS
    metrics = metric_bundle(rows)
    assert metrics["subject_count"] == 7
    assert metrics["mae"] == pytest.approx(2.0)
    assert metrics["rmse"] == pytest.approx(2.0)
    assert metrics["bias"] == pytest.approx(2.0)


def test_duplicate_missing_and_nonfinite_fail_closed():
    duplicate = fixture_csv().decode() + "SUB1,31,32\n"
    with pytest.raises(ComparisonError, match="Duplicate"):
        parse_subject_csv(duplicate.encode(), "mean_prediction", "fixture")
    missing = fixture_csv().decode().replace("SUB8,28.0,29.0\r\n", "")
    with pytest.raises(ComparisonError, match="Subject set mismatch"):
        parse_subject_csv(missing.encode(), "mean_prediction", "fixture")
    nonfinite = fixture_csv().decode().replace("SUB1,22.0,23.0", "SUB1,22.0,nan")
    with pytest.raises(ComparisonError, match="Non-finite"):
        parse_subject_csv(nonfinite.encode(), "mean_prediction", "fixture")


def test_agg_plot_outputs_are_nonempty(tmp_path):
    datasets = {}
    metrics = {}
    for index, experiment in enumerate(EXPERIMENTS):
        rows = parse_subject_csv(
            fixture_csv(experiment["prediction_field"], delta=float(index + 1)),
            experiment["prediction_field"],
            experiment["key"],
        )
        datasets[experiment["key"]] = rows
        metrics[experiment["key"]] = metric_bundle(rows)
    plot_errors(datasets, metrics, tmp_path, 100, "DejaVu Sans")
    assert matplotlib.get_backend().lower() == "agg"
    for name in ("age_prediction_error_scatter.png", "age_prediction_error_scatter.pdf"):
        path = tmp_path / name
        assert path.is_file() and path.stat().st_size > 0
