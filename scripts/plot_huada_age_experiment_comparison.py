#!/usr/bin/env python3
"""Compare three formal Huada age experiments from subject-level prediction CSVs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


EXPECTED_SUBJECTS = ("SUB1", "SUB2", "SUB3", "SUB4", "SUB6", "SUB7", "SUB8")
EXPERIMENTS = (
    "original_cv7",
    "dortmund_zero_shot",
    "dortmund_probe_transfer",
)
EXPECTED_METRICS = {
    "original_cv7": {"mae": 3.0339230112015665, "rmse": 3.5329356294661673},
    "dortmund_zero_shot": {"mae": 3.2228898805457704, "rmse": 3.9541168070989103},
    "dortmund_probe_transfer": {"mae": 2.343074738711675, "rmse": 2.754259918027419},
}
CSV_PRIORITY = {
    "original_cv7": ("oof_subject_predictions.csv", "subject_predictions.csv"),
    "dortmund_zero_shot": (
        "subject_predictions_labeled.csv", "subject_predictions.csv",
        "subject_predictions_all.csv",
    ),
    "dortmund_probe_transfer": ("oof_subject_predictions.csv", "subject_predictions.csv"),
}
SUBJECT_ALIASES = ("subject_id", "subject", "subj_id", "participant_id")
TRUE_AGE_ALIASES = ("true_age", "true_age_years", "age", "label")
PREDICTION_ALIASES = (
    "mean_prediction", "mean_predicted_age", "mean_predicted_age_years",
    "predicted_age", "prediction", "prediction_age",
)
COLORS = {
    "original_cv7": "#2F6BBA",
    "dortmund_zero_shot": "#E6862B",
    "dortmund_probe_transfer": "#2E8B57",
}
MARKERS = {
    "original_cv7": "o",
    "dortmund_zero_shot": "^",
    "dortmund_probe_transfer": "s",
}
ENGLISH_NAMES = {
    "original_cv7": "Original CV7",
    "dortmund_zero_shot": "Dortmund zero-shot",
    "dortmund_probe_transfer": "Dortmund probe transfer",
}


class ComparisonValidationError(RuntimeError):
    pass


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-cv7-dir", required=True)
    parser.add_argument("--dortmund-zero-shot-dir", required=True)
    parser.add_argument("--probe-transfer-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--language", choices=("zh", "en"), default="zh")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args(argv)
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    return args


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def find_subject_prediction_csv(path: str | Path, experiment: str) -> Path:
    candidate = Path(path).resolve()
    if candidate.is_file():
        if candidate.suffix.lower() != ".csv":
            raise ComparisonValidationError(f"Explicit prediction file is not CSV: {candidate}")
        return candidate
    if not candidate.is_dir():
        raise FileNotFoundError(candidate)
    for name in CSV_PRIORITY[experiment]:
        exact = candidate / name
        if exact.is_file():
            return exact
    available = sorted(
        item for item in candidate.glob("*subject*prediction*.csv") if item.is_file()
    )
    if len(available) == 1:
        return available[0]
    if not available:
        raise FileNotFoundError(f"No subject-level prediction CSV under {candidate}")
    raise ComparisonValidationError(
        f"Ambiguous subject-level prediction CSVs under {candidate}: {available}"
    )


def _resolve_field(fieldnames: Sequence[str] | None, aliases: Sequence[str], role: str) -> str:
    names = [] if fieldnames is None else list(fieldnames)
    lookup = {str(name).strip().lower(): str(name) for name in names}
    for alias in aliases:
        if alias in lookup:
            return lookup[alias]
    raise ComparisonValidationError(f"Missing {role} field; available={names}")


def _subject_number(subject_id: str) -> int:
    match = re.fullmatch(r"SUB(\d+)", subject_id.upper())
    if not match:
        raise ComparisonValidationError(f"Invalid subject ID: {subject_id!r}")
    return int(match.group(1))


def read_subject_predictions(path: str | Path) -> dict[str, dict[str, float]]:
    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        subject_field = _resolve_field(reader.fieldnames, SUBJECT_ALIASES, "subject ID")
        age_field = _resolve_field(reader.fieldnames, TRUE_AGE_ALIASES, "true age")
        prediction_field = _resolve_field(reader.fieldnames, PREDICTION_ALIASES, "subject prediction")
        output: dict[str, dict[str, float]] = {}
        for line_number, row in enumerate(reader, start=2):
            subject = str(row.get(subject_field, "")).strip().upper()
            if subject == "SUB5":
                continue
            if "included_in_metrics" in row and str(row["included_in_metrics"]).strip().lower() in {"false", "0", "no"}:
                continue
            if not subject:
                raise ComparisonValidationError(f"Empty subject ID at {csv_path}:{line_number}")
            _subject_number(subject)
            if subject in output:
                raise ComparisonValidationError(f"Duplicate subject {subject} in {csv_path}")
            try:
                true_age = float(row[age_field])
                prediction = float(row[prediction_field])
            except (TypeError, ValueError) as exc:
                raise ComparisonValidationError(
                    f"Invalid numeric value for {subject} in {csv_path}"
                ) from exc
            if not math.isfinite(true_age) or not math.isfinite(prediction):
                raise ComparisonValidationError(f"Non-finite value for {subject} in {csv_path}")
            output[subject] = {"true_age": true_age, "prediction": prediction}
    expected = set(EXPECTED_SUBJECTS)
    if set(output) != expected:
        raise ComparisonValidationError(
            f"Subject set mismatch in {csv_path}; missing={sorted(expected-set(output))}, "
            f"extra={sorted(set(output)-expected)}"
        )
    return dict(sorted(output.items(), key=lambda item: _subject_number(item[0])))


def regression_metrics(truth: Sequence[float], prediction: Sequence[float]) -> dict[str, Any]:
    y = np.asarray(truth, dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    if y.shape != p.shape or y.size == 0 or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ComparisonValidationError("Metric inputs must be finite, non-empty and equal-shaped")
    error = p - y
    denominator = float(np.sum((y - y.mean()) ** 2))
    if y.size < 2 or denominator <= 0:
        raise ComparisonValidationError("Subject metrics require at least two non-constant true ages")
    pearson = float(stats.pearsonr(y, p).statistic) if float(p.std()) > 0 else None
    spearman = float(stats.spearmanr(y, p).statistic) if float(p.std()) > 0 else None
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "r2": float(1.0 - np.sum(error ** 2) / denominator),
        "bias": float(np.mean(error)),
        "pearson_r": pearson,
        "spearman_rho": spearman,
        "subject_count": int(y.size),
    }


def combine_experiments(
    predictions: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    for experiment in EXPERIMENTS:
        if set(predictions[experiment]) != set(EXPECTED_SUBJECTS):
            raise ComparisonValidationError(f"Incomplete subject set for {experiment}")
    rows = []
    for subject in EXPECTED_SUBJECTS:
        ages = [float(predictions[experiment][subject]["true_age"]) for experiment in EXPERIMENTS]
        if max(ages) - min(ages) > 1e-12:
            raise ComparisonValidationError(f"True age mismatch for {subject}: {ages}")
        rows.append({
            "subject_id": subject,
            "true_age": ages[0],
            "prediction_original_cv7": float(predictions["original_cv7"][subject]["prediction"]),
            "prediction_dortmund_zero_shot": float(predictions["dortmund_zero_shot"][subject]["prediction"]),
            "prediction_dortmund_probe_transfer": float(predictions["dortmund_probe_transfer"][subject]["prediction"]),
        })
    truth = [row["true_age"] for row in rows]
    metrics = {
        experiment: regression_metrics(
            truth,
            [row[{
                "original_cv7": "prediction_original_cv7",
                "dortmund_zero_shot": "prediction_dortmund_zero_shot",
                "dortmund_probe_transfer": "prediction_dortmund_probe_transfer",
            }[experiment]] for row in rows],
        )
        for experiment in EXPERIMENTS
    }
    for experiment, expected in EXPECTED_METRICS.items():
        for key in ("mae", "rmse"):
            difference = abs(metrics[experiment][key] - expected[key])
            if difference > 5e-4:
                raise ComparisonValidationError(
                    f"{experiment} {key} mismatch: recomputed={metrics[experiment][key]}, "
                    f"expected={expected[key]}, abs_diff={difference}"
                )
    return rows, metrics


def choose_font(language: str) -> tuple[str, str, bool]:
    if language == "en":
        return "DejaVu Sans", "en", False
    for family in (
        "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC",
    ):
        try:
            path = font_manager.findfont(
                font_manager.FontProperties(family=[family]), fallback_to_default=False,
            )
        except ValueError:
            continue
        if Path(path).is_file():
            return family, "zh", False
    return "DejaVu Sans", "en", True


def _style(font_family: str) -> None:
    plt.rcParams.update({
        "font.family": font_family,
        "axes.unicode_minus": False,
        "font.size": 10.5,
        "axes.titlesize": 15,
        "axes.labelsize": 11.5,
        "legend.fontsize": 9.5,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _legend_label(experiment: str, metrics: Mapping[str, Mapping[str, Any]]) -> str:
    name = ENGLISH_NAMES[experiment]
    item = metrics[experiment]
    return f"{name} (MAE={item['mae']:.3f}, RMSE={item['rmse']:.3f})"


def plot_by_subject(
    rows: Sequence[Mapping[str, Any]], metrics: Mapping[str, Mapping[str, Any]],
    output_dir: Path, dpi: int, font_family: str, language: str,
) -> None:
    _style(font_family)
    x = np.arange(len(rows), dtype=float)
    truth = np.asarray([row["true_age"] for row in rows], dtype=float)
    offsets = {"true": -0.27, "original_cv7": -0.09, "dortmund_zero_shot": 0.09, "dortmund_probe_transfer": 0.27}
    fields = {
        "original_cv7": "prediction_original_cv7",
        "dortmund_zero_shot": "prediction_dortmund_zero_shot",
        "dortmund_probe_transfer": "prediction_dortmund_probe_transfer",
    }
    fig, ax = plt.subplots(figsize=(12.0, 6.5), constrained_layout=False)
    ax.scatter(
        x + offsets["true"], truth, color="#252525", marker="*", s=150,
        edgecolors="white", linewidths=0.7, zorder=5, label="True age",
    )
    for experiment in EXPERIMENTS:
        prediction = np.asarray([row[fields[experiment]] for row in rows], dtype=float)
        xpos = x + offsets[experiment]
        ax.vlines(
            xpos, truth, prediction, colors=COLORS[experiment], linewidth=1.2,
            alpha=0.35, zorder=1,
        )
        ax.scatter(
            xpos, prediction, color=COLORS[experiment], marker=MARKERS[experiment],
            s=66, edgecolors="white", linewidths=0.7, zorder=4,
            label=_legend_label(experiment, metrics),
        )
    ax.axhline(
        27.5, color="#9E9E9E", linestyle="-.", linewidth=1.1, alpha=0.8,
        label="Fixed 27.5-year baseline",
    )
    values = np.concatenate([
        truth,
        *[np.asarray([row[fields[e]] for row in rows], dtype=float) for e in EXPERIMENTS],
        np.asarray([27.5]),
    ])
    margin = max(1.0, float(np.ptp(values)) * 0.08)
    ax.set_ylim(float(values.min() - margin), float(values.max() + margin))
    ax.set_xlim(-0.65, len(rows) - 0.35)
    ax.set_xticks(x, [row["subject_id"] for row in rows])
    ax.set_xlabel("Subject ID")
    ax.set_ylabel("Age (years)")
    ax.set_title(
        "华大年龄预测：逐被试真实年龄与模型预测"
        if language == "zh" else "Huada Age Prediction: Subject-level Comparison"
    )
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.65)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.91, bottom=0.24)
    fig.savefig(output_dir / "age_predictions_by_subject.png", dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(output_dir / "age_predictions_by_subject.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _duplicate_age_offsets(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    by_age: dict[float, list[str]] = {}
    for row in rows:
        by_age.setdefault(float(row["true_age"]), []).append(str(row["subject_id"]))
    offsets = {}
    for subjects in by_age.values():
        values = np.linspace(-0.055, 0.055, len(subjects)) if len(subjects) > 1 else [0.0]
        offsets.update(zip(subjects, values))
    return offsets


def plot_scatter(
    rows: Sequence[Mapping[str, Any]], metrics: Mapping[str, Mapping[str, Any]],
    output_dir: Path, dpi: int, font_family: str, language: str,
) -> None:
    _style(font_family)
    fields = {
        "original_cv7": "prediction_original_cv7",
        "dortmund_zero_shot": "prediction_dortmund_zero_shot",
        "dortmund_probe_transfer": "prediction_dortmund_probe_transfer",
    }
    model_jitter = {"original_cv7": -0.085, "dortmund_zero_shot": 0.0, "dortmund_probe_transfer": 0.085}
    subject_jitter = _duplicate_age_offsets(rows)
    truth = np.asarray([row["true_age"] for row in rows], dtype=float)
    all_predictions = np.concatenate([
        np.asarray([row[fields[e]] for row in rows], dtype=float) for e in EXPERIMENTS
    ])
    all_values = np.concatenate([truth, all_predictions])
    margin = max(1.0, float(np.ptp(all_values)) * 0.09)
    lower, upper = float(all_values.min() - margin), float(all_values.max() + margin)
    fig, ax = plt.subplots(figsize=(8.2, 7.5), constrained_layout=False)
    ax.plot([lower, upper], [lower, upper], color="#777777", linestyle="--", linewidth=1.25, label="Ideal: y = x", zorder=1)
    label_offsets = {"original_cv7": (-8, -11), "dortmund_zero_shot": (4, 7), "dortmund_probe_transfer": (5, -11)}
    for experiment in EXPERIMENTS:
        prediction = np.asarray([row[fields[experiment]] for row in rows], dtype=float)
        xplot = np.asarray([
            float(row["true_age"]) + model_jitter[experiment] + subject_jitter[str(row["subject_id"])]
            for row in rows
        ])
        ax.scatter(
            xplot, prediction, color=COLORS[experiment], marker=MARKERS[experiment],
            s=70, edgecolors="white", linewidths=0.7, alpha=0.95, zorder=3,
            label=_legend_label(experiment, metrics),
        )
        for row, xp, yp in zip(rows, xplot, prediction):
            ax.annotate(
                str(row["subject_id"]), (xp, yp), xytext=label_offsets[experiment],
                textcoords="offset points", fontsize=7.3, color=COLORS[experiment],
                alpha=0.9,
            )
    ax.set_xlim(lower, upper); ax.set_ylim(lower, upper)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("True age (years)"); ax.set_ylabel("Predicted age (years)")
    ax.set_title(
        "华大年龄预测：预测年龄与真实年龄"
        if language == "zh" else "Huada Age Prediction: Predicted vs. True Age"
    )
    ax.grid(color="#E2E2E2", linewidth=0.7, alpha=0.65)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=1, frameon=False)
    note = (
        "横轴轻微抖动仅用于避免重叠；CSV和指标使用原始年龄。"
        if language == "zh" else
        "Small horizontal jitter is display-only; CSV values and metrics use exact ages."
    )
    fig.text(0.5, 0.025, note, ha="center", va="bottom", fontsize=8.5, color="#555555")
    fig.subplots_adjust(left=0.12, right=0.96, top=0.91, bottom=0.28)
    fig.savefig(output_dir / "age_predicted_vs_true_scatter.png", dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(output_dir / "age_predicted_vs_true_scatter.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def git_state(project_root: Path) -> dict[str, Any]:
    def command(*arguments: str) -> str | None:
        result = subprocess.run(
            ["git", *arguments], cwd=project_root, capture_output=True, text=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    status = command("status", "--porcelain")
    return {
        "commit": command("rev-parse", "HEAD"),
        "dirty": bool(status),
        "dirty_files": [] if not status else status.splitlines(),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {output_dir}")
    inputs = {
        "original_cv7": Path(args.original_cv7_dir).resolve(),
        "dortmund_zero_shot": Path(args.dortmund_zero_shot_dir).resolve(),
        "dortmund_probe_transfer": Path(args.probe_transfer_dir).resolve(),
    }
    csv_paths = {
        experiment: find_subject_prediction_csv(inputs[experiment], experiment)
        for experiment in EXPERIMENTS
    }
    predictions = {
        experiment: read_subject_predictions(csv_paths[experiment])
        for experiment in EXPERIMENTS
    }
    rows, metrics = combine_experiments(predictions)
    font_family, actual_language, font_fallback = choose_font(args.language)
    output_dir.mkdir(parents=True, exist_ok=False)
    atomic_csv(
        output_dir / "age_experiment_subject_comparison.csv", rows,
        ("subject_id", "true_age", "prediction_original_cv7", "prediction_dortmund_zero_shot", "prediction_dortmund_probe_transfer"),
    )
    metric_rows = [{"experiment": experiment, **metrics[experiment]} for experiment in EXPERIMENTS]
    atomic_csv(
        output_dir / "age_experiment_metrics.csv", metric_rows,
        ("experiment", "subject_count", "mae", "rmse", "r2", "bias", "pearson_r", "spearman_rho"),
    )
    atomic_csv(
        output_dir / "plot_issues.csv",
        ([{"severity": "info", "code": "font_fallback", "message": "No supported Chinese font was found; English labels were used."}] if font_fallback else []),
        ("severity", "code", "message"),
    )
    plot_by_subject(rows, metrics, output_dir, args.dpi, font_family, actual_language)
    plot_scatter(rows, metrics, output_dir, args.dpi, font_family, actual_language)
    summary = {
        "status": "passed", "subjects": len(rows), "experiments": len(EXPERIMENTS),
        "subject_ids": list(EXPECTED_SUBJECTS), "sub5_excluded": True,
        "all_true_ages_equal_across_experiments": True,
        "metrics_match_formal_results": True, "metrics": metrics,
        "png_generated": True, "pdf_generated": True,
        "requested_language": args.language, "actual_language": actual_language,
        "font_family": font_family, "font_fallback": font_fallback,
    }
    atomic_json(output_dir / "age_experiment_comparison_summary.json", summary)
    project_root = Path(__file__).resolve().parents[1]
    generated = [
        "age_experiment_subject_comparison.csv", "age_experiment_metrics.csv",
        "age_experiment_comparison_summary.json", "age_predictions_by_subject.png",
        "age_predictions_by_subject.pdf", "age_predicted_vs_true_scatter.png",
        "age_predicted_vs_true_scatter.pdf", "plot_issues.csv",
    ]
    provenance = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_result_directories": {key: str(path) for key, path in inputs.items()},
        "input_csv_paths": {key: str(path) for key, path in csv_paths.items()},
        "input_csv_sha256": {key: sha256_file(path) for key, path in csv_paths.items()},
        "subject_ids": list(EXPECTED_SUBJECTS),
        "plot_script": str(Path(__file__).resolve()),
        "plot_script_sha256": sha256_file(__file__),
        "git": git_state(project_root),
        "matplotlib_version": matplotlib.__version__,
        "requested_language": args.language, "actual_language": actual_language,
        "font_family": font_family, "font_fallback": font_fallback,
        "dpi": int(args.dpi),
        "output_sha256": {name: sha256_file(output_dir / name) for name in generated},
        "command": [sys.executable, *sys.argv],
    }
    atomic_json(output_dir / "plot_provenance.json", provenance)
    return {**summary, "output_dir": str(output_dir), "csv_paths": {k: str(v) for k, v in csv_paths.items()}}


def main(argv: Sequence[str] | None = None) -> int:
    result = run(parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
