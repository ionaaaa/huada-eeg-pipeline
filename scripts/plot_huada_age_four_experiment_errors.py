#!/usr/bin/env python3
"""Compare four Huada age experiments from remote subject-level predictions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from scipy import stats


SUBJECTS = ("SUB1", "SUB2", "SUB3", "SUB4", "SUB6", "SUB7", "SUB8")
EXPERIMENTS = (
    {
        "key": "old_zero_shot",
        "label": "Old checkpoint: strict zero-shot",
        "argument": "old_zero_shot_dir",
        "filename": "subject_predictions_labeled.csv",
        "prediction_field": "mean_predicted_age",
        "color": "#E68613",
        "marker": "^",
    },
    {
        "key": "new_zero_shot",
        "label": "New checkpoint: strict zero-shot",
        "argument": "new_zero_shot_dir",
        "filename": "subject_predictions_labeled.csv",
        "prediction_field": "mean_predicted_age",
        "color": "#D1495B",
        "marker": "D",
    },
    {
        "key": "old_probe_transfer",
        "label": "Old checkpoint: probe transfer CV7",
        "argument": "old_probe_transfer_dir",
        "filename": "oof_subject_predictions.csv",
        "prediction_field": "mean_prediction",
        "color": "#2A9D6F",
        "marker": "s",
    },
    {
        "key": "new_probe_transfer",
        "label": "New checkpoint: probe transfer CV7",
        "argument": "new_probe_transfer_dir",
        "filename": "oof_subject_predictions.csv",
        "prediction_field": "mean_prediction",
        "color": "#7551A8",
        "marker": "X",
    },
)


class ComparisonError(RuntimeError):
    pass


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssh-host", default="tione")
    parser.add_argument("--old-zero-shot-dir", required=True)
    parser.add_argument("--new-zero-shot-dir", required=True)
    parser.add_argument("--old-probe-transfer-dir", required=True)
    parser.add_argument("--new-probe-transfer-dir", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args(argv)
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    return args


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def remote_file(host: str, path: str) -> bytes:
    result = subprocess.run(
        ["ssh", host, "cat", path],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise ComparisonError(f"Failed to read {host}:{path}: {detail}")
    return result.stdout


def parse_subject_csv(content: bytes, prediction_field: str, source: str) -> dict[str, dict[str, float]]:
    text = content.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    required = {"subject_id", "true_age", prediction_field}
    missing = sorted(required.difference(reader.fieldnames or ()))
    if missing:
        raise ComparisonError(f"{source} missing fields: {missing}")
    rows: dict[str, dict[str, float]] = {}
    for raw in reader:
        subject = str(raw["subject_id"]).strip().upper()
        if subject == "SUB5":
            continue
        if subject in rows:
            raise ComparisonError(f"Duplicate subject {subject} in {source}")
        true_age = float(raw["true_age"])
        prediction = float(raw[prediction_field])
        if not math.isfinite(true_age) or not math.isfinite(prediction):
            raise ComparisonError(f"Non-finite age/prediction for {subject} in {source}")
        rows[subject] = {
            "true_age": true_age,
            "prediction": prediction,
            "error": prediction - true_age,
        }
    if set(rows) != set(SUBJECTS):
        raise ComparisonError(
            f"Subject set mismatch in {source}: expected={list(SUBJECTS)} actual={sorted(rows)}"
        )
    return rows


def metric_bundle(rows: Mapping[str, Mapping[str, float]]) -> dict[str, float | int]:
    truth = np.asarray([rows[subject]["true_age"] for subject in SUBJECTS], dtype=np.float64)
    prediction = np.asarray([rows[subject]["prediction"] for subject in SUBJECTS], dtype=np.float64)
    residual = prediction - truth
    pearson = stats.pearsonr(truth, prediction).statistic
    spearman = stats.spearmanr(truth, prediction).statistic
    return {
        "subject_count": int(truth.size),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "bias": float(np.mean(residual)),
        "r2": float(1.0 - np.sum(np.square(residual)) / np.sum(np.square(truth - truth.mean()))),
        "pearson_r": float(pearson),
        "spearman_rho": float(spearman),
    }


def select_font() -> str:
    available = {item.name for item in font_manager.fontManager.ttflist}
    for candidate in ("Microsoft YaHei", "Noto Sans CJK SC", "Source Han Sans SC", "Arial", "DejaVu Sans"):
        if candidate in available:
            return candidate
    return "DejaVu Sans"


def plot_errors(
    datasets: Mapping[str, Mapping[str, Mapping[str, float]]],
    metrics: Mapping[str, Mapping[str, Any]],
    output_dir: Path,
    dpi: int,
    font_family: str,
) -> None:
    plt.rcParams.update({
        "font.family": font_family,
        "font.size": 10,
        "axes.titlesize": 15,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })
    fig, ax = plt.subplots(figsize=(12.4, 6.7), constrained_layout=False)
    x = np.arange(len(SUBJECTS), dtype=float)
    offsets = np.linspace(-0.24, 0.24, len(EXPERIMENTS))
    for offset, experiment in zip(offsets, EXPERIMENTS):
        key = experiment["key"]
        errors = np.asarray([datasets[key][subject]["error"] for subject in SUBJECTS])
        positions = x + offset
        ax.vlines(positions, 0.0, errors, color=experiment["color"], alpha=0.30, linewidth=1.2, zorder=1)
        label = (
            f"{experiment['label']}  "
            f"(MAE={metrics[key]['mae']:.3f}, RMSE={metrics[key]['rmse']:.3f})"
        )
        ax.scatter(
            positions,
            errors,
            s=78,
            marker=experiment["marker"],
            color=experiment["color"],
            edgecolor="white",
            linewidth=0.8,
            label=label,
            zorder=3,
        )
    ax.axhline(0.0, color="#4F4F4F", linestyle="--", linewidth=1.2, alpha=0.85, label="No error")
    ax.set_xticks(x, SUBJECTS)
    ax.set_xlabel("Subject ID")
    ax.set_ylabel("Prediction - ground truth (years)")
    ax.set_title("Huada Age Prediction Error: Old vs. New Dortmund Checkpoints", pad=14)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.75)
    ax.grid(axis="x", visible=False)
    ax.spines[["top", "right"]].set_visible(False)
    all_errors = np.concatenate([
        np.asarray([datasets[experiment["key"]][subject]["error"] for subject in SUBJECTS])
        for experiment in EXPERIMENTS
    ])
    margin = max(1.0, float(np.ptp(all_errors)) * 0.10)
    ax.set_ylim(min(float(all_errors.min()) - margin, -1.0), max(float(all_errors.max()) + margin, 1.0))
    ax.legend(loc="upper left", frameon=False, ncol=2, bbox_to_anchor=(0.0, -0.16))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.91, bottom=0.27)
    fig.savefig(output_dir / "age_prediction_error_scatter.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / "age_prediction_error_scatter.pdf", bbox_inches="tight")
    plt.close(fig)


def git_state(root: Path) -> dict[str, Any]:
    def run(*parts: str) -> str:
        result = subprocess.run(["git", *parts], cwd=root, capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else ""
    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD") or None,
        "dirty": bool(status),
        "dirty_files": status.splitlines(),
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    datasets: dict[str, dict[str, dict[str, float]]] = {}
    input_paths: dict[str, str] = {}
    input_sha256: dict[str, str] = {}
    for experiment in EXPERIMENTS:
        remote_dir = str(getattr(args, experiment["argument"])).rstrip("/")
        remote_path = f"{remote_dir}/{experiment['filename']}"
        content = remote_file(args.ssh_host, remote_path)
        datasets[experiment["key"]] = parse_subject_csv(
            content, experiment["prediction_field"], f"{args.ssh_host}:{remote_path}"
        )
        input_paths[experiment["key"]] = remote_path
        input_sha256[experiment["key"]] = sha256_bytes(content)

    reference = datasets[EXPERIMENTS[0]["key"]]
    for experiment in EXPERIMENTS[1:]:
        current = datasets[experiment["key"]]
        for subject in SUBJECTS:
            if current[subject]["true_age"] != reference[subject]["true_age"]:
                raise ComparisonError(f"True age mismatch for {subject}: {experiment['key']}")

    metrics = {experiment["key"]: metric_bundle(datasets[experiment["key"]]) for experiment in EXPERIMENTS}
    comparison_rows = []
    for subject in SUBJECTS:
        row: dict[str, Any] = {"subject_id": subject, "true_age": reference[subject]["true_age"]}
        for experiment in EXPERIMENTS:
            key = experiment["key"]
            row[f"prediction_{key}"] = datasets[key][subject]["prediction"]
            row[f"error_{key}"] = datasets[key][subject]["error"]
        comparison_rows.append(row)
    comparison_fields = ["subject_id", "true_age"]
    for experiment in EXPERIMENTS:
        comparison_fields.extend([f"prediction_{experiment['key']}", f"error_{experiment['key']}"])
    write_csv(output_dir / "age_prediction_error_comparison.csv", comparison_rows, comparison_fields)

    metric_rows = []
    for experiment in EXPERIMENTS:
        key = experiment["key"]
        metric_rows.append({"experiment": key, "label": experiment["label"], **metrics[key]})
    metric_fields = ["experiment", "label", "subject_count", "mae", "rmse", "bias", "r2", "pearson_r", "spearman_rho"]
    write_csv(output_dir / "age_prediction_error_metrics.csv", metric_rows, metric_fields)

    font_family = select_font()
    plot_errors(datasets, metrics, output_dir, args.dpi, font_family)
    summary = {
        "status": "passed",
        "subjects": len(SUBJECTS),
        "subject_ids": list(SUBJECTS),
        "experiments": len(EXPERIMENTS),
        "all_true_ages_equal": True,
        "sub5_excluded": True,
        "error_definition": "subject_mean_prediction_minus_ground_truth_years",
        "metrics": metrics,
        "best_mae_experiment": min(metrics, key=lambda key: metrics[key]["mae"]),
        "best_rmse_experiment": min(metrics, key=lambda key: metrics[key]["rmse"]),
        "font_family": font_family,
    }
    write_json(output_dir / "age_prediction_error_summary.json", summary)
    write_csv(output_dir / "plot_issues.csv", [], ["severity", "issue", "detail"])

    project_root = Path(__file__).resolve().parents[1]
    output_hashes = {
        path.name: sha256_file(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "plot_provenance.json"
    }
    provenance = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ssh_host": args.ssh_host,
        "input_csv_paths": input_paths,
        "input_csv_sha256": input_sha256,
        "plot_script": str(Path(__file__).resolve()),
        "plot_script_sha256": sha256_file(__file__),
        "git": git_state(project_root),
        "matplotlib_version": matplotlib.__version__,
        "font_family": font_family,
        "dpi": args.dpi,
        "output_sha256": output_hashes,
        "command": [str(Path(__file__).resolve()), *[str(item) for item in vars(args).values()]],
    }
    write_json(output_dir / "plot_provenance.json", provenance)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run(args)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), **summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
