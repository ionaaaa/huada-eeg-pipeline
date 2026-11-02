"""Sex-stratified four-fold subject CV and OOF aggregation utilities."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


PROTOCOL = "sex_stratified_subject_cv4"
LABEL_TO_SEX = {0: "female", 1: "male"}
SEX_TO_LABEL = {"female": 0, "male": 1}


def _dump_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _write_csv(path, rows, fields):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_sex_stratified_subject_cv4(subject_labels, seed):
    """Build four deterministic balanced subject pairs and outer folds."""
    normalized = {str(subject): int(label) for subject, label in subject_labels.items()}
    females = sorted(subject for subject, label in normalized.items() if label == 0)
    males = sorted(subject for subject, label in normalized.items() if label == 1)
    unknown = sorted(subject for subject, label in normalized.items() if label not in (0, 1))
    if unknown:
        raise ValueError(f"Sex labels must be 0/1; invalid subjects: {unknown}")
    if len(females) != 4 or len(males) != 4 or len(normalized) != 8:
        raise ValueError(
            "sex_stratified_subject_cv4 requires exactly 4 female and 4 male subjects; "
            f"got female={len(females)}, male={len(males)}, total={len(normalized)}"
        )

    female_seed, male_seed = np.random.SeedSequence(int(seed)).spawn(2)
    female_rng = np.random.default_rng(female_seed)
    male_rng = np.random.default_rng(male_seed)
    females = [females[index] for index in female_rng.permutation(len(females))]
    males = [males[index] for index in male_rng.permutation(len(males))]
    pairs = [
        {
            "pair": pair_index,
            "subjects": [females[pair_index], males[pair_index]],
            "female": females[pair_index],
            "male": males[pair_index],
        }
        for pair_index in range(4)
    ]

    folds = []
    for fold in range(4):
        test_pair = fold
        validation_pair = (fold + 1) % 4
        train_pairs = [index for index in range(4) if index not in (test_pair, validation_pair)]
        train_subjects = sorted(
            subject for pair_index in train_pairs for subject in pairs[pair_index]["subjects"]
        )
        validation_subjects = sorted(pairs[validation_pair]["subjects"])
        test_subjects = sorted(pairs[test_pair]["subjects"])
        folds.append({
            "fold": fold,
            "train_pairs": train_pairs,
            "validation_pair": validation_pair,
            "test_pair": test_pair,
            "train_subjects": train_subjects,
            "validation_subjects": validation_subjects,
            "test_subjects": test_subjects,
        })

    payload = {
        "protocol": PROTOCOL,
        "fold_count": 4,
        "split_seed": int(seed),
        "pairs": pairs,
        "folds": folds,
        "subject_labels": normalized,
    }
    validate_cv4_split_payload(payload)
    return payload


def validate_cv4_split_payload(payload):
    if payload.get("protocol") != PROTOCOL or len(payload.get("folds", [])) != 4:
        raise ValueError("Expected sex_stratified_subject_cv4 with exactly four folds")
    subject_labels = {str(key): int(value) for key, value in payload["subject_labels"].items()}
    test_counts = Counter()
    for fold_info in payload["folds"]:
        train = set(fold_info["train_subjects"])
        validation = set(fold_info["validation_subjects"])
        test = set(fold_info["test_subjects"])
        if len(train) != 4 or len(validation) != 2 or len(test) != 2:
            raise ValueError(f"Invalid fold sizes: {fold_info}")
        if train & validation or train & test or validation & test:
            raise ValueError(f"Subject overlap in fold {fold_info['fold']}")
        if train | validation | test != set(subject_labels):
            raise ValueError(f"Fold {fold_info['fold']} does not cover all subjects")
        for split_name, subjects, expected in (
            ("train", train, {0: 2, 1: 2}),
            ("validation", validation, {0: 1, 1: 1}),
            ("test", test, {0: 1, 1: 1}),
        ):
            counts = Counter(subject_labels[subject] for subject in subjects)
            if dict(counts) != expected:
                raise ValueError(
                    f"Fold {fold_info['fold']} {split_name} is not sex-balanced: {dict(counts)}"
                )
        test_counts.update(test)
    if set(test_counts) != set(subject_labels) or any(count != 1 for count in test_counts.values()):
        raise ValueError(f"Every subject must be test exactly once: {dict(test_counts)}")
    return True


def split_rows(payload):
    labels = payload["subject_labels"]
    rows = []
    for fold_info in payload["folds"]:
        split_by_subject = {}
        for split_name, key in (
            ("train", "train_subjects"),
            ("validation", "validation_subjects"),
            ("test", "test_subjects"),
        ):
            for subject in fold_info[key]:
                split_by_subject[subject] = split_name
        for subject in sorted(labels):
            label = int(labels[subject])
            rows.append({
                "fold": fold_info["fold"],
                "subject_id": subject,
                "sex": LABEL_TO_SEX[label],
                "label": label,
                "split": split_by_subject[subject],
                "split_seed": payload["split_seed"],
            })
    return rows


def write_split_artifacts(output_dir, payload):
    output_dir = Path(output_dir)
    _write_csv(
        output_dir / "sex_cv4_subject_splits.csv",
        split_rows(payload),
        ["fold", "subject_id", "sex", "label", "split", "split_seed"],
    )
    _dump_json(output_dir / "sex_cv4_subject_splits.json", payload)


def binary_classification_metrics(true_labels, predicted_labels):
    truth = np.asarray(true_labels, dtype=np.int64).reshape(-1)
    predicted = np.asarray(predicted_labels, dtype=np.int64).reshape(-1)
    if truth.shape != predicted.shape or truth.size == 0:
        raise ValueError("Metric inputs must be non-empty and have equal shape")
    confusion = np.zeros((2, 2), dtype=np.int64)
    for true_label, predicted_label in zip(truth, predicted):
        if true_label not in (0, 1) or predicted_label not in (0, 1):
            raise ValueError("Binary metrics require labels 0/1")
        confusion[true_label, predicted_label] += 1
    recalls = []
    f1_values = []
    for label in range(2):
        tp = int(confusion[label, label])
        fn = int(confusion[label].sum() - tp)
        fp = int(confusion[:, label].sum() - tp)
        recall = tp / (tp + fn) if tp + fn else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        recalls.append(recall)
        f1_values.append(f1)
    return {
        "accuracy": float((truth == predicted).mean()),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1_values)),
        "confusion_matrix": confusion.tolist(),
    }


def validate_oof_window_predictions(rows, eligible_samples, subject_test_folds):
    """Validate exact OOF coverage and probability invariants."""
    expected = {
        str(sample_id): {"subject_id": str(value["subject_id"]), "label": int(value["label"])}
        for sample_id, value in eligible_samples.items()
    }
    sample_ids = [str(row["sample_id"]) for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        duplicates = [sample for sample, count in Counter(sample_ids).items() if count > 1]
        raise ValueError(f"Duplicate OOF sample_id values: {duplicates}")
    missing = sorted(set(expected).difference(sample_ids))
    extra = sorted(set(sample_ids).difference(expected))
    if missing or extra or len(rows) != len(expected):
        raise ValueError(f"OOF coverage mismatch: missing={missing}, extra={extra}")
    seen_subject_folds = defaultdict(set)
    for row in rows:
        sample_id = str(row["sample_id"])
        subject = str(row["subject_id"])
        label = int(row["true_label"])
        fold = int(row["fold"])
        if subject != expected[sample_id]["subject_id"] or label != expected[sample_id]["label"]:
            raise ValueError(f"OOF metadata mismatch for {sample_id}")
        if fold != int(subject_test_folds[subject]):
            raise ValueError(f"OOF fold mismatch for {subject}: {fold}")
        female = float(row["prob_female"])
        male = float(row["prob_male"])
        if not math.isfinite(female) or not math.isfinite(male):
            raise ValueError(f"Non-finite probability for {sample_id}")
        if not math.isclose(female + male, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"Probabilities do not sum to one for {sample_id}")
        if int(row["predicted_label"]) != int(np.argmax([female, male])):
            raise ValueError(f"predicted_label is not probability argmax for {sample_id}")
        seen_subject_folds[subject].add(fold)
    if any(len(folds) != 1 for folds in seen_subject_folds.values()):
        raise ValueError(f"A subject appears in multiple test folds: {dict(seen_subject_folds)}")
    if set(seen_subject_folds) != set(subject_test_folds):
        raise ValueError("OOF subjects do not match CV subjects")
    return True


def aggregate_subject_predictions(window_rows):
    """Aggregate window probabilities and votes with a deterministic tie-break.

    The function is shared by CV/OOF and external zero-shot evaluation.  Fold
    metadata is validated and returned only when it is present in the input.
    Vote ties use probability-mean argmax; an exact probability tie resolves
    to class 0 (female), matching NumPy's deterministic ``argmax`` behavior.
    """
    grouped = defaultdict(list)
    for row in window_rows:
        grouped[str(row["subject_id"])].append(row)
    subject_rows = []
    for subject_id in sorted(grouped):
        rows = grouped[subject_id]
        true_labels = {int(row["true_label"]) for row in rows}
        has_folds = any("fold" in row for row in rows)
        if has_folds and not all("fold" in row for row in rows):
            raise ValueError(f"Partially populated fold metadata for {subject_id}")
        folds = {int(row["fold"]) for row in rows} if has_folds else set()
        if len(true_labels) != 1 or (has_folds and len(folds) != 1):
            raise ValueError(f"Inconsistent subject rows for {subject_id}")
        true_label = next(iter(true_labels))
        mean_female = float(np.mean([float(row["prob_female"]) for row in rows]))
        mean_male = float(np.mean([float(row["prob_male"]) for row in rows]))
        probability_prediction = int(np.argmax([mean_female, mean_male]))
        votes = Counter(int(row["predicted_label"]) for row in rows)
        female_votes = int(votes.get(0, 0))
        male_votes = int(votes.get(1, 0))
        tie_break_used = female_votes == male_votes
        if female_votes > male_votes:
            majority_prediction = 0
        elif male_votes > female_votes:
            majority_prediction = 1
        elif mean_female > mean_male:
            majority_prediction = 0
        elif mean_male > mean_female:
            majority_prediction = 1
        else:
            majority_prediction = 0
        subject_row = {
            "subject_id": subject_id,
            "true_label": true_label,
            "true_sex": LABEL_TO_SEX[true_label],
            "num_windows": len(rows),
            "mean_prob_female": mean_female,
            "mean_prob_male": mean_male,
            "probability_mean_pred_label": probability_prediction,
            "probability_mean_pred_sex": LABEL_TO_SEX[probability_prediction],
            "female_vote_count": female_votes,
            "male_vote_count": male_votes,
            "majority_vote_pred_label": majority_prediction,
            "majority_vote_pred_sex": LABEL_TO_SEX[majority_prediction],
            "tie_break_used": tie_break_used,
        }
        if has_folds:
            subject_row["test_fold"] = next(iter(folds))
        subject_rows.append(subject_row)
    return subject_rows


def write_oof_artifacts(
    output_dir,
    split_payload,
    window_rows,
    per_fold_rows,
    eligible_samples,
    checkpoint_report_by_fold,
    head_input_dim,
    head_parameter_count,
    training_metadata,
):
    output_dir = Path(output_dir)
    subject_test_folds = {
        subject: fold_info["fold"]
        for fold_info in split_payload["folds"]
        for subject in fold_info["test_subjects"]
    }
    validate_oof_window_predictions(window_rows, eligible_samples, subject_test_folds)
    window_rows = sorted(window_rows, key=lambda row: (int(row["fold"]), str(row["sample_id"])))
    window_fields = [
        "fold", "sample_id", "subject_id", "sex", "true_label",
        "prob_female", "prob_male", "predicted_label", "predicted_sex",
        "num_valid_tokens", "checkpoint_path",
    ]
    _write_csv(output_dir / "sex_cv4_oof_window_predictions.csv", window_rows, window_fields)

    window_metrics = binary_classification_metrics(
        [row["true_label"] for row in window_rows],
        [row["predicted_label"] for row in window_rows],
    )
    window_metrics.update({
        "unit": "window",
        "num_windows": len(window_rows),
        "num_subjects": len({row["subject_id"] for row in window_rows}),
        "class_counts": dict(Counter(LABEL_TO_SEX[int(row["true_label"])] for row in window_rows)),
        "fold_count": 4,
    })
    _dump_json(output_dir / "sex_cv4_window_metrics.json", window_metrics)

    per_fold_fields = [
        "fold", "fold_seed", "num_windows", "num_subjects", "loss", "accuracy",
        "balanced_accuracy", "macro_f1", "confusion_matrix", "checkpoint_path",
        "checkpoint_sha256", "matched_key_count", "matched_parameter_fraction",
        "pooling", "head_lr", "weight_decay", "optimizer_name",
        "head_input_dim", "head_parameter_count",
    ]
    _write_csv(output_dir / "sex_cv4_per_fold_metrics.csv", per_fold_rows, per_fold_fields)

    subject_rows = aggregate_subject_predictions(window_rows)
    subject_fields = [
        "subject_id", "true_label", "true_sex", "num_windows",
        "mean_prob_female", "mean_prob_male", "probability_mean_pred_label",
        "probability_mean_pred_sex", "female_vote_count", "male_vote_count",
        "majority_vote_pred_label", "majority_vote_pred_sex", "tie_break_used", "test_fold",
    ]
    _write_csv(output_dir / "sex_cv4_oof_subject_predictions.csv", subject_rows, subject_fields)
    probability_metrics = binary_classification_metrics(
        [row["true_label"] for row in subject_rows],
        [row["probability_mean_pred_label"] for row in subject_rows],
    )
    majority_metrics = binary_classification_metrics(
        [row["true_label"] for row in subject_rows],
        [row["majority_vote_pred_label"] for row in subject_rows],
    )
    subject_metrics = {
        "unit": "subject",
        "num_subjects": len(subject_rows),
        "class_counts": dict(Counter(row["true_sex"] for row in subject_rows)),
        "probability_mean": probability_metrics,
        "majority_vote": majority_metrics,
    }
    _dump_json(output_dir / "sex_cv4_subject_metrics.json", subject_metrics)

    summary = {
        "protocol": PROTOCOL,
        "fold_count": 4,
        "split_seed": split_payload["split_seed"],
        "num_subjects": len(subject_rows),
        "num_windows": len(window_rows),
        "pooling": training_metadata["pooling"],
        "head_lr": float(training_metadata["head_lr"]),
        "weight_decay": float(training_metadata["weight_decay"]),
        "optimizer_name": training_metadata["optimizer_name"],
        "head_input_dim": int(head_input_dim),
        "head_parameter_count": int(head_parameter_count),
        "num_tokens": int(training_metadata["num_tokens"]),
        "d_model": int(training_metadata["d_model"]),
        "window_metrics": window_metrics,
        "subject_metrics": subject_metrics,
        "checkpoint_reports": checkpoint_report_by_fold,
        "primary_result_unit": "subject",
        "pilot_warning": "Eight subjects is an extremely small pilot and does not establish stable population generalization.",
    }
    _dump_json(output_dir / "sex_cv4_summary.json", summary)
    report = [
        "# 50M 性别分层被试级 4 折 OOF 报告", "",
        f"- protocol: `{PROTOCOL}`",
        f"- folds: 4",
        f"- subjects: {len(subject_rows)}",
        f"- windows: {len(window_rows)}",
        f"- pooling: {training_metadata['pooling']}",
        f"- head_lr: {training_metadata['head_lr']}",
        f"- weight_decay: {training_metadata['weight_decay']}",
        f"- optimizer: {training_metadata['optimizer_name']}",
        f"- head: {head_input_dim} -> 2 ({head_parameter_count} parameters)", "",
        "Window-level 指标以窗口为统计单位；subject-level 指标以 8 名被试为统计单位。",
        "本任务以 subject-level OOF 结果为更重要结果。8 名被试仍是极小样本 pilot，",
        "不能据此声称稳定的总体泛化性能。", "",
        f"- window accuracy: {window_metrics['accuracy']:.6f}",
        f"- subject probability-mean accuracy: {probability_metrics['accuracy']:.6f}",
        f"- subject majority-vote accuracy: {majority_metrics['accuracy']:.6f}",
    ]
    (output_dir / "sex_cv4_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary
