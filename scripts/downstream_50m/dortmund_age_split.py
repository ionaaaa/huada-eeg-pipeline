"""Deterministic subject-level split utilities for Dortmund age finetuning."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


SPLITS = ("train", "validation", "test")
ALGORITHM_VERSION = "dortmund-age-sex-margin-stratum-apportion-v2"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SubjectMetadata:
    subject_id: str
    age: float
    sex: str
    conditions: tuple[str, ...] = ()
    recording_ids: tuple[str, ...] = ()
    window_count: int = 0


def validate_ratios(train_ratio: float, validation_ratio: float, test_ratio: float) -> None:
    ratios = (float(train_ratio), float(validation_ratio), float(test_ratio))
    if any(value <= 0.0 for value in ratios):
        raise ValueError("All split ratios must be > 0")
    if not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"Split ratios must sum to 1.0; got {sum(ratios)!r}")


def allocate_split_counts(
    subject_count: int, train_ratio: float, validation_ratio: float, test_ratio: float
) -> dict[str, int]:
    validate_ratios(train_ratio, validation_ratio, test_ratio)
    total = int(subject_count)
    if total < 3:
        raise ValueError("At least three subjects are required")
    validation_count = int(round(total * float(validation_ratio)))
    test_count = int(round(total * float(test_ratio)))
    train_count = total - validation_count - test_count
    counts = {"train": train_count, "validation": validation_count, "test": test_count}
    if any(value <= 0 for value in counts.values()):
        raise ValueError(f"Split ratios produced an empty split: {counts}")
    return counts


def normalize_sex(value: Any) -> str:
    sex = str(value).strip().lower()
    aliases = {"f": "female", "female": "female", "m": "male", "male": "male"}
    if sex not in aliases:
        raise ValueError(f"Unknown or missing sex value: {value!r}")
    return aliases[sex]


def subjects_from_window_rows(rows: Iterable[dict[str, Any]]) -> list[SubjectMetadata]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["subject_id"])].append(row)
    result = []
    for subject_id in sorted(grouped):
        subject_rows = grouped[subject_id]
        ages = {float(row["age"]) for row in subject_rows}
        sexes = {normalize_sex(row["sex"]) for row in subject_rows}
        if len(ages) != 1 or len(sexes) != 1:
            raise ValueError(
                f"Inconsistent age/sex within subject {subject_id}: ages={ages}, sexes={sexes}"
            )
        age = ages.pop()
        if not math.isfinite(age):
            raise ValueError(f"Non-finite age for subject {subject_id}")
        conditions = tuple(sorted({str(row.get("eyes_condition", "")) for row in subject_rows}))
        recordings = tuple(sorted({str(row.get("recording_id", "")) for row in subject_rows}))
        result.append(SubjectMetadata(
            subject_id=subject_id,
            age=age,
            sex=sexes.pop(),
            conditions=conditions,
            recording_ids=recordings,
            window_count=len(subject_rows),
        ))
    return result


def _age_bin(age: float) -> str:
    lower = int(math.floor(float(age) / 2.0) * 2)
    return f"{lower:02d}-{lower + 1:02d}"


def resolve_strata(subjects: Sequence[SubjectMetadata]) -> dict[str, dict[str, str]]:
    exact_counts = Counter((float(row.age), row.sex) for row in subjects)
    bin_sex_counts = Counter((_age_bin(row.age), row.sex) for row in subjects)
    bin_counts = Counter(_age_bin(row.age) for row in subjects)
    result = {}
    for row in subjects:
        original = f"age={row.age:g}|sex={row.sex}"
        if exact_counts[(float(row.age), row.sex)] >= 3:
            level = "exact_age_x_sex"
            resolved = original
            reason = "exact age x sex cell has at least three subjects"
        elif bin_sex_counts[(_age_bin(row.age), row.sex)] >= 3:
            level = "two_year_age_bin_x_sex"
            resolved = f"age_bin={_age_bin(row.age)}|sex={row.sex}"
            reason = "exact age x sex cell is sparse (<3); merged to 2-year age bin x sex"
        else:
            level = "two_year_age_bin"
            resolved = f"age_bin={_age_bin(row.age)}"
            reason = (
                "exact age x sex and 2-year age bin x sex cells are sparse (<3); "
                "sex constraint retained in the global balancing objective"
            )
        if bin_counts[_age_bin(row.age)] < 3:
            reason += "; resolved age bin remains sparse and is balanced globally"
        result[row.subject_id] = {
            "stratum_original": original,
            "stratum_resolved": resolved,
            "stratification_level": level,
            "stratification_reason": reason,
        }
    return result


def _objective(
    candidate_split: str,
    subject: SubjectMetadata,
    assigned: dict[str, list[SubjectMetadata]],
    counts: dict[str, int],
    targets: dict[str, int],
    strata: dict[str, dict[str, str]],
    totals: dict[str, Counter],
    overall_mean: float,
    overall_std: float,
) -> float:
    split_size = len(assigned[candidate_split]) + 1
    expected_fraction = targets[candidate_split] / sum(targets.values())
    score = 0.0
    attributes = {
        "stratum": strata[subject.subject_id]["stratum_resolved"],
        "age": f"{subject.age:g}",
        "sex": subject.sex,
    }
    weights = {"stratum": 5.0, "age": 3.0, "sex": 2.0}
    for kind, value in attributes.items():
        current = sum(
            1 for row in assigned[candidate_split]
            if (
                strata[row.subject_id]["stratum_resolved"] if kind == "stratum"
                else f"{row.age:g}" if kind == "age" else row.sex
            ) == value
        )
        desired = totals[kind][value] * expected_fraction
        score += weights[kind] * ((current + 1 - desired) ** 2) / max(desired, 1.0)
    ages = np.asarray([row.age for row in assigned[candidate_split]] + [subject.age], dtype=np.float64)
    score += 0.5 * ((float(ages.mean()) - overall_mean) / max(overall_std, 1e-6)) ** 2
    if split_size >= 2:
        score += 0.25 * ((float(ages.std(ddof=0)) - overall_std) / max(overall_std, 1e-6)) ** 2
    remaining_after = targets[candidate_split] - split_size
    score += 1e-6 * remaining_after
    return float(score)


def _allocate_sex_margins(
    subjects: Sequence[SubjectMetadata], split_counts: dict[str, int], seed: int
) -> dict[str, dict[str, int]]:
    """Integer apportionment with exact sex-row and split-column margins."""
    sex_counts = Counter(row.sex for row in subjects)
    total = len(subjects)
    ideal = {
        (sex, split): sex_count * split_counts[split] / total
        for sex, sex_count in sex_counts.items() for split in SPLITS
    }
    targets = {
        sex: {split: int(math.floor(ideal[(sex, split)])) for split in SPLITS}
        for sex in sorted(sex_counts)
    }
    row_remaining = {
        sex: sex_counts[sex] - sum(targets[sex].values()) for sex in targets
    }
    column_remaining = {
        split: split_counts[split] - sum(targets[sex][split] for sex in targets)
        for split in SPLITS
    }
    rng = np.random.default_rng(int(seed) + 97_531)
    tie = {(sex, split): float(rng.random()) for sex in targets for split in SPLITS}
    while sum(row_remaining.values()):
        choices = [
            (sex, split) for sex in targets for split in SPLITS
            if row_remaining[sex] > 0 and column_remaining[split] > 0
        ]
        if not choices:
            raise AssertionError("Could not satisfy integer sex/split margins")
        sex, split = max(
            choices,
            key=lambda pair: (
                ideal[pair] - math.floor(ideal[pair]),
                column_remaining[pair[1]], tie[pair], pair,
            ),
        )
        targets[sex][split] += 1
        row_remaining[sex] -= 1
        column_remaining[split] -= 1
    if any(column_remaining.values()):
        raise AssertionError(f"Unfilled split margins: {column_remaining}")
    return targets


def _allocate_group_margins(
    group_counts: dict[str, int], split_counts: dict[str, int], seed: int
) -> dict[str, dict[str, int]]:
    """Apportion resolved strata to splits with exact row/column totals."""
    total = sum(group_counts.values())
    if total != sum(split_counts.values()):
        raise ValueError("Group and split margins differ")
    ideal = {
        (group, split): count * split_counts[split] / total
        for group, count in group_counts.items() for split in SPLITS
    }
    targets = {
        group: {split: int(math.floor(ideal[(group, split)])) for split in SPLITS}
        for group in sorted(group_counts)
    }
    row_remaining = {
        group: group_counts[group] - sum(targets[group].values()) for group in targets
    }
    column_remaining = {
        split: split_counts[split] - sum(targets[group][split] for group in targets)
        for split in SPLITS
    }
    rng = np.random.default_rng(int(seed))
    tie = {(group, split): float(rng.random()) for group in targets for split in SPLITS}
    while sum(row_remaining.values()):
        choices = [
            (group, split) for group in targets for split in SPLITS
            if row_remaining[group] > 0 and column_remaining[split] > 0
        ]
        if not choices:
            raise AssertionError("Could not satisfy stratum/split margins")
        group, split = max(
            choices,
            key=lambda pair: (
                ideal[pair] - math.floor(ideal[pair]),
                column_remaining[pair[1]], tie[pair], pair,
            ),
        )
        targets[group][split] += 1
        row_remaining[group] -= 1
        column_remaining[split] -= 1
    if any(column_remaining.values()):
        raise AssertionError(f"Unfilled stratum columns: {column_remaining}")
    return targets


def _assignment_balance_score(
    assigned: dict[str, list[SubjectMetadata]],
    split_counts: dict[str, int],
    strata: dict[str, dict[str, str]],
) -> float:
    total = sum(split_counts.values())
    all_rows = [row for split in SPLITS for row in assigned[split]]
    age_totals = Counter(f"{row.age:g}" for row in all_rows)
    stratum_totals = Counter(strata[row.subject_id]["stratum_resolved"] for row in all_rows)
    overall_ages = np.asarray([row.age for row in all_rows], dtype=np.float64)
    overall_mean = float(overall_ages.mean())
    overall_std = float(overall_ages.std(ddof=0))
    minimum, maximum = float(overall_ages.min()), float(overall_ages.max())
    score = 0.0
    for split in SPLITS:
        fraction = split_counts[split] / total
        rows = assigned[split]
        ages = Counter(f"{row.age:g}" for row in rows)
        groups = Counter(strata[row.subject_id]["stratum_resolved"] for row in rows)
        for value, count in age_totals.items():
            desired = count * fraction
            score += 3.0 * (ages[value] - desired) ** 2 / max(desired, 1.0)
        for value, count in stratum_totals.items():
            desired = count * fraction
            score += 5.0 * (groups[value] - desired) ** 2 / max(desired, 1.0)
        values = np.asarray([row.age for row in rows], dtype=np.float64)
        score += ((float(values.mean()) - overall_mean) / max(overall_std, 1e-6)) ** 2
        score += 0.5 * ((float(values.std(ddof=0)) - overall_std) / max(overall_std, 1e-6)) ** 2
        # Extremes are protocol-relevant; a split missing an extreme receives a
        # strong but finite penalty so it never overrides exact split margins.
        score += 10.0 * (minimum not in values) + 10.0 * (maximum not in values)
    return float(score)


def _refine_by_same_sex_swaps(
    assigned: dict[str, list[SubjectMetadata]],
    split_counts: dict[str, int],
    strata: dict[str, dict[str, str]],
    max_iterations: int = 100,
) -> dict[str, Any]:
    """Improve age/stratum balance without changing split or sex margins."""
    initial = _assignment_balance_score(assigned, split_counts, strata)
    current = initial
    swaps = 0
    for _ in range(max_iterations):
        best: tuple[float, str, int, str, int] | None = None
        for left_index, left in enumerate(SPLITS):
            for right in SPLITS[left_index + 1:]:
                for li, left_row in enumerate(assigned[left]):
                    for ri, right_row in enumerate(assigned[right]):
                        if left_row.sex != right_row.sex or left_row.age == right_row.age:
                            continue
                        assigned[left][li], assigned[right][ri] = right_row, left_row
                        candidate = _assignment_balance_score(assigned, split_counts, strata)
                        assigned[left][li], assigned[right][ri] = left_row, right_row
                        proposal = (candidate, left, li, right, ri)
                        if best is None or proposal < best:
                            best = proposal
        if best is None or best[0] >= current - 1e-12:
            break
        current, left, li, right, ri = best
        assigned[left][li], assigned[right][ri] = assigned[right][ri], assigned[left][li]
        swaps += 1
    return {"initial_score": initial, "final_score": current, "same_sex_swap_count": swaps}


def build_subject_split(
    subjects: Sequence[SubjectMetadata],
    *,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    split_seed: int = 42,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not subjects:
        raise ValueError("No subjects supplied")
    subject_ids = [row.subject_id for row in subjects]
    if len(subject_ids) != len(set(subject_ids)):
        raise ValueError("Duplicate subject_id values")
    counts = allocate_split_counts(
        len(subjects), train_ratio, validation_ratio, test_ratio
    )
    strata = resolve_strata(subjects)
    totals = {
        "stratum": Counter(strata[row.subject_id]["stratum_resolved"] for row in subjects),
        "age": Counter(f"{row.age:g}" for row in subjects),
        "sex": Counter(row.sex for row in subjects),
    }
    overall_ages = np.asarray([row.age for row in subjects], dtype=np.float64)
    rng = np.random.default_rng(int(split_seed))
    random_keys = {row.subject_id: float(rng.random()) for row in subjects}
    assigned: dict[str, list[SubjectMetadata]] = {name: [] for name in SPLITS}
    sex_targets = _allocate_sex_margins(subjects, counts, split_seed)
    for sex in sorted(sex_targets):
        by_group: dict[str, list[SubjectMetadata]] = defaultdict(list)
        for row in subjects:
            if row.sex == sex:
                by_group[strata[row.subject_id]["stratum_resolved"]].append(row)
        group_targets = _allocate_group_margins(
            {group: len(rows) for group, rows in by_group.items()},
            sex_targets[sex],
            split_seed + sum(ord(value) for value in sex),
        )
        for group in sorted(by_group):
            ordered = sorted(
                by_group[group], key=lambda row: (random_keys[row.subject_id], row.subject_id)
            )
            offset = 0
            # Seeded tie ordering prevents the same split receiving the first
            # subject of every sparse cell while preserving exact quotas.
            group_split_order = sorted(
                SPLITS,
                key=lambda split: (
                    -group_targets[group][split],
                    sha256_text(f"{split_seed}|{sex}|{group}|{split}"),
                ),
            )
            for split in group_split_order:
                take = group_targets[group][split]
                assigned[split].extend(ordered[offset:offset + take])
                offset += take
            if offset != len(ordered):
                raise AssertionError(f"Stratum allocation mismatch for {sex}/{group}")
    if {name: len(assigned[name]) for name in SPLITS} != counts:
        raise AssertionError("Strict split counts were not satisfied")
    refinement = _refine_by_same_sex_swaps(assigned, counts, strata)
    split_by_subject = {
        row.subject_id: split for split, rows in assigned.items() for row in rows
    }
    manifest = []
    by_id = {row.subject_id: row for row in subjects}
    for subject_id in sorted(by_id):
        row = by_id[subject_id]
        manifest.append({
            "subject_id": subject_id,
            "age": f"{row.age:.10g}",
            "sex": row.sex,
            "split": split_by_subject[subject_id],
            **strata[subject_id],
            "split_seed": int(split_seed),
            "window_count": int(row.window_count),
            "conditions": ";".join(row.conditions),
            "recording_ids": ";".join(row.recording_ids),
        })
    summary = summarize_split(
        manifest,
        requested_ratios={
            "train": float(train_ratio),
            "validation": float(validation_ratio),
            "test": float(test_ratio),
        },
        algorithm_version=ALGORITHM_VERSION,
    )
    summary["sex_split_targets"] = sex_targets
    summary["balance_refinement"] = refinement
    return manifest, summary


def summarize_split(
    manifest: Sequence[dict[str, Any]], *, requested_ratios: dict[str, float],
    algorithm_version: str = ALGORITHM_VERSION,
) -> dict[str, Any]:
    total = len(manifest)
    rows_by_split = {name: [row for row in manifest if row["split"] == name] for name in SPLITS}
    summary = {
        "algorithm_version": algorithm_version,
        "subject_count": total,
        "requested_ratios": requested_ratios,
        "splits": {},
        "subject_split_isolation": len({row["subject_id"] for row in manifest}) == total,
    }
    for name, rows in rows_by_split.items():
        ages = np.asarray([float(row["age"]) for row in rows], dtype=np.float64)
        summary["splits"][name] = {
            "subject_count": len(rows),
            "actual_ratio": len(rows) / total,
            "age_mean": float(ages.mean()),
            "age_std_ddof0": float(ages.std(ddof=0)),
            "age_min": float(ages.min()),
            "age_max": float(ages.max()),
            "sex_counts": dict(sorted(Counter(row["sex"] for row in rows).items())),
            "stratification_level_counts": dict(sorted(Counter(
                row["stratification_level"] for row in rows
            ).items())),
        }
    return summary


MANIFEST_FIELDS = [
    "subject_id", "age", "sex", "split", "stratum_original", "stratum_resolved",
    "stratification_level", "stratification_reason", "split_seed", "window_count",
    "conditions", "recording_ids",
]


def write_manifest(path: Path, rows: Sequence[dict[str, Any]]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in MANIFEST_FIELDS} for row in rows)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or list(rows[0]) != MANIFEST_FIELDS:
        raise ValueError("Split manifest schema mismatch")
    return rows


def distribution_rows(manifest: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    age_rows = []
    age_sex_rows = []
    for split in SPLITS:
        rows = [row for row in manifest if row["split"] == split]
        for age, count in sorted(Counter(float(row["age"]) for row in rows).items()):
            age_rows.append({"split": split, "age": age, "subject_count": count})
        for (age, sex), count in sorted(Counter(
            (float(row["age"]), row["sex"]) for row in rows
        ).items()):
            age_sex_rows.append({"split": split, "age": age, "sex": sex, "subject_count": count})
    return age_rows, age_sex_rows


def dump_json(path: Path, payload: Any) -> None:
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
