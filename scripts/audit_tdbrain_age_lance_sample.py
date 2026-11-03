#!/usr/bin/env python
"""Strictly limited, read-only age-label gate for a remote COS Lance dataset.

The script never writes to COS.  It inventories one directory level, reads Lance
metadata only, and materializes at most ten projected metadata rows *only* when
age semantics are documented.  All limits are enforced internally.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa


AUDIT_MODE = "limited_age_label_gate"
MAX_HEAD_ROWS_HARD = 10
MAX_SUBJECTS_HARD = 3
MAX_ROWS_PER_SUBJECT_HARD = 3
MAX_INVENTORY_ENTRIES = 100
MAX_SIDECAR_BYTES = 1_048_576
FORBIDDEN_COLUMNS = re.compile(
    r"(^|[._])(data|feature|features|signal|signals|token|tokens|mask|masks|tensor|waveform)([._]|$)",
    re.IGNORECASE,
)
AGE_EXPLICIT = {
    "age", "age_years", "subject_age", "participant_age", "session_age",
    "age_at_recording", "demographics.age", "metadata.age",
}
AGE_WEAK = {"target", "label", "label_value"}
ID_COLUMNS = {
    "sample_id", "subject_id", "participant_id", "session_id",
    "recording_id", "split", "task", "label", "label_name",
}
SIDE_CAR_RE = re.compile(
    r"(^|/)(readme(?:\.[a-z0-9]+)?|description\.(?:json|toml|txt|md)|"
    r"dataset(?:_info|_metadata)?\.(?:json|csv|toml|txt|md)|"
    r"metadata\.(?:json|csv|toml|txt|md)|versions\.toml|"
    r"manifest\.(?:json|csv|toml|txt)|schema\.json|config\.(?:json|toml|yaml|yml))$",
    re.IGNORECASE,
)
SAMPLED_HEADERS = [
    "sample_id", "subject_id", "participant_id", "session_id", "recording_id",
    "age_field", "age_value", "age_dtype", "task", "label", "label_name", "split",
]
CONSISTENCY_HEADERS = [
    "subject_key", "session_key", "rows_checked", "age_values", "age_consistent",
    "raw_age_matches_label", "assessment",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-uri", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-head-rows", type=int, default=10)
    parser.add_argument("--max-subjects", type=int, default=3)
    parser.add_argument("--max-rows-per-subject", type=int, default=3)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--stop-if-age-unconfirmed", action="store_true")
    parser.add_argument("--coscli", default=shutil.which("coscli") or "coscli")
    parser.add_argument(
        "--cos-endpoint",
        help="Explicit COS endpoint passed to coscli with -e (for example cos.ap-beijing.myqcloud.com).",
    )
    parser.add_argument(
        "--storage-options",
        type=Path,
        help="Optional JSON mapping for Lance S3-compatible read access. Never copied to output.",
    )
    return parser.parse_args(argv)


def enforce_limits(args: argparse.Namespace) -> None:
    if not args.metadata_only:
        raise ValueError("--metadata-only is mandatory")
    if not args.stop_if_age_unconfirmed:
        raise ValueError("--stop-if-age-unconfirmed is mandatory")
    if not (1 <= args.max_head_rows <= MAX_HEAD_ROWS_HARD):
        raise ValueError("--max-head-rows must be between 1 and 10")
    if not (1 <= args.max_subjects <= MAX_SUBJECTS_HARD):
        raise ValueError("--max-subjects must be between 1 and 3")
    if not (1 <= args.max_rows_per_subject <= MAX_ROWS_PER_SUBJECT_HARD):
        raise ValueError("--max-rows-per-subject must be between 1 and 3")
    if not str(args.source_uri).startswith("cos://"):
        raise ValueError("source URI must use cos://")


def redact(text: str) -> str:
    text = re.sub(r"(?i)(secret(?:id|key)?|token|authorization|credential)\s*[:=]\s*\S+", r"\1=<redacted>", text)
    text = re.sub(r"(?i)(q-signature|x-cos-security-token)=[^&\s]+", r"\1=<redacted>", text)
    return text


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], headers: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_coscli(
    coscli: str,
    args: list[str],
    run_log: list[str],
    cos_endpoint: str | None = None,
) -> subprocess.CompletedProcess[str]:
    allowed = {"ls", "stat", "cp"}
    if not args or args[0] not in allowed:
        raise AssertionError(f"disallowed COS operation: {args!r}")
    if args[0] == "cp":
        if len(args) < 3 or not args[1].startswith("cos://") or args[2].startswith("cos://"):
            raise AssertionError("COS cp is restricted to one remote source and one local destination")
        if "--recursive" in args or "-r" in args:
            raise AssertionError("recursive COS copy is forbidden")
    command = [coscli, *args]
    if cos_endpoint:
        command.extend(["-e", cos_endpoint])
    command.append("--disable-log")
    run_log.append("COMMAND " + " ".join(redact(part) for part in command))
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    safe = redact((completed.stdout or "") + (completed.stderr or "")).strip()
    run_log.append(f"EXIT {completed.returncode}")
    if safe:
        run_log.append(safe[:8000])
    return completed


def parse_ls_output(text: str, base_uri: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    bucket_match = re.match(r"cos://([^/]+)", base_uri)
    bucket = bucket_match.group(1) if bucket_match else ""
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.lower().startswith(("total", "last modified", "key", "----")):
            continue
        if "TOTAL OBJECTS" in stripped.upper() or stripped in {"|", "+"}:
            continue
        size = None
        if "|" in stripped:
            parts = [part.strip() for part in stripped.split("|")]
            if len(parts) < 3:
                continue
            key, object_type = parts[0], parts[1]
            if not key or key.lower() == "key" or set(key) <= {"-", "+"}:
                continue
            name = key if key.startswith("cos://") else f"cos://{bucket}/{key.lstrip('/')}"
            is_dir = object_type.upper() == "DIR" or key.endswith("/")
            size_text = parts[4] if len(parts) > 4 else ""
            match = re.search(r"([0-9.]+)\s*(B|KB|MB|GB)?", size_text, re.IGNORECASE)
            if match:
                multiplier = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}.get(
                    (match.group(2) or "B").upper(), 1
                )
                size = int(float(match.group(1)) * multiplier)
        else:
            uri_match = re.search(r"cos://[^\s]+", stripped)
            tokens = stripped.split()
            name = uri_match.group(0) if uri_match else tokens[-1]
            if name in {"-", "DIR"} or name.startswith(("Error", "ERROR")):
                continue
            if not name.startswith("cos://"):
                name = base_uri.rstrip("/") + "/" + name.lstrip("/")
            is_dir = stripped.endswith("/") or " DIR " in f" {stripped} "
        entries.append({"uri": name, "is_dir": is_dir, "size_bytes": size})
    unique = {entry["uri"]: entry for entry in entries}
    return list(unique.values())[:MAX_INVENTORY_ENTRIES]


def shallow_list(
    coscli: str,
    uri: str,
    run_log: list[str],
    cos_endpoint: str | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    completed = run_coscli(
        coscli,
        ["ls", uri.rstrip("/") + "/", "--limit", str(MAX_INVENTORY_ENTRIES)],
        run_log,
        cos_endpoint,
    )
    text = (completed.stdout or "") + (completed.stderr or "")
    if completed.returncode != 0:
        return [], redact(text).strip() or f"coscli ls exited {completed.returncode} without diagnostic text"
    return parse_ls_output(text, uri), None


def download_small_sidecars(
    entries: list[dict[str, Any]],
    output_dir: Path,
    coscli: str,
    run_log: list[str],
    cos_endpoint: str | None,
    inventory: dict[str, Any],
) -> list[dict[str, Any]]:
    sidecars: list[dict[str, Any]] = []
    local_dir = output_dir / ".sidecars"
    for entry in entries:
        size = entry.get("size_bytes")
        if entry.get("is_dir") or not SIDE_CAR_RE.search(entry["uri"]):
            continue
        if size is None or size > MAX_SIDECAR_BYTES:
            continue
        local_dir.mkdir(exist_ok=True)
        local_path = local_dir / Path(entry["uri"]).name
        completed = run_coscli(
            coscli,
            ["cp", entry["uri"], str(local_path), "--process-log=false", "--fail-output=false"],
            run_log,
            cos_endpoint,
        )
        if completed.returncode != 0 or not local_path.exists():
            continue
        text = local_path.read_text(encoding="utf-8", errors="replace")
        sidecars.append({"uri": entry["uri"], "text": text})
        inventory["small_local_metadata_downloads"].append({
            "kind": "sidecar", "remote_uri": entry["uri"],
            "local_path": str(local_path), "bytes": local_path.stat().st_size,
            "sha256": sha256(local_path),
        })
    return sidecars


def field_metadata(field: pa.Field) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, value in (field.metadata or {}).items():
        result[key.decode("utf-8", "replace")] = value.decode("utf-8", "replace")
    return result


def schema_to_json(schema: pa.Schema) -> dict[str, Any]:
    def field_obj(field: pa.Field) -> dict[str, Any]:
        item = {
            "name": field.name,
            "type": str(field.type),
            "nullable": field.nullable,
            "metadata": field_metadata(field),
        }
        if pa.types.is_struct(field.type):
            item["children"] = [field_obj(child) for child in field.type]
        return item

    return {
        "fields": [field_obj(field) for field in schema],
        "metadata": {
            key.decode("utf-8", "replace"): value.decode("utf-8", "replace")
            for key, value in (schema.metadata or {}).items()
        },
        "text": str(schema),
    }


def flatten_fields(schema: pa.Schema) -> list[tuple[str, pa.Field]]:
    output: list[tuple[str, pa.Field]] = []
    def visit(prefix: str, field: pa.Field) -> None:
        name = f"{prefix}.{field.name}" if prefix else field.name
        output.append((name, field))
        if pa.types.is_struct(field.type):
            for child in field.type:
                visit(name, child)
    for top in schema:
        visit("", top)
    return output


def _semantics_text(schema: pa.Schema, sidecars: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    if schema.metadata:
        parts.extend(value.decode("utf-8", "replace") for value in schema.metadata.values())
    for _, field in flatten_fields(schema):
        parts.extend(field_metadata(field).values())
    parts.extend(item.get("text", "") for item in sidecars)
    return "\n".join(parts)


def detect_age_candidates(schema: pa.Schema, sidecars: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str | None]:
    semantics = _semantics_text(schema, sidecars)
    lower_semantics = semantics.lower()
    candidates: list[dict[str, Any]] = []
    semantics_source = None
    for path, field in flatten_fields(schema):
        normalized = path.lower()
        explicit = normalized in AGE_EXPLICIT or normalized.split(".")[-1] in AGE_EXPLICIT
        weak = normalized in AGE_WEAK
        if not (explicit or weak or "age" in normalized):
            continue
        metadata_text = " ".join(field_metadata(field).values()).lower()
        combined_text = metadata_text + " " + lower_semantics
        age_semantics = bool(re.search(
            r"\bage\b|age[ _-]?(years?|at[ _-]recording|bin|group|class)|years?",
            combined_text,
        ))
        explicit_non_age_label = weak and not age_semantics and bool(re.search(
            r"depress(?:ion|ed)|\bcontrol\b|classification label|task\s*=\s*[\"']depression",
            combined_text,
        ))
        if explicit_non_age_label:
            continue
        documented = explicit or (weak and age_semantics)
        evidence: list[str] = []
        if explicit:
            evidence.append("explicit_age_field_name")
        if metadata_text and re.search(r"\bage\b|years?", metadata_text):
            evidence.append("field_metadata")
            semantics_source = semantics_source or "arrow_field_metadata"
        if weak and documented and "field_metadata" not in evidence:
            evidence.append("dataset_metadata_or_sidecar")
            semantics_source = semantics_source or "dataset_metadata_or_sidecar"
        type_text = str(field.type)
        is_numeric = pa.types.is_integer(field.type) or pa.types.is_floating(field.type)
        binned_text = combined_text
        is_binned = documented and bool(re.search(r"age[ _-]?(bin|group|class)|young\s*/\s*old|\d+\s*[-–]\s*\d+", binned_text))
        is_transformed = bool(re.search(r"z[- ]?score|standardi[sz]ed|normali[sz]ed|scaled", binned_text))
        candidates.append({
            "field_name": path,
            "dtype": type_text,
            "is_numeric": is_numeric,
            "semantics_documented": documented,
            "is_binned": is_binned,
            "is_transformed": is_transformed,
            "evidence": evidence,
        })
    if candidates and semantics_source is None and any(c["semantics_documented"] for c in candidates):
        semantics_source = "field_name"
    return candidates, semantics_source


def projection_for(schema: pa.Schema, age_field: str) -> list[str]:
    names = set(schema.names)
    selected = [name for name in schema.names if name in ID_COLUMNS and not FORBIDDEN_COLUMNS.search(name)]
    top_age = age_field.split(".")[0]
    if top_age in names and top_age not in selected and not FORBIDDEN_COLUMNS.search(top_age):
        selected.append(top_age)
    assert selected and all(not FORBIDDEN_COLUMNS.search(name) for name in selected)
    return selected


def load_storage_options(path: Path | None) -> dict[str, str] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("storage options JSON must be an object")
    return {str(key): str(value) for key, value in payload.items()}


def cos_to_lance_uri(uri: str) -> str:
    return "s3://" + uri[len("cos://"):]


def materialize_head(uri: str, options: dict[str, str], columns: list[str], limit: int):
    import lance
    dataset = lance.dataset(cos_to_lance_uri(uri), storage_options=options)
    scanner = dataset.scanner(columns=columns, limit=limit, scan_in_order=True, batch_readahead=1, fragment_readahead=1)
    table = scanner.to_table()
    if table.num_rows > limit:
        raise AssertionError("Lance scanner exceeded head-row limit")
    if set(table.column_names) != set(columns):
        raise AssertionError("unexpected columns materialized")
    return table


def normalize_age(value: Any) -> tuple[float | None, str | None]:
    if value is None:
        return None, "null"
    if isinstance(value, bool):
        return None, "boolean"
    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            return None, "non_finite"
        if number in {-1.0, 0.0, 999.0}:
            return None, "placeholder"
        return number, None
    text = str(value).strip()
    if not text or text.lower() in {"unknown", "na", "n/a", "none", "null", "-1", "999"}:
        return None, "placeholder"
    if re.fullmatch(r"\d+\s*[-–]\s*\d+", text) or text.lower() in {"young", "old", "young/old"}:
        return None, "bin"
    try:
        return float(text), None
    except ValueError:
        return None, "non_numeric"


def pick_nested(row: dict[str, Any], dotted: str) -> Any:
    value: Any = row
    for part in dotted.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def sample_rows_to_csv(rows: list[dict[str, Any]], age_field: str, age_dtype: str) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        output.append({
            "sample_id": row.get("sample_id", ""),
            "subject_id": row.get("subject_id", ""),
            "participant_id": row.get("participant_id", ""),
            "session_id": row.get("session_id", ""),
            "recording_id": row.get("recording_id", ""),
            "age_field": age_field,
            "age_value": pick_nested(row, age_field),
            "age_dtype": age_dtype,
            "task": row.get("task", ""),
            "label": row.get("label", ""),
            "label_name": row.get("label_name", ""),
            "split": row.get("split", ""),
        })
    return output


def limited_consistency(rows: list[dict[str, Any]], age_field: str, max_subjects: int, max_rows: int):
    subject_column = "subject_id" if any(row.get("subject_id") is not None for row in rows) else "participant_id"
    selected: list[str] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        subject = str(row.get(subject_column, "") or "")
        if subject not in selected:
            if len(selected) >= max_subjects:
                continue
            selected.append(subject)
        session = str(row.get("session_id", "") or "")
        if len(grouped[(subject, session)]) < max_rows:
            grouped[(subject, session)].append(row)
    output = []
    for (subject, session), group in grouped.items():
        ages = [pick_nested(row, age_field) for row in group]
        normalized = [normalize_age(value)[0] for value in ages]
        finite = [value for value in normalized if value is not None]
        consistent = len(set(finite)) <= 1 and len(finite) == len(group)
        label_matches = "unknown"
        if all("label" in row for row in group):
            comparisons = []
            for row, age in zip(group, normalized):
                label, error = normalize_age(row.get("label"))
                if error is None and age is not None:
                    comparisons.append(math.isclose(label, age, rel_tol=0, abs_tol=1e-6))
            if comparisons:
                label_matches = str(all(comparisons)).lower()
        output.append({
            "subject_key": subject,
            "session_key": session,
            "rows_checked": len(group),
            "age_values": json.dumps(ages, ensure_ascii=False),
            "age_consistent": str(consistent).lower(),
            "raw_age_matches_label": label_matches,
            "assessment": "consistent" if consistent else "conflict_or_missing",
        })
    return output, len(selected)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def base_summary(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "source_uri": args.source_uri,
        "audit_mode": AUDIT_MODE,
        "read_only": True,
        "resolved_lance_path": None,
        "dataset_path_ambiguous": False,
        "schema_read_success": False,
        "age_candidate_fields": [],
        "age_semantics_source": None,
        "age_field_confirmed": False,
        "age_field_name": None,
        "age_dtype": None,
        "age_unit": None,
        "age_is_numeric": False,
        "age_is_exact_or_integer_years": False,
        "age_is_binned": False,
        "age_is_transformed": False,
        "age_transform_recoverable": False,
        "age_sample_non_null": False,
        "schema_only_stage_completed": False,
        "metadata_rows_materialized": 0,
        "subjects_sampled": 0,
        "rows_per_subject_limit": args.max_rows_per_subject,
        "age_consistency_checked": False,
        "signal_column_materialized": False,
        "feature_column_materialized": False,
        "data_tensor_materialized": False,
        "full_fragment_scan_performed": False,
        "full_dataset_downloaded": False,
        "cos_objects_modified": False,
        "stop_status": None,
        "stop_reason": None,
        "eligible_for_full_age_audit": False,
        "limited_audit_result": None,
        "full_audit_started": False,
        "full_audit_recommended": False,
    }


def make_report(summary: dict[str, Any], schema_doc: dict[str, Any], inventory: dict[str, Any]) -> str:
    fields = [item.get("name") for item in schema_doc.get("fields", [])]
    warning = ""
    if summary["stop_status"] in {
        "STOP_AGE_PRESENT_BUT_EMPTY_IN_SAMPLE", "STOP_AGE_PRESENT_BUT_PLACEHOLDER",
        "STOP_AGE_PRESENT_BUT_TRANSFORMED_UNRECOVERABLE", "STOP_AGE_SEMANTICS_UNCONFIRMED",
    }:
        warning = "\n这是受限抽样中未确认年龄，不能据此证明全量数据绝对不存在年龄；按照本次要求，未继续扫描。\n"
    return f"""# TDBRAIN Lance 年龄标签受限抽样审计

- 源路径：`{summary['source_uri']}`
- 解析路径：`{summary['resolved_lance_path'] or '未解析'}`
- 模式：`{summary['audit_mode']}`，只读
- 停止状态：`{summary['stop_status']}`
- 结论：`{summary['limited_audit_result']}`

## Schema 闸门

- schema 读取成功：`{str(summary['schema_read_success']).lower()}`
- 顶层字段：`{json.dumps(fields, ensure_ascii=False)}`
- 年龄候选：`{json.dumps(summary['age_candidate_fields'], ensure_ascii=False)}`
- 年龄语义来源：`{summary['age_semantics_source'] or '无'}`
- 停止原因：{summary['stop_reason']}
{warning}
## 受限读取证明

- materialize metadata 行数：{summary['metadata_rows_materialized']}
- 抽样 subject 数：{summary['subjects_sampled']}
- signal/feature/data tensor materialized：`false`
- 全 fragment 扫描：`false`
- 全数据集下载：`false`
- COS 对象修改：`false`
- 全量年龄审计启动：`false`

## 后续建议

只有 `eligible_for_full_age_audit=true` 时才建议另行设计全量年龄审计；本次不会自动启动。
"""


def initialize_outputs(output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    write_csv(output_dir / "sampled_metadata_rows.csv", [], SAMPLED_HEADERS)
    write_csv(output_dir / "age_consistency_sample.csv", [], CONSISTENCY_HEADERS)
    (output_dir / "audit_errors.log").write_text("", encoding="utf-8")
    (output_dir / "run.log").write_text("", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    enforce_limits(args)
    initialize_outputs(args.output_dir)
    run_log = [f"created_at={datetime.now(timezone.utc).isoformat()}", f"audit_mode={AUDIT_MODE}"]
    errors: list[str] = []
    summary = base_summary(args)
    inventory: dict[str, Any] = {
        "source_uri": args.source_uri,
        "inventory_depth": "shallow",
        "entry_limit": MAX_INVENTORY_ENTRIES,
        "entries": [],
        "candidate_paths": [],
        "small_local_metadata_downloads": [],
        "cos_endpoint": args.cos_endpoint,
    }
    schema_doc: dict[str, Any] = {"fields": [], "metadata": {}, "text": ""}
    candidates: list[dict[str, Any]] = []
    sampled_csv: list[dict[str, Any]] = []
    consistency_csv: list[dict[str, Any]] = []
    try:
        entries, error = shallow_list(args.coscli, args.source_uri, run_log, args.cos_endpoint)
        inventory["entries"] = entries
        if error:
            summary.update({
                "stop_status": "STOP_REMOTE_ACCESS_FAILED",
                "stop_reason": error,
                "limited_audit_result": "age_candidate_not_usable_or_unverified",
            })
            raise RuntimeError(error)

        sidecars = download_small_sidecars(
            entries, args.output_dir, args.coscli, run_log, args.cos_endpoint, inventory
        )
        names = [entry["uri"].rstrip("/").split("/")[-1] for entry in entries]
        direct_markers = {"_versions", "data", "_transactions"}.intersection(names)
        if "_versions" in direct_markers:
            resolved = args.source_uri.rstrip("/")
            inventory["candidate_paths"] = [resolved]
        else:
            directory_candidates = [entry["uri"].rstrip("/") for entry in entries if entry["is_dir"]]
            inventory["candidate_paths"] = directory_candidates
            if len(directory_candidates) != 1:
                summary.update({
                    "dataset_path_ambiguous": True,
                    "stop_status": "STOP_DATASET_PATH_AMBIGUOUS",
                    "stop_reason": f"shallow inventory did not uniquely identify a Lance dataset; candidates={directory_candidates}",
                    "limited_audit_result": "age_candidate_not_usable_or_unverified",
                })
                raise RuntimeError(summary["stop_reason"])
            resolved = directory_candidates[0]
        summary["resolved_lance_path"] = resolved

        # Prefer direct remote metadata access only when explicit storage options were supplied.
        storage_options = load_storage_options(args.storage_options)
        dataset = None
        if storage_options is not None:
            import lance
            dataset = lance.dataset(cos_to_lance_uri(resolved), storage_options=storage_options)
            schema = dataset.schema
        else:
            version_uri = resolved.rstrip("/") + "/_versions/"
            version_entries, version_error = shallow_list(
                args.coscli, version_uri, run_log, args.cos_endpoint
            )
            if version_error:
                raise RuntimeError("could not list Lance manifests: " + version_error)
            manifests = [entry for entry in version_entries if re.search(r"\.manifest$", entry["uri"], re.IGNORECASE)]
            if not manifests:
                raise RuntimeError("no Lance manifest found in shallow _versions inventory")
            def manifest_number(item: dict[str, Any]) -> int:
                match = re.search(r"/(\d+)\.manifest$", item["uri"])
                return int(match.group(1)) if match else -1
            manifest = max(manifests, key=manifest_number)
            local_root = args.output_dir / ".manifest_only"
            local_version = local_root / "_versions"
            local_version.mkdir(parents=True)
            local_manifest = local_version / Path(manifest["uri"]).name
            completed = run_coscli(
                args.coscli,
                ["cp", manifest["uri"], str(local_manifest), "--process-log=false", "--fail-output=false"],
                run_log,
                args.cos_endpoint,
            )
            if completed.returncode != 0 or not local_manifest.exists():
                raise RuntimeError("failed to download the single Lance manifest for schema-only inspection")
            inventory["small_local_metadata_downloads"].append({
                "kind": "lance_manifest", "remote_uri": manifest["uri"],
                "local_path": str(local_manifest), "bytes": local_manifest.stat().st_size,
                "sha256": sha256(local_manifest),
            })
            import lance
            dataset = lance.dataset(local_root)
            schema = dataset.schema

        schema_doc = schema_to_json(schema)
        summary["schema_read_success"] = True
        summary["schema_only_stage_completed"] = True
        candidates, semantics_source = detect_age_candidates(schema, sidecars)
        summary["age_candidate_fields"] = [item["field_name"] for item in candidates]
        summary["age_semantics_source"] = semantics_source
        if not candidates:
            summary.update({
                "stop_status": "STOP_NO_AGE_FIELD", "stop_reason": "schema and available metadata contain no age candidate field",
                "limited_audit_result": "no_age_field_found",
            })
        else:
            documented = [item for item in candidates if item["semantics_documented"]]
            if not documented:
                summary.update({
                    "stop_status": "STOP_AGE_SEMANTICS_UNKNOWN", "stop_reason": "only weak or undocumented label/target semantics were found",
                    "limited_audit_result": "age_candidate_not_usable_or_unverified",
                })
            else:
                candidate = documented[0]
                summary.update({
                    "age_field_name": candidate["field_name"], "age_dtype": candidate["dtype"],
                    "age_is_numeric": candidate["is_numeric"], "age_is_binned": candidate["is_binned"],
                    "age_is_transformed": candidate["is_transformed"],
                })
                if candidate["is_binned"]:
                    summary.update({
                        "stop_status": "STOP_AGE_IS_BINNED", "stop_reason": "documented age candidate is a bin/class rather than raw numeric years",
                        "limited_audit_result": "age_candidate_not_usable_or_unverified",
                    })
                elif storage_options is None:
                    summary.update({
                        "stop_status": "STOP_REMOTE_ACCESS_FAILED",
                        "stop_reason": "age candidate passed schema semantics, but no explicit Lance-compatible storage options were supplied for projected row access",
                        "limited_audit_result": "age_candidate_not_usable_or_unverified",
                    })
                else:
                    projection = projection_for(schema, candidate["field_name"])
                    table = materialize_head(resolved, storage_options, projection, args.max_head_rows)
                    rows = table.to_pylist()
                    summary["metadata_rows_materialized"] = len(rows)
                    sampled_csv = sample_rows_to_csv(rows, candidate["field_name"], candidate["dtype"])
                    parsed = [normalize_age(pick_nested(row, candidate["field_name"])) for row in rows]
                    values = [value for value, error_name in parsed if error_name is None]
                    errors_found = Counter(error_name for value, error_name in parsed if error_name)
                    if not values and errors_found.get("null") == len(rows):
                        status = "STOP_AGE_PRESENT_BUT_EMPTY_IN_SAMPLE"
                    elif not values and (errors_found.get("placeholder") or errors_found.get("boolean")):
                        status = "STOP_AGE_PRESENT_BUT_PLACEHOLDER"
                    elif candidate["is_transformed"]:
                        status = "STOP_AGE_PRESENT_BUT_TRANSFORMED_UNRECOVERABLE"
                    elif not values or any(value < 5 or value > 89 for value in values):
                        status = "STOP_AGE_SEMANTICS_UNCONFIRMED"
                    else:
                        status = "AGE_FIELD_CONFIRMED_FOR_LIMITED_AUDIT"
                    if status != "AGE_FIELD_CONFIRMED_FOR_LIMITED_AUDIT":
                        summary.update({
                            "stop_status": status,
                            "stop_reason": f"limited metadata sample did not confirm usable raw age; sample diagnostics={dict(errors_found)}",
                            "limited_audit_result": "age_candidate_not_usable_or_unverified",
                        })
                    else:
                        consistency_csv, subjects = limited_consistency(
                            rows, candidate["field_name"], args.max_subjects, args.max_rows_per_subject
                        )
                        summary.update({
                            "age_field_confirmed": True, "age_unit": "years", "age_is_exact_or_integer_years": all(float(v).is_integer() for v in values),
                            "age_transform_recoverable": not candidate["is_transformed"], "age_sample_non_null": True,
                            "subjects_sampled": subjects, "age_consistency_checked": True,
                            "stop_status": status, "stop_reason": "usable numeric age in years confirmed in limited projected sample",
                            "eligible_for_full_age_audit": True, "limited_audit_result": "usable_age_field_confirmed",
                            "full_audit_recommended": True,
                        })
    except Exception as exc:
        errors.append(redact("".join(traceback.format_exception(type(exc), exc, exc.__traceback__))))
        if summary["stop_status"] is None:
            summary.update({
                "stop_status": "STOP_REMOTE_ACCESS_FAILED",
                "stop_reason": redact(str(exc)),
                "limited_audit_result": "age_candidate_not_usable_or_unverified",
            })
    finally:
        write_json(args.output_dir / "remote_inventory.json", inventory)
        write_json(args.output_dir / "schema.json", schema_doc)
        write_json(args.output_dir / "age_field_candidates.json", {"candidates": candidates})
        write_csv(args.output_dir / "sampled_metadata_rows.csv", sampled_csv, SAMPLED_HEADERS)
        write_csv(args.output_dir / "age_consistency_sample.csv", consistency_csv, CONSISTENCY_HEADERS)
        write_json(args.output_dir / "audit_summary.json", summary)
        (args.output_dir / "audit_report.md").write_text(make_report(summary, schema_doc, inventory), encoding="utf-8")
        (args.output_dir / "audit_errors.log").write_text("\n".join(errors), encoding="utf-8")
        (args.output_dir / "run.log").write_text("\n".join(run_log) + "\n", encoding="utf-8")
    print(json.dumps({
        "output_dir": str(args.output_dir.resolve()), "stop_status": summary["stop_status"],
        "metadata_rows_materialized": summary["metadata_rows_materialized"],
        "signal_or_feature_materialized": False, "full_scan_performed": False,
        "full_download_performed": False, "cos_modified": False,
    }, ensure_ascii=False, indent=2))
    return 0 if summary["stop_status"] != "STOP_REMOTE_ACCESS_FAILED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
