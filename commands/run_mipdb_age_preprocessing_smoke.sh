#!/usr/bin/env bash
set -euo pipefail

: "${HUADA_ROOT:=/opt/ml/input}"
: "${PYTHON_BIN:=python}"
: "${MIPDB_DATA_DIR:=${HUADA_ROOT}/data/MIPDB/mipdb_download/nm000153_py}"
: "${MIPDB_AGE_METADATA:=${HUADA_ROOT}/data/MIPDB/MIPDB_PublicFile.csv}"
: "${MIPDB_ALLOWLIST:=${HUADA_ROOT}/configs/mipdb_age_20_35_subjects.txt}"
: "${MIPDB_SOURCE_URI:=local://${MIPDB_DATA_DIR}}"
: "${TIONE_OUTPUT_ROOT:=/opt/ml/output}"
: "${MIPDB_SMOKE_SUBJECTS:=sub-A00058775 sub-A00062842}"
: "${RUN_TIMESTAMP:=$(date -u +%Y%m%d_%H%M%S)}"

fail() {
  echo "[mipdb-smoke] ERROR: $*" >&2
  exit 1
}

OUTPUT_DIR="${MIPDB_OUTPUT_DIR:-${TIONE_OUTPUT_ROOT}/mipdb_age_preprocessing_smoke_${RUN_TIMESTAMP}}"
read -r -a smoke_subjects <<< "$MIPDB_SMOKE_SUBJECTS"
(( ${#smoke_subjects[@]} >= 2 && ${#smoke_subjects[@]} <= 3 )) \
  || fail "MIPDB_SMOKE_SUBJECTS must contain 2-3 subject IDs"

[[ -d "$MIPDB_DATA_DIR" ]] || fail "MIPDB_DATA_DIR does not exist: $MIPDB_DATA_DIR"
[[ -f "$MIPDB_AGE_METADATA" ]] || fail "age metadata not found: $MIPDB_AGE_METADATA"
[[ -f "$MIPDB_ALLOWLIST" ]] || fail "subject allowlist not found: $MIPDB_ALLOWLIST"
[[ -f "$HUADA_ROOT/scripts/preprocess_mipdb_age_windows.py" ]] || fail "preprocessing script not found"
[[ -f "$HUADA_ROOT/configs/mipdb_age_preprocessing.toml" ]] || fail "MIPDB preprocessing config not found"
mkdir -p "$TIONE_OUTPUT_ROOT" || fail "cannot create TIONE_OUTPUT_ROOT"
[[ ! -e "$OUTPUT_DIR" ]] || fail "output already exists: $OUTPUT_DIR"

echo "[mipdb-smoke] subjects=${smoke_subjects[*]} max_windows_per_recording=2"
echo "[mipdb-smoke] OUTPUT_DIR=$OUTPUT_DIR"

"$PYTHON_BIN" -u "$HUADA_ROOT/scripts/preprocess_mipdb_age_windows.py" \
  --data-dir "$MIPDB_DATA_DIR" \
  --config "$HUADA_ROOT/configs/mipdb_age_preprocessing.toml" \
  --output-dir "$OUTPUT_DIR" \
  --age-metadata "$MIPDB_AGE_METADATA" \
  --subject-allowlist "$MIPDB_ALLOWLIST" \
  --source-uri "$MIPDB_SOURCE_URI" \
  --chunk-seconds 30 \
  --smoke \
  --smoke-subjects "${smoke_subjects[@]}" \
  --smoke-max-windows-per-recording 2 || {
    status=$?
    echo "[mipdb-smoke] preprocessing failed with exit status $status" >&2
    [[ -f "$OUTPUT_DIR/preprocessing_errors.log" ]] && tail -n 100 "$OUTPUT_DIR/preprocessing_errors.log" >&2
    exit "$status"
  }

echo "[mipdb-smoke] completed successfully: $OUTPUT_DIR"
