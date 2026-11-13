#!/usr/bin/env bash
set -euo pipefail

: "${HUADA_ROOT:=/opt/ml/input}"
: "${PYTHON_BIN:=python}"
: "${MIPDB_DATA_DIR:=${HUADA_ROOT}/data/MIPDB/mipdb_download/nm000153_py}"
: "${MIPDB_AGE_METADATA:=${HUADA_ROOT}/data/MIPDB/MIPDB_PublicFile.csv}"
: "${MIPDB_ALLOWLIST:=${HUADA_ROOT}/configs/mipdb_age_20_35_subjects.txt}"
: "${MIPDB_SOURCE_URI:=cos://jianyu-1442740494/huada/data/MIPDB/mipdb_download/nm000153_py}"
: "${TIONE_OUTPUT_ROOT:=/opt/ml/output}"
: "${MIPDB_MIN_AGE:=20}"
: "${MIPDB_MAX_AGE:=35}"
: "${RUN_TIMESTAMP:=$(date -u +%Y%m%d_%H%M%S)}"

fail() {
  echo "[mipdb-full] ERROR: $*" >&2
  exit 1
}

OUTPUT_DIR="${MIPDB_OUTPUT_DIR:-${TIONE_OUTPUT_ROOT}/mipdb_age_preprocessing_${RUN_TIMESTAMP}}"

echo "[mipdb-full] HUADA_ROOT=$HUADA_ROOT"
echo "[mipdb-full] MIPDB_DATA_DIR=$MIPDB_DATA_DIR"
echo "[mipdb-full] MIPDB_AGE_METADATA=$MIPDB_AGE_METADATA"
echo "[mipdb-full] MIPDB_ALLOWLIST=$MIPDB_ALLOWLIST"
echo "[mipdb-full] MIPDB_SOURCE_URI=$MIPDB_SOURCE_URI"
echo "[mipdb-full] AGE_RANGE=[$MIPDB_MIN_AGE,$MIPDB_MAX_AGE]"
echo "[mipdb-full] OUTPUT_DIR=$OUTPUT_DIR"
echo "[mipdb-full] subject/recording counts are discovered at runtime"

[[ -d "$HUADA_ROOT" ]] || fail "HUADA_ROOT does not exist: $HUADA_ROOT"
[[ -d "$MIPDB_DATA_DIR" ]] || fail "MIPDB_DATA_DIR does not exist: $MIPDB_DATA_DIR"
[[ -f "$MIPDB_AGE_METADATA" ]] || fail "age metadata not found: $MIPDB_AGE_METADATA"
[[ -f "$MIPDB_ALLOWLIST" ]] || fail "subject allowlist not found: $MIPDB_ALLOWLIST"
[[ -f "$HUADA_ROOT/scripts/preprocess_mipdb_age_windows.py" ]] || fail "preprocessing script not found"
[[ -f "$HUADA_ROOT/configs/mipdb_age_preprocessing.toml" ]] || fail "MIPDB preprocessing config not found"

if [[ "$PYTHON_BIN" == */* ]]; then
  [[ -x "$PYTHON_BIN" ]] || fail "PYTHON_BIN is not executable: $PYTHON_BIN"
else
  command -v "$PYTHON_BIN" >/dev/null || fail "PYTHON_BIN is not on PATH"
fi

echo "[mipdb-full] Python: $("$PYTHON_BIN" --version 2>&1)"
"$PYTHON_BIN" -c \
  'import lance, mne, numpy, pyarrow, scipy; print(
      "[mipdb-full] dependencies: lance=%s mne=%s numpy=%s pyarrow=%s scipy=%s"
      % (lance.__version__, mne.__version__, numpy.__version__, pyarrow.__version__, scipy.__version__)
  )' || fail "required Python dependencies are unavailable"

mkdir -p "$TIONE_OUTPUT_ROOT" || fail "cannot create TIONE_OUTPUT_ROOT: $TIONE_OUTPUT_ROOT"
[[ -w "$TIONE_OUTPUT_ROOT" ]] || fail "TIONE_OUTPUT_ROOT is not writable: $TIONE_OUTPUT_ROOT"
[[ ! -e "$OUTPUT_DIR" ]] || fail "output already exists: $OUTPUT_DIR"

args=(
  "$HUADA_ROOT/scripts/preprocess_mipdb_age_windows.py"
  --data-dir "$MIPDB_DATA_DIR"
  --config "$HUADA_ROOT/configs/mipdb_age_preprocessing.toml"
  --output-dir "$OUTPUT_DIR"
  --age-metadata "$MIPDB_AGE_METADATA"
  --subject-allowlist "$MIPDB_ALLOWLIST"
  --source-uri "$MIPDB_SOURCE_URI"
  --chunk-seconds 30
  --min-age "$MIPDB_MIN_AGE"
  --max-age "$MIPDB_MAX_AGE"
)

"$PYTHON_BIN" -u "${args[@]}" || {
  status=$?
  echo "[mipdb-full] preprocessing failed with exit status $status" >&2
  if [[ -f "$OUTPUT_DIR/preprocessing_errors.log" ]]; then
    tail -n 100 "$OUTPUT_DIR/preprocessing_errors.log" >&2
  fi
  exit "$status"
}

echo "[mipdb-full] completed successfully: $OUTPUT_DIR"
