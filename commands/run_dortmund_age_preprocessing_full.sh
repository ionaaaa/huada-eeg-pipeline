#!/usr/bin/env bash
set -euo pipefail

: "${HUADA_ROOT:=/opt/ml/input}"
: "${PYTHON_BIN:=python}"
: "${DORTMUND_DATA_DIR:=/opt/ml/input/data/Dortmund_Vital_Study}"
: "${DORTMUND_SOURCE_URI:=cos://jianyu-1442740494/huada/data/Dortmund_Vital_Study}"
: "${TIONE_OUTPUT_ROOT:=/opt/ml/output}"

fail() {
  echo "[dortmund-full] ERROR: $*" >&2
  exit 1
}

echo "[dortmund-full] HUADA_ROOT=$HUADA_ROOT"
echo "[dortmund-full] DORTMUND_DATA_DIR=$DORTMUND_DATA_DIR"
echo "[dortmund-full] TIONE_OUTPUT_ROOT=$TIONE_OUTPUT_ROOT"
echo "[dortmund-full] DORTMUND_SOURCE_URI=$DORTMUND_SOURCE_URI"

[[ -d "$HUADA_ROOT" ]] || fail "HUADA_ROOT does not exist: $HUADA_ROOT"
[[ -d "$DORTMUND_DATA_DIR" ]] || fail "DORTMUND_DATA_DIR does not exist: $DORTMUND_DATA_DIR"
[[ -f "$HUADA_ROOT/scripts/preprocess_dortmund_age_windows.py" ]] || fail "preprocessing script not found below HUADA_ROOT"
[[ -f "$HUADA_ROOT/configs/huada_resting_preprocessing.toml" ]] || fail "shared preprocessing config not found below HUADA_ROOT"
if [[ "$PYTHON_BIN" == */* ]]; then
  [[ -x "$PYTHON_BIN" ]] || fail "PYTHON_BIN is not executable: $PYTHON_BIN"
else
  command -v "$PYTHON_BIN" >/dev/null || fail "PYTHON_BIN is not on PATH: $PYTHON_BIN"
fi


echo "[dortmund-full] Python: $("$PYTHON_BIN" --version 2>&1)"
"$PYTHON_BIN" -c 'import lance, mne, numpy, pyarrow, scipy; print("[dortmund-full] dependencies: lance=%s mne=%s numpy=%s pyarrow=%s scipy=%s" % (lance.__version__, mne.__version__, numpy.__version__, pyarrow.__version__, scipy.__version__))' \
  || fail "required Python dependencies are unavailable in PYTHON_BIN"

mkdir -p "$TIONE_OUTPUT_ROOT" || fail "cannot create TIONE_OUTPUT_ROOT: $TIONE_OUTPUT_ROOT (use /opt/ml/output on Ti-ONE)"
[[ -w "$TIONE_OUTPUT_ROOT" ]] || fail "TIONE_OUTPUT_ROOT is not writable: $TIONE_OUTPUT_ROOT (use /opt/ml/output on Ti-ONE)"

STAMP="$(date -u +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${TIONE_OUTPUT_ROOT}/dortmund_age_preprocessing_full_${STAMP}"
[[ ! -e "$OUTPUT_DIR" ]] || fail "timestamped output already exists: $OUTPUT_DIR"
echo "[dortmund-full] OUTPUT_DIR=$OUTPUT_DIR"

"$PYTHON_BIN" -u "$HUADA_ROOT/scripts/preprocess_dortmund_age_windows.py" \
  --data-dir "$DORTMUND_DATA_DIR" \
  --config "$HUADA_ROOT/configs/huada_resting_preprocessing.toml" \
  --output-dir "$OUTPUT_DIR" \
  --source-uri "$DORTMUND_SOURCE_URI" \
  --chunk-seconds 30 \
  --expected-subjects 214 \
  --expected-recordings 428 || {
    status=$?
    echo "[dortmund-full] preprocessing failed with exit status $status" >&2
    if [[ -f "$OUTPUT_DIR/preprocessing_errors.log" ]]; then
      echo "[dortmund-full] tail of preprocessing_errors.log:" >&2
      tail -n 100 "$OUTPUT_DIR/preprocessing_errors.log" >&2
    fi
    exit "$status"
  }

echo "[dortmund-full] completed successfully: $OUTPUT_DIR"
