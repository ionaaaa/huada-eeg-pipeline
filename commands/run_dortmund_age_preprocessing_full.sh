#!/usr/bin/env bash
set -euo pipefail

: "${HUADA_ROOT:=/opt/ml/input}"
: "${PYTHON_BIN:=python}"
: "${DORTMUND_DATA_DIR:=/opt/ml/input/data/Dortmund_Vital_Study}"
: "${DORTMUND_SOURCE_URI:=cos://jianyu-1442740494/huada/data/Dortmund_Vital_Study}"
: "${TIONE_OUTPUT_ROOT:=/opt/ml/output}"

: "${DORTMUND_MIN_AGE:=20}"
: "${DORTMUND_MAX_AGE:=35}"
: "${DORTMUND_COHORT_ID:=age${DORTMUND_MIN_AGE}_${DORTMUND_MAX_AGE}_ses1_pre_profile_a}"
: "${RUN_TIMESTAMP:=$(date -u +%Y%m%d_%H%M%S)}"

fail() {
  echo "[dortmund-full] ERROR: $*" >&2
  exit 1
}

# Preserve the original 20-35 cohort defaults. Any expanded/restricted cohort
# must provide explicit count gates obtained from an inventory-only preflight.
if [[ -z "${DORTMUND_EXPECTED_SUBJECTS:-}" ]]; then
  if [[ "$DORTMUND_MIN_AGE" == "20" && "$DORTMUND_MAX_AGE" == "35" ]]; then
    DORTMUND_EXPECTED_SUBJECTS=214
  else
    fail "DORTMUND_EXPECTED_SUBJECTS must be set for non-default age range"
  fi
fi

if [[ -z "${DORTMUND_EXPECTED_RECORDINGS:-}" ]]; then
  if [[ "$DORTMUND_MIN_AGE" == "20" && "$DORTMUND_MAX_AGE" == "35" ]]; then
    DORTMUND_EXPECTED_RECORDINGS=428
  else
    fail "DORTMUND_EXPECTED_RECORDINGS must be set for non-default age range"
  fi
fi

OUTPUT_DIR="${DORTMUND_OUTPUT_DIR:-${TIONE_OUTPUT_ROOT}/dortmund_age_preprocessing_${DORTMUND_COHORT_ID}_${RUN_TIMESTAMP}}"

echo "[dortmund-full] HUADA_ROOT=$HUADA_ROOT"
echo "[dortmund-full] DORTMUND_DATA_DIR=$DORTMUND_DATA_DIR"
echo "[dortmund-full] DORTMUND_SOURCE_URI=$DORTMUND_SOURCE_URI"
echo "[dortmund-full] TIONE_OUTPUT_ROOT=$TIONE_OUTPUT_ROOT"
echo "[dortmund-full] DORTMUND_COHORT_ID=$DORTMUND_COHORT_ID"
echo "[dortmund-full] AGE_RANGE=[$DORTMUND_MIN_AGE,$DORTMUND_MAX_AGE]"
echo "[dortmund-full] EXPECTED_SUBJECTS=$DORTMUND_EXPECTED_SUBJECTS"
echo "[dortmund-full] EXPECTED_RECORDINGS=$DORTMUND_EXPECTED_RECORDINGS"
echo "[dortmund-full] OUTPUT_DIR=$OUTPUT_DIR"

[[ -d "$HUADA_ROOT" ]] || fail "HUADA_ROOT does not exist: $HUADA_ROOT"
[[ -d "$DORTMUND_DATA_DIR" ]] || fail "DORTMUND_DATA_DIR does not exist: $DORTMUND_DATA_DIR"
[[ -f "$HUADA_ROOT/scripts/preprocess_dortmund_age_windows.py" ]] \
  || fail "preprocessing script not found"
[[ -f "$HUADA_ROOT/configs/huada_resting_preprocessing.toml" ]] \
  || fail "shared preprocessing config not found"

if [[ "$PYTHON_BIN" == */* ]]; then
  [[ -x "$PYTHON_BIN" ]] || fail "PYTHON_BIN is not executable: $PYTHON_BIN"
else
  command -v "$PYTHON_BIN" >/dev/null || fail "PYTHON_BIN is not on PATH"
fi

echo "[dortmund-full] Python: $("$PYTHON_BIN" --version 2>&1)"
"$PYTHON_BIN" -c \
  'import lance, mne, numpy, pyarrow, scipy; print(
      "[dortmund-full] dependencies: lance=%s mne=%s numpy=%s pyarrow=%s scipy=%s"
      % (
          lance.__version__,
          mne.__version__,
          numpy.__version__,
          pyarrow.__version__,
          scipy.__version__,
      )
  )' || fail "required Python dependencies are unavailable"

mkdir -p "$TIONE_OUTPUT_ROOT" \
  || fail "cannot create TIONE_OUTPUT_ROOT: $TIONE_OUTPUT_ROOT"

[[ -w "$TIONE_OUTPUT_ROOT" ]] \
  || fail "TIONE_OUTPUT_ROOT is not writable: $TIONE_OUTPUT_ROOT"

[[ ! -e "$OUTPUT_DIR" ]] \
  || fail "output already exists: $OUTPUT_DIR"

args=(
  "$HUADA_ROOT/scripts/preprocess_dortmund_age_windows.py"
  --data-dir "$DORTMUND_DATA_DIR"
  --config "$HUADA_ROOT/configs/huada_resting_preprocessing.toml"
  --output-dir "$OUTPUT_DIR"
  --source-uri "$DORTMUND_SOURCE_URI"
  --chunk-seconds 30
  --min-age "$DORTMUND_MIN_AGE"
  --max-age "$DORTMUND_MAX_AGE"
  --cohort-id "$DORTMUND_COHORT_ID"
  --expected-subjects "$DORTMUND_EXPECTED_SUBJECTS"
  --expected-recordings "$DORTMUND_EXPECTED_RECORDINGS"
)

"$PYTHON_BIN" -u "${args[@]}" || {
  status=$?
  echo "[dortmund-full] preprocessing failed with exit status $status" >&2

  if [[ -f "$OUTPUT_DIR/preprocessing_errors.log" ]]; then
    echo "[dortmund-full] tail of preprocessing_errors.log:" >&2
    tail -n 100 "$OUTPUT_DIR/preprocessing_errors.log" >&2
  fi

  exit "$status"
}

echo "[dortmund-full] completed successfully: $OUTPUT_DIR"
