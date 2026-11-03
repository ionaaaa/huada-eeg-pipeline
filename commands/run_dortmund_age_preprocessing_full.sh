#!/usr/bin/env bash
set -euo pipefail

: "${HUADA_ROOT:=/opt/ml/input}"
: "${PYTHON_BIN:=python}"
: "${DORTMUND_DATA_DIR:=/opt/ml/input/data/Dortmund_Vital_Study}"
: "${DORTMUND_SOURCE_URI:=cos://jianyu-1442740494/huada/data/Dortmund_Vital_Study}"
: "${TIONE_OUTPUT_ROOT:=/opt/ml/output}"

test -d "$HUADA_ROOT"
test -d "$DORTMUND_DATA_DIR"
if [[ "$PYTHON_BIN" == */* ]]; then
  test -x "$PYTHON_BIN"
else
  command -v "$PYTHON_BIN" >/dev/null
fi

STAMP="$(date -u +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${TIONE_OUTPUT_ROOT}/dortmund_age_preprocessing_full_${STAMP}"
test ! -e "$OUTPUT_DIR"

exec "$PYTHON_BIN" "$HUADA_ROOT/scripts/preprocess_dortmund_age_windows.py" \
  --data-dir "$DORTMUND_DATA_DIR" \
  --config "$HUADA_ROOT/configs/huada_resting_preprocessing.toml" \
  --output-dir "$OUTPUT_DIR" \
  --source-uri "$DORTMUND_SOURCE_URI" \
  --chunk-seconds 30 \
  --expected-subjects 214 \
  --expected-recordings 428
