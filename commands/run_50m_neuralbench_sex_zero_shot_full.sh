#!/usr/bin/env bash
set -euo pipefail

: "${HUADA_ROOT:?Set HUADA_ROOT to the mounted project directory}"
: "${INPUT_LANCE:?Set INPUT_LANCE to the mounted model-token Lance directory}"
: "${NEURALBENCH_SEX_CKPT:?Set NEURALBENCH_SEX_CKPT to the mounted checkpoint file}"
: "${EXPECTED_CKPT_SHA256:?Set EXPECTED_CKPT_SHA256 to the authoritative digest}"
: "${TIONE_OUTPUT_ROOT:?Set TIONE_OUTPUT_ROOT to a persistent output directory}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-0}"
LOG_EVERY_BATCHES="${LOG_EVERY_BATCHES:-5}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
export PYTHONUNBUFFERED=1

test -d "$HUADA_ROOT"
test -d "$INPUT_LANCE"
test -f "$NEURALBENCH_SEX_CKPT"
mkdir -p "$TIONE_OUTPUT_ROOT"

cd "$HUADA_ROOT"
test -f "scripts/downstream_50m/run_neuralbench_sex_zero_shot.py"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${TIONE_OUTPUT_ROOT%/}/50m_neuralbench_sex_zero_shot_${STAMP}"
test ! -e "$OUTPUT_DIR"

"$PYTHON_BIN" -u scripts/downstream_50m/run_neuralbench_sex_zero_shot.py \
  --input-lance "$INPUT_LANCE" \
  --checkpoint "$NEURALBENCH_SEX_CKPT" \
  --expected-checkpoint-sha256 "$EXPECTED_CKPT_SHA256" \
  --output-dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --log-every-batches "$LOG_EVERY_BATCHES" \
  --log-level "$LOG_LEVEL"

printf 'Zero-shot output: %s\n' "$OUTPUT_DIR"
