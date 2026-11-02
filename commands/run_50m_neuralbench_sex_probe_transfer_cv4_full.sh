#!/usr/bin/env bash
set -euo pipefail

: "${HUADA_ROOT:?Set HUADA_ROOT to the mounted project directory}"
: "${INPUT_LANCE:?Set INPUT_LANCE to the mounted token Lance directory}"
: "${NEURALBENCH_SEX_CKPT:?Set NEURALBENCH_SEX_CKPT to the mounted checkpoint file}"
: "${EXPECTED_CKPT_SHA256:?Set EXPECTED_CKPT_SHA256 to the authoritative digest}"
: "${FEATURE_CACHE_ROOT:?Set FEATURE_CACHE_ROOT to a persistent cache directory}"
: "${TIONE_OUTPUT_ROOT:?Set TIONE_OUTPUT_ROOT to a persistent output directory}"

PYTHON_BIN="${PYTHON_BIN:-python}"
FEATURE_CACHE_POLICY="${FEATURE_CACHE_POLICY:-auto}"
FEATURE_CACHE_BATCH_SIZE="${FEATURE_CACHE_BATCH_SIZE:-8}"
CACHE_LOG_EVERY_BATCHES="${CACHE_LOG_EVERY_BATCHES:-5}"
ZERO_SHOT_RESULTS_DIR="${ZERO_SHOT_RESULTS_DIR:-}"
HEAD_LR="${HEAD_LR:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0}"
EPOCHS="${EPOCHS:-20}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-5}"
BATCH_SIZE="${BATCH_SIZE:-64}"
TRAIN_SEED="${TRAIN_SEED:-42}"
SPLIT_SEED="${SPLIT_SEED:-42}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-0}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
export PYTHONUNBUFFERED=1

test -d "$HUADA_ROOT"
test -d "$INPUT_LANCE"
test -f "$NEURALBENCH_SEX_CKPT"
mkdir -p "$FEATURE_CACHE_ROOT" "$TIONE_OUTPUT_ROOT"
if [[ -n "$ZERO_SHOT_RESULTS_DIR" ]]; then
  test -d "$ZERO_SHOT_RESULTS_DIR"
fi

cd "$HUADA_ROOT"
ENTRY="scripts/downstream_50m/run_neuralbench_sex_probe_transfer_cv4.py"
test -f "$ENTRY"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${TIONE_OUTPUT_ROOT%/}/50m_neuralbench_sex_probe_transfer_cv4_${STAMP}"
test ! -e "$OUTPUT_DIR"

OPTIONAL_ARGS=()
if [[ -n "$ZERO_SHOT_RESULTS_DIR" ]]; then
  OPTIONAL_ARGS=(--zero-shot-results-dir "$ZERO_SHOT_RESULTS_DIR")
fi

"$PYTHON_BIN" -u "$ENTRY" \
  --input-lance "$INPUT_LANCE" \
  --checkpoint "$NEURALBENCH_SEX_CKPT" \
  --expected-checkpoint-sha256 "$EXPECTED_CKPT_SHA256" \
  --feature-cache-root "$FEATURE_CACHE_ROOT" \
  --feature-cache-policy "$FEATURE_CACHE_POLICY" \
  --feature-cache-batch-size "$FEATURE_CACHE_BATCH_SIZE" \
  --cache-log-every-batches "$CACHE_LOG_EVERY_BATCHES" \
  --output-dir "$OUTPUT_DIR" \
  --head-lr "$HEAD_LR" \
  --weight-decay "$WEIGHT_DECAY" \
  --epochs "$EPOCHS" \
  --early-stopping-patience "$EARLY_STOPPING_PATIENCE" \
  --batch-size "$BATCH_SIZE" \
  --seed "$TRAIN_SEED" \
  --split-seed "$SPLIT_SEED" \
  --device "$DEVICE" \
  --num-workers "$NUM_WORKERS" \
  --log-level "$LOG_LEVEL" \
  "${OPTIONAL_ARGS[@]}"

printf 'Probe-transfer CV4 output: %s\n' "$OUTPUT_DIR"
