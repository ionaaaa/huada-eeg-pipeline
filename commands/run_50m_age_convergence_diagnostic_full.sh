#!/usr/bin/env bash
set -euo pipefail

: "${HUADA_ROOT:?Set HUADA_ROOT to the mounted project directory}"
: "${INPUT_LANCE:?Set INPUT_LANCE to the mounted token Lance directory}"
: "${MEAN_FEATURE_CACHE_LANCE:?Set MEAN_FEATURE_CACHE_LANCE to mean_features.lance}"
: "${FEATURE_CACHE_MANIFEST:?Set FEATURE_CACHE_MANIFEST to cache_manifest.json}"
: "${TIONE_OUTPUT_ROOT:?Set TIONE_OUTPUT_ROOT to a persistent output directory}"

PYTHON_BIN="${PYTHON_BIN:-python}"
FOLD="${FOLD:-0}"
SPLIT_SEED="${SPLIT_SEED:-42}"
TRAIN_SEED="${TRAIN_SEED:-42}"
HEAD_LR="${HEAD_LR:-0.0003}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MINI_SAMPLES_PER_SUBJECT="${MINI_SAMPLES_PER_SUBJECT:-4}"
MINI_MAX_EPOCHS="${MINI_MAX_EPOCHS:-1000}"
FULL_MAX_EPOCHS="${FULL_MAX_EPOCHS:-300}"
STEP_LOG_INTERVAL="${STEP_LOG_INTERVAL:-1}"
DEVICE="${DEVICE:-cuda}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
export PYTHONUNBUFFERED=1

test -d "$HUADA_ROOT"
test -d "$INPUT_LANCE"
test -d "$MEAN_FEATURE_CACHE_LANCE"
test -f "$FEATURE_CACHE_MANIFEST"
mkdir -p "$TIONE_OUTPUT_ROOT"

cd "$HUADA_ROOT"
test -f scripts/downstream_50m/diagnose_age_head_convergence.py

STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${TIONE_OUTPUT_ROOT%/}/50m_age_convergence_diagnostic_${STAMP}"
test ! -e "$OUTPUT_DIR"

"$PYTHON_BIN" -u scripts/downstream_50m/diagnose_age_head_convergence.py \
  --input-lance "$INPUT_LANCE" \
  --feature-cache-lance "$MEAN_FEATURE_CACHE_LANCE" \
  --feature-cache-manifest "$FEATURE_CACHE_MANIFEST" \
  --output-dir "$OUTPUT_DIR" \
  --fold "$FOLD" \
  --split-seed "$SPLIT_SEED" \
  --train-seed "$TRAIN_SEED" \
  --head-lr "$HEAD_LR" \
  --weight-decay "$WEIGHT_DECAY" \
  --batch-size "$BATCH_SIZE" \
  --mini-samples-per-subject "$MINI_SAMPLES_PER_SUBJECT" \
  --mini-max-epochs "$MINI_MAX_EPOCHS" \
  --full-max-epochs "$FULL_MAX_EPOCHS" \
  --step-log-interval "$STEP_LOG_INTERVAL" \
  --device "$DEVICE" \
  --log-level "$LOG_LEVEL"

printf 'Age convergence diagnostic output: %s\n' "$OUTPUT_DIR"
