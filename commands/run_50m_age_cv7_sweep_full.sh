#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1

HUADA_ROOT="${HUADA_ROOT:-/opt/ml/input}"
PYTHON_BIN="${PYTHON_BIN:-python}"
INPUT_LANCE="${INPUT_LANCE:-/opt/ml/input/data/model_tokens_full/20261030_140104/huada_model_tokens_8sub.lance}"
PRETRAINED_50M_CKPT="${PRETRAINED_50M_CKPT:-/opt/ml/input/data/checkpoint/pretrain_checkpoint_4.pt}"
SWEEP_CONFIG="${SWEEP_CONFIG:-/opt/ml/input/configs/50m_age_cv7_sweep.json}"
FEATURE_CACHE_ROOT="${FEATURE_CACHE_ROOT:-/opt/ml/input/data/downstream_feature_cache}"
FEATURE_CACHE_BUILD_ROOT="${FEATURE_CACHE_BUILD_ROOT:-/tmp/huada_downstream_feature_cache_build}"
FEATURE_CACHE_POLICY="${FEATURE_CACHE_POLICY:-auto}"
FEATURE_CACHE_BATCH_SIZE="${FEATURE_CACHE_BATCH_SIZE:-8}"
TIONE_OUTPUT_ROOT="${TIONE_OUTPUT_ROOT:-/opt/ml/input/downstream_outputs}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-64}"
TRAIN_SEED="${TRAIN_SEED:-42}"
SPLIT_SEED="${SPLIT_SEED:-42}"
TRAIN_LOG_EVERY_EPOCHS="${TRAIN_LOG_EVERY_EPOCHS:-5}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

test -d "$HUADA_ROOT"
test -d "$INPUT_LANCE"
test -f "$PRETRAINED_50M_CKPT"
test -f "$SWEEP_CONFIG"
mkdir -p "$TIONE_OUTPUT_ROOT" "$FEATURE_CACHE_ROOT" "$FEATURE_CACHE_BUILD_ROOT"

cd "$HUADA_ROOT"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${TIONE_OUTPUT_ROOT%/}/50m_age_cv7_sweep_${STAMP}"
test ! -e "$OUTPUT_DIR"

LANCE_AUTH=()
if [[ -n "${STORAGE_OPTIONS:-}" ]]; then
  test -f "$STORAGE_OPTIONS"
  LANCE_AUTH=(--storage-options "$STORAGE_OPTIONS")
fi

"$PYTHON_BIN" -u scripts/downstream_50m/run_age_cv7_sweep.py \
  --input-lance "$INPUT_LANCE" \
  "${LANCE_AUTH[@]}" \
  --checkpoint "$PRETRAINED_50M_CKPT" \
  --sweep-config "$SWEEP_CONFIG" \
  --feature-cache-root "$FEATURE_CACHE_ROOT" \
  --feature-cache-build-root "$FEATURE_CACHE_BUILD_ROOT" \
  --feature-cache-policy "$FEATURE_CACHE_POLICY" \
  --feature-cache-batch-size "$FEATURE_CACHE_BATCH_SIZE" \
  --output-dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --seed "$TRAIN_SEED" \
  --split-seed "$SPLIT_SEED" \
  --train-log-every-epochs "$TRAIN_LOG_EVERY_EPOCHS" \
  --log-level "$LOG_LEVEL"

printf 'Age CV7 output: %s\n' "$OUTPUT_DIR"
