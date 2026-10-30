#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1

HUADA_ROOT="${HUADA_ROOT:-/opt/ml/input}"
PYTHON_BIN="${PYTHON_BIN:-python}"
INPUT_LANCE="${INPUT_LANCE:-/opt/ml/input/data/model_tokens_full/20261030_140104/huada_model_tokens_8sub.lance}"
PRETRAINED_50M_CKPT="${PRETRAINED_50M_CKPT:-/opt/ml/input/data/checkpoint/pretrain_checkpoint_4.pt}"
SWEEP_CONFIG="${SWEEP_CONFIG:-/opt/ml/input/configs/50m_sex_cv4_sweep.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${TIONE_OUTPUT_ROOT:-/opt/ml/input/downstream_outputs}}"
FEATURE_CACHE_ROOT="${FEATURE_CACHE_ROOT:-/opt/ml/input/data/downstream_feature_cache}"
FEATURE_CACHE_POLICY="${FEATURE_CACHE_POLICY:-auto}"
FEATURE_CACHE_BATCH_SIZE="${FEATURE_CACHE_BATCH_SIZE:-8}"
CACHE_LOG_EVERY_BATCHES="${CACHE_LOG_EVERY_BATCHES:-5}"
TRAIN_LOG_EVERY_EPOCHS="${TRAIN_LOG_EVERY_EPOCHS:-5}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-64}"
TRAIN_SEED="${TRAIN_SEED:-42}"
SPLIT_SEED="${SPLIT_SEED:-42}"

test -d "$HUADA_ROOT"
test -d "$INPUT_LANCE"
test -f "$PRETRAINED_50M_CKPT"
test -f "$SWEEP_CONFIG"
test -d "$OUTPUT_ROOT"
mkdir -p "$FEATURE_CACHE_ROOT"

cd "$HUADA_ROOT"
STAMP="$(date +%Y%m%d_%H%M%S)"
SWEEP_OUTPUT="${OUTPUT_ROOT%/}/50m_sex_cv4_sweep_${STAMP}"
test ! -e "$SWEEP_OUTPUT"

"$PYTHON_BIN" -u -c 'import torch,lance,pyarrow; print({"torch":torch.__version__,"cuda":torch.cuda.is_available(),"lance":lance.__version__,"pyarrow":pyarrow.__version__})'

"$PYTHON_BIN" -u scripts/downstream_50m/run_sex_cv4_sweep.py \
  --input-lance "$INPUT_LANCE" \
  --checkpoint "$PRETRAINED_50M_CKPT" \
  --sweep-config "$SWEEP_CONFIG" \
  --output-dir "$SWEEP_OUTPUT" \
  --feature-cache-root "$FEATURE_CACHE_ROOT" \
  --feature-cache-policy "$FEATURE_CACHE_POLICY" \
  --feature-cache-batch-size "$FEATURE_CACHE_BATCH_SIZE" \
  --cache-log-every-batches "$CACHE_LOG_EVERY_BATCHES" \
  --train-log-every-epochs "$TRAIN_LOG_EVERY_EPOCHS" \
  --log-level "$LOG_LEVEL" \
  --device cuda \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --seed "$TRAIN_SEED" \
  --split-seed "$SPLIT_SEED"

printf 'Sweep output: %s\n' "$SWEEP_OUTPUT"
