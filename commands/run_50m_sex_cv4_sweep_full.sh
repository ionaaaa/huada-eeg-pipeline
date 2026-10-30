#!/usr/bin/env bash
set -euo pipefail

HUADA_ROOT="${HUADA_ROOT:-/opt/ml/input}"
PYTHON_BIN="${PYTHON_BIN:-python}"
INPUT_LANCE="${INPUT_LANCE:-/opt/ml/input/data/model_tokens_full/20261030_140104/huada_model_tokens_8sub.lance}"
PRETRAINED_50M_CKPT="${PRETRAINED_50M_CKPT:-/opt/ml/input/data/checkpoint/pretrain_checkpoint_4.pt}"
SWEEP_CONFIG="${SWEEP_CONFIG:-/opt/ml/input/configs/50m_sex_cv4_sweep.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/opt/ml/input/downstream_outputs}"

test -d "$HUADA_ROOT"
test -d "$INPUT_LANCE"
test -f "$PRETRAINED_50M_CKPT"
test -f "$SWEEP_CONFIG"
test -d "$OUTPUT_ROOT"

cd "$HUADA_ROOT"
STAMP="$(date +%Y%m%d_%H%M%S)"
SWEEP_OUTPUT="${OUTPUT_ROOT%/}/50m_sex_cv4_sweep_${STAMP}"
test ! -e "$SWEEP_OUTPUT"

"$PYTHON_BIN" scripts/downstream_50m/run_sex_cv4_sweep.py \
  --input-lance "$INPUT_LANCE" \
  --checkpoint "$PRETRAINED_50M_CKPT" \
  --sweep-config "$SWEEP_CONFIG" \
  --output-dir "$SWEEP_OUTPUT" \
  --device cuda \
  --epochs 5 \
  --batch-size 64 \
  --seed 42 \
  --split-seed 42

printf 'Sweep output: %s\n' "$SWEEP_OUTPUT"
