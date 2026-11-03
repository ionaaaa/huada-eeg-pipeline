#!/usr/bin/env bash
set -euo pipefail

: "${HUADA_ROOT:=/mnt/cfs/jianyu/huada-eeg-pipeline}"
: "${PYTHON_BIN:=python}"
: "${INPUT_LANCE:=/mnt/cfs/jianyu/huada-eeg-pipeline/data/model_tokens_full/fbmae_tokens/20261103_112918/dortmund_fbmae_tokens.lance}"
: "${TRAIN_RATIO:=0.8}"
: "${VALIDATION_RATIO:=0.1}"
: "${TEST_RATIO:=0.1}"
: "${SPLIT_SEED:=42}"
: "${SPLIT_OUTPUT:=${HUADA_ROOT}/downstream_output/dortmund_age_split_seed${SPLIT_SEED}}"

cd "${HUADA_ROOT}"
"${PYTHON_BIN}" scripts/downstream_50m/create_dortmund_age_split.py \
  --input-lance "${INPUT_LANCE}" \
  --output-dir "${SPLIT_OUTPUT}" \
  --train-ratio "${TRAIN_RATIO}" \
  --validation-ratio "${VALIDATION_RATIO}" \
  --test-ratio "${TEST_RATIO}" \
  --split-seed "${SPLIT_SEED}"
