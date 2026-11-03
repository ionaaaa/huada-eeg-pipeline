#!/usr/bin/env bash
set -euo pipefail

: "${HUADA_ROOT:=/mnt/cfs/jianyu/huada-eeg-pipeline}"
: "${PYTHON_BIN:=python}"
: "${INPUT_LANCE:=/mnt/cfs/jianyu/huada-eeg-pipeline/data/model_tokens_full/fbmae_tokens/20261103_112918/dortmund_fbmae_tokens.lance}"
: "${PRETRAINED_50M_CKPT:=${HUADA_ROOT}/data/checkpoint/pretrain_checkpoint_4.pt}"
: "${SPLIT_MANIFEST:=${HUADA_ROOT}/downstream_output/dortmund_age_split_seed42/subject_split_manifest.csv}"
: "${AUDIT_SUMMARY:=${HUADA_ROOT}/audit_dortmund_fbmae_tokens/20261103_124436/audit_summary.json}"
: "${OUTPUT_ROOT:=${HUADA_ROOT}/downstream_output}"
: "${TRAIN_SEED:=42}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_OUTPUT="${OUTPUT_ROOT}/dortmund_age_full_finetune_smoke_${STAMP}"

cd "${HUADA_ROOT}"
"${PYTHON_BIN}" scripts/downstream_50m/run_dortmund_age_full_finetune.py \
  --input-lance "${INPUT_LANCE}" \
  --checkpoint "${PRETRAINED_50M_CKPT}" \
  --split-manifest "${SPLIT_MANIFEST}" \
  --audit-summary "${AUDIT_SUMMARY}" \
  --output-dir "${RUN_OUTPUT}" \
  --epochs 3 --batch-size 4 --accumulation-steps 1 \
  --backbone-lr 1e-5 --head-lr 1e-4 --weight-decay 0.05 \
  --gradient-clip-norm 1.0 --scheduler linear_warmup_cosine_decay \
  --warmup-ratio 0.1 --early-stopping-patience 3 \
  --probe-layer-idx 8 --pooling mean --train-seed "${TRAIN_SEED}" \
  --amp-mode none --num-workers 0 --smoke --skip-test
