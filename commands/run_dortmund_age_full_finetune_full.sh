#!/usr/bin/env bash
set -euo pipefail

: "${HUADA_ROOT:=/mnt/cfs/jianyu/huada-eeg-pipeline}"
: "${INPUT_LANCE:=/mnt/cfs/jianyu/huada-eeg-pipeline/data/model_tokens_full/fbmae_tokens/20261103_112918/dortmund_fbmae_tokens.lance}"
: "${PRETRAINED_50M_CKPT:=${HUADA_ROOT}/data/checkpoint/pretrain_checkpoint_4.pt}"
: "${SPLIT_MANIFEST:=${HUADA_ROOT}/downstream_output/dortmund_age_split_seed42/subject_split_manifest.csv}"
: "${AUDIT_SUMMARY:=${HUADA_ROOT}/audit_dortmund_fbmae_tokens/20261103_124436/audit_summary.json}"
: "${OUTPUT_ROOT:=${HUADA_ROOT}/downstream_output}"
: "${NPROC_PER_NODE:=2}"
: "${TRAIN_SEED:=42}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_OUTPUT="${OUTPUT_ROOT}/dortmund_age_full_finetune_${STAMP}"

cd "${HUADA_ROOT}"
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
  scripts/downstream_50m/run_dortmund_age_full_finetune.py \
  --input-lance "${INPUT_LANCE}" \
  --checkpoint "${PRETRAINED_50M_CKPT}" \
  --split-manifest "${SPLIT_MANIFEST}" \
  --audit-summary "${AUDIT_SUMMARY}" \
  --output-dir "${RUN_OUTPUT}" \
  --epochs 100 --batch-size 32 --accumulation-steps 1 \
  --backbone-lr 1e-5 --head-lr 1e-4 --weight-decay 0.05 \
  --gradient-clip-norm 1.0 --scheduler linear_warmup_cosine_decay \
  --warmup-ratio 0.1 --early-stopping-patience 15 \
  --probe-layer-idx 8 --pooling mean --train-seed "${TRAIN_SEED}" \
  --amp-mode bf16 --num-workers 4
