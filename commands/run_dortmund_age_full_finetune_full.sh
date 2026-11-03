#!/usr/bin/env bash
set -euo pipefail

: "${HUADA_ROOT:=/opt/ml/cfs_input}"
: "${COS_ROOT:=/opt/ml/cos_input}"
: "${INPUT_LANCE:=${HUADA_ROOT}/data/model_tokens_full/fbmae_tokens/20261103_112918/dortmund_fbmae_tokens.lance}"
: "${SPLIT_MANIFEST:=${HUADA_ROOT}/downstream_output/dortmund_age_split_seed42_20261103_153000/subject_split_manifest.csv}"
: "${AUDIT_SUMMARY:=${HUADA_ROOT}/audit_dortmund_fbmae_tokens/20261103_124436/audit_summary.json}"
: "${OUTPUT_ROOT:=${HUADA_ROOT}/downstream_output}"
: "${NPROC_PER_NODE:=2}"
: "${TRAIN_SEED:=42}"
: "${EXPECTED_CKPT_SHA256:=97335b696b3ae9138dcb51c736f49ee1c6008fdc22fc42f13ea9a5301452f36e}"
: "${FULL_EPOCHS:=100}"
: "${PER_DEVICE_BATCH_SIZE:=2}"
: "${ACCUMULATION_STEPS:=16}"
: "${NUM_WORKERS:=4}"
: "${BACKBONE_LR:=1e-5}"
: "${HEAD_LR:=1e-4}"

if [[ -z "${PRETRAINED_50M_CKPT:-}" ]]; then
  if [[ -f "${HUADA_ROOT}/data/checkpoint/pretrain_checkpoint_4.pt" ]]; then
    PRETRAINED_50M_CKPT="${HUADA_ROOT}/data/checkpoint/pretrain_checkpoint_4.pt"
  else
    PRETRAINED_50M_CKPT="${COS_ROOT}/checkpoint/pretrain_checkpoint_4.pt"
  fi
fi
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_OUTPUT="${OUTPUT_ROOT}/dortmund_age_full_finetune_${STAMP}"

cd "${HUADA_ROOT}"
export PYTHONPATH="${HUADA_ROOT}/scripts/downstream_50m:${PYTHONPATH:-}"
test -d "${INPUT_LANCE}"
test -f "${SPLIT_MANIFEST}"
test -f "${PRETRAINED_50M_CKPT}"
printf '%s  %s\n' "${EXPECTED_CKPT_SHA256}" "${PRETRAINED_50M_CKPT}" | sha256sum -c -

AUDIT_ARGS=(--audit-policy record_only)
if [[ -f "${AUDIT_SUMMARY}" ]]; then
  AUDIT_ARGS+=(--audit-summary "${AUDIT_SUMMARY}")
fi

EFFECTIVE_GLOBAL_BATCH_SIZE=$((PER_DEVICE_BATCH_SIZE * NPROC_PER_NODE * ACCUMULATION_STEPS))
echo "mode=full_finetune"
echo "epochs=${FULL_EPOCHS}"
echo "per_device_batch_size=${PER_DEVICE_BATCH_SIZE}"
echo "world_size=${NPROC_PER_NODE}"
echo "accumulation_steps=${ACCUMULATION_STEPS}"
echo "effective_global_batch_size=${EFFECTIVE_GLOBAL_BATCH_SIZE}"
echo "backbone_lr=${BACKBONE_LR}"
echo "head_lr=${HEAD_LR}"

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
  scripts/downstream_50m/run_dortmund_age_full_finetune.py \
  --input-lance "${INPUT_LANCE}" \
  --checkpoint "${PRETRAINED_50M_CKPT}" \
  --split-manifest "${SPLIT_MANIFEST}" \
  "${AUDIT_ARGS[@]}" \
  --output-dir "${RUN_OUTPUT}" \
  --epochs "${FULL_EPOCHS}" \
  --batch-size "${PER_DEVICE_BATCH_SIZE}" \
  --accumulation-steps "${ACCUMULATION_STEPS}" \
  --backbone-lr "${BACKBONE_LR}" --head-lr "${HEAD_LR}" --weight-decay 0.05 \
  --gradient-clip-norm 1.0 --scheduler linear_warmup_cosine_decay \
  --warmup-ratio 0.1 --early-stopping-patience 15 \
  --probe-layer-idx 8 --pooling mean --train-seed "${TRAIN_SEED}" \
  --amp-mode bf16 --num-workers "${NUM_WORKERS}"

test -f "${RUN_OUTPUT}/_SUCCESS"
echo "FULL_OUTPUT=${RUN_OUTPUT}"
