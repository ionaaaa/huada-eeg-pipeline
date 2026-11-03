#!/usr/bin/env bash
set -euo pipefail

: "${HUADA_ROOT:=/opt/ml/cfs_input}"
: "${PYTHON_BIN:=python}"
: "${COS_ROOT:=/opt/ml/cos_input}"
: "${INPUT_LANCE:=${HUADA_ROOT}/data/model_tokens_full/fbmae_tokens/20261103_112918/dortmund_fbmae_tokens.lance}"
: "${SPLIT_MANIFEST:=${HUADA_ROOT}/downstream_output/dortmund_age_split_seed42_20261103_153000/subject_split_manifest.csv}"
: "${AUDIT_SUMMARY:=${HUADA_ROOT}/audit_dortmund_fbmae_tokens/20261103_124436/audit_summary.json}"
: "${OUTPUT_ROOT:=${HUADA_ROOT}/downstream_output}"
: "${TRAIN_SEED:=42}"
: "${NPROC_PER_NODE:=2}"
: "${EXPECTED_CKPT_SHA256:=97335b696b3ae9138dcb51c736f49ee1c6008fdc22fc42f13ea9a5301452f36e}"
: "${PER_DEVICE_BATCH_SIZE:=2}"

if [[ -z "${PRETRAINED_50M_CKPT:-}" ]]; then
  if [[ -f "${HUADA_ROOT}/data/checkpoint/pretrain_checkpoint_4.pt" ]]; then
    PRETRAINED_50M_CKPT="${HUADA_ROOT}/data/checkpoint/pretrain_checkpoint_4.pt"
  else
    PRETRAINED_50M_CKPT="${COS_ROOT}/checkpoint/pretrain_checkpoint_4.pt"
  fi
fi
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_OUTPUT="${OUTPUT_ROOT}/dortmund_age_full_finetune_smoke_${STAMP}"

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

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
  scripts/downstream_50m/run_dortmund_age_full_finetune.py \
  --input-lance "${INPUT_LANCE}" \
  --checkpoint "${PRETRAINED_50M_CKPT}" \
  --split-manifest "${SPLIT_MANIFEST}" \
  "${AUDIT_ARGS[@]}" \
  --output-dir "${RUN_OUTPUT}" \
  --epochs 2 --batch-size "${PER_DEVICE_BATCH_SIZE}" --accumulation-steps 1 \
  --samples-per-epoch 8 \
  --backbone-lr 1e-5 --head-lr 1e-4 --weight-decay 0.05 \
  --gradient-clip-norm 1.0 --scheduler linear_warmup_cosine_decay \
  --warmup-ratio 0.1 --early-stopping-patience 2 \
  --probe-layer-idx 8 --pooling mean --train-seed "${TRAIN_SEED}" \
  --amp-mode bf16 --num-workers 0 --smoke --skip-test \
  --smoke-train-subjects 3 --smoke-validation-subjects 2 \
  --smoke-windows-per-subject 4

test -f "${RUN_OUTPUT}/_SUCCESS"
echo "SMOKE_OUTPUT=${RUN_OUTPUT}"
