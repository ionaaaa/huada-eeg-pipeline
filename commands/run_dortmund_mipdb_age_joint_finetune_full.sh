#!/usr/bin/env bash
set -euo pipefail

: "${HUADA_ROOT:=/opt/ml/cfs_input}"
: "${PYTHON_BIN:=python}"
: "${DORTMUND_INPUT_LANCE:?Set DORTMUND_INPUT_LANCE to the frozen Dortmund token Lance}"
: "${MIPDB_INPUT_LANCE:?Set MIPDB_INPUT_LANCE to the all-block MIPDB token Lance}"
: "${DORTMUND_SPLIT_MANIFEST:?Set DORTMUND_SPLIT_MANIFEST to the frozen 172/21/21 manifest}"
: "${MIPDB_SPLIT_MANIFEST:=${HUADA_ROOT}/configs/mipdb_subject_split_manifest_v1.csv}"
: "${JOINT_SPLIT_MANIFEST:=${HUADA_ROOT}/configs/joint_subject_split_manifest_v1.csv}"
: "${JOINT_TARGET_TRANSFORM:=${HUADA_ROOT}/configs/joint_target_transform_v1.json}"
: "${JOINT_TRAINING_CONTRACT:=${HUADA_ROOT}/configs/joint_age_training_contract_v1.json}"
: "${TRAINING_PROFILES:=${HUADA_ROOT}/configs/mipdb_dortmund_fbmae_training_profiles_v1.json}"
: "${TRAINING_PROFILE:?Set TRAINING_PROFILE to joint_matched18 (B) or joint_native64_vs_sparse18 (C)}"
: "${PRETRAINED_50M_CKPT:=${HUADA_ROOT}/data/checkpoint/pretrain_checkpoint_4.pt}"
: "${EXPECTED_CKPT_SHA256:=97335b696b3ae9138dcb51c736f49ee1c6008fdc22fc42f13ea9a5301452f36e}"
: "${OUTPUT_ROOT:=${HUADA_ROOT}/downstream_output}"
: "${NPROC_PER_NODE:=2}"
: "${PER_DEVICE_BATCH_SIZE:=2}"
: "${ACCUMULATION_STEPS:=16}"
: "${FULL_EPOCHS:=100}"
: "${SAMPLES_PER_EPOCH:=6200}"
: "${BACKBONE_LR:=1e-5}"
: "${HEAD_LR:=1e-4}"
: "${WEIGHT_DECAY:=0.05}"
: "${WARMUP_RATIO:=0.1}"
: "${EARLY_STOPPING_PATIENCE:=15}"
: "${TRAIN_SEED:=42}"
: "${NUM_WORKERS:=0}"

case "${TRAINING_PROFILE}" in
  joint_matched18|joint_native64_vs_sparse18) ;;
  *) echo "Invalid TRAINING_PROFILE=${TRAINING_PROFILE}" >&2; exit 2 ;;
esac

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_OUTPUT="${OUTPUT_ROOT}/dortmund_mipdb_age_joint_${TRAINING_PROFILE}_${STAMP}"

test -d "${HUADA_ROOT}"
test -d "${DORTMUND_INPUT_LANCE}"
test -d "${MIPDB_INPUT_LANCE}"
test -f "${DORTMUND_SPLIT_MANIFEST}"
test -f "${MIPDB_SPLIT_MANIFEST}"
test -f "${JOINT_SPLIT_MANIFEST}"
test -f "${JOINT_TARGET_TRANSFORM}"
test -f "${JOINT_TRAINING_CONTRACT}"
test -f "${TRAINING_PROFILES}"
test -f "${PRETRAINED_50M_CKPT}"
test ! -e "${RUN_OUTPUT}"
printf '%s  %s\n' "${EXPECTED_CKPT_SHA256}" "${PRETRAINED_50M_CKPT}" | sha256sum -c -

cd "${HUADA_ROOT}"
export PYTHONPATH="${HUADA_ROOT}/scripts/downstream_50m:${PYTHONPATH:-}"

AUDIT_ARGS=(--audit-policy record_only)
if [[ -n "${AUDIT_SUMMARY:-}" && -f "${AUDIT_SUMMARY}" ]]; then
  AUDIT_ARGS+=(--audit-summary "${AUDIT_SUMMARY}")
fi

echo "training_profile=${TRAINING_PROFILE}"
echo "samples_per_epoch=${SAMPLES_PER_EPOCH}"
echo "effective_global_batch_size=$((PER_DEVICE_BATCH_SIZE * NPROC_PER_NODE * ACCUMULATION_STEPS))"
echo "mipdb_filter=block01"

"${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE}" \
  scripts/downstream_50m/run_dortmund_mipdb_age_joint_finetune.py \
  --input-lance "${DORTMUND_INPUT_LANCE}" \
  --mipdb-input-lance "${MIPDB_INPUT_LANCE}" \
  --checkpoint "${PRETRAINED_50M_CKPT}" \
  --split-manifest "${DORTMUND_SPLIT_MANIFEST}" \
  --mipdb-split-manifest "${MIPDB_SPLIT_MANIFEST}" \
  --joint-subject-split-manifest "${JOINT_SPLIT_MANIFEST}" \
  --joint-target-transform "${JOINT_TARGET_TRANSFORM}" \
  --joint-training-contract "${JOINT_TRAINING_CONTRACT}" \
  --training-profiles "${TRAINING_PROFILES}" \
  --training-profile "${TRAINING_PROFILE}" \
  --mipdb-block-id block01 \
  "${AUDIT_ARGS[@]}" \
  --output-dir "${RUN_OUTPUT}" \
  --epochs "${FULL_EPOCHS}" \
  --samples-per-epoch "${SAMPLES_PER_EPOCH}" \
  --batch-size "${PER_DEVICE_BATCH_SIZE}" \
  --accumulation-steps "${ACCUMULATION_STEPS}" \
  --backbone-lr "${BACKBONE_LR}" \
  --head-lr "${HEAD_LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --gradient-clip-norm 1.0 \
  --scheduler linear_warmup_cosine_decay \
  --warmup-ratio "${WARMUP_RATIO}" \
  --early-stopping-patience "${EARLY_STOPPING_PATIENCE}" \
  --probe-layer-idx 8 \
  --pooling mean \
  --train-seed "${TRAIN_SEED}" \
  --amp-mode bf16 \
  --num-workers "${NUM_WORKERS}"

test -f "${RUN_OUTPUT}/_SUCCESS"
echo "FULL_OUTPUT=${RUN_OUTPUT}"
