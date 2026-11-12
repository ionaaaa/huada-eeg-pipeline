#!/usr/bin/env bash
set -euo pipefail

: "${HUADA_ROOT:=/mnt/cfs/jianyu/huada-eeg-pipeline}"
: "${PYTHON_BIN:=python}"
: "${HUADA_TOKEN_LANCE:?Set HUADA_TOKEN_LANCE to the formal Huada FBMAE token Lance directory}"
: "${DORTMUND_AGE_CKPT:?Set DORTMUND_AGE_CKPT to best_full_checkpoint.pt}"
: "${OUTPUT_ROOT:=${HUADA_ROOT}/downstream_output}"
: "${DEVICE:=cuda}"
: "${BATCH_SIZE:=8}"
: "${NUM_WORKERS:=0}"

test -d "${HUADA_ROOT}"
test -d "${HUADA_TOKEN_LANCE}"
test -f "${DORTMUND_AGE_CKPT}"

actual_sha256="$(sha256sum "${DORTMUND_AGE_CKPT}" | awk '{print $1}')"
echo "checkpoint_sha256=${actual_sha256}"

timestamp="$(date +%Y%m%d_%H%M%S)"
run_output="${OUTPUT_ROOT}/50m_dortmund_age_huada_zero_shot_${timestamp}"
if [[ -e "${run_output}" ]]; then
  echo "Refusing to overwrite ${run_output}" >&2
  exit 2
fi

cd "${HUADA_ROOT}"
export PYTHONPATH="${HUADA_ROOT}/scripts/downstream_50m:${PYTHONPATH:-}"

checkpoint_args=()
if [[ -n "${EXPECTED_CKPT_SHA256:-}" ]]; then
  checkpoint_args+=(--expected-checkpoint-sha256 "${EXPECTED_CKPT_SHA256}")
fi

exec "${PYTHON_BIN}" scripts/downstream_50m/run_huada_dortmund_age_zero_shot.py \
  --input-lance "${HUADA_TOKEN_LANCE}" \
  --checkpoint "${DORTMUND_AGE_CKPT}" \
  "${checkpoint_args[@]}" \
  --output-dir "${run_output}" \
  --device "${DEVICE}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}"
