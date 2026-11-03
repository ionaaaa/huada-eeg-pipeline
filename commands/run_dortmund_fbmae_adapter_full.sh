#!/usr/bin/env bash
set -euo pipefail

HUADA_ROOT="${HUADA_ROOT:-/opt/ml/input}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DORTMUND_COMMON_DIR="${DORTMUND_COMMON_DIR:-${HUADA_ROOT}/data/preprocessed_common/dortmund_age_preprocessing_full_20261103_022751}"
DORTMUND_COMMON_LANCE="${DORTMUND_COMMON_LANCE:-}"
DORTMUND_SOURCE_URI="${DORTMUND_SOURCE_URI:-cos://jianyu-1442740494/huada/preprocessed_common/dortmund_age_preprocessing_full_20261103_022751/}"
TIONE_OUTPUT_ROOT="${TIONE_OUTPUT_ROOT:-/opt/ml/output}"
ADAPTER_BATCH_SIZE="${ADAPTER_BATCH_SIZE:-32}"
ADAPTER_NUM_WORKERS="${ADAPTER_NUM_WORKERS:-4}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${ADAPTER_OUTPUT_DIR:-${TIONE_OUTPUT_ROOT}/dortmund_fbmae_tokens_full/${RUN_TIMESTAMP}}"
ADAPTER_SCRIPT="${HUADA_ROOT}/scripts/adapt_dortmund_fbmae.py"
CHANNEL_PROFILE="${DORTMUND_CHANNEL_PROFILE:-${HUADA_ROOT}/configs/dortmund_fbmae_channel_profile.toml}"
ELECTRODE_VOCAB="${DORTMUND_ELECTRODE_VOCAB:-${HUADA_ROOT}/configs/fbmae_electrode_vocab_snapshot.json}"

command -v "${PYTHON_BIN}" >/dev/null 2>&1
test -f "${ADAPTER_SCRIPT}"
test -f "${CHANNEL_PROFILE}"
test -f "${ELECTRODE_VOCAB}"
test -d "${DORTMUND_COMMON_DIR}"
if [[ -n "${DORTMUND_COMMON_LANCE}" ]]; then
  test -d "${DORTMUND_COMMON_LANCE}"
fi
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "Refusing to overwrite existing output: ${OUTPUT_DIR}" >&2
  exit 2
fi

echo "HUADA_ROOT=${HUADA_ROOT}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "DORTMUND_COMMON_DIR=${DORTMUND_COMMON_DIR}"
echo "DORTMUND_COMMON_LANCE=${DORTMUND_COMMON_LANCE:-<auto-discover>}"
echo "DORTMUND_SOURCE_URI=${DORTMUND_SOURCE_URI}"
echo "CHANNEL_PROFILE=${CHANNEL_PROFILE}"
echo "ELECTRODE_VOCAB=${ELECTRODE_VOCAB}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "ADAPTER_BATCH_SIZE=${ADAPTER_BATCH_SIZE}"
echo "ADAPTER_NUM_WORKERS=${ADAPTER_NUM_WORKERS}"

args=(
  "${ADAPTER_SCRIPT}"
  --input-dir "${DORTMUND_COMMON_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --channel-profile "${CHANNEL_PROFILE}"
  --electrode-vocab "${ELECTRODE_VOCAB}"
  --source-uri "${DORTMUND_SOURCE_URI}"
  --batch-size "${ADAPTER_BATCH_SIZE}"
  --num-workers "${ADAPTER_NUM_WORKERS}"
)
if [[ -n "${DORTMUND_COMMON_LANCE}" ]]; then
  args+=(--input-lance "${DORTMUND_COMMON_LANCE}")
fi

"${PYTHON_BIN}" "${args[@]}"
echo "Dortmund FBMAE full conversion completed: ${OUTPUT_DIR}"
