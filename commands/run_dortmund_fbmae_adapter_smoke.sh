#!/usr/bin/env bash
set -euo pipefail

HUADA_ROOT="${HUADA_ROOT:-/opt/ml/input}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DORTMUND_COMMON_DIR="${DORTMUND_COMMON_DIR:-${HUADA_ROOT}/data/preprocessed_common/dortmund_age_preprocessing_full_20261103_022751}"
DORTMUND_COMMON_LANCE="${DORTMUND_COMMON_LANCE:-}"
DORTMUND_SOURCE_URI="${DORTMUND_SOURCE_URI:-cos://jianyu-1442740494/huada/preprocessed_common/dortmund_age_preprocessing_full_20261103_022751/}"
TIONE_OUTPUT_ROOT="${TIONE_OUTPUT_ROOT:-/opt/ml/output}"
: "${DORTMUND_MIN_AGE:=20}"
: "${DORTMUND_MAX_AGE:=35}"
: "${DORTMUND_COHORT_ID:=age${DORTMUND_MIN_AGE}_${DORTMUND_MAX_AGE}_ses1_pre_profile_a}"
ADAPTER_BATCH_SIZE="${ADAPTER_BATCH_SIZE:-8}"
ADAPTER_NUM_WORKERS="${ADAPTER_NUM_WORKERS:-0}"
ADAPTER_SMOKE_SUBJECTS="${ADAPTER_SMOKE_SUBJECTS:-}"
ADAPTER_SMOKE_MAX_WINDOWS="${ADAPTER_SMOKE_MAX_WINDOWS:-2}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${ADAPTER_OUTPUT_DIR:-${TIONE_OUTPUT_ROOT}/dortmund_fbmae_tokens_smoke/${RUN_TIMESTAMP}}"
ADAPTER_SCRIPT="${HUADA_ROOT}/scripts/adapt_dortmund_fbmae.py"
CHANNEL_PROFILE="${DORTMUND_CHANNEL_PROFILE:-${HUADA_ROOT}/configs/dortmund_fbmae_channel_profile.toml}"
ELECTRODE_VOCAB="${DORTMUND_ELECTRODE_VOCAB:-${HUADA_ROOT}/configs/fbmae_electrode_vocab_snapshot.json}"

fail() {
  echo "[dortmund-fbmae-smoke] ERROR: $*" >&2
  exit 1
}

if [[ -z "${DORTMUND_EXPECTED_SUBJECTS:-}" ]]; then
  if [[ "${DORTMUND_MIN_AGE}" == "20" && "${DORTMUND_MAX_AGE}" == "35" ]]; then
    DORTMUND_EXPECTED_SUBJECTS=214
  else
    fail "DORTMUND_EXPECTED_SUBJECTS must be set for non-default age range"
  fi
fi
if [[ -z "${DORTMUND_EXPECTED_RECORDINGS:-}" ]]; then
  if [[ "${DORTMUND_MIN_AGE}" == "20" && "${DORTMUND_MAX_AGE}" == "35" ]]; then
    DORTMUND_EXPECTED_RECORDINGS=428
  else
    fail "DORTMUND_EXPECTED_RECORDINGS must be set for non-default age range"
  fi
fi

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
echo "DORTMUND_COHORT_ID=${DORTMUND_COHORT_ID}"
echo "AGE_RANGE=[${DORTMUND_MIN_AGE},${DORTMUND_MAX_AGE}]"
echo "EXPECTED_SUBJECTS=${DORTMUND_EXPECTED_SUBJECTS}"
echo "EXPECTED_RECORDINGS=${DORTMUND_EXPECTED_RECORDINGS}"
echo "CHANNEL_PROFILE=${CHANNEL_PROFILE}"
echo "ELECTRODE_VOCAB=${ELECTRODE_VOCAB}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "ADAPTER_BATCH_SIZE=${ADAPTER_BATCH_SIZE}"
echo "ADAPTER_NUM_WORKERS=${ADAPTER_NUM_WORKERS}"
echo "ADAPTER_SMOKE_SUBJECTS=${ADAPTER_SMOKE_SUBJECTS:-<auto-select and record basis>}"
echo "ADAPTER_SMOKE_MAX_WINDOWS=${ADAPTER_SMOKE_MAX_WINDOWS}"

args=(
  "${ADAPTER_SCRIPT}"
  --input-dir "${DORTMUND_COMMON_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --channel-profile "${CHANNEL_PROFILE}"
  --electrode-vocab "${ELECTRODE_VOCAB}"
  --source-uri "${DORTMUND_SOURCE_URI}"
  --min-age "${DORTMUND_MIN_AGE}"
  --max-age "${DORTMUND_MAX_AGE}"
  --cohort-id "${DORTMUND_COHORT_ID}"
  --expected-subjects "${DORTMUND_EXPECTED_SUBJECTS}"
  --expected-recordings "${DORTMUND_EXPECTED_RECORDINGS}"
  --smoke
  --smoke-max-windows-per-recording "${ADAPTER_SMOKE_MAX_WINDOWS}"
  --batch-size "${ADAPTER_BATCH_SIZE}"
  --num-workers "${ADAPTER_NUM_WORKERS}"
)
if [[ -n "${DORTMUND_COMMON_LANCE}" ]]; then
  args+=(--input-lance "${DORTMUND_COMMON_LANCE}")
fi
if [[ -n "${ADAPTER_SMOKE_SUBJECTS}" ]]; then
  args+=(--smoke-subjects "${ADAPTER_SMOKE_SUBJECTS}")
fi

"${PYTHON_BIN}" "${args[@]}"
echo "Dortmund FBMAE smoke completed: ${OUTPUT_DIR}"
