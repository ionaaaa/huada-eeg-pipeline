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
ADAPTER_BATCH_SIZE="${ADAPTER_BATCH_SIZE:-32}"
ADAPTER_NUM_WORKERS="${ADAPTER_NUM_WORKERS:-4}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${ADAPTER_OUTPUT_DIR:-${TIONE_OUTPUT_ROOT}/dortmund_fbmae_tokens_full/${RUN_TIMESTAMP}}"
ADAPTER_SCRIPT="${HUADA_ROOT}/scripts/adapt_dortmund_fbmae.py"
CHANNEL_PROFILE="${DORTMUND_CHANNEL_PROFILE:-${HUADA_ROOT}/configs/dortmund_fbmae_channel_profile.toml}"
ELECTRODE_VOCAB="${DORTMUND_ELECTRODE_VOCAB:-${HUADA_ROOT}/configs/fbmae_electrode_vocab_snapshot.json}"

fail() {
  echo "[dortmund-fbmae-full] ERROR: $*" >&2
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

if [[ "${PYTHON_BIN}" == */* ]]; then
  [[ -x "${PYTHON_BIN}" ]] || fail "PYTHON_BIN is not executable: ${PYTHON_BIN}"
else
  command -v "${PYTHON_BIN}" >/dev/null 2>&1 \
    || fail "PYTHON_BIN is not on PATH: ${PYTHON_BIN}"
fi
[[ -f "${ADAPTER_SCRIPT}" ]] || fail "adapter script does not exist: ${ADAPTER_SCRIPT}"
[[ -f "${CHANNEL_PROFILE}" ]] || fail "channel profile does not exist: ${CHANNEL_PROFILE}"
[[ -f "${ELECTRODE_VOCAB}" ]] || fail "electrode vocabulary does not exist: ${ELECTRODE_VOCAB}"
[[ -d "${DORTMUND_COMMON_DIR}" ]] \
  || fail "DORTMUND_COMMON_DIR does not exist: ${DORTMUND_COMMON_DIR}"
if [[ -n "${DORTMUND_COMMON_LANCE}" ]]; then
  [[ -d "${DORTMUND_COMMON_LANCE}" ]] \
    || fail "DORTMUND_COMMON_LANCE does not exist or is not a directory: ${DORTMUND_COMMON_LANCE}"
fi
[[ ! -e "${OUTPUT_DIR}" ]] || fail "refusing to overwrite existing output: ${OUTPUT_DIR}"

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
  --batch-size "${ADAPTER_BATCH_SIZE}"
  --num-workers "${ADAPTER_NUM_WORKERS}"
)
if [[ -n "${DORTMUND_COMMON_LANCE}" ]]; then
  args+=(--input-lance "${DORTMUND_COMMON_LANCE}")
fi

if "${PYTHON_BIN}" -u "${args[@]}"; then
  :
else
  status=$?
  echo "[dortmund-fbmae-full] ERROR: adapter exited with status ${status}" >&2
  for artifact in adaptation_errors.log adapter_summary.json input_issues.csv adaptation_issues.csv; do
    if [[ -s "${OUTPUT_DIR}/${artifact}" ]]; then
      echo "[dortmund-fbmae-full] ${artifact}:" >&2
      tail -n 100 "${OUTPUT_DIR}/${artifact}" >&2
    fi
  done
  if [[ ! -d "${OUTPUT_DIR}" ]]; then
    echo "[dortmund-fbmae-full] no finalized output directory was created: ${OUTPUT_DIR}" >&2
  fi
  exit "${status}"
fi
echo "Dortmund FBMAE full conversion completed: ${OUTPUT_DIR}"
