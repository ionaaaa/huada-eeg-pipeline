#!/usr/bin/env bash
set -euo pipefail

: "${HUADA_ROOT:=/opt/ml/input}"
: "${PYTHON_BIN:=python}"
: "${MIPDB_COMMON_DIR:=${HUADA_ROOT}/data/preprocessed_common/mipdb_age_preprocessing_full}"
: "${MIPDB_COMMON_LANCE:=}"
: "${MIPDB_COMMON_SOURCE_URI:=cos://jianyu-1442740494/huada/preprocessed_common/mipdb_age_preprocessing_full/}"
: "${TIONE_OUTPUT_ROOT:=/opt/ml/output}"
: "${ADAPTER_BATCH_SIZE:=32}"
: "${ADAPTER_NUM_WORKERS:=4}"
: "${RUN_TIMESTAMP:=$(date -u +%Y%m%d_%H%M%S)}"

OUTPUT_DIR="${ADAPTER_OUTPUT_DIR:-${TIONE_OUTPUT_ROOT}/mipdb_fbmae_tokens_full/${RUN_TIMESTAMP}}"
ADAPTER_SCRIPT="${HUADA_ROOT}/scripts/adapt_mipdb_fbmae.py"
MAPPING_CONFIG="${MIPDB_MAPPING_CONFIG:-${HUADA_ROOT}/configs/mipdb_egi128_to_standard64_sparse18_v1.json}"
TRAINING_PROFILES="${MIPDB_TRAINING_PROFILES:-${HUADA_ROOT}/configs/mipdb_dortmund_fbmae_training_profiles_v1.json}"
ELECTRODE_VOCAB="${MIPDB_ELECTRODE_VOCAB:-${HUADA_ROOT}/configs/fbmae_electrode_vocab_snapshot.json}"

fail() {
  echo "[mipdb-fbmae-full] ERROR: $*" >&2
  exit 1
}

[[ -d "$MIPDB_COMMON_DIR" ]] || fail "MIPDB_COMMON_DIR does not exist: $MIPDB_COMMON_DIR"
[[ -f "$ADAPTER_SCRIPT" ]] || fail "adapter script not found: $ADAPTER_SCRIPT"
[[ -f "$MAPPING_CONFIG" ]] || fail "mapping config not found: $MAPPING_CONFIG"
[[ -f "$TRAINING_PROFILES" ]] || fail "training profiles not found: $TRAINING_PROFILES"
[[ -f "$ELECTRODE_VOCAB" ]] || fail "electrode vocabulary not found: $ELECTRODE_VOCAB"
[[ ! -e "$OUTPUT_DIR" ]] || fail "refusing to overwrite output: $OUTPUT_DIR"
if [[ -n "$MIPDB_COMMON_LANCE" ]]; then
  [[ -d "$MIPDB_COMMON_LANCE" ]] || fail "MIPDB_COMMON_LANCE does not exist: $MIPDB_COMMON_LANCE"
fi
if [[ "$PYTHON_BIN" == */* ]]; then
  [[ -x "$PYTHON_BIN" ]] || fail "PYTHON_BIN is not executable: $PYTHON_BIN"
else
  command -v "$PYTHON_BIN" >/dev/null || fail "PYTHON_BIN is not on PATH"
fi
mkdir -p "$TIONE_OUTPUT_ROOT" || fail "cannot create output root: $TIONE_OUTPUT_ROOT"

echo "[mipdb-fbmae-full] input=$MIPDB_COMMON_DIR"
echo "[mipdb-fbmae-full] output=$OUTPUT_DIR"
echo "[mipdb-fbmae-full] mapping=mipdb_egi128_to_standard64_sparse18_v1"
echo "[mipdb-fbmae-full] geometry=[640,100]; discovery is dynamic; no subject/recording count gates"

args=(
  "$ADAPTER_SCRIPT"
  --input-dir "$MIPDB_COMMON_DIR"
  --output-dir "$OUTPUT_DIR"
  --mapping-config "$MAPPING_CONFIG"
  --training-profiles "$TRAINING_PROFILES"
  --electrode-vocab "$ELECTRODE_VOCAB"
  --source-uri "$MIPDB_COMMON_SOURCE_URI"
  --batch-size "$ADAPTER_BATCH_SIZE"
  --num-workers "$ADAPTER_NUM_WORKERS"
)
if [[ -n "$MIPDB_COMMON_LANCE" ]]; then
  args+=(--input-lance "$MIPDB_COMMON_LANCE")
fi

"$PYTHON_BIN" -u "${args[@]}" || {
  status=$?
  echo "[mipdb-fbmae-full] adapter failed with status $status" >&2
  exit "$status"
}

echo "[mipdb-fbmae-full] completed: $OUTPUT_DIR"
