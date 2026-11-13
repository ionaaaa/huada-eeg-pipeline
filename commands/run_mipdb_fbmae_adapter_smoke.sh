#!/usr/bin/env bash
set -euo pipefail

: "${HUADA_ROOT:=/opt/ml/input}"
: "${PYTHON_BIN:=python}"
: "${MIPDB_COMMON_DIR:=${HUADA_ROOT}/mipdb_preprocessing_smoke/20261113_123132}"
: "${MIPDB_COMMON_LANCE:=${MIPDB_COMMON_DIR}/dataset.lance}"
: "${MIPDB_COMMON_SOURCE_URI:=local://${MIPDB_COMMON_DIR}}"
: "${TIONE_OUTPUT_ROOT:=/opt/ml/output}"
: "${MIPDB_SMOKE_SUBJECTS:=sub-A00058775 sub-A00062842}"
: "${ADAPTER_SMOKE_MAX_WINDOWS:=2}"
: "${RUN_TIMESTAMP:=$(date -u +%Y%m%d_%H%M%S)}"

OUTPUT_DIR="${ADAPTER_OUTPUT_DIR:-${TIONE_OUTPUT_ROOT}/mipdb_fbmae_tokens_smoke/${RUN_TIMESTAMP}}"
read -r -a smoke_subjects <<< "$MIPDB_SMOKE_SUBJECTS"
[[ ${#smoke_subjects[@]} -ge 1 ]] || { echo "empty smoke subjects" >&2; exit 1; }

args=(
  "$HUADA_ROOT/scripts/adapt_mipdb_fbmae.py"
  --input-dir "$MIPDB_COMMON_DIR"
  --input-lance "$MIPDB_COMMON_LANCE"
  --output-dir "$OUTPUT_DIR"
  --mapping-config "$HUADA_ROOT/configs/mipdb_egi128_to_standard64_sparse18_v1.json"
  --training-profiles "$HUADA_ROOT/configs/mipdb_dortmund_fbmae_training_profiles_v1.json"
  --electrode-vocab "$HUADA_ROOT/configs/fbmae_electrode_vocab_snapshot.json"
  --source-uri "$MIPDB_COMMON_SOURCE_URI"
  --batch-size 8
  --num-workers 0
  --smoke
  --smoke-subjects "${smoke_subjects[@]}"
  --smoke-max-windows-per-recording "$ADAPTER_SMOKE_MAX_WINDOWS"
)

[[ ! -e "$OUTPUT_DIR" ]] || { echo "refusing to overwrite: $OUTPUT_DIR" >&2; exit 1; }
mkdir -p "$TIONE_OUTPUT_ROOT"
"$PYTHON_BIN" -u "${args[@]}"
echo "[mipdb-fbmae-smoke] completed: $OUTPUT_DIR"
