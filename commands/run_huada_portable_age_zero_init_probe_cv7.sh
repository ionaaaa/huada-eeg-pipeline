#!/usr/bin/env bash
set -Eeuo pipefail
exec 2>&1

on_error() {
  local status=$?
  printf 'ERROR: zero-init probe CV7 failed status=%s line=%s command=%q\n' \
    "$status" "${BASH_LINENO[0]:-unknown}" "${BASH_COMMAND:-unknown}" >&2
  exit "$status"
}
trap on_error ERR

require_dir() { [[ -d "$1" ]] || { echo "ERROR: missing $2 directory: $1" >&2; return 2; }; }
require_file() { [[ -f "$1" ]] || { echo "ERROR: missing $2 file: $1" >&2; return 2; }; }

: "${HUADA_ROOT:?Set HUADA_ROOT}"
: "${PYTHON_BIN:?Set PYTHON_BIN}"
: "${MODEL_FAMILY:?Set MODEL_FAMILY to labram or reve}"
: "${NATIVE_CACHE_RUN_ROOT:?Set NATIVE_CACHE_RUN_ROOT}"
: "${SUBJECT_MANIFEST:?Set SUBJECT_MANIFEST}"
: "${TARGET_INPUT_CONTRACT:?Set TARGET_INPUT_CONTRACT}"
: "${EXPECTED_TARGET_CONTRACT_FILE_SHA256:?Set EXPECTED_TARGET_CONTRACT_FILE_SHA256}"
: "${PORTABLE_CHECKPOINT:?Set PORTABLE_CHECKPOINT}"
: "${PORTABLE_SIDECAR:?Set PORTABLE_SIDECAR}"
: "${EXPECTED_PORTABLE_CHECKPOINT_SHA256:?Set EXPECTED_PORTABLE_CHECKPOINT_SHA256}"
: "${EXPECTED_SOURCE_CHECKPOINT_SHA256:?Set EXPECTED_SOURCE_CHECKPOINT_SHA256}"
: "${EXPECTED_SOURCE_AGE_STD:?Set EXPECTED_SOURCE_AGE_STD}"
: "${DYNAMIC_PARITY_DIR:?Set DYNAMIC_PARITY_DIR}"
: "${STRICT_ZERO_SHOT_REFERENCE_DIR:?Set STRICT_ZERO_SHOT_REFERENCE_DIR}"
: "${FEATURE_CACHE_ROOT:?Set FEATURE_CACHE_ROOT}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT}"

case "$MODEL_FAMILY" in labram|reve) ;; *) echo "ERROR: invalid MODEL_FAMILY=$MODEL_FAMILY" >&2; exit 2;; esac

DEVICE="${DEVICE:-cuda:0}"
FEATURE_CACHE_POLICY="${FEATURE_CACHE_POLICY:-auto}"
FEATURE_CACHE_BATCH_SIZE="${FEATURE_CACHE_BATCH_SIZE:-8}"
BATCH_SIZE="${BATCH_SIZE:-64}"
HEAD_LR="${HEAD_LR:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0}"
EPOCHS="${EPOCHS:-100}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-15}"
TRAIN_SEED="${TRAIN_SEED:-42}"
SPLIT_SEED="${SPLIT_SEED:-42}"
EXPECTED_SOURCE_AGE_MEAN="${EXPECTED_SOURCE_AGE_MEAN:-27.5}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
DRY_RUN="${DRY_RUN:-0}"

[[ -x "$PYTHON_BIN" ]] || { echo "ERROR: Python is not executable: $PYTHON_BIN" >&2; exit 2; }
require_dir "$HUADA_ROOT" "Huada repository"

RUN_OUTPUT="$OUTPUT_ROOT/huada_${MODEL_FAMILY}_zero_init_probe_cv7_${RUN_ID}"
[[ ! -e "$RUN_OUTPUT" ]] || { echo "ERROR: output already exists: $RUN_OUTPUT" >&2; exit 2; }

if [[ "$DRY_RUN" == 0 ]]; then
  require_dir "$NATIVE_CACHE_RUN_ROOT" "native cache run"
  require_file "$NATIVE_CACHE_RUN_ROOT/_SUCCESS" "native cache success marker"
  require_file "$SUBJECT_MANIFEST" "subject manifest"
  require_file "$TARGET_INPUT_CONTRACT" "target input contract"
  require_file "$PORTABLE_CHECKPOINT" "portable checkpoint"
  require_file "${PORTABLE_CHECKPOINT}.manifest.json" "portable manifest"
  require_file "$PORTABLE_SIDECAR" "portable sidecar"
  require_dir "$DYNAMIC_PARITY_DIR" "dynamic parity"
  require_file "$DYNAMIC_PARITY_DIR/_SUCCESS" "dynamic parity success marker"
  require_dir "$STRICT_ZERO_SHOT_REFERENCE_DIR" "strict zero-shot reference"
  require_file "$STRICT_ZERO_SHOT_REFERENCE_DIR/_SUCCESS" "strict zero-shot success marker"
  mkdir -p "$FEATURE_CACHE_ROOT" "$OUTPUT_ROOT"
elif [[ "$DRY_RUN" != 1 ]]; then
  echo "ERROR: DRY_RUN must be 0 or 1" >&2
  exit 2
fi

export PYTHONPATH="$HUADA_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

ARGS=(
  -m scripts.native_backends.run_huada_portable_age_zero_init_probe_cv7
  --model-family "$MODEL_FAMILY"
  --native-cache-run-root "$NATIVE_CACHE_RUN_ROOT"
  --subject-manifest "$SUBJECT_MANIFEST"
  --target-input-contract "$TARGET_INPUT_CONTRACT"
  --expected-target-contract-file-sha256 "$EXPECTED_TARGET_CONTRACT_FILE_SHA256"
  --portable-checkpoint "$PORTABLE_CHECKPOINT"
  --portable-sidecar "$PORTABLE_SIDECAR"
  --expected-portable-checkpoint-sha256 "$EXPECTED_PORTABLE_CHECKPOINT_SHA256"
  --expected-source-checkpoint-sha256 "$EXPECTED_SOURCE_CHECKPOINT_SHA256"
  --expected-source-age-mean "$EXPECTED_SOURCE_AGE_MEAN"
  --expected-source-age-std "$EXPECTED_SOURCE_AGE_STD"
  --dynamic-parity-dir "$DYNAMIC_PARITY_DIR"
  --strict-zero-shot-reference-dir "$STRICT_ZERO_SHOT_REFERENCE_DIR"
  --feature-cache-root "$FEATURE_CACHE_ROOT"
  --feature-cache-policy "$FEATURE_CACHE_POLICY"
  --output-dir "$RUN_OUTPUT"
  --device "$DEVICE"
  --feature-cache-batch-size "$FEATURE_CACHE_BATCH_SIZE"
  --batch-size "$BATCH_SIZE"
  --head-lr "$HEAD_LR"
  --weight-decay "$WEIGHT_DECAY"
  --epochs "$EPOCHS"
  --early-stopping-patience "$EARLY_STOPPING_PATIENCE"
  --seed "$TRAIN_SEED"
  --split-seed "$SPLIT_SEED"
)

if [[ "$DRY_RUN" == 1 ]]; then
  ARGS+=(--dry-run)
fi

cd "$HUADA_ROOT"
printf 'model_family=%s\nfeature_cache_root=%s\noutput_dir=%s\n' \
  "$MODEL_FAMILY" "$FEATURE_CACHE_ROOT" "$RUN_OUTPUT"
printf 'probe_initialization=all_zero folds=7 candidates_per_fold=1 head_lr=%s weight_decay=%s\n' \
  "$HEAD_LR" "$WEIGHT_DECAY"
printf 'resolved_command='; printf ' %q' "$PYTHON_BIN" "${ARGS[@]}"; printf '\n'
exec "$PYTHON_BIN" "${ARGS[@]}"
