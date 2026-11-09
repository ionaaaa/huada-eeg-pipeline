#!/usr/bin/env bash
set -Eeuo pipefail
exec 2>&1

: "${HUADA_ROOT:?Set HUADA_ROOT to the Huada repository root}"
: "${PYTHON_BIN:?Set PYTHON_BIN to the model-forward Python runtime}"
: "${MODEL_FAMILY:?Set MODEL_FAMILY to labram or reve}"
: "${NATIVE_CACHE_RUN_ROOT:?Set NATIVE_CACHE_RUN_ROOT to a completed native cache run}"
: "${SUBJECT_MANIFEST:?Set SUBJECT_MANIFEST to the audited Huada subject_manifest.json}"
: "${EXPECTED_SUBJECT_MANIFEST_SHA256:?Set the full subject manifest SHA256}"
: "${TARGET_INPUT_CONTRACT:?Set TARGET_INPUT_CONTRACT to the frozen Huada target contract}"
: "${EXPECTED_TARGET_CONTRACT_FILE_SHA256:?Set the full target contract file SHA256}"
: "${PORTABLE_CHECKPOINT:?Set PORTABLE_CHECKPOINT to the formal portable v1 file}"
: "${PORTABLE_SIDECAR:?Set PORTABLE_SIDECAR to the formal rich sidecar JSON}"
: "${EXPECTED_PORTABLE_CHECKPOINT_SHA256:?Set the full portable file SHA256}"
: "${EXPECTED_SOURCE_CHECKPOINT_SHA256:?Set the full source FFT checkpoint SHA256}"
: "${DYNAMIC_PARITY_DIR:?Set DYNAMIC_PARITY_DIR to the passed source/portable parity result}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to a new-results parent directory}"
: "${EXPECTED_SOURCE_AGE_STD:?Set the exact full-precision source scaler std}"
: "${FBMAE_ZERO_SHOT_DIR:?Set FBMAE_ZERO_SHOT_DIR to the passed FBMAE strict zero-shot result}"

DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EXPECTED_SOURCE_AGE_MEAN="${EXPECTED_SOURCE_AGE_MEAN:-27.5}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
DRY_RUN="${DRY_RUN:-0}"

case "$MODEL_FAMILY" in
  labram|reve) ;;
  *) echo "ERROR: MODEL_FAMILY must be labram or reve" >&2; exit 2 ;;
esac

test -x "$PYTHON_BIN"
test -d "$HUADA_ROOT"
test -d "$NATIVE_CACHE_RUN_ROOT"
test -f "$NATIVE_CACHE_RUN_ROOT/_SUCCESS"
test -f "$SUBJECT_MANIFEST"
test -f "$TARGET_INPUT_CONTRACT"
test -f "$PORTABLE_CHECKPOINT"
test -f "${PORTABLE_CHECKPOINT}.manifest.json"
test -f "$PORTABLE_SIDECAR"
test -d "$DYNAMIC_PARITY_DIR"
test -f "$DYNAMIC_PARITY_DIR/_SUCCESS"
mkdir -p "$OUTPUT_ROOT"

RUN_OUTPUT="$OUTPUT_ROOT/huada_${MODEL_FAMILY}_dortmund_age_strict_zero_shot_${RUN_ID}"
test ! -e "$RUN_OUTPUT"

export PYTHONPATH="$HUADA_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

ARGS=(
  -m scripts.native_backends.run_huada_portable_age_zero_shot
  --model-family "$MODEL_FAMILY"
  --native-cache-run-root "$NATIVE_CACHE_RUN_ROOT"
  --subject-manifest "$SUBJECT_MANIFEST"
  --expected-subject-manifest-sha256 "$EXPECTED_SUBJECT_MANIFEST_SHA256"
  --target-input-contract "$TARGET_INPUT_CONTRACT"
  --expected-target-contract-file-sha256 "$EXPECTED_TARGET_CONTRACT_FILE_SHA256"
  --portable-checkpoint "$PORTABLE_CHECKPOINT"
  --portable-sidecar "$PORTABLE_SIDECAR"
  --expected-portable-checkpoint-sha256 "$EXPECTED_PORTABLE_CHECKPOINT_SHA256"
  --expected-source-checkpoint-sha256 "$EXPECTED_SOURCE_CHECKPOINT_SHA256"
  --dynamic-parity-dir "$DYNAMIC_PARITY_DIR"
  --output-dir "$RUN_OUTPUT"
  --device "$DEVICE"
  --batch-size "$BATCH_SIZE"
  --expected-source-age-mean "$EXPECTED_SOURCE_AGE_MEAN"
  --expected-source-age-std "$EXPECTED_SOURCE_AGE_STD"
)

test -d "$FBMAE_ZERO_SHOT_DIR"
test -f "$FBMAE_ZERO_SHOT_DIR/_SUCCESS"
ARGS+=(--comparison-result "fbmae=$FBMAE_ZERO_SHOT_DIR")

if [[ -n "${PEER_ZERO_SHOT_DIR:-}" ]]; then
  : "${PEER_MODEL_FAMILY:?Set PEER_MODEL_FAMILY when PEER_ZERO_SHOT_DIR is set}"
  test -d "$PEER_ZERO_SHOT_DIR"
  test -f "$PEER_ZERO_SHOT_DIR/_SUCCESS"
  ARGS+=(--comparison-result "${PEER_MODEL_FAMILY}=$PEER_ZERO_SHOT_DIR")
fi

if [[ "$DRY_RUN" == "1" ]]; then
  ARGS+=(--dry-run)
elif [[ "$DRY_RUN" != "0" ]]; then
  echo "ERROR: DRY_RUN must be 0 or 1" >&2
  exit 2
fi

cd "$HUADA_ROOT"
printf "model_family=%s\n" "$MODEL_FAMILY"
printf "native_cache_run_root=%s\n" "$NATIVE_CACHE_RUN_ROOT"
printf "portable_checkpoint=%s\n" "$PORTABLE_CHECKPOINT"
printf "dynamic_parity_dir=%s\n" "$DYNAMIC_PARITY_DIR"
printf "output_dir=%s\n" "$RUN_OUTPUT"
printf "training=false calibration=false prediction_clipping=false\n"
printf "resolved_command="
printf " %q" "$PYTHON_BIN" "${ARGS[@]}"
printf "\n"

exec "$PYTHON_BIN" "${ARGS[@]}"
