#!/usr/bin/env bash
set -euo pipefail

: "${HUADA_ROOT:?Set HUADA_ROOT to the project root}"
: "${HUADA_TOKEN_LANCE:?Set HUADA_TOKEN_LANCE to the formal Huada FBMAE token Lance}"
: "${DORTMUND_AGE_CKPT:?Set DORTMUND_AGE_CKPT to best_full_checkpoint.pt}"
: "${FEATURE_CACHE_ROOT:?Set FEATURE_CACHE_ROOT}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
HEAD_LR="${HEAD_LR:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0}"
EPOCHS="${EPOCHS:-100}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-15}"
BATCH_SIZE="${BATCH_SIZE:-64}"
TRAIN_SEED="${TRAIN_SEED:-42}"
SPLIT_SEED="${SPLIT_SEED:-42}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-0}"
FEATURE_CACHE_POLICY="${FEATURE_CACHE_POLICY:-auto}"
FEATURE_CACHE_BATCH_SIZE="${FEATURE_CACHE_BATCH_SIZE:-8}"

test -d "$HUADA_ROOT"
test -d "$HUADA_TOKEN_LANCE"
test -f "$DORTMUND_AGE_CKPT"
test "$(basename "$DORTMUND_AGE_CKPT")" = "best_full_checkpoint.pt"

actual_sha256="$(sha256sum "$DORTMUND_AGE_CKPT" | awk '{print $1}')"
printf 'checkpoint_sha256=%s\n' "$actual_sha256"

RUN_ID="$(date -u +%Y%m%d_%H%M%S)"
RUN_OUTPUT="$OUTPUT_ROOT/50m_dortmund_age_probe_transfer_cv7_$RUN_ID"
test ! -e "$RUN_OUTPUT"

cd "$HUADA_ROOT"
export PYTHONPATH="$HUADA_ROOT/scripts/downstream_50m:${PYTHONPATH:-}"

ARGS=(
  scripts/downstream_50m/run_huada_dortmund_age_probe_transfer_cv7.py
  --input-lance "$HUADA_TOKEN_LANCE"
  --checkpoint "$DORTMUND_AGE_CKPT"
  --feature-cache-root "$FEATURE_CACHE_ROOT"
  --feature-cache-policy "$FEATURE_CACHE_POLICY"
  --feature-cache-batch-size "$FEATURE_CACHE_BATCH_SIZE"
  --output-dir "$RUN_OUTPUT"
  --device "$DEVICE"
  --num-workers "$NUM_WORKERS"
  --head-lr "$HEAD_LR"
  --weight-decay "$WEIGHT_DECAY"
  --epochs "$EPOCHS"
  --early-stopping-patience "$EARLY_STOPPING_PATIENCE"
  --batch-size "$BATCH_SIZE"
  --seed "$TRAIN_SEED"
  --split-seed "$SPLIT_SEED"
)

if test -n "${EXPECTED_CKPT_SHA256:-}"; then
  ARGS+=(--expected-checkpoint-sha256 "$EXPECTED_CKPT_SHA256")
fi

if test -n "${FEATURE_CACHE_BUILD_ROOT:-}"; then
  ARGS+=(--feature-cache-build-root "$FEATURE_CACHE_BUILD_ROOT")
fi
if test -n "${ZERO_SHOT_REFERENCE_DIR:-}"; then
  ARGS+=(--zero-shot-reference-dir "$ZERO_SHOT_REFERENCE_DIR")
fi

printf 'output_dir=%s\n' "$RUN_OUTPUT"
printf 'head_lr=%s weight_decay=%s epochs=%s batch_size=%s\n' \
  "$HEAD_LR" "$WEIGHT_DECAY" "$EPOCHS" "$BATCH_SIZE"
"$PYTHON_BIN" "${ARGS[@]}"
