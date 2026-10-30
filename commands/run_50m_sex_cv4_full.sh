#!/usr/bin/env bash
set -euo pipefail

: "${HUADA_ROOT:?Set HUADA_ROOT to the mounted project directory}"
: "${INPUT_LANCE:?Set INPUT_LANCE to the formal model-token Lance URI or mount}"
: "${PRETRAINED_50M_CKPT:?Set PRETRAINED_50M_CKPT to the real 50M backbone checkpoint}"
: "${TIONE_OUTPUT_ROOT:?Set TIONE_OUTPUT_ROOT to a persistent existing directory}"

PYTHON_BIN="${PYTHON_BIN:-python}"
POOLING="${POOLING:-mean}"
HEAD_LR="${HEAD_LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-3}"
TRAIN_SEED="${TRAIN_SEED:-42}"
SPLIT_SEED="${SPLIT_SEED:-42}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-64}"

case "$POOLING" in
  mean|flatten) ;;
  *) echo "POOLING must be mean or flatten" >&2; exit 2 ;;
esac

test -d "$HUADA_ROOT"
test -f "$PRETRAINED_50M_CKPT"
test -d "$TIONE_OUTPUT_ROOT"
if [[ "$INPUT_LANCE" != *"://"* ]]; then
  test -d "$INPUT_LANCE"
fi

cd "$HUADA_ROOT"
test -d "scripts/downstream_50m"

LANCE_AUTH=()
if [[ -n "${STORAGE_OPTIONS:-}" ]]; then
  test -f "$STORAGE_OPTIONS"
  LANCE_AUTH=(--storage-options "$STORAGE_OPTIONS")
fi

safe_tag() {
  printf '%s' "$1" | tr '.+' 'p_' | tr -c '[:alnum:]_-' '_'
}

POOLING_TAG="$(safe_tag "$POOLING")"
LR_TAG="$(safe_tag "$HEAD_LR")"
WD_TAG="$(safe_tag "$WEIGHT_DECAY")"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${TIONE_OUTPUT_ROOT%/}/50m_sex_cv4_${POOLING_TAG}_lr${LR_TAG}_wd${WD_TAG}_${STAMP}"
test ! -e "$OUTPUT_DIR"

"$PYTHON_BIN" -c 'import torch,lance,pyarrow; print({"torch":torch.__version__,"cuda":torch.cuda.is_available(),"lance":lance.__version__,"pyarrow":pyarrow.__version__})'

"$PYTHON_BIN" scripts/downstream_50m/largeScale_fineTune.py \
  --input-lance "$INPUT_LANCE" \
  "${LANCE_AUTH[@]}" \
  --task sex_classification \
  --protocol sex_stratified_subject_cv4 \
  --folds 4 \
  --pooling "$POOLING" \
  --head-lr "$HEAD_LR" \
  --weight-decay "$WEIGHT_DECAY" \
  --output-dir "$OUTPUT_DIR" \
  --checkpoint "$PRETRAINED_50M_CKPT" \
  --device cuda \
  --mode linear_probe \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --seed "$TRAIN_SEED" \
  --split-seed "$SPLIT_SEED"

printf 'CV4 output: %s\n' "$OUTPUT_DIR"
