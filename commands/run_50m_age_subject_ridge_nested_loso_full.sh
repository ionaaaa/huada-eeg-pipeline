#!/usr/bin/env bash
set -euo pipefail

: "${HUADA_ROOT:?HUADA_ROOT is required}"
: "${INPUT_LANCE:?INPUT_LANCE is required}"
: "${MEAN_FEATURE_CACHE_LANCE:?MEAN_FEATURE_CACHE_LANCE is required}"
: "${FEATURE_CACHE_MANIFEST:?FEATURE_CACHE_MANIFEST is required}"
: "${TIONE_OUTPUT_ROOT:?TIONE_OUTPUT_ROOT is required}"

PYTHON_BIN="${PYTHON_BIN:-python}"
RIDGE_ALPHAS="${RIDGE_ALPHAS:-0.01 0.1 1 10 100 1000}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${TIONE_OUTPUT_ROOT}/50m_age_subject_ridge_nested_loso_${STAMP}"

test -d "${HUADA_ROOT}"
test -d "${INPUT_LANCE}"
test -d "${MEAN_FEATURE_CACHE_LANCE}"
test -f "${FEATURE_CACHE_MANIFEST}"
test ! -e "${OUTPUT_DIR}"
mkdir -p "${TIONE_OUTPUT_ROOT}"

cd "${HUADA_ROOT}"
read -r -a ALPHA_ARGS <<< "${RIDGE_ALPHAS}"
"${PYTHON_BIN}" scripts/downstream_50m/run_age_subject_ridge_nested_loso.py \
  --input-lance "${INPUT_LANCE}" \
  --feature-cache-lance "${MEAN_FEATURE_CACHE_LANCE}" \
  --feature-cache-manifest "${FEATURE_CACHE_MANIFEST}" \
  --output-dir "${OUTPUT_DIR}" \
  --alphas "${ALPHA_ARGS[@]}" \
  --aggregation mean \
  --feature-transform train_center \
  --target-transform train_subject_zscore \
  --selection-metric mae \
  --device cpu \
  --log-level "${LOG_LEVEL}"
