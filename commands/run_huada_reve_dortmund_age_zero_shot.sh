#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${PEER_ZERO_SHOT_DIR:?Set PEER_ZERO_SHOT_DIR to the passed LaBraM strict zero-shot result}"
export MODEL_FAMILY=reve
export PEER_MODEL_FAMILY=labram
exec bash "$SCRIPT_DIR/run_huada_portable_age_zero_shot.sh"
