#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MODEL_FAMILY=reve
exec bash "$SCRIPT_DIR/run_huada_portable_age_zero_init_probe_cv7.sh"
