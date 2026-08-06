#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-$ROOT/.venv}"

if [[ ! -f "$ROOT/.mlx-cuda-env" ]]; then
    echo "Run ./bootstrap_cuda.sh first" >&2
    exit 1
fi

# shellcheck disable=SC1091
source "$ROOT/.mlx-cuda-env"

mkdir -p "$ROOT/probe-results"
for mode in bf16 mxfp4 mxfp8; do
    "$VENV/bin/python" "$ROOT/probe_mxfp_cuda.py" \
        --mode "$mode" "$@" | tee "$ROOT/probe-results/$mode.json"
done
