#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UV_BIN="${UV_BIN:-uv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
VENV="${VENV:-$ROOT/.venv}"

command -v "$UV_BIN" >/dev/null 2>&1 || {
    echo "uv is required: https://docs.astral.sh/uv/" >&2
    exit 1
}

UV_PROJECT_ENVIRONMENT="$VENV" "$UV_BIN" sync \
    --project "$ROOT" \
    --frozen \
    --python "$PYTHON_VERSION"

cuda_root="$("$VENV/bin/python" - <<'PY'
from pathlib import Path
import nvidia

for base in map(Path, nvidia.__path__):
    candidate = base / "cuda_runtime"
    if (candidate / "include" / "cuda.h").is_file():
        print(candidate)
        break
else:
    raise SystemExit("nvidia-cuda-runtime-cu12 headers were not found")
PY
)"

cat >"$ROOT/.mlx-cuda-env" <<EOF
export CUDA_HOME="$cuda_root"
export CUDA_PATH="$cuda_root"
export LD_LIBRARY_PATH="$cuda_root/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
EOF

CUDA_HOME="$cuda_root" CUDA_PATH="$cuda_root" \
    "$VENV/bin/python" - <<'PY'
import mlx.core as mx

if "gpu" not in str(mx.default_device()).lower():
    raise SystemExit(f"expected a GPU device, got {mx.default_device()}")
print(f"MLX device: {mx.default_device()}")
PY

echo "Environment ready. Run: source $ROOT/.mlx-cuda-env"
