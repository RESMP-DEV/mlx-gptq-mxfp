# MLX CUDA MXFP

A small, reproducible environment for probing BF16, native MXFP4, and native
MXFP8 matrix multiplication through MLX on CUDA GPUs.

This repository exists separately from model weights and quantization code. It
contains no credentials, provider configuration, model paths, or benchmark
answers.

## Why the bootstrap installs CUDA headers

An NVIDIA driver and CUDA runtime are not always sufficient. Some MLX-CUDA
paths compile kernels on first use and search `CUDA_HOME` or `CUDA_PATH` for
`include/cuda.h`. Runtime-only cloud images can therefore load MLX and report a
GPU, but fail at the first MXFP operation.

`bootstrap_cuda.sh` installs the pinned Python CUDA runtime package, discovers
its headers without assuming `/usr/local/cuda`, and writes a local
`.mlx-cuda-env` file containing `CUDA_HOME`, `CUDA_PATH`, and
`LD_LIBRARY_PATH`.

## Run

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
./bootstrap_cuda.sh
./run_probe.sh --m 256 --n 4096 --k 4096 --iterations 100
```

The probe emits one JSON record for each format. It exercises the live driver;
device names and timings are observations, not inferred capability claims.

## Reusing the environment

```bash
source .mlx-cuda-env
.venv/bin/python -m mlx_lm.generate --model /path/to/model --prompt "Hello"
```

For a different environment location, set `VENV=/path/to/venv` consistently
for both scripts. `PYTHON_VERSION` and `UV_BIN` may also be overridden.
