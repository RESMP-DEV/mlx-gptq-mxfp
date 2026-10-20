# Validation

Validation performed on 2026-08-06 from a fresh checkout on an NVIDIA RTX PRO
6000 Blackwell Server Edition:

- The frozen `uv.lock` created a Python 3.12 environment.
- MLX reported `Device(gpu, 0)` through the live driver.
- The bootstrap discovered `cuda.h` from `nvidia-cuda-runtime-cu12` and wrote
  working `CUDA_HOME`, `CUDA_PATH`, and `LD_LIBRARY_PATH` exports.
- BF16, native MXFP4, and native MXFP8 matrix multiplications all executed.

The validation probe used a deliberately tiny shape and establishes execution,
not representative performance.

Publication checks:

- Tracked-file secret scan: zero findings.
- OSV-Scanner on the frozen lockfile: no issues found.
- `pip-audit --disable-pip` on hashed frozen requirements: no known
  vulnerabilities. Disabling installation allows CUDA-only Linux wheels to be
  audited from the macOS control machine.
