# Private DFlash-MLX fork for the calibrated Qwen3.8 MXFP4 target

Date: 2026-10-17

## Scope

This records the start of the private DFlash-MLX integration for
`RESMP-DEV/Qwen3.8-27B-MLX-GPTQ-MXFP4`. It covers the useful upstream runtime,
the first local-path integration fix, and validation. It does not claim a new
draft quantization, a CUDA port, upstream submission, or representative serving
performance.

## Live host state

Strix is reachable as `ssh strix`:

- hostname: `compute`;
- uptime: 32 minutes at the probe;
- RTX 5070 Laptop GPU: 8,151 MiB total, 2 MiB used, 0% utilization;
- root filesystem: 3.6 TB with 2.7 TB available.

No DFlash benchmark was launched there in this step. The upstream runtime is
Apple Silicon/MLX oriented, while the calibrated target already runs on the M4
Max. Strix remains available for bounded calibration or CUDA-side work that
does not require the custom Metal kernels.

## Fork

- Upstream: `https://github.com/bstnxbt/dflash-mlx`
- Upstream head: `60803233af4589e18588b9bacbb03880801c828a`
- Private mirror: `https://github.com/RESMP-DEV/dflash-mlx`
- Visibility: private
- Default branch: `main`
- Implementation branch: `work/local-mxfp4-dflash2`
- Implementation head: `9ced85b5e01a06ea5cbeda8ca026ec43735bc1d5`
- GitHub Actions are disabled at the repository level.

## First implementation change

Benchmarking exposed a local-path bug: a local
`Qwen3.8-27B-MLX-GPTQ-MXFP4` target matched the Qwen3.8 support spec, but a
local `z-lab-Qwen3.8-27B-DFlash2` draft path did not string-match the registry
reference exactly. The model therefore loaded the draft in BF16 unless W4 was
explicitly requested.

The private branch now recognizes both Hub references and local download
directory names when applying model-default draft quantization. Explicit
`--draft-quant none` still disables the default.

## Validation

- `ruff check dflash_mlx/runtime/registry.py`: exit 0.
- `ruff format --check dflash_mlx/runtime/registry.py`: exit 0.
- `pytest tests/test_runtime_bundle.py -q`: 13 passed.
- Real bundle probe with no explicit `--draft-quant`:
  - `effective_draft_quant=w4`;
  - `draft_quant_spec=w4`;
  - `draft_quant_source=model_default`.
- Real 128-token benchmark with no explicit quantization flag:
  - baseline: 27.541 tok/s;
  - DFlash2 W4: 52.564 tok/s;
  - speedup: 1.909×;
  - acceptance: 74.219%;
  - DFlash peak: 19.671 GB;
  - report records effective `draft_quant=w4`.

Machine receipt:
`build/reports/dflash/calibrated-mxfp4-default-w4-smoke/`.

A broader CLI test selection had four failures caused by the checked-out
runtime's incompatibility with the installed MLX-LM 0.32.2 server API and a
pre-existing test tokenizer stub, not by this registry change. The focused
runtime suite and real-model probe bound the changed behavior.
