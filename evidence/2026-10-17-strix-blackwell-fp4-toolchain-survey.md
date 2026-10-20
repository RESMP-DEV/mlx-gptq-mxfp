# Strix Blackwell FP4 toolchain and Ampere compatibility survey

**Recorded:** 2026-10-17 16:20 America/Los_Angeles.

This report follows the corrected Strix MXFP4 result in
`evidence/2026-10-17-strix-nvfp4-mxfp4-flops.md`. It answers two narrower questions:

1. What current Blackwell tooling should be used to infer SM120 MXFP4/NVFP4?
2. What happens when the same tooling reaches the Ampere `compute` machine?

The central correction is that native SM120 MXFP4 is not undocumented silicon. It is
present in the current PTX ISA, CUTLASS, and Triton surfaces. Our first pass missed it
because it stopped at PyTorch/cuBLASLt and at a stale Triton tutorial capability gate.

## Executive conclusions

1. **Strix/RTX 5070 Laptop is a native SM120 FP4 device.** PTX ISA 9.4 says
   `.e2m1` MMA plus `.kind`, `.block_scale`, and `.scale_vec_size` require
   `sm_120a`, with `kind::mxf4` and `kind::mxf4nvf4` accepting `.e2m1` operands and
   UE8M0 or UE4M3 scale metadata. The locally generated instruction is:

   ```text
   mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0
   ```

2. **CUTLASS 4.8.0 explicitly tests SM120 MXFP4, and those tests pass on this exact
   GPU.** The official test suite has
   `SM120_Device_Blockscaled_Gemm_mxf4t_mxf4n_f32n_tensor_op_f32`. Five MXFP4 tests
   passed on the RTX 5070 Laptop with CUDA 13.4 and GCC 13. This supersedes both the
   initial PyTorch/cuBLASLt interpretation and any claim that SM120 MXFP4 had to be
   inferred only through Triton.

3. **Triton support is current, but the bundled tutorial gate is misleading.** Triton
   PR #8494, merged 2025-10-20, is titled "[NVIDIA] Add native FP4 scaled_dot for
   SM120" and says that it replaces the previous decomposition fallback for
   MXFP4-by-MXFP4 and NVFP4-by-NVFP4. Triton 3.8 contains this support, but the
   tutorial's `supports_block_scaling()` path still checks only CUDA major versions
   `[10, 11]`. SM120 therefore has to bypass that specific tutorial gate.

4. **PyTorch 2.14.1's MXFP4 rejection is deliberate, not a hardware probe.** Commit
   `95d1df7d4e604fecea1f6977345217781c2cd990` disables CUDA MXFP4 outside major 10
   because B200 passed while RTX 5090 tests failed at that time. Its error text is the
   one observed on Strix: "MXFP4 scaling only supported in CUDA for B200/B300".

5. **Ampere cannot accelerate FP4 and must not use Triton's fallback for performance.**
   On the RTX 3090 Ti (SM86), PyTorch rejects both NVFP4 and MXFP4 `scaled_mm`.
   Triton 3.8 `tl.dot_scaled` passes its MXFP4 correctness check, but generated PTX
   contains `mma.sync...bf16.bf16`, not `mxf4nvf4`. At `8192x5120x4096`, that path
   reached **1.449 TFLOP/s** while direct BF16 cuBLAS reached **77.243 TFLOP/s**:
   the Triton fallback is **53.3x slower** than the normal BF16 control on this shape.
   Use Ampere only for dequantized BF16 controls or a separate W4A16 weight-only path.

6. **The local Strix result is not a global first.** Triton PR #8494, CUTLASS 4.8,
   MR-GPTQ/QuTLASS, QuACK, and FlashInfer already cover SM120 FP4. The local value is
   narrower: an exact RTX 5070 Laptop capability/performance measurement, matched
   MXFP4/NVFP4 kernels on that low-memory consumer part, and a documented Ampere
   compatibility boundary.

## Current toolchain matrix

| Layer | Current state checked 2026-10-17 | SM120 implication |
|---|---:|---|
| CUDA | Strix host toolkit 13.4.92; current PyPI CUDA runtime 13.4.92 | Use CUDA 13.4/GCC13 on Strix. Host CMake 4.2 failed its CUDA compiler-ID probe; isolated CMake 3.31.6 worked. |
| PTX ISA 9.4 | Documents `sm_120a` block-scaled MMA, E2M1, UE8M0/UE4M3, `scale_vec::2X/4X` | Canonical instruction contract for direct-kernel work. |
| CUTLASS C++ | Release 4.8.0, revision `098de2a652cf8f00fd70b2df54051c7eccbb855a` | Contains SM120 MXFP4-MXFP4 unit tests and SM120 dispatch schedules. Five MXFP4 tests passed locally. |
| CUTLASS Python DSL | PyPI `nvidia-cutlass-dsl` 4.8.1 | Current high-level CuTe DSL packaging for the 4.8 series. |
| Triton | Release 3.8.0; PR #8494 merged 2025-10-20 | Native SM120 MXFP4/NVFP4 `dot_scaled`; tutorial gate remains stale. |
| PyTorch | Release 2.14.1 | NVFP4 framework path works on Strix; CUDA MXFP4 deliberately B200/B300-gated. |
| FlashInfer | Release 0.7.2, 2026-10-15 | SM12x NVFP4 CuTe-DSL GEMM, SM120 W4A8 MegaMoE, and opt-in SM120 NVFP4 sparse-MLA paths. |
| vLLM | Release 0.31.0, 2026-10-05 | Uses FlashInfer 0.7-era kernels, QuTLASS builds, and SM120 tuned Triton configs; kernel dispatch still needs per-model proof. |
| SGLang | GitHub latest release 0.5.21; tag/package 0.5.22 exists | FlashInfer/TRT-LLM and CuTe-DSL NVFP4 paths plus SM120 attention work are present in source; no exact RTX 5070 Laptop claim was verified here. |
| QuACK/QuTLASS | QuACK release 0.6.5; MR-GPTQ paper v3 | Explicit `GemmSm120`, MXFP4/NVFP4 operands, and RTX 5090-tuned SM120 configs. |
| TensorRT-LLM / TensorRT-RTX | Latest GitHub release shown as 1.2.1; newer July changes are on main/nightly | Current main filters SM120/SM121 shared-memory-oversized MoE tactics. The current TensorRT-RTX support matrix lists CC12.x FP4 GEMM and CC8.6 no FP4 GEMM. |

## Live Strix CUTLASS validation

Pinned source:

```text
CUTLASS v4.8.0
098de2a652cf8f00fd70b2df54051c7eccbb855a
```

The host's first two configure attempts exposed a toolchain boundary:

- CMake 4.2 plus CUDA 13.4 rejected the CUDA compiler-ID probe.
- CUDA 13.4 with the default GCC 15 host compiler also failed CUDA's supported-host
  compiler check.
- Isolated CMake 3.31.6 plus `/usr/bin/gcc-13` configured successfully.

Commands, reduced to the decisive ones:

```bash
/home/kearm/cutlass-4.8.0/.cmake/bin/cmake \
  -S /home/kearm/cutlass-4.8.0 \
  -B /home/kearm/cutlass-4.8.0/build-sm120 \
  -GNinja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/gcc-13 \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.4/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 \
  -DCUTLASS_ENABLE_TESTS=ON \
  -DCUTLASS_TEST_LEVEL=0 \
  -DCUTLASS_NVCC_ARCHS=120a \
  -DCUTLASS_UNITY_BUILD_ENABLED=ON

ninja -C /home/kearm/cutlass-4.8.0/build-sm120 -j8 \
  cutlass_test_unit_bs_gemm_device_tensorop_sm120

.../cutlass_test_unit_bs_gemm_device_tensorop_sm120 \
  --gtest_filter='*mxf4t_mxf4n*'
```

Observed result:

```text
[==========] Running 5 tests from 1 test suite.
[ RUN ] ...128x128x256 ... OK
[ RUN ] ...128x64x256 ... OK
[ RUN ] ...128x32x256 ... OK
[ RUN ] ...128x16x256 ... OK
[ RUN ] ...128x8x256 ... OK
[  PASSED  ] 5 tests.
```

The test log is
`build/reports/strix-fp4-research-20261017/strix-cutlass-4.8/cutlass-sm120-mxfp4-test.log`.
The GPU returned to `0%` and `2 MiB` after the test.

## Live Ampere result

Host and environment:

- two RTX 3090 Ti GPUs, CUDA capability `(8,6)`, driver `595.99.02`
- host compiler toolkit: CUDA 12.4
- benchmark venv: PyTorch `2.14.1+cu130`, Triton `3.8.0`
- GPU0 was used by the probe. A resident rust-indexer process held 1,410 MiB at zero
  utilization. A separate DFlash2 distillation process was active on the other GPU and
  was preserved.

The small Triton correctness probe passed:

```text
✅ (pass mxfp4 - Triton only)
compiled_kernels 1
kernel ... {'tcgen05_scaled': False, 'tcgen05_any': False,
            'mma_bf16': True, 'mma_any': True}
```

Its PTX contains:

```text
mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
```

It does not contain `kind::mxf4nvf4`.

Performance and framework behavior:

| Path | Shape | Median | Notes |
|---|---:|---:|---|
| Triton `dot_scaled` MXFP4 | 8192x5120x4096 | 1.449 TFLOP/s | Correct but BF16 decomposition and unpack/scale overhead dominate. |
| BF16 `torch.matmul` control | 8192x5120x4096 | 77.243 TFLOP/s | Direct Ampere tensor-core control on GPU0. |
| Triton `dot_scaled` MXFP4 | 1024x1024x1024 | 1.305 TFLOP/s | Same fallback behavior. |
| BF16 `torch.matmul` control | 1024x1024x1024 | 48.771 TFLOP/s | Smaller-shape control. |
| PyTorch `scaled_mm` NVFP4 | 1024x1024x1024 | rejected | CC8.6 is below the CUDA FP4 API gate. |
| PyTorch `scaled_mm` MXFP4 | 1024x1024x1024 | rejected | Same CC8.6 gate. |

The exact PyTorch rejection is:

```text
NotImplementedError: torch._scaled_mm is only supported on CUDA devices
with compute capability >= 9.0 or 8.9, or ROCm MI300+
```

Therefore, a cross-architecture experiment must explicitly select one of:

- **Strix**: native MXFP4/NVFP4 tensor-core kernels.
- **Ampere**: dequantized BF16 reference, or an Ampere-specific W4A16 kernel.
- Never report Triton `dot_scaled` on Ampere as FP4 performance.

## Public work that changes the novelty assessment

### Triton

PR #8494, merged 2025-10-20:

> "Implement native FP4 scaled matmul support for SM120 by replacing the previous
> decomposition fallback."

It explicitly lists MXFP4-by-MXFP4 and NVFP4-by-NVFP4, and reports an RTX 5090 vLLM
E2E case where native MXFP4 takes 33 seconds versus 61 seconds on the prior branch.

### CUTLASS

CUTLASS 4.8's release notes say optimal code generation is with CUDA 13.4, describe
GeForce Blackwell block-scaled examples, and the repository contains:

```text
test/unit/gemm/device/sm120_blockscaled_tensorop_gemm/
  sm120_bs_gemm_mxf4_mxf4_f32_f32.cu
```

The test uses `cutlass::mx_float4_t<cutlass::float_e2m1_t>`, UE8M0 scale factors,
`arch::Sm120`, and `OpClassBlockScaledTensorOp`.

### PyTorch

Commit `95d1df7` says:

> "MXFP4 unit tests pass on B200, fail on RTX 5090 - disable non-B200 cases."

It installs the exact B200/B300 major-version check. This explains why framework
`scaled_mm` and the earlier cuBLASLt heuristic probe were insufficient architecture
probes.

### MR-GPTQ / QuTLASS / QuACK

MR-GPTQ v3 ("Bridging the Gap Between Promise and Performance for Microscaling FP4
Quantization", arXiv:2509.23202) describes:

- a comprehensive MXFP4/NVFP4 PTQ study;
- MXFP4 E8M0 scale fitting;
- SM100 and SM120 kernel implementations;
- MXFP4 throughput up to about 15% higher than NVFP4 on B200;
- RTX 5090 layerwise and end-to-end speedups.

QuACK 0.6.5 contains `GemmSm120`, explicit MXFP4/NVFP4 operand formats, and comments
citing RTX 5090 measurements over 2048³ through 8192³. It is therefore the closest
ready-made kernel stack for a local controlled comparison.

### FlashInfer and serving stacks

FlashInfer 0.7.2 adds:

- SM12x NVFP4 CuTe-DSL dense GEMM;
- SM120 W4A8 MXFP4/MXFP8 expert-parallel MoE;
- opt-in SM120 NVFP4 sparse-MLA decode/prefill for RTX PRO 6000/RTX 5090.

vLLM 0.31.0 pulls the FlashInfer 0.7 series, contains SM120 tuned persistent matmul
configs, and builds QuTLASS with current PyTorch. SGLang source exposes FlashInfer
TRT-LLM and CuTe-DSL NVFP4 paths. These are promising, but an RTX 5070 Laptop run must
capture kernel-selection evidence because 8 GiB memory and 36 SMs can select different
tactics than RTX 5090/RTX PRO 6000.

## Recommended local inference strategy

1. Use **Triton 3.8** for fast iteration on Strix because it already emits native SM120
   MXFP4 and NVFP4. Continue checking PTX for `kind::mxf4nvf4`; do not rely on API
   acceptance alone.
2. Use **CUTLASS 4.8/QuACK 0.6.5** as the performance and layout reference. The official
   MXFP4 test now gives a known-good control independent of Triton.
3. Treat **PyTorch 2.14.1 `scaled_mm`** as an NVFP4-only convenience path on Strix. It is
   not an MXFP4 architecture oracle.
4. For model inference, use **FlashInfer 0.7.2 or QuACK** rather than writing a custom
   GEMM first, but capture per-dispatch kernel names and PTX/SASS evidence. This is
   especially important on the 8 GiB RTX 5070 Laptop.
5. Keep **Ampere as a control host**: dequantized BF16 cuBLAS or W4A16 only. The measured
   1.449 TFLOP/s Triton fallback is unsuitable for any inference-performance claim.
6. Benchmark model shapes, not only large cubes. Existing DFlash2-shaped M=128 cases and
   BF16 LM-head bottlenecks can dominate while large-cube FP4 throughput looks excellent.

## Sources

- PTX ISA 9.4: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
- CUTLASS 4.8.0 release: https://github.com/NVIDIA/cutlass/releases/tag/v4.8.0
- CUTLASS SM120 MXFP4 test: https://github.com/NVIDIA/cutlass/blob/v4.8.0/test/unit/gemm/device/sm120_blockscaled_tensorop_gemm/sm120_bs_gemm_mxf4_mxf4_f32_f32.cu
- Triton 3.8.0 release: https://github.com/triton-lang/triton/releases/tag/v3.8.0
- Triton native SM120 FP4 PR: https://github.com/triton-lang/triton/pull/8494
- PyTorch MXFP4 gate commit: https://github.com/pytorch/pytorch/commit/95d1df7d4e604fecea1f6977345217781c2cd990
- FlashInfer 0.7.2 release: https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.7.2
- FlashInfer SM12x NVFP4 GEMM PR: https://github.com/flashinfer-ai/flashinfer/pull/5100
- FlashInfer SM120 NVFP4 sparse-MLA PR: https://github.com/flashinfer-ai/flashinfer/pull/5914
- vLLM 0.31.0 release: https://github.com/vllm-project/vllm/releases/tag/v0.31.0
- vLLM SM120 tuned matmul PR: https://github.com/vllm-project/vllm/pull/57456
- SGLang 0.5.21 release: https://github.com/sgl-project/sglang/releases/tag/v0.5.21
- SGLang 0.5.22 tag: https://github.com/sgl-project/sglang/tree/v0.5.22
- QuACK 0.6.5 release: https://github.com/Dao-AILab/quack/releases/tag/v0.6.5
- MR-GPTQ paper v3: https://arxiv.org/html/2509.23202
- TensorRT-LLM SM120 shared-memory filter PR: https://github.com/NVIDIA/TensorRT-LLM/pull/12704
- TensorRT-RTX current support matrix: https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/getting-started/support-matrix.html

## Research sub-agent status

Four read-only network-enabled workers were launched:

| Worker | Result | Handling |
|---|---|---|
| Triton/compiler | failed after 170 s; all configured CCR tiers returned `service_unavailable` | Replaced by direct Triton release/PR inspection. |
| CUTLASS/PyTorch | timed out at 1,200 s with empty output | Replaced by pinned CUTLASS source inspection and live unit tests. |
| NVIDIA architecture | timed out at 1,200 s with empty output | Replaced by PTX ISA and CUTLASS source inspection. |
| SOTA research | completed in 1,157.7 s | Key claims cross-checked. Its "first documented" and "unpublished tile configuration" claims were rejected after finding Triton PR #8494 and CUTLASS/QuACK source. |

The SOTA worker's useful, cross-checked contribution was locating MR-GPTQ, QuACK, and
the B200 matched-kernel MXFP4/NVFP4 direction. Its self-report is evidence, not
acceptance.

## Provenance and checks

- Repository: `/Users/kearm/mlx-gptq`
- Source revision: `7131c407af8b8991fb5a8148a49cf2f323c7afd3`
- Agent: Codex
- Recorded timezone: America/Los_Angeles
- Current official release metadata was queried live on 2026-10-17.
- Source-link check: all report URLs returned HTTP 200 after replacing the moved
  TensorRT-RTX support-matrix URL.
- CUTLASS 4.8.0 configure/build/test: exit 0; the MXFP4 GoogleTest filter passed 5/5.
- Ampere numerical probe: exit 0; generated PTX marker check found BF16 MMA and no
  native FP4 MMA.
- Ampere benchmark and BF16/API probe: exit 0 and JSON artifacts written.
- Strix GPU state after CUTLASS test: 0% utilization and 2 MiB used.
- No benchmark process remained on Ampere. An unrelated DFlash2 distillation process on
  the other GPU was identified and preserved.

Pre-existing repository dirty files were not modified:

```text
.gitignore
mlx_gptq/capture.py
mlx_gptq/sequential.py
pyproject.toml
tests/test_capture.py
uv.lock
tests/test_sequential_context.py
```

Task-owned durable/ignored additions are under `evidence/` and
`build/reports/strix-fp4-research-20261017/`. Remote reproducible artifacts remain at:

- Strix: `/home/kearm/cutlass-4.8.0` and its `build-sm120/` tree
- Ampere: `/home/kearm/reports/ampere-fp4-tooling-20261017`

## Artifact hashes

Core hashes are recorded in
`build/reports/strix-fp4-research-20261017/core-artifact-hashes.txt`. Important entries:

| Artifact | SHA-256 |
|---|---|
| prior corrected Strix report | `cc92ca2f119d0880917d0d03cef8a1b1d1540f0e251e7ef5c7197f8a2f649ef7` |
| Ampere Triton MXFP4 JSON | `dce11c1ba488a334e5d4d7b513e61523bffe414ae5f53acff9507b2d1cdff013` |
| Ampere BF16/API JSON | `bec2519351953e361144249925d3354975f62e9042d2aab784a8f724d3715dc3` |
| Ampere generated PTX | `d65c29b38698ca3c87d44aab4876691a71e41f49d0fe774311dacc069b68686d` |
| Strix CUTLASS MXFP4 test log | `8fec18703947538bed7de311ddbf88ba603a6eab6e9e0837aa56318eb0fd6038` |
| SOTA worker output | `163ab156392d8c54b31b112833bdf4c3d5309796ce3d0d835115ec4878162c26` |

## Non-claims

- No end-to-end vLLM, SGLang, TensorRT-LLM, or QuACK model serving run was performed.
- The Ampere result does not compare RTX 5070 and RTX 3090 Ti performance; it compares
  fallback versus direct BF16 execution on the RTX 3090 Ti.
- The Ampere benchmark had resident service memory on GPU0 and unrelated DFlash2 work on
  the other GPU. It establishes the fallback boundary, not a clean Ampere peak claim.
- No direct Ampere cuBLASLt FP4 heuristic probe was run; PyTorch's live capability gate
  and TensorRT-RTX's CC8.6 "FP4 GEMM: No" support matrix already establish the relevant
  architecture boundary for this survey.
- The public-source survey is current as of 2026-10-17 and prioritized to the local
  SM120/SM86 decision; it is not a complete catalog of every Blackwell kernel.
