# Tetra fused MXFP4 activation and weight packing

## Claim

The DFlash2 SM120 harness now prepares MXFP4 activations with Triton group
max, quantization, byte packing, and E8M0 scale preshuffle. It also repacks
packed uint32 E2M1 weights directly to Triton's byte layout without an int64
intermediate. The native matmul remains MXFP4 by MXFP4 and still lowers to
`kind::mxf4nvf4`.

This work ran on Tetra, a guest host with four RTX PRO 6000 Blackwell
Workstation Edition GPUs. The root-owned vLLM workers were not modified.

## Implementation

- `tools/mxfp4_kernel/mxfp4_matmul.py`
  - Adds fused activation group-max and quantize/pack Triton kernels.
  - Keeps E8M0 exponent selection in PyTorch on the small max tensor.
  - Writes activation padding deterministically without a padded input copy.
  - Adds direct uint32 E2M1 to Triton-byte weight repacking.
  - Uses the fused preparation path for the public matmul helper.
  - Corrects synthetic uint32 E2M1 packing to shifts of `4*i`, matching MLX's
    documented layout.
- `tools/mxfp4_kernel/e2m1_packing.py`
  - Keeps pure PyTorch layout helpers importable without Triton for regression
    tests.
- `tools/mxfp4_kernel/activation_quantizer_bench.py`
  - Compares codes and preshuffled scales byte-for-byte against the reference.
  - Bounds reference memory by comparing and timing aligned 128-row blocks.
- `tools/mxfp4_kernel/weight_repack_bench.py`
  - Checks direct weight repacking byte-for-byte on all ten DFlash2 projections.

The initial full-reference activation gate at FC `M=8192` failed to allocate
another 800 MiB transient tensor with only about 8 GiB free beside the resident
worker. The accepted gate therefore compares aligned 128-row blocks. This is a
memory-safe reference control, not a claim that the old full-tensor path could
run on Tetra.

## Provenance

Source parent: `0b5474d98fedc7af1530d6b953ceefbdb2ebc4a3`. The receipts were
produced with this kernel change dirty in the owned engine worktree; the final
commit follows this evidence file. The exact commands and per-case results are
in the three JSON receipts named below.

## Live results

Activation packer receipt:
`build/reports/dflash/tetra-activation-quantizer-20261020/results.json`

- Device: RTX PRO 6000 Blackwell Workstation Edition, CC 12.0.
- Software: PyTorch 2.14.0+cu130, Triton 3.8.0, CUDA 13.0.
- Cases: 16/16 passed across rows 5, 128, 2048, and 8192.
- Every case had byte-equal E2M1 codes and byte-equal preshuffled E8M0 scales.
- These are the four dominant projections, not all ten DFlash2 projections.
- FC `M=8192`: fused 1.2968 ms; 128-row-block reference 16.4647 ms.
- Gate/up/down projections at `M=8192`: fused 0.2725-0.8736 ms.

Weight repacker receipt:
`build/reports/dflash/tetra-activation-quantizer-20261020/weight-repack.json`

- Cases: 10/10 DFlash2 projections passed byte-for-byte.
- FC direct repack: 0.0923 ms.
- Gate/up/down direct repack: 0.0236-0.0310 ms.
- The first direct-repack attempt failed the byte gate and was rejected. It
  exposed the harness's nonstandard shift table; the corrected standard layout
  passes the roundtrip and all direct-repack cases.

Integrated matmul receipt:
`build/reports/dflash/tetra-activation-quantizer-20261020/mxfp4-matmul-m128.json`

- Cases: four dominant projections at `M=128`, all passed.
- Integrated GEMM coverage is not the full row-count or projection matrix.
- PTX markers: `mxf4nvf4=true`, `bf16_mma=false`, `fp8_mma=false`.
- Kernel throughput: 152.9-636.6 logical TFLOP/s for the selected shapes.
- Warm preparation: 0.156-0.551 ms. The first FC call was 484.89 ms because it
  included Triton JIT compilation; cold and warm values are recorded separately.
- Maximum per-element tolerance ratio: 0.2724 under the 5% relative plus 1%
  global-maximum absolute floor.
- Local regression suite: 77 passed, 15 skipped.

## Non-claims

- Synthetic operands prove packing, numerics, and kernel throughput only.
- They do not qualify DFlash2 model quality, acceptance rate, or end-to-end
  training throughput.
- The 128-row-block reference is a memory-bounded control, not the failed
  full-tensor control.
- Weight preparation is one-time for static projections; activation preparation
  is the repeated cost.
- No production DFlash2 checkpoint was loaded or modified.

## Review

A read-only review found no blockers. Its three concerns were repaired before
commit: unsupported dtypes and devices are rejected, the GEMM gate now uses
per-element relative error with an explicit absolute floor, and the receipts
carry source-parent, dirty-state, and exact-command provenance. The reviewer's
full output is retained under
`build/reports/dflash/workers/fused-mxfp4-final-review.txt`.
