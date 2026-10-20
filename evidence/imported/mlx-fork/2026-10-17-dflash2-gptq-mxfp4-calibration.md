# DFlash2 GPTQ MXFP4 calibration and distillation staging

Date: 2026-10-17

## Scope

This records the first campaign-owned native MXFP4 quantization of
`z-lab/Qwen3.8-27B-DFlash2`, measured against the calibrated
`RESMP-DEV/Qwen3.8-27B-MLX-GPTQ-MXFP4` target. It also records the staged
target-aware distillation harness. It does not claim a completed distillation
run, a public release, selector-codebook quantization, or representative
multi-user serving performance.

## Inputs and capture

The private DFlash-MLX capture harness ran 24 real corpus prompts through the
calibrated target and affine-W4 DFlash2. Each prompt generated 256 tokens with
EOS suppressed. The run captured 1,981 target-verified speculative cycles in
358.437 seconds.

Affine-W4 acceptance across the capture prompts was:

- minimum: 57.813%;
- mean: 66.590%;
- maximum: 79.297%.

Each captured cycle contains:

- projected target context;
- raw 25,600-wide target context where available;
- target-derived mask-block noise embedding;
- anchor and drafted IDs;
- target greedy IDs;
- target top-64 IDs and logits.

Capture receipt:
`build/reports/dflash/distill-capture/run6/receipt.json`.

## Sequential GPTQ solve

The solve used the private `mlx-gptq-mxfp` worktree at commit `0e7c94a`, the
DFlash2 PyTorch reference, and the existing MXFP4 safe-scale GPTQ solver.
Large Hessians were solved on CPU while bounded layer solves used the RTX 5070.
The full solve covered 1,981 cycles and 47 draft linear tensors:

- median relative error: `0.118944`;
- maximum relative error: `0.176118`;
- elapsed time: 789.614 seconds.

The two selector codebooks remain BF16 in this first artifact. Their combined
storage is approximately 254 MB and they were deliberately left outside this
first quantization boundary.

## Packed artifact

Local artifact:

`/Users/kearm/models/z-lab-Qwen3.8-27B-DFlash2-GPTQ-MXFP4-1981cycles`

Properties:

- native MLX MXFP4;
- bits: 4;
- group size: 32;
- calibrated tensors: 47;
- total checkpoint tensors: 128;
- size: 1,209,483,344 bytes;
- SHA-256:
  `b15239aa185d00370da0060c8676d07da1584790a326800efdac725e3588dadb`;
- selector codebooks: BF16.

The artifact is approximately 3.1x smaller than the 3,848,817,896-byte BF16
DFlash2 source.

The private runtime now detects a prequantized draft from its config and reports
`draft_quant_source=prequantized` without applying the affine-W4 registry
default again. A real 32-token no-flag smoke generated normally at 42.8 tok/s
with 68.8% acceptance.

## Validation

A strict MLX load materialized `fc`, layer-zero attention projections, MLP
projections, and the final layer projections as native MXFP4. The 256-token
real-target parity run matched every target token:

- exact target-ID match: 256/256;
- acceptance: 72.656%;
- DFlash2 tok/s: 51.852;
- stock target tok/s: 25.606.

Three-repeat, 256-token DFlash-only measurement on the standard math prompt:

- median throughput: 53.678 tok/s;
- median acceptance: 72.656%;
- peak MLX memory: 19.654 GB.

## Corpus and held-out comparisons

Across the same 24 capture prompts, calibrated MXFP4 was effectively tied with
affine W4:

- affine-W4 mean acceptance: 66.590%;
- calibrated-MXFP4 mean acceptance: 66.683%;
- mean delta: +0.093 percentage points;
- per-prompt results: 10 wins, 5 ties, 9 losses;
- calibrated-MXFP4 median throughput: 32.538 tok/s.

On a separate three-prompt held-out diagnostic:

- affine W4 mean acceptance was 67.198%;
- calibrated MXFP4 mean acceptance was 68.500%;
- mean acceptance delta was +1.302 percentage points;
- mean throughput delta was +0.342 tok/s;
- two prompts tied and one favored calibrated MXFP4.

These are small diagnostics, not broad quality claims. The main measured benefit
is durable memory reduction while retaining exact target verification.

## Distillation staging

The private quantizer branch includes a target-aware selector trainer. The
current bounded objective trains only
`candidate_selector.hidden_projection` with teacher-greedy cross-entropy over
captured target top-64 candidates. The transformer, target projection, and both
selector codebooks are frozen.

The DFlash2 PyTorch reference replay now consumes the already projected context
directly, matching the MLX runtime boundary. The initial CUDA uint32 indexing
defect was corrected locally. The trainer has passed Python compilation, Ruff,
and formatting checks, but it has not completed a training run.

Per the user's pause instruction, no further Strix work was launched after the
machine became needed for other measurements.

## Code receipts

- Private DFlash-MLX capture harness:
  `RESMP-DEV/dflash-mlx`, branch `work/local-mxfp4-dflash2`, commit
  `113af61`.
- Private GPTQ calibrator, packer, and staged trainer:
  `RESMP-DEV/mlx-gptq-mxfp`, branch `work/dflash2-calibration`, commit
  `0e7c94a`.
- Focused pack tests: 2 passed.
- Ruff and Python compilation passed for the new calibration, pack, capture,
  and distillation modules.

No Hugging Face upload was performed for this DFlash2 artifact.
