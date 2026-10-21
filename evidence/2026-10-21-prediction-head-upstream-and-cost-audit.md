# Prediction-head upstream and cost audit

## Claim

The current Qwen3.8-27B serving decision remains owned MXFP4 target plus the
local DFlash2 runtime at about 51 tok/s. The EAGLE-4 family is not yet
competitive on this M4 Max path, but the reason is not simply MXFP4 target
quality: EAGLE-4 emits only 1.684 tokens per cycle, recomputes a sequential
chain, and pays small-M compute/launch overhead. Selective W4 quantization of
the EAGLE-4 body improves its measured median from 32.083 to 34.522 tok/s
while preserving exact native IDs, but still trails DFlash2 badly.

## Upstream audit

### MLX

Local owned head before audit: `928ed68d1`.
Upstream head: `52a6c234ab35ddf46c8bd1517b849b59b610d078`.
There were seven upstream-only commits.

Relevant commits:

- `bff10220c`: staged affine QMM split-K kernels.
- `52a6c234a`: FP4-to-half fast route in `fp_qmm_t`.

An isolated full merge built as
`mlx==0.32.5.dev20261021+acba9bfa9`. Its own FP4-loader test passed all 24
subtests, but the merged quantized test file reported 252 failed test cases,
55 passed, 1 skipped, with 5,682 subtests passed. The existing fork-specific
split-K precision gate failed because errors were roughly 2x the required
single-rounding threshold.

The isolated `52a6c234a` cherry-pick built as
`mlx==0.32.5.dev20261021+eb06bb3b2`. Its FP4-loader test passed, but 12
split-K precision subtests failed on M4 Max.

Measured MXFP4 matmul p50 values:

| Shape | MLX 0.32.1 | FP4 fast route | Full upstream merge |
|---|---:|---:|---:|
| M=5, N=17408, K=5120 | 0.336 ms | 0.320 ms | 0.334 ms |
| M=8, N=17408, K=5120 | 0.475 ms | 0.417 ms | 0.432 ms |
| M=32, N=17408, K=5120 | 0.749 ms | 0.633 ms | 0.614 ms |

The isolated route was then run through the real EAGLE-4 model. Exact native
IDs passed, but warm throughput was 32.164 and 32.928 tok/s, below the current
34.522 tok/s selective-W4 median. DFlash2 was also effectively unchanged:
warm 51.852 tok/s upstream versus 51.984 tok/s baseline. Therefore neither QMM
commit is adopted for serving. They remain candidates only if a later
numerical review deliberately changes the split-K acceptance contract.

Other upstream MLX commits were M1/M2 SDPA fixes, CPU JIT cache configuration,
`broadcast_arrays` API behavior, and a scalar/contiguous-view memory-correctness
fix. They are not urgent for this M4 workload; the memory fix should be
revisited in the next broad synchronization after the QMM precision conflict
is understood.

### MLX-LM

Local owned head before sync: `d3eed60`.
Upstream head: `72be8c955d9a0d3099bd01865af288e0e947c3fa`.
There were six upstream-only commits.

Adopted into owned `main` at merge `066d9e6`:

- `e7d21f4`: require MLX 0.32.3, protecting `global_scale` quantized MoE APIs.
- `8b37582`: harden THINK-token detection for vocab and template variants.

Focused tokenizer validation with IPv4 DNS forced passed: 3 tests, 6
deselected. The owned `main` now requires `mlx>=0.32.3` on Darwin.

Deferred:

- `ab7f704` improves stop-word and tool-call-in-reasoning semantics, but changes
  the server TextStateMachine API and has two known textual merge points. It
  should be part of a full server synchronization, not a lone cherry-pick.

Irrelevant to Qwen3.8-27B in this audit: PLaMo3 YaRN, AFM7 KV-reuse offsets,
and ERNIE-4.5 correction-bias routing.

## New artifacts

Downloaded with `hfd` and full verification through a process-local IPv4 DNS
shim; the system network configuration was not changed.

| Artifact | Revision | Size | Purpose |
|---|---|---:|---|
| `RedHatAI/Qwen3.8-27B-speculator.dspark` | `87ca2fdc67f316f6c1f9ebc7ef7bbb0ad299a4ce` | 3,976,883,362 bytes | DSpark five-layer head |
| `mlx-community/Qwen3.8-27B-MTP-4bit` | `b643c01b6d3b094e325edb6ebd832e16c486c575` | 265,665,065 bytes | Native MTP W4 adapter |

The RedHat DSpark checkpoint uses the `speculators` config schema. A metadata
only normalized view was created at
`/Users/kearm/models/mlx-vlm-Qwen3.8-27B-DSpark`; it hard-links the original
verified safetensors payload. A local RTN W4 export is at
`/Users/kearm/models/mlx-vlm-Qwen3.8-27B-DSpark-W4` and is 1,209,966,546
bytes. These are research artifacts, not calibrated publications.

Compatible Medusa, Hydra, EAGLE-3, prompt-lookup, and n-gram artifacts were
not found for Qwen3.8-27B. The public ecosystem is dominated by DFlash2, DSpark,
and native MTP.

## Tensor geometry

EAGLE-4 BF16 checkpoint, 6,178,152,384 bytes:

- Target-shared embedding: 2.543 GB.
- Exact full head: 2.543 GB.
- One-layer body: 0.870 GB.
- Target-feature `fc`: 0.157 GB.
- Rank-128 selector: 0.065 GB.

The full file size is therefore misleading. During b2 chain drafting, the
actually streamed work is approximately two passes over body plus `fc` plus
rank retrieval, about 2.185 GB, while the embedding is looked up and only 128
exact-head rows are gathered. The new selective W4 mode quantizes only the
body and `fc`; the token embedding, exact full head, and candidate selector
remain full precision.

DFlash2:

- BF16 selector file: 3.849 GB.
- Runtime W4 body used by the winning path: 1.083 GB.
- Five non-causal draft layers and rank-256 selector.

Native MTP W4:

- Model payload: 238.931 MB.
- One recurrent one-token layer.
- Shares target embedding and full head.

DSpark BF16:

- Model payload: 3.977 GB.
- Five draft layers, eight auxiliary target layers.
- Markov rank 256 and confidence head.
- Local W4 payload: 1.210 GB.

## Measurements

All runs use the same 67-token prompt and 32 generated tokens. Model loading
is excluded. The mlx-vlm rows use upstream `mlx-vlm` 0.7.9 and MLX 0.32.4; the
local rows use the owned DFlash MLX runtime and MLX 0.32.1. This is a
cross-runtime comparison, not a single-kernel benchmark.

| Runtime | Head | Block | Median tok/s | Round behavior | Exact same-runtime IDs |
|---|---|---:|---:|---|---|
| Local | Native | default | 26.149 | one token | n/a |
| Local | EAGLE-4 BF16 draft | b2 | 32.083 | 1.684 tok/round | yes |
| Local | EAGLE-4 selective W4 | b2 | 34.522 | 1.684 tok/round | yes |
| Local | DFlash2 W4 selector | b8->b5 | 50.701 | 4.000 tok/round | yes |
| mlx-vlm | Native | default | 26.623 | one token | n/a |
| mlx-vlm | Native MTP W4 | b2 | 27.405 | 1.880 tok/round | yes |
| mlx-vlm | Native MTP W4 | b3 | 25.156 | 2.460 tok/round | yes |
| mlx-vlm | DFlash2 RTN MXFP4 | b8 | 28.762 | 2.580 tok/round, avg draft 2 | yes |
| mlx-vlm | DSpark BF16 | b5 | 24.680 | 3.560 tok/round | yes |
| mlx-vlm | DSpark W4 | b5 | 24.568 | 3.560 tok/round | yes |

The first MTP CLI smoke stopped at EOS and reported 100% draft acceptance over
six rounds. It is not included in the forced-32-token table.

Selective EAGLE-4 W4 runs were 25.727 cold and 34.522/34.710 warm. The BF16
control runs were 32.040/32.083/32.266. Exact native greedy IDs passed in all
selective-W4 runs.

## Cost model

Measured local cycle timing:

- DFlash2: 78.894 ms/cycle, 4.000 tokens/cycle.
  Naive target-plus-draft weight streaming is 19.110 GB, or 40.57 ms at 471
  GB/s, leaving about 38.32 ms of compute, attention, head work, and overhead.
- EAGLE-4 BF16 b2: 49.683 ms/cycle, 1.684 tokens/cycle.
  A naive full-file target-plus-draft model is 24.205 GB, but the actual chain
  does not stream the EAGLE embedding or complete exact head. Its small-M
  sequential work and low acceptance dominate.

Geometry-derived FLOPs, before attention and launch overhead:

- EAGLE-4: about 0.937 GFLOP per drafted token for body, `fc`, and candidate
  rescoring. Two chain steps plus target verification are about 6.96 GFLOP per
  1.684 emitted tokens, or 4.13 GFLOP/token.
- Native MTP: about 0.841 GFLOP per recurrent draft-token body step, plus a
  2.541 GFLOP full-vocabulary head pass per drafted token. At b2 this is about
  6.30 GFLOP per emitted token including verification.
- DFlash2: about 3.610 GFLOP for the five draft layers plus `fc` for the block.
  Two M=5 full-head passes, drafting and verification, are about 25.410 GFLOP.
  At four emitted tokens this is about 7.25 GFLOP/token.
- DSpark: about 3.316 GFLOP for five draft layers, 0.419 GFLOP for `fc`, plus
  Markov and full-head work. At the measured 3.56 tokens/round this remains
  compute-heavy on M4.

The M4 Max full-vocabulary BF16 head takes about 5.551 ms at M=1 and 5.584 ms
at M=5, showing that it is bandwidth-dominated rather than compute-limited at
decode batch sizes. A large BF16 matmul ceiling measured 14.05 TFLOP/s at
8192^3.

This validates the compute-plus-bandwidth theory: EAGLE has lower derived
FLOPs per emitted token than DFlash2, but its sequential M=1 launches and low
acceptance erase that advantage. DFlash2 pays more compute per round and
successfully amortizes it over four accepted tokens.

## NVIDIA comparison

The relevant NVIDIA measurements are not transferable to M4 Max:

- RTX 5070 SM120 MXFP4 DFlash2 shapes reached 294-308 TFLOP/s at M=8192 and
  117-204 TFLOP/s at M=128.
- M4 Max MXFP4 microbenchmarks in this audit reach about 10 TFLOP/s at large
  M and only 0.23-3.8 TFLOP/s at M=1-8 decode shapes.
- DSpark was trained/evaluated on H100/H200-class systems. Its reported
  validation acceptance length of 4.507 and long-context behavior do not
  transfer to Metal execution.

NVIDIA implementations can hide sequential draft work with higher tensor-core
throughput, CUDA graphs, and batched block kernels. On M4, full-head and
small-M launch costs change the ranking.

## Current decision

Keep the local DFlash2 W4 selector as the serving incumbent. Use explicit
`--draft-quant w4` for future EAGLE-4 measurements now that selective body
quantization is implemented. EAGLE-4 needs committed draft KV history and tree
proposals before it can be judged as a family ceiling.

## Non-claims

- These are short-context, single-request, 32-token decode measurements.
- The mlx-vlm rows use a different runtime and chat-template behavior from the
  local DFlash harness; they establish that those heads run, not identical
  kernel efficiency.
- DSpark and MTP exactness was checked against the same mlx-vlm native token
  stream, not the local raw-prompt harness token IDs.
- The DSpark W4 export is RTN research packaging, not calibrated GPTQ/MXFP4.
- No broad perplexity, KL, long-context, batching, or multimodal quality gate
  was run.
- No external MLX, MLX-LM, DFlash, EAGLE, or Hugging Face repository was
  modified.
