# Local BF16 PLE storage and later Optane placement

## Purpose

Run Qwen3.8-Flash-Next locally before the Optane volume is permanently mounted.
The first serving backend reads exact BF16 PLE rows from the verified source
shards on the internal SSD. The second backend moves those immutable shards to
Optane without changing values or tokenizer behavior. MXFP4 PLE is a separate,
measured alternative, not the default.

Priority order:

1. Correct teacher activations and PLE values.
2. Bounded resident memory and deterministic resumability.
3. End-to-end token latency, including queueing and page behavior.
4. Disk capacity savings.

## Ownership

| Boundary | Owner | Current position |
|---|---|---|
| Verified BF16 source | Mac and B550 | Complete, 144/144 files, full hash verification |
| Stage A calibration | B550 | Complete |
| Packed MXFP4 artifact | Mac and B550 | Complete and locally hash-verified |
| Local PLE backend | Mac runtime | Implemented; two real Metal smokes passed |
| Optane backend | Mac runtime + Thunderbolt volume | Wait for mount; copy exact BF16 shards |
| PLE format change | Campaign | Blocked on same-index quality and latency gates |

## Storage facts

- Pinned source revision: `de4b8e4d43b917e7706784d8bb445c9af86a3540`.
- Complete source tree: 144 files, `360,023,351,514` bytes.
- PLE logical rows: `320,001,536`.
- Values per row: `160`.
- BF16 bytes per row: `320`.
- Exact BF16 PLE plane: `102,400,491,520` bytes, `95.367 GiB`.
- Source PLE shards: 128 tensors, `shard_0` through `shard_127`.
- PLE tensors occur across source shards 5 through 37.
- Native MXFP4 PLE plane would be `27,200,130,560` bytes, `25.332 GiB`.

The verified BF16 PLE manifest is:
`build/reports/local-ple-storage/ple-bf16-source-manifest.json`.
Its SHA-256 is
`6e134b4cb51881af76b69d5b364ccc12dc1f83787f1ad3b9513ba58dbbba8ce2`.

The manifest deliberately names the original model shard files rather than
pretending that PLE is one physical tensor. A logical global row maps to:

```text
shard = row // rows_per_shard
row_in_shard = row % rows_per_shard
```

The source safetensors index supplies the file and tensor name for each shard.

## Tier L0: immutable local source

The internal-SSD source directory remains the master:

```text
/Users/kearm/models/Qwen3.8-Flash-Next-BF16
```

Invariants:

1. Do not rewrite source shards.
2. Do not merge the 128 PLE tensors into a new 95.37 GiB file.
3. Read only selected rows during calibration or inference.
4. Verify shard size and SHA-256 before enabling a backend.
5. Preserve the full Hugging Face receipt and revision.

This tier has already passed full verification on Mac and B550.

## Tier L1: internal-SSD local serving

The first local model uses:

- native MXFP4 resident backbone tensors;
- BF16 embedding and language-model head unless a later measured override says
  otherwise;
- exact BF16 PLE rows read from the original source shard files.

Expected logical footprint:

| Component | Estimate |
|---|---:|
| MXFP4 backbone | 63.73 GiB |
| BF16 embedding + LM head | 2.37 GiB |
| BF16 PLE source shards | 95.37 GiB |
| Total serving working set | 161.47 GiB |

The internal Data volume had approximately 2.6 TiB available when this document
was written. Tier L1 therefore fits without copying or converting the PLE plane.

Runtime contract:

1. A PLE backend descriptor selects `internal_ssd`.
2. The loader opens the 128 source safetensors files lazily.
3. Selected global IDs are mapped to shard and local row on the host.
4. Only requested rows are copied to MLX.
5. Admission is bounded across all in-flight shard files.
6. A fixed-prompt run must record row count, cache hits, page-read estimate,
   wait time, dequantize/project time, and resident peak.
7. Missing, truncated, or hash-mismatched shards disable the backend.

This is the local version to bring up before waiting for Optane. It is also the
correct control for later media comparisons because values remain bit-identical.

## Tier L2: Optane placement

Optane becomes a media change, not a model-format change.

Proposed target:

```text
/Volumes/OPTANE/qwen38-flashnext/ple-bf16/
```

Copy the 128 referenced model shard files listed in the manifest, or expose
dedicated per-PLE-shard files only if a later packer makes that necessary. Do
not convert the values during this move.

Promotion procedure:

1. Mount and identify the Optane filesystem and device serial.
2. Copy every referenced shard with resumable transfer.
3. Verify each file against the manifest SHA-256 and size.
4. Create a backend descriptor with `media: optane`.
5. Run the same prompt and same selected-row trace used for Tier L1.
6. Compare end-to-end tokens/s, p50/p99 PLE wait, admission queue depth, cache
   hits, and host/GPU memory.
7. Switch only when Optane meets the declared latency gate.

Fallback contract:

```text
ple_backend=optane,verified=false -> refuse to start
ple_backend=optane,io_error -> stop or fall back to internal_ssd explicitly
```

Silent fallback is not allowed.

## Why BF16 is the default PLE storage

The current Stage A teacher already uses exact selected BF16 PLE reads. The real
PLE layer completed without allocating the fused table. Quantizing PLE before
calibration would change the teacher for every later layer.

For serving, the logical size reduction is large:

```text
95.367 GiB BF16 -> 25.332 GiB MXFP4
```

But row width and page granularity matter:

- BF16 row: 320 bytes.
- MXFP4 row: 80 code bytes + 5 scale bytes.
- Page size on the prior Optane proof: 16 KiB.
- MXFP4 requires two planes and potentially two I/O operations per row.

Therefore a 3.765x byte reduction is not assumed to be a 3.765x random-I/O or
end-to-end latency improvement. Native BF16 is the exact, simpler first tier.

MXFP4 PLE may be introduced only as a parallel backend after:

1. the real token trace exists;
2. BF16 and MXFP4 are queried with identical indices;
3. MXFP4 wins end-to-end latency;
4. PLE logits or an equivalent quality gate pass;
5. conversion remains reproducible from the retained BF16 master.

## Open choices

| Topic | Current position | Decision gate |
|---|---|---|
| PLE media | Internal SSD first, Optane second | Same-trace latency and p99 comparison |
| PLE values | Exact BF16 | Quality gate before any MXFP4 serving claim |
| Embedding and LM head | BF16 | Full-artifact size and quality tradeoff |
| Shard-file layout | Use original model shard files | Profiling may justify split PLE-only files |
| Cache policy | Small bounded row cache | Real trace hit-rate and latency distribution |
| Admission bound | Reuse benchmark's bounded ticket model | Soak test under generation |

## Acceptance gates

### L1 internal-SSD gate

- Complete Stage A and Stage B backbone artifact.
- Load resident model without source-tree residency.
- Enable the internal-SSD BF16 PLE backend.
- Run two identical fixed prompts.
- Require finite logits, deterministic PLE row selection, no fused PLE
  allocation, bounded resident peak, and complete backend metrics.

### L2 Optane gate

- Optane is mounted and its device identity is recorded.
- Every referenced BF16 shard matches size and SHA-256.
- Same prompts and row IDs complete successfully.
- End-to-end and p99 latency satisfy the declared target.
- Failure injection proves explicit fallback or refusal.

## Work log

### 2026-10-07

- Mac and B550 full source receipts reached `status: verified`.
- Real Stage A began on B550.
- The real PLE layer completed using exact selected-row BF16 source reads.
- Official and diagnostic tokenizers produced identical full-corpus token
  streams, so the staged 32 by 2048 calibration windows are valid.
- Created the verified 128-entry BF16 PLE source manifest.
- Stage A completed all 48 layers, consumed 74,400 GPTQ records, produced
  native MXFP4/group-32 weights with exact BF16 PLE, and exited 0.
- Stage B completed 34 shards and 2,160 tensors. The resident backbone is
  `171,213,918,824` model bytes on disk; PLE, embedding, and LM head remain
  BF16.
- Implemented `ExternalPleTable` in the `dynamic-mxfp4` MLX-LM worktree. It
  parses safetensors headers, maps global PLE rows to physical offsets, reads
  only selected rows, coalesces contiguous requests, deduplicates requests,
  uses global-row cache keys, and exposes gather/cache/read metrics.
- Added the PLE runtime builder. It removes PLE keys from the loader index and
  streaming-prunes mixed safetensors files so no PLE tensor is loaded by MLX.
  Pure backbone shards are hard-linked on the same filesystem. Tokenizer JSON is
  preserved, while the full-artifact `model.sha256.json` is not copied into the
  runtime because its hashes do not describe pruned runtime shards.
- The tiny control passed exact selected-row parity, duplicate-row scatter,
  global cache-key coverage, PLE-free runtime weights, no resident PLE
  parameters, and resident-versus-external logits equality. Commands run:
  `ruff check` on the new files; `python -m unittest
  tests.test_qwen4_exp_ple tests.test_ple_runtime
  tests.test_models.TestModels.test_qwen4_exp`; all exited 0.
- The fixed-prompt smoke harness also passed on the tiny control through the
  real `generate_step` path: 8 generated tokens, 10 gathers, 72 requested rows,
  8 cache hits, 60 unique disk rows, and 1,920 bytes read. Resident parameters
  were `2,878,392` bytes.
- Built the real B550 stripped runtime at
  `/home/kearm/mlx-dynamic-mxfp4/models/qwen38-flashnext-mxfp4-ple-runtime`.
  It has 13 shards, 2,032 index keys, and `68,813,407,880` model bytes. The
  builder removed 128 PLE keys, omitted 21 PLE-only shards, and pruned the two
  PLE tensors from mixed shard 13.
- Verified the real B550 runtime: all 128 external PLE layouts are BF16 with
  `2,500,012` rows and 160 values; all 84 tensors rewritten from mixed shard 13
  match the source tensor byte ranges by SHA-256. Pure backbone files remain
  hard links to the packed artifact.
- Transferred the complete packed artifact from B550 to
  `/Users/kearm/models/qwen38-flashnext-mxfp4` at about 231 MB/s. All 34 shard
  SHA-256 hashes matched in 63.28 seconds.
- Built `/Users/kearm/models/qwen38-flashnext-mxfp4-ple-runtime` with 13
  shards, 2,032 index keys, and `68,813,407,880` model bytes. All 84 rewritten
  mixed-shard tensors match source byte ranges, 12 backbone files are hard
  links, and six tokenizer files are exact copies.
- Ran two identical real fixed-prompt Metal smokes. Each generated 32/32 greedy
  tokens and produced byte-identical text. Token rates were `1.130` and
  `1.433` tokens/s; resident parameters were `68,813,152,280` bytes; peak RSS
  was `46.80 GB` and `47.25 GB`. PLE gather time was `29.03 ms` and `27.29 ms`.
  Each run requested 672 rows, hit 144 cache rows, read 528 unique rows, and
  read `168,960` bytes.
- Recorded a real Metal System Trace with Xcode Instruments. Recording and TOC
  export exited 0 and produced a 199 MiB trace. Shader Timeline was disabled by
  the template, so no kernel-name attribution is claimed. Trace encoding
  intervals are overhead-influenced and are not accepted as steady latency.
- Measured greedy generation through 257 tokens. Excluding the first token,
  every 32-token bucket had p50 from `27.248` to `27.990` ms; sequence-length
  growth is not the current decode limiter.
- Implemented prompt-cache reset plus PLE row-cache clearing. Warmup now uses
  the real prompt and single-token cache paths, resets their state, and clears
  PLE rows before measurement. Focused `ruff` and nine unit tests exited 0.
  The standard real smoke generated the same 32-token text with `169.544` ms
  first token, `28.062` ms decode p50, `28.598` ms p95, and `1.039` s total
  generation. The earlier cache-less warmup still had `866.218` ms first token.
- Built an isolated trial runtime with the BF16 LM head converted to naive
  native MXFP4/group 32. Resident parameters dropped by `933,683,200` bytes
  and decode p50 dropped to `25.828` ms. The trial is not promoted: short-prompt
  top-1 agreement against BF16 was `40.0%`, mean top-10 overlap was `42.4%`,
  and max absolute logit delta was `31.281`. A calibrated output-head solve is
  required before it can become the default.
- Repaired request warmup to run one real `generate_step`, then reset cache and
  PLE state. A fresh logical request now has `120.220` ms first-token latency,
  `27.171` ms decode p50, and `0.965` s 32-token generation with unchanged
  output. Repeated requests in one process are also about `120` ms.
- Exact SwitchGLU gate/up MXFP4 fusion is retained. Larger PLE caches and
  deeper admission did not improve the fixed trace: both 1M-row arms still had
  144/672 row hits and 528 disk rows.
- Batched decode reached `117.519` aggregate tokens/s at batch 8 and `210.792`
  at batch 32. Batch 32 reused 6,873 of 20,992 requested PLE rows, but
  per-sequence throughput fell to `6.587` tokens/s.
- B550 replayed Stage A and produced calibrated MXFP4 and MXFP8 LM heads.
  MXFP4 and MXFP8 both changed the exact continuation near a step-10 near tie,
  so the exact BF16 head remains the runtime default despite MXFP8's 2.79%
  median weight error and `26.347` ms decode p50.

## Non-claims

- Optane is not currently mounted.
- The two real runs prove correctness and bounded residency, not tuned
  throughput.
- No p50/p99 PLE latency distribution exists yet.
- No BF16-versus-Optane or BF16-versus-MXFP4 real-trace comparison has run.
- No real-model generation-quality claim exists.
- The MXFP4 LM-head runtime is a performance trial only; naive conversion fails
  the short-context logit agreement gate.
- Calibrated MXFP4 and MXFP8 head variants are also rejected for the exact
  output artifact until the near-tie policy is changed deliberately.

## Work log, 2026-10-07 (performance hill climb)

- Added a bounded `issue`/`collect` ticket API to the external PLE table and an
  opt-in `ple_backend.prefetch` model path. Both preserve exact values, but early
  host-side issue synchronizes graph inputs: over three runs each, synchronous
  decode p50 median is 28.084 ms versus 28.704 ms with prefetch, so synchronous
  remains the default.
- Fixed a real defect exposed by batched tickets. Shard grouping used repeated
  `np.append` per row and checked the row cache once per requested row, which is
  quadratic and cost 62.4 ms of issue time at batch 32.
- One-pass list grouping plus a single global `np.unique(return_inverse)` dedup
  keeps byte-identical row selection and drops batch 32 issue time to 16.9 ms.
  Single-stream PLE gather time falls from 21.8 ms to 15.1 ms over 34 gathers.
- Batch scaling after the fix: batch 8 117.6 tokens/s, batch 32 213.4, batch 64
  244.3, batch 80 240.1, batch 96 228.4, batch 128 85.4. The throughput knee is
  batch 64; the latency knee remains batch 8.
- Resident peak memory at batch 64/128 is 34.0 GB RSS with 87.0 GB Metal peak.
- Optane remains disconnected; all PLE numbers are internal SSD.

## Additional probes, later on 2026-10-07

- Vectorizing the host n-gram head modulo changed no selected row but regressed
  real decode p50 to a 28.696 ms median over three runs. It was reverted.
- Disabling the packed GDN kernel produced 28.656 ms p50 and was rejected. The
  packed kernel remains default.
- Disabling MoE expert-index sorting at batch 32 dropped aggregate throughput
  from 213.4 to 175.5 tokens/s. The existing sort threshold remains.
- Moving router weights before the down projection saved 0.0817 ms per isolated
  layer but changed logits by 0.0009765625, so it fails the exactness gate.

## PLE cache-capacity tuning, 2026-10-07

- Added optional real row-ID tracing to the batch benchmark. The B=64 trace
  contains 41,984 requested rows across 32 tickets, 14,960 unique rows, and is
  retained under `build/reports/local-ple-storage/ple-trace/`.
- Replayed the identical trace against 4K, 8K, 16K, 32K, and 1M cache rows.
  Every capacity produced value SHA-256
  `255cf76761522984d01763c578952e7f7f8c1aaed052e9ad0a576d8cd540c732`.
- Direct replay median wall time: 4K `329.681 ms`, 8K `320.947 ms`,
  16K `312.830 ms`, 32K `311.031 ms`, 1M `312.165 ms`. 16K is the smallest
  capacity at the measured plateau and uses about 5 MiB of payload.
- Repeated model runs also favor 16K: single-stream p50 median `27.570 ms`
  versus `28.069 ms` for 4K, and B=64 aggregate `244.251` versus `243.569`
  tokens/s. Greedy text is unchanged.
- The Mac and B550 campaign runtimes now set `ple_backend.cache_rows=16384`.
- Replayed the same trace at admission depths 8, 16, 32, 64, 128, and 256.
  Median wall times were 283.346, 276.375, 272.267, 280.874, 277.517, and
  288.093 ms respectively. The existing depth 32 remains best.
- Added explicit `--max-batch-chunk` diagnostics. At B=128, monolithic decode
  measured 63.31 tokens/s while two B=64 chunks have 16.902 s combined steady
  decode wall time, or 242.34 tokens/s. This is not an exact-output runtime
  policy: sampled B=1 sequences differ from both B=128 and B=64 batch paths at
  near ties. Chunking remains diagnostic/scheduling-only.
- Re-tested moving router weights before the down projection with an opt-in
  path. Over three 64-token runs the p50 median was 27.284 ms off versus
  27.304 ms on, and greedy text diverged at the first near tie. It is not
  promoted.
- Re-tested Metal n-gram hashing after global row deduplication. Interleaved
  single-stream runs improved p50 median from 27.233 to 26.594 ms with unchanged
  text. Metal hashing is slower for batches, so the campaign default is
  `hash_backend=auto`: Metal for batch size 1 and host for larger requests.
  The runtime builder now emits this default, and a 64-token canonical run
  retained exact output with 26.476 ms p50.
- Fused the shared expert's packed MXFP4 gate and up projections into one
  quantized matmul. The isolated layer output is bit-identical and saves
  0.0207 ms/layer. All 48 shared experts fuse at load time. Repeated exact
  64-token runs improved p50 median from the prior 26.476 ms control to
  26.104 ms; p95 is 26.542 ms and first token is 117.060 ms.
- Fused full-attention key/value packed MXFP4 projections. All 12 full-attention
  layers fuse at load, K and V stay bit-identical, and the isolated layer saves
  0.0086 ms. Three exact 64-token runs improved p50 median from the prior
  26.104 ms control to 25.951 ms with unchanged output.
- Fused GDN projections that share an input: QKV+Z in all 36 linear-attention
  layers and B+A in the same layers. Every output is bit-identical. Three exact
  64-token runs improved p50 median from the prior 25.951 ms control to
  25.900 ms with unchanged output.
- Rejected three exact probes this cycle. Fusing Q with the packed K/V path
  saved 0.0494 ms per full-attention layer in isolation but did not improve real
  p50 over five runs (25.912 ms versus the 25.900 ms accepted baseline), so it
  was reverted. Fusing PLE key/value projections saves only 0.0023 ms. Caching
  the attention index lists as static attributes was within noise and reverted.
  A post-revert confirmation run measured 25.887 ms p50 with unchanged output.
- Added an explicit `greedy_fast` generation path. It selects argmax from raw
  logits and skips log-sum-exp normalization when normalized scores are not
  needed. The default API still returns normalized log probabilities, while the
  temperature-zero CLI uses the fast path automatically. Three exact 64-token
  benchmark runs improved p50 median from 25.900 to 25.814 ms with unchanged
  output.
- Rejected a vectorized Metal n-gram hash: exact and faster in isolation, but
  real p50 regressed to 25.940 ms. Rejected a mixed BF16/MXFP8 LM head: only 34
  BF16 rows were needed for canonical winners, but the mixed matmul was slower
  than BF16.
- Promoted `compile_moe=true` for exact batch-1 decode. The compiled wrapper is
  used only when batch and sequence length are both one; prompt prefill and
  batched requests stay eager. Three config-driven exact runs improved p50
  median from the 25.896 ms eager control to 25.336 ms; p95 is 25.912 ms and
  first token is 114.791 ms. Mac and B550 campaign configs now enable it.
- Sorting the ten batch-1 expert IDs regressed p50 to 27.308 ms. The existing
  sort threshold of 64 routed jobs remains.
- Promoted `compile_residual=true`. All 96 per-layer hyper-connection mixers and
  the final mixer compile for batch-1, sequence-length-1 decode only; prefill
  and batched requests stay eager. Three interleaved exact runs improved p50 in
  every pair, from 25.296/25.355/25.524 ms to 24.994/24.964/25.073 ms. Mean
  paired improvement is 0.381 ms p50 and 0.542 ms p95, with no warmup
  regression. A durable-config run measured 24.878 ms p50, 25.337 ms p95, and
  unchanged text. Receipt:
  `build/reports/local-ple-storage/compiled-residual-decode-20261007.json`.
- Rejected and reverted a temporary `compile_gdn` probe. Compiling only the GDN
  projection and output stages was exact, but median p50 improved only 0.142 ms
  while median p95 regressed 0.105 ms; paired results also did not win
  consistently. Receipt:
  `build/reports/local-ple-storage/gdn-compilation-rejected-20261007.json`.
- Rejected and reverted PLE post-projection compilation. It improved paired p50
  by only 0.056 to 0.127 ms, regressed p95 in two pairs, and produced a 10.502 s
  warmup outlier. The external lookup and cache-mutating convolution were never
  compiled. Receipt:
  `build/reports/local-ple-storage/ple-post-compilation-rejected-20261007.json`.
- Rejected and reverted full-attention projection/output compilation. Every p50
  pair improved, but one p95 pair regressed and one candidate warmup reached
  11.769 s. Receipt:
  `build/reports/local-ple-storage/attention-compilation-rejected-20261007.json`.
- Final post-probe audit passed the 20-test matrix and reproduced exact output
  at 24.971 ms p50 and 25.497 ms p95. No probe flags remain. The final source
  bundle SHA-256 is
  `af559d378dde6d071f51724ec1c891b2850c1dc4386110f70fa7e7c920f34f80`, and B550
  verified that bundle plus `compile_moe=true` and `compile_residual=true`.
- A cProfile pass found no material Python decode bottleneck, but attributed
  10.369 seconds of real model load to 48 runtime gate/up concatenations.
  Removing the per-layer MLX cache clear did not improve load time and was
  rejected. Receipt:
  `build/reports/local-ple-storage/fuse-cache-clear-rejected-20261007.json`.
- Added `text_config.prefused_moe` and a one-shard-at-a-time artifact builder.
  The tiny artifact has exact logits with zero `up_proj` modules. The real
  13-shard runtime contains 48 `gate_up_proj` pairs; all shard SHA-256 hashes
  verify. Three interleaved real runs preserved every continuation and reduced
  median cold-process load+parameter+forward time from 23.907 to 12.195
  seconds. Decode p50/p95 are not claimed to improve. Receipt:
  `build/reports/local-ple-storage/prefused-moe-runtime-20261007.json`.
- Final verification passed 21 focused tests and reproduced exact text from the
  pre-fused artifact. B550 holds the matching source bundle SHA-256
  `2e9f6ad90c6ac06fe1f3d226a354ec981a5bb608fd6ca01271c74919abac3002`; its
  existing split-layout runtime remains safely unchanged.
- The inherited split-runtime hash sidecar was removed and replaced by
  `prefused-runtime.sha256.json`. It describes 13 shards, 1,936 index tensors,
  68,813,394,804 model bytes, and zero split `switch_mlp` gate/up keys.
- Two interleaved rounds at batch 8, 32, and 64 confirm the split and
  pre-fused artifacts have equivalent batched throughput and decode latency.
  Mean B64 aggregate throughput is 246.885 versus 245.935 tokens/s; no batch
  decode improvement is claimed. Receipt:
  `build/reports/local-ple-storage/prefused-batch-parity-20261007.json`.
- Rejected and reverted an interleaved packed-GDN lane layout. It was exact and
  saved 0.02852 ms in the isolated decode kernel, but full-model p50 stayed at
  24.971/24.968 ms and every forward-warmup pair regressed by about 0.15 to
  0.20 s. Receipt:
  `build/reports/local-ple-storage/gdn-interleave-rejected-20261007.json`.

## Compiled block-injection addendum, later on 2026-10-07

The 96 per-token block-injection combines now share one compiled function,
used only for batch-1 sequence-1 decode; a single module-level capture serves
every call site instead of one capture per layer. Three interleaved real
64-token runs improved decode p50 median from 24.960 to 24.849 ms and p95
median from 25.480 to 25.260 ms, with unchanged output and lower warmup.
`text_config.compile_injection` is promoted in both Mac artifacts.

## Internal-SSD reader probes, later on 2026-10-07

- Page-coalesced `pread` is exact but slower on the real B64 trace: 5.125 ms
  versus 4.204 ms median. It is rejected.
- Per-shard `mmap` is exact and wins the isolated B64 replay, 1.583 ms versus
  4.176 ms, but regresses real decode p50 by about 2.0 ms and p95 by about
  8.1 ms under the full model working set. The implementation remains available
  behind `ple_backend.read_backend="mmap"` for diagnostics, but `pread` stays
  the default.
- macOS `F_NOCACHE` is exact but slower on the real single-stream trace replay,
  23.645 ms total versus 17.426 ms. It is rejected.
- The real single-stream trace contains 66 gathers and 1,184 rows. Direct
  numpy-row replay completes in about 17 ms once hot, while in-model gather
  timing is about 266 ms because it includes synchronization on the Metal hash
  dependency. This reinforces the earlier conclusion that PLE storage is not
  the current decode bottleneck.

## Additional exact rejections, later on 2026-10-07

- Fusing the two GDN Q/K RMS-normalization launches is bit-identical, but
  paired p50 deltas are +0.066, -0.270, and -0.228 ms and median p95 regresses
  by 0.136 ms. It was reverted. Receipt:
  `build/reports/local-ple-storage/qk-norm-fusion-rejected-20261007.json`.
- Decomposing the earlier full-attention compile into projection-only
  compilation won the first three p50 pairs but lost pairs 4 and 5; p95 lost
  the last three pairs. It was reverted after five interleaved pairs. Receipt:
  `build/reports/local-ple-storage/attention-projection-compile-rejected-20261007.json`.
- The final post-rejection source bundle, including the 32-test focused matrix,
  has SHA-256
  `9dd786761af44b1b875a8f076eadf4bcf5620b91214b7e4a0a0f312954c7a288`; B550
  verified the same bundle.

## Greedy token-only and attention-output probes, later on 2026-10-07

- Rejected and reverted a temporary `greedy_token_only` API plus compiled
  LM-head argmax. It preserved exact text but median p50 regressed from 24.760
  to 24.809 ms. The default `greedy_fast` logits contract remains unchanged.
  Receipt:
  `build/reports/local-ple-storage/greedy-token-only-rejected-20261007.json`.
- Promoted `compile_attention_output=true`. Only each full-attention layer's
  gated output projection compiles for batch-1 sequence-1 decode. Seven
  interleaved runs improved median p50 from 24.844 to 24.722 ms and median p95
  from 25.431 to 25.344 ms with unchanged text. A durable run reached 24.664 ms
  p50 and 25.308 ms p95. Receipt:
  `build/reports/local-ple-storage/compiled-attention-output-decode-20261007.json`.

## GDN all-projection fusion addendum, later on 2026-10-07

The already-packed QKV+Z and B+A GDN projections now optionally concatenate
into one MXFP4 gather per linear-attention layer. The split restores QKV, Z,
B, and A in their original order, so every output is bit-identical. Five
interleaved real runs with `fuse_gdn_all=true` improved median p50 from 24.720
to 24.590 ms and median p95 from 25.275 to 25.087 ms with unchanged text.
The promoted durable run reached 24.579 ms p50 and 25.291 ms p95. Receipt:
`build/reports/local-ple-storage/gdn-all-projection-fusion-20261007.json`.

## Shared GDN output compile addendum, later on 2026-10-07

The gated RMS norm and quantized GDN output projection now share one compiled
graph across all 36 linear-attention layers; each layer passes its norm weight
and packed output rows as arguments. Seven interleaved runs improved every
paired p50, with median p50 moving from 24.715 to 24.432 ms and median p95 from
25.315 to 25.006 ms. Six of seven p95 pairs improved. Median cold-process
preparation rose only 0.047 seconds, although the forward-warmup receipt field
rose 0.107 seconds. A durable run reached 24.560 ms p50 and 25.007 ms p95.
Receipt:
`build/reports/local-ple-storage/compiled-gdn-output-decode-20261007.json`.

## Router/shared-gate rejection, later on 2026-10-07

Concatenating the BF16 shared-expert gate row onto the BF16 router matmul was
bit-identical for 192/192 probes across all real layers, but regressed candidate
p50 in all five real pairs and reached a 25.898 ms p50 outlier. It was rejected
and reverted. Receipt:
`build/reports/local-ple-storage/router-shared-gate-fusion-rejected-20261007.json`.

## GDN projection-compile rejection, later on 2026-10-07

After `fuse_gdn_all`, a shared compiled projection/split graph was exact but
still did not improve the real model. The first two pairs occurred under
background load; once conditions cleared, pairs 3-5 all regressed p50 and p95.
The probe was rejected and reverted. Receipt:
`build/reports/local-ple-storage/gdn-projection-compile-rejected-20261007.json`.

## Shared attention-output rejection, later on 2026-10-07

Replacing the twelve accepted per-layer attention-output captures with one
shared parameterized graph was exact but did not clear the decode gate. Four of
five p50 pairs improved, yet median p50 regressed by 0.040 ms and only two of
five p95 pairs improved. It was rejected and reverted. Receipt:
`build/reports/local-ple-storage/shared-attention-output-rejected-20261007.json`.

## Batch and indexer/K/V checks, later on 2026-10-07

- Two current B8/B32/B64 runs show no material batch-throughput change from the
  later single-stream promotions: B32 and B64 move by less than 0.5 tok/s, and
  B8 varies by about 2.5 tok/s relative to the earlier campaign. Receipt:
  `build/reports/local-ple-storage/current-batch-after-gdn-all-20261007.json`.
- Fusing each QSA indexer K/Q projection onto its attention K/V gather is exact
  but improves median p50 by only 0.020 ms while median warmup regresses by
  0.408 seconds. It was rejected and reverted. Receipt:
  `build/reports/local-ple-storage/index-kv-fusion-rejected-20261007.json`.

## Pre-fused GDN artifact addendum, later on 2026-10-07

The real artifact now stores each linear-attention layer's QKV, Z, B, and A
rows in one packed GDN projection at artifact-build time. `prefused_gdn=true`
loads that tensor directly; `fuse_gdn_all=false` records that no runtime
concatenation is needed. The tiny control has exact logits with zero residual
split modules. All 36 real layers and all 13 rewritten shards verify by
SHA-256. Five interleaved real runs improved median p50 from 24.525 to 24.436
ms and median p95 from 25.252 to 24.953 ms with unchanged text. A durable run
reached 24.419 ms p50 and 25.188 ms p95. Receipt:
`build/reports/local-ple-storage/prefused-gdn-runtime-20261007.json`.

## PLE bounded-coalescing rejection, later on 2026-10-07

Bounded-gap `pread` coalescing from 512 bytes through 16 KiB was tested on the
real single-stream trace. Every threshold still issued 1,184 reads for 1,184
rows because the smallest same-shard row gap is 7,091,200 bytes. Values stayed
identical, but per-ticket median time did not improve. Bounded coalescing is
rejected; contiguous-range coalescing remains sufficient. Receipt:
`build/reports/local-ple-storage/ple-bounded-coalescing-rejected-20261007.json`.

## Unified shared-expert rejection, later on 2026-10-07

Appending the MXFP4 shared expert as an extra switch expert was bit-identical
in a focused block, but copies nearly the full routed expert tensors at load.
Two completed real pairs raised load to 17.4-17.5 seconds, forward warmup to
44.4-48.5 seconds, and decode p50 above 25.7 ms. It was rejected and reverted.
Receipt:
`build/reports/local-ple-storage/unified-shared-expert-rejected-20261007.json`.

## Post-promotion stage profile and shared-residual rejection, 2026-10-07

- A synchronized stage profile on the pre-fused-Gdn artifact found a roughly
  0.22-0.33 ms `mx.eval` floor per forced stage. PLE stood out at 1.542 ms
  median in that serialized view, but direct numpy-row replay issues in only
  0.054 ms and collects in 0.114 ms median. The difference is Metal graph
  synchronization, not table search or storage overhead. Receipts:
  `current-stage-profile-20261007.json` and
  `ple-direct-replay-current-20261007.json`.
- Replacing 97 per-layer residual compile captures with two shared graphs was
  bit-identical but regressed all five paired p50 values and median p95 by
  0.103 ms. It was rejected and reverted. Receipt:
  `shared-residual-rejected-20261007.json`.
- A k-only QSA projection for KV lengths below the 2,048-token budget was exact
  but slower end to end: every paired p50 regressed and median p95 increased by
  0.324 ms. It was rejected and reverted. Receipt:
  `short-context-k-only-rejected-20261007.json`.

## Attention input-fusion rejection, 2026-10-07

A runtime Q/gate + packed K/V MXFP4 concatenation was exact in a focused
attention test and in every real run. The pilot improved p50 by only 0.042 ms;
the required five-pair gate then produced a median paired p50 gain of -0.005 ms
and a median p95 gain of -0.093 ms. It was rejected and reverted. Receipt:
`build/reports/local-ple-storage/attention-input-fusion-rejected-20261007.json`.

## Chunked indexer-cache rejection, 2026-10-07

A chunked `_IndexerCache` avoided per-token concatenation and preserved exact
values, growth, trim, and state behavior in a focused test. Real decode did not
improve enough: five pairs regressed median p50 by 0.012 ms and improved median
p95 by only 0.018 ms. It was rejected and reverted. Receipt:
`build/reports/local-ple-storage/chunked-indexer-cache-rejected-20261007.json`.

## GDN pre-recursion compile rejection, 2026-10-07

A per-layer compiled graph covered the prefused projection split, short
convolution, SiLU, reshape, and Q/K normalization. Every output was exact, and
five pairs improved median p95 by 0.139 ms, but median p50 regressed by 0.014
ms. The recurrence and output stages stayed outside the graph. It failed the
both-medians promotion gate and was reverted. Receipt:
`build/reports/local-ple-storage/gdn-prefusion-compile-rejected-20261007.json`.

## Metal fast-sync rejection, 2026-10-07

The process-only `MLX_METAL_FAST_SYNC=1` mode preserved text but regressed
decode p50 by 0.272 ms and p95 by 0.463 ms in a paired pilot. It was rejected
on pilot evidence; no five-pair run was warranted. Receipt:
`build/reports/local-ple-storage/metal-fast-sync-rejected-20261007.json`.

## SDPA block override rejection, 2026-10-07

The default 64-block SDPA reduction remains best for the canonical short
context. A 32-block pilot regressed p50 and p95; 128 blocks was exact but
inconsistent, with the clean second five-pair block regressing median p50 by
0.009 ms. Receipt:
`build/reports/local-ple-storage/sdpa-blocks-rejected-20261007.json`.

## Metal command-buffer budget rejection, 2026-10-07

Pilots with `MLX_MAX_OPS_PER_BUFFER` set to 32, 64, and 100 all regressed p50
and p95. The default M4 Max scheduler budget of 50 operations and 50 MB stays.
Receipt: `build/reports/local-ple-storage/metal-op-buffer-rejected-20261007.json`.

## Metal memory-buffer rejection, 2026-10-07

Pilots at 25, 100, and 300 MB regressed p50 and p95. The best probe, 200 MB,
improved p50 by 0.146 ms but p95 by only 0.001 ms and was not promoted. The
default 50 MB budget stays. Receipt:
`build/reports/local-ple-storage/metal-mb-buffer-rejected-20261007.json`.

## Metal command-buffer launch setting, 2026-10-07

The canonical runtime now launches with `MLX_MAX_MB_PER_BUFFER=200` set before
Python starts. Two five-pair external blocks improved combined median p50 by
0.303 ms and p95 by 0.221 ms; a durable run reached 24.159 ms p50 and
24.775 ms p95. An in-process `os.environ` mutation failed to reproduce the
gain and is not part of the contract. Receipt:
`build/reports/local-ple-storage/metal-mb-buffer-launch-env-accepted-20261007.json`.

## Command-buffer operator cap at 200 MB, 2026-10-07

On top of the accepted 200 MB memory budget, 64- and 100-operation command
buffers both regressed p50 and p95. Keep the default 50-operation cap. Receipt:
`build/reports/local-ple-storage/metal-mb200-op-buffer-rejected-20261007.json`.

## Command-buffer memory bracket, 2026-10-07

Pilots at 150 and 250 MB did not beat the accepted 200 MB launch setting on
both metrics: 150 MB regressed p95, and 250 MB regressed p50 and p95. Retain
200 MB. Receipt: `build/reports/local-ple-storage/metal-mb-bracket-rejected-20261007.json`.

## BFS width rejection, 2026-10-07

With the accepted 200 MB command-buffer budget, BFS width 10 failed the p95
gate and width 40 produced a severe tail regression. Keep the default width
20. Receipt: `build/reports/local-ple-storage/mlx-bfs-width-rejected-20261007.json`.

## Correct fast-synchronization rejection, 2026-10-07

The prior fast-sync receipt used the misspelled `MLX_METAL_FAST_SYNC` variable
and was superseded. The real `MLX_METAL_FAST_SYNCH=1` setting regressed p50 by
0.241 ms and p95 by 0.411 ms alongside the accepted 200 MB budget. Receipt:
`build/reports/local-ple-storage/metal-fast-synch-rejected-20261007.json`.

## Metal cache-limit rejection, 2026-10-07

A 16 GiB buffer-cache cap looked promising through a post-load wrapper, but the
durable constructor and loader-config paths both regressed median p50. Zero
cache retention was much slower. The default platform cache limit remains.
Receipt: `build/reports/local-ple-storage/metal-cache-16g-rejected-20261007.json`.

## PLE cache-capacity 32,768 rejection, 2026-10-07

A 32,768-row cache did not change the canonical trace's 536 hits and 760 disk
rows. Five pairs regressed both median p50 and p95 by 0.039-0.040 ms, so the
16,384-row capacity remains. Receipt:
`build/reports/local-ple-storage/ple-cache-32768-rejected-20261007.json`.

## Adaptive PLE cache addendum, 2026-10-07

The canonical runtime now keeps 16,384 PLE rows for short prompts and grows to
32,768 before prefill only when the full prompt exceeds 2,048 tokens. A real
3,200-token run activates sparse attention, fills the short cache, and saves
864 disk rows in every measured pair while preserving exact output. Prompt
reset restores the short capacity. Receipt:
`build/reports/local-ple-storage/adaptive-ple-cache-accepted-20261007.json`.

## B550 adaptive-source sync, 2026-10-07

B550 received the adaptive-cache implementation and a complete source-only
validation bundle. Remote CPU tests passed with 16 runs and 8 skips. The
split-layout runtime remains unchanged, and committed swap still blocks
calibration launches. Receipt:
`build/reports/local-ple-storage/b550-adaptive-ple-cache-source-sync-20261007.json`.

## Long-context admission rejection, 2026-10-07

On the new 3,200-token trace, fixed admission depth 16 improved decode medians
but regressed first-token latency by 271.203 ms. Switching from 32 during
prefill to 16 during decode was verified but still regressed median decode p50
and first-token latency. Depth 64 regressed its pilot. Retain `max_inflight=32`.
Receipt: `build/reports/local-ple-storage/long-context-inflight-rejected-20261007.json`.

## Long-context chunk and QSA rejection, 2026-10-07

Prefill chunks of 1,024 and 2,048 tokens changed greedy output and increased
peak memory. The QSA complement mask preserved the selected block set in a
focused test but regressed real decode p50 and p95. Retain 512-token prefill
chunks and direct top-k mask construction. Receipts:
`build/reports/local-ple-storage/long-prefill-chunks-rejected-20261007.json` and
`build/reports/local-ple-storage/qsa-complement-mask-rejected-20261007.json`.

## Long-context SDPA block rejection, 2026-10-07

On the real 3,200-token trace, SDPA block counts 32, 128, and 256 all regressed
decode p50 and first-token latency despite exact text. Retain the default 64
blocks. Receipt: `build/reports/local-ple-storage/long-sdpa-blocks-rejected-20261007.json`.

## Long-context stage profile and incremental QSA rejection, 2026-10-07

The durable profiler reports 0.427 ms median per QSA call at 3,200 tokens.
Incremental immutable-block pooling was stepwise bit-exact and improved one clean
decode block, but its first-token/one-time construction cost failed the overall
gate and a decode-only build regressed. Direct QSA pooling remains. Receipt:
`build/reports/local-ple-storage/incremental-qsa-blocks-rejected-20261007.json`.

## Preallocated QSA block-cache rejection, 2026-10-07

A 256-block-capacity cache updated only new completed blocks in place. It was
stepwise exact, but the real five-pair gate regressed median p95 and first-token
latency. Direct historical pooling remains the durable behavior. Receipt:
`build/reports/local-ple-storage/preallocated-qsa-blocks-rejected-20261007.json`.

## Single-token QSA visibility rejection, 2026-10-07

Skipping the all-visible completed-block mask was exact in focused stepwise
tests but did not improve the real pilot's p95 or first-token latency. The
general QSA visibility path remains. Receipt:
`build/reports/local-ple-storage/fast-qsa-decode-rejected-20261007.json`.

## QSA own-tail fast path rejection, 2026-10-07

Constructing only the final partial-block tail was exact in focused stepwise
tests but regressed p50, p95, and first-token latency on the real artifact.
The general own-tail comparison remains. Receipt:
`build/reports/local-ple-storage/fast-qsa-tail-rejected-20261007.json`.

## QSA score contraction rejection, 2026-10-07

Batched matmul was bit-exact and faster in a synthetic shape diagnostic, but a
clean real-model block regressed p95 and first-token latency. QSA einsum remains
canonical. Receipt: `build/reports/local-ple-storage/matmul-qsa-scores-rejected-20261007.json`.

## Long-context host-hash rejection, 2026-10-07

A real 3,200-token run with `hash_backend=host` was exact but regressed p50,
p95, and first-token latency. Its lower reported PLE elapsed time was an issue/
collect accounting shift, not an end-to-end gain. Retain `hash_backend=auto`.
Receipt: `build/reports/local-ple-storage/long-host-hash-rejected-20261007.json`.

## Aligned compiled QSA score rejection, 2026-10-07

Rounding QSA blocks to a fixed aligned capacity avoided an unaligned score shape
and won in isolation, but the real model regressed median decode p50, p95, and
first-token latency. Direct eager scores remain canonical. Receipt:
`build/reports/local-ple-storage/aligned-qsa-scores-rejected-20261007.json`.

## Cache capacity at 8,192 tokens, 2026-10-07

A real 8,192-token run confirmed the promoted 32,768-row adaptive cache. Larger
caches reduced disk rows but regressed decode p50 and first-token latency, so
they were rejected. Receipt:
`build/reports/local-ple-storage/context-8192-cache-capacity-rejected-20261007.json`.

## Cache-capacity bracket at 8,192 tokens, 2026-10-07

Lower and intermediate cache capacities all regressed decode p50 and p95. The
32,768-row long-context capacity remains the measured local optimum. Receipt:
`build/reports/local-ple-storage/context-8192-cache-bracket-rejected-20261007.json`.

## Admission-depth bracket at 8,192 tokens, 2026-10-07

The real 8,192-token working set rechecked admission depths 16 and 64. Both
were exact and issued the same 104,982 disk rows, but both regressed decode
p50, decode p95, and first-token latency. Retain `max_inflight=32`. Receipt:
`build/reports/local-ple-storage/context-8192-inflight-rejected-20261007.json`.

## 8,192-token stage profile, 2026-10-08

The synchronized profiler reports PLE at 2.159 ms median, below the QSA, GDN,
MoE, residual, and injection work in the serialized view. Storage remains
bounded but is not the leading decode-stage candidate. Receipt:
`build/reports/local-ple-storage/context-8192-stage-profile-20261008.json`.

## Capacity-backed indexer rejection, 2026-10-08

An opt-in append-capacity cache removed full-history concatenation in focused
tests but regressed real p50, p95, first-token latency, and peak RSS. Direct
raw-key concatenation remains. Receipt:
`build/reports/local-ple-storage/capacity-indexer-cache-rejected-20261008.json`.

## Cached QSA RoPE rejection, 2026-10-08

Reusing exact immutable block-RoPE prefixes did not improve the real artifact.
It regressed p50, p95, first-token latency, and memory, so direct table
construction remains. Receipt:
`build/reports/local-ple-storage/cached-qsa-rope-rejected-20261008.json`.

## Full-context PLE result-cache rejection, 2026-10-08

An exact 16,384-entry result cache hit 31 of 72 eligible decode contexts and
reduced the apparent PLE interval dramatically, but exact end-to-end p50 and p95
regressed. The result is strong evidence that the measured PLE interval includes
graph synchronization rather than pure storage time. Receipt:
`build/reports/local-ple-storage/ple-result-cache-rejected-20261008.json`.

## Specialized decode-hash rejection, 2026-10-08

The exact three-token Metal hash specialization was bit-identical in focused
tests and won in isolation, but real p50, p95, and first-token latency regressed.
The general hash path remains. Receipt:
`build/reports/local-ple-storage/specialized-decode-hash-rejected-20261008.json`.

## Lazy PLE collect rejection, 2026-10-08

Letting PLE rows join the downstream graph was bit-exact and marginally improved
first token, but regressed decode p50 and p95. Eager row evaluation remains.
Receipt: `build/reports/local-ple-storage/lazy-ple-collect-rejected-20261008.json`.

## Same-layer injection-residual fusion rejection, 2026-10-08

Fusing each attention-output injection into its same-layer MoE residual graph
was bit-exact in focused tests but regressed real p50 and p95. Separate compiled
graphs remain. Receipt:
`build/reports/local-ple-storage/fused-injection-residual-rejected-20261008.json`.

## Upstream contiguous-sort conditional result, 2026-10-08

The upstream Metal sort-kernel selection fix was isolated in a separate MLX
build. It passes exact short-context promotion medians, but long-context pairs
remain inconclusive, so the canonical core build stays unchanged. Receipt:
`build/reports/local-ple-storage/upstream-sort-contiguous-kernel-conditional-20261008.json`.

Follow-up at 3,200 tokens rejected the isolated core for the canonical
multi-context runtime: p50, p95, and first-token medians all regressed despite
exact output.

## Final mixer and LM-head fusion rejection, 2026-10-08

Compiling the final mixer with the exact BF16 LM head preserved full raw logits
in focused tests but regressed real p50. Separate final-mixer and LM-head graphs
remain. Receipt:
`build/reports/local-ple-storage/compiled-lm-mixer-rejected-20261008.json`.

## GDN short-conv kernel rejection, 2026-10-08

The decode-only cache-update and four-tap convolution kernel was bit-exact and
improved median p95, but failed the required p50 median gate. Generic Conv1d
remains. Receipt:
`build/reports/local-ple-storage/gdn-short-conv-kernel-rejected-20261008.json`.

## Combined sort/GDN rejection, 2026-10-08

The isolated upstream-sort core and GDN short-conv kernel did not compose:
exact output regressed both decode medians. Receipt:
`build/reports/local-ple-storage/combined-sort-gdn-rejected-20261008.json`.

## Depthwise launch-geometry rejection, 2026-10-08

Forcing the canonical depthwise Conv1D launch from 1,024 to 512 threads was
exact but regressed real p95. The default launch geometry remains. Receipt:
`build/reports/local-ple-storage/depthwise-launch-512-rejected-20261008.json`.

## Fast PLE issue rejection, 2026-10-08

Batched cache checks and vectorized shard grouping improved direct replay, but
the real 8,192-token model regressed p50, p95, and first-token latency. The
standard issue path remains. Receipt:
`build/reports/local-ple-storage/fast-ple-issue-rejected-20261008.json`.

## Thresholded packed-GDN T=1 promotion, 2026-10-08

The packed GDN recurrence now has a bit-exact T=1 kernel for contexts through
4,096 tokens. Short and 3,200-token decode medians improve with exact output;
above 4,096 tokens the generic packed kernel remains active. Receipt:
`build/reports/local-ple-storage/gdn-packed-t1-threshold-accepted-20261008.json`.

## B550 packed-GDN source sync, 2026-10-08

The T=1 source bundle was synchronized and hash-verified on B550. CPU-only
validation passed with nine platform-specific skips. Because swap remained fully
committed, no calibration or GPU work was launched. Receipt:
`build/reports/local-ple-storage/b550-gdn-t1-source-sync-20261008.json`.

## Packed-GDN threshold boundary, 2026-10-08

Four clean 4,096-token pairs preserved exact output but improved no p95 pair.
The same-config A/A check was inconclusive because of external multi-second
spikes, so the canonical threshold was restored conservatively to 3,200 tokens.
The restored 4,096-token fallback smoke matched every gate output exactly.
Receipt: `build/reports/local-ple-storage/gdn-packed-t1-ctx4096-rejected-20261008.json`.

## GDN follow-up probes, 2026-10-08

A current Metal capture confirms 101 repeated GPU commands per steady decode and
a 2.770 ms median exact-BF16 head command, but shader attribution is unavailable
because profiler intervals export empty. Launch-geometry and float4-state probes
were rejected. An exact fused Q/K normalizer also failed the real p95 gate and
was reverted. Receipt:
`build/reports/local-ple-storage/gdn-fused-qk-rejected-20261008.json`.

## PLE cache-capacity boundary, 2026-10-08

The canonical 4,096-token trace shows why direct replay is not acceptance: a
65,536-row cache lowers replay disk work, but an eight-pair same-process model
gate regresses both decode medians. Exact BF16 rows and the 32,768-row
long-context cache remain canonical. Receipt:
`build/reports/local-ple-storage/ple-cache65536-ctx4096-rejected-20261008.json`.

## Upstream residency refresh rejection, 2026-10-08

The only new upstream Metal performance candidate after the pinned base adds an
optional residency refresh thread. One-second and 100-ms intervals do not remove
the first post-idle spike and do not improve warmed decode. The canonical MLX
build remains unchanged. Receipt:
`build/reports/local-ple-storage/upstream-residency-refresh-rejected-20261008.json`.

## Post-idle warm-up acceptance, 2026-10-08

The post-idle spike lives inside the first full model pass. A one-token
background warm-up after 90 seconds idle moved the work off the user path and
reduced the measured next first token from 863.790 ms to 110.689 ms with exact
output. The server now runs this warm-up proactively when the operator enables
an idle threshold. Receipts:
`build/reports/local-ple-storage/idle-full-warmup-accepted-20261008.json` and
`build/reports/local-ple-storage/idle-warmup-server-implementation-20261008.json`.

The full warm-up-enabled source bundle is synchronized to B550 and CPU-validated
there. No remote GPU or calibration work was launched while swap was full.
Receipt: `build/reports/local-ple-storage/b550-idle-warmup-source-sync-20261008.json`.

## Idle warm-up threshold, 2026-10-08

The cold full-model pass already costs about 0.92-1.09 s after a one-second
idle gap and does not grow with longer gaps, so the recommended threshold is
one second. Interleaved traffic with proactive warm-ups kept steady decode at
24.378 ms against 24.392 ms plain, with exact output. The flag stays disabled
by default. Receipt:
`build/reports/local-ple-storage/idle-warmup-threshold-and-no-degradation-20261008.json`.

A 4,096-token gate bounds the warm-up to short-prompt serving. After a 60-second
idle the long-context first token was 6,647.014 ms without a keeper and
6,448.527 ms with one, only 198.487 ms better, because PLE prefill dominates
there. Long-context first-token latency is unchanged by this flag. Receipt:
`build/reports/local-ple-storage/idle-warmup-longctx-scope-limited-20261008.json`.

Deferring the PLE collect evaluation reduces the direct per-token cost from
0.052 ms to 0.019 ms, which is below real decode variance and would change
synchronization structure, so the eager path stays canonical. Receipt:
`build/reports/local-ple-storage/ple-deferred-eval-not-promoted-20261008.json`.

The decode cost model was corrected and republished after attribution. The
routed MoE experts are the largest single measured component at 6.224 ms per
token, ahead of the exact BF16 head at about 2.77 ms and the packed-GDN
recurrence at 1.601 ms. Real PLE work is 0.183 ms per token, so the external
storage path is not a decode bottleneck. Isolated expert timing was rejected as
unreliable because a sub-projection appeared to cost more than the whole path.
Receipts: `build/reports/local-ple-storage/hillclimb-continuation-session-20261008-11.json`.

Attention, the LM head layout, and the compile boundaries were then closed.
The 12 full-attention output projections cost about 0.35 ms at 4,096 tokens.
An interleaved probe with real artifact weights confirms the transposed BF16
head is both fastest at 2.7052 ms and bitwise exact, while contiguous
transposition is slower and differs by 0.0039. Restricting decode compilation
to the MoE blocks is 0.657 ms slower short and 0.495 ms slower at 4,096 tokens
with zero wins in eight interleaved pairs, so full compilation stays canonical.
A sequential read had suggested the opposite, which is recorded as the reason
every candidate arm now requires interleaved measurement. Receipts:
`build/reports/local-ple-storage/lm-head-layout-interleaved-20261008.json`,
`build/reports/local-ple-storage/moe-only-compile-gate-20261008.json`, and
`build/reports/local-ple-storage/attention-and-compile-boundary-20261008.json`.

The PLE decode cost was corrected once more. A decode ticket requests only
16 unique rows in 16 pread calls totalling 5,120 bytes, which measures 0.0094
ms. The multi-millisecond figures seen earlier came from the eight prefill
tickets, each of which requests about 3,300 rows. Per-decode-token PLE I/O is
therefore negligible, and the thread-pool result is a prefill-path effect: pool
size 1 is 1.9x faster for prefill tickets but shows no consistent decode
benefit, so `max_inflight` stays 32. Receipts:
`build/reports/local-ple-storage/ple-read-cost-decompose-20261008.json`,
`build/reports/local-ple-storage/ple-decode-cost-correction-20261008.json`, and
`build/reports/local-ple-storage/ple-pool-size-not-promoted-20261008.json`.
