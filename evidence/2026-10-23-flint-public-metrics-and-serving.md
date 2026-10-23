# Flint public metrics and serving infrastructure

Date: 2026-10-23
Agent: Codex

## Scope

Publish the independent WikiText-2 correctness metrics on the public Hugging
Face model card, promote the successful Flint concurrent-serving harness into
supported project tooling, and determine whether paged attention is available
and usable for Flint on macOS.

Three read-only sub-agent audits covered model-card wording, paged-attention
feasibility, and serving implementation. The parent performed the public
upload, code integration, live model probes, and final acceptance.

## Public model-card update

The live Hugging Face README was byte-identical to the local publication
fixture before editing. The new section reports:

- MXFP4: PPL 21.29, target top-1 48.35%, agreement with BF16 73.20%.
- BF16: PPL 22.37, target top-1 48.83%.
- 32,752 identical scored transitions from WikiText-2 raw validation.
- Explicit definitions for perplexity, target top-1, and agreement.
- The non-claim that lower quantized perplexity does not show that
  quantization improved the model; task quality needs extensive benchmarking.

Upload result:

- Repository: `RESMP-DEV/Flint-40B-A2.5B-MLX-GPTQ-MXFP4`
- Commit: `edc81733f8fbafad6d653ca853b46e476697da37`
- Post-upload raw README bytes matched the local file exactly.
- Live model state remained public and ungated with 25 safetensors siblings.

## Supported serving tooling

The temporary ignored harnesses were promoted to:

- `scripts/flint_mlx_server.py`
- `scripts/flint_mlx_server_benchmark.py`
- `scripts/tests/test_flint_mlx_serving.py`

The server wrapper consumes one new `--listen-backlog` argument, validates it,
passes every other argument to stock `mlx_lm.server`, raises the
`ThreadingHTTPServer` listen backlog, and restores the class value after the
injected server-main function returns in tests. This preserves the mechanism
that prevented kernel connection resets during high-way bursts.

The benchmark client preserves the measured protocol:

- synchronized thread-barrier bursts;
- one persistent HTTP connection per request;
- streaming `/v1/completions`;
- temperature 0;
- no retry layer;
- unique prompt markers to defeat prefix-cache reuse;
- explicit Flint EOS suppression with `logit_bias`;
- per-request TTFT, request time, decode rate, p50/p95 decode latency, and
  output hash.

It now writes schema/provenance fields (run ID, prompt path and SHA-256, all
cases, timeout, suppressed token IDs, and token-count method), retains partial
rows and error strings when a burst fails, writes the report atomically, and
returns nonzero when any request fails. Token counts remain explicitly defined
as non-empty SSE text events, not independently verified tokenizer usage.

Representative launch:

```bash
PYTHONPATH=/Users/kearm/mlx-lm \
  /Users/kearm/mlx-lm/.venv/bin/python scripts/flint_mlx_server.py \
  --listen-backlog 1024 \
  --model /Users/kearm/mlx-stack/build/flint-resharded \
  --host 127.0.0.1 --port 8083 \
  --decode-concurrency 128 --prompt-concurrency 16 \
  --temp 0 --max-tokens 512
```

The benchmark CLI accepts arbitrary
`label:prompt_chars:concurrency:max_tokens` cases and does not hardcode the
model, prompt, output, or run identity.

Live acceptance used the repaired adapter and the real 25-shard model with a
short two-request burst:

- Requests successful: 2.
- Requests failed: 0.
- Generated non-empty events: 16.
- Aggregate completion rate: 27.98 tok/s.
- Sum per-stream decode rate: 152.51 tok/s.
- Report: `build/reports/flint/serving-tool-smoke.json`.

This short run proves the supported tool path and JSON contract; it is not a
repeat of the 128-way capacity result.

## Concurrent serving state

The current recommendation remains the repaired MLX-LM continuous batcher for
Flint. Its prior best measured practical point was 128 concurrent short-prompt
requests at 663.99 aggregate completion tok/s, 801.20 aggregate decode tok/s,
2.93 s median TTFT, and a 54 GB footprint.

Flint has 14 ordinary full-attention layers and 35 fixed 513-token rotating
sliding-window layers. The existing MLX-LM cache uses:

- `BatchKVCache` for full layers;
- `BatchRotatingKVCache` for sliding layers.

Both are dense key/value tensor caches. The current `RotatingKVCache` does not
provide quantized conversion, so `--kv-bits` is not a generic drop-in for this
hybrid model and disables batching where supported by other cache paths.

The DFlash target-batch runtime is not a Flint shortcut. Its target backend
currently resolves Qwen GDN/Gemma-family target operations and has no Flint
target adapter; routing this checkpoint through it would require architecture
integration and a separate parity gate.

## Paged attention on macOS

### MLX and MLX-LM boundary

Live source checks found:

- Upstream MLX `afc03c51105a8e25e70fa2ba0befb214e4f5ec6f` has no public
  paged-attention or block-table SDPA API.
- `mx.fast.scaled_dot_product_attention` accepts contiguous Q/K/V tensors, a
  mask, and attention sinks, not page tables.
- Upstream MLX-LM `59995f131cf2b48950a67d2eb36d76b2a554b989` has dense batched
  caches and continuous batching, not paged KV storage.

A Python-only page pool is not a serving solution: every decode step would
have to gather physical pages into dense K/V tensors for the existing kernel.

### vLLM Metal capability

The Apache-2.0 `vllm-project/vllm-metal` plugin is a credible macOS paged
attention lane. Current source contains:

- paged Metal decode and tiled varlen prefill kernels;
- block-table conversion;
- GQA support;
- sliding-window support;
- attention-sink parity tests.

The official Homebrew stable package was installed locally:

- vLLM: `0.32.0+cpu`
- vLLM Metal: `0.32.0`
- MLX: `0.32.2`
- MLX-LM: `0.32.0`
- The installed MLX-LM has no Flint model module.

The formula's native extension smoke passed. A direct paged-attention kernel
probe then used:

- 32 query heads, 4 KV heads, head dimension 128;
- context length 64, page size 16;
- bfloat16 K/V;
- non-zero per-head attention sinks.

Against `mx.fast.scaled_dot_product_attention` with sinks:

- maximum absolute difference: 0.001953125;
- mean absolute difference: 0.000000476837;
- output shape: `[1, 32, 128]`.

This proves that a real paged-attention kernel with GQA and sinks executes
correctly on this M4 Max. It does not prove Flint compatibility.

### Exact Flint blocker

A live `vllm serve` probe used the local 25-shard checkpoint, dummy load
format, 1,024-token maximum context, and two-sequence cap. It failed before
model loading with:

```text
The checkpoint you are trying to load has model type `flint`
but Transformers does not recognize this architecture.
```

Probe log:
`build/reports/flint-vllm-metal/flint-load-probe.log`.

Upstream vLLM contains a CUDA/FlashInfer-oriented Flint model, but vLLM Metal's
supported-model matrix does not list Flint and its installed MLX-LM dependency
does not include the fork-local Flint adapter. A future Flint lane therefore
needs at least:

1. Transformers/vLLM config recognition for `model_type=flint`;
2. the repaired Flint MLX-LM adapter in the plugin environment;
3. a model-adapter and hybrid-cache mapping;
4. MXFP4 weight-loading compatibility;
5. output parity and concurrent benchmark acceptance.

## Validation

Focused tool validation:

```text
ruff check: passed
ruff format --check: passed
scripts/tests/test_flint_mlx_serving.py: 13 passed
```

The root suite also passed:

```text
tests scripts/tests: 120 passed, 16 skipped
```

The quality dispatcher was updated to include `scripts/tests` in its root test
check. Full `just quality` results are recorded separately in its receipt; the
known pre-existing import lint issue in `scripts/local27_metrics_watcher.py`
is unrelated to this change.

## Non-claims

- The two-request smoke proves tool behavior, not a new capacity optimum.
- The vLLM Metal paged-kernel parity probe does not load or benchmark Flint.
- The failed vLLM Metal Flint probe is a current stable-package result, not a
  prediction about future releases.
- Installing vLLM Metal does not change or replace the working MLX-LM service.
- Public model-card metrics are correctness/fidelity measurements, not task
  quality or agentic benchmark results.
