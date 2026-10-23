# Flint 40B MLX performance receipt

Date: 2026-10-23
Agent: Codex

## Scope

Measure repaired Flint 40B GPTQ-MXFP4 inference on the local M4 Max, covering
single-stream latency and throughput, context scaling, recommended sampling,
and concurrent MLX-LM server batching. Determine a practical concurrency
operating point and verify that the benchmark left no process or service
running.

## Runtime and artifact identity

- Hardware: Apple M4 Max, 128 GiB unified memory, `applegpu_g16s`.
- Metal recommended working set: 115,448,725,504 bytes.
- MLX: 0.32.4.
- MLX-LM adapter:
  `b384a406ad5cac168877c32b048f31aa9c8db244`
  (`fix: correct Flint routed expert activation order`).
- Local model: `build/flint-resharded`, 25 shards, 20.5596 GiB parameters.
- Public model:
  `RESMP-DEV/Flint-40B-A2.5B-MLX-GPTQ-MXFP4`.
  Live HF API check on 2026-10-23 returned repository head
  `02658ab3aed9cb61b25a239ecb539c48911c5642`, public, ungated, with 25
  current safetensors siblings.

All throughput rows below are from the repaired adapter. Results from the
reversed routed-expert gate/up adapter are invalid and are not included.

## Direct generation

The harness used `generate_step(..., greedy_fast=True,
prefill_step_size=2048)`, reset MLX peak memory before every case, and wrote
per-repeat output hashes. Warm model load took 1.6487 s.

| Prompt tokens | Generated | Median TTFT | Median prefill | Median decode | p50 token latency | Peak MLX memory |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 256 | 47.5 ms | 336.5 tok/s | 126.98 tok/s | 7.80 ms | 20.67 GiB |
| 128 | 256 | 137.2 ms | 933.0 tok/s | 125.65 tok/s | 7.87 ms | 20.78 GiB |
| 1,024 | 128 | 552.7 ms | 1,852.7 tok/s | 121.59 tok/s | 8.12 ms | 21.43 GiB |
| 8,192 | 128 | 5.406 s | 1,515.3 tok/s | 111.97 tok/s | 8.89 ms | 21.95 GiB |
| 32,768 | 64 | 32.27 s | 1,015.7 tok/s | 91.89 tok/s | 10.71 ms | 22.61 GiB |

Recommended sampling (`temperature=1.0`, `top-p=0.95`, `top-k=20`) completed
three 256-token runs at 127.615, 129.400, and 128.409 tok/s. The CLI reported
35.481 decimal GB peak memory; this includes load-time high-water effects and
uses a different unit from the direct MLX GiB rows.

## Concurrent server batching

The server probes used streaming `/v1/completions`, temperature 0, a 128-token
cap, and unique prompts to prevent prefix-cache reuse. The direct benchmark
uses `greedy_fast`; the MLX-LM server path computes full-vocabulary
probabilities. This explains the lower one-way server decode rate and is not a
model-quality difference.

Stock `ThreadingHTTPServer` has a small listen backlog and reset connections
during high-way bursts. The benchmark wrapper therefore set
`ThreadingHTTPServer.request_queue_size = 1024` before launching the temporary
port 8083 server.

Short-prompt results:

| Concurrency | End-to-end aggregate | Sum per-stream decode | Median TTFT | Max TTFT | Errors |
|---:|---:|---:|---:|---:|---:|
| 1 | 67.43 tok/s | 90.99 tok/s | 0.502 s | 0.502 s | 0 |
| 4 | 239.98 tok/s | 252.38 tok/s | 0.119 s | 0.121 s | 0 |
| 8 | 319.52 tok/s | 337.74 tok/s | 0.195 s | 0.197 s | 0 |
| 16 | 405.31 tok/s | 439.69 tok/s | 0.403 s | 0.450 s | 0 |
| 32 | 552.59 tok/s | 615.95 tok/s | 0.727 s | 0.980 s | 0 |
| 64 | 592.26 tok/s | 675.66 tok/s | 1.379 s | 2.485 s | 0 |
| 96 | 631.92 tok/s | 741.00 tok/s | 2.156 s | 4.284 s | 0 |
| 128 | 663.99 tok/s | 801.20 tok/s | 2.934 s | 6.367 s | 0 |
| 192 | 635.77 tok/s | 840.05 tok/s | 4.700 s | 11.226 s | 0 |

The practical optimum is 128 concurrent short-prompt requests: 663.99
end-to-end completion tok/s, 801.20 aggregate decode tok/s, and 54 GB resident
and peak footprint. Although 192-way reached 840.05 aggregate decode tok/s,
its end-to-end completion rate fell to 635.77 tok/s, median TTFT rose to
4.700 s, max TTFT reached 11.226 s, and its 92 GB footprint drove approximately
8 GB of swap use. It is not a recommended operating point.

Sampled server footprints were about 30 GB resident and 34 GB peak at 64-way,
54 GB resident and peak at 128-way, and 92 GB at 192-way.

For roughly 1K-token prompts, eight concurrent requests reached 128.72
aggregate completion tok/s, 298.66 aggregate decode tok/s, and 4.47 s median
TTFT. Sixteen reached 139.85 completion tok/s, 368.01 decode tok/s, and 8.88 s
median TTFT. Long-prompt concurrency should therefore remain much lower than
the short-prompt operating point.

## Cleanup and health

Final cleanup checks at 2026-10-23 10:22:53 -0700 found:

- No TCP listener on port 8083.
- No `run_flint_server_high_backlog.py`, `mlx_lm.server`,
  `benchmark_flint_server.py`, or `benchmark_mlx_flint.py` process.
- `dev.resmp.local27-mxfp4` and `dev.resmp.local27-admission` remained not
  loaded, as in the pre-benchmark state.
- `memory_pressure -Q` reported 83% system-wide free memory.
- `vm.swapusage` still reported 7,941.06 MiB used after the 192-way memory
  probe. This is retained swap space under low current pressure, not active
  model residency.
- The highest-RSS listed processes were unrelated system and desktop
  processes; no Flint model process remained.

## Verification artifacts

- Direct harness: `build/reports/flint/benchmark_mlx_flint.py`
- Direct result: `build/reports/flint/flint-m4-single-stream-benchmark.json`
- Sampling result:
  `build/reports/flint/flint-m4-recommended-sampling-benchmark.log`
- Server harness: `build/reports/flint/benchmark_flint_server.py`
- High-backlog wrapper:
  `build/reports/flint/run_flint_server_high_backlog.py`
- Server results:
  `build/reports/flint/flint-m4-server-batching-benchmark.json`,
  `flint-m4-server-batching-64way.json`,
  `flint-m4-server-batching-96way.json`,
  `flint-m4-server-batching-128way.json`, and
  `flint-m4-server-batching-192way.json`.
- The 32-way JSON is intentionally empty because the one-off serializer passed
  a `PosixPath` to `json.dumps`. The valid measured 24-way, 32-way, and
  long-prompt rows are in `flint-m4-server-batching-32way.log`; they were not
  rerun only to regenerate JSON.

These paths are ignored build evidence. The receipt and architecture entry are
the durable tracked records.

## Non-claims

- This is one M4 Max point-in-time measurement, not a guarantee across Apple
  Silicon models, OS versions, MLX releases, or checkpoint revisions.
- Server concurrency values are workload-specific. The recommended 128-way
  point applies to the measured short prompts and 128-token cap, not arbitrary
  prompts or generation lengths.
- The temporary benchmark wrapper changes the Python HTTP server class backlog
  for measurement; it is not a claim that the stock MLX-LM server is
  production-security hardened or permanently configured with this wrapper.
- `--kv-bits` disables the batching path and was not used in these concurrency
  measurements.
- Full-vocabulary server sampling and direct `greedy_fast` generation are not
  interchangeable throughput implementations.
- Throughput does not establish benchmark-score quality. Quality and remote
  object integrity are covered by
  `evidence/2026-10-23-flint-mlx-eval-publication.md`.
