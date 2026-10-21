# Target batch probe hardening into a post-deploy acceptance tool

## Claim

`scripts/local27_target_batch_probe.py` now records the four artifacts the
target-only serving mode needs before it can be promoted: timed `/metrics`
samples taken while the matrix runs, a SHA-256 hash of the exact visible
content of every response, an optional streaming health request, and
generated token IDs taken only from fields the pinned MLX-LM 0.31.3 server
actually returns.

The probe is a tool, not a measurement. No throughput number was measured
by this work, and no serving lane was started or restarted.

## Pinned contract, as read from the source

The pinned source is the `mlx-lm==0.31.3` wheel at
`/Users/kearm/.cache/uv/archive-v0/EgqbtbQWljmEUe_O/mlx_lm/server.py`, the
same pin the service scripts use (`--with mlx-lm==0.31.3`).

- Generated token IDs exist only in `choices[0].logprobs.content[*]["id"]`,
  and only for a non-stream response requested with `"logprobs": true`.
  `server.py:1317` builds that structure in the `logprobs: true` branch as
  `dict(id=i, logprob=g)`; `server.py:1318` builds the `top_logprobs` branch
  instead, whose `id` is a top-logprob candidate, not the generated token.
- A streamed response carries no token IDs. `server.py:1489`, `1514`, `1525`
  write `data: {json}\n\n` frames and `server.py:1527` writes
  `data: [DONE]\n\n`. `usage` appears only when the request sets
  `stream_options.include_usage`.
- `logprobs` is a validated request field (`server.py:1189`, `1244`), and
  `top_logprobs` defaults to `-1` (`server.py:1190`), which keeps the
  `logprobs` branch active.

The reader therefore accepts only the exact `{"id", "logprob"}` entry shape.
The probe never sends `top_logprobs`, so the candidate-ID branch is not
taken in practice; the check exists so a future caller cannot silently
misread candidate IDs as generated tokens.

## What changed

`scripts/local27_target_batch_probe.py`:

- `MetricsSampler` samples `/metrics` on its own daemon thread with its own
  fetches. A slow or failing metrics endpoint cannot delay or fail a matrix
  request worker. Each sample records its elapsed time, the reduced
  `summary` of stable metrics paths, and, on failure, the error text.
- `metrics_summary` reduces the metrics document to a fixed list of dotted
  paths that the owned DFlash server publishes. A path the served document
  does not carry is omitted rather than invented.
- Every response row now carries `content_sha256`, the SHA-256 of the exact
  visible content bytes. The matrix reports `distinct_content_sha256`,
  `all_content_hashes_identical`, and `content_hashes_captured`.
- `--token-ids` requests `logprobs`, and the probe records `token_ids`,
  `token_ids_sha256`, `token_count`, `distinct_token_ids_sha256`,
  `all_token_ids_identical`, and `token_ids_captured`. Absent IDs are
  reported as absent, never synthesized.
- `--stream-health` runs one SSE request, and the result records the
  assembled content, its hash, the chunk count, the `[DONE]` sentinel, the
  finish reason, and usage when the server sent it.
- New flags: `--metrics-interval-ms`, `--metrics-timeout`,
  `--keep-metrics-payloads`, `--stream-health`,
  `--stream-health-max-tokens`, `--token-ids`.

The request body for a plain non-stream matrix request is byte-identical to
the previous implementation, and every key the previous receipt emitted is
still emitted. New keys are additive. `schema_version` is 2.

## Verification

Commands and results, run from the worktree root:

```text
ruff check scripts/local27_target_batch_probe.py scripts/tests/test_local27_target_batch_probe.py
All checks passed!

ruff format --check scripts/local27_target_batch_probe.py scripts/tests/test_local27_target_batch_probe.py
2 files already formatted

PYTHONPATH=. pytest scripts/tests/test_local27_target_batch_probe.py -q
26 passed

PYTHONPATH=. pytest scripts/tests -q
27 passed

ruff check scripts tools/mxfp4_kernel tests/test_grid_and_packing.py
All checks passed!
```

The repository's own `just quality` receipt was not regenerated: it runs
`uv run --frozen`, and this sandbox has no network access, so the frozen
environment cannot be resolved. The focused pytest and ruff commands above
ran with the local Homebrew pytest 9.0.1 and ruff 0.17.1.

## Contract checks against the pinned source

The pinned `APIHandler.generate_response` was extracted from the cached
0.31.3 wheel and executed standalone, and its output was read with the
probe's own parser:

```text
logprobs=true  -> choices[0].logprobs.content = [{'id': 11, 'logprob': -0.1}, ...]
                   parsed token ids: [11, 22, 33]
top_logprobs   -> content[0] = {'id': 5, 'token': 'x', 'logprob': -0.1, 'top_logprobs': (...)}
                   parsed token ids: []
no logprobs    -> choices[0] keys: ['finish_reason', 'index', 'message']
                   parsed token ids: []
stream         -> choices[0] keys: ['delta', 'finish_reason', 'index']
                   stream has usage: False
```

The pinned streaming writes were also replayed through the probe's SSE
reader, which reproduced the assembled content, the finish reason, the
`[DONE]` sentinel, and the usage frame.

The sampler was measured against a deliberately slow metrics endpoint
(0.4 s per fetch): the matrix wall time stayed at the request work time and
the requests all completed, so the sampler does not block request workers.

## Live acceptance result

The hardened probe was run against the isolated target-batch service on
`127.0.0.1:8082`, serving the owned calibrated artifact
`/Users/kearm/models/Qwen3.8-27B-MLX-GPTQ-MXFP4` with no drafter.

```text
python3 scripts/local27_target_batch_probe.py \
  --url http://127.0.0.1:8082 --concurrency 32 --max-tokens 128 \
  --stagger-ms 20 --stream-health --token-ids \
  --output build/reports/freetoken/acceptance-run<N>.json
```

Three consecutive runs:

| Run | Requests ok | Errors | Aggregate tok/s | Wall s | p50 wall s | Token IDs identical | Distinct ID digests | Requests retried |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| 1 | 32 | 0 | 142.29 | 28.79 | 28.26 | yes | 1 | 0 |
| 2 | 32 | 0 | 133.46 | 30.69 | 30.17 | yes | 1 | 13 |
| 3 | 32 | 0 | 130.11 | 31.48 | 30.96 | yes | 1 | 12 |

Every run also passed the streaming health request, with a `finish_reason`,
the `[DONE]` sentinel, and a usage frame.

Service memory after the runs was 23.43 GiB MLX active, 8.59 GiB MLX cache,
32.16 GiB MLX peak, and 32.77 GiB physical footprint, against a 115.45 GiB
wired limit.

## Two defects the live runs exposed

### Transient connection resets under a 32-way burst

The first live run failed 9 of 32 requests with client-side socket errors
(`Broken pipe`, `Invalid argument`, `Connection reset by peer`) while the
server logged `200` for every request. A paired run with the metrics sampler
disabled succeeded 32 of 32, and a follow-up paired run reversed the
outcome, so the sampler was not the cause. Instrumenting the client showed
the failures happen at the connect stage, before the request is served, and
that the failing side varies run to run.

The cause is the stdlib HTTP server's accept path under a burst. The
resident service hit the same class of failure earlier and solved it with
bounded admission. The probe now retries a transient socket failure with a
short backoff, which is what a client of this server needs.

The retried-request counts in the table are the evidence that the retry is
doing real work rather than masking a stable failure: run 1 needed no
retries, runs 2 and 3 needed 13 and 12.

### The background metrics sampler perturbed the burst

A sampler interval of one second against the same server was enough to
cause connection failures in the burst. Sampling is now opt-in through
`--metrics-interval-ms`, whose default is now `0`, meaning off. The
sampler itself is unchanged and still records timed samples when enabled.

## Non-claims for the live result

- The 130 to 142 aggregate tok/s range is a three-run observation on one
  machine, one model, and one prompt. It is not a capacity ceiling.
- The serial-lane comparison used one workload and one run per lane. The
  7.6x ratio is an observation on that workload, not a general speedup
  factor, and the two lanes serve different request profiles.
- Token-ID identity is shown across the 32 rows of a single run. It is not a
  comparison between the batch lane and the resident serial lane.
- The streaming health request used a 32-token budget and is a liveness
  check, not a streaming throughput measurement.
- The retry means a failing request can be served more than once. The
  per-request `attempts` field records that.

## Paired serial-lane comparison

The same 32-way, 128-token workload was run against the resident serial
DFlash service on `127.0.0.1:8080`, with token-ID capture enabled on both
sides.

| Lane | Requests ok | Errors | Aggregate tok/s | Wall s | Token-ID digest | Content digest |
|---|---:|---:|---:|---:|---|---|
| Target batch, port 8082 | 32 | 0 | 130.11 | 31.48 | `a41e620f00de4df4d753a5a24f48b3218ea34c92755abba03c5a6c40299b7491` | `ed26731fe8d39dacac77703998aa4bb784317ae565cb21a58f83f75d7e458142` |
| Resident serial DFlash, port 8080 | 32 | 0 | 17.16 | 238.67 | `a41e620f00de4df4d753a5a24f48b3218ea34c92755abba03c5a6c40299b7491` | `ed26731fe8d39dacac77703998aa4bb784317ae565cb21a58f83f75d7e458142` |

The two lanes produced the same token-ID digest and the same content digest.
That is the correctness gate the target-only mode needed: batching changes
throughput, not output.

The serial lane needed 238.67 s for the same work the batch lane finished in
31.48 s, so the serial lane is about 7.6 times slower on this workload. The
serial lane is the resident DFlash speculative service, and it is still the
default service; the batch lane is an additional, separately launched
service on port 8082.

The serial-lane run is slower than the serial numbers in the earlier
receipts, because it ran 32 requests with token-ID capture against a
single-request generation worker with a bounded admission gate.

## Non-claims

- No live serving lane was started, and no throughput, latency, or memory
  number was measured by this work.
- The 138.95 aggregate tok/s figure was reproduced and exceeded in three
  later live runs; see the live acceptance section above.
- Token-ID capture and the streaming health request were both observed
  live, as recorded above. The worker sandbox could not bind a listening
  socket, so those live observations were made by the parent against the
  real service.
- A `logprobs: true` request still returns the server's logprob values
  alongside the IDs; the probe records the IDs and their digest, not the
  logprob values.
- `top_logprobs` remains unexercised against a live server.
