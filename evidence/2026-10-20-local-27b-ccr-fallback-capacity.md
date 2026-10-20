# Local Qwen3.8 27B CCR fallback capacity

## Claim

The owned calibrated Qwen3.8-27B GPTQ MXFP4 artifact can serve an OpenAI
Chat Completions fallback through an isolated CCR Rust route. With thinking
disabled, normal OpenAI `content` is returned. The useful short-context decode
operating point is 32 concurrent requests with CCR `maxInflight: 32`, producing
about 142 aggregate completion tokens/s.

## Artifacts

- Target: `/Users/kearm/models/Qwen3.8-27B-MLX-GPTQ-MXFP4`
- Control: `/Users/kearm/models/mlx-community-Qwen3.8-27B-4bit`
- MLX LM: `/Users/kearm/mlx-lm` at `d3eed60`, MLX 0.32.4
- CCR Rust: `/Users/kearm/ccr-rust` at `f12ee5c`
- Temporary MLX listener: `127.0.0.1:8081`
- Temporary CCR listener: `127.0.0.1:8091`

The MLX server command used:

```bash
python -m mlx_lm.server \
  --model /Users/kearm/models/Qwen3.8-27B-MLX-GPTQ-MXFP4 \
  --host 127.0.0.1 --port 8081 \
  --decode-concurrency 64 --prompt-concurrency 16 \
  --temp 0 --max-tokens 512 \
  --chat-template-args '{"enable_thinking":false}'
```

The isolated CCR provider used:

```json
{
  "name": "local27",
  "api_base_url": "http://127.0.0.1:8081/v1",
  "api_key": "${LOCAL27_API_KEY}",
  "protocol": "openai",
  "maxInflight": 32,
  "models": ["/Users/kearm/models/Qwen3.8-27B-MLX-GPTQ-MXFP4"]
}
```

## Required compatibility facts

- `--chat-template-args '{"enable_thinking":false}'` is mandatory for this
  Qwen template. Without it, early responses place text in `message.reasoning`
  and leave OpenAI `message.content` null.
- The provider model ID must be the exact local model path. A friendly alias
  caused MLX LM to query Hugging Face and return 404.
- CCR's default provider AIMD initial limit is 8. A 32-way burst exhausted the
  tier immediately unless `maxInflight: 32` pinned the initial limit and
  ceiling.
- The MLX HTTP server warns that it is not production-oriented and provides
  only basic security checks. It must remain loopback-only behind CCR.

## Measured capacity

All decode cases used 128 generated tokens per request.

| Path | Concurrency | Aggregate tok/s | Median request tok/s | p50 latency |
|---|---:|---:|---:|---:|
| Calibrated MXFP4 direct | 1 | 43.2 | 22.4 | 5.72 s |
| Calibrated MXFP4 direct | 32 | 138.2 | 5.02 | 25.49 s |
| Community 4-bit direct | 1 | 44.5 | 22.7 | 5.63 s |
| Community 4-bit direct | 32 | 141.1 | 5.14 | 24.91 s |
| Calibrated MXFP4 through CCR | 16 | 84.8 | 5.77 | 22.18 s |
| Calibrated MXFP4 through CCR | 32 | 142.0 | 5.19 | 24.67 s |

CCR with `maxInflight: 32` preserved the direct MLX batch throughput. A
representative OpenAI request through CCR returned exactly `CCR_READY` in
normal `message.content`.

## Long-context constraint

Prompt ingestion, not decode batching, limits realistic agent concurrency.

- One approximately 8k-token prompt: 44.16 s, 182.3 prompt tokens/s.
- One approximately 32k-token prompt: 273.12 s, 117.4 prompt tokens/s.
- Two simultaneous approximately 8k-token prompts through CCR: 79.74 s wall
  time, 201.1 aggregate prompt tokens/s.

Two concurrent 8k prompts improved aggregate prefill only from about 182 to
201 tokens/s. The useful long-prefill concurrency is therefore about two
simultaneous 8k requests, not 32. Longer contexts should be queued or admitted
at concurrency one or two.

## Rejected MTP server mode

The standalone BF16 MTP drafter was also exercised with four draft tokens. A
server compatibility repair allowed a fresh empty prompt cache, while still
rejecting populated-cache reuse. MTP mode then worked, but:

- MLX LM marks a draft-model server non-batchable.
- Measured throughput was 10.9 median request tok/s.
- Draft acceptance was 242/520 proposals, or 46.54%.
- The plain batched server was substantially better for aggregate CCR
  fallback capacity.

Therefore MTP is not the fallback serving mode. The dirty MTP worktree remains
preserved for later single-stream optimization; this capacity decision does not
claim batched MTP support.

## Recommendation

Use the calibrated owned MXFP4 target, not the community control: throughput is
effectively tied while the prior one-prompt diagnostic recorded 44.141% BF16
token agreement for the calibrated target versus 1.953% for community 4-bit.

For short-context overflow, configure the loopback MLX server as an OpenAI
final CCR fallback with `maxInflight: 32` and cap worker admission at 32. For
long agent contexts, use a separate low-concurrency lane around one or two
in-flight prefills; do not allow 32 fresh 8k-plus prompts to compete directly.

## Non-claims

- This was an isolated capacity probe, not a change to deployed CCR listeners.
- No production CCR policy or listener configuration was modified.
- Exact-token agreement is not a broad quality claim.
- The 32-way decode result does not predict long-context prefill capacity.
- No model artifact or Hugging Face repository was changed.

