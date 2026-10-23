# Flint 224-expert MXFP4 artifact and local serving

Date: 2026-10-23
Agent: Codex

## Artifact

`/Users/kearm/models/RecursiveSI-Flint-40B-A2.5B-MLX-GPTQ-MXFP4-224`

- 224 of 256 routed experts, top-6 routing unchanged.
- Sliced from the calibrated full-256 MLX GPTQ-MXFP4 checkpoint with q/scales
  preserved.
- Mask SHA-256:
  `b34167c369b0ac419b21fc406dbd57540f7962b44ca5e3088d666f62ac20619b`.
- Expert-ID-map SHA-256:
  `d731ae02d470bdfa9e9b17984e7cb500d426da6862bfbc048167176f145bb949`.
- Parameter bytes: `19,556,113,984`, down from `22,075,711,040` for full 256.
- `artifact-manifest.json` SHA-256:
  `c29e79662aa028531ca7726f9bb9b7fcecc8490f38a3c39f0ccaad6f44cb0ce3`.
- `SHA256SUMS` SHA-256:
  `85bda633960b71e8e28e291fd06400fea7c6c315d48b2c8add6a47cdc32c6f33`.

`shasum -a 256 -c SHA256SUMS` passed for all 16 files, including all five
18.2 GiB payload shards.

## Quality

The physical artifact reproduced the runtime-pruned QDQ result exactly on 32
spread agentic-coding windows:

- full 256 MXFP4 PPL: `6.207109417615344`;
- 224 MXFP4 PPL: `6.2060948161381635`;
- NLL delta: `-0.00016347132623195648`;
- bootstrap 95% CI: `[-0.04942917847074568, 0.04476901260204613]`;
- full top-1: `0.668914794921875`;
- 224 top-1: `0.666748046875`.

This is teacher-forced token quality. No task-level agentic benchmark ran.

## Service

LaunchAgent: `dev.resmp.flint224-mxfp4`

- URL: `http://127.0.0.1:8084`
- Launcher: `/Users/kearm/mlx-stack/scripts/flint224_mxfp4_service.sh`
- Wrapper: `/Users/kearm/mlx-stack/scripts/flint_mlx_server.py`
- Model: the physical 224 MXFP4 artifact above.
- Limits: decode concurrency 4, prompt concurrency 2, prefill step 2048,
  default max tokens 2048, prompt cache size 32, backlog 1024.
- The runner starts Python through `env -i` with HOME, PATH, PYTHONPATH,
  PYTHONUNBUFFERED, and HF offline flags only. Inherited credential variables
  are not passed to the model process.

Lifecycle:

```bash
/Users/kearm/mlx-stack/scripts/flint224_mxfp4_service.sh restore
/Users/kearm/mlx-stack/scripts/flint224_mxfp4_service.sh status
/Users/kearm/mlx-stack/scripts/flint224_mxfp4_service.sh evict
```

## Live validation

- `/health`: `{"status":"ok"}`.
- `/v1/models`: includes the full local 224 model path.
- Non-stream completion: 64 tokens in 1.171 s.
- Streaming completion: SSE keepalive and token events received.
- Non-stream logprobs: four generated rows, each with 20 top logprobs, so the
  Shannon-entropy controller path works.
- A four-request synchronized burst completed 4/4 with no errors.
- Four-request benchmark using a 4,096-character real holdout prompt and 32
  generated tokens per request: aggregate completion 36.38 tok/s, sum of
  per-stream decode 222.73 tok/s, median first token 2.945 s, group wall
  3.518 s.
- One-request control on the same prompt: aggregate completion 43.42 tok/s,
  per-stream decode 90.82 tok/s, first token 0.395 s, request wall 0.737 s.

The concurrency numbers are a short local smoke, not a benchmark claim.

## Non-claims

- No QLoRA, CPT, or SFT has run.
- No task-level agentic benchmark has run.
- The inherited full-256 calibration predates the RoPE layout repair.
- The artifact is local-only and is not uploaded.
