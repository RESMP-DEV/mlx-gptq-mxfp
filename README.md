# MLX GPTQ MXFP

GPTQ-calibrated quantization for **standard MLX (mlx-lm) models**, with the
expensive calibration running on **CUDA** where it's fast, and only the final
packing on the Mac.

Why: `mlx_lm.convert` is round-to-nearest (no calibration), and mlx-lm's
DWQ/AWQ run on the Mac — far too slow for large MoE models. This tool solves
GPTQ directly on the MLX affine, native MXFP4, or native MXFP8 grid. MXFP4
uses E2M1 values with one E8M0 scale per 32 weights; MXFP8 uses E4M3 values
with the same E8M0 grouping. Codes, scales, and packing are verified bit-exact
against live `mlx.core`, so the calibrated result loads as a 100% standard
mlx-lm model. No custom kernels or runtime changes.

## MoE: every expert gets real calibration

transformers ≥ 5 stores experts as fused 3D parameters and dispatches with
`F.linear(tokens, weight_3d[expert])`. Stage A intercepts those calls with a
`TorchFunctionMode`, matching each expert's weight slice by data pointer — so
**each expert accumulates a Hessian from exactly the tokens the router sent
to it**. Experts are then solved in batches (one batched Cholesky + column
loop across many experts at once) spread over all GPUs. Experts that received
zero calibration tokens are logged and fall back to data-free rounding.

The router/gate, embeddings, and `lm_head` stay in bf16 by default. The latter
two can be explicitly RTN-quantized with `--head-bits/--embed-bits`; they are
never silently data-free-quantized in a calibrated artifact.

## Workflow

### Stage A — calibrate on the CUDA box

```bash
# env: python 3.11+, torch (CUDA), transformers, safetensors, accelerate, tqdm
uv venv .venv && uv pip install -p .venv/bin/python -e /path/to/mlx-gptq

python -m mlx_gptq.calibrate \
    --model Qwen/Qwen3-30B-A3B \
    --output ./calib-qwen3-30b \
    --dataset ~/calibration_datav3.txt \
    --nsamples 256 --seqlen 2048 --batch-size 4 \
    --bits 4 --group-size 64 \
    --devices cuda:0,cuda:1,cuda:2,cuda:3 --vram-gb 16
```

For native microscaling output, select `--mode mxfp4` or `--mode mxfp8`.
Both modes require the native group size of 32; their bit width is selected
automatically. Existing deterministic token windows can be reused with
`--calibration-tokens /path/to/calib_tokens.npy` instead of `--dataset`.
The MXFP solver runs sequential GPTQ inside each native 32-value scale group;
cross-group compensation is intentionally excluded because it crosses
independent MXFP scale boundaries. `--mxfp-algorithm auto` uses native-scale
groupwise GPTQ for MXFP8 and, for MXFP4, activation-Hessian E8M0 scale search
with a per-row/group non-regression fallback. Both variants can be selected
explicitly with `--mxfp-algorithm groupwise|safe-scale-search`.

- The model is **never fully loaded**: a meta skeleton is built and each
  layer's weights stream from the safetensors shards on demand. RAM usage is
  activations + one layer + captured inputs, independent of model size.
- Datasets: `file.txt`, `file.jsonl[:field]`, `hf:name[:split[:field]]`, and
  `hfchat:name[#config][:split[:maxrows]]` — comma-separated to mix. Windows
  are sampled GPTQ-style from the EOS-joined token stream.
- `hfchat:` renders chat/agent datasets through the model's own chat template
  (tool schemas included), so calibration covers real `<tool_call>` token
  distributions. Streams, handles messages(role/content/tool_calls),
  ShareGPT(from/value), and function_call/function-role schemas.
  Agentic mix that renders clean through Qwen templates:
    hfchat:Agent-Ark/Toucan-1.5M#Qwen3:train:400            (real MCP tool loops)
    hfchat:Kwai-Klear/SWE-smith-mini_swe_agent_plus-trajectories-66k:train:40
                                                            (long terminal/code agent runs)
    hfchat:minpeter/xlam-function-calling-60k-parsed#xlam-function-calling-60k:train:400
                                                            (parallel/multiple tool calls)
    hfchat:NousResearch/hermes-function-calling-v1#func_calling:train:200
                                                            (tool use + json mode)
- `--override 'REGEX=BITS'` for mixed precision (e.g.
  `--override '.*down_proj.*=8'`). Packable bits: 2/4/8. Note experts of one
  proj share a stacked MLX tensor, so overrides must hit all of them equally.
- Crash-safe: artifacts append per layer; re-running with the same `--output`
  resumes (finished layers replay their quantized weights for propagation).
- `--clip mse` (default) does a GPTQ-style range-shrink search per group;
  scales/biases are rounded to bf16 *before* error feedback, so the solver
  optimizes exactly what inference will see.

### Stage B — pack on the Mac

```bash
rsync -a cuda-host:calib-qwen3-30b ./
python -m mlx_gptq.pack \
    --hf-path Qwen/Qwen3-30B-A3B \
    --calib ./calib-qwen3-30b \
    --mlx-path ./Qwen3-30B-A3B-4bit-gptq \
    --verify
```

Runs `mlx_lm.convert` (RTN, same bits/group, router excluded), then rewrites
the shards in place with the calibrated `q/scales/biases` — per-expert
artifacts are stacked into the `switch_mlp` tensors. Shapes/dtypes are
unchanged so the index and config stay valid. Injection fails loudly if any
artifact goes unconsumed (naming mismatch) or bits/group disagree with the
converted config.

### LFM ShortConv AWQ + GPTQ mixed quantization

`mlx_gptq.awq_lfm` starts from a partially quantized LFM model whose ShortConv
`in_proj` and `out_proj` modules remain BF16. It keeps all existing GPTQ
modules unchanged, learns activation-aware channel scales for the ShortConv
projections, fuses those scales through `operator_norm` and the convolution's
`C` branch, and packs only those projections as native MXFP4:

```bash
python -m mlx_gptq.awq_lfm \
  --model ./lfm-gptq-mxfp4-conv-bf16 \
  --calibration ./calibration/calib_tokens.npy \
  --output ./lfm-gptq-awq-mxfp4
```

The output uses MLX's per-module quantization configuration. `AWQ_RECEIPT.json`
records the calibration shape, selected scale exponents, module count, and
weight hashes. `mlx_gptq.mix` can compose matched BF16/MXFP4/MXFP8 module-level
controls without dequantizing or requantizing either source.

## Validated

- Bit-exact grid/packing vs `mlx.core` (2/4/8-bit, incl. stacked expert tensors).
- Qwen3-0.6B end-to-end: wikitext PPL **24.60 (GPTQ) vs 25.71 (RTN)** at 4-bit
  g64 with only 16×512 calibration tokens.
- Tiny Qwen3-MoE end-to-end: per-expert routed-token capture (token counts
  match top-k routing exactly), per-expert artifacts stacked into
  `switch_mlp`, final MLX logits cosine 0.991 vs the torch reference carrying
  identical weights; router kept bf16.

## Supported architectures

Anything whose decoder layers live at `model.layers` (else `--layers-attr`,
e.g. `model.language_model.layers` for ConditionalGeneration wrappers) and
whose MoE uses the standard fused-expert pattern: Qwen3-MoE, Qwen3.5-MoE
(incl. the hybrid linear-attention layers — their in/out projections are
plain `nn.Linear` and get calibrated too), DeepSeek-V2/V3 (bf16 checkpoints),
GLM-4.x-MoE, Mixtral, dense Llama/Qwen/etc. Both per-expert and fused raw
checkpoint layouts materialize correctly. Known gaps:

- **GPT-OSS**: transposed/interleaved fused checkpoint layout — not yet.
- **FP8 checkpoints** (DeepSeek-V3/Kimi native): needs a block-dequant step in
  the shard reader — planned; use bf16 checkpoints meanwhile.
- 3/5/6-bit MLX packing (different byte layout). RTN via mlx-lm still works
  for those (e.g. `--head-bits 6` is fine — that's packed by mlx itself).

## Tests

```bash
.venv/bin/python -m pytest tests/
```

Includes bit-exactness of grid + packing against `mlx.core`, GPTQ-beats-RTN
on correlated inputs, batched-solve == per-item solve, and fused-expert
capture with routing-count invariants.

## Repository scope

This repository contains generalized source code and synthetic tests only. It
does not contain model weights, access tokens, private calibration corpora, or
pre-tokenized calibration windows. Generated artifacts are deliberately kept
in separate model repositories so code review and model access remain
independent.
