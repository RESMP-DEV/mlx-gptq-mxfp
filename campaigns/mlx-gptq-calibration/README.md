# MLX GPTQ Calibration

Reproducible GPTQ calibration campaigns with the Mac as the authority for Git,
credentials, packing, and acceptance. A bounded SSH-accessed CUDA host performs
the expensive layer-wise solve; calibrated artifacts return to the Mac and are
packed as standard `mlx-lm` models.

## Campaigns

| Campaign | Source | Output |
| --- | --- | --- |
| Qwen3.8 27B | `Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | Native MLX MXFP4, group size 32, activation-Hessian GPTQ |
| Gemma 4 26B-A4B | `google/gemma-4-26B-A4B-it-qat-q4_0-unquantized@f1e06dc520982d9b9edd76859fdb7ab209449949` | Native MLX MXFP4 with routed-token Hessians |

Both use tokenizer-matched matrices from
`RESMP-DEV/ptq-calibration-corpus@ffa2e74b9361d573167373140f10dfd30c69a327`
and the engine at
`RESMP-DEV/mlx-gptq-mxfp@7131c407af8b8991fb5a8148a49cf2f323c7afd3`.
Token IDs are never reused across tokenizers.

The Qwen checkpoint is a vision-language wrapper, but calibration and packing
target its text model. The engine maps the Transformers, checkpoint, and MLX
layer namespaces explicitly; `mlx-lm` drops the unused vision tower.

The similarly named Gemma `...-unquantized-assistant` repository is only a
four-layer speculative drafter and is not the MoE target.

## Compressed-tensors CUDA output

`campaigns/qwen3.8-27b-mxfp4-vllm.json` reuses the same verified source model
and tokenizer-matched calibration matrix, but emits a standard
`compressed-tensors` MXFP4A16 artifact with llm-compressor's sequential GPTQ
pipeline. This is the CUDA/vLLM counterpart to the native MLX artifact:

- llm-compressor `0.14.0`; the upstream commit is a recipe reference, while the
  installed distribution is independently pinned by its installation `RECORD`
  SHA-256
- `GPTQModifier(targets="Linear", scheme="MXFP4A16", ignore=["lm_head"])`
- group size 32, damping 0.01, static activation order, and seed 17
- all 256 pinned calibration windows at 2,048 tokens, in original order
- the text-only `Qwen3_5ForCausalLM` graph; the vision tower and MTP drafter are
  not calibration targets

On Ampere, vLLM's compressed-tensors MXFP4 scheme selects its Marlin W4A16 path
(minimum compute capability 8.0). This is weight-only MXFP4, not native W4A4
FP4 tensor-core execution. `scripts/quantize_mxfp4_vllm.py` verifies the exact
package versions, calibration hash, source revision receipt, and output hashes
before writing its run receipt.

The B550 run compressed all 496 eligible text linear tensors in 3,606 seconds.
The raw llm-compressor output preserves the conditional wrapper and MTP shard;
`scripts/convert_mxfp4_text_only.py` therefore flattens the text graph to
`Qwen3_5ForCausalLM`. Its required `--tokenizer-source` restores
`tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`, and the
chat template byte-for-byte from the verified source checkpoint. This matters:
Transformers rewrote the saved tokenizer in the first packaging pass, which
changed Hindi input such as `नी` from one source token to two tokens. That first
artifact is superseded and must not be used.

The corrected, atomic, campaign-hash-bound artifact is
`Qwen3.8-27B-vLLM-MXFP4A16-GPTQ-text-only-source-tokenizer-v4` on B550. vLLM
`0.32.0` resolves it as `Qwen3_5ForCausalLM`, selects
`MarlinMxFp4LinearKernel`, loads 17.09 GiB of weights on one SM86 RTX 3090 Ti,
and completes a coherent deterministic generation. The exact hashes, service
restoration, superseded-tokenizer regression, and non-claims are recorded in
[`receipts/qwen3.8-27b-mxfp4-vllm/b550-summary.json`](receipts/qwen3.8-27b-mxfp4-vllm/b550-summary.json).
This is operational acceptance for CUDA/vLLM pilot use, not an accuracy claim;
matched BF16 perplexity or task parity remains unmeasured.

The A100 precision ladder now supplies the matched quality and throughput
control missing from that B550 operational run. The current fleet
recommendation is the private
`RESMP-DEV/Qwen3.8-27B-MXFP4A16-GPTQ-A100@a100-vllm-0.32.0-v3` at revision
`95709c64d92b8bc9747388638d6da00db4c0adff`: measured code-holdout perplexity
increased by 1.6366% while single-request decode was 2.1372x BF16 on the same
A100-SXM4-80GB. Task accuracy and batched serving remain unmeasured. The
matched protocol, host-specific hashes, publication receipt, and non-claims are
in [`docs/qwen38-precision-ladder.md`](docs/qwen38-precision-ladder.md).

## Gemma hybrid BF16 variants

The Gemma campaign keeps router, gate, shared-expert gate, and score-correction
paths out of low-bit conversion. It supports either the QAT router or a small
ordinary-parent BF16 router overlay.

See [docs/hybrid-bf16-surgery.md](docs/hybrid-bf16-surgery.md) for the ablation
matrix, routing measurements, overlays, and control models.

## Quick start

The orchestration package itself has no third-party runtime dependency:

```bash
uv sync
uv run mlx-gptq-calibration verify
uv run mlx-gptq-calibration probe --host YOUR_CUDA_SSH_ALIAS
```

Select Gemma with:

```bash
GEMMA=campaigns/gemma4-26b-a4b-qat-mxfp4.json
uv run mlx-gptq-calibration --campaign "$GEMMA" verify
uv run mlx-gptq-calibration --campaign "$GEMMA" probe --host YOUR_CUDA_SSH_ALIAS
```

The practical floor is one CUDA GPU with at least 8,000 MiB VRAM. Calibration
streams one decoder layer at a time and derives per-GPU budgets from live free
memory. Every eligible GPU is used independently; mixed-capacity devices are
not throttled to the smallest card.

Prepare and run the remote stage:

```bash
uv run mlx-gptq-calibration prepare --host YOUR_CUDA_SSH_ALIAS
uv run mlx-gptq-calibration start --host YOUR_CUDA_SSH_ALIAS
uv run mlx-gptq-calibration status --host YOUR_CUDA_SSH_ALIAS
uv run mlx-gptq-calibration fetch --host YOUR_CUDA_SSH_ALIAS
```

Preparation reuses complete pinned snapshots when possible. Otherwise it uses
`hfd`, resumes the transfer, and verifies every LFS SHA-256 or Git blob before
calibration begins.

Fetch the optional Gemma parent-router overlay with:

```bash
uv run mlx-gptq-calibration --campaign "$GEMMA" fetch-overlay
```

## Pack locally

Stage B requires the exact pinned source model and an MLX-enabled engine
environment. For Qwen:

```bash
hfd download Qwen/Qwen3.8-27B \
  --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --local-dir ~/models/Qwen-Qwen3.8-27B-1d4bf0f

uv sync --directory ~/mlx-gptq --frozen --extra mlx
~/mlx-gptq/.venv/bin/python ~/mlx-gptq/tools/patch_mlx_lm_tf512.py

uv run mlx-gptq-calibration pack \
  --model ~/models/Qwen-Qwen3.8-27B-1d4bf0f
```

Each phase writes a machine-readable receipt under the ignored `runs/`
directory. Calibration is resumable from completed layer artifacts.

## Validation

For matched BF16-to-quantized GGUF validation, run:

```bash
uv run mlx-gptq-logit-trace \
  --llama-perplexity /path/to/llama-perplexity \
  --reference /path/to/reference.gguf \
  --candidate /path/to/candidate.gguf \
  --corpus /path/to/heldout.txt \
  --output-dir runs/logit-trace \
  --chunks 32 --ctx-size 512 --batch-size 512
```

The trace records full-vocabulary KL, top-1 agreement, probability deltas,
paired perplexity, timings, commands, and artifact hashes. Long-context runs can
use `patches/llama-perplexity-tail-logits.patch` to score only a bounded tail
while retaining the full prefix.

### M4 and CUDA MLX parity

A live 2026-08-24 probe compared the same packed Qwen3.8 MXFP4 artifact under
MLX 0.32.0 and `mlx-lm` 0.31.3 on an M4 Max and an RTX 3090 Ti. Across 64
teacher-forced continuation positions:

- Top-1 token agreement: `64/64`
- Mean logit cosine similarity: `0.999835`
- Minimum logit cosine similarity: `0.998585`
- Mean absolute target log-probability delta: `0.00655`

For this probe, the deviation is within expected cross-backend numerical
variance and did not change a top-1 decision. This is not bit-exact parity:
free-running generation can diverge after an early near-tie and then amplify
that difference autoregressively.

The CUDA server also required an explicit-key sampling shim because global RNG
sampling was degenerate inside its worker thread. The shim lives in
`scripts/mlx_lm_server_cuda_dist.py`. Long benchmark runs should disable retained
prompt caches with `--prompt-cache-size 0` on 24 GB cards.

## Safety properties

- Model, dataset, engine, and tokenizer-specific calibration inputs are pinned.
- Packing refuses an engine checkout at the wrong commit.
- Hardware and memory budgets come from live probing, not host labels.
- Remote work is campaign-scoped and does not own repository credentials.
- Published model repositories are private by default.
