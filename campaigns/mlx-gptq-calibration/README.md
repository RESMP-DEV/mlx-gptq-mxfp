# MLX GPTQ Calibration

Reproducible calibration campaigns that keep the Mac authoritative while a
bounded SSH-accessed CUDA host performs the expensive GPTQ solve. The returned
artifacts are packed and verified locally as standard `mlx-lm` models.

The first campaign is pinned to:

- Model: `Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- Corpus: `RESMP-DEV/ptq-calibration-corpus@ffa2e74b9361d573167373140f10dfd30c69a327`
- Engine: `RESMP-DEV/mlx-gptq-mxfp@7131c407af8b8991fb5a8148a49cf2f323c7afd3`
- Output: native MLX MXFP4, group size 32, activation-Hessian GPTQ with safe
  E8M0 scale search

The first campaign consumes the dataset's default Qwen3.8-tokenized
`data/calibration_256x2048.npy`. The model-neutral `calibration.txt` remains
available for targets without a published matching matrix; token IDs are never
reused across tokenizers.

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

## Architecture

```text
Mac (Git, credentials, pins, acceptance)
  | SSH: prepare/start/status
  | rsync: tokenizer-matched calibration.npy ->
  v
CUDA host (`~/models` checkpoint library + layer-streamed GPTQ)
  | rsync: calibrated q/scales artifacts ->
  v
Mac (mlx-lm conversion, injection, generation verification)
```

The checkpoint is a native vision-language wrapper. Transformers exposes a
text-only causal runtime at `model.layers`, while the raw checkpoint uses
`model.language_model.layers` and MLX uses `language_model.model.layers`. The
pinned engine maps those three namespaces explicitly. The vision tower is not
used to generate calibration activations and `mlx-lm` drops it during
conversion, so the final artifact is the Qwen3.8 text-generation model.

The second campaign targets the actual Gemma 4 MoE checkpoint:

- Model: `google/gemma-4-26B-A4B-it-qat-q4_0-unquantized@f1e06dc520982d9b9edd76859fdb7ab209449949`
- Architecture: 25.2B total parameters, 3.8B active, 128 experts with top-8 routing
- Source: BF16 weights extracted from Google's Q4_0 quantization-aware training
- Output: native MLX MXFP4 with routed-token Hessians for every active expert

The similarly named `...-unquantized-assistant` repository is only an 839 MB,
four-layer dense speculative-decoding drafter. It is pinned in the campaign as
an optional companion, but it is not substituted for the MoE target.

## Hybrid BF16 router experiment

The Gemma campaign also pins its ordinary BF16 parent and can fetch only the
21.8 MB router state as a safetensors overlay. This supports two distinct
mixed-precision variants without downloading another 51.6 GB checkpoint:

- QAT-derived GPTQ MXFP4 with the QAT router retained in BF16.
- The same packed model with the ordinary parent BF16 router transplanted.

The campaign's packing regex explicitly keeps `router.*`, `gate`, shared-expert
gates, and score-correction paths out of low-bit conversion. This also prevents
the custom pack predicate from accidentally overriding Gemma 4's native router
policy and quantizing `router.proj` to ordinary 4-bit RTN.

See [docs/hybrid-bf16-surgery.md](docs/hybrid-bf16-surgery.md) for the evidence,
ablation matrix, routing metrics, the pinned Qwen3.6-35B-A3B non-QAT control, and
the artifact-level audit of JoyAI-LLM Flash as a second genuine QAT MoE.

Long-context evaluation uses the separately isolated
`agentic_coding_holdout/heldout` config at
`RESMP-DEV/ptq-calibration-corpus@6ef2bbf949b94192ee8bb78d45a2fc840a10cff5`.
It contains 2,429,702 Gemma 4 tokens of commit-pinned repository source, tests,
and documentation with zero normalized 20-word-shingle overlap against the
calibration text.

## Quick start

The project itself has no third-party runtime dependency:

```bash
uv sync
uv run mlx-gptq-calibration verify
uv run mlx-gptq-calibration probe --host YOUR_CUDA_SSH_ALIAS
```

Select campaign two with:

```bash
GEMMA=campaigns/gemma4-26b-a4b-qat-mxfp4.json
uv run mlx-gptq-calibration --campaign "$GEMMA" verify
uv run mlx-gptq-calibration --campaign "$GEMMA" probe --host YOUR_CUDA_SSH_ALIAS
```

The practical floor is one CUDA GPU reporting at least 8,000 MiB VRAM. This is
not a full-model residency requirement: the pipeline streams one decoder layer
at a time and derives the per-GPU expert-solve budget from live free memory,
while reserving headroom for the driver, layer, and activation working set.
Sub-8 GB operation can be forced with `--allow-underprovisioned`, but is likely
to lose the practical advantage over local MLX calibration. There is no
multi-GPU requirement or recommendation; every eligible GPU present is used.
Budgets are computed independently, so mixed 24 GB, 32 GB, and 96 GB devices
retain their own expert-chunk capacity instead of being throttled to the
smallest GPU.

Prepare the isolated remote environment and pinned 55.6 GB checkpoint:

```bash
uv run mlx-gptq-calibration prepare --host YOUR_CUDA_SSH_ALIAS
```

`prepare` discovers complete pinned snapshots in `~/models` and the Hugging
Face cache, transferring them with checksummed `rsync`; it falls back to the
independent `hfd` CLI on the CUDA host. The fallback disables Xet, pins the Hub
revision, resumes through aria2, and verifies every LFS SHA-256 or Git blob
before calibration can start. Install `hfd` in the remote `PATH` (or at
`~/.local/bin/hfd`). Remote checkpoints live in a revision-suffixed `~/models`
directory so other campaigns and tools can reuse them. An explicit local source
can be selected with `--local-model PATH`.

Start the resumable CUDA stage in the background, then monitor it:

```bash
uv run mlx-gptq-calibration start --host YOUR_CUDA_SSH_ALIAS
uv run mlx-gptq-calibration status --host YOUR_CUDA_SSH_ALIAS
```

Fetch completed calibration artifacts:

```bash
uv run mlx-gptq-calibration fetch --host YOUR_CUDA_SSH_ALIAS
```

Fetch the pinned parent-BF16 router overlay for the Gemma hybrid:

```bash
uv run mlx-gptq-calibration --campaign "$GEMMA" fetch-overlay
```

Run a matched full-vocabulary logit trace after exporting two variants to GGUF:

```bash
uv run mlx-gptq-logit-trace \
  --llama-perplexity /path/to/llama-perplexity \
  --reference /path/to/qat-bf16.gguf \
  --candidate /path/to/candidate.gguf \
  --corpus /path/to/agentic-coding-long-context.txt \
  --output-dir runs/gemma4-logit-trace \
  --chunks 32 --ctx-size 512 --batch-size 512
```

The trace records full-vocabulary KL, tail KL, top-1 token agreement, token
probability deltas, paired perplexity, commands, timings, sizes, and SHA-256
identities. It recomputes the perplexity ratio from the unclipped reference
pass because llama.cpp's compact log-probability trace clips its far tail.
Use `--reuse-reference-logits` to compare additional hybrid variants against
the same stored reference trace without rerunning the 50 GB BF16 model.

For long contexts, apply
[`patches/llama-perplexity-tail-logits.patch`](patches/llama-perplexity-tail-logits.patch)
to the pinned llama.cpp source and score a bounded tail while retaining the
entire prefix:

```bash
uv run mlx-gptq-logit-trace \
  --llama-perplexity /path/to/patched/llama-perplexity \
  --reference /path/to/qat-bf16.gguf \
  --candidate /path/to/candidate.gguf \
  --corpus /path/to/agentic-coding-long-context.txt \
  --output-dir runs/gemma4-logit-trace-262144 \
  --ctx-size 262144 --tail-tokens 256 --chunks 1 \
  --batch-size 4096 --ubatch-size 512 --flash-attn on \
  --gpu-layers 20 --no-kv-offload
```

The primary long-context workload is a deterministic repository-context stream
built from commit-pinned Apache-2.0 source repositories. It includes source,
tests, and documentation, but excludes SWE-bench Science task prompts,
verifiers, trajectories, and answer patches:

```bash
uv run scripts/build_agentic_coding_holdout.py \
  --calibration-text /path/to/calibration.txt \
  --workspace /path/to/pinned-repositories \
  --output-dir /path/to/agentic-coding-holdout
```

The Ultra-FineWeb-derived stream remains a separately reported general-text
control. Neither holdout may be used as GPTQ calibration input.

Stage B needs the exact campaign source model on the Mac and an MLX-ready
environment in the pinned engine checkout. For Qwen3.8:

```bash
hfd download Qwen/Qwen3.8-27B \
  --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --local-dir ~/models/Qwen-Qwen3.8-27B-1d4bf0f

uv sync --directory ~/mlx-gptq --frozen --extra mlx
~/mlx-gptq/.venv/bin/python ~/mlx-gptq/tools/patch_mlx_lm_tf512.py

uv run mlx-gptq-calibration pack \
  --model ~/models/Qwen-Qwen3.8-27B-1d4bf0f
```

Every phase writes a machine-readable receipt below the ignored
`runs/qwen3.8-27b-mxfp4/` directory. Calibration is crash-safe and resumes from
the layer artifacts already present on the CUDA host.

## Local artifacts

The Mac currently has the verified Qwen3.8 token matrix used by campaign one
and the Gemma 4 matrix used by campaign two, so each is sent directly instead
of being downloaded again. `~/Qwen3.6-27B` has the same
Qwen3.5-family 64-layer shape and is useful for compatibility experiments, but
its weights, config, tokenizer metadata, and 15-shard index do not match the
pinned Qwen3.8 18-shard checkpoint. It is never accepted as the model input.

## Safety properties

- Model, dataset, engine, and tokenizer-specific calibration content are
  independently pinned.
- Remote work lives under a campaign-specific directory and never owns Git or
  credentials for this project.
- The remote start is detached, logged, resumable, and refuses duplicate runs.
- Hardware requirements are proven from live `nvidia-smi`, not host labels.
- Packing refuses an engine checkout at the wrong commit.
- Published model repositories are private by default.
