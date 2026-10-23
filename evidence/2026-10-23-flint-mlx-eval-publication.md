# Flint MLX evaluation and publication receipt

Date: 2026-10-23
Agent: Codex

## Scope

Unload the resident local27 service if necessary, test the owned Flint
GPTQ-MXFP4 checkpoint on Apple Silicon, repair any adapter defect found, apply
a quality gate, and publish the artifact under RESMP-DEV.

## Runtime state

- The resident local27 service was already down after the restart:
  `dev.resmp.local27-mxfp4` and `dev.resmp.local27-admission` were not loaded,
  and port 8081 had no listener.
- The official Apple control was `RecursiveSI/flint-int4-gguf` revision
  `9c47151036bc2c34331142dda04488bceab2bc9f`, Q4_1, downloaded with `hfd`
  and fully verified at 25.99 GB.
- The control runtime was llama.cpp revision
  `a37cc46104eb50900d5ab4a9ba02a0abdac039db`, built locally with Metal and
  the merged Flint support commit.

## Adapter repair

The first MLX tests were incoherent:

- BF16 greedy output was non-language text.
- MXFP4 greedy output was non-language text.
- Initial eight-window perplexity was about 9.7 million for BF16 and 12.4
  million for MXFP4.

Root cause: `mlx_lm/models/flint.py` passed the raw `swiglu(gate, x)` function
to `SwitchGLU`. `SwitchGLU` invokes `activation(x_up, x_gate)`, so every routed
expert swapped its gate and up projections. The dense and shared-expert MLPs
were correct.

The fix removes the raw function and uses `SwitchGLU`'s standard activation.
The new regression test checks routed-expert gate/up order against an explicit
SwiGLU calculation.

- MLX-LM fix: `RESMP-DEV/mlx-lm` commit
  `b384a406ad5cac168877c32b048f31aa9c8db244`
  (`fix: correct Flint routed expert activation order`)
- Test command: `PYTHONPATH=. uv run --no-project --with 'mlx==0.32.4'
  --with pytest --with numpy python -m pytest -q tests/test_flint.py`
- Result: 4 passed.
- Lint command: `uv run --no-project --with ruff --with pytest --with numpy
  ruff check mlx_lm/models/flint.py tests/test_flint.py`
- Result: passed.

The eager PyTorch calibration skeleton already used the correct gate/up order,
so the existing GPTQ calibration artifacts remained valid after the MLX
adapter repair.

## Corrected evaluation

All perplexity rows use the same held-out agentic-coding corpus, eight windows
of 2,048 tokens, and the base checkpoint tokenizer.

| Arm | Runtime | Result |
| --- | --- | ---: |
| BF16 source | MLX-LM | PPL 8.5360 |
| Owned GPTQ MXFP4 | MLX-LM | PPL 7.4038 |
| Official Apple Q4_1 | llama.cpp Metal | PPL 6.1512 |

Direct comparison over all 16,384 target tokens:

- MXFP4 versus BF16 next-token top-1 agreement: 77.8625%.
- Mean target log-probability delta, MXFP4 minus BF16: +0.14230 nats.
- Minimum and maximum per-token deltas were -19.6959 and +24.9337 nats.

Corrected greedy behavior:

- BF16 and MXFP4 both generated coherent reasoning and identified Paris as
  the capital of France.
- The 25-shard local publication fixture generated the same coherent prefix
  and loaded with 35.481 GiB peak memory.

Single-stream M4 Max performance:

- Owned MXFP4 decode, 256 tokens: 131.532 tok/s.
- BF16 decode, 48 tokens: 66.796 tok/s.
- Official Q4_1 decode, 256 tokens: 106.67 tok/s.
- Owned MXFP4 prefill, 8,194 prompt tokens: 1,563.674 tok/s.

## Publication

- Public repository:
  `https://huggingface.co/RESMP-DEV/Flint-40B-A2.5B-MLX-GPTQ-MXFP4`
- Final model-card head:
  `02658ab3aed9cb61b25a239ecb539c48911c5642`
- Weight commit:
  `c8bf433ce383c84262b6d3f5881be5b8c4338665`
- Adapter commit:
  `b384a406ad5cac168877c32b048f31aa9c8db244`
- Format: MXFP4, group size 32, 49 layers, 25 safetensors shards.
- Published shard bytes: 22,075,859,864.
- Remote state is public, has 25 current safetensors files, and has no stale
  five-shard filenames.
- Every one of the 25 remote LFS objects matched the local file size and
  SHA-256 before the final two README-only commits.

The initial five-shard publication attempt encountered Xet CAS instability and
an effectively hung Xet pre-push. The working transfer path was Git LFS with
Hugging Face's multipart adapter, the machine-global Xet adapter suppressed
for the push, and the model resharded into 25 roughly 1 GB files. Sixteen
concurrent multipart transfers completed all 22 GB at 9.1 MB/s.

## Verification artifacts

- Corrected BF16 perplexity:
  `build/reports/flint/fixed-eval-ppl-bf16.log`
- Corrected MXFP4 perplexity:
  `build/reports/flint/fixed-eval-ppl-owned-mxfp4.log`
- Official Q4_1 control:
  `build/reports/flint/official-int4-gguf-ppl.log`
- Output parity:
  `build/reports/flint/fixed-parity-8window.json`
- Local 25-shard load:
  `build/reports/flint/resharded-load-greedy.log`
- Remote hash and size verification:
  `build/reports/flint/remote-verification.json`
- Final public-head state:
  `build/reports/flint/remote-final-head.json`

## Non-claims

- This does not reproduce upstream Terminal-Bench, SWE-bench Pro, or
  SWE-bench Verified scores on the quantized artifact.
- Perplexity and top-1 agreement measure fidelity on one held-out coding
  corpus; they do not prove general agentic quality.
- The official Q4_1 control used llama.cpp, while BF16 and MXFP4 used MLX-LM.
  The tokenizer, hardware, text, window count, and sequence length match, but
  the runtimes differ.
- Throughput is single-stream M4 Max data, not a serving benchmark.
- The full public repository was not downloaded again after upload. Remote
  integrity is proven by matching every current remote LFS size and SHA-256;
  final post-weight commits changed only README text.
