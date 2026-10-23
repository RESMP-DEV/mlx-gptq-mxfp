# Flint independent perplexity and upstream audit

Date: 2026-10-23
Agent: Codex

## Scope

Run a second correctness gate on the repaired public Flint MXFP4 artifact using
an independent natural-language corpus, compare it with the pinned BF16 source,
check the model-level and MLX-LM upstream refs, and classify the checkpoint's
attention and prompt-cache representation.

## Runtime identity

- Hardware: Apple M4 Max, 128 GiB unified memory.
- MLX: 0.32.4.
- MLX-LM: `b384a406ad5cac168877c32b048f31aa9c8db244`.
- BF16 source: `/Users/kearm/models/RecursiveSI-flint-BF16-2a247be`.
- MXFP4 artifact: `build/flint-resharded`, identical weight lineage to the
  public 25-shard release.

The published tokenizer JSON and tokenizer configuration are serialization
different from the source files, but tokenization was checked over the complete
evaluation fixture. Both tokenizers produced the same 32,768 token IDs. The
chat template is byte-identical.

## Independent perplexity result

Corpus fixture:

- `Salesforce/wikitext`, configuration `wikitext-2-raw-v1`, validation split.
- Parquet SHA-256:
  `204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c`.
- Rows: 3,760; non-empty rows: 2,461; characters: 1,142,150.
- The source tokenizer was applied to rows in order without BOS/EOS insertion.
- The first 32,768 tokens were divided into sixteen fixed 2,048-token windows.
- Token-array SHA-256:
  `290eee254adf1725b939809db511baf4f467326457cb189f693d1fd72262a428`.
- Both arms scored the same 32,752 next-token transitions.

| Arm | Mean NLL | PPL | PPL SE | Target top-1 | Peak memory |
|---|---:|---:|---:|---:|---:|
| BF16 source | 3.1075680852 | 22.366584579 | 0.04796326 | 48.8337% | 82.067 GB |
| GPTQ MXFP4 | 3.0581631809 | 21.288418255 | 0.04732859 | 48.3512% | 35.481 GB |

Paired comparison:

- MXFP4/BF16 PPL ratio: 0.951795666.
- Mean NLL delta, MXFP4 minus BF16: -0.049404904 nats/token.
- Standard error over the sixteen paired windows: 0.016944180 nats/token.
- Paired window-bootstrap 95% CI: `[-0.082137590, -0.018019710]` nats/token.
- Exact MXFP4-versus-BF16 top-1 agreement: 73.1986%.
- Predictions differed on 8,778 of 32,752 positions.

The lower MXFP4 perplexity is repeatable on this fixture, but lower perplexity
does not by itself prove better general quality: BF16 remains the fidelity
reference, MXFP4 has slightly lower target top-1 accuracy, and the two models
disagree on 26.8% of next-token choices.

The direct evaluator uses cache-free forward passes. Therefore this is a
forward-numerics correctness gate, not a generation or cache performance test.

Regression checks with MLX 0.32.4:

```text
tests/test_flint.py: 4 passed
tests/test_prompt_cache.py: 25 passed
```

Verification artifacts:

- `build/reports/flint-correctness/wikitext2-bf16.json`
- `build/reports/flint-correctness/wikitext2-mxfp4.json`
- `build/reports/flint-correctness/wikitext2-paired-summary.json`
- `build/reports/flint-correctness/wikitext2-bf16.log`
- `build/reports/flint-correctness/wikitext2-mxfp4.log`

## Upstream audit

### Model-level upstream

Live Hugging Face API checks found:

- `RecursiveSI/flint` head:
  `2a247be0c0aec895ccb0ac7ca92229ec6df37288`
- `RecursiveSI/flint-int4-gguf` head:
  `9c47151036bc2c34331142dda04488bceab2bc9f`

These match the already pinned source and official-control revisions. There is
no new model-level revision to recalibrate against or carry forward.

### MLX-LM upstream

The fetch-only `upstream/main` was refreshed to:
`59995f131cf2b48950a67d2eb36d76b2a554b989`.

Upstream contains no Flint model implementation or tests. The fork therefore
remains the only source of the Flint adapter and its routed-expert correction.

The only broadly portable upstream fixes found were:

- `33fdd16`: preserve NVFP4 global scales during dtype conversion and
  dequantization.
- `72be8c9`: repair AFM7 RoPE offsets for left-padded KV-reuse batches.

Neither affects this MXFP4 Flint artifact. The remaining upstream commits add or
repair unrelated model families, GLM DSA sparse prefill, MiniMax loading,
TextMachine behavior, or test layout. They were not imported merely to sync
history.

## Attention and cache classification

The model is a fixed hybrid of ordinary softmax attention layers:

- 14 `full_attention` layers at indices
  `0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 48`.
- 35 `sliding_attention` layers with a fixed 513-token window.
- 32 query heads, 4 grouped-query KV heads, and head dimension 128.

`FlintModel.make_cache()` creates:

- ordinary `KVCache` for full-attention layers;
- `RotatingKVCache(max_size=513)` for sliding layers.

Both classes retain standard key and value tensors. There is no linear-state,
recurrent-state, delta-rule, or other nonlinear compressed cache in this
implementation.

`softmax_type: learnable` is implemented as one learnable offset per query
head, passed as attention sinks to the standard scaled-dot-product attention
kernels. The public artifact has all 49 `[32]` `softmax_offset` tensors, and
all 1,568 loaded offsets are non-zero. This modifies attention logits but not
the KV representation.

With ordinary 16-bit K/V tensors, the theoretical cache cost is:

- full-attention layers: 28 KiB per token across all 14 full layers;
- sliding layers: 35.068 MiB steady-state per sequence once their 513-token
  windows are full.

This excludes logits, activations, masks, metadata, and runtime allocation
slack. The current `RotatingKVCache` also does not implement its
`to_quantized` conversion path, so generic quantized KV conversion is not a
drop-in property of the hybrid cache.

## Non-claims

- WikiText-2 is one natural-language domain and sixteen contiguous windows; it
  does not replace agentic, long-context, instruction-following, or coding
  evaluations.
- The paired bootstrap is over windows, not independent token draws, because
  neighboring tokens in natural language are correlated.
- A lower quantized PPL than BF16 is not evidence that quantization improved
  the model. It is a measured corpus-level effect with lower top-1 fidelity.
- The upstream audit proves refs at check time and does not claim future
  upstream revisions are absent.
- Cache classification is source and live-tensor evidence, not a separate
  boundary-crossing generation parity benchmark.
