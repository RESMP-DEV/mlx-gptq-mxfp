# Hybrid BF16/4-bit MoE model surgery

## Claim under test

A QAT checkpoint may be a better source for 4-bit expert and projection weights
than its ordinary BF16 parent, while routing and other sensitive boundaries may
benefit from remaining in BF16. A more speculative variant transplants the BF16
router from the non-QAT parent into the QAT-derived 4-bit model.

The transplant is not assumed to win. The QAT router and QAT experts may have
co-adapted, so a numerically cleaner parent router can still choose worse experts.
The experiment must therefore distinguish precision recovery from cross-checkpoint
compatibility.

Both candidate routers are already stored in BF16. Retaining the QAT router is a
real precision-preservation policy; replacing it with the parent router is not an
upcast, but cross-checkpoint component transfer.

## Verified checkpoint compatibility

Live Hugging Face API and safetensors-header probes on 2026-08-22 established:

- QAT source: `google/gemma-4-26B-A4B-it-qat-q4_0-unquantized`
  at `f1e06dc520982d9b9edd76859fdb7ab209449949`.
- Parent source: `google/gemma-4-26B-A4B-it`
  at `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`.
- Both checkpoints contain exactly 1,013 tensors and 51,611,872,412 tensor
  bytes in two shards.
- Tensor names, shapes, and dtypes match 1,013/1,013. Their configs differ only
  in the recorded Transformers version.
- The complete router state is 90 BF16 tensors and 21,803,520 bytes. The 30
  `router.proj.weight` tensors account for 21,626,880 bytes; the remainder is
  `router.scale` and `router.per_expert_scale`.
- The transplant is materially different, not a near-copy: base-vs-QAT router
  projections have mean cosine 0.9862 and mean relative L2 difference 0.1677;
  only 0.98% of projection elements are bit-identical on average. The
  per-expert scales are identical, while router normalization scales have mean
  relative L2 difference 0.0412.

This makes an exact byte-level router transplant possible without downloading
the full parent checkpoint. `fetch-overlay` retrieves only those pinned tensor
ranges and records per-tensor hashes.

## Important packing correction

Gemma 4's native MLX implementation assigns `router.proj` an 8-bit policy. The
calibration engine excludes `router.*` from GPTQ, but its previous custom packing
predicate accidentally overrode the architecture policy and did not match
`router.proj`; the router would therefore have fallen back to ordinary 4-bit RTN.

The campaign now explicitly preserves these module families in BF16:

```text
(^|\.)(gate|router)($|\.)|shared_expert_gate|e_score
```

Norms, scalar gates, biases, and other tensors without a quantizable linear
module remain floating-point automatically. Token embeddings and the tied output
head also remain BF16 unless a campaign explicitly requests otherwise.

## Gemma 4 ablation matrix

All variants must use the same tokenizer, calibration windows, sequence lengths,
seed, GPTQ artifacts, prompts, and evaluator versions.

1. `qat-bf16`: unquantized QAT checkpoint; quality reference for the QAT family.
2. `qat-gptq-router4`: diagnostic reproduction of the accidental 4-bit RTN
   router policy. This is a negative/control variant, not a release candidate.
3. `qat-gptq-router8`: architecture-default 8-bit router control.
4. `qat-gptq-routerbf16`: QAT-derived GPTQ MXFP4 with its own QAT router in BF16.
5. `qat-gptq-base-routerbf16`: variant 4 with the exact parent BF16 router state
   transplanted after packing.
6. `base-bf16`: ordinary parent BF16 checkpoint.
7. `base-gptq-routerbf16`: non-QAT GPTQ control with its own router in BF16.

Variants 4 and 5 isolate the central question. Variants 1, 6, and 7 prevent a
QAT benefit from being misattributed to the transplant.

## Published Gemma 4 26B-A4B Q4 versus BF16 differential

Google does not publish a separate task-score table for the 26B-A4B BF16 and
QAT Q4_0 artifacts. The same capability table is reproduced on both model
cards, so those values cannot be interpreted as a measured zero delta.

Google's technical report does publish the memory differential for the text
model: 52.0 GB BF16 versus 16.2 GB Q4_0, a 35.8 GB or 68.85% reduction. For the
active weights it reports 7.6 GB versus 2.8 GB, a 4.8 GB or 63.16% reduction.

The strongest public matched quality comparison found is Unsloth's logit study,
which uses the unquantized QAT BF16 model as reference:

| 26B-A4B export | Disk | Mean KL to QAT BF16 | 99.9% KL | Token top-1 agreement |
| --- | ---: | ---: | ---: | ---: |
| Unsloth `UD-Q4_K_XL` | 14.25 GB | 0.09788 | 2.7087 | 85.63% |
| Naive Q4_0 conversion | 14.44 GB | 0.36094 | 4.5420 | 70.20% |

The better 4-bit export therefore improves top-1 agreement by 15.43 percentage
points and reduces mean KL by 72.88% relative to the naive Q4_0 conversion. It
still disagrees with the BF16 reference's top token on 14.37% of measured token
positions. This is meaningful distribution-level evidence, but it is not a
downstream benchmark accuracy score and it does not isolate router errors.

### Independent B550 trace

We independently reproduced the strong-Q4 comparison on B550's two RTX 3090 Ti
GPUs with Unsloth llama.cpp `8ce6f6fb2`, a BF16 GGUF converted from the pinned
QAT checkpoint, the pinned `UD-Q4_K_XL` artifact, and the first 32 contiguous
512-token WikiText-2 test contexts. The last 255 predictions in each context
were scored, for 8,160 evaluated token positions.

| Measurement | B550 result | Unsloth published result |
| --- | ---: | ---: |
| Mean KL to QAT BF16 | 0.109056 | 0.09788 |
| 99.9% KL | 2.854649 | 2.7087 |
| Token top-1 agreement | 85.123% | 85.63% |
| Paired Q4/BF16 perplexity ratio | 1.00559 | not reported |

The independent top-1 result is within 0.51 percentage points of the published
number, and the KL values are directionally consistent. The absolute WikiText
perplexities are high and are not presented as model-quality scores; only the
paired same-runtime differential is useful here. llama.cpp stores the reference
trace as clipped 16-bit log probabilities with a `-16` floor, so its reconstructed
base perplexity is biased downward. The harness instead obtains BF16 perplexity
from the unclipped reference pass and uses the compact trace only for KL,
probability-delta, and argmax comparisons.

The machine-readable receipt is
[`receipts/gemma4-b550-q4-logit-trace.json`](../receipts/gemma4-b550-q4-logit-trace.json),
with the raw reference and candidate logs retained under `receipts/logs/`.

This unusually large 26B-A4B gap supports the sensitivity hypothesis: a generic
Q4_0 conversion loses substantially more fidelity on the MoE than the carefully
mixed 4-bit export. Our BF16-router variants should therefore be compared with
both QAT BF16 and the strongest available 4-bit export, not only with naive Q4_0.

### Long-context protocol

The 512-token trace is only the short-context baseline. Gemma 4 advertises
262,144 tokens, while 25 layers use a 1,024-token sliding-attention window and
five layers use full attention. We test the architectural boundary and the
25%, 50%, 75%, and 100% regions with the same prefix at 1,024, 8,192, 65,536,
131,072, 196,608, and 262,144 tokens.

Two claims are kept separate:

- **Advertised context** means that the exact 262,144-token prompt loads,
  completes prefill, and produces finite tail logits without an OOM.
- **Effective context** means that retrieval and answer quality remain useful
  as evidence moves deeper into the window. Passkey, multi-needle, and
  long-document QA are evaluated independently from the natural-text logit
  trace.

For distribution fidelity, every checkpoint scores the final 256 predictions
against QAT BF16. A patched `llama-perplexity` keeps the complete prefix and
requests full-vocabulary logits only for that tail; otherwise the stock tool
would allocate and serialize half of a 256K context. CUDA provides the rapid
Q4/hybrid screen, llama.cpp provides the partially offloaded BF16 authority,
and MLX validates the final GPTQ MXFP4 artifacts.

Long-context scoring uses two separate distributions rather than presenting a
generic web corpus as a proxy for coding-agent workloads:

- The **general-text control** is a deterministic transformed subset of
  `openbmb/Ultra-FineWeb-L1` `CC-MAIN-2025-51`, pinned at
  `10b9ba18466215c0ba495299dfffd798af1027f2`.
- The **agentic-coding primary track** is assembled from public source, tests,
  and documentation at six Apache-2.0 repository commits indexed by
  `OpenMOSS-Team/SWE-bench-Science@7e7084280fdf5d21811ff04c7acb0b738984c705`.
  The index supplies only repository URLs, base commits, and license gates; no
  benchmark instructions, verifier material, trajectories, or answer patches
  enter this corpus.

The coding stream encodes a selected file tree followed by complete files and
targets a 55/25/20 character mix of source, tests, and documentation for each
repository. This approximates the context an agent receives while navigating a
real workspace without pretending that incidental technical keywords make an
ordinary web page an agentic-coding example. Both streams reject any file or
document sharing a normalized 20-word shingle with the published calibration
text. They are evaluation-only holdouts and must never enter GPTQ calibration.
Metrics are reported separately for the two distributions; they are not pooled
into one headline number.

The first published coding holdout contains 788 files and 2,429,702 Gemma 4
tokens without BOS. Its combined UTF-8 stream SHA-256 is
`e615a4149b1b16e5327d8cf79cbb70b07a4007c29cc2302ffc13291cc36455cd`;
the construction receipt records zero shared 20-word shingles across the whole
stream. It is pinned at
`RESMP-DEV/ptq-calibration-corpus@6ef2bbf949b94192ee8bb78d45a2fc840a10cff5`
under the `agentic_coding_holdout` config and `heldout` split. The local evidence
copy is [`receipts/agentic-coding-holdout.json`](../receipts/agentic-coding-holdout.json).
The complete machine-readable protocol is
[`experiments/gemma4-long-context-matrix.json`](../experiments/gemma4-long-context-matrix.json).

## Measurements

Quality measurements:

- held-out token negative log-likelihood and perplexity with one tokenizer;
- mean `KL(P_reference || P_variant)` and top-1 token agreement against both
  `qat-bf16` and `base-bf16`;
- deterministic generation token hashes on a fixed prompt set;
- task suites relevant to the model card, reported with confidence intervals.

Routing measurements, captured per layer and token before expert execution:

- top-8 expert-set overlap and exact-set agreement;
- top-1 expert agreement;
- Jensen-Shannon divergence of normalized top-8 routing weights;
- routing-margin change between the eighth and ninth experts;
- expert-load histogram divergence and dead/overloaded expert counts.

Operational measurements:

- serialized bytes and effective bits per weight;
- peak resident memory;
- prefill and decode throughput reported separately;
- first-token latency and output-token latency.

The main success criterion is a statistically reliable quality improvement from
variant 4 or 5 over variant 3 without a material throughput or memory regression.
A routing metric moving toward a BF16 reference is supporting mechanism evidence,
not sufficient evidence by itself.

## Research context

The specific QAT-plus-parent-router combination was not found in the literature
search, but its two constituent risks are well documented:

- **EAC-MoE** (ACL 2025) demonstrates that low-bit quantization causes
  expert-selection shift. It keeps routers at original precision and calibrates
  them layer by layer with a TopK-MSE objective.
- **EAQuant** aligns full-precision and quantized routing distributions using
  reconstruction and KL objectives, again treating routing consistency as a
  first-class quantization target.
- **RouteQuant / Router Choice Matters** reports that a full-precision frozen
  router is not sufficient because quantized upstream layers change its input
  distribution. It aligns top-k ranks and routing margins after quantization.
- **Component Transfer Can Exceed Full Model Performance** (GEM 2026) directly
  swaps routers, attention modules, and experts between two post-trained OLMoE
  checkpoints. Router-only transfer provides little benefit or harms quality;
  the authors attribute the instability to router-expert co-adaptation.
- **HARC** studies routing breakdown after MoE model merging and uses a
  Hessian-aware, training-free router calibration to realign the merged router.

The evidence therefore favors `qat-gptq-routerbf16` as the primary candidate.
`qat-gptq-base-routerbf16` remains a useful high-risk ablation. If it misroutes,
the next principled variant is not blind interpolation: calibrate the parent or
QAT router against the quantized hidden-state distribution using top-k/rank and
margin preservation.

## Non-QAT MoE replication

`campaigns/qwen3.6-35b-a3b-mxfp4.json` pins
`Qwen/Qwen3.6-35B-A3B@995ad96eacd98c81ed38be0c5b274b04031597b0`.
It keeps the 40 primary routers (`mlp.gate`) and shared-expert gates in BF16 while
calibrating the large expert projections to MXFP4. Because no QAT checkpoint is
involved, this campaign tests the general mixed-precision surgery hypothesis and
cannot establish a QAT-specific gain.

## Second QAT MoE: JoyAI-LLM Flash

`jdopensource/JoyAI-LLM-Flash` is a second provenance-backed QAT MoE, and its
official INT4 artifact is direct prior art for the least speculative part of this
experiment. The v2 technical report says that training inserted fake INT4
quantize/dequantize operators, used straight-through estimators, and retained
high-precision master weights. It also says this QAT makes simple round-to-nearest
low-bit export accurate.

Live artifact inspection on 2026-08-22 established:

- BF16 QAT master: `jdopensource/JoyAI-LLM-Flash` at
  `8d31d53f43de15f3bd5e34fea03bd8e186611fc8`.
- Official packed model: `jdopensource/JoyAI-LLM-Flash-INT4` at
  `8778cd62685ffd7f1856d1a3de8eaca26aa6c3e5`.
- The model has 48B total parameters, 3B active, 40 layers, 256 routed experts,
  top-8 routing, and one shared expert.
- The INT4 artifact contains exactly 29,952 packed expert projections:
  39 MoE layers x 256 experts x 3 projections. They use symmetric group-size-64
  INT4 with FP32 scales.
- Its declared exclusions keep self-attention, shared experts, the dense layer,
  and `lm_head` in floating point. The router is not a `Linear` target and remains
  BF16 as well.
- All 78 router tensors (`gate.weight` plus `e_score_correction_bias` in each of
  39 MoE layers) are byte-identical between the pinned BF16 and INT4 releases.
  This covers 40,914,432 bytes, not a name-only or shape-only comparison.
- The complete BF16 checkpoint is 98,575,966,632 bytes; the official INT4
  checkpoint is 30,862,354,128 bytes across the same 42 safetensors files.

JoyAI therefore demonstrates **QAT-trained expert weights plus the same QAT
router retained in BF16**. It does not demonstrate the riskier Gemma ablation of
replacing a QAT router with a router from a separately trained non-QAT parent.
That cross-checkpoint transplant remains the novel question.

The earlier `zhuyksir/qwen3_30b_a3b_nvfp4_qat` lead remains unverified because
it has no model card or training receipt. NVIDIA Model Optimizer publishes the
underlying QAT/QAD workflow, but that alone does not establish the provenance of
an individual checkpoint.

## JoyAI replication matrix

1. `qat-bf16`: the official high-precision QAT master.
2. `qat-official-int4`: the official group-size-64 RTN artifact with routed
   experts packed and the router, attention, shared experts, dense layer, and
   output head retained in floating point.
3. `qat-gptq-routerbf16`: GPTQ/MXFP4 routed experts sourced from the QAT master,
   with the byte-identical QAT router retained in BF16.
4. `qat-gptq-router4`: negative control that also quantizes the router.

Variants 2 and 3 isolate the value of activation-aware GPTQ versus the official
QAT-plus-RTN export while preserving the same sensitive-component boundary.

## Commands

Fetch the 21.8 MB parent-router overlay:

```bash
CAMPAIGN=campaigns/gemma4-26b-a4b-qat-mxfp4.json
uv run mlx-gptq-calibration --campaign "$CAMPAIGN" fetch-overlay
```

Pack the QAT-router BF16 baseline:

```bash
uv run mlx-gptq-calibration --campaign "$CAMPAIGN" pack \
  --model ~/models/google-gemma-4-26B-A4B-it-qat-q4_0-unquantized-f1e06dc \
  --output ~/models/Gemma-4-26B-A4B-QAT-GPTQ-MXFP4-router-bf16
```

Pack and transplant the parent BF16 router:

```bash
uv run mlx-gptq-calibration --campaign "$CAMPAIGN" pack \
  --model ~/models/google-gemma-4-26B-A4B-it-qat-q4_0-unquantized-f1e06dc \
  --output ~/models/Gemma-4-26B-A4B-QAT-GPTQ-MXFP4-base-router-bf16 \
  --overlay runs/gemma4-26b-a4b-qat-mxfp4/overlays/gemma4-base-bf16-router.safetensors
```

Each operation emits a JSON receipt under the ignored campaign run directory.

## References

- Gemma 4 technical report: <https://arxiv.org/abs/2607.02770>
- Unsloth Gemma 4 QAT analysis:
  <https://unsloth.ai/docs/models/gemma-4/qat#qat-analysis>
- EAC-MoE: <https://aclanthology.org/2025.acl-long.633/>
- EAQuant: <https://arxiv.org/abs/2506.13329>
- Router Choice Matters / RouteQuant:
  <https://openreview.net/forum?id=kPgLp47bJf>
- Component Transfer Can Exceed Full Model Performance:
  <https://aclanthology.org/2026.gem-main.7/>
- HARC: <https://arxiv.org/abs/2606.03391>
- NVIDIA Model Optimizer QAT/QAD workflow:
  <https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/llm_qat>
- JoyAI-LLM Flash technical report v2, section 4.1:
  <https://arxiv.org/html/2604.03044v2#S4.SS1>
- JoyAI-LLM Flash BF16 QAT master:
  <https://huggingface.co/jdopensource/JoyAI-LLM-Flash>
- JoyAI-LLM Flash official INT4 artifact:
  <https://huggingface.co/jdopensource/JoyAI-LLM-Flash-INT4>
