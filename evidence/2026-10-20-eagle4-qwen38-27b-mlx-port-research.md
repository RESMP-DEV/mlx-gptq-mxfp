# EAGLE-4 Qwen3.8 27B MLX port research

## Claim

The official EAGLE-4 Qwen3.8-27B drafter is a one-layer BF16 Llama-style
model with target-feature projection, a trained rank-128 candidate selector,
and a full 248,320-row output head. A correctness-first MLX port is feasible
through the DFlash MLX runtime, but it requires exact target-layer capture,
draft-only cache handling, and tree-safe Qwen GDN state branching. No B300
throughput result transfers to Apple Silicon.

## Provenance

| Source | Revision or checksum | Local path |
|---|---|---|
| Official inference code | `f6149b060b21f0912282d4410a01f2335f913372` | `/Users/kearm/research/ml/eagle-hq/EAGLE-4` |
| Draft checkpoint | revision `f58a5985223040552ad404fba98f0dd27f5f676a` | `/Users/kearm/models/EAGLE-Corp-Qwen3.8-27B-EAGLE4` |
| Draft weights | SHA-256 `c90c82867b5d6878daca0ada2fdb0dadfc56171770f93c8ee3bd9f796f1d5df3`, 6,178,152,384 bytes | `model.safetensors` |
| Paper PDF | SHA-256 `9b3f4833f7dcb49643edac3756b9aa93eaf740c4ba88e574adbf21f6bc126901` | `build/reports/eagle4/paper/EAGLE4_Paper.pdf` |
| Named target | revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | `/Users/kearm/models/Qwen-Qwen3.8-27B-1d4bf0f` |

The official clone is clean, its fetch URL is the upstream repository, and its
push URL is `/Users/kearm/.config/no-upstream-push`. The local draft artifact
has no `.aria2` control file and its full SHA-256 was recomputed during this
consolidation.

Four bounded sub-agent audits completed with process exit code zero. Their
full reports are ignored runtime evidence under
`build/reports/eagle4/workers/`:

- `paper-audit.txt`
- `inference-code-audit.txt`
- `artifact-audit.txt`
- `mlx-implementation-map.txt`

The corresponding worker status files record 1,012.011 s, 1,044.418 s,
926.136 s, and 726.838 s elapsed respectively. Each completed report was
inspected by the parent before this record was written.

## Checkpoint contract

The draft config declares `LlamaForCausalLMEagle4`, hidden size 5120, target
hidden size 5120, one layer, 40 attention heads, 40 KV heads, head dimension
128, MLP width 16384, vocabulary 248320, RoPE base 10,000,000, and sliding
window 4096. The three target auxiliary layers are `3`, `31`, and `59`. The
checkpoint has 19 BF16 tensors:

```text
efficient_lm_head.weight                         [248320, 128]
efficient_lm_head_down_proj.weight               [128, 5120]
embed_tokens.weight                              [248320, 5120]
fc.weight                                        [5120, 15360]
fc_norm.0.weight                                 [5120]
fc_norm.1.weight                                 [5120]
fc_norm.2.weight                                 [5120]
layers.0.hidden_norm.weight                      [5120]
layers.0.input_layernorm.weight                  [5120]
layers.0.mlp.down_proj.weight                    [5120, 16384]
layers.0.mlp.gate_proj.weight                    [16384, 5120]
layers.0.mlp.up_proj.weight                      [16384, 5120]
layers.0.post_attention_layernorm.weight         [5120]
layers.0.self_attn.k_proj.weight                 [5120, 10240]
layers.0.self_attn.o_proj.weight                 [5120, 5120]
layers.0.self_attn.q_proj.weight                 [5120, 10240]
layers.0.self_attn.v_proj.weight                 [5120, 10240]
lm_head.weight                                   [248320, 5120]
norm.weight                                      [5120]
```

The 19-tensor list was read directly from the safetensors header during this
consolidation. There is no `fc_norm.3`; strict loaders must reject it.

The draft checkpoint has no tokenizer. Runtime tokenization, chat template,
and stop behavior must come from the pinned Qwen target. In particular, the
target generation configuration includes EOS IDs 248046 and 248044, while the
draft config alone lists only 248044.

## Drafter input and candidate head

At seed/prefill time:

1. Capture target hidden states after decoder layers 3, 31, and 59.
2. RMS-normalize each state independently.
3. Concatenate them to width 15360.
4. Apply `fc.weight` and its norm contract to width 5120.
5. Independently normalize the current token embedding and projected target
   feature with `input_layernorm` and `hidden_norm`.
6. Concatenate them to width 10240 and run the one sliding-window Llama layer.
7. The layer's final normalized output seeds the next draft depth because the
   config enables `norm_output`.

At deeper depths the drafter reuses its own prior normalized hidden state; it
does not receive fresh target auxiliary states for every speculative depth.

The low-rank head is not a substitute output head:

```text
approximate_scores = hidden @ efficient_lm_head_down_proj.T @ efficient_lm_head.T
candidate_ids      = top_k(approximate_scores, 128)
exact_scores       = lm_head[candidate_ids] @ hidden
```

The rank-128 factors select candidates. Gathered rows from the full draft LM
head produce the exact scores used for probabilities and tree policy. The
factors are already trained; inference must not recompute them by SVD.

## Official tree and verification behavior

The official drafter expands best frontier nodes by parent score multiplied by
proposal probability, verifies nodes parent-before-child, and compacts only the
accepted path. SGLang adapts depth and width from measured target and draft
costs. vLLM's fixed schedules and B300 cost tables are hardware-specific and
must not be copied to MLX.

For greedy decoding, the target computes a token at each proposed node. The
verifier walks from the root while each child token equals the target token,
then commits the accepted path plus one target bonus token. Sampling requires
a single clearly specified policy: either standard residual sampling or the
official fixed-tree direct target-sampling walk. Randomness must be stable by
request seed and row position.

The Qwen target is hybrid. A generic KV tree mask is insufficient: GDN state
must branch per proposed node, tree-safe causal convolution and recurrence must
run, and rejected branches must leave no target state residue.

## MLX implementation decision

The implementation proceeds in correctness-first phases:

1. Parse and strictly validate the EAGLE-4 schema, then prove a finite tiny
   synthetic draft forward and exact candidate-head agreement against a
   full-head reference.
2. Capture only target layers 3, 31, and 59 during prompt prefill and accepted
   target extension. Do not retain all 60 hidden states for long contexts.
3. Use top-k 1 and the full head for the first chain-only generation path.
   Maintain a separate draft KV cache and compare exact greedy IDs against
   native no-speculation generation.
4. Add rank-128 retrieval and benchmark it against the full head only after
   exact chain parity.
5. Add draft tree branching and Qwen GDN target-state branching, then compact
   accepted state.
6. Profile Metal and build M4-specific cost tables before considering dynamic
   schedules or serving integration.

The existing DFlash tree verification and recurrent rollback machinery is the
best local starting point, but DFlash checkpoint assumptions cannot be reused
without an explicit EAGLE branch. In particular, the existing generic loader
currently rejects draft-owned `lm_head` weights; EAGLE-4 must permit only its
validated full-head and efficient-head contract.

## Compatibility gates

- Loader/schema gate: exact names, shapes, dtypes, finite values, and strict
  mapping with no inferred tensor names.
- Draft-forward gate: finite normalized states and reference-parity tiny
  tensors.
- Candidate-head gate: selected IDs and exact scores agree with full-head
  reference for deterministic inputs.
- Target-capture gate: selected layer outputs equal a manual reference loop;
  no off-by-one is permitted.
- Chain greedy gate: exact token IDs and target-position logprobs equal native
  target-only generation on short prompts.
- Cache gate: accepted-path cache equals a fresh cache replaying only accepted
  tokens; rejected branches leave no residue.
- Quantization gate: compare the official drafter against the pinned BF16
  target first, then the owned calibrated MXFP4 target. Quantization may alter
  hidden states, near-ties, acceptance, and output distribution.
- Long-context gate: repeat exactness at selected long-context depths without
  all-layer hidden-state retention.
- Performance receipt: measured M4 Max command, source revision, hardware,
  cache state, acceptance lengths, and non-claims.

## License and publication boundary

EAGLE-4 is source-available under the Eagle-4 license, not OSI open source.
The license's software definition includes architecture, weights, parameters,
and configuration, and its terms include trademark, attribution, notice, and
competing-offering restrictions. Internal MLX research is in scope. Do not
vendor or redistribute official source or weights into a public repository
without a separate license decision. The Qwen target's Apache-2.0 license does
not relicense the drafter.

## Implementation bring-up

The first correctness-only implementation landed through worker commit
`5dcc124` and merge commit `d556853`:

- `runtime/dflash-mlx/dflash_mlx/draft/eagle4.py`
- `runtime/dflash-mlx/dflash_mlx/draft/checkpoint.py`
- `runtime/dflash-mlx/dflash_mlx/runtime/loading.py`
- `runtime/dflash-mlx/dflash_mlx/runtime/bundle.py`
- `runtime/dflash-mlx/tests/test_eagle4_draft.py`
- `runtime/dflash-mlx/tests/test_eagle4_target_capture.py`

The parser accepts only the dense, rotary, sliding-window, full-vocabulary,
rank-128 low-rank profile represented by the pinned Qwen checkpoint. It checks
config/weight dtype consistency and strict names, shapes, and dtypes. The model
implements independent aux-state norms, target-feature `fc`, independent token
and hidden-feature norms, the fused 2H QKV layer, per-row RoPE, sliding-window
attention, target-embedding sharing, full logits, rank-128 candidate selection,
and exact full-head selected-row rescoring.

The implementation is a clean re-expression of the pinned EAGLE-4 formulas, not
a vendored copy. Its source provenance and
`LicenseRef-Eagle-4-Source-Available` boundary are stated in the module header
and this receipt. It remains an internal RESMP research port until a separate
license decision permits redistribution.

An independent read-only review confirmed the formula, shape, tensor-name,
aux-layer, target-embedding, and candidate-head mapping. Its blocker was that
ordinary runtime dispatch would hand the incompatible model to the existing
DFlash backend. The repair keeps `load_draft_bundle` useful for direct schema
and numerical bring-up, while `load_runtime_bundle` now rejects EAGLE-4 with a
clear error until a dedicated engine backend exists. The review also led to
dtype/config cross-checks, stricter missing-config diagnostics, and this
provenance record.

Validation at merged head `d556853`:

```bash
PYTHONPATH=/Users/kearm/mlx-stack/runtime/dflash-mlx:/Users/kearm/mlx-stack \
  uv run --no-project \
  --with mlx==0.32.1 --with mlx-lm==0.31.3 --with pytest \
  pytest tests/test_eagle4_draft.py tests/test_eagle4_target_capture.py \
  tests/test_target_qwen_tree.py tests/test_draft_config_schema.py \
  tests/test_runtime_bundle.py tests/test_gemma4_draft.py -q
```

Result: 69 passed.

The real 6.2 GB checkpoint was then lazy-loaded from the merged head. It
exposed exactly 19 BF16 tensors with configured dtype BF16. A one-token
real-weight Metal smoke produced finite hidden state `[1,1,5120]`, finite full
logits `[1,1,248320]`, 128 selected candidates, and selected-logit rescoring
against the full-head reference with maximum absolute difference `0`.

## Non-claims

- The real-weight smoke used a random target feature vector; it does not prove
  27B target-hook numerical parity.
- No dedicated EAGLE-4 draft backend, draft KV cache, tree state branching, or
  end-to-end speculative generation is implemented yet.
- `load_runtime_bundle` intentionally rejects EAGLE-4 before the existing
  DFlash backend receives it.
- The B300 speeds in the official model card are not Apple Silicon results.
- Compatibility with the owned calibrated MXFP4 target is not yet measured.
- No official upstream repository, issue tracker, or model repository was
  modified.
- Hugging Face Papers exact-title search returned unrelated papers, including
  P-EAGLE, and no valid EAGLE-4 paper ID was found; cite the pinned PDF and
  official repository instead of inventing one.
