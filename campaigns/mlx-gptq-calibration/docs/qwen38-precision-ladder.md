# Qwen3.8-27B precision ladder (Modal A100)

## Purpose

Decide which precision level of Qwen3.8-27B this fleet should actually serve,
and record the evidence for that recommendation. The earlier B550 work proved
the MXFP4A16 artifact *runs* on CUDA; it explicitly did not measure quality.
This ladder supplies the missing matched control.

## Design

One source, one corpus, one harness, varying only precision.

| Arm | Weights | Activations | Role |
| --- | --- | --- | --- |
| `bf16` | BF16 | BF16 | Matched reference, upper bound on quality |
| `mxfp4-gptq` | MXFP4 (E2M1, group 32, E8M0 scales) | BF16 | Candidate serving precision |

Both arms are the *same* upstream model, `Qwen/Qwen3.8-27B` at revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, flattened to the same
`Qwen3_5ForCausalLM` text-only graph with the same tokenizer bytes, and served
by the same vLLM build with identical settings.

### Why BF16 activations in both arms

A100 is SM80 (Ampere). Native FP8 *activation* compute needs SM89+; on Ampere
vLLM would run lower-precision activations through dequantizing fallback paths.
Keeping activations BF16 in both defined arms means weight representation is
the precision variable: BF16 weights provide the matched reference, while
MXFP4 packed weights execute through the Marlin W4A16 path. This avoids
attributing a framework fallback to model precision.

## Held-out corpus

The evaluation corpus is the `agentic-coding` long-context holdout from
`RESMP-DEV/ptq-calibration-corpus` at revision
`6ef2bbf949b94192ee8bb78d45a2fc840a10cff5`:

- path `data/heldout/agentic-coding/agentic-coding-long-context.txt`
- 6,504,524 bytes, SHA-256
  `e615a4149b1b16e5327d8cf79cbb70b07a4007c29cc2302ffc13291cc36455cd`
- 6 commit-pinned Apache-2.0 repositories (diffrax, xarray, cvxpy, monai,
  pyscf, quimb), task prompts, trajectories, and verifier material excluded.

It is the correct control because its own construction receipt proves zero
overlap with the calibration text: `combined_stream_shared_twenty_word_shingles`
is 0 across 773,048 shingles, and the referenced calibration file hash
`244a35f5289efad6397342a6782350e464a5f180d8406d2969200722d7da5fb9` is
byte-identical to `calibration.txt` at the Qwen calibration revision
`ffa2e74b9361d573167373140f10dfd30c69a327`. The corpus used to calibrate the
MXFP4 artifact and the corpus used to score it are provably disjoint.

## Provenance chain

1. `Qwen/Qwen3.8-27B@1d4bf0f` is downloaded to the Modal `qwen38-models`
   volume and then verified file-by-file against the Hugging Face published
   pins for that revision: every size checked, and every LFS-tracked file
   SHA-256 checked against its published oid. The resulting
   `pipeline_content_manifest_sha256` is
   `968425eaa11944b2b8d285acb995c034c52ccdc057a661991abfb955aa377182`,
   identical to the B550 run's manifest for the same revision.
2. The calibration matrix is pinned by SHA-256
   `d7dd281b1f1756a6492361e1dfb0c65fed259f8a3d65eabbcb51809b50bd35d7`,
   identical to the B550 campaign pin.
3. The MXFP4 arm is produced by the already-reviewed repository scripts
   (`scripts/quantize_mxfp4_vllm.py`, `scripts/convert_mxfp4_text_only.py`)
   rather than a reimplementation, so the same recipe, guards, and receipts
   apply.
   The historical artifact was solved on one `NVIDIA A100-SXM4-40GB`; all
   quality and throughput measurements below were run on one
   `NVIDIA A100-SXM4-80GB`. These are distinct provenance and measurement
   boundaries, and GPTQ is not assumed bitwise reproducible across GPU models.
4. The A100 campaign records, per runtime package, the B550 pin and the
   A100-measured value in `runtime_provenance`, so a reader can tell a
   reviewed pin from a live measurement. The only differences are the torch
   CUDA build tag (`2.14.0` vs `2.14.0+cu130`) and the llm-compressor
   installation `RECORD` hash, which differs because it is a different build
   of the same pinned version.

## Measurements

For each arm, on one NVIDIA A100-SXM4-80GB (SM80), driver 610.57.04,
CUDA 13.3, vLLM 0.32.0, torch 2.13.0+cu130, transformers 5.19.0,
causal-conv1d 1.7.0, and flash-linear-attention 0.5.2:

- **Held-out perplexity.** Sliding-window negative log-likelihood over the
  holdout at 2,048-token windows with 512-token stride and the 64-window cap.
  The harness uses explicit causal-logit cross entropy and scores only the
  final positions not already carried by the prior window; the initial corpus
  token has no predecessor and is not counted. Reported as perplexity plus
  evaluated-label count, so a coverage difference cannot masquerade as a
  quality difference.
- **Throughput.** Deterministic decode under vLLM 0.32.0 with a fixed
  prompt, fixed seed, fixed `max_model_len`, fixed `max_num_batched_tokens`,
  and a warmup pass before the timed pass. Reported as generated tokens per
  second.

Every receipt binds the model shard hashes, the config hash, the tokenizer
hashes, the holdout hash, the runtime versions, and the live GPU identity.

## Measured result

| Arm | Held-out perplexity | Mean NLL | Single-request decode |
| --- | ---: | ---: | ---: |
| BF16 reference | 2.4590172978 | 0.8997617977 | 28.78 tok/s |
| MXFP4A16 GPTQ | 2.4992624858 | 0.9159956827 | 61.51 tok/s |

The MXFP4 arm has a 1.016366 perplexity ratio (a 1.6366% increase) and a
0.016234 nats/token penalty while decoding 2.1372x faster than BF16. Both arms
used the same A100-SXM4-80GB device name, identical holdout bytes, identical
34,303-causal-label coverage, and identical vLLM settings.

Current recommendation: serve the A100-built MXFP4A16 GPTQ text-only artifact
when single-request A100 decode speed matters more than a 1.64% code-corpus
perplexity penalty. Its text shard SHA-256 is
`1b6cf393189f59763dbd330cf85fefd2508cfca63db7c13aaea8d53da71e1dd7`, and its
config SHA-256 is
`873538762602808f3b895701b479f2f085d917e390d85a9788aa26b796660055`.
The consolidated evidence record is
[`receipts/qwen38-precision-ladder/result.json`](../receipts/qwen38-precision-ladder/result.json).

The recommended artifact is published privately as
`RESMP-DEV/Qwen3.8-27B-MXFP4A16-GPTQ-A100`, pinned revision
`95709c64d92b8bc9747388638d6da00db4c0adff` and tag
`a100-vllm-0.32.0-v3`. The publisher verified all ten packaged file hashes,
the exact 20-file remote set, private visibility, and the remote LFS SHA-256
before recording [`hf-publication.json`](../receipts/qwen38-precision-ladder/hf-publication.json).

This recommendation is intentionally local to the measured A100/vLLM protocol.
It does not qualify task accuracy, batched serving capacity, B550, or another
engine. The A100 and B550 GPTQ artifacts have different shard hashes because
GPTQ is not bitwise reproducible across devices; consume the artifact whose
receipt matches the execution host and evaluation objective.

## Non-claims

- Perplexity on a code corpus is not a task or agentic score.
- Single-request decode is not batched serving capacity.
- One GPU, one run per arm: no cross-host or cross-protocol pooling, and no
  claim about other GPUs or other batch shapes.
- The MXFP4 arm on Ampere runs the Marlin weight-only W4A16 path; it is not
  native W4A4 tensor-core MXFP4.
- A ladder result on A100 does not by itself qualify B550 (SM86) or any
  other host.

## Status

The source verification, BF16 flattening, A100 MXFP4 quantization/packaging,
matched perplexity, and matched throughput stages are implemented and
exercised. The recommended artifact is also published privately and pinned.
The durable receipts live in
`receipts/qwen38-precision-ladder/`; Modal-volume run copies are local build
state and are not tracked.
