---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
library_name: vllm
pipeline_tag: text-generation
tags:
- mxfp4
- gptq
- compressed-tensors
- text-generation
- private-research
---

# Qwen3.8-27B MXFP4A16 GPTQ, A100 build

This private research artifact is the measured A100 build of a text-only
Qwen3.8-27B compressed-tensors model. It is intended for the local fleet and is
not a universal leaderboard result.

## Pinned artifact

- Base model: `Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- Source content manifest SHA-256:
  `968425eaa11944b2b8d285acb995c034c52ccdc057a661991abfb955aa377182`
- Architecture: `Qwen3_5ForCausalLM`, text-only graph
- Quantization: calibrated MXFP4A16 GPTQ, E2M1 weights, E8M0 group scales,
  group size 32; BF16 activations and LM head
- Quantization hardware: one NVIDIA A100-SXM4-40GB
- Quantized text linear tensors: 496
- `model.safetensors` SHA-256:
  `1b6cf393189f59763dbd330cf85fefd2508cfca63db7c13aaea8d53da71e1dd7`
- `config.json` SHA-256:
  `873538762602808f3b895701b479f2f085d917e390d85a9788aa26b796660055`

The tokenizer files were restored byte-for-byte from the verified source
checkpoint. The vision tower and MTP drafter are intentionally absent.

This A100 build is not bit-identical to the earlier B550 build. Select the
build whose host-specific receipt matches the intended execution and evaluation
protocol; do not interchange them by name alone.

## Calibration and holdout

- Calibration corpus: `RESMP-DEV/ptq-calibration-corpus@ffa2e74b9361d573167373140f10dfd30c69a327`
- Calibration matrix: `data/calibration_256x2048.npy`, shape `256 x 2048`,
  SHA-256 `d7dd281b1f1756a6492361e1dfb0c65fed259f8a3d65eabbcb51809b50bd35d7`
- Holdout: `agentic-coding-long-context.txt` at corpus revision
  `6ef2bbf949b94192ee8bb78d45a2fc840a10cff5`, SHA-256
  `e615a4149b1b16e5327d8cf79cbb70b07a4007c29cc2302ffc13291cc36455cd`
- Holdout overlap: zero shared normalized 20-word shingles with calibration text

## Measured A100 result

Both arms used the same source revision, holdout bytes, 34,303 evaluated
tokens, `NVIDIA A100-SXM4-80GB`, vLLM `0.32.0`, torch `2.13.0+cu130`,
Transformers `5.19.0`, causal-conv1d `1.7.0`, and flash-linear-attention
`0.5.2`.

| Arm | Holdout perplexity | Mean NLL | Single-request decode |
| --- | ---: | ---: | ---: |
| BF16 reference | 2.4590172978 | 0.8997617977 | 28.78 tok/s |
| MXFP4A16 GPTQ | 2.4992624858 | 0.9159956827 | 61.51 tok/s |

The MXFP4 build has a 1.6366% perplexity increase and 0.016234 nats/token NLL
penalty while decoding 2.1372x faster. It is recommended when single-request
A100 decode speed matters more than that measured code-corpus quality penalty.

On Ampere this format uses vLLM's Marlin weight-only W4A16 path. It is not
native W4A4 FP4 tensor-core execution.

## Non-claims

- Held-out code perplexity is not task, instruction-following, or agentic
  benchmark accuracy.
- Single-request decode throughput is not batched serving capacity.
- The result qualifies this artifact on one A100-SXM4-80GB and vLLM 0.32.0
  only; it does not qualify B550, PCIe A100s, or another engine.
- No universal leaderboard or public-score claim is made.

Exact source, calibration, packaging, perplexity, throughput, and comparison
receipts are retained under `receipts/qwen38-precision-ladder/`.
