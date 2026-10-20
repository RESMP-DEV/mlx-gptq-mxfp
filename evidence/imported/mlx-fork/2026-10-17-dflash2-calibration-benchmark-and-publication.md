# Qwen3.8-27B DFlash2 calibration benchmark and publication

Date: 2026-10-17

## Scope

This records the first local DFlash2 diagnostic against the calibrated private
Qwen3.8-27B GPTQ MXFP4 target, the uncalibrated community 4-bit control, and
the pinned BF16 source. It also records Hugging Face publication state. It
does not claim representative model quality, multi-user serving throughput,
vision support, or any Strix execution.

## Inputs

- Calibrated target: `/Users/kearm/models/Qwen3.8-27B-MLX-GPTQ-MXFP4`.
- Community 4-bit control:
  `/Users/kearm/models/mlx-community-Qwen3.8-27B-4bit`, pinned Hub revision
  `10c35caafbb80f7dc6a7a432cdd11af10a6d4818`; all three model shards and the
  tokenizer SHA-256 checks passed.
- BF16 source: `/Users/kearm/models/Qwen-Qwen3.8-27B-1d4bf0f`, pinned revision
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- DFlash2 draft:
  `/Users/kearm/models/z-lab-Qwen3.8-27B-DFlash2`, Hub revision
  `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`. Its 3,848,817,896-byte
  `model.safetensors` has SHA-256
  `67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c`.
- Runtime: `bstnxbt/dflash-mlx` commit
  `60803233af4589e18588b9bacbb03880801c828a`, MLX 0.32.4, MLX-LM 0.32.2,
  Python 3.12.15, Apple M4 Max with 128 GB unified memory.

## Environment

`dflash doctor` reached the model checks with Python, MLX, MLX-LM, Metal,
scripts, runtime config, chip profile, L2 cache, and verify environment all
passing. It warned that NAX matrix kernels are unavailable before M5 and exit
code 2 therefore represented the expected M4 fallback. No Strix process was
run: `ssh strix` reported the host down and one configured WoL packet did not
restore it within 60 seconds.

## Benchmarks

The DFlash CLI ran the same 144-token chat-template prompt for 256 generated
tokens, three repeats, block size 16, adaptive verification, explicit affine
W4 draft weights, and EOS suppressed. Explicit `--draft-quant w4` was required
because local target paths do not resolve the runtime registry's implicit W4
default; without it, DFlash2 loaded BF16. Machine receipts:

- `build/reports/dflash/calibrated-mxfp4-256x3/`;
- `build/reports/dflash/community-4bit-256x3/`;
- `build/reports/dflash/calibrated-mxfp4-w4-dflash-256x3/`;
- `build/reports/dflash/community-4bit-w4-dflash-256x3/`.

| Target | Baseline median | DFlash2 median | Speedup | Acceptance | DFlash peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| Calibrated MXFP4 | 26.385 tok/s | 53.156 tok/s | 2.029× | 72.266% | 19.688 GB |
| Community 4-bit | 28.048 tok/s | 69.410 tok/s | 2.475× | 72.266% | 16.775 GB |

With DFlash2 left BF16 on the calibrated target, median throughput was 49.580
tok/s and peak memory was 22.293 GB. Explicit W4 improved the diagnostic by
about 3.576 tok/s while reducing peak memory by about 2.606 GB.

Separate internal parity runs recorded all 256 baseline and DFlash2 token IDs
for each explicit-W4 target and confirmed exact equality in both cases:

- calibrated MXFP4: 26.313 -> 51.996 tok/s, acceptance 72.266%;
- community 4-bit: 27.971 -> 62.850 tok/s, acceptance 72.266%.

The same 256-token generation was then run from each target without DFlash and
compared with the BF16 source token stream:

| Target | BF16 token agreement | First divergence | Baseline tok/s |
| --- | ---: | ---: | ---: |
| Calibrated MXFP4 | 113/256 = 44.141% | 97 | 27.424 |
| Community 4-bit | 5/256 = 1.953% | 3 | 28.658 |

Receipt: `build/reports/dflash/calibration-target-token-parity.json`. This is
one prompt and exact-token agreement is not perplexity or task quality.

## Publication

The polished target card was uploaded and made public at
`RESMP-DEV/Qwen3.8-27B-MLX-GPTQ-MXFP4`, revision
`a462c37ed4ff7fe054b3ed0a7424ec5ba4d104ad`.

The polished FlashNext card, Qwen Community 1.0 license, 34 safetensors shards,
index, tokenizer, and manifests were uploaded and made public at
`RESMP-DEV/Qwen3.8-Flash-Next-GPTQ-MXFP4`, revision
`b66f3cdedce0a64f1fbd4e91c3a89aabddcc00b4`. Hub metadata reports 48 files and
171,226,728,144 bytes of used storage. All 34 remote LFS OIDs match the local
`model.sha256.json` shard SHA-256 values; the verification receipt is
`build/reports/hf-publication/flashnext-remote-verification.json`.
