# DFlash2 generation-3 combined-corpus promotion and Triton kernel

Date: 2026-10-19

## Scope

This records the third DFlash2 distillation generation, which combined the
legacy 24-prompt capture with the Fable training prompts, the Triton grouped
dynamic convolution, and the first candidate to beat the incumbent on both live
gates.

## Combined capture

Generation 3 trains on a combined corpus built on B550 at
`/home/kearm/dflash2-distill/combined-incumbent-capture` by symlinking cycles
from both captures. The 25 Fable live-gate prompts are excluded; the legacy
24-prompt corpus is included because the incumbent already trained on it.

- cycles: 11,273 across 127 prompt groups
- sources: generation-1 corpus (24 prompts) + Fable training prompts (103 of 128)
- index/receipt written with provenance and a non-claim about corpus overlap

## Optimizer sweeps

Two 16-trial studies on separate RTX 3090 Ti GPUs, both resuming from the
promoted head on the combined capture:

- eager study `combined-g3-gpu0`, seed 43: best rank 8, alpha 16, predicted
  predecessors, LR `1.031001037545979e-6`, weight decay
  `1.2416974848056795e-4`, 2 epochs, held-out prefix `0.790449`.
- Triton study `combined-g3-gpu1-triton`, seed 47: best rank 8, alpha 8,
  teacher predecessors, LR `1.1174468616178302e-6`, weight decay
  `2.869073336644482e-5`, 2 epochs, held-out prefix `0.786364`.

Study summaries: `build/reports/dflash/g3-receipts/combined-g3-gpu*-summary.json`.

## Full runs

Both full-corpus runs trained two epochs over 11,273 cycles with offline W&B
logging, synced to the dedicated instance `retis_labs/dflash2-distillation`:

- eager run `g3-r8-combined-full`, W&B run `oq19e0ql`, 18,204 steps, best
  held-out prefix `0.793221`, top-1 `0.871488`.
- Triton run `g3-triton-r8-combined-full`, W&B run `nb2rwk1d`, 18,292 steps,
  best held-out prefix `0.790197`, top-1 `0.868555`.

The eager full run reported 636.1 seconds of wall time while another tenant job
was sharing GPU0; the Triton full run reported 259.2 seconds. Those wall-clock
numbers are not a clean comparison because the two runs did not execute under
matched GPU occupancy. The controlled kernel measurements below are the
evidence for the Triton path.

## Live gates and the promoted candidate

No single interpolated generation-3 candidate crossed both live gates. The
combined generation-2 Fable adaptation plus generation-3 combined adaptation at
strengths `0.0625` and `0.1875` did:

- legacy 24-prompt corpus: `0.671875` vs incumbent `0.6712239583333334`,
  delta `+0.00065104166666663` (5 wins, 14 ties, 5 losses).
- Fable 25-prompt held-out: `0.64921875` vs incumbent `0.6471875`, delta
  `+0.00203125` (11 wins, 5 ties, 9 losses).
- exact 256-token parity: `token_match: true`, smoke acceptance `0.7109375`.

Promoted artifact:
`/Users/kearm/models/z-lab-Qwen3.8-27B-DFlash2-G3-g20.0625-g30.1875`, model
SHA-256
`c90e0fc65dddbe4107b1b417d771c3be188c87197e89cf4b14415f56abc866f0`, blend
receipt SHA-256
`e53a6bb5e890d7e3368053302189766c3ef68d2eb7887152ece1ea9eac4f64d5`.

Paired receipts:
`build/reports/dflash/g3-g20.0625-g30.1875-vs-g1-corpus24.json` and
`build/reports/dflash/g3-g20.0625-g30.1875-vs-g1-heldout25.json`.
Diagnostic corpus throughput was `25.239` and `25.651` tok/s on the two gates;
thermal pressure was not controlled.

Rejected single-generation interpolations, all exact-parity clean:
`g3-eager-r8-blend{0.0625,0.1250,0.1563,0.1875,0.250,0.500,1.000}` and
`g3-triton-r8-blend{0.250,0.500,1.000}`. The `0.1875` eager blend was the
best legacy-only candidate at `0.6722005208333334` but lost Fable at
`0.6465625`, so it was not promoted.

## Triton grouped dynamic convolution

`mlx_gptq/triton_dynamic_conv.py` replaces the DFlash2 PyTorch
`GroupedDynamicCausalConv` Python loop with one Triton launch, with an autograd
wrapper so it trains. It is verified against the eager path in forward and
backward on CUDA (`tests/test_triton_dynamic_conv.py`).

Controlled kernel measurement on one RTX 3090 Ti
(`triton-dynamic-conv-benchmark.json`): isolated convolution is
`1.50x` to `1.60x` faster than eager at sequence lengths 3 to 5.

Kernel-shape tuning selected `DFLASH_TRITON_BLOCK_GROUPS=16`,
`DFLASH_TRITON_NUM_WARPS=1`. Over three repeats of a full measured DFlash2
train step (`triton-full-step-repeat.tsv`):

- eager mean `0.0081503` seconds per cycle;
- Triton mean `0.0078824` seconds per cycle;
- full-step gain about `3.3%`.

Nsight Compute was attempted but blocked by
`ERR_NVGPUCTRPERM`; profiling switched to Nsight Systems, whose kernel summary
is in `g3-receipts/triton-nsys-dynamic-conv-kernels.csv` and shows the fused
`_grouped_dynamic_conv_kernel` replacing the eager elementwise/addcmul kernel
chain. The default trainer path stays eager; `--triton-dynamic-conv` enables
the kernel.

## Code

`RESMP-DEV/mlx-gptq-mxfp` branch `work/dflash2-calibration`:
`d3d27bf` (Tune Triton DFlash2 training path), `de111f7` (Benchmark full
DFlash2 training steps), `3595be0` (Add Triton DFlash2 dynamic convolution),
`b8dbdbe` (Bind resumed sweeps to their capture corpus). Local tests: 30 passed,
3 skipped; the CUDA Triton test passes on B550.

## Non-claims

- Offline prefix accuracy is a proxy and is not acceptance.
- The Fable held-out prompts come from the same eval shard as the Fable training
  prompts; they are not an out-of-domain generalization test.
- Throughput and memory are diagnostic because thermal pressure and co-tenant
  occupancy are not controlled.
- The pinned BF16 Qwen3.8-27B target was not used as the capture teacher.
- No rejected artifact was uploaded to Hugging Face.
