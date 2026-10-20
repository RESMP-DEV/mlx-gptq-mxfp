# Promoted DFlash2 generation-3 Hugging Face publication

Date: 2026-10-19

## Repository

- Hub repository: `RESMP-DEV/Qwen3.8-27B-DFlash2-G3-Combined`
- URL: `https://huggingface.co/RESMP-DEV/Qwen3.8-27B-DFlash2-G3-Combined`
- visibility: private
- revision: `93d507ea14d63bb7063914725d4fa6b5749ac83d`
- last modified: `2026-10-19T15:08:50Z`

The repository contains six files:

- `.gitattributes`
- `README.md`
- `blend.json`
- `config.json`
- `metrics.json`
- `model.safetensors`

No trace dataset, prompt file, capture cycle, selector tensor, adapter file, or
W&B log was uploaded.

## Checkpoint verification

The remote LFS object is byte-for-byte accounted for by its SHA-256 OID:

- local `model.safetensors` SHA-256:
  `c90e0fc65dddbe4107b1b417d771c3be188c87197e89cf4b14415f56abc866f0`
- remote LFS OID:
  `c90e0fc65dddbe4107b1b417d771c3be188c87197e89cf4b14415f56abc866f0`
- local and remote size: `3,848,817,885` bytes

Other staged hashes:

- `config.json`:
  `873e3556509b0da06e29654ba00d4944888d4b5e8a33afde25f7eb27d321e980`
- `blend.json`:
  `e53a6bb5e890d7e3368053302189766c3ef68d2eb7887152ece1ea9eac4f64d5`
- `README.md`:
  `181b02a530b4441d2a547bae321d6322d5d8dc111a7097a08f6840d09fd2cbdf`
- `metrics.json`:
  `a4744d7c8093e19b1f7ef795976df6f3bd0ef4fcf743e9b7e450cc94880fcf61`

## Recorded metrics

The model card and `metrics.json` record the promoted generation-3 result:

- legacy 24-prompt mean acceptance: `0.671875`, versus incumbent
  `0.6712239583333334` (`+0.0006510416666666297`);
- Fable 25-prompt held-out mean acceptance: `0.64921875`, versus incumbent
  `0.6471875` (`+0.002031249999999929`);
- exact 256-token parity: `token_match: true`;
- parity-run acceptance: `0.7109375`.

The card also records the combined 11,273-cycle training capture, its 127
prompt groups, the excluded 25 Fable live-gate prompts, the calibrated MXFP4
teacher provenance, diagnostic-only throughput, and the trace-data omission.

## Evidence

Creation and upload logs:

- `build/reports/dflash/hf-g3-repo-create.json`
- `build/reports/dflash/hf-g3-upload.log`

The upload completed at Hub revision
`93d507ea14d63bb7063914725d4fa6b5749ac83d`. An authenticated Hub tree query
confirmed all six filenames and the model LFS OID above. An unauthenticated
tree query returned HTTP 401, confirming the private visibility.

## Namespace correction

The initial creation used `Kearm/Qwen3.8-27B-DFlash2-G3-Combined` because an
older HF-specific default named the personal namespace. That default conflicted
with the user's standing standard and was corrected. The repository was moved
with `hf repos move` to
`RESMP-DEV/Qwen3.8-27B-DFlash2-G3-Combined`; the move receipt is
`build/reports/dflash/hf-g3-repo-move.json`.

Post-move verification confirmed:

- repository ID and author: `RESMP-DEV/Qwen3.8-27B-DFlash2-G3-Combined`;
- visibility: private;
- revision unchanged: `93d507ea14d63bb7063914725d4fa6b5749ac83d`;
- model LFS OID unchanged:
  `c90e0fc65dddbe4107b1b417d771c3be188c87197e89cf4b14415f56abc866f0`;
- no distinct G3 repository remains under the `Kearm` author listing. The old
  namespace is a Hub rename redirect, not a second copy.

## Non-claims

Publication does not change the acceptance evidence. The metrics remain paired
runtime measurements, not broad task-quality, natural-EOS, long-context, or
controlled thermal-throughput proof. The repository intentionally does not
include the training traces or capture tensors.

## Rename, model-tree fix, and public visibility

Per a later user instruction, the repository was renamed and made public.

- Move receipt: `build/reports/dflash/hf-dflash2-rename.json`
  (`RESMP-DEV/Qwen3.8-27B-DFlash2-G3-Combined` -> `RESMP-DEV/DFlash2`).
- Card-update receipt: `build/reports/dflash/hf-dflash2-card-update.log`.
- Public-settings receipt: `build/reports/dflash/hf-dflash2-public.json`.
- Public info: `build/reports/dflash/hf-dflash2-public-info.json`.

Final state:

- Hub repository: `RESMP-DEV/DFlash2`
- URL: `https://huggingface.co/RESMP-DEV/DFlash2`
- visibility: public (unauthenticated API returns HTTP 200; `private: false`)
- revision after rename and card update:
  `035b76311db31244f0f15bf37cc539bdcaee34fb`
- old name redirects with HTTP 307, not a duplicate repository

The model card was rewritten to remove the confusing Hub model tree. The
`base_model` frontmatter field was dropped because it produced a
`Finetuned (575)` tree entry that mislabeled a small speculative draft head as
a fine-tune of the full Qwen target. Post-change Hub metadata reports
`baseModels: null` and `childrenModelCount` all zero, confirming the tree panel
is gone. The card now states plainly that the artifact is draft-only, has no
tokenizer or target weights, and must run inside a DFlash2 speculative runtime
with `Qwen/Qwen3.8-27B` as the verifier. The Qwen relationship is kept as a
prose sentence rather than a machine-readable base-model link.

The six-file contents, the `model.safetensors` LFS OID
`c90e0fc65dddbe4107b1b417d771c3be188c87197e89cf4b14415f56abc866f0`, and the
recorded acceptance and parity metrics are unchanged by the rename. Publication
visibility does not change the acceptance evidence.
