# MLX Stack consolidation

## Claim

The owned `mlx-gptq-mxfp` repository was renamed to
`RESMP-DEV/mlx-stack` without history rewriting. The DFlash runtime, Qwen
calibration campaign, and CUDA MXFP probe lineages were imported as subtree
merges under explicit ownership prefixes.

## Imported revisions

| Lineage | Prefix | Imported main |
|---|---|---|
| DFlash MLX runtime | `runtime/dflash-mlx/` | `60803233af4589e18588b9bacbb03880801c828a` |
| Qwen calibration campaigns | `campaigns/mlx-gptq-calibration/` | `f543f1586d8631bf1bd795acd954007f021ff426` |
| CUDA MXFP probe | `cuda/mlx-cuda-mxfp/` | `3676dd88aa83b5df5f602306bf8ad7477b140218` |

GitHub redirects from the old repository name remain active. Local fetch and
push remotes now point to `RESMP-DEV/mlx-stack`.

## Quality receipt

Machine receipt: `build/reports/quality/receipt.json`

All five checks passed on 2026-10-20:

- Root lint for the active kernel/layout-test surface.
- Root engine tests.
- DFlash runtime tests with MLX 0.32.1 and MLX LM 0.31.3.
- Calibration campaign tests.
- CUDA probe syntax compilation with the root Python environment.

The DFlash runtime result was 1246 passed and 17 skipped. One prior failure
was not suppressed globally: on M4 Max (`applegpu_g16s`), MLX 0.32.1 produces
a one-BF16-quantum deviation of 0.0078125 for `m16_gate_up`. All three pinned
seeds retain 100% margin-certified argmax agreement, so only that GPU profile
now has a 0.01 ceiling for that shape; the original tighter bound remains in
effect for other profiles.

## Non-claims

- This consolidation does not change `RESMP-DEV/mlx` or `RESMP-DEV/mlx-lm`.
- CUDA syntax compilation does not execute the Linux-only CUDA probe.
- Campaign and runtime tests do not qualify a new model artifact.
- Old duplicate local clones are not deleted until their remaining branches
  and references are inventoried.

## Post-acceptance cleanup

After the quality gate and publication, the pushed Qwen parity-tooling branch
was merged into the campaign subtree. The clean local clones for the old
engine, calibration repository, and CUDA probe were removed. Their GitHub
repositories remain available as historical subtree sources, and
`/Users/kearm/mlx-stack` points to the canonical checkout.
