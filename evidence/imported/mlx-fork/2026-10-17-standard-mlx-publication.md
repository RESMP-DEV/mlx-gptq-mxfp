# Standard MLX public-fork publication

Date: 2026-10-17

## Scope

This records publication of the standard MLX fork to the public RESMP-DEV
remote after the user authorized building in public. It does not create or
modify an upstream contribution.

## Published state

- Public remote: `RESMP-DEV/mlx`
- Branch: `main`
- Head: `56428f725ec76d84ec0070ba539682a6485ea840`
- Previous public head: `53aa4566732c6e86a09fe3101e44f1a354ab1f84`
- Upstream head remains: `b6642e604843776705b57b1eaecb6f57e519b793`

The delta is current upstream plus the local Strata/PLE architecture and
benchmark records, the selective-upstream sync tooling, and the local revert of
upstream `494c8a6c9`. The revert preserves the established Qwen exact-output
contract.

## Validation

Before publication, the focused live-metal suites passed:

```text
PYTHONPATH="$PWD/python" .venv/bin/python -m pytest \
  python/tests/test_quantized.py \
  python/tests/test_blas.py \
  python/tests/test_conv.py \
  python/tests/test_fast_sdpa.py \
  python/tests/test_reduce.py \
  python/tests/test_extension.py -q

172 passed, 13 skipped, 12668 subtests passed in 13.46s
```

The prior merge receipt also records exact short-context and 4,096-token Qwen
output pairs for this core lineage.

## Publication checks

- The public repository is public and its default branch is `main`.
- GitHub Actions are disabled at the repository level.
- Git and the GitHub API both report the published head.
- No credential pattern or confidential optimization-methodology term was found
  in the publication delta.
- The publication delta contains provenance paths in evidence documents but no
  credential values.
- Upstream was not pushed and no upstream PR was opened.

## Non-claims

This publication does not claim a generic MLX API addition, cross-platform
coverage beyond the focused suites, Optane attachment, distributed execution, or
upstream acceptance. The exact-output exclusion of upstream `494c8a6c9` remains
local until the full numerical gate budget is run.
