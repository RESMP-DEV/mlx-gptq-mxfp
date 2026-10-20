# Contributing

## Workflow

1. Preserve unrelated dirty work and use a focused branch.
2. Implement against the owning layout boundary in `AGENTS.md`.
3. Run `just quality`.
4. Add or update a dated evidence receipt for live hardware or provider claims.
5. Make an atomic conventional commit and push to the owned RESMP-DEV remote.

## Code standards

- Python uses `uv`, current typing, `ruff`, and `pathlib`.
- Kernel changes require byte-level layout tests plus a live hardware receipt.
- Synthetic throughput results never substitute for model-quality gates.
- Keep machine state under ignored `build/`; sanitize durable evidence before
  committing it under `evidence/`.

## Work records

A trusted result records the claim, source revision and dirty state, exact
commands, exit codes or measurements, and explicit non-claims. Review worker
output is evidence; the parent's consolidated receipt is acceptance.
