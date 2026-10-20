# Repository Guidelines

## Project Structure & Module Organization

Core Python code lives in `mlx_gptq_calibration/`: CLI orchestration in `cli.py`,
campaign validation in `campaign.py`, and specialized logic in `hybrid.py` and
`logit_trace.py`. Reproducible model settings
belong in `campaigns/*.json`; do not hard-code checkpoint revisions or calibration
hashes in ad hoc scripts. Tests are under `tests/`, helper dataset builders under
`scripts/`, experiment matrices under `experiments/`, and supporting evidence under
`receipts/`. Keep generated run output in the ignored `runs/` directory.

## Build, Test, and Development Commands

- `uv sync --frozen` creates the locked Python 3.11+ environment.
- `uv run python -m unittest discover -s tests` runs the complete test suite.
- `uvx ruff check .` checks imports, modernization, and style rules.
- `uv build` produces source and wheel distributions through Hatchling.
- `uv run mlx-gptq-calibration verify` validates the default campaign and calls the
  live Hugging Face API for pinned revisions.
- `uv run mlx-gptq-calibration probe --host <ssh-alias>` checks real CUDA capacity;
  never infer hardware support from a hostname or documentation alone.

## Coding Style & Naming Conventions

Use four-space indentation, type annotations, `pathlib.Path`, and focused pure helpers
where practical. Ruff targets Python 3.11 with a 100-character line limit and rules
`E`, `F`, `I`, and `UP`. Use `snake_case` for modules, functions, and variables;
`PascalCase` for classes; and descriptive kebab-case campaign filenames such as
`gemma4-26b-a4b-qat-mxfp4.json`. Preserve pinned revisions, hashes, and namespace
mappings explicitly.

## Testing Guidelines

Tests use the standard-library `unittest` framework. Name files `test_<area>.py`, test
classes `<Area>Tests`, and methods `test_<behavior>`. Add regression coverage for
campaign validation, generated commands, artifact integrity, and failure paths. Mock
or isolate filesystem cases with `tempfile`; reserve live API, SSH, and GPU probes for
explicit integration verification.

## Commit & Pull Request Guidelines

Recent commits use short, imperative subjects such as `Use tokenizer-matched
calibration matrices`. Keep commits scoped and include tests or receipts with behavior
changes. Pull requests should describe pipeline impact, list verification results,
link issues, and identify remote hardware used. Inspect thread-aware review state,
invoke configured reviewers, and resolve actionable findings with evidence. Do not
treat an unavailable optional reviewer as approval or bypass required gates.
