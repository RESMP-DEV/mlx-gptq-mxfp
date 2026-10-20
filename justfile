default:
    @just --list

quality:
    @PYTHONPATH={{justfile_directory()}} uv run --frozen python scripts/check_stack.py
