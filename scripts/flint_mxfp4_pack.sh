#!/usr/bin/env bash
set -euo pipefail

REPO=/Users/kearm/mlx-stack
SOURCE="${SOURCE:-/Users/kearm/models/RecursiveSI-flint-BF16-2a247be}"
CALIB="${CALIB:-/Users/kearm/mlx-stack/build/flint-calibration/artifacts-mxfp4}"
OUTPUT="${OUTPUT:-/Users/kearm/models/RecursiveSI-Flint-40B-A2.5B-MLX-GPTQ-MXFP4}"
HOLDOUT="${HOLDOUT:-/Users/kearm/mlx-stack/build/flint-calibration/holdout/data/heldout/agentic-coding/agentic-coding-long-context.txt}"
REVISION=2a247be0c0aec895ccb0ac7ca92229ec6df37288

usage() {
	echo "Usage: SOURCE=... CALIB=... OUTPUT=... HOLDOUT=... $0 [pack|evaluate]" >&2
	exit 2
}

check_inputs() {
	python3 - "$SOURCE" "$CALIB" "$HOLDOUT" "$REVISION" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

source, calib, holdout, revision = sys.argv[1:]
config = json.loads((Path(source) / "config.json").read_text())
assert config["model_type"] == "flint"
assert config["architectures"] == ["FlintForCausalLM"]
manifest = json.loads((Path(calib) / "manifest.json").read_text())
assert manifest["mode"] == "mxfp4"
assert (manifest["bits"], manifest["group_size"]) == (4, 32)
assert manifest["algorithm"] == "safe-scale-search"
assert manifest["nsamples"] == 256 and manifest["seqlen"] == 2048
assert hashlib.sha256((Path(calib) / "calib_tokens.npy").read_bytes()).hexdigest() == (
    "240415002e578a293c92f81af897d34279f2dee04e5c48b233a6d3205a143f77"
)
assert hashlib.sha256(Path(holdout).read_bytes()).hexdigest() == (
    "e615a4149b1b16e5327d8cf79cbb70b07a4007c29cc2302ffc13291cc36455cd"
)
assert len(list(Path(source).glob("model-*.safetensors"))) == 15
print("flint pack inputs verified", revision)
PY
}

pack() {
	if [ -e "$OUTPUT" ]; then
		echo "output already exists: $OUTPUT" >&2
		exit 1
	fi
	check_inputs
	mkdir -p "$REPO/build/reports/flint"
	"$REPO/.venv/bin/python" -m mlx_gptq.validate_artifacts \
		--model "$SOURCE" --calib "$CALIB" \
		--output "$REPO/build/reports/flint/artifact-validation.json"
	PYTHONPATH="$REPO:/Users/kearm/mlx-lm" uv run --no-project \
		--with 'mlx==0.32.4' --with torch --with 'transformers>=5.17' \
		--with accelerate --with safetensors --with numpy \
		python -m mlx_gptq.pack \
		--hf-path "$SOURCE" \
		--calib "$CALIB" \
		--mlx-path "$OUTPUT" \
		--verify
}

evaluate() {
	check_inputs
	for arm in "$SOURCE" "$OUTPUT"; do
		PYTHONPATH="/Users/kearm/mlx-lm" /Users/kearm/mlx-lm/.venv/bin/python \
			-m mlx_gptq.eval_ppl --model "$arm" \
			--text "$HOLDOUT" --seqlen 2048 --windows 8
	done
}

case "${1:-pack}" in
pack) pack ;;
evaluate) evaluate ;;
*) usage ;;
esac
