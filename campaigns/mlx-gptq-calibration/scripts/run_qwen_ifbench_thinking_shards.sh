#!/usr/bin/env bash
set -euo pipefail

HARNESS=/home/kearm/benchmarks/mlx-gptq-card-comparison/harnesses/openbench-6a080a2f
RUN=/home/kearm/benchmarks/mlx-gptq-card-comparison/qwen/ifbench-thinking-xhigh
PY=/home/kearm/benchmarks/mlx-gptq-card-comparison/qwen/run-qwen-ifbench-thinking.py
PORT=${1:?port}
OFFSET=${2:?offset must be 0 or 1}
MAX_TOKENS=${3:-32768}

mkdir -p "$RUN/chunks"
cd "$HARNESS"

for start in $(seq "$OFFSET" 2 299); do
  end=$((start + 1))
  chunk=$(printf '%04d' "$start")
  result="$RUN/chunks/chunk-$chunk.json"
  if test -f "$result" &&
    jq -e '.status == "success" and .error == null and .results.completed_samples == 1' "$result" >/dev/null; then
    continue
  fi
  rm -f "$result"
  VLLM_API_KEY=local .venv/bin/python "$PY" \
    --port "$PORT" \
    --start "$start" \
    --end "$end" \
    --log-dir "$RUN/chunks" \
    --logfile "chunk-$chunk" \
    --max-tokens "$MAX_TOKENS"
done
