#!/usr/bin/env bash
set -euo pipefail

HARNESS=/home/kearm/benchmarks/mlx-gptq-card-comparison/harnesses/openbench-6a080a2f
RUN=/home/kearm/benchmarks/mlx-gptq-card-comparison/qwen/ifbench-thinking-xhigh
PY=/home/kearm/benchmarks/mlx-gptq-card-comparison/qwen/run-qwen-ifbench-thinking.py
PORT=${1:-8210}
MAX_TOKENS=${2:-32768}

mkdir -p "$RUN/chunks"
cd "$HARNESS"

for start in $(seq 0 10 290); do
  end=$((start + 10))
  chunk=$(printf '%04d' "$start")
  result="$RUN/chunks/chunk-$chunk.json"
  if test -f "$result" &&
    jq -e '.status == "success" and .error == null and .results.completed_samples == 10' "$result" >/dev/null; then
    continue
  fi
  VLLM_API_KEY=local .venv/bin/python "$PY" \
    --port "$PORT" \
    --start "$start" \
    --end "$end" \
    --log-dir "$RUN/chunks" \
    --logfile "chunk-$chunk" \
    --max-tokens "$MAX_TOKENS"
done
