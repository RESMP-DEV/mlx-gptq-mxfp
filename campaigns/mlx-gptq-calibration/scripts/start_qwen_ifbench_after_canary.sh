#!/usr/bin/env bash
set -euo pipefail

BASE=/home/kearm/benchmarks/mlx-gptq-card-comparison/qwen/ifbench-thinking-xhigh
CANARY_DIR=${1:-$BASE/canary-tp2-v3}
CANARY_PID_FILE="$CANARY_DIR/runner.pid"
CANARY_RESULT="$CANARY_DIR/sample-0000.json"
FULL_RUNNER=/home/kearm/benchmarks/mlx-gptq-card-comparison/qwen/run-qwen-ifbench-thinking-all.sh

canary_pid=$(sed -n '1p' "$CANARY_PID_FILE")
while kill -0 "$canary_pid" 2>/dev/null; do
  sleep 15
done

jq -e '
  .status == "success" and
  .error == null and
  .results.completed_samples == 1 and
  .samples[0].output.choices[0].stop_reason == "stop" and
  ([.samples[0].scores[].metadata.reasoning_parser] | all(. == "structured")) and
  ([.samples[0].scores[].metadata.reasoning_closed] | all(. == true)) and
  ([.samples[0].scores[].metadata.final_answer_chars] | all(. > 0))
' "$CANARY_RESULT" >/dev/null

exec "$FULL_RUNNER" 8210 32768
