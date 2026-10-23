#!/usr/bin/env bash
set -euo pipefail

readonly REPO=/Users/kearm/mlx-stack
readonly MLX_LM=/Users/kearm/mlx-lm
readonly MODEL=/Users/kearm/models/RecursiveSI-Flint-40B-A2.5B-MLX-GPTQ-MXFP4-224
readonly LABEL=dev.resmp.flint224-mxfp4
readonly PLIST="$REPO/scripts/$LABEL.plist"
readonly STATE=/Users/kearm/.local/state/flint224-mxfp4
readonly PORT=8084

mkdir -p "$STATE/logs"

run_server() {
	exec env -i \
		HOME="$HOME" \
		PATH="/usr/bin:/bin:/usr/sbin:/sbin" \
		PYTHONPATH="$MLX_LM" \
		PYTHONUNBUFFERED=1 \
		HF_HUB_OFFLINE=1 \
		TRANSFORMERS_OFFLINE=1 \
		"$MLX_LM/.venv/bin/python" "$REPO/scripts/flint_mlx_server.py" \
		--listen-backlog 1024 \
		--model "$MODEL" \
		--host 127.0.0.1 --port "$PORT" \
		--decode-concurrency 4 \
		--prompt-concurrency 2 \
		--prefill-step-size 2048 \
		--max-tokens 2048 \
		--prompt-cache-size 32 \
		--temp 0 \
		--log-level INFO
}

loaded() {
	launchctl print "gui/$(id -u)/$1" >/dev/null 2>&1
}

case "${1:-run}" in
run)
	run_server
	;;
restore)
	if ! loaded "$LABEL"; then
		launchctl bootstrap "gui/$(id -u)" "$PLIST"
	fi
	launchctl kickstart -k "gui/$(id -u)/$LABEL"
	;;
evict)
	if loaded "$LABEL"; then
		launchctl bootout "gui/$(id -u)/$LABEL"
	fi
	;;
status)
	if loaded "$LABEL"; then
		launchctl print "gui/$(id -u)/$LABEL" | sed -n '1,100p'
	else
		echo "$LABEL is not loaded"
	fi
	curl -fsS --max-time 5 "http://127.0.0.1:$PORT/health"
	echo
	curl -fsS --max-time 5 "http://127.0.0.1:$PORT/v1/models"
	echo
	;;
*)
	echo "Usage: $0 {run|restore|evict|status}" >&2
	exit 2
	;;
esac
