#!/usr/bin/env bash
set -euo pipefail

readonly REPO=/Users/kearm/mlx-stack
UV=/Users/kearm/.local/bin/uv
readonly LABEL=dev.resmp.local27-mlx-batch
readonly PLIST="$REPO/scripts/$LABEL.plist"
readonly STATE=/Users/kearm/.local/state/local27-mlx-batch
readonly URL=http://127.0.0.1:8084/metrics

mkdir -p "$STATE/logs"

run_server() {
	if [ ! -x "$UV" ]; then
		echo "uv is missing at $UV" >&2
		exit 78
	fi
	export HF_HUB_OFFLINE=1
	export TRANSFORMERS_OFFLINE=1
	exec "$UV" run --no-project \
		--with mlx==0.32.1 --with mlx-lm==0.31.3 \
		python -m mlx_lm.server \
		--model /Users/kearm/models/Qwen3.8-27B-MLX-GPTQ-MXFP4 \
		--host 127.0.0.1 --port 8084 \
		--decode-concurrency 32 --prompt-concurrency 8 \
		--temp 0 --max-tokens 512 \
		--chat-template-args '{"enable_thinking":false}'
}

loaded() {
	launchctl print 'gui/'"$(id -u)"'/'"$LABEL" >/dev/null 2>&1
}

case "${1:-run}" in
run)
	run_server
	;;
restore)
	if ! loaded; then
		launchctl bootstrap "gui/$(id -u)" "$PLIST"
	fi
	launchctl kickstart -k "gui/$(id -u)/$LABEL"
	;;
evict)
	launchctl bootout "gui/$(id -u)/$LABEL"
	echo "Evicted $LABEL."
	;;
status)
	if loaded; then
		launchctl print "gui/$(id -u)/$LABEL" | sed -n '1,60p'
	else
		echo "$LABEL is not loaded"
	fi
	curl -fsS --max-time 3 "$URL" || true
	;;
*)
	echo "Usage: $0 {run|restore|evict|status}" >&2
	exit 2
	;;
esac
