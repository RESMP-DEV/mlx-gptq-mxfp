#!/usr/bin/env bash
set -euo pipefail

# The runtime code can come from a different checkout than this script. The
# environment override points the service at the checkout that carries the
# target-batch implementation; without it the script uses its own repo.
REPO="${LOCAL27_TARGET_BATCH_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}"
readonly REPO
readonly UV=/Users/kearm/.local/bin/uv
readonly LABEL=dev.resmp.local27-target-batch
readonly PLIST="$REPO/scripts/$LABEL.plist"
readonly STATE=/Users/kearm/.local/state/local27-target-batch
readonly URL=http://127.0.0.1:8082/metrics
readonly MODEL=/Users/kearm/models/Qwen3.8-27B-MLX-GPTQ-MXFP4

mkdir -p "$STATE/logs" "$STATE/diagnostics"

run_server() {
	if [ ! -x "$UV" ]; then
		echo "uv is missing at $UV" >&2
		exit 78
	fi
	export PYTHONPATH="$REPO/runtime/dflash-mlx:$REPO"
	export HF_HUB_OFFLINE=1
	export TRANSFORMERS_OFFLINE=1
	exec "$UV" run --no-project \
		--with mlx==0.32.1 --with mlx-lm==0.31.3 \
		python -m dflash_mlx.cli serve \
		--model "$MODEL" \
		--host 127.0.0.1 --port 8082 \
		--target-batch-only \
		--decode-concurrency 64 --prompt-concurrency 16 \
		--temp 0 --max-tokens 512 \
		--chat-template-args '{"enable_thinking":false}' \
		--prompt-cache-size 32 \
		--wired-limit auto --cache-limit 8GB \
		--diagnostics basic --diagnostics-dir "$STATE/diagnostics" \
		--log-level INFO
}

loaded() {
	launchctl print "gui/$(id -u)/$LABEL" >/dev/null 2>&1
}

wait_ready() {
	local deadline=$((SECONDS + 300))
	while (( SECONDS < deadline )); do
		if curl -fsS --max-time 2 "$URL" |
			jq -e --arg model "$MODEL" \
				'.server.mode == "target_batch" and .server.model == $model' \
			>/dev/null; then
			return 0
		fi
		sleep 1
	done
	echo "target-batch service did not become ready at $URL" >&2
	return 1
}

case "${1:-usage}" in
run)
	run_server
	;;
install)
	plutil -lint "$PLIST"
	if loaded; then
		echo "$LABEL is already loaded"
	else
		launchctl bootstrap "gui/$(id -u)" "$PLIST"
	fi
	launchctl kickstart "gui/$(id -u)/$LABEL"
	wait_ready
	;;
uninstall|stop)
	if loaded; then
		launchctl bootout "gui/$(id -u)/$LABEL"
	fi
	echo "$LABEL is stopped."
	;;
restart)
	if ! loaded; then
		launchctl bootstrap "gui/$(id -u)" "$PLIST"
	fi
	launchctl kickstart -k "gui/$(id -u)/$LABEL"
	wait_ready
	;;
status)
	if loaded; then
		launchctl print "gui/$(id -u)/$LABEL" | sed -n '1,100p'
	else
		echo "$LABEL is not loaded"
	fi
	curl -fsS --max-time 3 "$URL" | jq '{server,memory,rates}' || true
	;;
health)
	loaded || {
		echo "$LABEL is not loaded" >&2
		exit 1
	}
	curl -fsS --max-time 5 "$URL" |
		jq -e --arg model "$MODEL" \
			'.server.mode == "target_batch" and .server.model == $model'
	;;
usage|*)
	echo "Usage: $0 {run|install|stop|restart|status|health}" >&2
	exit 2
	;;
esac
