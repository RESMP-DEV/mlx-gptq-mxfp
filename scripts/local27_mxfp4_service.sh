#!/usr/bin/env bash
set -euo pipefail

readonly REPO=/Users/kearm/mlx-stack
UV=/Users/kearm/.local/bin/uv
readonly LABEL=dev.resmp.local27-mxfp4
readonly PROXY_LABEL=dev.resmp.local27-admission
readonly WATCH_LABEL=dev.resmp.local27-metrics-watch
readonly PLIST="$REPO/scripts/$LABEL.plist"
readonly PROXY_PLIST="$REPO/scripts/$PROXY_LABEL.plist"
readonly WATCH_PLIST="$REPO/scripts/$WATCH_LABEL.plist"
readonly STATE=/Users/kearm/.local/state/local27-mxfp4
readonly URL=http://127.0.0.1:8081/metrics

mkdir -p "$STATE/logs" "$STATE/diagnostics" /Users/kearm/.cache/alphaheng/local27-prefix-l2

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
		--model /Users/kearm/models/Qwen3.8-27B-MLX-GPTQ-MXFP4 \
		--draft /Users/kearm/models/z-lab-Qwen3.8-27B-DFlash2-GPTQ-MXFP4-1981cycles \
		--host 127.0.0.1 --port 8081 \
		--temp 0 --max-tokens 512 \
		--chat-template-args '{"enable_thinking":false}' \
		--prefill-step-size 512 \
		--prefix-cache --prefix-cache-max-bytes 4GB \
		--prefix-cache-l2 --prefix-cache-l2-dir /Users/kearm/.cache/alphaheng/local27-prefix-l2 \
		--prefix-cache-l2-max-bytes 64GB \
		--prompt-cache-size 32 \
		--wired-limit auto --cache-limit 4GB \
		--diagnostics basic --diagnostics-dir "$STATE/diagnostics" \
		--log-level INFO
}

loaded() {
	launchctl print 'gui/'"$(id -u)"'/'"$1" >/dev/null 2>&1
}

bootout_if_loaded() {
	if loaded "$1"; then
		launchctl bootout "gui/$(id -u)/$1"
	fi
}

case "${1:-run}" in
run)
	run_server
	;;
restore)
	if ! loaded "$LABEL"; then
		launchctl bootstrap "gui/$(id -u)" "$PLIST"
	fi
	if ! loaded "$PROXY_LABEL"; then
		launchctl bootstrap "gui/$(id -u)" "$PROXY_PLIST"
	fi
	if ! loaded "$WATCH_LABEL"; then
		launchctl bootstrap "gui/$(id -u)" "$WATCH_PLIST"
	fi
	launchctl kickstart -k "gui/$(id -u)/$LABEL"
	launchctl kickstart "gui/$(id -u)/$PROXY_LABEL"
	launchctl kickstart "gui/$(id -u)/$WATCH_LABEL"
	launchctl print "gui/$(id -u)/$LABEL" | sed -n '1,60p'
	launchctl print "gui/$(id -u)/$PROXY_LABEL" | sed -n '1,40p'
	;;
evict)
	bootout_if_loaded "$PROXY_LABEL"
	bootout_if_loaded "$LABEL"
	echo "Evicted $PROXY_LABEL and $LABEL. Run '$0 restore' to reload the resident model."
	;;
status)
	if loaded "$LABEL"; then
		launchctl print "gui/$(id -u)/$LABEL" | sed -n '1,100p'
	else
		echo "$LABEL is not loaded"
	fi
	if loaded "$PROXY_LABEL"; then
		launchctl print "gui/$(id -u)/$PROXY_LABEL" | sed -n '1,60p'
	else
		echo "$PROXY_LABEL is not loaded"
	fi
	curl -fsS --max-time 3 http://127.0.0.1:8080/healthz | jq .
	curl -fsS --max-time 3 "$URL" | jq '{server,memory,rates,prefix_cache}'
	;;
*)
	echo "Usage: $0 {run|restore|evict|status}" >&2
	exit 2
	;;
esac
