#!/usr/bin/env bash
# Resilient Hugging Face download using the hf CLI and xet transport.
#
# The aria2c backend collides with peer sessions that kill aria2c, so this
# path uses hf_transfer/xet, which resumes from the hub cache on retry.
set -uo pipefail

REPO="${REPO:?REPO required}"
REVISION="${REVISION:?REVISION required}"
DEST="${DEST:?DEST required}"
LOG="${LOG:-$DEST.hf.log}"
MAX_TRIES="${MAX_TRIES:-200}"
DELAY="${DELAY:-45}"
HF_BIN="${HF_BIN:-/Users/kearm/.local/bin/hf}"

mkdir -p "$DEST"
attempt=0
while [ "$attempt" -lt "$MAX_TRIES" ]; do
	attempt=$((attempt + 1))
	echo "=== attempt $attempt $(date -u +%FT%TZ) ===" >>"$LOG"
	if "$HF_BIN" download "$REPO" \
		--revision "$REVISION" \
		--local-dir "$DEST" \
		--include "*.json" \
		--include "*.safetensors" \
		--include "*.jinja" \
		--include "LICENSE" >>"$LOG" 2>&1; then
		echo "=== complete $(date -u +%FT%TZ) ===" >>"$LOG"
		exit 0
	fi
	echo "=== attempt $attempt failed, sleeping ${DELAY}s ===" >>"$LOG"
	sleep "$DELAY"
done
echo "=== exhausted $MAX_TRIES attempts ===" >>"$LOG"
exit 1
