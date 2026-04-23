#!/usr/bin/env bash
set -euo pipefail

: "${SUMMARIZER_PROFILE:=ollama-qwen3.5-4b}"
: "${WINDOW_SECONDS:=300}"

ollama serve >/var/log/ollama.log 2>&1 &
OLLAMA_PID=$!

cleanup() {
    kill "$OLLAMA_PID" 2>/dev/null || true
    wait "$OLLAMA_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for i in $(seq 1 60); do
    if curl -sf http://127.0.0.1:11434/api/tags >/dev/null; then
        break
    fi
    sleep 1
done

exec ./tools/run_pipeline.sh --window "$WINDOW_SECONDS" --summarizer "$SUMMARIZER_PROFILE"
