#!/usr/bin/env bash
set -euo pipefail

: "${SUMMARIZER_PROFILE:=vllm-qwen3.5-4b}"
: "${WINDOW_SECONDS:=300}"

exec ./tools/run_pipeline.sh --window "$WINDOW_SECONDS" --profile "$SUMMARIZER_PROFILE"
