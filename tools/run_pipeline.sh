#!/usr/bin/env bash
set -euo pipefail

# Default values
WINDOW=300
SUMMARIZER=ollama-qwen3.5-9b

# Parse named arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -w|--window)
      WINDOW="$2"
      shift 2
      ;;
    -s|--summarizer)
      SUMMARIZER="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  -w, --window <seconds>      Window duration in seconds (default: 300)"
      echo "  -s, --summarizer <profile>   Summarizer profile (default: ollama-qwen3.5-9b)"
      echo "  -h, --help                  Show this help message"
      echo ""
      echo "Examples:"
      echo "  $0 -w 600 -s ollama-llama3.1-8b"
      echo "  $0 --window 300 --summarizer ollama-qwen3.5-9b"
      exit 0
      ;;
    -*)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

PROFILE="pipelines/summarizer/${SUMMARIZER}.json"
if [ ! -f "$PROFILE" ]; then
  echo "Profile not found: $PROFILE"
  echo "Available profiles:"
  ls pipelines/summarizer/*.json | xargs -n1 basename | sed 's/\.json//'
  exit 1
fi

ASSEMBLED="./tmp/config-${WINDOW}s-${SUMMARIZER}.json"
uv run python tools/assemble_config.py "$WINDOW" "$SUMMARIZER" "$ASSEMBLED"

echo "Window:     ${WINDOW}s  |  Summarizer: ${SUMMARIZER}"
echo ""
uv run python -m juturna launch --config "$ASSEMBLED"
