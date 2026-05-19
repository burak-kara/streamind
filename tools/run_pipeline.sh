#!/usr/bin/env bash
set -euo pipefail

# Default values
WINDOW=300
SUMMARIZER=""

usage() {
  cat <<EOF
Usage: $0 [OPTIONS]

Options:
  -w, --window <seconds>      Window duration in seconds (default: 300)
  -s, --summarizer <profile>  Summarizer profile under pipelines/summarizer/<profile>.json (required)
  -h, --help                  Show this help message

Examples:
  $0 -s vllm-qwen3-8b
  $0 --window 30 --summarizer vllm-qwen3-8b

Judge runs offline after the pipeline exits:
  uv run python tools/eval_quality.py results/<model>/<window>/ --judge-profile <judge_profile>
EOF
}

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
      usage
      exit 0
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
    *)
      echo "Unexpected positional arg: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${SUMMARIZER}" ]]; then
  available=$(ls pipelines/summarizer/*.json 2>/dev/null | xargs -n1 basename | sed 's/\.json//' || true)
  echo "ERROR: --summarizer is required." >&2
  echo "Available profiles: ${available:-(none — create one under pipelines/summarizer/)}" >&2
  exit 1
fi

PROFILE="pipelines/summarizer/${SUMMARIZER}.json"
if [[ ! -f "$PROFILE" ]]; then
  echo "ERROR: profile not found: $PROFILE" >&2
  echo "Available profiles:" >&2
  ls pipelines/summarizer/*.json 2>/dev/null | xargs -n1 basename | sed 's/\.json//' >&2 || echo "  (none)" >&2
  exit 1
fi

# Preflight: GPU + local model dir present (vLLM is in-process, not a daemon).
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found. CUDA pipeline requires a GPU host." >&2
  exit 1
fi

MODEL_PATH=$(uv run python -c "import json,sys; print(json.load(open('${PROFILE}'))['configuration']['model_name'])")
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "ERROR: model directory missing or empty: ${MODEL_PATH}" >&2
  LOCAL_NAME=$(basename "${MODEL_PATH}")
  echo "Run: ./tools/fetch_models.sh <hf_id> ${LOCAL_NAME}" >&2
  exit 1
fi

ASSEMBLED="./tmp/config-${WINDOW}s-${SUMMARIZER}.json"
uv run python tools/assemble_config.py "$WINDOW" "$SUMMARIZER" "$ASSEMBLED"

echo "Window:     ${WINDOW}s"
echo "Summarizer: ${SUMMARIZER}  (model: ${MODEL_PATH})"
echo ""
uv run python -m juturna launch --config "$ASSEMBLED"
