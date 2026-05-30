#!/usr/bin/env bash
set -euo pipefail

# Shared launch-prep (runtime env + model preflight) — single source of truth,
# also sourced by tools/compare_summarizers.sh.
source "$(dirname "$0")/lib/pipeline_prep.sh"

# Default values
WINDOW=300
PROFILE_NAME=""
ASR=""

usage() {
  cat <<EOF
Usage: $0 --profile <name> [OPTIONS]

Options:
  -p, --profile <name>   Summarizer profile under pipelines/summarizer/<name>.json (required)
  -w, --window <sec>     Window duration in seconds (default: 300)
  -a, --asr <model>      Override ASR model_name (e.g. small.en, large-v3-turbo)
  -h, --help             Show this help message

Examples:
  $0 -p vllm-qwen3.5-4b
  $0 --window 30 --profile vllm-qwen3.5-4b
  $0 -p vllm-qwen3.5-4b --asr small.en

Judge runs offline after the pipeline exits:
  uv run python tools/eval_quality.py results/<model>/<window>/ --judge-profile <judge_profile>
EOF
}

while [[ $# -gt 0 ]]; do
  case $1 in
    -p|--profile)
      PROFILE_NAME="$2"
      shift 2
      ;;
    -w|--window)
      WINDOW="$2"
      shift 2
      ;;
    -a|--asr)
      ASR="$2"
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

if [[ -z "${PROFILE_NAME}" ]]; then
  available=$(ls pipelines/summarizer/*.json 2>/dev/null | xargs -n1 basename | sed 's/\.json//' || true)
  echo "ERROR: --profile is required." >&2
  echo "Available profiles: ${available:-(none — create one under pipelines/summarizer/)}" >&2
  exit 1
fi

PROFILE="pipelines/summarizer/${PROFILE_NAME}.json"
if [[ ! -f "$PROFILE" ]]; then
  echo "ERROR: profile not found: $PROFILE" >&2
  echo "Available profiles:" >&2
  ls pipelines/summarizer/*.json 2>/dev/null | xargs -n1 basename | sed 's/\.json//' >&2 || echo "  (none)" >&2
  exit 1
fi

# Preflight: GPU + local model dir present (vLLM is in-process, not a daemon).
require_gpu || exit 1
require_model "$PROFILE" summarizer || exit 1
MODEL_PATH=$(model_path_of "$PROFILE")

ASSEMBLED="./tmp/config-${WINDOW}s-${PROFILE_NAME}.json"
ASR_ARGS=""
if [[ -n "${ASR}" ]]; then
  ASR_ARGS="--asr ${ASR}"
  ASSEMBLED="./tmp/config-${WINDOW}s-${PROFILE_NAME}-asr-${ASR}.json"
fi
uv run python tools/assemble_config.py "$WINDOW" "$PROFILE_NAME" "$ASSEMBLED" ${ASR_ARGS}

# Runtime env (FlashInfer sampler off + CUBLAS on LD_LIBRARY_PATH for ASR).
prep_runtime_env

echo "Window:     ${WINDOW}s"
echo "Summarizer: ${PROFILE_NAME}  (model: ${MODEL_PATH})"
if [[ -n "${ASR}" ]]; then
  echo "ASR:        ${ASR}  (override)"
fi
echo ""
uv run python -m juturna launch --config "$ASSEMBLED"
