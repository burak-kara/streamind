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
  $0 -p vllm-qwen3.5-4b-awq
  $0 --window 30 --profile vllm-qwen3.5-4b-awq
  $0 -p vllm-qwen3.5-4b-awq --asr small.en

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

# Each launch gets its own timestamped results root so reruns never clobber and
# nothing is deleted — results/<stamp>/<model>/<window>/window_*.json. The
# transmitter appends <model>/<window> itself; we only steer its results_dir.
STAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/${STAMP}"
OUT_DIR="${RESULTS_DIR}/$(model_dir_of "$PROFILE")/${WINDOW}"

ASSEMBLED="./tmp/config-${WINDOW}s-${PROFILE_NAME}.json"
ASR_ARGS=""
if [[ -n "${ASR}" ]]; then
  ASR_ARGS="--asr ${ASR}"
  ASSEMBLED="./tmp/config-${WINDOW}s-${PROFILE_NAME}-asr-${ASR}.json"
fi
uv run python tools/assemble_config.py "$WINDOW" "$PROFILE_NAME" "$ASSEMBLED" \
  --results-dir "$RESULTS_DIR" ${ASR_ARGS}

# Runtime env (FlashInfer sampler off + CUBLAS on LD_LIBRARY_PATH for ASR).
prep_runtime_env

echo "Window:     ${WINDOW}s"
echo "Summarizer: ${PROFILE_NAME}  (model: ${MODEL_PATH})"
echo "Output:     ${OUT_DIR}/"
if [[ -n "${ASR}" ]]; then
  echo "ASR:        ${ASR}  (override)"
fi
echo ""

# Launch in its own process group so Ctrl-C / kill tears down the whole tree
# (uv -> python -> vLLM EngineCore worker) and reclaims VRAM. Killing just the
# top pid orphans the workers, which keep holding GPU memory. Negative pid in
# kill(1) = process group.
LAUNCH_PGID=""
teardown() {
  [ -n "$LAUNCH_PGID" ] || return 0
  local pgid="$LAUNCH_PGID"; LAUNCH_PGID=""
  kill -TERM -"$pgid" 2>/dev/null || return 0
  for _ in $(seq 1 15); do kill -0 -"$pgid" 2>/dev/null || return 0; sleep 1; done
  kill -KILL -"$pgid" 2>/dev/null
}
trap 'teardown' EXIT
trap 'teardown; exit 130' INT TERM

# --auto: start/stop pipeline without interactive prompt (detached containers
# have no stdin — juturna launch would otherwise EOFError waiting for input).
set -m
uv run python -m juturna launch --config "$ASSEMBLED" --auto &
LAUNCH_PGID=$!
set +m
wait "$LAUNCH_PGID"
LAUNCH_PGID=""
