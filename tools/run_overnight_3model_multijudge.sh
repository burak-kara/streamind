#!/usr/bin/env bash
# Overnight 3-model multi-judge battery on oneill 300s.
#
# Pipeline runs (each: setup + warmup + ~75min stream + auto-stop):
#   1. Qwen2.5-7B-Instruct (downloads + creates profile if missing)
#   2. Qwen3.5-9B (profile already exists from friend's merge)
#
# Then multi-judge (Mistral + Phi-4 + Gemma-3) the EXISTING 4B base oneill
# results so we have apples-to-apples consensus scores for all 3 models.
#
# Estimated total: ~4 hours
#   - download (~10 min if 7B needed)
#   - 7B pipeline + multi-judge: ~95 + 20 min
#   - 9B pipeline + multi-judge: ~95 + 20 min
#   - 4B multi-judge only: ~20 min
#
# Safe to leave running overnight. Final summary printed at end + saved to
# results/3model_battery_<timestamp>.txt
set -uo pipefail
cd "$(dirname "$0")/.."

# env that the pipeline needs (mirrors run_pipeline.sh)
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONUNBUFFERED=1

# --- Config ---
AUDIO="datasets/rev16/26_Episode_338_-_Special_Guest_Rob_O'Neill:_The_Man_Who_Killed_Osama_Bin_Laden/audio.opus"
WINDOW=300
TIMEOUT_BUFFER=180   # extra seconds past audio length (warmup + final flush)

# Pipelines to run: "profile-name:result-label"
PIPELINE_RUNS=(
  "vllm-qwen2.5-7b-instruct:qwen2.5-7b-instruct"
  "vllm-qwen3.5-9b:qwen3.5-9b"
)
# Result dirs to multi-judge (includes existing 4B base run)
JUDGE_TARGETS=(
  "qwen3.5-4b"
  "qwen2.5-7b-instruct"
  "qwen3.5-9b"
)
# Judge profiles to score with (eval_multi_judge.py requires explicit list)
JUDGE_PROFILES=(
  vllm-mistral-small-24b-awq
  vllm-phi-4-awq
  vllm-gemma3-27b-it-int4-awq
)
# --------------

STAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p tmp results
SUMMARY="results/3model_battery_${STAMP}.txt"
MLOG="results/3model_battery_${STAMP}.log"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$MLOG"; }

log "=== Overnight 3-model multi-judge battery ==="
log "audio: $AUDIO"
log "window: ${WINDOW}s"
log "pipeline runs: ${PIPELINE_RUNS[*]}"
log "multi-judge targets: ${JUDGE_TARGETS[*]}"

# Pre-flight checks
if [ ! -f "$AUDIO" ]; then
  log "FATAL: audio not found: $AUDIO"
  exit 1
fi
DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$AUDIO")
TIMEOUT=$(awk "BEGIN{printf \"%d\", $DUR + $TIMEOUT_BUFFER}")
log "audio duration: ${DUR}s  pipeline timeout: ${TIMEOUT}s"

if ! docker compose ps janus 2>/dev/null | grep -qE 'Up|running'; then
  log "Janus not running — starting it"
  docker compose up -d janus
  sleep 8
fi

# Auto-setup Qwen2.5-7B-Instruct (download + profile if missing)
if [ ! -d models/qwen2.5-7b-instruct ]; then
  log "Downloading Qwen2.5-7B-Instruct (~16 GB BF16, ~10 min)"
  ./tools/fetch_models.sh Qwen/Qwen2.5-7B-Instruct qwen2.5-7b-instruct 2>&1 | tee -a "$MLOG"
fi

if [ ! -f pipelines/summarizer/vllm-qwen2.5-7b-instruct.json ]; then
  log "Creating Qwen2.5-7B-Instruct profile"
  cat > pipelines/summarizer/vllm-qwen2.5-7b-instruct.json <<'PROFILE'
{
  "name": "summarizer",
  "type": "proc",
  "mark": "summarizer_vllm",
  "configuration": {
    "model_name": "./models/qwen2.5-7b-instruct",
    "prompt_template_file": "summarize_prompt.txt",
    "dtype": "bfloat16",
    "gpu_memory_utilization": 0.85,
    "max_model_len": 2048,
    "max_tokens": 256,
    "temperature": 0.3,
    "top_p": 0.9,
    "repetition_penalty": 1.05,
    "enforce_eager": false
  }
}
PROFILE
fi

# --- Helpers ---

# Wait for GPU memory to drop below 500 MiB (5 min max). Used between phases.
wait_gpu_free() {
  log "waiting for GPU to be free..."
  for _ in $(seq 1 60); do
    local used
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
      | head -1 | tr -d ' ')
    if [ -n "$used" ] && [ "$used" -lt 500 ]; then
      log "GPU free (${used} MiB)"; return 0
    fi
    sleep 5
  done
  log "WARN GPU still busy after 5 min — proceeding"
}

# Run one pipeline: launch juturna with --auto/--timeout, stream audio, wait for autostop.
run_pipeline() {
  local profile="$1"
  local label="$2"
  local outdir="results/${label}/${WINDOW}"

  log "=== ${label}: pipeline run ==="

  if [ -d "$outdir" ]; then
    local archive="results/${label}_pre3model_${STAMP}"
    log "archiving existing results: $outdir -> $archive"
    mv "$outdir" "$archive"
  fi

  local assem="tmp/3model-${profile}-${WINDOW}s.json"
  uv run python tools/assemble_config.py "$WINDOW" "$profile" "$assem" >>"$MLOG" 2>&1

  local plog="tmp/${label}_3model_${STAMP}.pipelog"
  : > "$plog"

  log "launching juturna (timeout=${TIMEOUT}s, autostop after stream)"
  uv run python -m juturna launch -c "$assem" --auto --timeout "$TIMEOUT" > "$plog" 2>&1 &
  local pid=$!

  # wait for "pipe started" (up to 8 min warmup)
  local started=0
  for _ in $(seq 1 480); do
    if grep -q "pipe started" "$plog"; then started=1; break; fi
    if ! kill -0 "$pid" 2>/dev/null; then
      log "ERROR juturna died during warmup for ${label} (see $plog)"
      return 1
    fi
    sleep 1
  done
  if [ "$started" -ne 1 ]; then
    log "ERROR ${label} did not reach 'pipe started' in 8 min"
    kill "$pid" 2>/dev/null || true
    return 1
  fi
  sleep 3

  log "${label}: pipeline up — streaming ~${DUR}s real-time"
  uv run python tools/send_audio.py "$AUDIO" >>"$plog" 2>&1 \
    || log "WARN ${label}: send_audio returned nonzero"

  log "${label}: stream done, waiting for pipeline auto-stop"
  wait "$pid" 2>/dev/null

  if [ ! -d "$outdir" ]; then
    log "ERROR ${label}: no results dir at $outdir"
    return 1
  fi
  local nwin
  nwin=$(ls "$outdir"/window_*.json 2>/dev/null | wc -l)
  log "${label}: ${nwin} windows produced"
  if [ "$nwin" -lt 5 ]; then
    log "WARN ${label}: only ${nwin} windows — pipeline may have truncated"
  fi
}

# Multi-judge a result dir (Mistral + Phi-4 + Gemma-3 sequentially).
multi_judge() {
  local label="$1"
  local outdir="results/${label}/${WINDOW}"

  if [ ! -d "$outdir" ]; then
    log "SKIP multi-judge ${label}: no results at $outdir"
    return 1
  fi
  log "=== ${label}: multi-judge (${JUDGE_PROFILES[*]}) ==="
  uv run python tools/eval_multi_judge.py "$outdir/" \
    --judge-profiles "${JUDGE_PROFILES[@]}" \
    --audio "$AUDIO" >>"$MLOG" 2>&1 \
    || log "WARN ${label}: multi-judge command returned nonzero"
  if [ -f "$outdir/judge_report_multi.txt" ]; then
    log "${label}: multi-judge done -> $outdir/judge_report_multi.txt"
  else
    log "WARN ${label}: no judge_report_multi.txt produced"
  fi
}

# --- PHASE 1: pipeline runs ---
for spec in "${PIPELINE_RUNS[@]}"; do
  profile="${spec%%:*}"
  label="${spec##*:}"
  if [ ! -f "pipelines/summarizer/${profile}.json" ]; then
    log "SKIP ${label}: profile pipelines/summarizer/${profile}.json missing"
    continue
  fi
  run_pipeline "$profile" "$label" || log "WARN ${label} pipeline failed; continuing"
  wait_gpu_free
done

# --- PHASE 2: multi-judge all targets ---
for label in "${JUDGE_TARGETS[@]}"; do
  multi_judge "$label"
  wait_gpu_free
done

# --- PHASE 3: summary ---
{
  echo "=================================================="
  echo "  3-MODEL MULTI-JUDGE BATTERY SUMMARY  (${STAMP})"
  echo "=================================================="
  echo
  for label in "${JUDGE_TARGETS[@]}"; do
    echo "## ${label}"
    if [ -f "results/${label}/${WINDOW}/judge_report_multi.txt" ]; then
      cat "results/${label}/${WINDOW}/judge_report_multi.txt"
    else
      echo "  (no judge_report_multi.txt — run failed or skipped)"
    fi
    echo
  done
  echo "=================================================="
  echo "Master log: $MLOG"
} | tee "$SUMMARY"

log "=== DONE ==="
log "Summary: $SUMMARY"
log "Master log: $MLOG"
