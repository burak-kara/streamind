#!/usr/bin/env bash
#
# compare_summarizers.sh — long-running summarizer bake-off on a single audio file.
#
# For each summarizer profile given on the command line: assemble the pipeline,
# stream the audio through Janus in real time, auto-stop, then score the windows
# with the FIXED cross-family multi-judge panel. Emits a leaderboard at the end.
#
# Judges are fixed (CLAUDE.md "Current Model Choices"): Mistral-Small-24B-AWQ,
# Phi-4-AWQ, Gemma-3-27B-it-int4-AWQ. Summarizers are the iterated variable.
#
# Usage:
#   ./tools/compare_summarizers.sh --audio <file> --profiles <p1> [p2 ...] [options]
#
# Required:
#   --audio <file>        any ffmpeg-decodable audio (wav/opus/mp3/m4a)
#   --profiles <p...>     one or more summarizer profile names under
#                         pipelines/summarizer/ (without .json), iterated in order
#
# Options:
#   -w, --window <sec>          window duration        (default 300, submission window)
#   -r, --runs <n>              runs per profile       (default 1; >1 measures variance)
#   -b, --timeout-buffer <sec>  seconds past audio     (default 180; warmup + flush)
#   -c, --min-coverage <frac>   min first-run coverage (default 0.85; abort if truncated)
#   -h, --help                  show this help
#
# Example:
#   ./tools/compare_summarizers.sh \
#       --audio datasets/rev16/oneill/audio.opus \
#       --profiles vllm-qwen3-4b-2507 vllm-qwen3.5-4b
#
# Safe to leave running overnight. All artifacts land under results/.
set -uo pipefail
cd "$(dirname "$0")/.."

# Shared launch-prep (runtime env + model preflight) — same lib run_pipeline.sh uses.
source tools/lib/pipeline_prep.sh

# PYTHONUNBUFFERED so juturna's "pipe started" line reaches the log we poll
# promptly. (vLLM/CUBLAS runtime env is set later via prep_runtime_env.)
export PYTHONUNBUFFERED=1

# --- fixed config ---
JUDGE_PROFILES=(
  vllm-mistral-small-24b-awq
  vllm-phi-4-awq
  vllm-gemma3-27b-it-int4-awq
)

# --- defaults (override via flags) ---
WINDOW=300
RUNS=1
TIMEOUT_BUFFER=180
MIN_COV=0.85
AUDIO=""
PROFILES=()

# Print the leading comment block (shebang excluded) as help.
usage() {
  awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' "$0"
  exit "${1:-0}"
}

while [ $# -gt 0 ]; do
  case "$1" in
    --audio)             AUDIO="${2:-}"; shift 2 ;;
    --profiles)
      shift
      while [ $# -gt 0 ] && [[ "$1" != -* ]]; do PROFILES+=("$1"); shift; done
      ;;
    -w|--window)         WINDOW="${2:-}"; shift 2 ;;
    -r|--runs)           RUNS="${2:-}"; shift 2 ;;
    -b|--timeout-buffer) TIMEOUT_BUFFER="${2:-}"; shift 2 ;;
    -c|--min-coverage)   MIN_COV="${2:-}"; shift 2 ;;
    -h|--help)           usage 0 ;;
    --)                  shift; break ;;
    -*)  echo "ERROR: unknown flag $1" >&2; usage 1 ;;
    *)   echo "ERROR: unexpected arg '$1' (pass audio via --audio, models via --profiles)" >&2; usage 1 ;;
  esac
done

[ -n "$AUDIO" ] || { echo "ERROR: --audio <file> is required" >&2; usage 1; }
[ "${#PROFILES[@]}" -ge 1 ] || { echo "ERROR: --profiles needs >=1 summarizer profile" >&2; usage 1; }

# --- logging / output layout ---
STAMP=$(date +%Y%m%d_%H%M%S)
# Audio label -> short slug. Fixtures are <episode>/audio.opus, so the filename
# alone ("audio") is identical across episodes — fold in the parent dir for
# identity. Then keep alnum, collapse every other run of chars to a single '_',
# and cap 32 so the directory name stays readable instead of a wall of
# underscores. (Episode names are long and punctuation-heavy.)
_fname=$(basename "$AUDIO"); _fname=${_fname%.*}
_pdir=$(basename "$(dirname "$AUDIO")")
LABEL=$(printf '%s_%s' "$_pdir" "$_fname" | tr -c 'A-Za-z0-9' '_' | tr -s '_' | sed 's/^_//; s/_$//' | cut -c1-32)
LABEL=${LABEL:-audio}
# Output tree: results/<audio_label>/<stamp>/<model>/<window>/run<K>/ . The
# <stamp> is unique per invocation, so repeat invocations never clobber each
# other and nothing is deleted. Leaderboard, master log, and per-run pipeline
# logs live at the <stamp> root alongside the model dirs.
COMPARE_DIR="results/${LABEL}/${STAMP}"
mkdir -p tmp "$COMPARE_DIR/logs"
SUMMARY="$COMPARE_DIR/leaderboard.txt"
MLOG="$COMPARE_DIR/run.log"
# log to stderr so $(run_pipeline ...) captures only the result dir it echoes.
log()  { echo "[$(date '+%F %T')] $*" | tee -a "$MLOG" >&2; }
die()  { log "FATAL: $*"; exit 1; }

# --- preflight (fail fast before any multi-hour run) ---
command -v ffprobe >/dev/null || die "ffprobe not found (need ffmpeg)"
require_gpu || die "no CUDA GPU (see above)"
[ -f "$AUDIO" ] || die "audio not found: $AUDIO"

for p in "${PROFILES[@]}"; do
  pf="pipelines/summarizer/$p.json"
  [ -f "$pf" ] || die "summarizer profile missing: $pf"
  require_model "$pf" summarizer || die "summarizer weights missing for $p (see above)"
done
for j in "${JUDGE_PROFILES[@]}"; do
  jf="pipelines/judge/$j.json"
  [ -f "$jf" ] || die "judge profile missing: $jf"
  require_model "$jf" judge || die "judge weights missing for $j (see above)"
done

DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$AUDIO")
[ -n "$DUR" ] || die "could not read duration of $AUDIO"
TIMEOUT=$(awk "BEGIN{printf \"%d\", $DUR + $TIMEOUT_BUFFER}")

if ! docker compose ps janus 2>/dev/null | grep -qE 'Up|running'; then
  log "Janus not running — starting it"
  docker compose up -d janus || die "failed to start Janus"
  sleep 8
fi

log "=== summarizer comparison ==="
log "audio=$AUDIO dur=${DUR}s timeout=${TIMEOUT}s window=${WINDOW}s runs=$RUNS"
log "summarizers: ${PROFILES[*]}"
log "judges (fixed): ${JUDGE_PROFILES[*]}"

# --- helpers ---

# Block until GPU memory drops below 500 MiB (5 min cap). Used between phases so
# the next model/judge loads into a clean card.
wait_gpu_free() {
  log "waiting for GPU to free..."
  for _ in $(seq 1 60); do
    local used
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    if [ -n "$used" ] && [ "$used" -lt 500 ]; then log "GPU free (${used} MiB)"; return 0; fi
    sleep 5
  done
  log "WARN GPU still busy after 5 min — proceeding"
}

# Run one pipeline; echoes the final result dir on success, nothing on failure.
run_pipeline() {
  local profile="$1" run="$2" model_dir="$3"
  # Pipeline always writes to this fixed dir (model + window, not run-aware).
  local src="results/${model_dir}/${WINDOW}"
  # Structured destination: <audio_label>/<stamp>/<model>/<window>/run<K>. The
  # run index K is per-model (assigned by the caller), so two profiles sharing
  # one model_dir — same weights, different prompt — land in run1, run2 and never
  # collide. Caller records which profile produced each run for the leaderboard.
  local dst="${COMPARE_DIR}/${model_dir}/${WINDOW}/run${run}"
  local assem="tmp/compare-${profile}-${WINDOW}s.json"
  local plog="${COMPARE_DIR}/logs/${profile}_run${run}.pipelog"

  log "=== ${profile} run ${run}: pipeline ==="
  uv run python tools/assemble_config.py "$WINDOW" "$profile" "$assem" >>"$MLOG" 2>&1 \
    || { log "ERROR assemble_config failed for $profile"; return 1; }
  rm -rf "$src"; : > "$plog"

  uv run python -m juturna launch -c "$assem" --auto --timeout "$TIMEOUT" > "$plog" 2>&1 &
  local pid=$!

  # wait for warmup -> "pipe started" (8 min cap), bail if juturna dies
  local started=0
  for _ in $(seq 1 480); do
    grep -q "pipe started" "$plog" && { started=1; break; }
    kill -0 "$pid" 2>/dev/null || { log "ERROR juturna died in warmup ($profile) — see $plog"; return 1; }
    sleep 1
  done
  [ "$started" -eq 1 ] || { log "ERROR ${profile}: no 'pipe started' in 8 min"; kill "$pid" 2>/dev/null; return 1; }
  sleep 3

  log "${profile}: streaming ~${DUR}s real-time"
  uv run python tools/send_audio.py "$AUDIO" >>"$plog" 2>&1 || log "WARN send_audio nonzero ($profile)"
  log "${profile}: stream done — waiting for auto-stop"
  wait "$pid" 2>/dev/null

  [ -d "$src" ] || { log "ERROR ${profile}: no results at $src"; return 1; }
  mkdir -p "$(dirname "$dst")"; rm -rf "$dst"; mv "$src" "$dst"
  echo "$dst"
}

# coverage = max(window 'to') / duration ; prints a 0..1 float
coverage_of() {
  python3 - "$1" "$DUR" <<'PY'
import sys, glob, json
d, dur = sys.argv[1], float(sys.argv[2])
tos = []
for f in glob.glob(f"{d}/window_*.json"):
    try: tos.append(json.load(open(f)).get("to", 0))
    except Exception: pass
print(f"{(max(tos)/dur if tos and dur else 0):.2f}")
PY
}

multi_judge() {
  local dst="$1" profile="$2"
  log "=== ${profile}: multi-judge (${JUDGE_PROFILES[*]}) ==="
  uv run python tools/eval_multi_judge.py "$dst/" \
    --judge-profiles "${JUDGE_PROFILES[@]}" \
    --audio "$AUDIO" >>"$MLOG" 2>&1 \
    || log "WARN ${profile}: multi-judge returned nonzero"
  [ -f "$dst/judge_report_multi.txt" ] \
    && log "${profile}: judged -> $dst/judge_report_multi.txt" \
    || log "WARN ${profile}: no judge_report_multi.txt produced"
}

# Runtime env (FlashInfer off + CUBLAS on LD_LIBRARY_PATH for ASR) — must be set
# before any juturna launch. Same call run_pipeline.sh makes.
prep_runtime_env

# --- run comparison ---
declare -a DONE_DIRS=()
declare -A RUN_IDX=()       # per-model_dir run counter -> contiguous run1,run2…
declare -A DIR_PROFILE=()   # result dir -> profile (model dir alone can't tell
                            # apart two prompt variants of the same weights)
first_done=0
for profile in "${PROFILES[@]}"; do
  model_dir=$(model_dir_of "pipelines/summarizer/$profile.json")
  for ((N=1; N<=RUNS; N++)); do
    K=$(( ${RUN_IDX[$model_dir]:-0} + 1 )); RUN_IDX[$model_dir]=$K
    if dst=$(run_pipeline "$profile" "$K" "$model_dir"); then
      DIR_PROFILE["$dst"]=$profile
      cov=$(coverage_of "$dst")
      nwin=$(ls "$dst"/window_*.json 2>/dev/null | wc -l | tr -d ' ')
      log "${profile} -> ${model_dir}/${WINDOW}/run${K}: ${nwin} windows, coverage ${cov}"
      if [ "$first_done" -eq 0 ]; then
        first_done=1
        if awk "BEGIN{exit !($cov < $MIN_COV)}"; then
          die "first-run coverage ${cov} < ${MIN_COV} — audio truncated. Fix send_audio/Janus before a long comparison run."
        fi
        log "first-run coverage ${cov} OK — continuing"
      fi
      wait_gpu_free
      multi_judge "$dst" "$profile"
      DONE_DIRS+=("$dst")
      wait_gpu_free
    else
      log "WARN ${profile} run ${N} failed; continuing"
      [ "$first_done" -eq 0 ] && die "first run produced nothing — aborting before wasting hours."
    fi
  done
done

# --- summary / leaderboard ---
{
  echo "=================================================="
  echo "  SUMMARIZER COMPARISON — ${LABEL}  (${STAMP})"
  echo "  window=${WINDOW}s  judges=${JUDGE_PROFILES[*]}"
  echo "=================================================="
  echo
  echo "## Leaderboard (final source score, +Janus — higher is better)"
  for d in "${DONE_DIRS[@]}"; do
    rpt="$d/judge_report_multi.txt"
    score=$(grep -iE 'final source score' "$rpt" 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+' | tail -1)
    # label = profile @ <model>/<window>/run<K> (relative to the stamp root)
    printf '%s\t%s\t%s\n' "${score:-0}" "${DIR_PROFILE[$d]:-?}" "${d#"$COMPARE_DIR"/}"
  done | sort -rn | awk -F'\t' '{printf "  %-28s %-32s %s\n", $2, $3, ($1=="0"?"N/A":$1)}'
  echo
  for d in "${DONE_DIRS[@]}"; do
    echo "## ${DIR_PROFILE[$d]:-?}  (${d#"$COMPARE_DIR"/})"
    [ -f "$d/judge_report_multi.txt" ] && cat "$d/judge_report_multi.txt" || echo "  (no report)"
    echo
  done
  echo "Master log: $MLOG"
} | tee "$SUMMARY"

log "=== DONE -> $SUMMARY ==="
