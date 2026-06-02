#!/usr/bin/env bash
#
# compare_summarizers.sh — long-running summarizer bake-off over one or more audio files.
#
# For each audio file × each summarizer profile: assemble the pipeline, stream
# the audio through Janus in real time, auto-stop, then score the windows with
# the FIXED cross-family multi-judge panel. Emits one leaderboard per audio.
#
# Judges are fixed (CLAUDE.md "Current Model Choices"): Mistral-Small-24B-AWQ,
# Phi-4-AWQ, Gemma-3-27B-it-int4-AWQ. Summarizers and audios are the iterated
# variables.
#
# Usage:
#   ./tools/compare_summarizers.sh --audios <a1> [a2 ...] --profiles <p1> [p2 ...] [options]
#
# Required:
#   --audios <a...>       one or more ffmpeg-decodable audio files (wav/opus/mp3/m4a),
#                         iterated in order; each gets its own results tree
#   --profiles <p...>     one or more summarizer profile names under
#                         pipelines/summarizer/ (without .json), iterated in order
#
# Options:
#   -w, --window <sec>          window duration        (default 300, submission window)
#   -r, --runs <n>              runs per profile       (default 1; >1 measures variance)
#   -b, --timeout-buffer <sec>  seconds past audio     (default 180; warmup + flush)
#   -c, --min-coverage <frac>   min first-run coverage (default 0.85; abort if truncated)
#   -d, --max-duration <sec>    cap audio streaming    (default 0 = no cap; e.g. 1800 = 30 min)
#   -h, --help                  show this help
#
# Example:
#   ./tools/compare_summarizers.sh \
#       --audios datasets/rev16/*/audio.opus \
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
MAX_DUR=0
AUDIOS=()
PROFILES=()

# Print the leading comment block (shebang excluded) as help.
usage() {
  awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' "$0"
  exit "${1:-0}"
}

while [ $# -gt 0 ]; do
  case "$1" in
    --audios)
      shift
      while [ $# -gt 0 ] && [[ "$1" != -* ]]; do AUDIOS+=("$1"); shift; done
      ;;
    --profiles)
      shift
      while [ $# -gt 0 ] && [[ "$1" != -* ]]; do PROFILES+=("$1"); shift; done
      ;;
    -w|--window)         WINDOW="${2:-}"; shift 2 ;;
    -r|--runs)           RUNS="${2:-}"; shift 2 ;;
    -b|--timeout-buffer) TIMEOUT_BUFFER="${2:-}"; shift 2 ;;
    -c|--min-coverage)   MIN_COV="${2:-}"; shift 2 ;;
    -d|--max-duration)   MAX_DUR="${2:-}"; shift 2 ;;
    -h|--help)           usage 0 ;;
    --)                  shift; break ;;
    -*)  echo "ERROR: unknown flag $1" >&2; usage 1 ;;
    *)   echo "ERROR: unexpected arg '$1' (pass audio via --audios, models via --profiles)" >&2; usage 1 ;;
  esac
done

[ "${#AUDIOS[@]}"   -ge 1 ] || { echo "ERROR: --audios needs >=1 audio file" >&2; usage 1; }
[ "${#PROFILES[@]}" -ge 1 ] || { echo "ERROR: --profiles needs >=1 summarizer profile" >&2; usage 1; }

# --- global logger ---
# Spans the whole run (all audios). Inside the per-audio loop MLOG is repointed
# into that audio's own run.log; here it captures shared preflight + the final
# index of every leaderboard produced.
GSTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p tmp results
MLOG="results/compare_${GSTAMP}.log"
log()  { echo "[$(date '+%F %T')] $*" | tee -a "$MLOG" >&2; }
die()  { log "FATAL: $*"; exit 1; }

# --- pipeline process-group teardown ---
# A juturna launch is a tree: uv -> python -> vLLM EngineCore worker(s). Killing
# only the top pid (uv) leaves the python + vLLM workers orphaned (reparented to
# init), still holding VRAM — `ps` shows nothing, yet `nvidia-smi` reports memory
# used. So we start each launch in its OWN process group (set -m makes the '&'
# job a group leader: PGID == its pid) and signal the WHOLE group on stop /
# Ctrl-C / any exit. Negative pid = process group in kill(1).
PIPE_PID=""   # PGID of the running launch; empty when none in flight

stop_pipeline() {
  [ -n "$PIPE_PID" ] || return 0
  local pgid="$PIPE_PID"; PIPE_PID=""
  kill -TERM -"$pgid" 2>/dev/null || return 0   # group already gone
  for _ in $(seq 1 15); do kill -0 -"$pgid" 2>/dev/null || return 0; sleep 1; done
  log "pipeline group $pgid ignored TERM — sending KILL"
  kill -KILL -"$pgid" 2>/dev/null
}

# EXIT covers normal end + die(); INT/TERM cover Ctrl-C / kill of the script.
trap 'stop_pipeline' EXIT
trap 'stop_pipeline; exit 130' INT TERM

# --- preflight (fail fast before any multi-hour run) ---
# Audio-independent checks run once; weights/GPU don't change between audios.
command -v ffprobe >/dev/null || die "ffprobe not found (need ffmpeg)"
require_gpu || die "no CUDA GPU (see above)"
for a in "${AUDIOS[@]}"; do
  [ -f "$a" ] || die "audio not found: $a"
done
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

if ! docker compose ps janus 2>/dev/null | grep -qE 'Up|running'; then
  log "Janus not running — starting it"
  docker compose up -d janus || die "failed to start Janus"
  sleep 8
fi

log "=== summarizer comparison: ${#AUDIOS[@]} audio(s) × ${#PROFILES[@]} profile(s) ==="
log "audios:         ${AUDIOS[*]}"
log "summarizers:    ${PROFILES[*]}"
log "judges (fixed): ${JUDGE_PROFILES[*]}"
[ "${MAX_DUR:-0}" -gt 0 ] && log "max-duration:   ${MAX_DUR}s per audio"

# --- helpers ---
# These operate on per-audio globals set by the outer loop below: AUDIO, DUR,
# TIMEOUT, LABEL, COMPARE_DIR, WINDOW, MLOG.

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
  # Steer the transmitter into a per-run staging dir UNDER this audio's tree (it
  # appends <model_dir>/<window> itself), so nothing is ever written to the repo
  # root results/<model>/<window>. Cleaner than the old shared-path + mv, and a
  # killed run can't strand windows at the root.
  local stage="${COMPARE_DIR}/.stage/${profile}-run${run}"
  local produced="${stage}/${model_dir}/${WINDOW}"
  # Structured destination: <audio_label>/<stamp>/<model>/<window>/run<K>. The
  # run index K is per-model (assigned by the caller), so two profiles sharing
  # one model_dir — same weights, different prompt — land in run1, run2 and never
  # collide. Caller records which profile produced each run for the leaderboard.
  local dst="${COMPARE_DIR}/${model_dir}/${WINDOW}/run${run}"
  local assem="tmp/compare-${profile}-${WINDOW}s.json"
  local plog="${COMPARE_DIR}/logs/${profile}_run${run}.pipelog"

  log "=== ${profile} run ${run}: pipeline ==="
  uv run python tools/assemble_config.py "$WINDOW" "$profile" "$assem" \
    --results-dir "$stage" >>"$MLOG" 2>&1 \
    || { log "ERROR assemble_config failed for $profile"; return 1; }
  rm -rf "$stage"; : > "$plog"

  # Launch in its own process group (PGID == pid) so stop_pipeline can tear down
  # the whole uv -> python -> vLLM tree and reclaim VRAM. set -m flips on job
  # control just for this fork; we flip it back to avoid the [1]+ Done spam.
  set -m
  uv run python -m juturna launch -c "$assem" --auto --timeout "$TIMEOUT" > "$plog" 2>&1 &
  local pid=$!
  set +m
  PIPE_PID=$pid

  # wait for warmup -> "pipe started" (8 min cap), bail if juturna dies
  local started=0
  for _ in $(seq 1 480); do
    grep -q "pipe started" "$plog" && { started=1; break; }
    kill -0 "$pid" 2>/dev/null || { log "ERROR juturna died in warmup ($profile) — see $plog"; stop_pipeline; return 1; }
    sleep 1
  done
  [ "$started" -eq 1 ] || { log "ERROR ${profile}: no 'pipe started' in 8 min"; stop_pipeline; return 1; }
  sleep 3

  log "${profile}: streaming ~${DUR}s real-time"
  _send_args=("$AUDIO")
  [ "${MAX_DUR:-0}" -gt 0 ] && _send_args+=(--max-duration "$MAX_DUR")
  uv run python tools/send_audio.py "${_send_args[@]}" >>"$plog" 2>&1 || log "WARN send_audio nonzero ($profile)"
  log "${profile}: stream done — waiting for auto-stop"
  wait "$pid" 2>/dev/null
  # Even on a clean auto-stop, vLLM's in-process unload is leaky — an EngineCore
  # worker can linger in the group, holding the CUDA context. The NEXT model then
  # dies at init_device ("CUDA driver initialization failed"). Sweep the whole
  # group so each model starts on a truly clean GPU (no-op if already gone).
  stop_pipeline

  [ -d "$produced" ] || { log "ERROR ${profile}: no results at $produced"; rm -rf "$stage"; return 1; }
  mkdir -p "$(dirname "$dst")"; rm -rf "$dst"; mv "$produced" "$dst"; rm -rf "$stage"
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

# --- run comparison: outer loop over audios ---
declare -a ALL_SUMMARIES=()
for AUDIO in "${AUDIOS[@]}"; do
  # Per-audio output tree + logger. Each audio gets its own <stamp> so trees are
  # independent (results/<audio_label>/<stamp>/…) and never clobber.
  STAMP=$(date +%Y%m%d_%H%M%S)
  # Audio label -> short slug. Fixtures are <episode>/audio.opus, so the filename
  # alone ("audio") is identical across episodes — fold in the parent dir for
  # identity. Then keep alnum, collapse every other run of chars to a single '_',
  # and cap 32 so the directory name stays readable.
  _fname=$(basename "$AUDIO"); _fname=${_fname%.*}
  _pdir=$(basename "$(dirname "$AUDIO")")
  LABEL=$(printf '%s_%s' "$_pdir" "$_fname" | tr -c 'A-Za-z0-9' '_' | tr -s '_' | sed 's/^_//; s/_$//' | cut -c1-32)
  LABEL=${LABEL:-audio}
  COMPARE_DIR="results/${LABEL}/${STAMP}"
  mkdir -p "$COMPARE_DIR/logs"
  SUMMARY="$COMPARE_DIR/leaderboard.txt"
  CSV="$COMPARE_DIR/leaderboard.csv"
  MLOG="$COMPARE_DIR/run.log"

  DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$AUDIO")
  [ -n "$DUR" ] || die "could not read duration of $AUDIO"
  if [ "${MAX_DUR:-0}" -gt 0 ] && awk "BEGIN{exit !($DUR > $MAX_DUR)}"; then
    log "audio ${DUR}s exceeds --max-duration ${MAX_DUR}s — capping"
    DUR=$MAX_DUR
  fi
  TIMEOUT=$(awk "BEGIN{printf \"%d\", $DUR + $TIMEOUT_BUFFER}")

  log "=== audio: $AUDIO ==="
  log "dur=${DUR}s timeout=${TIMEOUT}s window=${WINDOW}s runs=$RUNS -> $COMPARE_DIR"

  # Per-audio state — reset each iteration.
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

  # --- per-audio summary / leaderboard ---
  # Build the sorted leaderboard rows ONCE as TSV:
  #   final_score <TAB> avg_bk <TAB> avg_l <TAB> profile <TAB> run
  # (descending by final_score), then render both the human table (SUMMARY) and
  # the machine table (CSV) from the same data so they can never disagree.
  # Scores come from judge_scores_multi.json (machine-readable, avoids fragile
  # text grep). run = last path component only (e.g. "run2").
  LEADER_ROWS=$(
    for d in "${DONE_DIRS[@]}"; do
      profile="${DIR_PROFILE[$d]:-?}"
      rel="${d#"$COMPARE_DIR"/}"
      run="${rel##*/}"   # last component = run<K>
      json="$d/judge_scores_multi.json"
      if [ -f "$json" ]; then
        _s=$(python3 - "$json" <<'PYEOF'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    c = d.get("consensus", {})
    final = c.get("final_source_score") or 0
    avg_bk = c.get("avg_bk") or 0
    avg_l = c.get("avg_l") or 0
    print(f"{final:.2f}\t{avg_bk:.2f}\t{avg_l:.2f}")
except Exception:
    print("0\t0\t0")
PYEOF
)
        IFS=$'\t' read -r _final _avg_bk _avg_l <<< "$_s"
      else
        _final=0; _avg_bk=0; _avg_l=0
      fi
      printf '%s\t%s\t%s\t%s\t%s\n' \
        "${_final:-0}" "${_avg_bk:-0}" "${_avg_l:-0}" "$profile" "$run"
    done | sort -rn
  )

  # Leaderboard table only -> CSV (rank,profile,run,avg_bk,avg_l,score).
  # Fields carry no commas, so plain unquoted CSV is safe.
  {
    echo "rank,profile,run,avg_bk,avg_l,score"
    printf '%s\n' "$LEADER_ROWS" | awk -F'\t' 'NF{
      bk = ($2=="0" ? "N/A" : $2)
      al = ($3=="0" ? "N/A" : $3)
      sc = ($1=="0" ? "N/A" : $1)
      print NR","$4","$5","bk","al","sc
    }'
  } > "$CSV"

  {
    echo "=================================================="
    echo "  SUMMARIZER COMPARISON — ${LABEL}  (${STAMP})"
    echo "  audio=${AUDIO}"
    echo "  window=${WINDOW}s  judges=${JUDGE_PROFILES[*]}"
    echo "=================================================="
    echo
    echo "## Leaderboard (final = avg B+K + avg L + Janus — higher is better)"
    printf '  %-28s %-8s %7s %6s %7s\n' "profile" "run" "avg_BK" "avg_L" "score"
    printf '%s\n' "$LEADER_ROWS" | awk -F'\t' 'NF{
      bk = ($2=="0" ? "   N/A" : sprintf("%7.2f", $2))
      al = ($3=="0" ? "  N/A"  : sprintf("%6.2f", $3))
      sc = ($1=="0" ? "   N/A" : sprintf("%7.2f", $1))
      printf "  %-28s %-8s %s %s %s\n", $4, $5, bk, al, sc
    }'
    echo
    for d in "${DONE_DIRS[@]}"; do
      echo "## ${DIR_PROFILE[$d]:-?}  (${d#"$COMPARE_DIR"/})"
      [ -f "$d/judge_report_multi.txt" ] && cat "$d/judge_report_multi.txt" || echo "  (no report)"
      echo
    done
    echo "Run log: $MLOG"
  } | tee "$SUMMARY"

  log "=== audio done -> $SUMMARY ==="
  ALL_SUMMARIES+=("$SUMMARY")

  # Rebuild cross-stamp ranking for this audio dir (all stamps seen so far).
  AUDIO_DIR="$(dirname "$COMPARE_DIR")"
  log "Updating cross-stamp leaderboard: $AUDIO_DIR"
  uv run python tools/postproc/rank_audio.py "$AUDIO_DIR" >> "$MLOG" 2>&1 \
    || log "WARN rank_audio.py failed for $AUDIO_DIR (non-fatal)"

  MLOG="results/compare_${GSTAMP}.log"   # restore global logger for the next boundary
done

log "=== ALL DONE: ${#ALL_SUMMARIES[@]} leaderboard(s) ==="
for s in "${ALL_SUMMARIES[@]}"; do log "  $s"; done
