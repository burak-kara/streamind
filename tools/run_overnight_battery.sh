#!/usr/bin/env bash
# Overnight model comparison. Uses juturna's --auto/--timeout (no interactive
# prompts) and the FlashInfer-sampler workaround run_pipeline.sh relies on.
# Per (profile x run): launch pipeline, stream recording real-time, auto-stop,
# label results, judge, append CSV. Aborts if the FIRST run truncates.
set -uo pipefail
cd "$(dirname "$0")/.."

# --- env that the pipeline needs (mirrors run_pipeline.sh) ---
export VLLM_USE_FLASHINFER_SAMPLER=0   # box has CUDA runtime but no nvcc
export PYTHONUNBUFFERED=1              # so we can detect "pipe started" in the log

# ---- config (edit these) ----
AUDIO="docs/datasets/rev16/26_Episode_338_-_Special_Guest_Rob_O'Neill:_The_Man_Who_Killed_Osama_Bin_Laden.opus"
LABEL=oneill
WINDOW=300
RUNS=2
JUDGE=vllm-mistral-small-24b-awq
PROFILES=(vllm-qwen3-4b-2507 vllm-qwen3.5-4b vllm-llama-3.1-8b)
MIN_COV=0.85
TIMEOUT_BUFFER=120     # extra seconds past audio length (warmup + final flush)
# -----------------------------

mkdir -p tmp results
STAMP=$(date +%Y%m%d_%H%M%S)
CSV="results/battery_${LABEL}_${STAMP}.csv"; MLOG="results/battery_${LABEL}_${STAMP}.log"
echo "model,run,B,K,L,C,final,n_windows,coverage" > "$CSV"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$MLOG"; }

DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$AUDIO")
TIMEOUT=$(awk "BEGIN{printf \"%d\", $DUR + $TIMEOUT_BUFFER}")
log "audio=$AUDIO dur=${DUR}s timeout=${TIMEOUT}s profiles=${PROFILES[*]} runs=$RUNS"

first_done=0
for PROFILE in "${PROFILES[@]}"; do
  PF="pipelines/summarizer/${PROFILE}.json"
  [ -f "$PF" ] || { log "MISSING $PF — skip"; continue; }
  MPATH=$(python3 -c "import json;print(json.load(open('$PF'))['configuration']['model_name'])")
  SAN=$(echo "$MPATH" | tr '/:' '_-')
  for ((N=1;N<=RUNS;N++)); do
    log "=== $PROFILE run $N ==="
    ASSEM="tmp/battery-${PROFILE}-${WINDOW}s.json"
    uv run python tools/assemble_config.py "$WINDOW" "$PROFILE" "$ASSEM" >>"$MLOG" 2>&1
    rm -rf "results/${SAN}/${WINDOW}"
    PLOG="tmp/${SAN}_${LABEL}_run${N}.pipelog"; : > "$PLOG"
    uv run python -m juturna launch -c "$ASSEM" --auto --timeout "$TIMEOUT" > "$PLOG" 2>&1 &
    PID=$!
    # wait until the pipeline is actually running (auto-started after warmup)
    for _ in $(seq 1 240); do grep -q "pipe started" "$PLOG" && break; sleep 1; done
    sleep 2
    log "pipeline up — streaming (~${DUR}s real-time)…"
    uv run python tools/send_audio.py "$AUDIO" >>"$PLOG" 2>&1 || log "WARN send_audio returned nonzero"
    log "stream finished — waiting for pipeline auto-stop"
    wait "$PID" 2>/dev/null
    SRC="results/${SAN}/${WINDOW}"; DST="results/${SAN}/${WINDOW}_${LABEL}_run${N}"
    if [ ! -d "$SRC" ]; then
      log "WARN no results ($PROFILE run $N) — see $PLOG"
      [ "$first_done" -eq 0 ] && { log "ABORT: first run produced nothing."; exit 1; }
      continue
    fi
    rm -rf "$DST"; mv "$SRC" "$DST"
    log "judging $DST"
    uv run python tools/eval_quality.py "$DST" --judge-profile "$JUDGE" >>"$PLOG" 2>&1
    LINE=$(python3 - "$DST" "$DUR" <<'PY'
import sys,re,glob,json
d,dur=sys.argv[1],float(sys.argv[2])
try: txt=open(f"{d}/judge_report.txt").read()
except FileNotFoundError: txt=""
avg=next((l for l in txt.splitlines() if l.split()[:1]==["avg"]),"")
nums=re.findall(r"-?\d+\.?\d*",avg); b,k,l,c=(nums+["","","",""])[:4]
m=re.search(r"Final source score.*?=\s*([\d.]+)",txt); final=m.group(1) if m else c
tos=[json.load(open(f)).get("to",0) for f in glob.glob(f"{d}/window_*.json")]
cov=(max(tos)/dur) if (tos and dur) else 0
print(f"{b},{k},{l},{c},{final},{len(tos)},{cov:.2f}")
PY
)
    echo "$PROFILE,$N,$LINE" >> "$CSV"; log "-> $PROFILE run $N: $LINE"
    if [ "$first_done" -eq 0 ]; then
      first_done=1; COV=$(echo "$LINE" | awk -F, '{print $NF}')
      awk "BEGIN{exit !($COV < $MIN_COV)}" && { log "ABORT: first-run coverage $COV < $MIN_COV (truncated). Fix send_audio first."; exit 1; }
      log "first-run coverage $COV OK — continuing"
    fi
  done
done
log "DONE -> $CSV"; column -s, -t "$CSV" | tee -a "$MLOG"
