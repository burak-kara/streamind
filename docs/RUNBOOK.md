# STREAMIND Runbook

Operational guide: start services, feed audio, run the pipeline, score output.
Single source of truth for run commands. `CLAUDE.md` references this file.

All full-pipeline work runs on **`uni-lab`** (RTX 4090, 24 GB, CUDA 12.4). vLLM
does not install on Apple Silicon — local mac is unit-tests only (vLLM mocked).
Repo on uni-lab lives at `~/Desktop/streamind`.

```bash
ssh uni-lab
cd ~/Desktop/streamind
git pull
uv sync --extra dev          # vLLM + torch cu124 — Linux/CUDA only
```

---

## 0. One-time setup

Fetch model weights into `./models/<local_name>/` (idempotent; `models/` is
gitignored — never commit weights). Pipeline aborts at warmup if the dir is
missing.

```bash
./tools/fetch_models.sh <hf_id> <local_name>
# e.g. (locked submission summarizer)
./tools/fetch_models.sh cyankiwi/Qwen3.5-4B-AWQ-BF16-INT4 qwen3.5-4b-awq
```

ASR model (`deepdml/faster-whisper-large-v3-turbo-ct2`) auto-resolves to
`./models/faster-whisper-large-v3-turbo` if present, else falls back to HF
download in dev (baked into the Docker image for submission).

---

## 1. Start Janus (audio gateway)

Janus is the WebRTC server that feeds RTP audio to the pipeline. First run
builds the image from source (~15 min).

```bash
docker compose up -d janus          # start
docker compose logs -f janus        # follow logs
docker compose down                 # stop
```

Ports: `8088` (HTTP API), `10000-10050/udp` (WebRTC media).
The pipeline's `audio_rtp` node listens on `0.0.0.0:8888`; Janus RTP-forwards
to that port via `host.docker.internal`.

`docker compose up -d` (no service) also starts the `pipeline` container —
for local dev run the pipeline natively (§3) instead, and only start `janus`.

---

## 2. Run the pipeline

```bash
./tools/run_pipeline.sh -p <summarizer_profile>          # 300s window (default)
./tools/run_pipeline.sh -w 30 -p <summarizer_profile>    # short 30s window
./tools/run_pipeline.sh -p <profile> -a small.en         # ASR override (A/B)

# e.g.
./tools/run_pipeline.sh -p vllm-qwen3.5-4b-awq          # locked submission profile
```

Flags (`tools/run_pipeline.sh`):

| Flag | Meaning | Default |
| ---- | ------- | ------- |
| `-p, --profile <name>` | summarizer profile under `pipelines/summarizer/<name>.json` | **required** |
| `-w, --window <seconds>` | rolling window duration | `300` |
| `-a, --asr <model>` | override ASR `model_name` (e.g. `small.en`, `large-v3-turbo`) | config-base value |
| `-h, --help` | usage | — |

What it does: preflight (`nvidia-smi` + model dir present) → assembles
`config-base.json` + summarizer profile into `./tmp/config-<w>s-<profile>.json`
→ exports CUDA libs + disables FlashInfer JIT → `juturna launch`.

Output JSON written to `results/<stamp>/<model>/<window>/window_*.json` with
challenge keys: `from`, `to`, `summary`, `keywords` (exactly 3), `proc_time`.
Each launch gets its own `<stamp>` (`YYYYMMDD_HHMMSS`) run root, so **reruns never
clobber and nothing is deleted** — prior runs sit side by side. The launcher
prints the exact `Output:` dir; pass that path (or its `<model>/<window>`
subdir) to the judge. Delete old stamp dirs by hand when you no longer need them.

Leave the pipeline running and feed audio from a second session (§ next).

---

## 3. Feed audio

Inject any ffmpeg-decodable file (wav, opus, mp3, m4a) through Janus while the
pipeline runs. Separate terminal / ssh session.

```bash
uv run python tools/send_audio.py <audio_file>
# default rev16 fixture (~36 min → ~7×300s windows):
uv run python tools/send_audio.py 'datasets/rev16/10_Creating_Your_Own_Lane_in_Podcasting_ft_@Favyfav_of_@latinoswholunch/audio.opus'
```

Flags (`tools/send_audio.py`):

| Flag | Meaning | Default |
| ---- | ------- | ------- |
| `audio_path` | input file (positional) | **required** |
| `--janus-url` | Janus HTTP API base | `http://localhost:8088/janus` |
| `--room` | VideoRoom room ID | `1234` |
| `--pipeline-host` | host where `audio_rtp` listens | `host.docker.internal` |
| `--pipeline-port` | UDP port of `audio_rtp` | `8888` |
| `--max-duration <sec>` | cap streaming to N seconds (e.g. `1800` = 30 min); audio beyond this is never sent | no cap |

Heartbeat logs elapsed/total every 60s and surfaces WebRTC state, so mid-stream
truncation is visible without log spelunking. Same path in dev and production:
audio always enters through Janus → `audio_rtp` (a Juturna built-in; there is no
file-source node).

---

## 4. Score the output (offline judge)

Judges run **after** the pipeline exits — the summarizer must be unloaded first
to free VRAM (a single GPU cannot hold summarizer + judge co-resident).

### 4a. Single judge

```bash
uv run python tools/eval_quality.py results/<stamp>/<model>/300/ \
  --judge-profile vllm-<judge_name> \
  --audio 'datasets/rev16/<episode>/audio.opus'
```

`--audio` enables ASR WER and auto-discovers sibling `chunks.json` for
ground-truth comparison.

Flags (`tools/eval_quality.py`):

| Flag | Meaning |
| ---- | ------- |
| `results_dir` | dir with `window_*.json` (recursive); default `./results` |
| `--judge-profile <name>` | profile under `pipelines/judge/<name>.json` (**required**) |
| `--audio <file>` | source audio → enables WER + auto-resolves reference `.txt` |
| `--reference <txt>` | ground-truth transcript, overrides auto-resolution |
| `--ground-truth <chunks.json>` | reference summaries/keywords; auto-discovered from `--audio` dir |
| `--filter <substr>` | only score windows whose path matches |
| `--json-out <path>` | machine-readable summary JSON |
| `--report-out <path>` | score table path; default `<results_dir>/judge_report.txt` |
| `--no-janus-bonus` | drop the +4 Janus bonus from final source score |
| `--judge-tag <label>` | namespace per-window output under `judge/<label>/` (used by multi-judge) |

### 4b. Multi-judge panel (de-bias)

Cross-family panel, run **sequentially** — one subprocess per judge so process
exit fully reclaims VRAM before the next loads (robust vs leaky in-process vLLM
unload). Reports each judge's C side-by-side + consensus mean/stdev; low stdev
corroborates the score, high stdev flags judge bias.

```bash
uv run python tools/eval_multi_judge.py results/<stamp>/<model>/300/ \
  --judge-profiles vllm-mistral-small-24b-awq vllm-phi-4-awq vllm-gemma3-27b-it-int4-awq \
  --audio 'datasets/rev16/<episode>/audio.opus'

# e.g.
uv run python tools/eval_multi_judge.py results/qwen3.5-4b-prompt-tune-v2/300/ \
  --judge-profiles vllm-mistral-small-24b-awq vllm-phi-4-awq vllm-gemma3-27b-it-int4-awq \
  --audio 'datasets/rev16/10_Creating_Your_Own_Lane_in_Podcasting_ft_@Favyfav_of_@latinoswholunch/audio.opus' 

uv run python tools/eval_multi_judge.py results/qwen3.5-4b/300/ --judge-profiles vllm-mistral-small-24b-awq vllm-phi-4-awq vllm-gemma3-27b-it-int4-awq --audio 'datasets/rev16/26_Episode_338_-_Special_Guest_Rob_O'Neill:_The_Man_Who_Killed_Osama_Bin_Laden'
```

Flags (`tools/eval_multi_judge.py`): `--judge-profiles <p...>` (**required**,
run in order), plus passthrough to each child: `--audio`, `--reference`,
`--ground-truth`, `--filter`, `--no-janus-bonus`, and `--tmp-dir` (intermediate
json-out location, default `./tmp`).

Outputs next to the results:

| File | Contents |
| ---- | -------- |
| `judge_report_multi.txt` | combined table: C per judge + mean + stdev + consensus |
| `judge_scores_multi.json` | per-window + per-judge + consensus, machine-readable |
| `judge_report_<profile>.txt` | each judge's own table |
| `judge/<profile>/window_N.json` | per-window judge JSON, namespaced (no clobber) |

Qwen judges are excluded — Qwen is the summarizer family, so a Qwen judge would
self-bias. Current panel: Mistral-Small-24B-AWQ, Phi-4-AWQ, Gemma-3-27B-it-int4-AWQ.

### 4c. Summarizer comparison (unattended bake-off, multi-judge)

`tools/compare_summarizers.sh` chains steps 2–4b for every (audio × summarizer)
pair: assemble → stream → auto-stop → multi-judge → leaderboard. Judges are
**fixed** to the panel above; summarizers and audios are the iterated arguments.
Safe to leave running overnight.

```bash
# bake-off two summarizers on one episode at the 300 s submission window
./tools/compare_summarizers.sh \
  --audios 'datasets/rev16/26_Episode_338_Special_Guest_Rob_O'Neill/audio.opus' \
  --profiles vllm-qwen3.5-4b-awq


./tools/compare_summarizers.sh \
  --runs 2 \
  --audios 'datasets/rev16/10_Creating_Your_Own_Lane_in_Podcasting/audio.opus' 'datasets/rev16/11_Podcast_Tips_From_Berry/audio.opus' 'datasets/rev16/27_What_We_Own_is_Sacred_Because_We_Are_Sacred/audio.opus' \
  --profiles vllm-qwen3.5-4b-awq

# same profiles across every rev16 episode in one unattended run
./tools/compare_summarizers.sh \
  --audios datasets/rev16/*/audio.opus \
  --profiles vllm-qwen3.5-4b-awq
```

Flags: `--audios <a...>` (required, iterated — each audio gets its own results
tree + leaderboard), `--profiles <p...>` (required, iterated),
`-w|--window <sec>` (default 300), `-r|--runs <n>` (default 1),
`-b|--timeout-buffer <sec>` (default 180), `-c|--min-coverage <frac>` (default
0.85, abort if first run truncates), `-d|--max-duration <sec>` (default: no cap
— cap streaming per audio, e.g. `1800` for 30 min; timeout + coverage are
computed against the capped duration). Preflight verifies every audio + GPU + every
summarizer and judge model dir once, before any multi-hour run.

All artifacts land under one per-invocation root
`results/<audio_label>/<stamp>/` (audio_label = parent-dir + filename slug,
capped 32 chars; stamp = `YYYYMMDD_HHMMSS`):

```
results/<audio_label>/<stamp>/
  leaderboard.txt                       # leaderboard + every judge report
  run.log                               # master log
  logs/<profile>_run<K>.pipelog         # per-run pipeline/vLLM log
  <model>/<window>/run<K>/              # that run's windows + judge_report_multi.txt
```

The `run<K>` index is **per-model**, so two profiles sharing one model dir —
same weights, different prompt — land in `run1`, `run2` and never collide. The
leaderboard records which profile produced each `run<K>`. Windows are written
straight into a per-run staging dir under the stamp root (`.stage/`) and moved
into place — nothing is ever written to the repo-root `results/<model>/`.

#### Run unattended over SSH (tmux)

A multi-audio bake-off runs for hours. Run it inside **tmux** so an SSH drop
doesn't kill it (a bare `./tools/... &` dies with the login shell's SIGHUP).

```bash
ssh uni-lab
tmux new -s bakeoff                       # fresh session (or: tmux attach -t bakeoff)
cd ~/Desktop/streamind
./tools/compare_summarizers.sh --runs 3 \
  --audios datasets/rev16/*/audio.opus \
  --profiles vllm-qwen3.5-4b-awq
```

Detach (leaves it running): prefix `Ctrl-b` **then** `d`. Reattach later:
`tmux attach -t bakeoff`. List sessions: `tmux ls`.

If `Ctrl-b d` does nothing, the prefix is remapped or the keys aren't reaching
tmux. Workarounds, in order:

- Detach by command instead of keybinding — from any pane: `tmux detach`
  (or run `tmux detach-client -s bakeoff` from a second SSH session).
- Check the prefix: `tmux show-options -g prefix`. If it's e.g. `C-a`, use
  `Ctrl-a d`.
- Nested tmux (local + remote tmux): press the prefix **twice** to reach the
  inner session — `Ctrl-b Ctrl-b d`.
- No tmux at all — `nohup ./tools/compare_summarizers.sh … >/dev/null 2>&1 &
  disown`; the script's own `run.log` is the live progress (`tail -f`).

**Termination is clean.** Ctrl-C in the attached session, `tmux kill-session`,
or any script exit tears down the whole pipeline process group
(`uv → python → vLLM`), so VRAM is released. If a run is ever interrupted by a
hard `kill -9` of the script (which skips the trap), reclaim VRAM by hand:
`nvidia-smi` to find the leftover python PID, then `kill <pid>`.

---

## 5. Unit tests (local mac, no GPU)

vLLM is mocked, so these run on Apple Silicon.

```bash
uv sync --extra dev
.venv/bin/pytest tests/
```

---

## 6. Mid-development sync (local mac → uni-lab)

Push uncommitted changes without a commit/pull cycle:

```bash
rsync -av --exclude='.venv' --exclude='results' --exclude='__pycache__' \
  --exclude='models' --exclude='tmp' --exclude='.git' --exclude='.remember' \
  . uni-lab:~/Desktop/streamind/
```

---

## Gotchas

- **Pipeline aborts at warmup if `./models/<name>` is missing** — run `fetch_models.sh`.
- **`destination_endpoint` in `config-base.json`** must be set to the challenge POST URL before submission — currently `""` (results still write locally when empty).
- Use `./tmp` (not `/tmp`) for temp files — permission issues.
- Janus is a black box: do not modify its internals. We use Janus 0.x; 1.x compatibility unverified.
- Scoring contract source of truth: [`docs/CHALLENGE.md`](CHALLENGE.md). Hardware: [`docs/MACHINE-SPECS.md`](MACHINE-SPECS.md).
