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
# e.g.
./tools/fetch_models.sh Qwen/Qwen3.5-4B qwen3.5-4b
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
./tools/run_pipeline.sh -s <summarizer_profile>          # 300s window (default)
./tools/run_pipeline.sh -w 30 -s <summarizer_profile>    # short 30s window
./tools/run_pipeline.sh -s <profile> -a small.en         # ASR override (A/B)

# e.g.
./tools/run_pipeline.sh -s vllm-qwen3.5-9b -a small.en
```

Flags (`tools/run_pipeline.sh`):

| Flag | Meaning | Default |
| ---- | ------- | ------- |
| `-s, --summarizer <profile>` | profile under `pipelines/summarizer/<profile>.json` | **required** |
| `-w, --window <seconds>` | rolling window duration | `300` |
| `-a, --asr <model>` | override ASR `model_name` (e.g. `small.en`, `large-v3-turbo`) | config-base value |
| `-h, --help` | usage | — |

What it does: preflight (`nvidia-smi` + model dir present) → assembles
`config-base.json` + summarizer profile into `./tmp/config-<w>s-<profile>.json`
→ exports CUDA libs + disables FlashInfer JIT → `juturna launch`.

Output JSON written to `results/<model>/<window>/window_*.json` with challenge
keys: `from`, `to`, `summary`, `keywords` (exactly 3), `proc_time`.

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

Heartbeat logs elapsed/total every 60s and surfaces WebRTC state, so mid-stream
truncation is visible without log spelunking. Same path in dev and production —
do **not** swap to an `audio_file` source for testing.

---

## 4. Score the output (offline judge)

Judges run **after** the pipeline exits — the summarizer must be unloaded first
to free VRAM (a single GPU cannot hold summarizer + judge co-resident).

### 4a. Single judge

```bash
uv run python tools/eval_quality.py results/<model>/300/ \
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
uv run python tools/eval_multi_judge.py results/<model>/300/ \
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
