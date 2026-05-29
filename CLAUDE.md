# STREAMIND Project — Claude Agent Instructions

## Challenge Summary

STREAMIND Grand Challenge: build **real-time AI meeting intelligence pipeline**. Receives live audio stream, transcribes incrementally, aggregates context, produces structured LLM outputs (summary + keywords) with minimal latency.

See **Scoring Constraints** section for formula and limits. Full spec in [`docs/CHALLENGE.md`](docs/CHALLENGE.md).

## Pipeline Architecture

6 sequential stages, all **Juturna nodes**:

1. **Audio Reception** — Receive Opus-encoded mono RTP stream via WebRTC (through Janus gateway)
2. **Incremental ASR** — Transcribe short, consecutive, partially overlapping audio chunks
3. **Novel Chunk Extraction** — Deduplicate overlapping content; extract new portion
4. **Window Aggregation** — Accumulate transcript into 300-second rolling context windows
5. **Summarization** — vLLM in-process produces one summary + exactly 3 keywords per window
6. **Transmission** — Store result locally and POST to challenge destination endpoint

## Key Frameworks

### Juturna

- Open-source Python framework for real-time AI pipeline prototyping
- Node-based composable architecture
- **All pipeline stages must be implemented as Juturna nodes**
- Repo: <https://github.com/meetecho/juturna>

### Janus

- Open-source WebRTC server for audio delivery to pipeline
- Repo: <https://github.com/meetecho/janus-gateway>

### vLLM (inference backend)

- In-process Python library: `from vllm import LLM, SamplingParams`
- **No daemon.** Summarizer node loads the model into its own process at warmup.
- Loads weights from a **local filesystem path only** (`./models/<name>`). Pipeline refuses to start if the directory is missing.

## Scoring Constraints

- Per-chunk score: `C_i = B_i + K_i + L_i` — quality gates latency reward
- **Final score = average of all chunk scores + Janus bonus (if applicable)**, then min-max normalised across submissions
- `L_i` only applies when `B_i ≥ 10` — prioritize summary quality over raw latency
- Each window output **must include exactly 3 keywords**; wrong count hurts `K_i`
- Missing/extra keywords penalized: −2 per irrelevant keyword
- **Janus bonus**: +4 flat for using Janus instead of `ffmpeg` as the audio source

## Evaluation Environment

- Production: single **RTX Pro 4500** GPU (Blackwell, **32 GB GDDR7**, 896 GB/s, FP4/FP8/INT8 inference, no NVLink) — size all models to fit in VRAM
- **Lab machine** (`uni-lab`): RTX 4090, 24 GB, CUDA 12.4 — primary dev host. **Tighter VRAM than production (24 < 32 GB), so it is the binding constraint for live models during dev.** Access via `ssh uni-lab`.
- All inference is CUDA-native (vLLM); no MLX, no Ollama, no other middleware.

## Model Packaging

**Model weights ship with the submission**; pipeline code never downloads at runtime.

- Local dev: `./tools/fetch_models.sh <hf_id> <local_name>` populates `./models/<local_name>/`. Run once after clone.
- Submission Docker (M3): same script invoked during `docker build`, so the image carries the weights. Container has zero network dependency at runtime.
- Pipeline configs reference local paths (`./models/<local_name>`), never HF ids.
- `models/` is `.gitignore`d — never commit weights.

## Key References

- [`docs/CHALLENGE.md`](docs/CHALLENGE.md) — official challenge spec (scoring contract source of truth)
- [`docs/CLAUDE.md`](docs/CLAUDE.md) — docs folder navigation
- [`docs/documentation/CLAUDE.md`](docs/documentation/CLAUDE.md) — Juturna/Janus reference
- [`docs/plans/cuda-native-pipeline-base-first.md`](docs/plans/cuda-native-pipeline-base-first.md) — active plan (M0/M1 done, M2–M4 ahead)

## Directory Structure

```text
plugins/nodes/        # Juturna node implementations
  source/             # _audio_file (WAV file source for local testing)
  proc/               # _audio_chunker, _novel_extractor, _transcriber_whisper,
                      # _window_aggregator, _hallucination_filter,
                      # _summarizer_vllm, _summarizer_common (shared helpers)
  sink/               # _result_transmitter, _judge_common (shared scorer)
pipelines/            # Pipeline configs
  config-base.json    # Base pipeline (all nodes except summarizer)
  summarizer/         # vLLM summarizer profiles — one JSON per model
  judge/              # Offline judge profiles — consumed only by tools/eval_quality.py
models/               # LLM weights (gitignored) — populated by tools/fetch_models.sh
tools/
  fetch_models.sh     # Dev helper: download an HF model into ./models/<name>
  assemble_config.py  # Merges base + summarizer profile into a complete config
  run_pipeline.sh     # Launcher: ./run_pipeline.sh --window <s> --summarizer <profile>
  eval_quality.py     # Offline judge harness — uses vLLM, runs after pipeline exits
  send_audio.py       # Inject a WAV file through Janus for local testing
  finetune/           # QLoRA data prep + training on rev16 — PARKED until M4 (Ollama refs remain)
datasets/             # Audio corpora — audio gitignored, text/json tracked
  rev16/<episode>/    #   audio.opus, transcript.txt, chunks.json (ground truth)
  ietf/               #   <name>.opus (gitignored), <name>.opus.txt (transcript)
tests/                # Unit + integration tests
results/              # Output JSON files written by result_transmitter
docs/                 # Challenge spec, plans, approach write-up
.claude/skills/       # Project-specific Claude Code skills
```

## Quick Start

```bash
# --- Local (Apple Silicon) — unit tests only, vLLM is mocked ---
uv sync --extra dev
.venv/bin/pytest tests/

# --- Lab (uni-lab, RTX 4090, CUDA 12.4) — full pipeline ---
# Login once, then run commands directly. Repo lives at ~/Desktop/streamind on uni-lab.
ssh uni-lab
cd ~/Desktop/streamind

git pull
uv sync --extra dev

# Populate model once (idempotent)
./tools/fetch_models.sh <hf_id> <local_name>

# Start Janus (first run builds the Docker image — takes ~15 min)
docker compose up -d janus

# Run pipeline
./tools/run_pipeline.sh -s vllm-<local_name>
./tools/run_pipeline.sh -w 30 -s vllm-<local_name>   # short window

# Offline judge after the pipeline exits (judge model loaded sequentially,
# summarizer must be unloaded first to free VRAM).
# --audio enables ASR WER + auto-discovers chunks.json for ground-truth comparison.
uv run python tools/eval_quality.py results/<local_name>/300/ \
  --judge-profile vllm-<judge_name> \
  --audio 'datasets/rev16/10_Creating_Your_Own_Lane_in_Podcasting_ft_@Favyfav_of_@latinoswholunch/audio.opus'

# Multi-judge: score with a cross-family panel run sequentially (one subprocess
# per judge → full VRAM reclaim between models), report side-by-side C + consensus.
# Writes judge_report_multi.txt + judge_scores_multi.json next to the results.
uv run python tools/eval_multi_judge.py results/<local_name>/300/ \
  --judge-profiles vllm-mistral-small-24b-awq vllm-phi-4-awq vllm-gemma3-27b-it-int4-awq \
  --audio 'datasets/rev16/10_Creating_Your_Own_Lane_in_Podcasting_ft_@Favyfav_of_@latinoswholunch/audio.opus'

# Inject test audio through Janus (separate terminal/ssh session, while pipeline runs)
# Default rev16 fixture (~36 min — yields ~7×300s windows). Any ffmpeg-decodable format works.
uv run python tools/send_audio.py 'datasets/rev16/10_Creating_Your_Own_Lane_in_Podcasting_ft_@Favyfav_of_@latinoswholunch/audio.opus'

# --- From local mac, mid-development sync (uncommitted changes) ---
rsync -av --exclude='.venv' --exclude='results' --exclude='__pycache__' --exclude='models' . uni-lab:~/Desktop/streamind/
```

## Runtime Requirements

- **Janus** running via Docker on uni-lab: `docker compose up -d` (builds from source on first run)
- **vLLM** is a base dependency (no extra needed). `uv sync` installs it; **Linux + CUDA only — will not install on Apple Silicon**. All dev happens on `uni-lab`.
- ASR model (`deepdml/faster-whisper-large-v3-turbo-ct2` via faster-whisper) is baked into Docker at build time or fetched locally via `snapshot_download`. In dev, transcriber auto-resolves `./models/faster-whisper-large-v3-turbo` if present, otherwise falls back to HF download.
- LLM weights must be present under `./models/<name>/` before pipeline launch — pipeline aborts at warmup otherwise.
- Pipeline configs assembled at launch: `config-base.json` + profile from `pipelines/summarizer/`
- `audio_rtp` node listens on `0.0.0.0:8888`; Janus forwards RTP to that port

## Current Model Choices

- **Summarizer:** `Qwen/Qwen3.5-4B` (Apache-2.0, BF16, ~10 GB w/ KV at 2K). Profile: `pipelines/summarizer/vllm-qwen3.5-4b.json`. Local dir: `./models/qwen3.5-4b/`.
- **Judge (offline only) — cross-family panel.** Multiple judges from different families avoid single-model bias. Run sequentially (a single GPU — 24 GB dev 4090 or 32 GB prod Pro 4500 — cannot hold the full panel co-resident) via `tools/eval_multi_judge.py`, one subprocess per judge so process exit reclaims VRAM before the next loads. Combined report (`judge_report_multi.txt` + `judge_scores_multi.json`) shows each judge's C side-by-side plus a consensus mean + stdev — low stdev corroborates the score, high stdev flags bias. Qwen judges excluded (summarizer family → self-bias). Single-judge `tools/eval_quality.py` still works unchanged.
  - `stelterlab/Mistral-Small-24B-Instruct-2501-AWQ` (~13 GB AWQ-int4, Apache-2.0, text-only). Profile: `pipelines/judge/vllm-mistral-small-24b-awq.json`. Local dir: `./models/mistral-small-24b-awq/`. Previous pick `cyankiwi/Qwen3.5-27B-AWQ-BF16-INT4` dropped 2026-05-20: not cleanly int4 (~26 GB on disk → OOM on 24 GB).
  - `casperhansen/phi-4-awq` (Microsoft, ~8.5 GB AWQ-int4, MIT). Profile: `pipelines/judge/vllm-phi-4-awq.json`. Local dir: `./models/phi-4-awq/`.
  - `gaunernst/gemma-3-27b-it-int4-awq` (Google, ~18 GB int4-AWQ). Profile: `pipelines/judge/vllm-gemma3-27b-it-int4-awq.json`. Local dir: `./models/gemma3-27b-it-int4-awq/`. Uses `dtype: bfloat16` (Gemma overflows at fp16) and `gpu_memory_utilization: 0.90` (18 GB weights leave tight KV room on the 24 GB dev card). Requires a vLLM build with Gemma-3 support.
- **Fallback summarizer:** `Qwen/Qwen3.5-9B` if 4B B_i averages < 15. Same prompt + tooling; just swap profile.

See [`plugins/nodes/CLAUDE.md`](plugins/nodes/CLAUDE.md) for node layout. No MLX/Ollama variants exist.

## Gotchas

- **Local Apple-Silicon dev cannot run the pipeline.** vLLM does not install. Use uni-lab. Unit tests still pass locally because vLLM is mocked.
- **Pipeline aborts at warmup if `./models/<name>` is missing.** Run `./tools/fetch_models.sh` to populate. Never commit weights — `.gitignore` handles `models/`.
- `destination_endpoint` in `pipelines/config-base.json` must be set to the challenge POST URL before submission — currently `""` (results still write locally when empty). Set in M3.
- **Challenge output format**: `result_transmitter` maps internal keys to challenge-required keys: `window_start` → `from`, `window_end` → `to`, `latency` → `proc_time`. Output must contain exactly: `from`, `to`, `summary`, `keywords` (3 items), `proc_time`.
- `encoding_clock_chan: "opus/48000/1"` in `config-base.json` declares mono Opus per challenge spec. Verify against actual Janus stream before submission.
- Local + production same environment: both go through Janus → `audio_rtp`. Don't swap to `audio_file` for testing.
- To inject an audio file into pipeline: `uv run python tools/send_audio.py <file>` (accepts wav, opus, mp3, m4a — anything ffmpeg decodes)
- `uv sync` and `uv sync --extra dev` must run on `uni-lab` — vLLM + torch cu124 wheels do not install on Apple Silicon.
- **ASR model override**: `./tools/run_pipeline.sh -a <model>` overrides ASR model without editing config-base.json. Useful for A/B comparisons (e.g. `-a small.en` vs default `large-v3-turbo`).
- `tools/finetune/` is **parked** until M4 (`prepare_rev16.py` and `eval_finetuned.py` still use `ollama` Python client — will be rewritten to vLLM in M4).
- To download YouTube video: see `docs/documentation/yt-dlp-guide.md`
- Use `./tmp` (not `/tmp`) for temp files to avoid permission issues.

## Skills

| Skill | Purpose |
| -------- | ----------- |
| `/run-pipeline` | Check GPU + model dir, launch pipeline |
| `/benchmark-pipeline` | Parse `results/*/window_*.json`, compute L_i per chunk, flag format violations |
| `/tune-prompt` | Test summarization prompt against sample transcript via vLLM |
| `/swap-model` | Switch the local model path in `pipelines/summarizer/*.json` and run `fetch_models.sh` |
| `/add-node` | Scaffold new Juturna node with correct structure |
| `/prep-submission` | Pre-submission checklist — config, format, CUDA smoke test, Docker baking |
| `/remote-finetune` | **PARKED — M4 only.** QLoRA pipeline (prepare → train → merge → swap into vLLM) |

## Constraints and Disallowed Approaches

- **No workarounds**: challenge tests real-time processing. No pre-processing entire audio or non-streaming models.
- **Real-time only**: process audio as received, no waiting for full audio.
- **No daemons / no middleware**: inference is in-process via vLLM. No Ollama, no MLX server, no HTTP LLM backend.
- **No env-specific deps**: solution must be portable. CUDA + vLLM is the single supported path; both ship in the submission Docker.
- **No model download at runtime**: weights ship with the image. Pipeline aborts if the local model dir is missing.
- **No separate envs**: same code + config run in all contexts without modification.
- **No hardcoded local paths**: use relative paths only (`./models/<name>`, `./results/`, `./tmp/`).
