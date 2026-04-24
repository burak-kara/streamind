# STREAMIND Project — Claude Agent Instructions

## Challenge Summary

STREAMIND Grand Challenge: build **real-time AI meeting intelligence pipeline**. Receives live audio stream, transcribes incrementally, aggregates context, produces structured LLM outputs (summary + keywords) with minimal latency.

Scoring:

- Per-chunk: `C_i = B_i + K_i + L_i`
- Per-audio source: `S_audio = avg_i(C_i)`
- Janus bonus: `+4` added to overall score
- Submission ranking: min-max normalised across submissions

- `B_i` max 25: LLM judge on 5 Likert criteria — factual consistency, relevance, coherence, fluency, conciseness
- `K_i` max 6: +2 per relevant keyword, −2 per irrelevant; exactly 3 required
- `L_i` max ~6: `10·e^(−0.5·proc_time)`, **only if `B_i ≥ 10`**; ~0.09 at 9s, ~2.2 at 3s, ~6.1 at 1s
Quality gates latency reward — `B_i < 10` zeroes out `L_i` entirely.

## Pipeline Architecture

6 sequential stages, all **Juturna nodes**:

1. **Audio Reception** — Receive Opus-encoded mono RTP stream via WebRTC (through Janus gateway)
2. **Incremental ASR** — Transcribe short, consecutive, partially overlapping audio chunks
3. **Novel Chunk Extraction** — Deduplicate overlapping content; extract new portion
4. **Window Aggregation** — Accumulate transcript into 300-second rolling context windows
5. **Summarization** — LLM produces one summary + exactly 3 keywords per window
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

## Scoring Constraints

- Per-chunk score: `C_i = B_i + K_i + L_i` — quality gates latency reward
- **Final score = average of all chunk scores + Janus bonus (if applicable)**, then min-max normalised across submissions
- `L_i` only applies when `B_i ≥ 10` — prioritize summary quality over raw latency
- Each window output **must include exactly 3 keywords**; wrong count hurts `K_i`
- Missing/extra keywords penalized: −2 per irrelevant keyword

## Evaluation Environment

- Production: single **RTX Pro 4500** GPU — size all models to fit within its VRAM
- MLX profiles run on Apple Silicon locally; submission must target RTX Pro 4500

## Key References

- [`docs/CLAUDE.md`](docs/CLAUDE.md) — docs folder contents and navigation guide (documentation, plans, challenge specification)
- [`docs/documentation/CLAUDE.md`](docs/documentation/CLAUDE.md) — Juturna/Janus links and local documentation index

## Directory Structure

```text
plugins/nodes/        # Juturna node implementations
  source/             # _audio_file (WAV file source for local testing)
  proc/               # _audio_chunker, _novel_extractor, _transcriber_whisper, _window_aggregator, _summarizer_llm, _summarizer_mlx
  sink/               # _result_transmitter, _judge_llm (async LLM-as-judge), _judge_common (shared scorer helper)
pipelines/            # Pipeline configs
  config-base.json    # Base pipeline (all nodes except summarizer)
  summarizer/         # Summarizer profiles — one JSON node definition per profile
    ollama-qwen3.5-9b.json        # Ollama, higher quality
    ollama-qwen3.5-4b.json        # Ollama, submission default (CUDA target)
    mlx-Qwen3.5-2B-OptiQ-4bit.json  # MLX, fastest (Apple Silicon)
    mlx-Qwen3.5-4B-OptiQ-4bit.json  # MLX, balanced
    mlx-Qwen3.5-9B-OptiQ-4bit.json  # MLX, highest quality
  judge/              # Optional async LLM-as-judge profiles (off by default; opt in with --judge)
    ollama-qwen3.5-4b.json        # Cheap scoring; pair with the 9b summarizer
    ollama-qwen3.5-9b.json        # Higher-quality scoring
tools/
  assemble_config.py  # Merges base + summarizer (+ optional judge) into a complete config
  run_pipeline.sh     # Launcher: ./run_pipeline.sh --window <s> --summarizer <profile> [--judge <profile>]
  eval_quality.py     # Offline judge harness — shares the scorer module with the in-pipeline judge_llm node
tests/                # Unit + integration tests, fixtures/
results/              # Output JSON files written by result_transmitter
docs/                 # Challenge spec, TODO, plans
.claude/skills/       # Project-specific Claude Code skills
```

## Quick Start

```bash
# Install (requires uv + Python 3.12) — include mlx for production inference
uv sync --extra dev --extra mlx

# Start Janus (first run builds the Docker image — takes ~15 min)
docker compose up

# Run unit tests (no Janus or Ollama required)
.venv/bin/pytest tests/

# Run pipeline (requires Janus)
./tools/run_pipeline.sh                                        # 300s window, default (ollama-qwen3.5-9b)
./tools/run_pipeline.sh --window 30                            # 30s window for faster iteration
./tools/run_pipeline.sh -s ollama-qwen3.5-4b                   # Ollama submission default (CUDA target)

# MLX profiles (Apple Silicon native, no Ollama server needed)
./tools/run_pipeline.sh -w 30 -s mlx-Qwen3.5-2B-OptiQ-4bit    # fastest (~1GB)
./tools/run_pipeline.sh -w 30 -s mlx-Qwen3.5-4B-OptiQ-4bit    # balanced (~2GB)
./tools/run_pipeline.sh -w 30 -s mlx-Qwen3.5-9B-OptiQ-4bit    # highest quality (~4.5GB)

# Add async LLM-as-judge (writes B/K/L/C scores per window to results/.../judge/, never blocks the live path)
./tools/run_pipeline.sh -s ollama-qwen3.5-9b -j ollama-qwen3.5-4b

# Inject test audio through Janus (in a separate terminal while pipeline is running)
# Use the 10-min fixture so the 300s production window fires at least once
uv run python tools/send_audio.py tests/fixtures/youtube_15min.wav
```

## Runtime Requirements

- **Janus** running via Docker: `docker compose up -d` (builds from source on first run)
- **MLX** (default): no server required — model downloads from HuggingFace on first run
- **Ollama** (alternative): running locally on `http://127.0.0.1:11434`; pull model with `ollama pull qwen3.5:9b-16k`
- ASR model (`small.en` via faster-whisper) downloads automatically on first run
- Pipeline configs assembled at launch: `config-base.json` + profile from `pipelines/summarizer/`
- `audio_rtp` node listens on `0.0.0.0:8888`; Janus forwards RTP to that port

## Current Model Choices

See [`plugins/nodes/CLAUDE.md`](plugins/nodes/CLAUDE.md) for node layout, model choices, summarizer details.

## Gotchas

- To download YouTube video: see `docs/documentation/yt-dlp-guide.md`
- `destination_endpoint` in `pipelines/config-base.json` must be set to challenge POST URL before submission — currently `""` (results still write locally when empty)
- Prompt templates: `summarize_prompt_ollama.txt` (Ollama, `/no_think`), `summarize_prompt_mlx_qwen3.txt` (MLX Qwen3.5, `/no_think`), `summarize_prompt_mlx.txt` (legacy mlx, no `/no_think`). Set via `prompt_template_file` in each pipeline config.
- Results written to `results/{sanitized_model}/{window_duration}/window_N.json`, optionally POSTed to `destination_endpoint`. Model name sanitization: `:` → `-`, `/` → `_` (filesystem compatibility).
- **Challenge output format**: result_transmitter maps internal keys to challenge-required keys: `window_start` → `from`, `window_end` → `to`, `latency` → `proc_time`. Output must contain exactly: `from`, `to`, `summary`, `keywords` (3 items), `proc_time`.
- `encoding_clock_chan: "opus/48000/2"` in `config-base.json` declares stereo Opus; verify against actual Janus stream before submission — change to `opus/48000/1` if Janus sends mono
- Local + production same environment: both go through Janus → `audio_rtp`. Don't swap to `audio_file` for testing.
- To inject WAV file into pipeline locally: `uv run python tools/send_audio.py <file.wav>`

## Skills

| Skill | Purpose |
| -------- | ----------- |
| `/run-pipeline` | Check Ollama + model, launch pipeline |
| `/benchmark-pipeline` | Parse `results/*/window_*.json`, compute L_i per chunk, flag format violations |
| `/tune-prompt` | Test summarization prompt against sample transcript via Ollama |
| `/swap-model` | Switch ASR or LLM model in `config.json`, verify availability |
| `/add-node` | Scaffold new Juturna node with correct structure |

## Constraints and Disallowed Approaches

- **No workarounds**: challenge tests real-time processing. No pre-processing entire audio or non-streaming models.
- **Real-time only**: process audio as received, no waiting for full audio.
- **No env-specific deps**: solution must be portable, no hardware/software configs not commonly available.
- **No separate envs**: same code + config run in all contexts without modification.
- **No hardcoded local paths**: use relative paths only.
- Use `./tmp` (not `/tmp`) for temp files to avoid permission issues.
