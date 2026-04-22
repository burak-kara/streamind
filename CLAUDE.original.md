# STREAMIND Project — Claude Agent Instructions

## Challenge Summary

STREAMIND is a Grand Challenge to build a **real-time AI meeting intelligence pipeline**. The system receives a live audio stream from a meeting, transcribes it incrementally, aggregates context, and produces structured LLM outputs (summary + keywords) with minimal latency.

Scoring: `C_i = B_i + K_i + L_i` (per-chunk, averaged across all chunks, min-max normalised)

- `B_i` max 25: LLM judge on 5 Likert criteria — factual consistency, relevance, coherence, fluency, conciseness
- `K_i` max 6: +2 per relevant keyword, −2 per irrelevant; exactly 3 required
- `L_i` max ~6: `10·e^(−0.5·proc_time)`, **only if `B_i ≥ 10`**; ~0.09 at 9s, ~2.2 at 3s, ~6.1 at 1s
- Janus bonus: flat +4 (already earned by architecture)
Quality gates latency reward — `B_i < 10` zeroes out `L_i` entirely.

## Pipeline Architecture

The pipeline has 6 sequential stages, all implemented as **Juturna nodes**:

1. **Audio Reception** — Receive Opus-encoded mono RTP stream via WebRTC (through Janus gateway)
2. **Incremental ASR** — Transcribe short, consecutive, partially overlapping audio chunks
3. **Novel Chunk Extraction** — Deduplicate overlapping content between consecutive chunks; extract only the new portion
4. **Window Aggregation** — Accumulate transcript into 300-second rolling context windows
5. **Summarization** — LLM produces one summary + exactly 3 keywords per window
6. **Transmission** — Store result locally and POST to the challenge destination endpoint

## Key Frameworks

### Juturna

- Open-source Python framework for real-time AI data pipeline prototyping
- Node-based composable architecture
- **All pipeline stages must be implemented as Juturna nodes**
- Repo: <https://github.com/meetecho/juturna>

### Janus

- Open-source WebRTC server used for audio delivery to the pipeline
- Repo: <https://github.com/meetecho/janus-gateway>

## Scoring Constraints

- Per-chunk score: `C_i = B_i + K_i + L_i` — quality gates latency reward
- **Final score = average of all chunk scores**, min-max normalised across submissions
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
  sink/               # _result_transmitter
pipelines/            # Pipeline configs
  config-base.json    # Base pipeline (all nodes except summarizer)
  summarizer/         # Summarizer profiles — one JSON node definition per profile
    ollama-qwen3.5-9b.json        # Ollama, production default
    ollama-qwen3-1.7b.json        # Ollama, fast / lower latency
    mlx-Qwen3.5-2B-OptiQ-4bit.json  # MLX, fastest (Apple Silicon)
    mlx-Qwen3.5-4B-OptiQ-4bit.json  # MLX, balanced
    mlx-Qwen3.5-9B-OptiQ-4bit.json  # MLX, highest quality
tools/
  assemble_config.py  # Merges base + profile into a complete config for Juturna
  run_pipeline.sh     # Launcher: ./run_pipeline.sh --window <seconds> --summarizer <profile>
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
./tools/run_pipeline.sh -s ollama-qwen3-1.7b                   # Ollama fast alternative

# MLX profiles (Apple Silicon native, no Ollama server needed)
./tools/run_pipeline.sh -w 30 -s mlx-Qwen3.5-2B-OptiQ-4bit    # fastest (~1GB)
./tools/run_pipeline.sh -w 30 -s mlx-Qwen3.5-4B-OptiQ-4bit    # balanced (~2GB)
./tools/run_pipeline.sh -w 30 -s mlx-Qwen3.5-9B-OptiQ-4bit    # highest quality (~4.5GB)

# Inject test audio through Janus (in a separate terminal while pipeline is running)
# Use the 10-min fixture so the 300s production window fires at least once
uv run python tools/send_audio.py tests/fixtures/youtube_15min.wav
```

## Runtime Requirements

- **Janus** running via Docker: `docker compose up -d` (builds from source on first run)
- **MLX** (default): no server required — model downloads from HuggingFace on first run
- **Ollama** (alternative): running locally on `http://127.0.0.1:11434`; pull model with `ollama pull qwen3.5:9b-16k`
- ASR model (`small.en` via faster-whisper) downloads automatically on first run
- Pipeline configs are assembled at launch time: `config-base.json` + a profile from `pipelines/summarizer/`
- The `audio_rtp` node listens on `0.0.0.0:8888`; Janus forwards RTP to that port

## Current Model Choices

See [`plugins/nodes/CLAUDE.md`](plugins/nodes/CLAUDE.md) for node directory layout, model choices, and summarizer node details.

## Gotchas

- To download a YouTube video: see `docs/documentation/yt-dlp-guide.md`
- `destination_endpoint` in `pipelines/config-base.json` must be set to the challenge POST URL before submission — currently `""` (results still write locally when empty)
- Prompt templates: `summarize_prompt_ollama.txt` (Ollama, with `/no_think`), `summarize_prompt_mlx_qwen3.txt` (MLX Qwen3.5 profiles, with `/no_think`), and `summarize_prompt_mlx.txt` (legacy mlx, without `/no_think`). Configured via `prompt_template_file` in each pipeline config.
- Results are written to `results/{sanitized_model}/{window_duration}/window_N.json` and optionally POSTed to `destination_endpoint`. Model name sanitization: `:` → `-`, `/` → `_` (filesystem compatibility).
- **Challenge output format**: result_transmitter maps internal keys to challenge-required keys: `window_start` → `from`, `window_end` → `to`, `latency` → `proc_time`. Output must contain exactly: `from`, `to`, `summary`, `keywords` (3 items), `proc_time`.
- `encoding_clock_chan: "opus/48000/2"` in `config-base.json` declares stereo Opus; verify against actual Janus stream before submission — change to `opus/48000/1` if Janus sends mono
- Local and production use the same environment: both go through Janus → `audio_rtp`. Do not swap to `audio_file` for testing.
- To inject a WAV file into the pipeline locally: `uv run python tools/send_audio.py <file.wav>`

## Skills

| Skill | Purpose |
| -------- | ----------- |
| `/run-pipeline` | Check Ollama + model, then launch the pipeline |
| `/benchmark-pipeline` | Parse `results/window_*.json` and report latency stats + score estimate |
| `/tune-prompt` | Test the summarization prompt against a sample transcript via Ollama |
| `/swap-model` | Switch ASR or LLM model in `config.json` and verify availability |
| `/add-node` | Scaffold a new Juturna node with correct structure |

## Constraints and Disallowed Approaches

- **Do not introduce workarounds**: The challenge is designed to test real-time processing. Avoid any approaches that would circumvent the latency requirement, such as pre-processing the entire audio or using non-streaming models.
- **All processing must be done in real-time**: The system should process audio as it is received, without waiting for the entire audio to be available.
- **Do not introduce environment-specific dependencies**: The solution should be portable and not rely on specific hardware or software configurations that are not commonly available.
- **Do not introduce environemnts**: No Local or Production environments — the same code and config should run in both contexts without modification.
- **Do not write local file paths**: All file paths should be relative and not hardcoded to specific local directories.
- Use `./tmp` folder located in this project instead of `/tmp` for temporary files to avoid permission issues in some environments.
