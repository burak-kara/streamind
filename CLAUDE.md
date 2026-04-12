# STREAMIND Project — Claude Agent Instructions

## Challenge Summary

STREAMIND is a Grand Challenge to build a **real-time AI meeting intelligence pipeline**. The system receives a live audio stream from a meeting, transcribes it incrementally, aggregates context, and produces structured LLM outputs (summary + keywords) with minimal latency.

Scoring: `Score = LLM-as-judge(summary quality, 0–30) − latency_penalty`

Latency directly subtracts from score. Every millisecond matters.

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

- Per-chunk score: `C_i = S_i − latency_i` where `S_i ∈ [0, 30]` (LLM-as-judge)
- **Final score = average of all chunk scores** across the audio source
- Latency is subtracted directly from each chunk's score — minimize end-to-end processing time
- Each window output **must include exactly 3 keywords**
- Missing keywords or malformed output will hurt the score

## Key References

- Full challenge specification: `docs/CHALLENGE.md`
- Juturna framework: <https://github.com/meetecho/juturna>
- Janus WebRTC server: <https://github.com/meetecho/janus-gateway>

### Juturna Local Documentation (`docs/documentation/juturna/`)

| File | Contents |
|------|----------|
| `README.md` | Overview and getting started |
| `node-development.md` | Juturna node development guide |
| `package.html` | Full API reference |
| `audio-transcription.html` | Audio pipeline tutorial |
| `explain/1- rationale.html` | Design philosophy and scope |
| `explain/2- entities.html` | Core entities: Pipeline, Node, Message |
| `explain/3- pipelines.html` | Pipeline lifecycle and config format |
| `explain/4- nodes.html` | Node types, threading model, lifecycle |
| `explain/5- messages-and-payloads.html` | Payload types, immutability, Draft |
| `explain/6- remote-services.html` | Remotizing nodes as microservices |
| `how-to/1- why-and-how-to.html` | Practical introduction |
| `how-to/2- create-costum-nodes.html` | Step-by-step custom node guide |
| `how-to/3- create-pipelines.html` | Pipeline creation workflow |
| `how-to/4- cli-tool.html` | All CLI commands reference |
| `how-to/5- constants-and-environment.html` | Env vars and global constants |
| `how-to/6- remotise-node.html` | Remote node deployment |
| `how-to/7- observability.html` | Telemetry and logging |

### Janus Local Documentation

- `docs/documentation/janus/README.md` — Janus overview and getting started

## Directory Structure

```text
plugins/nodes/        # Juturna node implementations
  source/             # _audio_file (WAV file source for local testing)
  proc/               # _audio_chunker, _novel_extractor, _transcriber_whisper, _window_aggregator, _summarizer_llm
  sink/               # _result_transmitter
pipelines/            # Single source of truth for pipeline config
  config.json         # Production pipeline (qwen3.5:9b-16k, 300s windows)
  config-fast.json    # Fast mode (qwen3:1.7b, 300s windows)
tests/                # Unit + integration tests, fixtures/
results/              # Output JSON files written by result_transmitter
docs/                 # Challenge spec, TODO, plans
.claude/skills/       # Project-specific Claude Code skills
```

## Quick Start

```bash
# Install (requires uv + Python 3.12)
uv sync --extra dev

# Start Janus (first run builds the Docker image — takes ~15 min)
docker compose up -d

# Run unit tests (no Janus or Ollama required)
.venv/bin/pytest tests/

# Run pipeline (requires Janus + Ollama)
uv run python -m juturna launch --config pipelines/config.json
# OR for 30s windows:
uv run python -m juturna launch --config pipelines/config-30s.json

# Inject test audio through Janus (in a separate terminal while pipeline is running)
# Use the 10-min fixture so the 300s production window fires at least once
uv run python tools/send_audio.py tests/fixtures/youtube_15min.wav
```

## Runtime Requirements

- **Janus** running via Docker: `docker compose up -d` (builds from source on first run)
- **Ollama** running locally on `http://127.0.0.1:11434`
- Pull the LLM model before first run: `ollama pull qwen3.5:9b-16k`
- ASR model (`small.en` via faster-whisper) downloads automatically on first run
- `pipelines/config.json` is the single production config — edit it directly
- The `audio_rtp` node listens on `0.0.0.0:8888`; Janus forwards RTP to that port

## Current Model Choices

| Stage         | Model                     | Notes                                      |
|---------------|---------------------------|--------------------------------------------|
| ASR           | `faster-whisper small.en` | `device: auto`, int8, English-only         |
| Summarization | `qwen3.5:9b-16k` via Ollama     | Structured JSON output, stop-drain logic   |

## Gotchas

- `destination_endpoint` in `pipelines/config.json` must be set to the challenge POST URL before submission — currently `""` (results still write locally when empty)
- Prompt template for summarizer lives at `plugins/nodes/proc/_summarizer_llm/summarize_prompt.txt`
- Results are written to `results/window_N.json` and optionally POSTed to `destination_endpoint`
- **Challenge output format**: result_transmitter maps internal keys to challenge-required keys: `window_start` → `from`, `window_end` → `to`, `latency` → `proc_time`. Output must contain exactly: `from`, `to`, `summary`, `keywords` (3 items), `proc_time`.
- `encoding_clock_chan: "opus/48000/2"` in `config.json` declares stereo Opus; verify against actual Janus stream before submission — change to `opus/48000/1` if Janus sends mono
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
