# Pipeline Nodes

## Directory Layout

| Folder | Node | Role |
| --------- | ------ | ------ |
| `source/_audio_file/` | `audio_file` | WAV file source for local testing via Janus |
| `proc/_audio_chunker/` | `audio_chunker` | Splits RTP audio into short overlapping chunks |
| `proc/_novel_extractor/` | `novel_extractor` | Deduplicates overlapping chunk content |
| `proc/_transcriber_whisper/` | `transcriber_whisper` | ASR via faster-whisper |
| `proc/_window_aggregator/` | `window_aggregator` | Accumulates transcript into rolling windows |
| `proc/_summarizer_mlx/` | `summarizer_mlx` | LLM summarization — native MLX (default) |
| `proc/_summarizer_llm/` | `summarizer_llm` | LLM summarization — Ollama backend |
| `proc/_summarizer_common/` | — (helper) | Shared keyword post-processing for both summarizer nodes (loaded via importlib, not a Juturna node) |
| `proc/_hallucination_filter/` | `hallucination_filter` | Post-processes LLM output to remove hallucinations |
| `sink/_result_transmitter/` | `result_transmitter` | Writes results locally and POSTs to endpoint |

## Current Model Choices

| Stage | Model | Notes |
| --------- | --------- | --------- |
| ASR | `faster-whisper small.en` | `device: auto`, int8, English-only |
| Summarization | `Qwen3.5-2B-OptiQ-4bit` (mlx) | Fastest MLX option; ~1GB; `temp=0.2` |
| Summarization | `Qwen3.5-4B-OptiQ-4bit` (mlx) | Balanced MLX option; ~2GB; `temp=0.3` |
| Summarization | `Qwen3.5-9B-OptiQ-4bit` (mlx) | Highest quality MLX; ~4.5GB; `temp=0.4` |
| Summarization | `qwen3.5:9b-16k` (Ollama) | Ollama default; requires Ollama running |

## Summarizer Nodes

Two separate Juturna nodes — switch by changing `mark` in the pipeline config:

- **`summarizer_mlx`** (default) — Native Apple Silicon inference via `mlx-lm`. No server required. Install with `uv sync --extra mlx`. Model names are HuggingFace IDs. Qwen3.5 profiles use `summarize_prompt_mlx_qwen3.txt` (includes `/no_think` to suppress chain-of-thought). Config params: `model_name`, `prompt_template_file`, `num_predict`, `temp`, `top_p`, `repetition_penalty`.
- **`summarizer_llm`** (alternative) — Ollama backend. Requires Ollama at `http://127.0.0.1:11434`. Prompt: `summarize_prompt_ollama.txt` (includes `/no_think`). Config params: `endpoint`, `model_name`, `num_ctx`, `num_predict`.
