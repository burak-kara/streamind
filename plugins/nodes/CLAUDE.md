# Pipeline Nodes

## Directory Layout

| Folder | Node | Role |
| --------- | ------ | ------ |
| `source/_audio_file/` | `audio_file` | WAV file source for local testing via Janus |
| `proc/_audio_chunker/` | `audio_chunker` | Splits RTP audio into short overlapping chunks |
| `proc/_novel_extractor/` | `novel_extractor` | Deduplicates overlapping chunk content |
| `proc/_transcriber_whisper/` | `transcriber_whisper` | ASR via faster-whisper |
| `proc/_window_aggregator/` | `window_aggregator` | Accumulates transcript into rolling windows |
| `proc/_hallucination_filter/` | `hallucination_filter` | Post-processes LLM output to remove hallucinations |
| `proc/_summarizer_vllm/` | `summarizer_vllm` | LLM summarization — native CUDA via vLLM in-process |
| `proc/_summarizer_common/` | — (helper) | Shared keyword post-processing (loaded via importlib, not a Juturna node) |
| `sink/_result_transmitter/` | `result_transmitter` | Writes results locally and POSTs to endpoint |
| `sink/_judge_common/` | — (helper) | Shared judge prompt + B/K/L/C math, used by `tools/eval_quality.py` (loaded via importlib, not a Juturna node) |

## Judge — Offline Only

There is no live judge sink in the pipeline. `tools/eval_quality.py` runs **after** the pipeline exits: it loads a judge model with vLLM, iterates `results/<model>/<window>/window_*.json`, and writes `results/.../judge/window_N.json` with `B/K/L/C` scores. Sequential loading avoids VRAM co-residency.

Rationale: the RTX Pro 4500 has 24 GB. A 8B+ summarizer + 7B+ judge co-resident does not fit. At submission time the judge is irrelevant (held-out evaluator scores us); during development we score sequentially.

Output schema: same as the live window JSON plus `B_breakdown`, `B`, `K`, `L`, `C`, `keyword_flags`, `judge_model`. Never touches the official submission file.

## Summarizer

**`summarizer_vllm`** — Native CUDA inference via [vLLM](https://github.com/vllm-project/vllm) `LLM` class, in-process. No daemon. Requires `uv sync --extra cuda` (Linux + CUDA only).

- **Model source:** local filesystem path (`./models/<name>`). Never an HF id at runtime. Pipeline aborts at warmup if the directory is missing — clear error directs the user to `tools/fetch_models.sh`.
- **Config (`config.toml` defaults):** `model_name`, `prompt_template_file` (default `summarize_prompt.txt`), `dtype` (default `float16`), `gpu_memory_utilization` (0.85), `max_model_len` (2048), `max_tokens` (150), `temperature` (0.3), `top_p` (0.9), `repetition_penalty` (1.05), `enforce_eager` (false).
- **Output contract:** strict JSON `{"summary": ..., "keywords": [3 strings]}`. Output goes through the shared `keywords.ensure_three_keywords()` helper which guarantees the count and bans generic terms.
- **Prompt:** `summarize_prompt.txt` — model-agnostic, includes `/no_think` to suppress thinking traces on Qwen-family models.

## Model Choices

Active summarizer model is committed in `pipelines/summarizer/vllm-<name>.json` and the weights live under `./models/<name>/` (gitignored). One profile only; no MLX/Ollama variants exist.

Selection criteria:
- ≤ ~20 GB fp16 to leave VRAM headroom on the 24 GB RTX Pro 4500
- Permissive license for redistribution in the submission Docker image
- Strong instruction-following + reliable JSON output
- Available on HuggingFace
