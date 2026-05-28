# STREAMIND — Approach

> **Status: submission-ready.** Pipeline runs on native-CUDA vLLM, packaged via the `Dockerfile` with model weights baked in (zero runtime network dependency). Schema verified end-to-end inside the container.

## Challenge restatement

Real-time meeting intelligence from a live audio stream. Per-chunk score
`C_i = B_i + K_i + L_i` (B: LLM judge, 5 Likert criteria incl. conciseness,
max 25; K: keyword relevance, max 6; L: latency reward,
`10·e^(−0.5·proc_time)` gated by `B_i ≥ 10`). Audio-level score averages
chunk scores and adds a flat `+4` Janus bonus.

## Pipeline architecture

Eight sequential Juturna nodes, wired in `pipelines/config-base.json`:

1. `audio_rtp` — receives mono Opus/48k RTP on `0.0.0.0:8888`. Janus
   gateway forwards audio into this port. Config: `channels: 1`,
   `encoding_clock_chan: "opus/48000/1"`.
2. `audio_chunker` — splits the live stream into 5 s overlapping chunks
   (1 s overlap) so the ASR sees audio incrementally, not as one blob.
3. `transcriber_whisper` — faster-whisper `large-v3-turbo`, `device: auto`
   (GPU when available), `compute_type: float16`. OpenAI's distilled
   large-v3 (~809M params) — near large-v3 WER at half the size. Loads
   ~2 GB on GPU alongside the summarizer; the 4B summarizer plus ASR plus
   KV cache fits comfortably in 24 GB. Linked against CTranslate2 with
   PyTorch-bundled cuBLAS via an `LD_LIBRARY_PATH` export in
   `tools/run_pipeline.sh` so the stack stays self-contained.
   `language: "en"` set explicitly (multilingual model, no `.en` suffix).
4. `hallucination_filter` — post-processes ASR output to drop known
   hallucinatory sentences before aggregation.
5. `novel_extractor` — deduplicates the overlapping ASR output, keeps
   only the new content each chunk contributes.
6. `window_aggregator` — concatenates novel transcript segments into a
   rolling 300-second context window. Emits one message per completed
   window.
7. `summarizer_vllm` — **native CUDA via vLLM in-process.** Loads the
   model from `./models/<name>/` at warmup, generates `{summary,
   keywords: [3]}` JSON per window. No daemon, no middleware. Refuses
   to start if the model directory is missing.
8. `result_transmitter` — maps internal keys to the challenge contract
   (`from`, `to`, `summary`, `keywords`, `proc_time`), writes the result
   locally, and POSTs to `destination_endpoint` when configured.

## Key design decisions

- **Janus WebRTC bonus.** Audio enters through a Janus gateway in both
  local dev and the evaluation environment — no environment-specific code
  paths. This earns the flat `+4` audio-level bonus at zero architectural
  cost.
- **ASR model — faster-whisper `large-v3-turbo`.** OpenAI's distilled
  large-v3 (~809M params, ~2 GB float16). Near-identical WER to
  large-v3 at half the compute. Runs on GPU (float16) co-resident with
  vLLM — the 4B summarizer + KV cache + ASR fit in 24 GB with margin.
  GPU placement removes the ~2 s/chunk CPU bottleneck the int8/CPU
  variant had, lifting WER (better transcripts → higher B_i) without
  costing latency.
- **Summarizer backend — vLLM in-process.** Chosen over Ollama (server),
  MLX (Apple-only), and raw `transformers` (slower):
  - **No daemon.** Inference is a Python library call. The submission
    Docker has one process to launch; nothing to start, nothing to
    health-check, nothing to crash independently of the pipeline.
  - **Paged attention + batching.** vLLM's continuous batching and
    paged KV cache keep `proc_time` low even at larger model sizes,
    which is exactly the lever for `L_i = 10·e^(−0.5·proc_time)`.
  - **Single-GPU fit.** A 8–9 B fp16 model fits inside the 24 GB
    RTX Pro 4500 with headroom for KV cache; vLLM's
    `gpu_memory_utilization` knob controls the reservation explicitly.
- **Model weights ship with the submission.** `tools/fetch_models.sh`
  populates `./models/<name>/` during development; the same script is
  invoked at `docker build` time so the resulting image carries weights.
  The pipeline aborts at warmup if the directory is missing — failures
  are loud, never silent.
- **Single summarizer profile.** One Juturna node, one JSON profile
  under `pipelines/summarizer/vllm-<name>.json`. No MLX/Ollama variants
  to maintain or drift between.
- **Judge runs offline, sequentially.** `tools/eval_quality.py` is the
  sole judge path. After the pipeline exits and the summarizer is
  unloaded, the judge model is loaded with vLLM, scores every window
  in `results/.../`, and writes `results/.../judge/window_N.json`. This
  avoids VRAM co-residency (an 8 B summarizer + 7 B judge cannot share
  24 GB at fp16) and reflects reality: the held-out judge is the only
  one that matters at submission time.
- **Prompt engineering for conciseness + specificity.** Prompts demand
  1–2 sentences, facts present in the transcript only, and specific
  noun phrases for keywords. This doubles up: higher `B_i` conciseness
  score and lower `proc_time` (smaller `max_tokens` suffices), which
  boosts `L_i`.
- **Keyword validation.** `ensure_three_keywords` strips banned generic
  terms (`general`, `discussion`, `meeting`, …) that would be scored as
  irrelevant, and backfills missing slots by extracting proper-noun /
  content words from the transcript. Turns the previous `−6` worst
  case on LLM failure into a `+0` floor: even a broken summarizer no
  longer tanks `K_i`.
- **Robust JSON extraction.** The summarizer strips ChatML special
  tokens (`<|im_end|>`, `<|endoftext|>`) and extracts the outermost
  balanced JSON object from the raw output — tolerates trailing stop
  tokens or commentary the model may emit after the JSON.
- **Hallucination filter.** Inline post-processing guards against
  invented names, numbers, or facts — a known failure mode for small
  LLMs on short transcripts. Protects `B_i` factual-consistency.
- **Quality gates latency.** The scoring rule zeroes `L_i` when
  `B_i < 10`. The pipeline is tuned so each window clears that floor
  before optimizing latency further. Bigger models / longer prompts are
  preferred to faster but lower-quality combinations.
- **Extractive fallback (never empty summary).** The summarizer node has
  two guarded paths into a deterministic extractive summary (first
  sentence(s) of the transcript, capped at 300 chars) under
  `_summarizer_common/extractive.py`: (a) transcripts shorter than
  `min_transcript_chars=80` skip the LLM entirely; (b) LLM failures
  (parse error, empty summary, exception) fall through to the same
  helper. Output is always non-empty, faithful (verbatim), and
  effectively zero proc_time — preserves the L bonus on edge windows.
- **End-of-stream flush watchdog.** The window aggregator runs an
  inactivity watchdog (default 30 s). When no chunks arrive for
  `flush_timeout` seconds, the partial window is flushed with its
  actual `chunk_end` (not `window_start + duration`). Captures the tail
  of finite audio that would otherwise be dropped.

## Measured performance

Per-window averages on the rev16 oneill fixture (~75 min audio,
15 × 300 s windows), `tools/eval_quality.py` with the
`vllm-mistral-small-24b-awq` judge:

| Model | B (≤25) | K (≤6) | L (≤10) | C (avg) | Janus | **Final** |
|-------|---------|--------|---------|---------|-------|-----------|
| Qwen3.5-4B (base) | 22.0 | 6.0 | 5.69 | 33.69 | +4 | **37.69** |

Container smoke-tested end-to-end with a Crime Town podcast preview
(~3 min): all 7 windows landed with the challenge-format schema;
`proc_time` 0.22–0.78 s on LLM windows (`L_i ≈ 7.8`), <1 ms on
extractive-fallback windows. The end-of-stream flush captured the
final 4 s partial window correctly.

## Submission packaging

- **`Dockerfile`** targets `nvidia/cuda:12.4.1-runtime-ubuntu22.04`,
  installs Python 3.12 + uv, runs `uv sync` (vLLM is a base dependency;
  no extras needed in the image). The build then invokes
  `./tools/fetch_models.sh Qwen/Qwen3.5-4B qwen3.5-4b` to bake the
  summarizer weights, and `huggingface_hub.snapshot_download(...)` to
  bake `faster-whisper-large-v3-turbo`. The resulting image carries
  every model byte; **zero network dependency at runtime**.
- **`docker-compose.yml`** runs the Janus service and the pipeline
  service. Janus forwards RTP to the pipeline container on UDP/8888.
  `WINDOW_SECONDS` and `SUMMARIZER_PROFILE` are env-overridable for
  rapid iteration (`WINDOW_SECONDS=30 docker compose up -d`).
- **Entry point** `docker/entrypoint-pipeline.sh` calls
  `tools/run_pipeline.sh --window $WINDOW_SECONDS --summarizer
  $SUMMARIZER_PROFILE`. The launch uses juturna's `--auto` flag so the
  container starts without interactive input.
- **`destination_endpoint`** is left empty in the committed
  `pipelines/config-base.json`. Per the challenge spec, the receiving
  URL was not published as of submission; result files are written to
  `./results/<model>/<window_s>/window_N.json` and the host volume
  mount exposes them. **Set this string to the evaluation endpoint and
  the transmitter will POST every window** (httpx, configurable
  timeout) in addition to the local write.
- **`.dockerignore`** keeps the image lean (~20 GB content vs ~150 GB
  with default context) by excluding `models/` (baked separately),
  `datasets/`, `results/`, `.venv/`, and `tools/finetune/runs/`.
