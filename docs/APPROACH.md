# STREAMIND — Approach

> **Status: in flight.** Pipeline runs on native-CUDA vLLM. Measured performance section will be updated during M2 (prompt tune) and M3 (submission packaging).

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
3. `transcriber_whisper` — faster-whisper `large-v3-turbo`, `device: cpu`,
   int8. OpenAI's distilled large-v3 (~809M params) — near large-v3 WER
   at half the size. Runs on CPU to avoid VRAM contention with vLLM;
   processes 5 s chunks in ~2-3 s on modern CPU. `language: "en"` set
   explicitly (multilingual model, no `.en` suffix).
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
  large-v3 (~809M params, ~1.5 GB int8). Near-identical WER to
  large-v3 at half the compute. Runs on CPU (int8) to keep GPU VRAM
  free for vLLM; processes 5 s chunks well within real-time on
  uni-lab/production CPUs. Better transcripts feed higher B_i.
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

## Measured performance

To be populated in M2 against the chosen base model on uni-lab (RTX 4090,
CUDA 12.4). The 30-second-window prototype on Apple Silicon MLX hit
`B=25, K=6, L=2.13, C=33.13` at `proc_time=3.1 s`; native CUDA on a
larger model is expected to improve `B` and roughly hold or improve
`L` after batching.

## Submission packaging (planned, M3)

- `Dockerfile` targets `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`.
  Installs Python 3.12 + uv, runs `uv sync` (vLLM ships as a base
  dependency; no extras needed in the image), and during build runs
  `./tools/fetch_models.sh <hf_id> <local_name>` (the script pulls
  `huggingface-hub` from `--extra dev` for the build stage only) so the
  resulting image carries the model weights. Zero network dependency at
  runtime.
- `docker-compose.yml` runs both the Janus service and the pipeline
  service. Janus forwards RTP to the pipeline container on UDP/8888.
- Entry point: `tools/run_pipeline.sh` with the committed summarizer
  profile and a 300 s window — same command used in development.
