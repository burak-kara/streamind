# STREAMIND — Approach

## Challenge restatement

Real-time meeting intelligence from a live audio stream. Per-chunk score
`C_i = B_i + K_i + L_i` (B: LLM judge, 5 Likert criteria incl. conciseness,
max 25; K: keyword relevance, max 6; L: latency reward,
`10·e^(−0.5·proc_time)` gated by `B_i ≥ 10`). Audio-level score averages
chunk scores and adds a flat `+4` Janus bonus.

## Pipeline architecture

Seven sequential Juturna nodes, wired in `pipelines/config-base.json`:

1. `audio_rtp` — receives mono Opus/48k RTP on `0.0.0.0:8888`. Janus
   gateway forwards audio into this port. Config: `channels: 1`,
   `encoding_clock_chan: "opus/48000/1"`.
2. `audio_chunker` — splits the live stream into 5s overlapping chunks
   (1s overlap) so the ASR sees audio incrementally, not as one blob.
3. `transcriber_whisper` — faster-whisper `small.en`, `device: auto`,
   int8. Cheap on CPU, fast on GPU. English-only because the challenge
   transcripts are English.
4. `hallucination_filter` — post-processes ASR output to drop known
   hallucinatory sentences before aggregation.
5. `novel_extractor` — deduplicates the overlapping ASR output, keeps
   only the new content each chunk contributes.
6. `window_aggregator` — concatenates novel transcript segments into a
   rolling 300-second context window. Emits one message per completed
   window.
7. `summarizer_mlx` (Apple Silicon) or `summarizer_llm` (Ollama,
   CUDA-ready) — produces `{summary, keywords: [x3]}` JSON via a local
   LLM.
8. `result_transmitter` — maps internal keys to the challenge contract
   (`from`, `to`, `summary`, `keywords`, `proc_time`), writes the result
   locally, and POSTs to `destination_endpoint` when configured.

## Key design decisions

- **Janus WebRTC bonus.** Audio enters through a Janus gateway in both
  local dev and the evaluation environment — no environment-specific code
  paths. This earns the flat `+4` audio-level bonus at zero architectural
  cost.
- **ASR model choice — faster-whisper `small.en`.** Accuracy/latency
  sweet spot on English audio. `medium` adds measurable latency without a
  quality win big enough to matter for a 1-2 sentence summary.
- **Summarizer choice — Qwen3.5 4B via Ollama.** Evaluation hardware is
  one RTX Pro 4500 (no MLX). 4B fits with room to spare and scored at
  ceiling on the internal judge harness at 3.1s proc_time on Apple
  Silicon MLX; CUDA is expected to be faster still. MLX profiles are
  kept for local Apple Silicon development.
- **Two summarizer nodes, not a shim.** `summarizer_mlx` (native MLX)
  and `summarizer_llm` (Ollama) are separate Juturna nodes with the same
  output schema. Selecting the production path is a one-line config
  change, not a runtime branch. Shared keyword post-processing lives in
  `plugins/nodes/proc/_summarizer_common/keywords.py` so both nodes stay
  in lock-step.
- **Prompt engineering for conciseness + specificity.** Prompts demand
  1-2 sentences, facts present in the transcript only, and specific noun
  phrases for keywords. This doubles up: higher `B_i` conciseness score
  and lower `proc_time` (smaller `num_predict` suffices), which boosts
  `L_i`.
- **Keyword validation.** `ensure_three_keywords` strips banned generic
  terms (`general`, `discussion`, `meeting`, …) that would be scored as
  irrelevant, and backfills missing slots by extracting proper-noun /
  content words from the transcript. This turns the previous `−6` worst
  case on LLM failure into a `+0` floor: even a broken summarizer no
  longer tanks `K_i`.
- **Robust JSON extraction.** Both summarizer nodes strip ChatML special
  tokens (`<|im_end|>`, `<|endoftext|>`) and extract the outermost
  balanced JSON object from the raw LLM output. Caught and fixed after
  a smoke test where `mlx-lm` left the stop token attached and every
  window emitted an empty summary.
- **Hallucination filter.** Inline post-processing guards against
  invented names, numbers, or facts — a known failure mode for small
  LLMs on short transcripts. Protects `B_i` factual-consistency.
- **Quality gates latency.** The scoring rule zeroes `L_i` when
  `B_i < 10`. The pipeline is tuned so each window clears that floor
  before optimizing latency further. Bigger models / longer prompts are
  preferred to faster but lower-quality combinations.

## Novel contributions

- **Banned-keyword filter + transcript-derived backfill** in
  `ensure_three_keywords`. Fixes a `−6 per chunk` failure mode where the
  error path and padding both emitted generic terms.
- **Single-image Docker submission.** Ollama + pre-pulled
  `qwen3.5:4b` model baked into the CUDA image, started alongside the
  pipeline via an entrypoint, so the submission boots end-to-end with
  one `docker compose up`.
- **Local quality harness (`tools/eval_quality.py`).** Runs the
  challenge rubric against our own outputs with a local judge model, so
  prompt and model changes can be measured before committing them.

## Measured performance

Local Apple Silicon run on a 120s slice of `youtube_15min.wav` @ 30s
windows, MLX 4B profile, judge = local `qwen3.5:9b` via
`tools/eval_quality.py`. Numbers are directional — the hidden judge may
score differently.

| State | B_i | K_i | L_i | C_i | proc_time |
|---|---|---|---|---|---|
| Pre-Phase-1 baseline (30s, 14 windows) | 24.7 | 6.0 | 1.34 | 32.06 | 4.09s |
| Pre-Phase-1 baseline (300s, 3 windows) | 25.0 | 6.0 | 0.09 | 31.09 | 9.56s |
| **Post-Phase-1 smoke (30s, 3 windows)** | **25.0** | **6.0** | **2.13** | **33.13** | **3.10s** |

Phase-1 changes (tightened prompts, `num_predict: 150`, banned-keyword
filter, JSON extraction fix) cut `proc_time` by 24 % and lifted `L_i` by
59 % against the 30s-window baseline, holding `B_i` and `K_i` at
ceiling. With the Janus `+4` audio-level bonus, expected audio-level
score on this fixture is `≈ 37`. CUDA numbers on the RTX Pro 4500 are
expected to improve proc_time further; the `L_i` curve flattens at
roughly 1 s (`L ≈ 6.1`).

## Submission packaging

- `Dockerfile` targets `nvidia/cuda:12.3.0-runtime-ubuntu22.04`,
  installs Python 3.12 + uv, syncs project deps, installs Ollama, and
  pulls `qwen3.5:4b` at build time so the container is self-contained
  at runtime.
- `docker-compose.yml` runs both the Janus service and the pipeline
  service. Janus forwards RTP to the pipeline container on UDP/8888.
- Entry point: `docker/entrypoint-pipeline.sh` starts Ollama, waits
  for readiness, then launches `tools/run_pipeline.sh` with the
  submission profile (`ollama-qwen3.5-4b`, 300s window).
