# Things To Do

Active plan: [`docs/plans/cuda-native-pipeline-base-first.md`](plans/cuda-native-pipeline-base-first.md).

## Milestones

- [x] **M0 — Cleanup.** Delete MLX/Ollama nodes, configs, prompts, tests; drop `ollama`/`mlx-lm` deps; add `cuda` extra; rewrite CLAUDE.md + skills + docs to a single native-CUDA path.
- [x] **M1 — Base CUDA pipeline working.** vLLM in-process summarizer node, `tools/fetch_models.sh`, one summarizer profile, one judge profile, refactored `run_pipeline.sh` + `eval_quality.py`, unit tests green (mocked vLLM). Smoke pipeline + offline judge run on uni-lab against a 30 s and a 30 min audio fixture; output JSON keys exactly `{from, to, summary, keywords[3], proc_time}`.
  - [x] Core pipeline: vLLM node, model selection, profiles, smoke runs, offline judge, extractive fallback, WER alignment, warmup, Dockerfile, fixtures migration.
  - [ ] **Pipeline truncation (deferred).** Two issues: (1) Janus session may drop early (timeout vs aiortc — not yet diagnosed); (2) end-of-audio flush — last partial window never summarized. See standalone item below.
  - [ ] **Model variant audit (in progress — teammate).** Qwen3.5-4B multimodal arch wastes VRAM on vision encoder. Investigating text-only variants.
- [x] **M2 — Prompt + sampling tune.** Iterate `summarize_prompt.txt` and SamplingParams on the chosen model; rerun offline judge; lock the best prompt.
  - [x] ASR: exposed beam_size, no_speech_threshold, initial_prompt params; switched to GPU float16
  - [x] Prompt: iterated through 3 versions (longer summaries → named-entity keywords → 2-3 sentences for latency). Locked at 2-3 sentences + mixed topic/entity keywords.
  - [x] Hallucination filter: expanded with common Whisper noise phrases
  - [x] Eval: A/B results in `results/qwen3.5-4b-prompt-tune{,-v2}/`. Best C=35.58 (baseline), v2 C=32.15 (higher R-L/K-J but worse K/L).
  - [ ] Record final B/K/L/C in `docs/APPROACH.md` (deferred to M3).
- [ ] **M3 — Submission packaging.** Rewrite `Dockerfile` (CUDA base, weights baked via `tools/fetch_models.sh` during build), populate `destination_endpoint`, end-to-end smoke inside container, finalize `docs/APPROACH.md`, build submission bundle (code + config + sample results + Dockerfile + approach).
  - [x] **Bake faster-whisper into Docker image** — `Dockerfile` now runs `snapshot_download('deepdml/faster-whisper-large-v3-turbo-ct2')` at build; transcriber node auto-resolves `./models/faster-whisper-large-v3-turbo` if present.
  - [x] **Upgrade ASR to large-v3-turbo** — Switched from `small.en` (244M) to `large-v3-turbo` (809M, OpenAI distilled large-v3). Near large-v3 WER at half compute. Now GPU float16 (was CPU int8). Added WER interpretation guide to judge reports.
- [ ] **M4 — Finetune (later).** Rewrite `prepare_rev16.py` teacher distillation off Ollama (use vLLM in-process), run QLoRA → merge → swap merged dir into the vLLM profile, A/B against base via `tools/eval_quality.py`. Gated on M3.

- [x] ~~Should we warmup the transcript model to GPU?~~ — Addressed: ASR now runs GPU float16 (`device: auto`).
- [x] **Pipeline truncation / end-of-audio handling.** Fixed: window_aggregator now has an inactivity watchdog (default 30s). When no chunks arrive for `flush_timeout` seconds, partial window is flushed with correct `window_end` (actual last `chunk_end`, not `window_start + duration`). Root cause: audio_rtp (framework) treats ffmpeg exit code 0 as crash and loops forever — cannot be modified, so watchdog works around it.

- [ ] Make sure audio reception is working robustly in Janus (deferred — no known issues yet, but we haven't done long runs on uni-lab with the new pipeline).
  - [ ] No costumization. 
  - [ ] Check the compatiblity with the latest Janus version. We use 0.x but Janus is now at 1.x — need to verify that our Janus client code still works and that there are no regressions in audio reception.

## Bookmarks

- [vLLM](https://github.com/vllm-project/vllm)
- [Qwen on HuggingFace](https://huggingface.co/Qwen)
- [rev16 dataset (Whisper subset)](https://huggingface.co/datasets/distil-whisper/rev16/tree/main/whisper_subset) — now per-episode dirs with `chunks.json` ground truth
- [MeetingBank dataset](https://meetingbank.github.io/)
