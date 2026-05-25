# Things To Do

Active plan: [`docs/plans/cuda-native-pipeline-base-first.md`](plans/cuda-native-pipeline-base-first.md).

## Milestones

- [x] **M0 — Cleanup.** Delete MLX/Ollama nodes, configs, prompts, tests; drop `ollama`/`mlx-lm` deps; add `cuda` extra; rewrite CLAUDE.md + skills + docs to a single native-CUDA path.
- [x] **M1 — Base CUDA pipeline working.** vLLM in-process summarizer node, `tools/fetch_models.sh`, one summarizer profile, one judge profile, refactored `run_pipeline.sh` + `eval_quality.py`, unit tests green (mocked vLLM). Smoke pipeline + offline judge run on uni-lab against a 30 s and a 30 min audio fixture; output JSON keys exactly `{from, to, summary, keywords[3], proc_time}`.
  - [x] Core pipeline: vLLM node, model selection, profiles, smoke runs, offline judge, extractive fallback, WER alignment, warmup, Dockerfile, fixtures migration.
  - [ ] **Pipeline truncation (deferred).** `send_audio.py` heartbeat logging added. Root cause (Janus session timeout vs aiortc drop) not yet diagnosed — needs uni-lab log capture run.
  - [ ] **Model variant audit (in progress — teammate).** Qwen3.5-4B multimodal arch wastes VRAM on vision encoder. Investigating text-only variants.
- [ ] **M2 — Prompt + sampling tune.** Iterate `summarize_prompt.txt` and SamplingParams on the chosen model; rerun offline judge; lock the best prompt; record B/K/L/C in `docs/APPROACH.md`. The summary lenght can be longer given the provided example chunks in `datasets/rev16`.
- [ ] **M3 — Submission packaging.** Rewrite `Dockerfile` (CUDA base, weights baked via `tools/fetch_models.sh` during build), populate `destination_endpoint`, end-to-end smoke inside container, finalize `docs/APPROACH.md`, build submission bundle (code + config + sample results + Dockerfile + approach).
  - [x] **Bake faster-whisper into Docker image** — `Dockerfile` now runs `snapshot_download('deepdml/faster-whisper-large-v3-turbo-ct2')` at build; transcriber node auto-resolves `./models/faster-whisper-large-v3-turbo` if present.
  - [x] **Upgrade ASR to large-v3-turbo** — Switched from `small.en` (244M) to `large-v3-turbo` (809M, OpenAI distilled large-v3). Near large-v3 WER at half compute. Runs CPU int8 to avoid VRAM contention. Added WER interpretation guide to judge reports.
- [ ] **M4 — Finetune (later).** Rewrite `prepare_rev16.py` teacher distillation off Ollama (use vLLM in-process), run QLoRA → merge → swap merged dir into the vLLM profile, A/B against base via `tools/eval_quality.py`. Gated on M3.

- [ ] Should we warmup the transcript model to GPU?
- [ ] When the audio ends, the pipeline should create the regular output instead of hanging or failing. Currently, no summary is produced, and the judge waits indefinitely for the summary until it times out. This is likely due to the vLLM node not receiving any more audio chunks and not producing any output, causing the judge to wait indefinitely.

## Bookmarks

- [vLLM](https://github.com/vllm-project/vllm)
- [Qwen on HuggingFace](https://huggingface.co/Qwen)
- [rev16 dataset (Whisper subset)](https://huggingface.co/datasets/distil-whisper/rev16/tree/main/whisper_subset) — now per-episode dirs with `chunks.json` ground truth
- [MeetingBank dataset](https://meetingbank.github.io/)
