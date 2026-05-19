# Things To Do

Active plan: [`docs/plans/cuda-native-pipeline-base-first.md`](plans/cuda-native-pipeline-base-first.md).

## Milestones

- [x] **M0 — Cleanup.** Delete MLX/Ollama nodes, configs, prompts, tests; drop `ollama`/`mlx-lm` deps; add `cuda` extra; rewrite CLAUDE.md + skills + docs to a single native-CUDA path.
- [ ] **M1 — Base CUDA pipeline working.** vLLM in-process summarizer node, `tools/fetch_models.sh`, one summarizer profile, one judge profile, refactored `run_pipeline.sh` + `eval_quality.py`, unit tests green (mocked vLLM). Smoke pipeline + offline judge run on uni-lab against a 30 s and a 30 min audio fixture; output JSON keys exactly `{from, to, summary, keywords[3], proc_time}`.
  - [x] `_summarizer_vllm/` node + `tools/fetch_models.sh` + tests (mocked vLLM passes)
  - [x] Pick concrete model id + judge model id (summarizer: `Qwen/Qwen3.5-4B`; judge: `cyankiwi/Qwen3.5-27B-AWQ-BF16-INT4`)
  - [x] Commit `pipelines/summarizer/vllm-qwen3.5-4b.json` + `pipelines/judge/vllm-qwen3.5-27b-awq.json`
  - [ ] Source a ≥30 min audio fixture (current `youtube_15min.wav` only yields 3 windows)
  - [ ] Smoke run + judge run on uni-lab
- [ ] **M2 — Prompt + sampling tune.** Iterate `summarize_prompt.txt` and SamplingParams on the chosen model; rerun offline judge; lock the best prompt; record B/K/L/C in `docs/APPROACH.md`.
- [ ] **M3 — Submission packaging.** Rewrite `Dockerfile` (CUDA base, weights baked via `tools/fetch_models.sh` during build), populate `destination_endpoint`, end-to-end smoke inside container, finalize `docs/APPROACH.md`, build submission bundle (code + config + sample results + Dockerfile + approach).
- [ ] **M4 — Finetune (later).** Rewrite `prepare_rev16.py` teacher distillation off Ollama (use vLLM in-process), run QLoRA → merge → swap merged dir into the vLLM profile, A/B against base via `tools/eval_quality.py`. Gated on M3.

## Bookmarks

- [vLLM](https://github.com/vllm-project/vllm)
- [Qwen on HuggingFace](https://huggingface.co/Qwen)
- [rev16 dataset (Whisper subset)](https://huggingface.co/datasets/distil-whisper/rev16/tree/main/whisper_subset)
- [MeetingBank dataset](https://meetingbank.github.io/)
