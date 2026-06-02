# Things To Do

Active plan: [`docs/plans/cuda-native-pipeline-base-first.md`](plans/cuda-native-pipeline-base-first.md).

## Milestones

- [x] **M0 — Cleanup.** Removed MLX/Ollama nodes, configs, prompts, tests; dropped `ollama`/`mlx-lm` deps; added `cuda` extra; rewrote CLAUDE.md + skills + docs to CUDA-only path.
- [x] **M1 — Base CUDA pipeline.** vLLM in-process summarizer, `tools/fetch_models.sh`, profiles, `run_pipeline.sh`, `eval_quality.py`, unit tests (mocked vLLM). Smoke-tested on uni-lab. Output keys `{from, to, summary, keywords[3], proc_time}` verified. Aggregator inactivity watchdog fixed end-of-audio flush. Vision-encoder VRAM waste fixed via `limit_mm_per_prompt`.
- [x] **M2 — Prompt + sampling tune.** ASR on GPU float16; beam_size/no_speech_threshold exposed. Prompt locked at 2–3 sentences + mixed topic/entity keywords (v2). Hallucination filter expanded. Best multi-judge score: `qwen3.5-4b` 38.02–38.33.
- [ ] **M3 — Submission packaging (DEFERRED).** Rewrite `Dockerfile` (CUDA base, weights baked via `tools/fetch_models.sh`), populate `destination_endpoint` (currently `""`), end-to-end container smoke, finalize `docs/APPROACH.md`, build submission bundle. **Model locked: `qwen3.5-4b-awq` (profile `vllm-qwen3.5-4b-awq.json`).** M3 unblocked.
  - [x] faster-whisper large-v3-turbo baked into Docker image; transcriber auto-resolves `./models/faster-whisper-large-v3-turbo`.
- [~] **M4 — Finetune (abandoned for submission).** QLoRA retargeted to vLLM; three distillation passes all scored below base on multi-judge panel. Decision: dropped. FT profiles/results kept for record only.

## Next Steps (active — M3 packaging)

Model locked: **`qwen3.5-4b-awq`** (profile `vllm-qwen3.5-4b-awq.json`, dir `./models/qwen3.5-4b-awq/`).

- [x] **Model locked.** `qwen3.5-4b-awq` selected as submission summarizer.
- [ ] **Record locked B/K/L/C breakdown in `docs/APPROACH.md`.**
- [ ] **M3: Dockerfile** — bake `qwen3.5-4b-awq` weights, populate `destination_endpoint`, end-to-end container smoke test.

### Future work (post-submission)

- **Finetune:** revisit only if a teacher/data change demonstrably beats base on multi-judge panel.

## Completed Standalone Items

- [x] **Multi-judge eval (de-bias).** `tools/eval_multi_judge.py` runs fixed cross-family panel (Mistral-Small-24B-AWQ + Phi-4-AWQ + Gemma-3-27B-it-int4-AWQ) sequentially; combined `judge_report_multi.txt` + `judge_scores_multi.json`; B+K separated from objective L; unit tests in `tests/test_eval_multi_judge.py`.
- [x] **Summarizer comparison runner.** `tools/compare_summarizers.sh` — takes `--audios` + `--profiles`, streams every (audio × profile) pair, scores with fixed 3-judge panel, emits leaderboard. Preflight verifies GPU + all model dirs; aborts if first-run coverage < 0.85. Bake-offs run across rev16 (ep10, ep11, ep27) and IETF (Computing-Aware Tr, MoQ). `tools/postproc/rank_audio.py` generates cross-stamp rankings per audio dir.
- [x] **Pipeline truncation fix.** Aggregator inactivity watchdog (30s default) flushes partial window on audio end.
- [x] **ASR upgrade.** `large-v3-turbo` (809M, GPU float16) replacing `small.en`. WER interpretation guide in judge reports.
- [x] **Leaderboard infra.** CSV export from compare_summarizers; `tools/postproc/rebuild_results.py` regenerates reports from existing `judge_scores_multi.json`; cross-stamp `leaderboard.csv` per audio dir.

## Bookmarks

- [vLLM](https://github.com/vllm-project/vllm)
- [Qwen on HuggingFace](https://huggingface.co/Qwen)
- [rev16 dataset (Whisper subset)](https://huggingface.co/datasets/distil-whisper/rev16/tree/main/whisper_subset)
- [MeetingBank dataset](https://meetingbank.github.io/)
