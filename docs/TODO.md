# Things To Do

Active plan: [`docs/plans/cuda-native-pipeline-base-first.md`](plans/cuda-native-pipeline-base-first.md).

## Milestones

- [x] **M0 — Cleanup.** Delete MLX/Ollama nodes, configs, prompts, tests; drop `ollama`/`mlx-lm` deps; add `cuda` extra; rewrite CLAUDE.md + skills + docs to a single native-CUDA path.
- [x] **M1 — Base CUDA pipeline working.** vLLM in-process summarizer node, `tools/fetch_models.sh`, one summarizer profile, one judge profile, refactored `run_pipeline.sh` + `eval_quality.py`, unit tests green (mocked vLLM). Smoke pipeline + offline judge run on uni-lab against a 30 s and a 30 min audio fixture; output JSON keys exactly `{from, to, summary, keywords[3], proc_time}`.
  - [x] Core pipeline: vLLM node, model selection, profiles, smoke runs, offline judge, extractive fallback, WER alignment, warmup, Dockerfile, fixtures migration.
  - [x] Pipeline truncation / end-of-audio flush — fixed via aggregator inactivity watchdog (see standalone item below).
  - [x] Model variant audit — done. Vision-encoder VRAM waste fixed via `limit_mm_per_prompt` (Qwen3.5/Gemma OOM commits). Newer base `Qwen3-4B-Instruct-2507` tested and now leads (~39.4, see Model bake-off below).
- [x] **M2 — Prompt + sampling tune.** Iterate `summarize_prompt.txt` and SamplingParams on the chosen model; rerun offline judge; lock the best prompt.
  - [x] ASR: exposed beam_size, no_speech_threshold, initial_prompt params; switched to GPU float16
  - [x] Prompt: iterated through 3 versions (longer summaries → named-entity keywords → 2-3 sentences for latency). Locked at 2-3 sentences + mixed topic/entity keywords.
  - [x] Hallucination filter: expanded with common Whisper noise phrases
  - [x] Eval: A/B results in `results/qwen3.5-4b-prompt-tune{,-v2}/`. Best C=35.58 (baseline), v2 C=32.15 (higher R-L/K-J but worse K/L).
  - [ ] Record final B/K/L/C in `docs/APPROACH.md` (deferred to M3).
- [ ] **M3 — Submission packaging (DEFERRED — future work).** Rewrite `Dockerfile` (CUDA base, weights baked via `tools/fetch_models.sh` during build), populate `destination_endpoint` (still `""`), end-to-end smoke inside container, finalize `docs/APPROACH.md`, build submission bundle (code + config + sample results + Dockerfile + approach). Held until the model is locked (see Next Steps) so the image bakes the final pick once.
  - [x] **Bake faster-whisper into Docker image** — `Dockerfile` now runs `snapshot_download('deepdml/faster-whisper-large-v3-turbo-ct2')` at build; transcriber node auto-resolves `./models/faster-whisper-large-v3-turbo` if present.
  - [x] **Upgrade ASR to large-v3-turbo** — Switched from `small.en` (244M) to `large-v3-turbo` (809M, OpenAI distilled large-v3). Near large-v3 WER at half compute. Now GPU float16 (was CPU int8). Added WER interpretation guide to judge reports.
- [~] **M4 — Finetune (attempted — regressed below base; on hold).** QLoRA workflow retargeted off Ollama to vLLM runtime (`tools/finetune/finetune_summarizer.py` r=16/alpha=32, `merge_lora.py`). Three distillation passes evaluated; **all scored below base** on oneill/300s single-judge:
  - `qwen3.5-4b-ft` (plain) = 34.59
  - `qwen3.5-4b-ft-gemini` = 35.98
  - `qwen3.5-4b-ft-full-gemini` (full Gemini relabel) = 36.47
  - vs base `qwen3.5-4b` = 38.02–38.33 (multi-judge), `qwen3-4b-instruct-2507` = 39.43.
  - **Caveat:** FT scored single-judge, base multi-judge — not strict apples-to-apples. FT runs were never re-scored on the multi-judge panel. Gemini relabel monotonically improved FT but never closed the gap to base.
  - **Decision (2026-05-30): dropped for submission.** Consistent regression vs base; not worth the added pipeline complexity. FT profiles/results kept for the record, not shipped. Revisit only as post-submission research.

## Next Steps (active — lock the model)

Priority before any packaging: settle the summarizer pick on a fair, single panel. Current numbers mix single- and multi-judge runs and different dataset slices, so they don't strictly compare.

- [ ] **Same-panel bake-off: `qwen3-4b-instruct-2507` vs `qwen3.5-4b`.** Run both through `tools/eval_multi_judge.py` on the *same* dataset (oneill 300s) with the *same* 3-judge panel. Pick the winner as the locked submission summarizer.
- [ ] **Confirm the locked pick's profile is production-clean** — sampling params, `limit_mm_per_prompt` vision-cache skip, prompt template (locked V2). No leftover experimental flags.
- [ ] **Record the locked B/K/L/C (per-criterion breakdown) in `docs/APPROACH.md`.** Carries over the long-open "record final numbers" item from M2.
- [ ] Once locked, unblock M3 (Dockerfile bakes the final model, endpoint, container smoke).

### Future work (post-lock / post-submission)

- **Model upgrade candidate:** `Qwen3-4B-Instruct-2507` scored ~39.43 (single-judge) vs `qwen3.5-4b` 38.02–38.33 (multi-judge) — currently the strongest base. Submission Dockerfile still bakes `qwen3.5-4b`; switching requires re-baking the image. Deferred with the rest of M3 packaging.
- **Finetune (abandoned for submission):** revisit only as research — needs a teacher/data change that demonstrably beats base on the multi-judge panel before reconsidering.

- [x] ~~Should we warmup the transcript model to GPU?~~ — Addressed: ASR now runs GPU float16 (`device: auto`).
- [x] **Pipeline truncation / end-of-audio handling.** Fixed: window_aggregator now has an inactivity watchdog (default 30s). When no chunks arrive for `flush_timeout` seconds, partial window is flushed with correct `window_end` (actual last `chunk_end`, not `window_start + duration`). Root cause: audio_rtp (framework) treats ffmpeg exit code 0 as crash and loops forever — cannot be modified, so watchdog works around it.

- [x] **Multi-judge eval (de-bias scores) — STANDARD eval going forward (2026-05-30).** All model comparisons use the fixed cross-family panel; single-judge `eval_quality.py` is retained only as a per-judge component (no longer the comparison basis). `tools/eval_multi_judge.py` runs the panel (Mistral-Small-24B-AWQ + Phi-4-AWQ + Gemma-3-27B-it-int4-AWQ) sequentially, one subprocess per judge for full VRAM reclaim. Combined `judge_report_multi.txt` + `judge_scores_multi.json` report per-judge C side-by-side + consensus mean/stdev (stdev = bias signal). `eval_quality.py` gained `--judge-tag` to namespace per-window output. Unit tests in `tests/test_eval_multi_judge.py`. Multi-judge reports produced for base models (oneill 300s): `qwen3.5-4b` 38.02, archived-oneill 38.33, `qwen3.5-9b` 35.67.

- [ ] **Summarizer comparison runner — `tools/compare_summarizers.sh` (this branch).** Unified the two overnight scripts (`run_overnight_battery.sh` + `run_overnight_3model_multijudge.sh`, both deleted) into one generic runner: takes any audio file + a list of summarizer profiles, streams each through Janus at the default 300s window, then scores with the fixed 3-judge panel and prints a leaderboard. Preflight fails fast (audio, GPU, every summarizer + judge model dir) before any multi-hour run; aborts if first-run coverage < 0.85 (truncation guard); waits for GPU to drain between phases. Result dir derived from `basename(model_name)` to match `result_transmitter` (fixes the old `._models_` bug).
  - [ ] **Pending: e2e run on uni-lab** — `./tools/compare_summarizers.sh --audio datasets/rev16/oneill/audio.opus --profiles vllm-qwen3-4b-2507 vllm-qwen3.5-4b` is the bake-off that locks the submission model (see Next Steps).

- [ ] Try different chunk durations and overlaps to find the optimal balance between latency and summary quality. Shorter chunks may reduce latency but could lead to less coherent summaries, while longer chunks may improve summary quality at the cost of increased latency.
- [ ] Make sure the ASR model choice is configured via `plugins/nodes/proc/_transcriber_whisper/config.toml`.
- [ ] Make sure audio reception is working robustly in Janus (deferred — no known issues yet, but we haven't done long runs on uni-lab with the new pipeline).
  - [ ] No costumization. Janus should be a black box that feeds us audio chunks; we shouldn't have to modify it or worry about its internals.
  - [ ] Check the compatiblity with the latest Janus version. We use 0.x but Janus is now at 1.x — need to verify that our Janus client code still works and that there are no regressions in audio reception.
- [ ] Feed run logs to identfy any issue.
  - [ ] Add more debug logging around audio reception, chunk processing, and model inference to identify any bottlenecks or failures during long runs.
  - [ ] Monitor GPU utilization and memory usage to ensure the pipeline is running efficiently and to catch any potential OOM issues early.

## Bookmarks

- [vLLM](https://github.com/vllm-project/vllm)
- [Qwen on HuggingFace](https://huggingface.co/Qwen)
- [rev16 dataset (Whisper subset)](https://huggingface.co/datasets/distil-whisper/rev16/tree/main/whisper_subset) — now per-episode dirs with `chunks.json` ground truth
- [MeetingBank dataset](https://meetingbank.github.io/)
