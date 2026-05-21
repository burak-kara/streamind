# Things To Do

Active plan: [`docs/plans/cuda-native-pipeline-base-first.md`](plans/cuda-native-pipeline-base-first.md).

## Milestones

- [x] **M0 — Cleanup.** Delete MLX/Ollama nodes, configs, prompts, tests; drop `ollama`/`mlx-lm` deps; add `cuda` extra; rewrite CLAUDE.md + skills + docs to a single native-CUDA path.
- [ ] **M1 — Base CUDA pipeline working.** vLLM in-process summarizer node, `tools/fetch_models.sh`, one summarizer profile, one judge profile, refactored `run_pipeline.sh` + `eval_quality.py`, unit tests green (mocked vLLM). Smoke pipeline + offline judge run on uni-lab against a 30 s and a 30 min audio fixture; output JSON keys exactly `{from, to, summary, keywords[3], proc_time}`.
  - [x] `_summarizer_vllm/` node + `tools/fetch_models.sh` + tests (mocked vLLM passes)
  - [x] Pick concrete model id + judge model id (summarizer: `Qwen/Qwen3.5-4B`; judge: `stelterlab/Mistral-Small-24B-Instruct-2501-AWQ` — swapped 2026-05-20 from `cyankiwi/Qwen3.5-27B-AWQ-BF16-INT4` which OOMed at 26 GB on-disk vs claimed int4)
  - [x] Commit `pipelines/summarizer/vllm-qwen3.5-4b.json` + `pipelines/judge/vllm-mistral-small-24b-awq.json`
  - [x] Source a ≥30 min audio fixture — using `docs/datasets/rev16/10_Creating_Your_Own_Lane_in_Podcasting_ft_@Favyfav_of_@latinoswholunch.opus` (36 min, ~7×300s windows)
  - [x] Smoke run + judge run on uni-lab — 30 s window on rev16 podcast 10 ran; avg C=34.29 (Final 38.29 with Janus +4 / 45), but surfaced three issues (see follow-up block below).
  - [x] Offline judge: ASR WER vs rev16 ground-truth `.txt` (independent signal — does NOT affect B/K/L). `--audio` + `--reference` flags on `eval_quality.py`. Per Plan 2: ASR errors are scored separately so summarizer skill isolated from ASR fidelity.
  - [x] **M1 follow-up: empty-summary windows.** Replaced the silent `summary=""` failure path with a deterministic extractive fallback in `plugins/nodes/proc/_summarizer_common/extractive.py`. Both short-transcript (< `min_transcript_chars=80`) and LLM-failure cases now emit verbatim transcript snippets so every chunk ships a non-empty, factual-consistency-friendly summary at ~0 ms cost. Plus diagnostic logging of exception type + raw output for next-run debugging. See active plan §M1 follow-up A.
  - [x] **M1 follow-up: WER alignment.** Added `_sliding_wer` in `tools/eval_quality.py`: tries slice positions at ±30/20/10 % around the proportional anchor and returns the minimum WER. Also prints a one-shot `WARN` when any judged window is < 120 s (proportional slice on flat ground-truth is noisy at sub-minute granularity). Forced alignment via faster-whisper word timestamps remains a deferred upgrade. See active plan §M1 follow-up B.
  - [ ] **M1 follow-up: pipeline truncation.** Half-done. `tools/send_audio.py` now logs a 60 s heartbeat (`streamed Xs / Ys  pc.state=...`) so mid-stream truncation surfaces in stdout. Still pending: re-run on uni-lab with `tee tmp/pipeline.log` + `tee tmp/send_audio.log`, grep for ICE/disconnect/destroy, then fix the underlying disconnect (likely Janus session timeout or aiortc connection drop). See active plan §M1 follow-up C.
  - [ ] Use audio files under `docs/datasets/` as fixtures, not `tests/fixtures/`
  - [ ] Refactor and cleanup `Dockerfile` and `docker-compose.yml` (remove Ollama + MLX, add CUDA base, bake in weights via `tools/fetch_models.sh` during build)
  - [ ] **Model variant audit** — `Qwen/Qwen3.5-4B` resolves as `Qwen3_5ForConditionalGeneration` (multimodal w/ Qwen2VL image processor); vision encoder cache + image-item profiling allocate wasted VRAM. Investigate text-only variants (other providers, distilled bases). Reweight against B_i quality before swap.
  - [ ] **Extend summarizer warmup** — current `Say hello.` (max_tokens=5) warmup does NOT trigger Triton JIT for `_zero_kv_blocks_kernel`, `_compute_slot_mapping_kernel`, `_causal_conv1d_fwd_kernel` (mamba), `_fused_post_conv_kernel`. JIT fires during first real window → inflated proc_time → poisons L_i for window_0. Warmup with ~1500-token dummy transcript matching real chunk shape.
- [ ] **M2 — Prompt + sampling tune.** Iterate `summarize_prompt.txt` and SamplingParams on the chosen model; rerun offline judge; lock the best prompt; record B/K/L/C in `docs/APPROACH.md`.
- [ ] **M3 — Submission packaging.** Rewrite `Dockerfile` (CUDA base, weights baked via `tools/fetch_models.sh` during build), populate `destination_endpoint`, end-to-end smoke inside container, finalize `docs/APPROACH.md`, build submission bundle (code + config + sample results + Dockerfile + approach).
  - [ ] **Bake faster-whisper into Docker image** — pipeline launch issues HTTP GET to `huggingface.co/api/models/Systran/faster-whisper-small.en` even though model is the runtime "auto-download exception". Submission container has no network; must pre-populate the HF cache during `docker build`.
- [ ] **M4 — Finetune (later).** Rewrite `prepare_rev16.py` teacher distillation off Ollama (use vLLM in-process), run QLoRA → merge → swap merged dir into the vLLM profile, A/B against base via `tools/eval_quality.py`. Gated on M3.

## Bookmarks

- [vLLM](https://github.com/vllm-project/vllm)
- [Qwen on HuggingFace](https://huggingface.co/Qwen)
- [rev16 dataset (Whisper subset)](https://huggingface.co/datasets/distil-whisper/rev16/tree/main/whisper_subset)
- [MeetingBank dataset](https://meetingbank.github.io/)
