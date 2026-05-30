---
name: prep-submission
description: Pre-submission checklist — validate config, output format, and native CUDA vLLM pipeline on uni-lab before submitting
disable-model-invocation: true
---

Complete each step before submitting. All CUDA testing runs on uni-lab.
Active summarizer profile and model are committed in `pipelines/summarizer/vllm-<name>.json` — substitute the name everywhere below.

1. Set `destination_endpoint` in `pipelines/config-base.json` (currently `""` — submission deadline blocker).
2. Verify `encoding_clock_chan: "opus/48000/1"` (challenge specifies mono Opus).
3. Confirm `pipelines/summarizer/vllm-<name>.json` exists and `configuration.model_name` points to a relative path under `./models/` (never an HF id at runtime).
4. Sync to uni-lab: `rsync -av --exclude='.venv' --exclude='results' --exclude='__pycache__' --exclude='models' . uni-lab:~/Desktop/streamind/`
5. Verify CUDA + vLLM on uni-lab:
   `ssh uni-lab "cd ~/Desktop/streamind && nvidia-smi && uv run python -c 'from vllm import LLM, SamplingParams; print(LLM.__module__)'"`
6. Ensure weights present: `ssh uni-lab "cd ~/Desktop/streamind && ls ./models/<name>/config.json"` — if missing, run `./tools/fetch_models.sh <hf_id> <name>`.
7. 30 s smoke test (audio_rtp via Janus, NOT audio_file):
   `ssh uni-lab "cd ~/Desktop/streamind && ./tools/run_pipeline.sh -w 30 -p vllm-<name>"`
   In parallel: `ssh uni-lab "cd ~/Desktop/streamind && uv run python tools/send_audio.py 'datasets/rev16/10_Creating_Your_Own_Lane_in_Podcasting_ft_@Favyfav_of_@latinoswholunch/audio.opus'"`
8. Pull results: `rsync -av uni-lab:~/Desktop/streamind/results/ ./results/`
9. Validate output JSON: keys MUST be exactly `from`, `to`, `summary`, `keywords` (length 3), `proc_time` — no extras.
10. Offline judge (sequential — summarizer must be unloaded first). `--audio` enables ASR WER (independent of B/K/L):
    `ssh uni-lab "cd ~/Desktop/streamind && uv run python tools/eval_quality.py results/<name>/30/ --judge-profile vllm-<judge_name> --audio 'datasets/rev16/10_Creating_Your_Own_Lane_in_Podcasting_ft_@Favyfav_of_@latinoswholunch/audio.opus'"`
11. Full 300 s test: at least one window fires and POSTs successfully to `destination_endpoint`.
12. Grep guard: `rg -i 'ollama|mlx' --type py --type json -g '!tools/finetune/**' -g '!docs/**'` returns zero matches.
13. VRAM budget during run: `nvidia-smi` peak < 24 GB.
14. **No-network sanity** (M3): docker run with `--network none` still loads model and produces results — proves weights baked into image.
15. Confirm `Dockerfile` includes `RUN ./tools/fetch_models.sh ...` so the image carries the weights (no runtime download).
16. Submission bundle includes: code (`plugins/`, `tools/`), `pipelines/config-base.json` + assembled config, sample results JSONs, `Dockerfile`, `docs/APPROACH.md`.
