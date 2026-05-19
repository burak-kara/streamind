---
name: prep-submission
description: Pre-submission checklist — validate config, output format, and CUDA pipeline on uni-lab before submitting
disable-model-invocation: true
---

Complete each step before submitting. All CUDA testing runs on uni-lab.

1. Set `destination_endpoint` in `pipelines/config-base.json` (currently `""`)
2. Verify `encoding_clock_chan` matches actual Janus stream: mono=`opus/48000/1`, stereo=`opus/48000/2`
3. Confirm summarizer profile is `ollama-qwen3.5-4b` (submission CUDA default)
4. Pull model on uni-lab if absent: `ssh uni-lab "ollama pull qwen3.5:4b"`
5. Sync to uni-lab: `rsync -av --exclude='.venv' --exclude='results' --exclude='__pycache__' . uni-lab:~/streamind/`
6. Run 30s smoke test with judge on uni-lab:
   `ssh uni-lab "cd ~/streamind && ./tools/run_pipeline.sh -w 30 -s ollama-qwen3.5-4b -j ollama-qwen3.5-9b"`
7. Pull results: `rsync -av uni-lab:~/streamind/results/ ./results/`
8. Validate output format: check results/ for `from`/`to`/`summary`/`keywords`(3 items)/`proc_time` — no extra keys
9. Confirm audio goes through `audio_rtp` (Janus path), NOT `audio_file`
10. Run full 300s test to confirm at least one window fires and posts successfully
