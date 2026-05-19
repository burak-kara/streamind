---
name: remote-finetune
description: Sync repo to uni-lab and run QLoRA finetune pipeline (prepare → train → merge → swap into vLLM)
disable-model-invocation: true
---

**PARKED — M4 only.** Do not run until the base CUDA pipeline (M1) is green
and prompt tuning (M2) + submission packaging (M3) are complete. The current
finetune workflow still references Ollama internally (`prepare_rev16.py`
teacher distillation, `export_to_ollama.sh`); both will be replaced with
vLLM equivalents when M4 starts. Until then, treat this skill as historical.

---

Sync and run finetune on uni-lab (RTX 4090, CUDA 12.4).

1. Sync uncommitted changes: `rsync -av --exclude='.venv' --exclude='results' --exclude='__pycache__' --exclude='models' . uni-lab:~/streamind/`
2. Install finetune deps: `ssh uni-lab "cd ~/streamind && uv sync --extra finetune"`
3. Prepare data: `ssh uni-lab "cd ~/streamind && uv run python tools/finetune/prepare_rev16.py"` — **WILL FAIL until M4 swaps the Ollama teacher for vLLM.**
4. Train: `ssh uni-lab "cd ~/streamind && uv run python tools/finetune/finetune_summarizer.py"`
5. Merge adapter: `ssh uni-lab "cd ~/streamind && uv run python tools/finetune/merge_lora.py"` — outputs a merged HF dir.
6. **M4 plan**: drop GGUF/Ollama export entirely. Copy the merged dir under `./models/<name>-ft/` and point the vLLM profile `model_name` to it. No `export_to_ollama.sh` step needed.
7. Evaluate: A/B base vs finetuned via `tools/eval_quality.py` against the same `results/` set.

Steps are NOT idempotent — run in order. If repo is already pushed, replace step 1 with `ssh uni-lab "cd ~/streamind && git pull"`.
