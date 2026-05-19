---
name: remote-finetune
description: Sync repo to uni-lab and run QLoRA finetune pipeline (prepare → train → merge → export → eval)
disable-model-invocation: true
---

Sync and run finetune on uni-lab (RTX 4090, CUDA 12.4).

1. Sync uncommitted changes: `rsync -av --exclude='.venv' --exclude='results' --exclude='__pycache__' . uni-lab:~/streamind/`
2. Install finetune deps: `ssh uni-lab "cd ~/streamind && uv sync --extra finetune"`
3. Prepare data: `ssh uni-lab "cd ~/streamind && uv run python tools/finetune/prepare_rev16.py"`
4. Train: `ssh uni-lab "cd ~/streamind && uv run python tools/finetune/finetune_summarizer.py"`
5. Merge adapter: `ssh uni-lab "cd ~/streamind && uv run python tools/finetune/merge_lora.py"`
6. Export to Ollama: `ssh uni-lab "cd ~/streamind && bash tools/finetune/export_to_ollama.sh"`
7. Evaluate: `ssh uni-lab "cd ~/streamind && uv run python tools/finetune/eval_finetuned.py"`
8. Pull results back: `rsync -av uni-lab:~/streamind/results/ ./results/`

Steps are NOT idempotent — run in order. merge_lora.py must complete before export_to_ollama.sh.
If repo is already committed and pushed, replace step 1 with: `ssh uni-lab "cd ~/streamind && git pull"`
