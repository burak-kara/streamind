---
name: run-pipeline
description: Check Ollama is running and the required model is pulled, then launch the pipeline
disable-model-invocation: true
---

Run the following checks before launching, stopping on first failure with a clear message:

1. Verify Ollama is reachable:
   ```
   curl -s http://127.0.0.1:11434/api/tags
   ```
   If it fails, print: "Ollama is not running. Start it with: ollama serve"

2. Confirm `qwen3:8b` is in the model list. If not, print:
   "Model not found. Pull it with: ollama pull qwen3:8b"

3. Launch the pipeline:
   ```
   uv run python -m juturna run pipelines/config.json
   ```
