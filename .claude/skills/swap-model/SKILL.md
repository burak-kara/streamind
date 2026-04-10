---
name: swap-model
description: Switch the ASR or LLM model in pipelines/config.json and verify availability
disable-model-invocation: true
---

Ask the user:
1. Which stage to update: **ASR** (transcriber_whisper) or **LLM** (summarizer_llm)?
2. Which model name to use?

**ASR model options** (faster-whisper): `tiny.en`, `base.en`, `small.en`, `medium.en`
- `tiny.en` = fastest, lowest quality (~32x realtime on CPU)
- `base.en` = good balance (~16x realtime on CPU)
- `small.en` = better quality, ~3–4x slower than tiny on CPU

**LLM model options** (Ollama): `qwen3:4b`, `qwen3:8b`, `llama3.2:3b`, `llama3.2:1b`
- Smaller = faster latency, lower quality
- For this challenge, latency matters: prefer smaller models unless quality is suffering

---

Once the user provides their choices, execute:

## For ASR model swap

```bash
uv run python - <<PYEOF
import json

cfg = json.load(open("pipelines/config.json"))
for node in cfg["pipeline"]["nodes"]:
    if node["mark"] == "transcriber_whisper":
        old = node["configuration"]["model_name"]
        node["configuration"]["model_name"] = "NEW_MODEL_HERE"
        print(f"ASR: {old} -> {node['configuration']['model_name']}")
        break
json.dump(cfg, open("pipelines/config.json", "w"), indent=2)
print("pipelines/config.json updated.")
PYEOF
```

Replace `NEW_MODEL_HERE` with the user's chosen model name.

Note: faster-whisper downloads ASR models automatically on first run. No pull step needed.

## For LLM model swap

```bash
uv run python - <<PYEOF
import json

cfg = json.load(open("pipelines/config.json"))
for node in cfg["pipeline"]["nodes"]:
    if node["mark"] == "summarizer_llm":
        old = node["configuration"]["model_name"]
        node["configuration"]["model_name"] = "NEW_MODEL_HERE"
        print(f"LLM: {old} -> {node['configuration']['model_name']}")
        break
json.dump(cfg, open("pipelines/config.json", "w"), indent=2)
print("pipelines/config.json updated.")
PYEOF
```

Then verify the model is available in Ollama:

```bash
curl -s http://127.0.0.1:11434/api/tags | python3 -c "
import sys, json
d = json.load(sys.stdin)
names = [m['name'] for m in d.get('models', [])]
target = 'NEW_MODEL_HERE'
if any(target in n for n in names):
    print(f'Model {target} is available.')
else:
    print(f'Model {target} not found. Pull it with: ollama pull {target}')
"
```

If the model is not found, run:
```bash
ollama pull NEW_MODEL_HERE
```

## After any swap

Print a summary:
```
Model swapped:
  Stage: [ASR|LLM]
  Old:   [old_model]
  New:   [new_model]

Run /benchmark-pipeline after the next pipeline run to compare latency.
```
