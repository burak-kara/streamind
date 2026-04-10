---
name: tune-prompt
description: Test the current summarization prompt against a sample transcript via Ollama and show the structured output
disable-model-invocation: true
---

Run these steps in order:

## 1. Verify Ollama is reachable

```bash
curl -s http://127.0.0.1:11434/api/tags | python3 -c "import sys,json; d=json.load(sys.stdin); print('Ollama OK, models:', [m['name'] for m in d.get('models',[])])" 2>/dev/null || echo "ERROR: Ollama not running. Start with: ollama serve"
```

Stop if Ollama is unreachable.

## 2. Pick a sample transcript

Check if there are any result files or fixture text to use:

```bash
uv run python - <<'EOF'
import json, glob

# Try to use a results file first (has real pipeline output)
files = sorted(glob.glob("results/window_*.json"))
if files:
    d = json.load(open(files[0]))
    # Results don't store the raw transcript — use a fixture or hardcoded sample
    print("USING_SAMPLE")
else:
    print("USING_SAMPLE")
EOF
```

Use this hardcoded sample transcript for testing (representative of meeting content):

```
SAMPLE_TRANSCRIPT="The team discussed migrating the authentication service to OAuth 2.0.
John raised concerns about the token expiry defaults and suggested setting them to 24 hours for mobile clients.
Sarah confirmed that the API gateway already supports PKCE flow.
The group agreed to start with a staged rollout affecting 10 percent of users next Tuesday.
Action item: John to update the migration runbook by end of week."
```

## 3. Read the current prompt template

```bash
cat plugins/nodes/proc/_summarizer_llm/summarize_prompt.txt
```

## 4. Run the prompt against Ollama

```bash
uv run python - <<'PYEOF'
import json, urllib.request, os

prompt_tmpl = open("plugins/nodes/proc/_summarizer_llm/summarize_prompt.txt").read()
sample = """The team discussed migrating the authentication service to OAuth 2.0.
John raised concerns about the token expiry defaults and suggested setting them to 24 hours for mobile clients.
Sarah confirmed that the API gateway already supports PKCE flow.
The group agreed to start with a staged rollout affecting 10 percent of users next Tuesday.
Action item: John to update the migration runbook by end of week."""

prompt = prompt_tmpl.replace("{transcript}", sample)

# Read model from config
cfg = json.load(open("pipelines/config.json"))
model = next(
    (n["configuration"]["model_name"] for n in cfg["pipeline"]["nodes"] if n["mark"] == "summarizer_llm"),
    "qwen3.5:9b-16k"
)

print(f"Model: {model}")
print(f"Prompt length: {len(prompt)} chars\n")
print("--- LLM OUTPUT ---")

payload = json.dumps({
    "model": model,
    "prompt": prompt,
    "stream": False,
    "format": {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "keywords": {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 3}
        },
        "required": ["summary", "keywords"]
    },
    "options": {"stop": ["</think>"], "temperature": 0}
}).encode()

req = urllib.request.Request(
    "http://127.0.0.1:11434/api/generate",
    data=payload,
    headers={"Content-Type": "application/json"}
)
resp = urllib.request.urlopen(req, timeout=120)
result = json.loads(resp.read())
raw = result.get("response", "")

try:
    parsed = json.loads(raw)
    print(f"summary: {parsed.get('summary', 'MISSING')}")
    print(f"keywords: {parsed.get('keywords', 'MISSING')}")
    kw = parsed.get('keywords', [])
    if len(kw) != 3:
        print(f"WARNING: {len(kw)} keywords returned (need exactly 3)")
    else:
        print("keywords count: OK (3)")
except json.JSONDecodeError:
    print(f"RAW (not valid JSON): {raw}")

eval_ms = result.get("eval_duration", 0) / 1e6
load_ms = result.get("load_duration", 0) / 1e6
print(f"\nTiming: load={load_ms:.0f}ms  inference={eval_ms:.0f}ms  total={(load_ms+eval_ms):.0f}ms")
PYEOF
```

## 5. Prompt improvement tips

After showing output, print these guidelines if the output looks poor:

- **Vague summary**: Add explicit instruction: "Include specific decisions, action items, and names mentioned."
- **Wrong keyword count**: Reinforce in prompt: "You MUST return EXACTLY 3 keywords. No more, no fewer."
- **Generic keywords**: Add: "Keywords must be specific noun phrases from the transcript, not generic terms like 'discussion'."
- **Slow inference**: Shorten the prompt — every token costs latency.

To edit the prompt: `Edit plugins/nodes/proc/_summarizer_llm/summarize_prompt.txt`
Then re-run `/tune-prompt` to compare.
