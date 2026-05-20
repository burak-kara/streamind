---
name: tune-prompt
description: Test the current summarization prompt against a sample transcript via vLLM on uni-lab, show structured output, and measure inference latency
disable-model-invocation: true
---

vLLM only runs on uni-lab (CUDA). All steps below execute remotely.
The active summarizer profile lives at `pipelines/summarizer/vllm-<name>.json`.

## 1. Pick the active profile

```bash
ls pipelines/summarizer/vllm-*.json
```

If more than one, ask the user which to tune.

## 2. Verify weights present on uni-lab

```bash
ssh uni-lab "cd ~/Desktop/streamind && ls models/<name>/config.json"
```

If missing: `ssh uni-lab "cd ~/Desktop/streamind && ./tools/fetch_models.sh <hf_id> <name>"`.

## 3. Read the current prompt

```bash
cat plugins/nodes/proc/_summarizer_vllm/summarize_prompt.txt
```

## 4. Drive the model against a sample transcript on uni-lab

The script reads the same profile JSON the pipeline uses, so prompt + sampling
params match production exactly. Override the prompt file via `--prompt` to
test a candidate without editing the on-disk template.

```bash
ssh uni-lab "cd ~/Desktop/streamind && uv run python - <<'PYEOF'
import json, sys, time
from pathlib import Path
from vllm import LLM, SamplingParams

profile = json.loads(Path('pipelines/summarizer/vllm-<NAME>.json').read_text())
cfg = profile['configuration']

prompt_path = Path('plugins/nodes/proc/_summarizer_vllm') / cfg['prompt_template_file']
prompt_tmpl = prompt_path.read_text()

sample = '''The team discussed migrating the authentication service to OAuth 2.0.
John raised concerns about the token expiry defaults and suggested setting them to 24 hours for mobile clients.
Sarah confirmed that the API gateway already supports PKCE flow.
The group agreed to start with a staged rollout affecting 10 percent of users next Tuesday.
Action item: John to update the migration runbook by end of week.'''

prompt = prompt_tmpl.format(transcript=sample)

llm = LLM(
    model=cfg['model_name'],
    dtype=cfg.get('dtype', 'float16'),
    gpu_memory_utilization=cfg.get('gpu_memory_utilization', 0.85),
    max_model_len=cfg.get('max_model_len', 2048),
    enforce_eager=cfg.get('enforce_eager', False),
)
sampling = SamplingParams(
    temperature=cfg.get('temperature', 0.3),
    top_p=cfg.get('top_p', 0.9),
    max_tokens=cfg.get('max_tokens', 256),
    repetition_penalty=cfg.get('repetition_penalty', 1.05),
)

t0 = time.time()
out = llm.chat(
    [{'role': 'user', 'content': prompt}],
    sampling_params=sampling,
    chat_template_kwargs={'enable_thinking': False},
    use_tqdm=False,
)
elapsed = time.time() - t0
text = out[0].outputs[0].text

print(f'Model: {cfg[\"model_name\"]}')
print(f'Inference: {elapsed*1000:.0f} ms')
print(f'--- raw output ---\\n{text}\\n--- end ---')

import re, json as _json
clean = re.sub(r'<\\|[^|]+\\|>', '', text).strip()
clean = re.sub(r'<think>.*?</think>', '', clean, flags=re.DOTALL)
clean = re.sub(r'^```[a-z]*\\n?', '', clean)
clean = re.sub(r'\\n?```$', '', clean).strip()
m = re.search(r'\\{.*\\}', clean, re.DOTALL)
if not m:
    print('NO JSON FOUND'); sys.exit(0)
parsed = _json.loads(m.group(0))
print(f'summary:  {parsed.get(\"summary\", \"MISSING\")}')
kws = parsed.get('keywords', [])
print(f'keywords: {kws}')
if len(kws) != 3:
    print(f'WARNING: {len(kws)} keywords returned (need exactly 3)')
PYEOF
"
```

Replace `<NAME>` with the actual profile name (without `.json`).

## 5. Prompt improvement tips

After showing output, suggest these if the output is weak:

- **Vague summary**: add "Include specific decisions, action items, and names mentioned."
- **Wrong keyword count**: reinforce "You MUST return EXACTLY 3 keywords."
- **Generic keywords**: add "Keywords must be specific noun phrases from the transcript, not generic terms like 'discussion'."
- **Slow inference**: shorten prompt; every token costs latency (L_i depends on `proc_time`).
- **Hallucinations**: add "Only state facts explicitly present in the transcript. Do not infer."

To edit: `Edit plugins/nodes/proc/_summarizer_vllm/summarize_prompt.txt`. Then rerun.
