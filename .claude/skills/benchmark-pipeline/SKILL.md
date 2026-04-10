---
name: benchmark-pipeline
description: Run tests, parse results/window_*.json, and report latency stats + estimated score impact
disable-model-invocation: true
---

Run these steps in order, stopping on failure with a clear message:

## 1. Run unit tests

```bash
uv run pytest tests/ -x -q 2>&1 | tail -20
```

If tests fail, stop and show the failure.

## 2. Check for result files

```bash
ls results/window_*.json 2>/dev/null || echo "NO_RESULTS"
```

If `NO_RESULTS`, print: "No results found. Run `/run-pipeline` first to generate output."

## 3. Parse and report latency stats

Run this Python snippet:

```bash
uv run python - <<'EOF'
import json, glob, sys

files = sorted(glob.glob("results/window_*.json"))
if not files:
    print("No result files found.")
    sys.exit(1)

latencies = []
for f in files:
    d = json.load(open(f))
    lat = d.get("latency", None)
    wid = d.get("window_id", "?")
    summary_len = len(d.get("summary", ""))
    kw = d.get("keywords", [])
    kw_ok = "OK" if len(kw) == 3 else f"WARN: {len(kw)} keywords (need 3)"
    print(f"  window {wid}: latency={lat:.2f}s  summary={summary_len} chars  keywords={kw_ok}")
    if lat is not None:
        latencies.append(lat)

if latencies:
    avg = sum(latencies) / len(latencies)
    print(f"\nLatency summary: avg={avg:.2f}s  min={min(latencies):.2f}s  max={max(latencies):.2f}s")
    # Score estimate: quality maxes at 30, latency subtracts directly
    # Show range based on optimistic (30) and conservative (20) quality assumptions
    print(f"\nEstimated score range (quality 20–30, avg latency {avg:.2f}s):")
    print(f"  Optimistic:    {30 - avg:.2f}  (quality=30)")
    print(f"  Conservative:  {20 - avg:.2f}  (quality=20)")
    if avg > 30:
        print(f"\n  WARNING: avg latency {avg:.2f}s exceeds max quality score (30). Score will be negative.")
EOF
```

## 4. Print a one-line recommendation

If avg latency > 20s: "Consider a faster LLM (e.g. qwen3:4b) or smaller ASR model. Use `/swap-model` to switch."
If avg latency 10–20s: "Latency is marginal. Profile the LLM inference time vs ASR time to find the bottleneck."
If avg latency < 10s: "Latency looks good. Focus on improving summary quality with `/tune-prompt`."
