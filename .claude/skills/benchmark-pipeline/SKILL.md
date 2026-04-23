---
name: benchmark-pipeline
description: Parse results/*/window_*.json, compute L_i per chunk, flag format violations, estimate C_i range
disable-model-invocation: true
---

Run these steps in order.

## 1. Pick the results directory

```bash
ls results/ 2>/dev/null || { echo "ERROR: no results/ dir. Run the pipeline first."; exit 1; }
```

If the user passed a model/window subpath, use it. Otherwise scan all model folders under `results/`.

## 2. Parse every window file and compute L_i

```bash
uv run python - <<'EOF'
import glob, json, math, sys
from statistics import mean, median

REQUIRED = {"from", "to", "summary", "keywords", "proc_time"}
BANNED = {"general", "discussion", "meeting", "topic", "content",
          "summary", "overview", "information", "points", "items"}

paths = [
    p for p in sorted(glob.glob("results/**/window_*.json", recursive=True))
    if "/debug/" not in p and p.rsplit("/", 1)[-1].removeprefix("window_").removesuffix(".json").isdigit()
]
if not paths:
    print("no window_*.json under results/")
    sys.exit(1)

rows = []
violations = []

for p in paths:
    try:
        d = json.load(open(p))
    except Exception as e:
        violations.append(f"{p}: invalid JSON ({e})")
        continue

    missing = REQUIRED - d.keys()
    extra = d.keys() - REQUIRED
    if missing:
        violations.append(f"{p}: missing keys {sorted(missing)}")
    if extra:
        violations.append(f"{p}: unexpected keys {sorted(extra)}")

    kws = d.get("keywords") or []
    if len(kws) != 3:
        violations.append(f"{p}: keywords count {len(kws)} (want 3)")
    banned_hits = [k for k in kws if str(k).lower() in BANNED]
    if banned_hits:
        violations.append(f"{p}: banned keywords {banned_hits}")
    if any(not str(k).strip() for k in kws):
        violations.append(f"{p}: empty keyword slot")

    proc = float(d.get("proc_time", 0.0))
    L = 10.0 * math.exp(-0.5 * proc)
    rows.append((p, proc, L, kws, d.get("summary", "")))

rows.sort(key=lambda r: r[1])
print(f"{'window':60} {'proc_time':>9} {'L_i':>6}")
print("-" * 78)
for p, proc, L, _, _ in rows:
    print(f"{p[-60:]:60} {proc:>9.2f} {L:>6.2f}")

if rows:
    procs = [r[1] for r in rows]
    Ls = [r[2] for r in rows]
    print("-" * 78)
    print(f"n={len(rows)}  proc mean={mean(procs):.2f}s median={median(procs):.2f}s  "
          f"L mean={mean(Ls):.2f} (gated: requires B_i>=10)")

    # C_i estimates assuming max K_i=+6. B sweeps over plausible judge scores.
    print()
    print("Estimated chunk C_i = B + K + L  (K assumed +6, avg L shown)")
    print(f"{'B assumption':>14}   C_i estimate")
    for B in (10, 15, 18, 20, 22):
        L_eff = mean(Ls) if B >= 10 else 0.0
        C = B + 6 + L_eff
        print(f"{B:>14}   {C:.2f}")

if violations:
    print()
    print("FORMAT VIOLATIONS:")
    for v in violations:
        print(f"  - {v}")
else:
    print()
    print("Format OK: no violations found.")
EOF
```

## 3. Summarize for the user

Report:

- window count scanned
- mean + median `proc_time`
- mean `L_i`
- list of any format violations (empty keywords, wrong count, banned terms, missing required keys)
- rough `C_i` range across plausible `B_i` assumptions

If violations exist, suggest the most likely cause (prompt drift, LLM error path, wrong profile selected) and the file to inspect.
