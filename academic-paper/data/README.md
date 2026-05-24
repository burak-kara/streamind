# Paper data

Captured benchmark outputs used by `figures/*.tex` and tables in `main.tex`.

Each CSV is regenerated from `results/*/window_*.json` via
`tools/eval_quality.py` after a benchmark pass on the target hardware.

| File | Source | Used by |
|------|--------|---------|
| `model-sweep.csv` (TODO) | Per-profile runs (Qwen3.5 2B/4B/9B, CUDA) on rev16, 300 s windows | `figures/model-sweep.tex` |
| `per-window.csv` (TODO) | CUDA Qwen3.5-4B on rev16, 300 s windows | Table in §5.3 |
| `component-ablation.csv` (TODO) | Four single-component removals on CUDA path | Table in §5.5 |

CSV schema (uniform):

```
profile,window,from_s,to_s,proc_time_s,B,K,L,C
```

`B` is the LLM-judge Likert sum (max 25). `K` follows the challenge keyword
formula (max 6). `L = 10 * exp(-0.5 * proc_time)` if `B >= 10`, else 0.
`C = B + K + L`. Audio-level score = mean(C) + 4 (Janus bonus).

**Status**: All CSVs are placeholders pending benchmark runs on uni-lab.
