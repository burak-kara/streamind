# Paper data

Captured benchmark outputs used by `figures/*.tex` and tables in `main.tex`.

Each CSV is regenerated from `results/*/window_*.json` via
`tools/eval_quality.py` after a benchmark pass on the target hardware.

| File | Source | Used by |
|------|--------|---------|
| `phase1-ablation.csv` (TODO) | Pre-Phase-1 + Post-Phase-1 runs on `youtube_15min.wav`, 30\,s windows | `figures/phase1-ablation.tex`, Table T3 |
| `model-sweep.csv` (TODO) | Per-profile runs (Qwen3.5 2B/4B/9B × MLX/CUDA) on `youtube_15min.wav`, 300\,s windows | `figures/model-sweep.tex`, Table T2 |
| `per-window.csv` (TODO) | CUDA Qwen3.5-4B on `youtube_15min.wav`, 300\,s windows | Table T4 (per-window scores) |
| `component-ablation.csv` (TODO) | Four single-component removals on the CUDA path | Table T5 ($\Delta$ vs full pipeline) |

CSV schema (uniform):

```
profile,window,from_s,to_s,proc_time_s,B,K,L,C
```

`B` is the LLM-judge Likert sum (max 25). `K` follows the challenge keyword
formula (max 6). `L = 10 * exp(-0.5 * proc_time)` if `B >= 10`, else 0.
`C = B + K + L`. Audio-level score = mean(C) + 4 (Janus bonus).

**Status**: All CSVs are placeholders pending the CUDA benchmark pass.
See the paper plan at `/Users/burak/.claude/plans/inherited-bouncing-reef.md`
§9 for the dependency chain.
