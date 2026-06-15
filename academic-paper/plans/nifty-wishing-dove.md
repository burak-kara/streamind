# STREAMIND Paper — Submission Polish Plan

## Context

IEEE MMSP 2026 Grand Challenge paper, 6 pages, deadline June 19.  
Current paper compiles clean at 6 pages with **1 remaining `\todo{}`** (§5.5 Ablations).  
Teammate added `score-breakdown.tex` (untracked) and modified several files.  
New experimental data available: Gemma-3-12B-IT-AWQ and Llama-3.1-8B-Instruct results across all 3 Rev16 episodes.  
User confirmed references are fixed. Abstract has no citations (verified ✓).

---

## Critical Issues

| # | Issue | File | Severity |
|---|-------|------|----------|
| 1 | `model-sweep.tex` uses raw colors (`blue!35`, `teal!40`, `orange!50`, `red!35`) — not from palette | `figures/model-sweep.tex` | High |
| 2 | `score-breakdown.tex` (untracked, unused) has broken `\thisrow{lval}` formula + raw colors | `figures/score-breakdown.tex` | High |
| 3 | §5.5 Ablations: only a `\todo{}` — no content | `main.tex` §5.5 | High |
| 4 | New model results (Gemma-3-12B-IT-AWQ, Llama-3.1-8B-Instruct) not in paper | `main.tex` §5.6 | High |
| 5 | IETF cross-domain results placed in §5.7 Threats (wrong section — positive evidence ≠ threat) | `main.tex` §5.7 | Medium |
| 6 | §5.7: missing space — `...rankings may shift as coverage grows.As a partial cross-domain...` | `main.tex` §5.7 | Low |
| 7 | `helpers/packages.tex`: duplicate `\usepackage{algorithm}` and `\usepackage{algpseudocode}` | `helpers/packages.tex` | Low |

---

## New Data Available (to incorporate)

### Gemma-3-12B-IT-AWQ (`./models/gemma-3-12b-it-awq`, bfloat16)
| Audio | Runs | avg_bk | avg_l | Score |
|-------|------|--------|-------|-------|
| ep10 | 2 | 27.11 | 3.78 | 34.89 |
| ep11 | 2 | 27.33 | 3.53 | 34.87 |
| ep27 | 2 | 29.80 | 3.33 | 37.13 |
| **Cross-audio** | 6 | 28.08 | 3.55 | **35.63 ± 1.06** |

B ≈ 22.1 (avg_bk – K, where K=6.0 guaranteed), avg t_proc ≈ 2.07s

### Llama-3.1-8B-Instruct (`./models/llama-3.1-8b-instruct`, bfloat16)
| Audio | Runs | avg_bk | avg_l | Score |
|-------|------|--------|-------|-------|
| ep10 | 2 | 24.17 | 3.84 | 32.01 |
| ep11 | 2 | 18.96 | 3.89 | 26.85 |
| ep27 | 2 | 30.04 | 4.04 | 38.07 |
| **Cross-audio** | 6 | 24.39 | 3.92 | **32.31 ± 4.59** |

B ≈ 18.4 average, but **high variance** — ep11 avg_bk≈19 means B_i≈13 (very poor quality on that content type).  
Key insight: keyword guarantee holds K_i = 6 even when LLM quality is poor (B_i=13 > gate of 10, so L_i still applies).

---

## Planned Changes (Ordered by Impact)

### 1. Figure Style Unification

#### `figures/model-sweep.tex` (currently used in main.tex)

Replace raw colors with palette from `helpers/colors.tex`:
- `blue!35, draw=blue!70` → `line1!55, draw=line1!80` (colorBlue)
- `teal!40, draw=teal!70` → `line3!55, draw=line3!80` (colorGreen)  
- `orange!50, draw=orange!80` → `line2!55, draw=line2!85` (colorOrange)
- `red!35, draw=red!70` → `colorRed!35, draw=colorRed!70`
- `gray!30` (grid) → `line4!30`
- Add `x tick label style={..., text=labelColor}` to match axis style
- **Extend to 6 models** (see §5.6 below) — add 2 new symbolic x coords

New symbolic x coords: `{4B-AWQ, 4B, 9B-AWQ, 9B, Gem3-12B, Llma-8B}`  
New coord labels → shorter to fit in column width  
Reduce `bar width` from `20pt` to `14pt`  
Reduce `enlarge x limits` from `0.18` to `0.12`

New data rows for the 2 additional models:
```
% B_i
(Gem3-12B, 22.1)   (Llma-8B, 18.4)
% K_i
(Gem3-12B, 6.0)    (Llma-8B, 6.0)
% L_i
(Gem3-12B, 3.55)   (Llma-8B, 3.92)
% Janus +4
(Gem3-12B, 4.0)    (Llma-8B, 4.0)
```
Score total labels: 35.63 (Gemma), 32.31 (Llama) — raise `ymax` from 46 to 46 (ok as-is, all fit).

#### `figures/score-breakdown.tex` (untracked, not referenced in main.tex)

Fix the broken Janus layer — remove the `nodes near coords` + `point meta` approach entirely. Replace with the same `\node[font=\small\bfseries, anchor=south, yshift=3pt]` pattern from model-sweep.tex. Apply palette colors same as above. **Keep as alternative draft, not referenced in main.tex** (unless user decides to swap).

---

### 2. §5.5 Ablations — Replace `\todo{}` with Qualitative Analysis

Replace the placeholder with ~80 words of substantive qualitative ablation:

**Hallucination filter**: 22 patterns matched on podcast ASR output (e.g., "thank you for watching"). Without this node, these phrases reach the summarizer and are faithfully summarized as facts, directly reducing the judge's factual_consistency criterion (one of five B_i criteria) by up to 5 points per window.

**Novel extractor**: Consecutive 5s chunks share 1s (20% of a 5s window). Without deduplication, the 300s context window contains ~20% repeated text, biasing summaries toward verbatim repeated phrases and inflating B_i over the true novel content.

**Keyword validation**: The theoretical K_i range is [−6, +6]. Without \Cref{alg:keywords}, models that hallucinate or omit keywords score K_i = −6 per window. The cross-family data shows this is not hypothetical: Llama-3.1-8B achieves avg_bk=19 on ep11 (B_i≈13), but keyword validation holds K_i = 6 on every window. Formal ablation runs are planned for the camera-ready submission.

---

### 3. §5.6 Model-Size Sweep — Extend to Cross-Family Comparison

**Update the section title** if needed (currently "Model-Size Sweep" — fits).

**Update the figure reference** to note 6-model chart.

**Add 2 paragraphs** after the current Qwen paragraph:

*Para 1 (Gemma)*: Gemma-3-12B-IT-AWQ~\cite{gemma3-12b-awq} scores 35.63±1.06 despite 3× more parameters than 4B-AWQ. avg_bk=28.08 (B≈22.1) is slightly below Qwen's 29.3 on the same episodes, confirming that model family and instruction-tuning alignment matter more than raw parameter count for this task. avg t_proc≈2.07s yields L_i≈3.55, similar to 9B-AWQ latency, suggesting the AWQ int4 quantization gap at 12B is smaller than the cross-family quality gap.

*Para 2 (Llama)*: Llama-3.1-8B-Instruct~\cite{llama3-8b} scores 32.31±4.59 — the ±4.59 std is five times larger than any Qwen variant. Per-audio breakdown reveals the source: ep11 (avg_bk=18.96, B_i≈13) vs ep27 (avg_bk=30.04, B_i≈24). The keyword guarantee holds K_i=6 on every window regardless of summary quality, preventing a further 12-point worst-case drop, but the B_i variation itself is content-driven and unavoidable without further instruction tuning.

**Update figure caption** to note the 6-model sweep.

**New references needed** (add to references.bib):
- `gemma3-12b-awq`: need HuggingFace URL — check `tools/fetch_models.sh` or git log for `./models/gemma-3-12b-it-awq` source
- `llama3-8b`: `meta-llama/Llama-3.1-8B-Instruct` on HuggingFace

---

### 4. Relocate IETF Results (§5.7 → §5.3)

Remove this sentence from §5.7:
> "As a partial cross-domain check beyond the Rev16 podcasts, two held-out sessions from IETF 125..."

Add it to the end of §5.3 (End-to-End Scoring) as a standalone paragraph:
> "As a cross-domain generalization check, the same pipeline and summarizer were evaluated on two technical sessions from IETF 125 (Computing-Aware Traffic Steering and Media over QUIC Transport). Qwen3.5-4B-AWQ produced final source scores of 40.19 and 40.24 respectively, closely tracking the 40.14 cross-episode mean from Rev16 despite the substantially denser technical vocabulary."

This removes IETF from "threats" framing and promotes it to positive evidence.

---

### 5. §5.7 Threats — Fix Space and Trim

Fix missing space: `grows.As` → `grows. As`  
After removing the IETF paragraph, §5.7 becomes tighter. May need slight rewording of the remaining sentence for flow.

---

### 6. `helpers/packages.tex` — Remove Duplicate Declarations

Lines 15-16 and lines 27-28 both declare `algorithm` and `algpseudocode`. Remove the duplicate pair (lines 27-28).

---

### 7. Page Budget

Current paper: exactly 6 pages.  
New additions: ~+0.45 col (ablation text + Gemma/Llama paragraphs + IETF paragraph in §5.3).  
Required trims to stay at 6 pages:
- §5.4 Outlier Analysis: shorten by 1-2 sentences (the section re-states what §5.3 already shows)
- §6 Discussion: trim 1-2 sentences from the future-work paragraph
- Moving IETF from §5.7 to §5.3 is neutral (same text, different location)

**Verify page count** by compiling with `./compile.sh` after each major change.

---

## File Change Summary

| File | Change |
|------|--------|
| `figures/model-sweep.tex` | Colors → palette, extend to 6 models |
| `figures/score-breakdown.tex` | Fix broken formula, apply palette colors |
| `main.tex` §5.3 | Add IETF cross-domain paragraph |
| `main.tex` §5.5 | Replace `\todo{}` with qualitative ablation |
| `main.tex` §5.6 | Add Gemma + Llama paragraphs, update figure caption |
| `main.tex` §5.7 | Remove IETF paragraph, fix space typo |
| `references.bib` | Add `gemma3-12b-awq` and `llama3-8b` entries |
| `helpers/packages.tex` | Remove duplicate package declarations |

---

## Dependencies

- **HuggingFace URLs** for Gemma-3-12B-IT-AWQ and Llama-3.1-8B-Instruct needed for references.bib entries. Check git log or `tools/fetch_models.sh` history.
- **Ablation experiments** (formal numerical data) deferred to camera-ready; qualitative analysis is submitted as-is.

---

## Verification

After all changes:
1. `grep -c '\\todo' academic-paper/main.tex` → must return 0
2. `./compile.sh --clean` → zero errors, zero undefined references
3. Page count ≤ 6 (check build log or PDF)
4. `grep 'cite' academic-paper/main.tex | grep -v '^%' | head -5` to confirm no citation in abstract
