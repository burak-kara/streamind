# STREAMIND Paper Plan — IEEE MMSP 2026 Grand Challenge

## Context

Submission paper for the IEEE MMSP 2026 STREAMIND Grand Challenge. The repo
already has a working 7-node Juturna pipeline that ingests Opus/RTP via Janus,
incrementally transcribes with `faster-whisper small.en`, deduplicates
overlapping ASR output, aggregates a 300 s context window, summarises with
Qwen3.5-4B/9B, and POSTs results to a challenge endpoint.

**Current implementation status** (as of plan writing):

- The MLX (Apple Silicon) summariser path is built, exercised on a 120 s
  YouTube fixture, and produced the only measured numbers we have today
  (`B = 25`, `K = 6`, `L = 2.13`, `C = 33.13`, `proc_time = 3.10 s` on the
  30 s window post-Phase-1 ablation).
- The CUDA / Ollama summariser path *is the intended submission target* but
  has **not been built or measured** yet. RTX 4090 dev box is provisioned;
  RTX Pro 4500 evaluation hardware is the official target. Both still need
  the full implementation + benchmark pass.
- Project remains under active development. The paper plan assumes the CUDA
  pipeline is implemented and benchmarked *before* the paper is drafted.

Paper deadline: **June 19, 2026**. **Page limit: 6 pages IEEE conference
(IEEEtran, two-column, conference option).** Authors as already set in
`main.tex` (5 × Ozyegin University + Burak Kara × Inria).

The challenge scoring formula is `C_i = B_i + K_i + L_i` with `L_i` gated on
`B_i ≥ 10`. Janus delivery earns a flat `+4` audio-level bonus. This gating
structure dictates the paper's narrative: *quality first, latency second,
keyword discipline always.*

The paper needs to: (a) describe the pipeline clearly enough that the
implementation is reproducible from the system description; (b) justify each
design decision against the scoring formula; (c) report measured quality,
keyword, and latency numbers across summariser model sizes and ablate the
optimisation steps that moved C_i from 32.06 → 33.13 + 4.

---

## 1. Narrative arc

**Hook** — Real-time meeting intelligence is bottlenecked by a non-linear
trade-off: latency only counts when summary quality clears a floor. A pipeline
optimised for raw speed but producing low-quality summaries scores worse than a
slower pipeline producing useful ones.

**Problem framing** — Decompose the challenge: ingest live Opus/RTP; produce
exactly 3 keywords and a coherent summary every 300 s; minimise `proc_time`
without sacrificing `B_i ≥ 10`. Map each scoring sub-term to a concrete
engineering decision.

**Solution** — A 7-node Juturna pipeline with three contributions that
directly target the score components:
1. *Streaming chunk deduplication* — preserves temporal coherence across
   overlapping 5 s ASR chunks while keeping the aggregated window
   non-redundant.
2. *Keyword guarantee algorithm* — `ensure_three_keywords` strips banned
   generic terms and back-fills missing slots from transcript proper nouns,
   converting the worst-case `−6` LLM-failure mode into a `+0` floor.
3. *Quality-gated latency optimisation* — prompt engineering (1–2 sentences,
   `num_predict ≤ 150`, `/no_think`) and 4-bit OptiQ quantisation jointly cut
   `proc_time` 24 % while keeping `B_i` at ceiling.

**Operational claims** — Janus delivery in both dev and eval (no env-specific
code paths). Single-image Docker submission boots end-to-end with
`docker compose up`.

**Evaluation** — Ablation showing each contribution's score impact. Model
size sweep (2B / 4B / 9B) across Apple Silicon MLX + CUDA Ollama. Per-window
breakdown showing scoring stability.

---

## 2. Section outline

> Target length: **6 pages** IEEE conference, two-column, IEEEtran
> `\documentclass[conference]{IEEEtran}`. References fit in the page budget.

### 1. Introduction (≈ 0.75 pp)
- Live-audio intelligence motivation; meeting/podcast/IETF-session use cases.
- The STREAMIND Grand Challenge scoring formula and why latency-gating-on-
  quality drives every design choice.
- Contributions enumerated (3, matching narrative above).
- Paper roadmap.

### 2. Related Work and Background (≈ 0.5 pp)
- Juturna framework (node graph, message payloads).
- Janus WebRTC gateway as standardised audio ingestion.
- Streaming ASR (Whisper family, faster-whisper int8).
- LLM summarisation under latency constraints; small-model quantisation
  (4-bit OptiQ Qwen3.5).
- Position the work as a *systems* contribution wrapping these primitives.

### 3. Pipeline Architecture (≈ 2.0 pp — the meat)
Subsections, one per stage, each ≤ 0.3 pp:
- 3.1 Audio reception over Janus/Opus/RTP.
- 3.2 Incremental ASR with overlapping 5 s chunks (1 s overlap).
- 3.3 Hallucination filter (Whisper noise tags + small-LLM defence).
- 3.4 Novel chunk extraction (suffix-prefix word match, `SequenceMatcher`
  fallback).
- 3.5 Window aggregation (300 s rolling buffer).
- 3.6 Summarisation backends (MLX vs Ollama, identical schema).
- 3.7 Keyword validation (banned terms, transcript back-fill).
- 3.8 Result transmission (challenge-contract field mapping).

### 4. Implementation and Deployment (≈ 0.75 pp)
- Configuration assembly (`config-base.json` + summariser profile).
- Single-image Docker container (CUDA 12.3, Ollama daemon + pre-pulled model).
- `docker-compose.yml` topology (Janus + pipeline services).
- Reproducibility checklist (dataset, model versions, RTP payload type,
  encoding clock channel).

### 5. Evaluation (≈ 1.5 pp)
- 5.1 Setup — datasets (`youtube_15min.wav`, optionally `rev16`, `ietf`),
  judge harness (`tools/eval_quality.py`, local Qwen3.5-9B as judge), hardware
  (Apple Silicon M-series, NVIDIA RTX Pro 4500).
- 5.2 End-to-end scoring — per-window B / K / L / C and audio-level mean
  across summariser profiles.
- 5.3 Latency profile — `proc_time` distribution and `L_i` curve; where time
  is spent (ASR vs LLM).
- 5.4 Ablations — Phase-1 vs Pre-Phase-1 (prompt tightening, keyword
  validation, JSON extraction fix); with/without hallucination filter;
  with/without `/no_think`.
- 5.5 Model-size sweep — 2B / 4B / 9B Qwen3.5 trade-off.
- 5.6 Threats to validity — local judge ≠ official judge; MLX numbers
  approximate CUDA; tested on small audio set.

### 6. Discussion (≈ 0.25 pp)
- What the scoring formula encourages vs what real users want.
- Limits of static prompt tuning; future work: per-window adaptive prompts,
  speculative decoding, batched ASR.

### 7. Conclusion (≈ 0.25 pp)
- Recap of contributions + final claimed score.

### References (≈ 0.5 pp)
Juturna, Janus, Whisper, Qwen3, faster-whisper, MLX-LM, Ollama, IEEE MMSP
spec; relevant streaming-summarisation prior work.

---

## 3. Figures

| # | Figure | Purpose | Source |
|---|--------|---------|--------|
| F1 | Pipeline architecture (TikZ block diagram) | One-shot overview of the 7 nodes, message types, RTP/Janus path | New; build with `tikz-inet` (already loaded in `helpers/packages.tex`) |
| F2 | Scoring formula visualisation | `L_i` curve with `B_i` gate (cliff at `B_i = 10`) — communicates *why* quality must come first | pgfplots, single panel |
| F3 | Chunk-overlap timing diagram | Shows how 5 s chunks slide every 4 s and how the novel extractor isolates new words | TikZ |
| F4 | Phase-1 ablation bar chart | `proc_time`, `L_i`, `C_i` pre vs post Phase-1 fixes | pgfplots, grouped bars |
| F5 | Model-size sweep | Bar chart of `B_i / K_i / L_i / C_i` across 2B / 4B / 9B (MLX) and 4B / 9B (Ollama CUDA) | pgfplots |
| F6 (optional) | Latency stage breakdown | Stacked bar: ASR ms / dedup ms / aggregation ms / LLM ms per window | pgfplots; requires per-stage instrumentation |

### 4. Tables

| # | Table | Content |
|---|-------|---------|
| T1 | Pipeline nodes summary | name, type, model, key hyper-parameters, role (one row per Juturna node) |
| T2 | Summariser profiles | profile name, backend, model, params count, quantisation, temp, top_p, repetition_penalty, target hardware |
| T3 | Phase-1 ablation | (Pre/Post) × (`B_i`, `K_i`, `L_i`, `C_i`, `proc_time`) |
| T4 | Per-window scores on `youtube_15min.wav` | window 0..N: `from`, `to`, `proc_time`, B, K, L, C |
| T5 (optional) | Component ablation | rows: `−hallucination_filter`, `−banned_keywords`, `−novel_extractor`, `−/no_think`, baseline; columns: ΔB, ΔK, ΔL, ΔC |

---

## 5. Evaluation strategy

**Datasets to use (priority order):**
1. `youtube_15min.wav` (already in `tests/fixtures/`) — provides ≥ 3 full
   300 s windows and exists today.
2. `rev16` Whisper subset — 30 podcast episodes; pick a 30-min subset for
   audio-level scoring.
3. `ietf` IETF125 sessions — 8 conference recordings; sanity-check generalises
   to a different speech register.

**Hardware:**
- Apple Silicon (MLX 2B/4B/9B) — local dev numbers; available *today*.
- RTX 4090 dev box (Ollama Qwen3.5-4B/9B) — interim CUDA numbers; available
  once the CUDA implementation lands. Use as the headline figures if RTX Pro
  4500 access is delayed.
- RTX Pro 4500 (Ollama Qwen3.5-4B/9B) — *the* target submission hardware.
  Numbers here are required to validate the submission claim. **Dependency**:
  the CUDA pipeline must be implemented and the benchmark suite ported
  before any numbers can be produced.

Paper drafting cannot finalise §5 (Evaluation) until at least one CUDA
benchmark pass has run. Suggested project sequencing:
  1. Implement CUDA path (Ollama summariser node + Dockerfile for CUDA).
  2. Smoke-test on RTX 4090 dev box; capture per-window numbers.
  3. If RTX Pro 4500 access exists, repeat there; otherwise extrapolate /
     disclose.
  4. Run all four ablations on the CUDA path.
  5. Then freeze §5 of the paper.

**Judging:**
- Use `tools/eval_quality.py` with local `qwen3.5:9b-16k` as judge. Disclose
  in §5.1 that this is a self-judge approximation and the official MMSP judge
  may differ. Treat the numbers as *directional*.

**Metrics reported per window:**
- `B_i` (Likert sum across 5 dimensions)
- `K_i` (challenge formula)
- `L_i` (`10·e^(−0.5·proc_time)` with `B_i ≥ 10` gate)
- `C_i = B_i + K_i + L_i`
- `proc_time` (s, wall-clock window-to-output)

**Aggregate metrics:**
- Audio-level `S = avg_i(C_i) + 4` (Janus bonus).
- Per-stage latency mean ± std (optional, if instrumentation lands).

**Ablations to run (all four confirmed in scope):**
- `−banned_keywords` in `ensure_three_keywords` → predicts large `K_i` drop.
- `−hallucination_filter` → predicts modest `B_i` drop.
- `−novel_extractor` (let the window aggregator see raw overlapping ASR) →
  predicts noisier summaries.
- Phase-1 vs Pre-Phase-1 (prompt tightening + `num_predict ≤ 150` +
  banned-keyword filter + JSON extraction fix) — numbers already captured
  in `docs/APPROACH.md`; re-emit as Table T3 / Figure F4.

---

## 6. Files to create or modify (paper repo only)

Plan-phase only — implementation happens after plan approval.

- `academic-paper/main.tex` — flesh out the section skeleton already in place.
- `academic-paper/references.bib` — extend with Whisper, Qwen3, MLX-LM, Ollama,
  Janus, Juturna, faster-whisper, IEEE MMSP CFP.
- `academic-paper/figures/pipeline.tex` *(new)* — F1 TikZ diagram.
- `academic-paper/figures/scoring-curve.tex` *(new)* — F2 pgfplots curve.
- `academic-paper/figures/chunk-overlap.tex` *(new)* — F3 timing diagram.
- `academic-paper/figures/phase1-ablation.tex` *(new)* — F4 grouped bars.
- `academic-paper/figures/model-sweep.tex` *(new)* — F5 grouped bars.
- `academic-paper/data/` *(new)* — CSV inputs for pgfplots tables (generated
  by re-running `tools/eval_quality.py` against captured `results/*/`).

Existing helper files (`helpers/packages.tex`, `commands.tex`, `colors.tex`,
`glossaries.tex`, `tikz-styles.tex`) already provide the macro and styling
base — no rework needed.

---

## 7. Verification

- `./compile.sh --draft` produces `main.pdf` with all sections, figures
  rendering, table macros resolving, and no `??` cross-references.
- `./compile.sh` (full build) → bibliography compiles without warnings;
  cross-references resolved; final PDF copied to `academic-paper/main.pdf`.
- Spot-check: every `\todo{…}` token cleared before camera-ready (search the
  source for `\todo`).
- Cross-check: every claim in §3–§5 has a citation, a measured number, or a
  code-path reference; no unsupported assertions.

---

## 8. Resolved decisions (from clarifying questions)

- **Page limit**: 6 pages, IEEE conference (`IEEEtran`).
- **Evaluation scope**: Report BOTH MLX dev numbers AND CUDA target numbers.
  Requires a benchmark pass on the remote GPU box before camera-ready.
- **Ablations**: All four — Phase-1 fixes, banned-keyword + back-fill,
  hallucination filter, novel chunk extraction.
- **Authors**: Author block in `main.tex` is correct as-is.

## 9. Outstanding risks and dependencies

**Paper-blocking dependencies (must clear before §5 can be drafted):**

- CUDA pipeline implementation. Ollama summariser node, CUDA Dockerfile,
  pre-pulled Qwen3.5-4B/9B in the image, `docker compose up` smoke. None of
  this has been built; all measurements today are MLX. The paper plan
  assumes this lands first.
- CUDA benchmark pass. After the pipeline is built, run
  `tools/eval_quality.py` against captured `results/*/` on the RTX 4090 dev
  box (primary) and RTX Pro 4500 (if accessible).
- Four-way ablation run on CUDA. Phase-1, banned-keyword, hallucination
  filter, novel chunk extraction — each requires a separate config and
  benchmark pass.

**Non-paper-blocking submission risks (tracked elsewhere):**

- `destination_endpoint` URL still empty in `config-base.json`. Submission
  blocker, not a paper blocker. Tracked in
  `docs/plans/submission-readiness-remaining.md`.
- Official MMSP 2026 author kit not in repo. Page limit (6 pp) confirmed
  by the user, but the author kit PDF still needs to be downloaded to
  confirm column width / font / bibliography style match `IEEEtran`
  defaults.
- Local judge (`qwen3.5:9b-16k`) is not the official challenge judge.
  Disclose in §5.1 and treat numbers as directional.
