# STREAMIND Academic Paper — Full Content Plan

## Context

IEEE MMSP 2026 Grand Challenge paper, 6 pages, deadline June 19. Template/skeleton exists in `academic-paper/main.tex` with `\todo{}` placeholders. Figures F1-F3 are built. Need to fill all sections with real content, run remaining experiments, and produce camera-ready PDF.

Key constraint: paper describes only the vLLM pipeline (no MLX/Ollama). ASR model choice undecided (A/B data exists for 30s windows, need 300s). Model sweep (2B/4B/9B) planned but not yet run.

---

## Additional Advice (Beyond User's 5 Areas)

1. **WER ≠ Score paradox** — Both ASR models produce WER=0.69 but different C scores. Strong finding: transcription accuracy is not the bottleneck; keyword stability is. Present this explicitly.

2. **Outlier window analysis** — Window 7 is consistently worst in both configs. Root cause: topic transition with informal speech. Shows per-window variance is content-driven, not model-driven. Adds analytical depth reviewers value.

3. **Prompt-to-Likert alignment** — Each prompt clause maps to a specific judge criterion. Present this as intentional design methodology, not accident.

4. **Extractive fallback as safety net** — When LLM fails, verbatim transcript excerpt preserves B_i factual_consistency at zero latency cost. Practical contribution worth 1-2 sentences.

5. **Grand Challenge paper strategy** — Reviewers evaluate a working system against a shared task. Score decomposition (B/K/L per window) matters more than headline numbers. Honest limitations > glossed weaknesses.

6. **Keyword guarantee algorithm** — Strongest contribution. Converts a 12-point K_i swing (-6 to +6) into a 6-point swing (0 to +6). Present as Algorithm 1 with pseudocode.

7. **Future work from research** — SGLang (+29% throughput), MoE models (Qwen3-30B-A3B), encoder-decoder BART path. All grounded in `docs/research/model-engine-alternatives.md`.

---

## Section-by-Section Content Plan

### §1 Introduction (~0.7 pp)

**Can write now.** No data dependencies.

- P1: Motivation — real-time meeting intelligence (remote work, IETF, podcasts). Gap between batch and streaming summarization.
- P2: Scoring formula as design constraint — present C_i = B_i + K_i + L_i with B_i ≥ 10 gate. Frame: "quality first, latency second, keyword discipline always." Mention Janus +4.
- P3: Three contributions:
  1. Streaming chunk deduplication (temporal coherence across overlapping 5s ASR chunks)
  2. Keyword guarantee algorithm (worst-case -6 → floor +0)
  3. Quality-gated latency optimization (avg proc_time 0.6s, L_i ~7.4, B_i at ceiling)
- P4: Paper roadmap (2 sentences)

### §2 Background and Related Work (~0.5 pp)

**Can write now.**

- P1: Juturna (node-graph framework) + Janus (WebRTC gateway, +4 bonus, same code path dev/eval)
- P2: Streaming ASR — Whisper family, faster-whisper/CTranslate2 int8. `condition_on_previous_text=False` for streaming
- P3: LLM summarization under latency — small instruction-tuned LLMs, vLLM paged attention. Position as *systems* contribution wrapping existing primitives

**Add to references.bib:** vLLM (Kwon et al. SOSP 2023), CTranslate2

### §3 System Architecture (~2.0 pp) — THE MEAT

**Can write now** (except final ASR model pick in §3.2).

Opening: pipeline overview → Figure 1 (exists). Seven stages. Scoring-aware design principle.

| Subsection | Length | Scoring Link | Key Detail |
|------------|--------|-------------|------------|
| 3.1 Audio Reception | 0.1 pp | +4 bonus | Janus, Opus/48k/mono, UDP/8888 |
| 3.2 Incremental ASR | 0.35 pp | B_i | 5s chunks/1s overlap, A/B comparison table, WER≠score paradox |
| 3.3 Hallucination Filter | 0.15 pp | B_i factual | 11 exact-match patterns, regex tag strip |
| 3.4 Novel Extraction | 0.2 pp | B_i coherence | Suffix-prefix match + SequenceMatcher 0.8 fallback |
| 3.5 Window Aggregation | 0.1 pp | — | 300s rolling, graceful stop |
| 3.6 Summarization | 0.4 pp | B_i + L_i | vLLM in-process, prompt design, thinking suppression, extractive fallback, CUDA warmup |
| 3.7 Keyword Validation | 0.2 pp | K_i | Algorithm 1 pseudocode, banned terms, backfill |
| 3.8 Result Transmission | 0.1 pp | — | Challenge contract mapping |

**Key content in §3.6 (Summarization):**
- Quote exact prompt from `summarize_prompt.txt`
- Prompt-to-Likert mapping: "1-2 sentences" → conciseness, "only facts explicitly present" → factual_consistency, "specific noun phrases" → K_i
- `enable_thinking=False` → fewer output tokens → lower proc_time → higher L_i
- Sampling: T=0.3, top_p=0.9, rep_penalty=1.05, max_tokens=256
- Extractive fallback: <80 char transcripts → verbatim excerpt (B_i safe, L_i preserved)

**Key content in §3.7 (Keywords):**
- Present as Algorithm 1 (pseudocode box)
- 4-step: cleanup → ban filter → dedup → backfill from transcript frequency ranking
- Safety math: converts K_i range from [-6, +6] to [0, +6]

### §4 Implementation and Deployment (~0.5 pp)

**Can write now.**

- Config assembly: `config-base.json` + summarizer profile → `assemble_config.py`
- Docker: CUDA 12.4 runtime, Python 3.12 + uv, pre-baked weights. Zero network at runtime
- `docker-compose.yml`: Janus + pipeline services
- Reproducibility: dataset ID, model revision pins, RTP params, hardware spec

**Table 1: Pipeline Node Summary** (8 rows × 5 columns: node, type, model/algo, params, scoring target)

### §5 Evaluation (~1.5 pp)

#### 5.1 Setup (0.2 pp) — Can write now
- Dataset: rev16 podcast, ~36 min
- Judge: Mistral-Small-24B-AWQ (cross-family from Qwen). Disclosure: local approximation
- Hardware: RTX 4090, CUDA 12.4
- WER: jiwer + sliding-window proportional alignment (independent signal)

#### 5.2 ASR Model Comparison (0.3 pp) — DATA READY (30s)

**Table 2** from existing results:

| Metric | small.en | large-v3-turbo |
|--------|----------|---------------|
| avg B | 23.6 | 23.8 |
| avg K | 4.9 | 4.5 |
| avg L | 7.42 | 7.37 |
| avg C | 36.00 | 35.73 |
| final | 40.00 | 39.73 |
| proc_time | 0.60s | 0.62s |
| WER | 0.69 | 0.69 |

Key claims:
- Identical WER, different C scores → WER isn't the bottleneck
- small.en wins on keyword stability (K diff = +0.4)
- Paradox: smaller ASR model → higher pipeline score

#### 5.3 End-to-End Scoring (0.3 pp) — NEEDS 300s RUN
- Table 3: per-window B/K/L/C at 300s windows
- Fallback: use 30s data with disclosure if 300s unavailable

#### 5.4 Outlier Analysis (0.15 pp) — DATA READY
- Window 7: consistently worst (both configs). B=19-22, K=-2 to +2
- Root cause: topic transition, informal speech
- Shows variance is content-driven

#### 5.5 Ablations (0.3 pp) — NEEDS ABLATION RUNS
- Table 4: full pipeline vs -keyword_validation vs -hallucination_filter vs -novel_extractor
- Columns: ΔB, ΔK, ΔL, ΔC

#### 5.6 Model-Size Sweep (0.2 pp) — NEEDS SWEEP RUN
- Figure 5: Qwen3.5-2B/4B/9B on CUDA
- Key question: does 2B ever drop B_i < 10? (loses L_i entirely)

#### 5.7 Threats to Validity (0.1 pp) — Can write now

### §6 Discussion (~0.15 pp)

**Can write now.**
- Scoring incentives vs real-world needs
- Static prompt limitation → future: content-adaptive prompting
- Future work: SGLang (+29%), MoE (Qwen3-30B-A3B), BART path, structured-output decoding

### §7 Conclusion (~0.15 pp)

**Write after results finalized.** Recap 3 contributions + headline score.

---

## Experiments To Run (Priority Order)

### P1 — Paper-blocking

| # | Experiment | Purpose | Produces |
|---|-----------|---------|----------|
| 1 | 300s window run (both ASR models, ≥2 audio files) | Submission-granularity scores | Table 3, ASR decision |
| 2 | Summarizer sweep: Qwen3.5-2B/4B/9B on CUDA | Model-size tradeoff | Figure 5 data |

### P2 — Strongly recommended

| # | Experiment | Purpose | Produces |
|---|-----------|---------|----------|
| 3 | Component ablation (4 configs) | Contribution validation | Table 4 |
| 4 | Second audio source (different rev16 ep or IETF) | Generalization | Paragraph in §5.7 |
| 5 | Per-stage latency breakdown (ASR vs summarizer) | Where proc_time is spent | Optional figure or inline numbers |

### P3 — Nice to have

| # | Experiment | Purpose | Produces |
|---|-----------|---------|----------|
| 6 | Phase-1 ablation on CUDA | Update F4 with CUDA data | Figure 4 update |

---

## Figures & Tables Status

| ID | Content | Status | Blocker |
|----|---------|--------|---------|
| F1 | Pipeline architecture (TikZ) | ✅ DONE | — |
| F2 | L_i scoring curve | ✅ DONE | — |
| F3 | Chunk-overlap timing | ✅ DONE | — |
| F4 | Phase-1 ablation bars | ⚠️ EXISTS, has MLX data | Exp #6 for CUDA update |
| F5 | Model-size sweep | ❌ PLACEHOLDER | Exp #2 |
| T1 | Pipeline node summary | ✍️ Write now | — |
| T2 | ASR model comparison | ✅ DATA READY | — |
| T3 | Per-window 300s scores | ❌ NEEDS DATA | Exp #1 |
| T4 | Component ablation | ❌ NEEDS DATA | Exp #3 |
| A1 | Keyword guarantee algorithm | ✍️ Write now | — |

---

## Writing Schedule

**Week 1 (May 25-31):** Write §1-§4, Algorithm 1, Table 1, Table 2, §5.1, §5.2, §5.4, §5.7, §6. Run Exp #1-#2 on uni-lab.

**Week 2 (Jun 1-7):** Write §5.3, §5.6 with data from Exp #1-#2. Run Exp #3-#4. Fill Figure 5 data. Make ASR model decision.

**Week 3 (Jun 8-14):** Write §5.5 (ablations), §7 (conclusion), finalize abstract. Update Figure 4 if Exp #6 done. Page budget check.

**Jun 15-19:** Proofread, compile, verify cross-refs, clear all `\todo{}`, check page count. Submit.

---

## Structural Changes to main.tex

1. **Remove all Ollama/MLX references** from `\todo{}` markers in §3.6 and Table T2
2. **Add ASR comparison** to §3.2 (inline table or separate Table 2)
3. **Add Algorithm environment** for keyword guarantee (packages already loaded)
4. **Rename "Phase-1 ablation"** → "Prompt and Output Engineering" (reviewers don't know internal milestones)
5. **Add vLLM citation** to references.bib
6. **Update summarizer profiles table** — only vLLM Qwen3.5-{2B,4B,9B}
7. **Consider dropping F4** if over page budget (MLX-era data less relevant to CUDA submission)

---

## Page Budget

| Section | Pages |
|---------|-------|
| Title/authors/abstract/keywords | 0.3 |
| §1 Introduction | 0.7 |
| §2 Background | 0.5 |
| §3 Architecture (+ F1, F2, F3, A1) | 2.0 |
| §4 Implementation (+ T1) | 0.5 |
| §5 Evaluation (+ T2, T3, T4, F5) | 1.5 |
| §6 Discussion | 0.15 |
| §7 Conclusion | 0.15 |
| References | 0.3 |
| **Total** | **~6.1** |

Compression if over: tighten §3.1/3.5/3.8, shorten §6, merge T1+T2, or drop F4.

---

## Verification

- `./compile.sh --draft` compiles with no errors
- `./compile.sh` full build: bibliography resolves, cross-refs work
- `grep -c '\\todo' main.tex` returns 0 before submission
- Every claim in §3-§5 backed by citation, measured number, or code reference
- Page count ≤ 6
