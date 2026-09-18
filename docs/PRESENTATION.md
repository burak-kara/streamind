# Presentation Outline — STREAMIND (IEEE MMSP 2026 Grand Challenge)

Talk length: ~16 min, 17 slides. Source of truth for every number/claim: the
accepted paper (`academic-paper/main.tex`) and the official challenge spec
(`docs/CHALLENGE.md`). This is a content outline only — build the actual deck
(PowerPoint / Google Slides / Beamer) from it.

**Reusable visuals** — three figures already exist as TikZ source in the paper and
should be compiled standalone rather than redrawn:
- `academic-paper/figures/pipeline.tex` — 8-node architecture diagram (slide 7)
- `academic-paper/figures/scoring-curve.tex` — latency-bonus curve (slide 3)
- `academic-paper/figures/model-sweep.tex` — stacked bar chart, 6 model configs (data now shown as a table on slide 15 instead; figure still available if preferred)

Compile each by wrapping in a minimal `\documentclass{standalone}` doc (or via
`academic-paper/compile.sh` machinery), then export to PNG at presentation
resolution.

---

## 1. Cover

**Content**
- Title: "A Real-Time Juturna Pipeline for Live Audio Summarization and Keyword Extraction"
- STREAMIND — Streaming Transcription and Real-Time Extraction for AI Meeting INtelligence Distillation
- Mervegul Parlak, Burak Kara, Alperen F. Zengin, Ali C. Begen — Ozyegin University / Rennes
- IEEE MMSP 2026 Grand Challenge

**Speaker notes**: Introduce the team and frame this as a Grand Challenge submission, not a general research talk — the whole design is shaped by one external scoring function, which the next few slides will make concrete.

**Visual**: none (text-only, no logos).

---

## 2. The Challenge — Problem & Goal

**Content**
- Goal: live audio in → structured summary + 3 keywords out, in real time
- Reference pipeline (6 steps): WebRTC/RTP reception → incremental ASR → novel-chunk extraction → 300s window aggregation → summarization → transmission
- Evaluation datasets: Rev16 (30 podcast episodes) and IETF125 (8 live conference sessions)

**Speaker notes**: The challenge bridges live audio reception and structured LLM inference — bring your own pipeline that ingests a live RTP stream and progressively emits summaries, not a batch transcription tool. 300-second windows on a 30-minute recording yield 6 progressive summary chunks, which is the unit everything downstream gets scored on.

**Visual**: reuse `docs/images/challenge_overview.png` (challenge-provided diagram).

---

## 3. The Challenge — Scoring Contract

**Content**
- $C_i = B_i + K_i + L_i$ per chunk
- $B_i \in [5,25]$ — 5-dimension LLM-judged quality (factual consistency, relevance, coherence, fluency, conciseness)
- $K_i = 2(N_{rel} - N_{irrel}) \in [-6,6]$ — keyword relevance
- $L_i = 10e^{-0.5 \cdot proc_i}$, **only if** $B_i \ge 10$
- Janus WebRTC bonus: flat **+4**
- Final score = average chunk score per source, then min-max normalized across submissions

**Speaker notes**: The latency term is gated on a quality floor — you cannot game speed by emitting garbage instantly. This single design choice shapes every architectural decision in the rest of the talk: quality first, then squeeze latency.

**Visual**: reuse `academic-paper/figures/scoring-curve.tex`.

---

## 4. Why This Is Hard

**Content**
- Streaming, not batch: overlapping ASR chunks need deduplication before they reach the summarizer
- Fixed 300s windows required to produce progressive (not just final) output
- Single-GPU ceiling: RTX Pro 4500, 32GB, at eval time — ASR + LLM must co-reside
- Juturna itself is a young framework (1.0-beta) — this challenge also stress-tests the tool

**Speaker notes**: Two separate hard constraints stack here — a systems constraint (real-time, overlapping, GPU-bounded) and a tooling constraint (building on a beta framework). Worth naming both since it explains some of the defensive engineering choices later (fallback paths, guarantees).

**Visual**: none — bullet slide.

---

## 5. Background — Juturna Framework

**Content**
- Meetecho's open-source real-time AI pipeline library
- Node-based, message-driven, async/threaded architecture
- Three node roles sharing one abstraction: source / proc / sink
- Each node: a typed message transformer with a defined lifecycle (configure → warmup → start → stop → destroy)

**Speaker notes**: Juturna's node graph maps directly onto the challenge's staged pipeline description — that's essentially why the reference implementation and every submission looks like a chain of nodes. It's a fairly thin abstraction, which is part of why it works for fast prototyping under a deadline.

**Visual**: none, or a simple 3-box diagram (source → proc → sink) if useful.

---

## 6. Background — Janus WebRTC Gateway

**Content**
- Real WebRTC ingestion (ICE/DTLS/SRTP negotiation, live conferencing audio)
- Contrast: challenge baseline uses `ffmpeg`-simulated RTP (a file replayed over UDP, no real negotiation)
- Janus's RTP-forward capability feeds Juturna's `audio_rtp` source node directly
- This is why adopting Janus earns the challenge's +4 bonus

**Speaker notes**: The bonus exists because Janus demonstrates the "real" version of audio reception — actual WebRTC negotiation from a live client, not a toy pipe. Both Janus and Juturna come from Meetecho, so this also showcases their intended combined stack.

**Visual**: none — carry over from architecture slide context.

---

## 7. Solution — Architecture Overview

**Content**
- 8-node Juturna pipeline, one node per stage
- `audio_rtp` → `audio_chunker` → `transcriber_whisper` → `hallucination_filter` → `novel_extractor` → `window_aggregator` → `summarizer_vllm` → `result_transmitter`

**Speaker notes**: This is the centerpiece diagram of the talk — walk left to right, naming each node's job in one clause. Flag which three nodes target which scoring term: ASR/hallucination-filter/novel-extractor protect $B_i$, keyword logic inside the summarizer targets $K_i$, and the summarizer's prompt/quantization choices target $L_i$.

**Visual**: reuse `academic-paper/figures/pipeline.tex` — full-width, this is the main visual anchor of the deck.

---

## 8. Solution — Contribution 1: Streaming Chunk Deduplication

**Content**
- Problem: 5s audio chunks with 1s overlap → duplicated transcript text
- Greedy suffix-prefix match, 10 words down to 1, first exact match wins
- `SequenceMatcher` fallback (similarity ≥ 0.8) to absorb ASR variation across the overlap
- Removes ~20% redundant text per 300s window

**Speaker notes**: Without this, roughly a fifth of every context window handed to the summarizer would just be repeated speech — directly hurting conciseness and coherence, both $B_i$ sub-scores.

**Visual**: none, or a small before/after text-overlap diagram.

---

## 9. Solution — Contribution 2: Keyword Guarantee

**Content**
- Problem: challenge requires exactly 3 keywords; raw LLM output is inconsistent (0, 2, 5, duplicates, generic filler)
- Pipeline: strip banned generic terms ("meeting", "discussion", "topic") → dedupe (case-insensitive) → backfill from transcript token-frequency ranking → fallback terms as last resort
- Guarantees the 3-keyword *format* — does **not** guarantee positive $K_i$

**Speaker notes**: Be explicit about the limit here — this is a format guarantee, not a quality guarantee. Under a weak summarizer (Llama-3.1-8B in our sweep) two windows still scored $K_i = -6$ because the fallback keywords inherited low relevance from an already-poor summary. Worth stating plainly since it's an honest constraint, not a flaw to hide.

**Visual**: none — bullet slide.

---

## 10. Solution — Contribution 3: Quality-Gated Latency Optimization

**Content**
- Prompt engineering + `enable_thinking=False` (thinking-mode suppression) + int4-AWQ quantization
- Result: $B_i$ held above 23/25 while average processing time is **0.76s**
- This is the direct target of the challenge's exponential latency bonus $L_i = 10e^{-0.5 \cdot proc_i}$

**Speaker notes**: This is the punchline of the "solution" section — quality didn't have to be traded for speed here; the three contributions compound so that a sub-second pipeline still scores near-ceiling on quality. Segue into results.

**Visual**: none — or point back to the scoring-curve figure (slide 3) and mark where 0.76s lands on it.

---

## 11. Evaluation Methodology — LLM-as-a-Judge

**Content**
- $B_i$ and $K_i$ are LLM-judged — no ground-truth reference summary exists, so scoring follows the **G-Eval** LLM-as-judge paradigm
- **Cross-family judge panel**: Mistral-Small-24B-Instruct, Phi-4, Gemma-3-27B-IT (all int4-AWQ) — deliberately excludes the Qwen family (the summarizer's own family) to avoid self-evaluation bias
- Each judge scores every summary against the same 5 Likert criteria the challenge defines for $B_i$, plus a per-keyword boolean relevance flag for $K_i$
- Judges run **sequentially**, isolated subprocesses, VRAM reclaimed between loads — single 24GB dev GPU can't host all three at once
- Every configuration run **≥3 times** to absorb judge non-determinism; reported numbers are means ± standard deviation across runs
- Panel consensus = mean ± std **across the 3 judges** too — low dispersion → confidence, high dispersion → flags judge-specific bias, treated with caution

**Speaker notes**: Since $B_i$ and $K_i$ come from an LLM judge, not a fixed reference metric, the methodology itself needs to be defensible — this is the guardrail section before the numbers. Two independent sources of noise are controlled separately: within-judge stochasticity (handled by ≥3 pipeline runs per config) and between-judge bias (handled by scoring with three model families unrelated to the summarizer, then reporting their spread, not just their mean). Be upfront in Q&A: this panel is a proxy for the hidden challenge judge — where all three agree, scores are more trustworthy, but they remain directional, not absolute.

**Visual**: none — or a simple diagram (transcript + summary → 3 parallel judges → mean ± std); not part of the paper's existing figure set, would need to be drawn fresh.

---

## 12. Results — Headline Number

**Content**
- Cross-episode mean final score: **40.14 ± 0.85 / 45** (Rev16, submission config)
- Per-window breakdown (episode 27, 5 windows): avg $B_i$ = 24.4, $K_i$ = 6.0 (ceiling on every window), $L_i$ = 6.65, avg proc = 0.82s
- With Janus +4 bonus: **41.09** for this episode

**Speaker notes**: Lead with this number, then immediately show it's not a one-off by pointing at the per-window table — quality and keyword score stay essentially flat across the whole 25-minute episode.

**Visual**: simple table reproducing the paper's Table III (5 rows + average).

---

## 13. Results — Cross-Domain Validation

**Content**
- IETF125 technical meeting sessions (out-of-domain from the podcast-tuned pipeline): mean score **40.22**
- Closely tracks the Rev16 podcast mean (40.14)

**Speaker notes**: The pipeline and prompt were developed against podcast audio; this is the evidence that it generalizes to a genuinely different domain (structured technical meetings) without retuning.

**Visual**: none, or a two-bar comparison (Rev16 vs IETF125).

---

## 14. Results — Does ASR Accuracy Matter?

**Content**
- `small.en`: WER 0.259 → final score **40.44**
- `large-v3-turbo`: WER 0.252 → final score 40.14
- The *lower-WER* model scores marginally *lower* overall

**Speaker notes**: Counterintuitive finding worth pausing on — at 300-second summarization granularity, transcription accuracy isn't the bottleneck; small ASR errors get smoothed out by the summarizer. Submission still ships `large-v3-turbo` for non-English robustness, which the two-episode English-only comparison doesn't test.

**Visual**: small 2-row comparison table (Table II).

---

## 15. Results — Model-Size & Cross-Family Sweep

**Content**

| Model | $B_i$ (/25) | $K_i$ (/6) | $L_i$ (/10) | proc$_i$ (s) | Janus | Total (/45, mean ± std) |
|---|---|---|---|---|---|---|
| Qwen3.5-4B-AWQ (submission) | 23.3 | 6.0 | 6.84 | 0.76 | +4.0 | **40.14 ± 0.85** |
| Qwen3.5-4B-BF16 | 23.2 | 6.0 | 5.77 | ~1.10† | +4.0 | 38.97 ± 0.91 |
| Qwen3.5-9B-AWQ | 23.0 | 6.0 | 4.15 | ~1.76† | +4.0 | 37.15 ± 1.53 |
| Qwen3.5-9B-BF16 | 23.2 | 6.0 | 2.92 | 2.51 | +4.0 | 36.12 ± 1.42 |
| Gemma-3-12B-IT-AWQ | 22.1 | 6.0 | 3.55 | 2.07 | +4.0 | 35.63 ± 1.06 |
| Llama-3.1-8B-Instruct | 18.4 | 6.0 | 3.92 | ~1.87† | +4.0 | 32.31 ± 4.59 |

† proc$_i$ not stated directly in the paper's prose for this row — derived by inverting the paper's own $L_i = 10e^{-0.5 \cdot \text{proc}_i}$ against its reported $L_i$. Every other cell (including the un-marked proc$_i$ values 0.76, 2.51, 2.07) is quoted directly from `academic-paper/main.tex` §5.6 or the hardcoded coordinates in `academic-paper/figures/model-sweep.tex`.

- AWQ int4 cuts latency ~35% at 4B with essentially flat $B_i+K_i$ (29.3 vs 29.2 across the two 4B variants)
- $K_i$ sits at ceiling (6.0) for every configuration — the keyword guarantee holds regardless of summarizer choice
- Score spread is driven almost entirely by $L_i$ (proc$_i$), not by $B_i$, within the Qwen family

**Speaker notes**: Bigger models don't win under this scoring function — quality plateaus quickly across the Qwen family while latency keeps climbing, so the reward curve favors the smallest model that still clears the quality floor. Llama-3.1-8B's large variance (episode-to-episode $B_i$ swinging 13→24) is also worth flagging as a robustness gap in that model, not in the pipeline. If asked about the two `†` rows, be upfront that those proc$_i$ values are back-calculated from the formula, not separately measured/reported.

**Visual**: reuse `academic-paper/figures/model-sweep.tex`; the table above can replace or accompany it as a data-dense complement.

---

## 16. Honest Limitations

**Content**
- Ablations are qualitative, not isolated toggle-off experiments — can't cleanly separate the contribution of dedup vs. keyword-guarantee vs. prompt engineering individually (flagged by both reviewers)
- Hallucination filter is a fixed 22-pattern list — blind to novel or semantic hallucinations outside it
- Novel-extractor's exact-match approach can misalign under heavy disfluency or noisy overlap (drop/duplicate words)

**Speaker notes**: State these plainly and don't soften them — they're exactly what both reviewers flagged, and the paper already commits to them as future work (§5.7, threats to validity). Owning this directly builds more credibility in Q&A than skipping past it.

**Visual**: none — bullet slide.

---

## 17. Closing

**Content**
- Real-time, quality-gated, sub-second summarization pipeline
- 40.14/45 cross-episode mean, generalizes to a second domain (IETF125: 40.22)
- Open source: `github.com/streaming-university/streamind`
- Thank you — questions

**Speaker notes**: Close on the generalization result, not just the headline score — it's the strongest signal that this isn't overfit to one dataset. Open the floor for questions.

**Visual**: none (text-only, no logos).
