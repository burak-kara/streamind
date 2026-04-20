# STREAMIND — Scoring Formula Realignment Roadmap

## Context

The challenge scoring formula was completely redesigned. The old formula `C_i = S_i - latency_i` (0–30 opaque judge minus latency penalty) was replaced with an explicit component model. This requires targeted changes across: keyword error handling (immediate score penalty), prompt engineering (new conciseness criterion), latency strategy (bonus replaces penalty, exponential decay), and submission packaging (Dockerfile now required).

The Janus +4 bonus is already earned by our architecture.

## New Scoring Reference

```
C_i = B_i + K_i + L_i + 4 (Janus, free)

B_i  max 25  — LLM judge on 5 Likert criteria:
               factual consistency, relevance, coherence, fluency, CONCISENESS (new)

K_i  max 6   — +2 per relevant keyword, −2 per irrelevant; exactly 3 required
               Current bug: "general" fallback = −2 pts each (up to −6 per chunk)

L_i  max ~6  — 10·e^(−0.5·proc_time), ONLY if B_i ≥ 10
               At 9.35s (current): L_i ≈ 0.09 (negligible)
               At 3s:  L_i ≈ 2.2
               At 1s:  L_i ≈ 6.1

Gate: B_i < 10 → L_i = 0. Quality must come before speed.
```

## ⚠️ Critical Architecture Decision Required

The evaluation environment is an **RTX Pro 4500** (NVIDIA GPU). The current pipeline uses **MLX**, which is Apple Silicon only and **cannot run on NVIDIA hardware**. The submission Dockerfile must target CUDA. This means Phase 2 model optimization should benchmark on an Ollama/CUDA backend for the submission path, not MLX. Local development can continue on MLX.

---

## Phase 1 — Immediate Score-Floor Fixes
_Independent, no dependencies, highest urgency_

### 1.1 Fix keyword error-path bug (CRITICAL)
**Files**: `plugins/nodes/proc/_summarizer_mlx/summarizer_mlx.py`, `plugins/nodes/proc/_summarizer_llm/summarizer_llm.py`

**Problem**: In the `except` block, `keywords = []` is set without calling `_ensure_three_keywords`. Even worse, `_ensure_three_keywords` pads with `"general"` — a semantically irrelevant term that scores −2 pts each.

**Two-part fix**:

**Part A** — Call `_ensure_three_keywords` in the error branch (both files):
```python
# Change this:
summary = ""
keywords = []
# To this:
summary = ""
keywords = self._ensure_three_keywords([])
```

**Part B** — Replace the `"general"` fallback in `_ensure_three_keywords`. Extract fallback terms from the transcript instead of using a static string. Simple approach: take the most frequent capitalized words (proper nouns) from the transcript string, or extract nouns using basic regex. No extra ML needed.

Also add a banned-keyword filter before the length check:
```python
BANNED_KEYWORDS = {"general", "discussion", "meeting", "topic", "content",
                   "summary", "overview", "information", "points", "items"}
kw = [k for k in kw if k.lower() not in BANNED_KEYWORDS]
```

**New tests to add**: `tests/test_summarizer_mlx.py` (currently absent), mirroring the existing `tests/test_summarizer_llm.py` — include an error-path test that mocks `generate()` to raise an exception and asserts `len(keywords) == 3` with no banned terms.

**Acceptance**: No `keywords: []` in any result file; no `"general"` in keyword output during a pipeline run.

### 1.2 Tighten prompts for conciseness
**Files**: `summarize_prompt_mlx_qwen3.txt`, `summarize_prompt_ollama.txt`, `summarize_prompt_mlx.txt`

Change summary instruction from `"2-4 sentences capturing the key points"` to `"1-2 sentences. Be concise and specific. Include decisions, action items, or named entities mentioned."` 

Change keyword instruction from `"single words or short phrases representing main topics"` to `"specific noun phrases from the transcript. Never use generic terms like 'discussion', 'meeting', or 'topic'."` 

Double leverage: better conciseness Likert score AND fewer output tokens → lower proc_time.

**Acceptance**: Summaries under 80 words; no generic keyword terms.

### 1.3 Lower `num_predict` caps
**Files**: All `pipelines/summarizer/*.json`

After tightening prompts to 1-2 sentences, outputs fit in ~128 tokens. Lower to `num_predict: 150` (safety margin). This hard-limits generation time and acts as a conciseness guardrail.

**Acceptance**: Valid JSON emitted within the cap (not truncated).

### 1.4 Fix CLAUDE.md scoring formula
**File**: `CLAUDE.md`

Replace the Challenge Summary section's scoring line and the Scoring Constraints section's formula with the new `C_i = B_i + K_i + L_i` formula and component breakdown. Also fix or remove the reference to `/benchmark-pipeline` skill in the Skills table — this file does not exist yet.

---

## Phase 2 — Latency Optimization
_Gated by Phase 3.1 quality harness — do not swap models without quality validation_

### 2.1 Establish proc_time baseline post-Phase 1
Run 30s window tests across profiles. Phase 1 prompt tightening should already reduce proc_time. Record new baselines before further changes.

### 2.2 Build CUDA/Ollama submission path
**Context**: The submission must run on RTX Pro 4500. Current pipeline uses MLX (Apple Silicon only). A CUDA-compatible inference path is needed for submission.

Options (ranked by ease):
1. **Ollama** (already implemented): `summarizer_llm.py` with `qwen3.5:2b` or `qwen3.5:4b` model — simplest path, Ollama has CUDA support out of the box.
2. **vLLM**: faster inference on CUDA, but adds complexity.

**Recommended**: Use the existing `summarizer_llm` node with a Qwen3.5-2B or Qwen3.5-4B Ollama model for the submission. The submission `run_pipeline.sh` should default to `ollama-qwen3.5-4b` (add this profile to `pipelines/summarizer/`).

**File to create**: `pipelines/summarizer/ollama-qwen3.5-4b.json`

### 2.3 Target proc_time goals
- **3s**: `L_i ≈ 2.2` — achievable with 2B/4B model + prompt tightening
- **1s**: `L_i ≈ 6.1` — stretch goal, requires fast hardware + small model

For local Apple Silicon testing, continue using MLX 2B profile. For submission benchmarking, use Ollama on the RTX environment.

---

## Phase 3 — Quality Tuning
_Build eval harness first; use it to gate all model and prompt changes_

### 3.1 Build local quality evaluation harness
**New file**: `tools/eval_quality.py`

Script that reads `results/*/window_*.json` + `results/*/debug/window_*_transcript.txt`, calls the local LLM-as-judge on the 5 Likert criteria, scores keywords for relevance, and outputs a score table with B_i / K_i / L_i per window.

This harness is the single tool that validates every change in Phases 2 and 3.

### 3.2 Prompt A/B testing
Use `tools/eval_quality.py` to test prompt variants against `tests/fixtures/youtube_15min.wav` (15 min → 3 windows, fast iteration). Key variants:
- **A**: Phase 1 baseline (1-2 sentences, specific noun phrases)
- **B**: Add `"Only state facts explicitly present in the transcript. Do not infer."`
- **C**: Add `"Keywords must be specific terms that identify the meeting's unique subject matter."`
- **D**: Structured output hint: summary starts with main topic, then one supporting fact

Pick the best combination; apply to both prompt files.

### 3.3 Dataset testing
Run the full pipeline against the `rev16` and `ietf` datasets in `docs/datasets/` using the production-ready profile. Use `tools/eval_quality.py` to score all windows. Identify and fix any patterns of keyword failure or summary quality drops.

---

## Phase 4 — Submission Packaging

### 4.1 Dockerfile
**New file**: `Dockerfile` (project root)

Must target CUDA (RTX Pro 4500 evaluation environment):
```
FROM nvidia/cuda:12.3.0-runtime-ubuntu22.04
# Python 3.12, uv, sync without --extra mlx
# Install and configure Ollama with Qwen3.5-4B (pre-pulled at build time)
# Copy pipeline code
# EXPOSE 8888/udp (RTP)
# ENTRYPOINT: ./tools/run_pipeline.sh -s ollama-qwen3.5-4b
```

Multi-container note: Janus must also be running. Update `docker-compose.yml` to add a `pipeline` service that builds from the new Dockerfile alongside the existing `janus` service.

**Acceptance**: `docker compose up` starts both Janus and the pipeline; `uv run python tools/send_audio.py tests/fixtures/youtube_15min.wav` produces result files.

### 4.2 Approach document
**New file**: `docs/APPROACH.md`

1-2 pages: pipeline architecture, key design decisions (Janus for WebRTC bonus, faster-whisper, model choice), scoring strategy (quality floor first, then latency), any novel contributions (keyword validation, hallucination filter).

### 4.3 Benchmark skill
**New file**: `.claude/skills/benchmark-pipeline/skill.md`

Reads `results/*/window_*.json`, computes `L_i = 10·exp(−0.5·proc_time)` for each window, flags format violations (empty keywords, fewer than 3), and shows estimated C_i range across B_i assumptions.

### 4.4 Final submission checklist
- `destination_endpoint` set to challenge POST URL
- `encoding_clock_chan` verified against actual Janus stream (`opus/48000/2` vs `opus/48000/1`)
- All result files: valid JSON, exactly `from`, `to`, `summary`, `keywords` (3 items), `proc_time`
- `pytest tests/` passes
- `docker compose build && docker compose up` succeeds

---

## Execution Order

```
Phase 1 (all parallel, no deps)
  1.1 Keyword fix + tests
  1.2 Prompt tightening
  1.3 num_predict caps
  1.4 CLAUDE.md update

Phase 3.1 (eval harness) ← needed before Phase 2 model swaps

Phase 2 + Phase 3.2-3.3 (parallel once harness exists)

Phase 4 (last, needs arch finalized)
```

## Estimated Score Ceiling

| State | B_i | K_i | L_i | Janus | Total |
|-------|-----|-----|-----|-------|-------|
| Current (4B MLX, 9.35s) | ~15 | ~2 | ~0.1 | 4 | ~21 |
| After Phase 1 | ~18 | ~5 | ~0.5 | 4 | ~27 |
| After Phase 2+3 (3s proc_time) | ~20 | ~6 | ~2.2 | 4 | ~32 |
| Stretch (1s proc_time) | ~22 | ~6 | ~6.1 | 4 | ~38 |

## Files Modified

| Phase | File |
|-------|------|
| 1.1 | `plugins/nodes/proc/_summarizer_mlx/summarizer_mlx.py` |
| 1.1 | `plugins/nodes/proc/_summarizer_llm/summarizer_llm.py` |
| 1.1 | `tests/test_summarizer_mlx.py` (new) |
| 1.2 | `plugins/nodes/proc/_summarizer_mlx/summarize_prompt_mlx_qwen3.txt` |
| 1.2 | `plugins/nodes/proc/_summarizer_llm/summarize_prompt_ollama.txt` |
| 1.2 | `plugins/nodes/proc/_summarizer_mlx/summarize_prompt_mlx.txt` |
| 1.3 | `pipelines/summarizer/*.json` |
| 1.4 | `CLAUDE.md` |
| 2.2 | `pipelines/summarizer/ollama-qwen3.5-4b.json` (new) |
| 3.1 | `tools/eval_quality.py` (new) |
| 4.1 | `Dockerfile` (new) |
| 4.1 | `docker-compose.yml` |
| 4.2 | `docs/APPROACH.md` (new) |
| 4.3 | `.claude/skills/benchmark-pipeline/skill.md` (new) |
