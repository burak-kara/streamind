# CLAUDE.md — academic-paper

IEEE MMSP 2026 Grand Challenge paper: *"STREAMIND: A Real-Time AI Meeting Intelligence Pipeline"*

Describes the system built for the STREAMIND Grand Challenge — live Opus/RTP audio → incremental ASR → novel chunk extraction → window aggregation → LLM summarization + keyword extraction, all via Juturna nodes with Janus WebRTC ingestion.

## Compilation

Run from the `academic-paper/` directory:

```bash
./compile.sh          # full build: pdflatex → bibtex → pdflatex × 2; output at main.pdf
./compile.sh --draft  # single pdflatex pass — fast, no bibliography
./compile.sh --clean  # wipe build/ then full build
```

Output goes to `build/`; final `main.pdf` is copied to the `academic-paper/` root.

## Document Structure

```text
academic-paper/
  main.tex              # Root document (IEEEtran, conference mode)
  references.bib        # BibTeX citations
  compile.sh            # Build script
  build/                # All LaTeX aux/log/pdf output (gitignored)
  helpers/
    packages.tex        # \usepackage declarations and pgfplots/siunitx setup
    commands.tex        # Custom macros, math shorthands, annotation commands
    colors.tex          # Color palette + TikZ aliases
    glossaries.tex      # \newacronym entries for streaming/networking terms
    tikz-styles.tex     # pgfplots cycle lists, axis styles, legend styles
  figures/              # \input fragments embedded in figure environments
    pipeline.tex          # F1 — pipeline block diagram (TikZ)
    scoring-curve.tex     # F2 — L_i = 10*exp(-0.5*proc_time) curve (pgfplots)
    chunk-overlap.tex     # F3 — chunking + novel-extraction timing (TikZ)
    phase1-ablation.tex   # F4 — Pre/Post Phase-1 bar chart (pgfplots)
    model-sweep.tex       # F5 — Qwen3.5 2B/4B/9B sweep (pgfplots)
  data/                 # CSV inputs for pgfplots tables
    README.md             # CSV schema + which figure consumes which file
```

## Custom Macros

Defined in `helpers/commands.tex`:

| Macro | Expands to |
|-------|------------|
| `\ie`, `\eg`, `\etc`, `\etal` | Standard Latin abbreviations |
| `\td` | `\textemdash` |
| `\cA`–`\cY` | Calligraphic math letters `\mathcal{A}` … |
| `\E`, `\P` | Expectation / probability (`\mathds`) |
| `\ssection{title}` | Bold inline heading (no numbering) |
| `\todo{text}` | Red TODO annotation |
| `\quotes{text}` | ``text'' |

Key acronyms from `helpers/glossaries.tex` (use `\gls{key}`):

| Key | Expansion |
|-----|-----------|
| `webrtc` | WebRTC |
| `rtp` | RTP |
| `asr` | ASR |
| `llm` | LLM |
| `awq` | AWQ (Activation-aware Weight Quantization) |
| `vram` | VRAM |
| `ai` | AI |
| `gpu`, `cpu` | GPU, CPU |
| `api` | API |

## Challenge Context

Scoring formula (see root `CLAUDE.md`): `C_i = B_i + K_i + L_i`. Paper must explain:

- Pipeline architecture (7 Juturna nodes)
- Model choices (faster-whisper ASR, Qwen3.5 summarizer via in-process vLLM, int4-AWQ where VRAM-constrained)
- Inference-engine choice (vLLM over SGLang / TensorRT-LLM / Ollama)
- Latency optimisation strategy
- Janus WebRTC integration and Janus bonus (+4)
- Evaluation via cross-family multi-judge panel (Mistral-Small-24B, Phi-4, Gemma-3-27B), consensus mean/stdev

## Academic Writing & Proofread Mode

Proofreader for IEEE conference manuscripts in multimedia systems and real-time AI.

Primary task: correct grammar, syntax, and style to IEEE conference standards.

### Operating rules

- Fix subject-verb agreement, tense consistency, article usage, punctuation.
- American English, serial comma.
- Preserve passive voice standard for methodology. Convert to active only for clarity.
- Do not alter technical meaning. Flag ambiguity with `% TODO: Ambiguous — do you mean X or Y?`
- Use American English (e.g., "optimize", "analyze", "modeling").

### LaTeX formatting rules

Output: inline LaTeX code block using ` ```latex ```.
No markdown styling.
Each sentence on its own line. No `\\` for line breaks.

Required packages (loaded via `helpers/packages.tex`):

- `siunitx`: `\num{}` for standalone numbers, `\SI{}{}` for numbers with units.
- `glossaries`: `\gls{}` for acronyms.

Preserve all custom macros from `helpers/commands.tex` unchanged.

### When asked to extend content

- Add substantive technical content consistent with the surrounding argument.
- Mark additions with: `% ADDED: [brief rationale]`
- Do not fabricate citations or benchmark results.

### When asked for a scientific review

Switch to reviewer mode: evaluate as a senior IEEE MMSP reviewer. Provide Major and Minor Comments. Do not mix with proofreading output.

### References

- Use `\cite{}` with keys from `references.bib`.
- Do not fabricate references. If a citation is missing, flag with `% TODO: Missing citation for [claim]`.
- Use this format for non academic misc/online resources:

```latex
@misc{label,
 title        = {TTitle of the resource},
year         = {year},
 author       = {Author Name(s)},
 note         = {Accessed on <date>},
 howpublished = {[Online] Available: \url{<URL>}}
}
```
