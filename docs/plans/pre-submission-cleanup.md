# Pre-Submission Code Cleanup

## Context

Before the M3 submission we want the repo to carry only what the locked pipeline
needs — no abandoned experiments, orphan configs, or stale docs. The summarizer
is locked to `qwen3.5-4b-awq`; **M4 finetuning was attempted and abandoned**
(all FT runs scored below base). A safety tag `pre-cleanup` was created at main
(`a9b4f59`) so every removal below is recoverable from git history.

Three independent review passes found: **the Python logic is clean** (no dead
functions/classes — Juturna lifecycle methods correctly excluded). The stale
surface is entirely (a) the abandoned finetune toolchain + its artifacts, (b)
orphan config profiles, and (c) doc/code drift. Findings are evidence-backed
(reference counts via `rg`).

## Tier A — Stale removals (high confidence)

Each item below has **0 active references** in the locked pipeline.

- [ ] **4 finetuned summarizer profiles** (M4 abandoned, 0 refs):
  `pipelines/summarizer/vllm-qwen3.5-4b-ft.json`, `…-ft-gemini.json`,
  `…-ft-full-gemini.json`, `vllm-qwen3-4b-2507-ft.json`.
- [ ] **Stale FT result dirs** (tracked, ~bloat):
  `results/prev_results/qwen3.5-4b-ft/`, `…-ft-gemini/`, `…-ft-full-gemini/`.
- [ ] **`requirements.txt`** — lists `ollama`, superseded by `pyproject.toml` (uv
  is the single source). The only `requirements.txt` matches in `rg` are inside
  Juturna's vendored HTML docs, not our file → 0 real references. Remove.
- [ ] **`pipelines/audio_src/_ffmpeg_launcher.sh` + `_session_in.sdp`** — legacy
  ffmpeg ingest path, replaced by Janus RTP, 0 config references.

## Tier B — Scope decisions (need user input)

- [ ] **Finetune toolchain `tools/finetune/`** (7 `.py`, `prepare_rev16.py` still
  imports the removed `ollama` client) + `tests/test_prepare_rev16.py` + the
  FT-only `dev` deps (`peft`, `trl`, `bitsandbytes`, `accelerate`, `datasets`).
  Options: **remove** (tag preserves it) / keep but isolate deps into a separate
  `finetune` extra / keep as-is.
- [ ] **`plugins/nodes/source/_audio_file/`** — WAV file source. 0 external
  references; the challenge + Docker use `audio_rtp` via Janus exclusively, and
  CLAUDE.md actively discourages `audio_file`. Options: **remove** / keep for
  local testing.
- [ ] **Alternate non-FT summarizer profiles** — `vllm-qwen3.5-4b` (non-AWQ),
  `vllm-qwen3.5-9b`, `vllm-qwen3.5-9b-awq`, `vllm-qwen3-4b-2507`,
  `vllm-llama-3.1-8b`. Used by the dev `compare_summarizers.sh` bake-off, NOT the
  submission; they ship in the image (Dockerfile `COPY pipelines/`). Options:
  keep (dev tooling) / prune to the locked profile only (lean submission).

## Tier C — Doc / config drift fixes (low risk)

- [ ] `plugins/nodes/CLAUDE.md`: `max_tokens` documented `150` → actual code
  default `256` (profile overrides to `384`). Fix to `256`.
- [ ] `plugins/nodes/CLAUDE.md`: prompt directive documented `/no_think` → actual
  `summarize_prompt.txt` uses `/nothink`. Fix.
- [ ] `plugins/nodes/CLAUDE.md`: add one line that the chosen profile overrides
  `config.toml` defaults at assembly time.
- [ ] `pipelines/config-base.json` (or RUNBOOK): note the `summarizer` node is
  injected at runtime by `assemble_config.py` (referenced in links, not in
  `nodes[]`).
- [ ] Note `payload_type: 97` / `opus/48000/1` rationale vs the CHALLENGE.md
  PCMU/8k example (mono-Opus per spec, resampled to 16k for Whisper).
- [ ] Scrub any remaining `ollama`/`mlx` mentions that become wrong once finetune
  is gone (keep deliberate "no Ollama/MLX variants" historical notes).

## Tier D — Code polish (low)

- [ ] `keywords.py`: extract `REQUIRED_KEYWORD_COUNT = 3` (replaces 4 hardcoded
  `3`s); comment the `+5` candidate buffer.
- [ ] Type hints: `ensure_three_keywords(keywords: list[str]) -> list[str]`;
  `result_transmitter.update(...) -> None`.

## Tier E — Test additions (medium)

- [ ] Add `DESTINATION_ENDPOINT` env-fallback tests to
  `tests/test_result_transmitter.py` (config wins; env used when config empty) —
  covers behavior added in M3.

## Verification

- `rg -i 'ollama|mlx' -g '!docs/**'` returns only deliberate historical notes (no
  live deps/imports).
- `uv lock` succeeds after dropping FT deps; `.venv/bin/pytest tests/` green
  (minus the removed finetune test).
- `tools/run_pipeline.sh -p vllm-qwen3.5-4b-awq` still assembles + the Docker
  build still bakes only the locked profile.
- `git status` clean; nothing referenced-but-missing (grep each removed name).

## Out of scope
- No behavior changes to the live pipeline nodes (logic reviewed clean).
- Finetune research can be resumed from the `pre-cleanup` tag if revived.
