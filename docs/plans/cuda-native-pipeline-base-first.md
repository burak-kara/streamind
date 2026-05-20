# STREAMIND Cleanup — Native CUDA Pipeline, Base-First

## Context

Project pivoted: development moved to `uni-lab` (RTX 4090, CUDA 12.4); submission target is RTX Pro 4500 (single GPU, 24 GB). Drop all middleware — no Ollama daemon, no MLX. Replace with **vLLM in-process** (`from vllm import LLM`) so summarizer runs natively in the pipeline Python process. Submission must be portable, self-contained, real-time. **Model weights ship with the submission** (baked into Docker image); pipeline code never downloads at runtime.

**Working order:** get base CUDA pipeline running end-to-end with an off-the-shelf open-source model first. Only after the pipeline is stable, measured, and meets format/latency targets do we revisit finetuning. This plan covers M0 + M1 in detail and sketches M2–M4 as follow-on milestones.

User decisions:

- **Inference backend:** vLLM in-process (`LLM` class, no server)
- **Legacy code:** delete MLX + Ollama entirely (no extras kept)
- **Model packaging:** weights shipped with submission; no runtime HF download; dev-only fetch script
- **Model identity:** picked 2026-05-19 — summarizer `Qwen/Qwen3.5-4B` (Apache-2.0, ~10 GB BF16); judge (offline) `cyankiwi/Qwen3.5-27B-AWQ-BF16-INT4` (~14 GB AWQ-int4)
- **Finetune:** parked — `tools/finetune/` left in place but unused until base pipeline green

---

## Alignment with `docs/CHALLENGE.md`

| Challenge requirement | Where addressed in plan |
| --- | --- |
| Opus-encoded **mono** RTP via Janus | `pipelines/config-base.json` — verify `encoding_clock_chan: "opus/48000/1"` |
| Incremental ASR over short overlapping chunks | `_transcriber_whisper` (unchanged) + `_audio_chunker` (unchanged) |
| Novel-chunk dedup | `_novel_extractor` (unchanged) |
| **300 s** window aggregation | `_window_aggregator` default; `run_pipeline.sh -w 300` |
| Structured summary + 3 keywords | `_summarizer_vllm` reuses `keywords.ensure_three_keywords()` |
| Local store + POST to destination endpoint | `_result_transmitter` (unchanged); endpoint populated in M3 |
| **B_i** (LLM judge, 5 Likert, max 25) | Offline `tools/eval_quality.py` using `_judge_common/scorer.py` |
| **K_i** (3 keywords, ±2, max 6) | `keywords.ensure_three_keywords()` guarantees count; judge scores relevance |
| **L_i** (`10·e^(−0.5·proc_i)` if B_i ≥ 10) | `proc_time` measured end-to-end in `_result_transmitter`; judge applies gate |
| **Janus bonus (+4)** | Pipeline source is `audio_rtp` behind Janus, not `audio_file` |
| Output keys: `from`, `to`, `summary`, `keywords`, `proc_time` | `_result_transmitter` key remap; verification step 6 grep-guards |
| **Single RTX Pro 4500 (24 GB)** fits all models | VRAM budget check in verification; summarizer-only at submission (judge offline) |
| Submission deliverables: code + config + progressive summaries + **Dockerfile** + approach `.md` | M3 builds Dockerfile; `docs/APPROACH.md` rewritten in M0; `pipelines/config-base.json` + assembled config shipped; results dir produced by smoke run |
| Test against 30+ min audio (6 summary chunks) | Verification step 5 uses 30+ min source (existing 15 min fixture insufficient — see open item below) |

---

## Phasing

| Milestone | Goal | Scope |
| --- | --- | --- |
| **M0 — Cleanup** | Single CUDA path in repo | Delete MLX/Ollama nodes, configs, prompts, tests; drop `ollama`/`mlx-lm` deps; rewrite CLAUDE.md, skills, docs to match |
| **M1 — Base CUDA pipeline working** | End-to-end vLLM summarizer on uni-lab producing valid submission JSON | New `_summarizer_vllm` node; one summarizer profile; one judge profile (offline); pyproject `cuda` extra; `tools/fetch_models.sh` dev script; pipeline loads from local `./models/` path; 30 s + 30 min smoke tests pass |
| **M2 — Prompt + sampling tune** | Maximise B/K on chosen base model | Iterate `summarize_prompt.txt`, sampling params; rerun offline judge; lock best prompt |
| **M3 — Submission packaging** | Self-contained Docker on CUDA 12.4 + populated `destination_endpoint` + format guard | `Dockerfile` rewrite, model **baked into image** at build time (no runtime download), end-to-end smoke in container, `docs/APPROACH.md` final |
| **M4 — Finetune (later)** | Quality lift via QLoRA on rev16 | Replace `prepare_rev16.py` teacher (Ollama → vLLM), run finetune → merge → swap merged dir into vLLM profile, A/B vs base |

This plan executes **M0 + M1** now. M2/M3 follow once M1 is green. M4 is gated on M2/M3.

---

## Architecture changes (M0 + M1)

| Stage | Before | After |
| --- | --- | --- |
| Summarizer | `_summarizer_llm` (Ollama HTTP) / `_summarizer_mlx` (Apple) | `_summarizer_vllm` (CUDA in-process, vLLM `LLM`, **local path only**) |
| Judge | `_judge_llm` Juturna sink (Ollama, async in-pipeline) | Offline-only: `tools/eval_quality.py` loads judge model sequentially after summarizer is unloaded. **No live judge sink.** |
| Backend dep | `ollama`, `mlx-lm` extras | `vllm>=0.6` in new `cuda` extra |
| Profiles | 5 summarizer + 2 judge configs | 1 summarizer + 1 judge config (judge config consumed only by `eval_quality.py`) |
| Prompt files | 3 variants (`_ollama`, `_mlx`, `_mlx_qwen3`) | 1 (`summarize_prompt.txt`) |
| Model source | HF download at first call (Ollama pull / MLX load) | **Local `./models/<name>/` directory only.** Dev fetch via `tools/fetch_models.sh`; submission Docker fetches during `docker build` |

vLLM init (warmup): `LLM(model="./models/<name>", dtype="float16", gpu_memory_utilization=<frac>, max_model_len=2048, enforce_eager=False)`. Generation: `llm.generate([prompt], SamplingParams(temperature=0.3, top_p=0.9, max_tokens=150))`. The `model_name` config field is **always a filesystem path** — never an HF id at runtime. Same path used in dev and submission.

**Base model pick (M1):** `Qwen/Qwen3.5-4B` (Apache-2.0, BF16 native, ~8 GB on disk, ~10 GB with KV at 2K). Judge: `cyankiwi/Qwen3.5-27B-AWQ-BF16-INT4`. Picked 2026-05-19 against constraints:

- ≤ ~20 GB fp16 to leave VRAM headroom on 24 GB Pro 4500 — **Qwen3.5-4B: ~10 GB ✓**
- Permissive license (Apache 2.0 / MIT / Qwen / Llama community) for redistribution in the submission image — **Apache-2.0 ✓**
- Strong instruction-following + JSON output reliability for the summary contract — **IFEval 91.5 on Qwen3.5-4B ✓**
- Available on HuggingFace (or another permanent mirror) so `fetch_models.sh` is reproducible — **HF live ✓**

Fallback (if B_i averages < 15 on offline judge): `Qwen/Qwen3.5-9B` — same family, +~2 IFEval, ~20 GB fp16, ~2 s proc_time → L_i ≈ 3.7. Same prompt + tooling.

**Judge path:** offline sequential. Summarizer pipeline runs, writes `results/.../window_N.json`. After pipeline exits (or via separate run), `tools/eval_quality.py results/.../` loads judge LLM, scores each window, writes `results/.../judge/window_N.json`. Eliminates VRAM co-residency problem.

---

## Model packaging strategy (new)

Pipeline code **never** triggers a model download. Three contexts:

1. **Local dev (uni-lab):** `./tools/fetch_models.sh <hf_id> <local_name>` populates `./models/<local_name>/`. Run once after clone. Idempotent.
2. **CI / unit tests:** vLLM is mocked; no model needed.
3. **Submission Docker:** `Dockerfile` runs the same `fetch_models.sh` (or inlined `huggingface-cli download`) during `docker build`, so the resulting image carries the weights. At runtime the container has no network dependency.

Layout:

```text
./models/
  <local_name>/           # weights, tokenizer, config.json — committed paths in .gitignore
  ...
tools/fetch_models.sh     # wraps `huggingface-cli download <hf_id> --local-dir ./models/<local_name>`
```

`.gitignore` adds `models/` so weights are never committed to git.

Pipeline configs reference local paths only:

```json
{
  "mark": "summarizer_vllm",
  "configuration": {
    "model_name": "./models/<local_name>",
    "dtype": "float16",
    "gpu_memory_utilization": 0.85,
    "max_model_len": 2048,
    "max_tokens": 150,
    "temperature": 0.3,
    "top_p": 0.9,
    "prompt_template_file": "summarize_prompt.txt"
  }
}
```

`_summarizer_vllm.summarizer_vllm` validates that `model_name` resolves to an existing local directory at warmup; raises a clear error directing the user to run `tools/fetch_models.sh` if missing.

---

## File deletions (M0)

```text
plugins/nodes/proc/_summarizer_mlx/                              # MLX node
plugins/nodes/proc/_summarizer_llm/                              # Ollama node (replaced)
pipelines/summarizer/mlx-Qwen3.5-2B-OptiQ-4bit.json
pipelines/summarizer/mlx-Qwen3.5-4B-OptiQ-4bit.json
pipelines/summarizer/mlx-Qwen3.5-9B-OptiQ-4bit.json
pipelines/summarizer/ollama-qwen3.5-4b.json
pipelines/summarizer/ollama-qwen3.5-9b.json
pipelines/judge/ollama-qwen3.5-4b.json
pipelines/judge/ollama-qwen3.5-9b.json
plugins/nodes/sink/_judge_llm/                                   # deleted; judge is now offline-only via tools/eval_quality.py
tests/test_summarizer_mlx.py
tests/test_summarizer_llm.py                                     # rewritten as _vllm variant
tests/test_judge_llm.py                                          # rewritten to cover offline eval_quality.py path
# Old prompt templates folded into single file (deleted via dir removals above):
#   plugins/nodes/proc/_summarizer_llm/summarize_prompt_ollama.txt
#   plugins/nodes/proc/_summarizer_mlx/summarize_prompt_mlx*.txt
```

`tools/finetune/` left **untouched** in M0. Files there (`export_to_ollama.sh`, `modelfile.template`, `prepare_rev16.py` Ollama call) become an M4 concern.

---

## File creations (M1)

```text
plugins/nodes/proc/_summarizer_vllm/
  __init__.py
  summarizer_vllm.py                # mirrors old summarizer_llm.py structure;
                                    # vllm.LLM warmup from local dir; reuses
                                    # JSON parsing + keywords.ensure_three_keywords()
  node.json                         # Juturna manifest (match _summarizer_llm)
  summarize_prompt.txt              # single prompt (model-appropriate chat template)

pipelines/summarizer/vllm-qwen3.5-4b.json   # mark: summarizer_vllm,
                                            # model_name: ./models/qwen3.5-4b

pipelines/judge/vllm-qwen3.5-27b-awq.json   # consumed only by tools/eval_quality.py

tools/fetch_models.sh                       # dev-only HF download helper

.gitignore                                  # add `models/` (if not already)
```

No `_judge_vllm/` Juturna sink — judge runs offline only.

---

## File modifications

### Code (M0 + M1)

- `pyproject.toml` — drop `ollama` from base deps; drop `mlx` extra; add:

  ```toml
  cuda = ["vllm>=0.6.0", "torch==2.5.1", "huggingface-hub[cli]"]
  ```

  Leave `finetune` extra alone (stays, unused until M4). `huggingface-hub[cli]` powers `fetch_models.sh`.
- `plugins/nodes/proc/_summarizer_common/keywords.py` — **unchanged** (reused by new node).
- `plugins/nodes/sink/_judge_common/scorer.py` — **unchanged** (prompt + parser + B/K/L/C math reused).
- `pipelines/config-base.json` — verify `encoding_clock_chan: "opus/48000/1"` (mono per CHALLENGE.md). `destination_endpoint` stays empty for M1; populated in M3.
- `tools/run_pipeline.sh` — drop MLX/Ollama profile defaults; default `--summarizer vllm-qwen3.5-4b`; **drop `--judge` flag entirely** (judge is now offline-only); drop Ollama pre-flight check; add `nvidia-smi` + local-model-path pre-flight (refuse to start if `./models/<name>` missing).
- `tools/assemble_config.py` — drop judge merging logic; verify glob matches new summarizer profile name.
- `tools/eval_quality.py` — rewrite as offline judge runner: load `pipelines/judge/vllm-qwen3.5-27b-awq.json`, instantiate `vllm.LLM` from local path, iterate `results/<model>/<window>/window_*.json`, write `results/.../judge/window_N.json` via `_judge_common/scorer.py`. CLI: `eval_quality.py <results-dir> [--judge-profile vllm-qwen3.5-27b-awq]`.
- `tools/send_audio.py` — unchanged.
- `tools/finetune/*` — **untouched** in M0/M1. Revisit in M4.

### Tests (M0 + M1)

- `tests/test_summarizer_vllm.py` (new) — mock `vllm.LLM` to avoid CUDA dep at test time; assert prompt format, JSON parse, keyword guarantees, **clear error when local model dir missing**.
- `tests/test_eval_quality.py` (new, replaces old `test_judge_llm.py`) — mock vLLM; assert offline scorer iterates result JSONs, writes judge JSONs with correct B/K/L/C schema.
- `tests/test_keywords.py` — unchanged.
- Audio pipeline tests (`test_audio_*`, `test_chunker_*`, `test_window_aggregator*`, etc.) — unchanged.

### CLAUDE.md files (M0)

- `CLAUDE.md` (root): rewrite **Evaluation Environment**, **Quick Start**, **Runtime Requirements**, **Current Model Choices**, **Gotchas**, **Skills** sections. Strip all MLX + Ollama references. Quick Start: install `uv sync --extra dev` on uni-lab; run `./tools/fetch_models.sh <hf_id> <local_name>` once; drop Ollama daemon step; keep Janus. New gotchas: vLLM only installs on Linux + CUDA; pipeline refuses to start if `./models/<name>` missing.
- `docs/CLAUDE.md`: drop Ollama/MLX cross-references; point to new vLLM node; mention `models/` dir.
- `docs/documentation/CLAUDE.md`: drop Ollama doc links; keep Juturna + Janus.
- `plugins/nodes/CLAUDE.md`: rewrite node table (single summarizer, no live judge); remove MLX/Ollama model-choice rows.

### Skills (M0)

- `.claude/skills/run-pipeline/SKILL.md` — drop `ollama serve` / `ollama pull`; add `python -c "import vllm"` + `nvidia-smi` + `ls ./models/<name>` checks; note ~30 s warmup on first model load.
- `.claude/skills/swap-model/SKILL.md` — swap target is **local path** in `pipelines/summarizer/*.json` plus a `fetch_models.sh` call; drop Ollama `pull` step.
- `.claude/skills/prep-submission/SKILL.md` — rewrite checklist (model present under `./models/`, vLLM smoke test, VRAM budget, format check, Janus bonus verification, endpoint populated). M3 polishes further.
- `.claude/skills/benchmark-pipeline/SKILL.md` — minor: update example commands.
- `.claude/skills/tune-prompt/SKILL.md` — refactor to drive vLLM directly against the new profile (used heavily in M2).
- `.claude/skills/remote-finetune/SKILL.md` — leave content intact; add note at top: **"Activate in M4 only — base pipeline must be green first."**

### Docs (M0 — light pass; M2/M3 deepen)

- `docs/APPROACH.md` — rewrite inference section: native CUDA via vLLM in-process, model weights shipped in image, no runtime download. Justify choice (no daemon, batched, paged attention, fits 24 GB), explicitly contrast with old Ollama/MLX path. Polished as final submission artifact in M3.
- `docs/TODO.md` — replace MLX/Ollama items with the milestone list above; mark M0/M1 as active.
- `docs/plans/cloud-development-submission-plan.md` — mark superseded by this plan.
- `docs/plans/given-the-changes-in-streamed-biscuit.md` — review; likely archive.
- `docs/plans/submission-readiness-remaining.md` — refresh with vLLM-era blockers; defer Docker/endpoint items to M3.

### Docker / submission packaging (M3, sketch only here)

- `Dockerfile` — base `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`; install `uv`, run `uv sync` (no extras — vLLM is a base dep; dev tooling and finetune deps stay out of the image); **`RUN ./tools/fetch_models.sh <hf_id> <local_name>` during build** so weights are baked into the image (uses `huggingface-hub` from `--extra dev` during build, not at runtime); entrypoint runs `tools/run_pipeline.sh`. No HF download at container start.
- `docker-compose.yml` — keep Janus; drop any Ollama service; GPU passthrough on pipeline service.

Not built in this plan execution; flagged so the M3 hand-off knows what to expect.

---

## Open items requiring user input

1. ~~**Model id**~~ — picked 2026-05-19: summarizer `Qwen/Qwen3.5-4B`, judge `cyankiwi/Qwen3.5-27B-AWQ-BF16-INT4`.
2. **30+ min test audio** — current `tests/fixtures/youtube_15min.wav` gives only 3 windows at 300 s. CHALLENGE.md recommends ≥ 30 min (6 chunks). Need either a longer fixture or a `rev16`/`ietf` sample for M1 verification step 5.

---

## Critical files to read before execution

- `plugins/nodes/proc/_summarizer_llm/summarizer_llm.py` — template for `_summarizer_vllm`
- `plugins/nodes/sink/_judge_llm/judge_llm.py` — reference for offline judge logic (reuses `_judge_common/scorer.py`)
- `plugins/nodes/sink/_judge_common/scorer.py` — reused unchanged
- `plugins/nodes/proc/_summarizer_common/keywords.py` — reused unchanged
- `plugins/nodes/sink/_result_transmitter/result_transmitter.py` — confirm `proc_time` is measured end-to-end, key remap is intact
- `pipelines/config-base.json` — mono encoding + Janus source check
- `tools/run_pipeline.sh` — CLI surface
- `pyproject.toml` — extras
- `docs/CHALLENGE.md` — scoring contract source of truth
- `.claude/skills/prep-submission/SKILL.md` — checklist source of truth

---

## Verification (M0 + M1 done bar)

Run on `uni-lab` (RTX 4090, CUDA 12.4):

1. **Install:** `ssh uni-lab "cd ~/Desktop/streamind && uv sync --extra dev"` — succeeds without manual flags.
2. **Model fetch (dev only):** `ssh uni-lab "cd ~/Desktop/streamind && ./tools/fetch_models.sh <hf_id> <local_name>"` populates `./models/<local_name>/`. Re-run is a no-op.
3. **Import smoke:** `ssh uni-lab "cd ~/Desktop/streamind && uv run python -c 'from vllm import LLM, SamplingParams; print(LLM.__module__)'"`
4. **Unit tests:** `uv run pytest tests/ -x` locally (mocked vLLM) and on uni-lab — all pass. Includes test that asserts pipeline refuses to start when `./models/<name>` is absent.
5. **30 s smoke pipeline:**

   ```bash
   ssh uni-lab "cd ~/Desktop/streamind && ./tools/run_pipeline.sh -w 30 -s vllm-qwen3.5-4b"
   # in parallel: uv run python tools/send_audio.py tests/fixtures/youtube_15min.wav
   ```

   ≥ 1 `results/.../window_*.json` with exactly keys `{from, to, summary, keywords[3], proc_time}`.
6. **30 min window + offline judge** (requires 30+ min fixture — see open item 3):

   ```bash
   ssh uni-lab "cd ~/Desktop/streamind && ./tools/run_pipeline.sh -s vllm-qwen3.5-4b"
   # after pipeline exits:
   ssh uni-lab "cd ~/Desktop/streamind && uv run python tools/eval_quality.py results/<local_name>/300/"
   ```

   ≥ 6 summary windows + `results/.../judge/window_*.json` with B/K/L/C scores. Sequential — summarizer LLM unloaded before judge LLM loads.
7. **Format check:** `/benchmark-pipeline` on result dir — no format violations; K_i computed; L_i applied only when B_i ≥ 10; `proc_time` measured end-to-end.
8. **Janus bonus:** confirm pipeline source is `audio_rtp` (not `audio_file`) by reading the assembled config emitted by `assemble_config.py`.
9. **Grep guard:** `rg -i 'ollama|mlx' --type py --type json -g '!docs/**' -g '!tools/finetune/**'` returns **zero** matches.
10. **VRAM budget:** during run, `nvidia-smi` shows peak GPU memory < 24 GB (RTX Pro 4500 target).
11. **No-network sanity:** `unshare -n` (or `ip link set <iface> down` in a throwaway env) — pipeline still starts and produces results, proving no runtime download.

M0 + M1 complete when 1–11 pass. M2 (prompt tune) and M3 (Docker + endpoint + final APPROACH.md) follow as separate workstreams once base pipeline is green. M4 (finetune) gated on M3.
