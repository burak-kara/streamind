# Final Submission Commit — Public Repo

## Goal

Publish the locked pipeline to a **new public repo as a single commit** that
hides all development history and carries only what the submission needs to
build, run, and be graded. No results, eval/finetune scripts, alternate models,
datasets, or dev docs.

Decisions (this session): **runnable demo** (ship `send_audio.py` so a friend
can feed audio) + **fresh README only** (no CHALLENGE/RUNBOOK).

## Challenge deliverable contract (`docs/CHALLENGE.md` §"Submission output")

Submissions must contain: (1) code for all custom components, (2) the pipeline
config file, (3) the progressive summary objects, (4) a `Dockerfile`, (5) a
short approach markdown. All five are covered below.

## Manifest — INCLUDE

Build / runtime infra:
- `Dockerfile`, `docker-compose.yml`, `.dockerignore`
- `docker/entrypoint-pipeline.sh`
- `docker/janus/Dockerfile` + `docker/janus/conf/*.jcfg` (3 files) — Janus = +4 bonus
- `pyproject.toml`, `uv.lock` — **unchanged** (proven; editing forces a uni-lab relock)

Custom components (only nodes the locked pipeline loads):
- `plugins/nodes/proc/_audio_chunker/`
- `plugins/nodes/proc/_transcriber_whisper/`
- `plugins/nodes/proc/_hallucination_filter/`
- `plugins/nodes/proc/_novel_extractor/`
- `plugins/nodes/proc/_window_aggregator/`
- `plugins/nodes/proc/_summarizer_vllm/` — **minus** `summarize_prompt-v1.txt` (stale)
- `plugins/nodes/proc/_summarizer_common/` (`keywords.py`, `extractive.py`) — loaded by
  `summarizer_vllm.py` via `importlib` from the sibling dir; required
- `plugins/nodes/sink/_result_transmitter/`

Configs:
- `pipelines/config-base.json`
- `pipelines/summarizer/vllm-qwen3.5-4b-awq.json` (locked profile)
- `pipelines/config-submission.json` (assembled 8-node config; clean paths verified)

Operational tools (only the run path):
- `tools/fetch_models.sh`, `tools/assemble_config.py`, `tools/run_pipeline.sh`,
  `tools/lib/pipeline_prep.sh`, `tools/send_audio.py`

Deliverables / docs:
- `docs/APPROACH.md` (required §5; ships near-as-is)
- `submission/sample_outputs/window_0..4.json` (required §3 — progressive summary objects)
- **NEW** `README.md` (build/run/output quickstart, written fresh)
- **NEW** `.gitignore` (minimal: `models/ results/ tmp/ __pycache__/ .venv/ .DS_Store`)

## Manifest — EXCLUDE (development work)

- `results/` (4213 files), `datasets/` (audio + transcripts)
- `tests/` (dev verification)
- `tools/eval_quality.py`, `tools/eval_multi_judge.py`, `tools/compare_summarizers.sh`,
  `tools/postproc/`, `tools/finetune/`, `tools/youtube-download.sh`
- `pipelines/judge/*`, `pipelines/summarizer/{vllm-llama-3.1-8b-instruct, vllm-gemma-3-12b-it-awq}.json`
- `plugins/nodes/sink/_judge_common/` (eval-only), `plugins/nodes/CLAUDE.md`
- `docs/` everything except APPROACH.md: `CHALLENGE.md`, `RUNBOOK.md`, `TODO.md`,
  `LESSONS.md`, `CLAUDE.md`, `MACHINE-SPECS.md`, `plans/`, `documentation/`, `images/`
- Root `CLAUDE.md`, `AGENTS.md`, `.claude/`, `.remember/`

## Notes on weights & deps

- **Weights are NOT in the repo** (correct — git can't hold ~7 GB, and they're
  gitignored). The `Dockerfile` fetches + bakes them at build (`fetch_models.sh`
  → Qwen3.5-4B-AWQ; `snapshot_download` → faster-whisper). Build needs network
  **once**; runtime is offline (`HF_HUB_OFFLINE=1`). README states this plainly.
- `send_audio.py` host deps (`aiortc`, `httpx`) already live in the `dev` extra.
  README documents `uv sync --extra dev` for the host that injects audio. No
  pyproject/lock edits → no relock.

## Commit mechanism — staging dir + fresh `git init` (primary)

Cleanest "single commit, no shared history" — physically isolated, fully
auditable before push.

```bash
SRC=$(pwd)
OUT=./tmp/streamind-submission
rm -rf "$OUT" && mkdir -p "$OUT"

# copy manifest preserving structure (rsync -R anchors at SRC root)
cd "$SRC"
rsync -R \
  Dockerfile docker-compose.yml .dockerignore \
  docker/entrypoint-pipeline.sh docker/janus/Dockerfile docker/janus/conf/*.jcfg \
  pyproject.toml uv.lock \
  pipelines/config-base.json pipelines/config-submission.json \
  pipelines/summarizer/vllm-qwen3.5-4b-awq.json \
  plugins/nodes/proc/_audio_chunker plugins/nodes/proc/_transcriber_whisper \
  plugins/nodes/proc/_hallucination_filter plugins/nodes/proc/_novel_extractor \
  plugins/nodes/proc/_window_aggregator plugins/nodes/proc/_summarizer_vllm \
  plugins/nodes/proc/_summarizer_common plugins/nodes/sink/_result_transmitter \
  tools/fetch_models.sh tools/assemble_config.py tools/run_pipeline.sh \
  tools/lib/pipeline_prep.sh tools/send_audio.py \
  docs/APPROACH.md submission/sample_outputs \
  "$OUT/"

# drop the stale prompt + any __pycache__
rm -f "$OUT/plugins/nodes/proc/_summarizer_vllm/summarize_prompt-v1.txt"
find "$OUT" -name __pycache__ -type d -prune -exec rm -rf {} +

# add fresh README.md + .gitignore (generated, see below)
# ... write files ...

cd "$OUT"
git init -q && git add -A
git commit -m "STREAMIND: real-time meeting-intelligence pipeline (challenge submission)"
git branch -M main
# user supplies the new public remote and pushes:
git remote add origin <NEW_PUBLIC_REPO_URL>
git push -u origin main
```

Acceptance gate before push:
- `git -C ./tmp/streamind-submission ls-files` == manifest exactly (no
  results/datasets/tests/eval/finetune/judge/alt-profiles/dev-docs).
- `rg -i 'ollama|mlx|uni-lab|rsync|judge|finetune' ./tmp/streamind-submission`
  → only intended mentions (none in code paths).
- Single commit, `main` branch, no parents.
- **(Optional, on uni-lab)** clean-room build from the staging dir:
  `docker compose build` succeeds; `docker compose up` + `send_audio.py` →
  ≥1 valid `window_*.json` with exact `{from,to,summary,keywords[3],proc_time}`.

## Division of labor

I prepare the staging dir + fresh README/.gitignore + the local single commit.
**User** runs the clean-room build verification on uni-lab (per workflow) and
performs `git remote add` + `git push` to the new public repo (their creds).

## Risk

- **Dropping a needed file** → build/run breaks. Mitigation: manifest derived
  from the actual `COPY`/`source`/`importlib` chain; acceptance gate runs a
  clean-room build before the repo goes public.
- **`pyproject` extras leak dev intent** (`dev`, `finetune` list pytest/ollama/etc.).
  Kept to avoid a relock; they install nothing in the image. Optional follow-up:
  trim to base + a `client` extra and relock on uni-lab.
