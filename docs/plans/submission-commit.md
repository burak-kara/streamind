# Final Submission Commit — Public Repo

## Goal

Publish the locked pipeline to a **new public repo as a single commit** that
hides all development history and carries only what the submission needs to
build, run, and be graded. No results, eval/finetune scripts, alternate models,
datasets, or dev docs.

Decisions (this session): **runnable demo** (ship `send_audio.py` so a friend
can feed audio) + **fresh README only** (no CHALLENGE/RUNBOOK) + **trim deps**
(submission-only `client` extra, relock on uni-lab).

> **Status: executed.** See the [Outcome](#outcome) section at the bottom for
> the actual commit (`d0c23bd`, rootless, 51 files, verified). The mechanism and
> deps sections below describe what was actually done.

## Challenge deliverable contract (`docs/CHALLENGE.md` §"Submission output")

Submissions must contain: (1) code for all custom components, (2) the pipeline
config file, (3) the progressive summary objects, (4) a `Dockerfile`, (5) a
short approach markdown. All five are covered below.

## Manifest — INCLUDE

> **Superseded for the final commit** — see [Update (2026-06-25)](#update-2026-06-25--single-config-submission)
> at the bottom. This list reflects `d0c23bd` (51 files); the shipped submission
> is `4733d02` (45 files, single config, approach folded into README).

Build / runtime infra:
- `Dockerfile`, `docker-compose.yml`, `.dockerignore`
- `docker/entrypoint-pipeline.sh`
- `docker/janus/Dockerfile` + `docker/janus/conf/*.jcfg` (3 files) — Janus = +4 bonus
- `pyproject.toml` (extras **trimmed** to a single `client = [aiortc]` group) + `uv.lock`
  (**relocked on uni-lab** to match — see Outcome)

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
- **NEW** `.gitignore` (minimal: `models/ results/ tmp/ __pycache__/ *.py[cod] .venv/ .DS_Store`)

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
- `send_audio.py` host deps: `httpx` is already a **base** dependency; `aiortc`
  is the only extra needed. `pyproject` extras were trimmed from `dev` +
  `finetune` (pytest/torch/transformers/ollama/peft/…) down to a single
  `client = [aiortc]` group — the dev/finetune names were the last "development
  work" leaking into the public tree. README documents `uv sync --extra client`
  for the host that injects audio. Trimming pyproject made `uv.lock` stale, so it
  was **relocked on uni-lab** (`uv lock` can't resolve vLLM on Apple Silicon).
  The trim is **submission-only** — `main` keeps the full `dev`/`finetune` extras.

## Commit mechanism — rootless orphan branch (as executed)

A staging-dir + `git init` was the original plan; once history was collapsed to
a single `main`, a **rootless orphan branch** was used instead — git-native,
in-repo, auditable via `git ls-files`, and it pushes to the new public repo's
`main` sending only that one parent-less commit's objects (no dev history).

```bash
# 1. orphan branch off the clean working tree
git checkout --orphan submission
git rm -r --cached -q .            # empty index; working tree intact on disk

# 2. write the two new files
#    - .gitignore  (minimal, overwrites the dev one in this branch only)
#    - README.md   (fresh build/run/output quickstart)

# 3. stage exactly the manifest, then drop the stale prompt
git add Dockerfile docker-compose.yml .dockerignore README.md .gitignore \
        docker/entrypoint-pipeline.sh docker/janus \
        pyproject.toml uv.lock \
        pipelines/config-base.json pipelines/config-submission.json \
        pipelines/summarizer/vllm-qwen3.5-4b-awq.json \
        plugins/nodes/proc/_audio_chunker plugins/nodes/proc/_transcriber_whisper \
        plugins/nodes/proc/_hallucination_filter plugins/nodes/proc/_novel_extractor \
        plugins/nodes/proc/_window_aggregator plugins/nodes/proc/_summarizer_vllm \
        plugins/nodes/proc/_summarizer_common plugins/nodes/sink/_result_transmitter \
        tools/fetch_models.sh tools/assemble_config.py tools/run_pipeline.sh \
        tools/lib/pipeline_prep.sh tools/send_audio.py \
        docs/APPROACH.md submission/sample_outputs
git rm --cached -q plugins/nodes/proc/_summarizer_vllm/summarize_prompt-v1.txt

# 4. single rootless commit
git commit -m "STREAMIND: real-time meeting-intelligence pipeline (challenge submission)"

# 5. restore the dev tree (untracked dev files block a plain switch; -f is safe —
#    they're byte-identical to main and get restored from main)
git checkout -f main
```

The trimmed `.gitignore` (`__pycache__/`, `models/`, etc.) must exist **before**
`git add` so it auto-excludes caches/weights from the staged dirs.

### Conduit + relock + public push (on uni-lab)

The trimmed `pyproject` needs a matching `uv.lock`, and `uv lock` can't resolve
vLLM on Apple Silicon → relock on uni-lab. The branch reaches uni-lab via the
private origin (never rsync):

```bash
# from the Mac: push the orphan branch as a conduit
git push -u origin submission

# on uni-lab:
git fetch origin
git checkout -B submission origin/submission
uv lock                              # regenerate uv.lock for the trimmed pyproject
git add uv.lock
git commit --amend --no-edit         # keep ONE rootless commit (new SHA)
uv sync --no-install-project         # sanity: lean build resolves
git remote add public <NEW_PUBLIC_REPO_URL>
git push public submission:main
```

Acceptance gate (run, all passed):

- `git ls-tree -r --name-only <commit>` == manifest exactly (51 files; no
  results/datasets/tests/eval/finetune/judge/alt-profiles/dev-docs).
- Rootless (no parent), single commit.
- All 5 `submission/sample_outputs/window_*.json` have exact
  `{from,to,summary,keywords[3],proc_time}`.
- `uv.lock` relock proof: `ollama/peft/trl/bitsandbytes/accelerate/jiwer/pytest`
  = 0; `aiortc`/`vllm` present.
- uni-lab `docker compose build` + run → valid windows.

## Division of labor

I built the orphan branch + fresh README/.gitignore + the local rootless commit,
and pushed it to private `origin` as a conduit. **User** relocked on uni-lab
(`uv lock` + `--amend`), confirmed the build, and performs the final
`git remote add public` + `git push public submission:main` (their creds, new repo).

## Risk

- **Dropping a needed file** → build/run breaks. Mitigation: manifest derived
  from the actual `COPY`/`source`/`importlib` chain; acceptance gate + the
  uni-lab clean-room build ran before the repo goes public. (Resolved — verified.)
- **Orphan-branch working-tree clutter:** main's tracked files become untracked
  on the orphan branch; `git checkout -f main` restores them safely (identical
  content). Handled.

## Outcome

- **Commit:** `36435d1` (Mac, rootless, 51 files, trimmed pyproject + stale lock)
  → relocked + `--amend` on uni-lab → **`d0c23bd`** (rootless, 51 files,
  consistent `uv.lock`). Build passed on uni-lab.
- **Verified** (`d0c23bd`): all 5 challenge deliverables present; output schema
  exact (3 keywords; windows 0→300→…→1305 s); zero excluded dev artifacts; relock
  confirmed clean.
- **Pushed to private `origin/submission`** (conduit). The two remaining steps —
  add the new public remote and `git push public submission:main` — are deferred
  to the user.
- `main` keeps the full dev/`finetune` extras; the trim lives only on the
  submission branch.

## Update (2026-06-25) — single-config submission

After `d0c23bd`, the submission was simplified so the container launches **one
pre-assembled config directly**. The multi-model assembly step (base + profile
→ `./tmp/config.json`) exists only to A/B summarizers during dev; a locked
single-model submission does not need it. Deltas since `d0c23bd`:

- **Entry point** `docker/entrypoint-pipeline.sh` rewritten self-contained: sets
  the two CUDA-startup envs (`VLLM_WORKER_MULTIPROC_METHOD=spawn` + cuBLAS
  `LD_LIBRARY_PATH`) inline, then `exec`s
  `juturna launch --config pipelines/config-submission.json --auto`. No runtime
  assembly.
- **Dropped 5 now-dev-only files:** `tools/{assemble_config.py, run_pipeline.sh,
  lib/pipeline_prep.sh}`, `pipelines/config-base.json`,
  `pipelines/summarizer/vllm-qwen3.5-4b-awq.json`. `pipelines/config-submission.json`
  is the sole config (deliverable §2).
- **Approach folded into `README.md`** (`## Approach` section); `docs/` removed
  entirely. Deliverable §5 satisfied by README.
- **README run section** reframed from "feed a local audio file" to "feed a
  remote audio source" — documents the Janus VideoRoom (room 1234) →
  `rtp_forward` (`audio_pt 97`) → `audio_rtp` `udp/8888` ingress contract.
- **Dockerfile/compose** dropped the dead `SUMMARIZER_PROFILE` / `WINDOW_SECONDS`
  env (window is fixed in the config); compose keeps `DESTINATION_ENDPOINT`.
- `pyproject.toml` / `uv.lock` unchanged from `d0c23bd` → **no relock**.

Manifest is now **45 files** (was 51). SHA chain (all rootless, amended in a Mac
worktree off `d0c23bd`): `d67caa8` (entrypoint + drop 5) → `d59727c`
(approach→README, drop `docs/`) → `0dd007e` (remote-audio README) →
**`4733d02`** (pyproject comment). Scope: **submission branch only** — `main` is
untouched and keeps `run_pipeline.sh` + `assemble_config.py` for host dev
bakeoffs.

Re-verified on `4733d02`: rootless, 45 files, all 5 deliverables, output schema
exact, zero dev leakage; the lean entrypoint started cleanly on uni-lab (image
build is unchanged by the doc edits). Remaining: `git push -f origin submission`,
then publish to the public repo.
