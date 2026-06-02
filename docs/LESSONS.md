# Lessons

Patterns captured after user corrections. Review at session start.

## Deployment to uni-lab — git flow, never rsync

**Correction (2026-06-02):** I synced changes to uni-lab with `rsync` instead
of pushing and pulling. Reason at the time: uni-lab's `origin` was HTTPS with
no creds and the SSH deploy key was denied, so `git fetch`/`checkout` failed.
That left uni-lab's tree as uncommitted diffs on a foreign branch — messy.

**Rule:** Always **edit locally → commit → push → pull on uni-lab**. Do not
rsync the working tree. If commits need reshaping, amend or merge them locally
before pushing. uni-lab git auth works for the user; if a fetch fails on my
side, surface it and let the user pull rather than falling back to rsync.

## uni-lab execution — user drives

**Correction (2026-06-02):** I ran `docker compose build`, `docker run`, and
other commands directly on uni-lab. The user runs commands on uni-lab
themselves.

**Rule:** Do **not** execute commands on uni-lab unless the user explicitly
asks. Prepare the exact command sequence and hand it over; the user runs it and
reports results. Read-only probes are still fine only when needed to plan, but
default to handing off execution.

## Locked summarizer HF id

**Correction (2026-06-02):** Docs (CLAUDE.md) labeled the locked summarizer
`Qwen/Qwen3.5-4B-AWQ`, which does not exist on HuggingFace — `docker build`
failed with "Model not found". The real repo is
`cyankiwi/Qwen3.5-4B-AWQ-BF16-INT4` (AWQ-int4 of Qwen3.5-4B, Apache-2.0),
confirmed via the original `fetch_models.sh` invocation in shell history.

**Rule:** Verify an HF id against the actual fetch command / a live repo before
trusting a prose label in docs. The authoritative source for a baked model is
the command that produced the local dir, not the summary table.
