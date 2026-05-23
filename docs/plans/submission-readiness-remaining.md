# STREAMIND — Submission Readiness: Remaining Items

> **SUPERSEDED — 2026-05-19.** Replaced by [cuda-native-pipeline-base-first.md](cuda-native-pipeline-base-first.md). Item list predates the vLLM pivot — endpoint + Docker items will be refreshed in M3.

Snapshot as of 2026-04-23. All prior plan work
(`docs/plans/given-the-changes-in-streamed-biscuit.md` and
`~/.claude/plans/given-the-missing-poitns-curried-bumblebee.md`) is complete
except the items below. 46 tests pass. Post-Phase-1 smoke measured B=25,
K=6, L=2.13, C=33.13, proc_time=3.10s (MLX 4B, 30s window, judge=qwen3.5:9b
local).

## Hard blockers

1. **`destination_endpoint` unset.** `pipelines/config-base.json` still has
   `"destination_endpoint": ""`. Blocked on challenge POST URL.
2. **CUDA build unverified.** `Dockerfile` never built end-to-end. Risks
   that only surface at build time:
   - `nvidia/cuda:12.3.0-runtime-ubuntu22.04` pull (amd64-only; Rosetta on
     Apple Silicon slows build further)
   - `deadsnakes` PPA install for Python 3.12
   - `uv sync --extra dev --no-install-project` — `juturna` must be
     installable from wherever it lives
   - Ollama install script on minimal Ubuntu (no systemd)
   - Build-time `ollama pull qwen3.5:4b` requires the daemon reachable
     mid-build (current Dockerfile launches `ollama serve &` then polls
     `/api/tags`)
3. **End-to-end container smoke unrun.** `docker compose up` + Janus →
   pipeline-container UDP/8888 → `send_audio.py` never exercised. GPU
   passthrough in `docker-compose.yml` (`deploy.resources.reservations.
   devices`) is unvalidated.

## Measurement gaps

4. **CUDA `proc_time` unknown.** Measured 3.10s is MLX on Apple Silicon.
   RTX Pro 4500 `proc_time` will differ; `L_i` target is 2.2–6.1 at 1–3 s.
5. **No held-out-style validation.** `tools/eval_quality.py` only run on a
   120 s slice of `tests/fixtures/youtube_15min.wav`. `datasets/rev16/`
   (ground-truth `.txt` transcripts, 30 podcast episodes) and
   `datasets/ietf/` (10 conference sessions) untouched.
6. **Judge calibration unknown.** Local `qwen3.5:9b` judge returned
   ceiling-25/25 on every window. Real hidden judge almost certainly
   scores lower. No way to calibrate without a scoring sample from the
   organizers.

## Config risks

7. **`encoding_clock_chan: "opus/48000/1"` — mono-correctness unverified.**
   Fixed for CHALLENGE.md §31 but the smoke test used a forced-mono
   120 s clip (`tmp/smoke_120s_mono.wav`). aiortc negotiates channel count
   from the WAV file's encoding. If the challenge rig WAV is stereo but
   declared mono in SDP, behavior is unknown. Verify by sending both a
   mono and a native-stereo WAV through `send_audio.py` once the rig
   arrives.
8. **`payload_type: 97` hardcoded** in `pipelines/config-base.json:22`
   and `tools/send_audio.py:158`. CHALLENGE.md §98-100 shows the example
   uses `"payload_type": 0` with `PCMU/8000` — different codec. The
   actual PT the challenge rig sends for Opus is unconfirmed. Mismatch
   means ffmpeg SDP decode fails silently. **Action**: ask organizers
   what PT their Janus forwards use.
9. **POST path untested.** `result_transmitter` writes local files fine
   (verified via smoke). The HTTP POST branch (triggered only when
   `destination_endpoint` is non-empty) has no unit or integration test.
   Before submission, set endpoint to a local `python -m http.server` or
   httpbin URL once, run one window, confirm the POST body matches the
   challenge contract.

## Polish / code health

10. **`@pytest.mark.integration` not registered** in `pyproject.toml`.
    Raises `PytestUnknownMarkWarning` every run. Fix:
    ```toml
    [tool.pytest.ini_options]
    markers = ["integration: requires Janus + full network stack"]
    ```
11. **`BANNED_KEYWORDS` duplicated** in
    `plugins/nodes/proc/_summarizer_common/keywords.py` (authoritative),
    `tools/eval_quality.py`, and `.claude/skills/benchmark-pipeline/
    SKILL.md`. Duplication is intentional (tooling + node loaders are
    separate modules with no shared import path). Revisit if a fourth
    consumer appears.
12. **Uncommitted working tree.** Pre-existing modifications plus this
    session's refactor: `Dockerfile`, `docker/entrypoint-pipeline.sh`,
    `docker-compose.yml`, `docs/APPROACH.md`, new tests, shared
    `keywords.py`, SKILL.md, eval_quality.py, Phase-1 fixes. Nothing
    committed yet — no checkpoint on the green state.
13. **No submission archive.** CHALLENGE.md §171-179 asks for: custom
    components, pipeline config, progressive summary objects, Dockerfile,
    approach MD. No bundling script; no decision on what format the
    organizers want (zip, tar, git tag, PR link).

## Documentation gaps

14. **APPROACH.md measured numbers are MLX, not CUDA.** Append a second
    row once the RTX Pro 4500 build runs.
15. **No live submission checklist.** Plan `Phase 4.4` items (endpoint set,
    encoding verified, tests green, build green, archive built) are not
    tracked as checkable items in any doc.

## Priority to close

1. **Verify on CUDA hardware.** Single most blocker-closing action.
   Push image to a cloud RTX node, run `docker compose up`, send audio,
   inspect results. Closes #2, #3, #4, #7 and replaces MLX numbers in
   APPROACH.md (#14).
2. **Contact organizers** for the submission `destination_endpoint` URL
   and the RTP `payload_type` they forward. Closes #1, #8.
3. **rev16 run** (one ~15-min episode, MLX locally, or in the CUDA
   container) with the ground-truth transcript loaded into
   `eval_quality.py` as a sanity check on `B_i`/`K_i`. Partial close of
   #5.
4. **Git commit + tag** `submission-v0` once 1–3 are green. Add a tiny
   `tools/make_submission.sh` that `git archive`s the tagged ref into a
   zip and copies `results/` + `Dockerfile` + `docs/APPROACH.md`.
   Closes #12, #13.
5. **Register `integration` marker**, then enable `pytest -m integration`
   in a local shell against a running Janus. Closes #10.

## Out of scope / accepted

- #6 (judge calibration) — no viable action without organizer data.
- #11 (BANNED_KEYWORDS duplication) — intentional, revisit on 4th
  consumer.
- #15 (live checklist) — this document acts as the checklist until
  submission day.
