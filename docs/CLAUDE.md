# Docs Folder Guide

Maps `docs/` dir for agent nav. Start here for specs, plans, reference.

## Challenge

- `CHALLENGE.md`: Full challenge spec — scoring formula, endpoint contract, submission format. Source of truth.
- `APPROACH.md`: Brief markdown explaining the design — shipped with the submission per challenge requirements. Updated in M3 with final numbers.
- `images/challenge_overview.png`: Pipeline architecture diagram
- `images/challenge_scoring.png`: Scoring breakdown diagram
- `../datasets/`: Audio corpora. rev16: per-episode dirs (`audio.opus`, `transcript.txt`, `chunks.json`). IETF: flat (`*.opus`, `*.opus.txt`). Audio gitignored; text/json tracked.

## Task Tracking

- `TODO.md`: Active milestone list (M0–M4) — read before new work

## Plans

`plans/` holds active plans.

- `plans/cuda-native-pipeline-base-first.md`: **ACTIVE.** M0 cleanup + M1 base CUDA pipeline + M2/M3/M4 sketch. Native vLLM in-process; model weights ship with submission.

## Machine

- `MACHINE-SPECS.md`: `uni-lab` hardware snapshot (RTX 4090, CUDA 12.4, Debian 13)

## Framework & Tool Documentation

All ref docs for Juturna, Janus, tools in `docs/documentation/`.

See [`docs/documentation/CLAUDE.md`](documentation/CLAUDE.md) for full index.
