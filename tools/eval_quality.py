#!/usr/bin/env python3
"""Local quality evaluation harness for summarizer output.

Reads result windows + their debug transcripts, asks a local Ollama LLM to
score on the challenge's 5 Likert criteria, scores keyword relevance, and
emits a per-window table with B_i / K_i / L_i and the chunk score C_i.

This harness is directional only. The official hidden judge may differ.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable

import ollama


# Load the shared scorer module by path. Mirrors the loader pattern used by
# the in-pipeline judge node so the script and node can never drift.
# The sys.modules registration is required so @dataclass inside the module
# can resolve its own __module__ in Python 3.12+.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_scorer_spec = importlib.util.spec_from_file_location(
    "judge_scorer",
    _REPO_ROOT / "plugins" / "nodes" / "sink" / "_judge_common" / "scorer.py",
)
_scorer = importlib.util.module_from_spec(_scorer_spec)
assert _scorer_spec.loader is not None
sys.modules["judge_scorer"] = _scorer
_scorer_spec.loader.exec_module(_scorer)


@dataclass
class Scored:
    path: Path
    proc_time: float
    b_breakdown: dict[str, int]
    b: int
    keyword_flags: list[bool]
    k: int
    l: float
    c: float
    summary: str
    keywords: list[str]
    transcript_len: int


def _read_transcript(window_path: Path) -> str:
    stem = window_path.stem  # window_N
    candidates = [
        window_path.parent / "debug" / f"{stem}_transcript.txt",
        window_path.parent.parent / "debug" / f"{stem}_transcript.txt",
    ]
    for c in candidates:
        if c.exists():
            return c.read_text()
    return ""


def _iter_window_files(root: Path) -> Iterable[Path]:
    for p in sorted(root.rglob("window_*.json")):
        # Skip debug artifacts and the judge node's own output dir.
        if "debug" in p.parts or "judge" in p.parts:
            continue
        if p.name.startswith("window_") and not p.stem[len("window_"):].isdigit():
            continue
        yield p


def _score_one(client: ollama.Client, model: str, window: Path) -> Scored | None:
    try:
        result = json.loads(window.read_text())
    except Exception as e:
        print(f"skip {window}: {e}", file=sys.stderr)
        return None
    transcript = _read_transcript(window)
    summary = result.get("summary", "")
    keywords = result.get("keywords", [])
    prompt = _scorer.build_prompt(transcript=transcript, summary=summary, keywords=keywords)
    response = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"num_predict": 256, "num_ctx": 4096, "temperature": 0.0},
        think=False,
    )
    try:
        b_breakdown, keyword_flags = _scorer.parse_judge_response(response.message.content)
    except (ValueError, json.JSONDecodeError) as e:
        print(f"skip {window}: judge parse failed: {e}", file=sys.stderr)
        return None

    proc_time = float(result.get("proc_time", 0.0))
    score = _scorer.compute_score(b_breakdown, keyword_flags, proc_time)
    return Scored(
        path=window,
        proc_time=proc_time,
        b_breakdown=score.b_breakdown,
        b=score.b,
        keyword_flags=score.keyword_flags,
        k=score.k,
        l=score.l,
        c=score.c,
        summary=summary,
        keywords=keywords,
        transcript_len=len(transcript),
    )


def _print_table(scored: list[Scored]) -> None:
    header = f"{'window':60} {'B':>3} {'K':>3} {'L':>5} {'C':>6} {'proc':>6}"
    print(header)
    print("-" * len(header))
    for s in scored:
        rel = str(s.path.relative_to(Path.cwd())) if Path.cwd() in s.path.parents else str(s.path)
        rel = rel[-60:].rjust(60)
        print(f"{rel} {s.b:>3} {s.k:>3} {s.l:>5.2f} {s.c:>6.2f} {s.proc_time:>6.2f}")
    if scored:
        print("-" * len(header))
        print(
            f"{'avg':60} "
            f"{mean(s.b for s in scored):>3.1f} "
            f"{mean(s.k for s in scored):>3.1f} "
            f"{mean(s.l for s in scored):>5.2f} "
            f"{mean(s.c for s in scored):>6.2f} "
            f"{mean(s.proc_time for s in scored):>6.2f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--judge-model", default="qwen3.5:4b")
    parser.add_argument("--judge-endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--filter", default=None, help="substring match on window path")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    if not args.results_dir.exists():
        print(f"results dir not found: {args.results_dir}", file=sys.stderr)
        return 1

    client = ollama.Client(host=args.judge_endpoint)
    windows = list(_iter_window_files(args.results_dir))
    if args.filter:
        windows = [w for w in windows if args.filter in str(w)]
    if not windows:
        print("no window files matched", file=sys.stderr)
        return 1

    scored: list[Scored] = []
    for w in windows:
        s = _score_one(client, args.judge_model, w)
        if s is not None:
            scored.append(s)

    _print_table(scored)

    if args.json_out:
        args.json_out.write_text(json.dumps(
            [
                {
                    "path": str(s.path),
                    "proc_time": s.proc_time,
                    "B_breakdown": s.b_breakdown,
                    "B": s.b,
                    "keyword_flags": s.keyword_flags,
                    "K": s.k,
                    "L": s.l,
                    "C": s.c,
                    "summary": s.summary,
                    "keywords": s.keywords,
                }
                for s in scored
            ],
            indent=2,
        ))

    return 0


if __name__ == "__main__":
    sys.exit(main())
