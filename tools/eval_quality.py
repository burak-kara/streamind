#!/usr/bin/env python3
"""Local quality evaluation harness for summarizer output.

Reads result windows + their debug transcripts, asks a local Ollama LLM to
score on the challenge's 5 Likert criteria, scores keyword relevance, and
emits a per-window table with B_i / K_i / L_i and the chunk score C_i.

This harness is directional only. The official hidden judge may differ.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable

import ollama


LIKERT_CRITERIA = (
    "factual_consistency",
    "relevance",
    "coherence",
    "fluency",
    "conciseness",
)

JUDGE_PROMPT = """You are a strict evaluator for meeting summaries. Given the TRANSCRIPT and the CANDIDATE JSON (summary + 3 keywords), output ONLY a raw JSON object (no markdown) with these fields:

- "factual_consistency": integer 1-5 — does the summary state only facts present in the transcript?
- "relevance": integer 1-5 — does the summary capture the main point of the transcript?
- "coherence": integer 1-5 — is the summary logically ordered and readable?
- "fluency": integer 1-5 — grammatical, natural English?
- "conciseness": integer 1-5 — short and specific, no padding?
- "keyword_relevance": array of exactly 3 booleans — is each keyword a specific, relevant term from the transcript (not a generic word like "meeting", "discussion", "topic")?

TRANSCRIPT:
{transcript}

CANDIDATE:
{candidate}
"""


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


def _strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```[a-z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    return text.strip()


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
        # Skip debug artifacts like `debug/window_N_llm_raw.json`.
        if "debug" in p.parts:
            continue
        if p.name.startswith("window_") and not p.stem[len("window_"):].isdigit():
            # Only accept `window_<int>.json`, not `window_N_llm_raw.json` etc.
            continue
        yield p


def _score_one(client: ollama.Client, model: str, window: Path) -> Scored | None:
    try:
        result = json.loads(window.read_text())
    except Exception as e:
        print(f"skip {window}: {e}", file=sys.stderr)
        return None
    transcript = _read_transcript(window)
    candidate = json.dumps(
        {"summary": result.get("summary", ""), "keywords": result.get("keywords", [])},
        ensure_ascii=False,
    )
    prompt = JUDGE_PROMPT.format(transcript=transcript, candidate=candidate)
    response = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"num_predict": 256, "num_ctx": 4096, "temperature": 0.0},
        think=False,
    )
    content = _strip_fences(response.message.content)
    try:
        judged = json.loads(content)
    except json.JSONDecodeError:
        print(f"skip {window}: judge returned non-JSON: {content[:120]!r}", file=sys.stderr)
        return None

    b_breakdown = {c: int(judged.get(c, 0)) for c in LIKERT_CRITERIA}
    b = sum(b_breakdown.values())
    flags_raw = judged.get("keyword_relevance") or []
    keyword_flags = [bool(f) for f in flags_raw][:3]
    while len(keyword_flags) < 3:
        keyword_flags.append(False)
    k = sum(2 if f else -2 for f in keyword_flags)

    proc_time = float(result.get("proc_time", 0.0))
    if b >= 10:
        l_val = 10.0 * math.exp(-0.5 * proc_time)
    else:
        l_val = 0.0
    c = b + k + l_val
    return Scored(
        path=window,
        proc_time=proc_time,
        b_breakdown=b_breakdown,
        b=b,
        keyword_flags=keyword_flags,
        k=k,
        l=l_val,
        c=c,
        summary=result.get("summary", ""),
        keywords=result.get("keywords", []),
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
