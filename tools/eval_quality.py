#!/usr/bin/env python3
"""Offline LLM-as-judge harness for summarizer output.

Loads a judge model with vLLM in-process, walks `results/<model>/<window>/`,
scores every window on the challenge's 5 Likert criteria, derives K and L
per the scoring formula, and writes one judge JSON per window under a
`judge/` subdirectory next to the source results.

The judge runs **sequentially after the summarizer pipeline exits** — at
submission time only the summarizer occupies VRAM; the judge is an eval
tool, not part of the live path.

Directional only. The official hidden judge may differ.
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
    stem = window_path.stem
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
        if "debug" in p.parts or "judge" in p.parts:
            continue
        if p.name.startswith("window_") and not p.stem[len("window_"):].isdigit():
            continue
        yield p


def _load_judge_profile(profile_name: str) -> dict:
    profile_path = _REPO_ROOT / "pipelines" / "judge" / f"{profile_name}.json"
    if not profile_path.exists():
        raise FileNotFoundError(
            f"Judge profile not found: {profile_path}. Expected file under "
            f"pipelines/judge/ with the suffix `.json`."
        )
    data = json.loads(profile_path.read_text())
    cfg = data.get("configuration", data)
    return cfg


def _build_llm(cfg: dict):
    from vllm import LLM, SamplingParams

    model_path = Path(cfg["model_name"])
    if not model_path.exists() or not (model_path / "config.json").exists():
        raise FileNotFoundError(
            f"Judge model directory not found: {model_path}. "
            f"Run `./tools/fetch_models.sh <hf_id> {model_path.name}` first."
        )

    llm = LLM(
        model=str(model_path),
        dtype=cfg.get("dtype", "float16"),
        gpu_memory_utilization=cfg.get("gpu_memory_utilization", 0.85),
        max_model_len=cfg.get("max_model_len", 4096),
        enforce_eager=cfg.get("enforce_eager", False),
    )
    sampling = SamplingParams(
        temperature=cfg.get("temperature", 0.0),
        top_p=cfg.get("top_p", 1.0),
        max_tokens=cfg.get("max_tokens", 256),
    )
    return llm, sampling


def _generate(llm, sampling, prompt: str) -> str:
    out = llm.chat(
        [{"role": "user", "content": prompt}],
        sampling_params=sampling,
        use_tqdm=False,
    )
    return out[0].outputs[0].text


def _score_one(llm, sampling, window: Path) -> Scored | None:
    try:
        result = json.loads(window.read_text())
    except Exception as e:
        print(f"skip {window}: {e}", file=sys.stderr)
        return None
    transcript = _read_transcript(window)
    summary = result.get("summary", "")
    keywords = result.get("keywords", [])
    prompt = _scorer.build_prompt(transcript=transcript, summary=summary, keywords=keywords)
    try:
        raw = _generate(llm, sampling, prompt)
        b_breakdown, keyword_flags = _scorer.parse_judge_response(raw)
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


def _write_per_window(scored: list[Scored], judge_model: str) -> None:
    """Mirror the in-pipeline judge's output layout: results/.../judge/window_N.json."""
    for s in scored:
        window_data = json.loads(s.path.read_text())
        judge_dir = s.path.parent / "judge"
        judge_dir.mkdir(parents=True, exist_ok=True)
        out_path = judge_dir / s.path.name
        out_path.write_text(json.dumps(
            {
                **window_data,
                "B_breakdown": s.b_breakdown,
                "B": s.b,
                "keyword_flags": s.keyword_flags,
                "K": s.k,
                "L": s.l,
                "C": s.c,
                "judge_model": judge_model,
            },
            indent=2,
        ))


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
    parser.add_argument("results_dir", type=Path, nargs="?", default=Path("results"),
                        help="Directory containing window_*.json files (recursive). Default: ./results")
    parser.add_argument("--judge-profile", default=None,
                        help="Profile name under pipelines/judge/<name>.json. Required.")
    parser.add_argument("--filter", default=None, help="substring match on window path")
    parser.add_argument("--json-out", type=Path, default=None,
                        help="Optional summary JSON; per-window judge files always written.")
    args = parser.parse_args()

    if not args.results_dir.exists():
        print(f"results dir not found: {args.results_dir}", file=sys.stderr)
        return 1
    if args.judge_profile is None:
        print("--judge-profile is required (e.g. --judge-profile vllm-qwen-7b)", file=sys.stderr)
        return 1

    cfg = _load_judge_profile(args.judge_profile)
    judge_model_id = cfg.get("model_name", args.judge_profile)

    windows = list(_iter_window_files(args.results_dir))
    if args.filter:
        windows = [w for w in windows if args.filter in str(w)]
    if not windows:
        print("no window files matched", file=sys.stderr)
        return 1

    llm, sampling = _build_llm(cfg)

    scored: list[Scored] = []
    for w in windows:
        s = _score_one(llm, sampling, w)
        if s is not None:
            scored.append(s)

    _write_per_window(scored, judge_model=judge_model_id)
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
