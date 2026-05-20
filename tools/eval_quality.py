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
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable, Optional

# FlashInfer sampler JIT-compiles kernels via nvcc at first use; uni-lab has
# CUDA runtime but no toolkit. Same workaround `run_pipeline.sh` uses for the
# summarizer. Must be set BEFORE the `vllm` import inside _build_llm.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")


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
    asr_wer: Optional[float] = None
    asr_text: str = ""
    reference_slice: str = ""


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


def _audio_duration(path: Path) -> float:
    out = subprocess.check_output([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ])
    return float(out.strip())


def _slice_proportional(text: str, start_sec: float, end_sec: float, total_sec: float) -> str:
    """Char-proportional slice. Approximates time-aligned segment of a flat
    transcript (no timestamps). Boundary fuzz ±10% — speech rate non-uniform.
    """
    if total_sec <= 0 or not text:
        return ""
    L = len(text)
    start = max(0, min(L, int(start_sec / total_sec * L)))
    end = max(0, min(L, int(end_sec / total_sec * L)))
    if end <= start:
        return ""
    return text[start:end]


def _compute_wer(reference: str, hypothesis: str) -> Optional[float]:
    if not reference.strip() or not hypothesis.strip():
        return None
    try:
        import jiwer
    except ImportError:
        print("jiwer not installed; skip WER. Add to dev extra.", file=sys.stderr)
        return None
    return float(jiwer.wer(reference, hypothesis))


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


def _score_one(
    llm,
    sampling,
    window: Path,
    reference_text: str = "",
    total_duration: float = 0.0,
) -> Scored | None:
    try:
        result = json.loads(window.read_text())
    except Exception as e:
        print(f"skip {window}: {e}", file=sys.stderr)
        return None
    transcript = _read_transcript(window)
    summary = result.get("summary", "")
    keywords = result.get("keywords", [])
    # Per Plan 2: judge sees ASR transcript (status quo). Ref slice + WER are
    # independent ASR-fidelity signal that does NOT feed B/K/L.
    prompt = _scorer.build_prompt(transcript=transcript, summary=summary, keywords=keywords)
    raw = _generate(llm, sampling, prompt)
    try:
        b_breakdown, keyword_flags = _scorer.parse_judge_response(raw)
    except (ValueError, json.JSONDecodeError) as e:
        print(f"skip {window}: judge parse failed: {e}", file=sys.stderr)
        print(f"  raw judge output (first 500 chars): {raw[:500]!r}", file=sys.stderr)
        return None

    proc_time = float(result.get("proc_time", 0.0))
    score = _scorer.compute_score(b_breakdown, keyword_flags, proc_time)

    asr_wer: Optional[float] = None
    reference_slice = ""
    if reference_text and total_duration > 0:
        start_sec = float(result.get("from", 0.0))
        end_sec = float(result.get("to", 0.0))
        reference_slice = _slice_proportional(reference_text, start_sec, end_sec, total_duration)
        asr_wer = _compute_wer(reference_slice, transcript)

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
        asr_wer=asr_wer,
        asr_text=transcript,
        reference_slice=reference_slice,
    )


def _write_per_window(scored: list[Scored], judge_model: str) -> None:
    """Mirror the in-pipeline judge's output layout: results/.../judge/window_N.json."""
    for s in scored:
        window_data = json.loads(s.path.read_text())
        judge_dir = s.path.parent / "judge"
        judge_dir.mkdir(parents=True, exist_ok=True)
        out_path = judge_dir / s.path.name
        payload = {
            **window_data,
            "B_breakdown": s.b_breakdown,
            "B": s.b,
            "keyword_flags": s.keyword_flags,
            "K": s.k,
            "L": s.l,
            "C": s.c,
            "judge_model": judge_model,
        }
        if s.asr_wer is not None:
            payload["asr_wer"] = s.asr_wer
            payload["asr_transcript"] = s.asr_text
            payload["reference_slice"] = s.reference_slice
        out_path.write_text(json.dumps(payload, indent=2))


def _print_table(scored: list[Scored]) -> None:
    has_wer = any(s.asr_wer is not None for s in scored)
    wer_hdr = f" {'WER':>5}" if has_wer else ""
    header = f"{'window':60} {'B':>3} {'K':>3} {'L':>5} {'C':>6} {'proc':>6}{wer_hdr}"
    print(header)
    print("-" * len(header))
    for s in scored:
        rel = str(s.path.relative_to(Path.cwd())) if Path.cwd() in s.path.parents else str(s.path)
        rel = rel[-60:].rjust(60)
        line = f"{rel} {s.b:>3} {s.k:>3} {s.l:>5.2f} {s.c:>6.2f} {s.proc_time:>6.2f}"
        if has_wer:
            line += f" {s.asr_wer:>5.2f}" if s.asr_wer is not None else f" {'-':>5}"
        print(line)
    if scored:
        print("-" * len(header))
        avg_line = (
            f"{'avg':60} "
            f"{mean(s.b for s in scored):>3.1f} "
            f"{mean(s.k for s in scored):>3.1f} "
            f"{mean(s.l for s in scored):>5.2f} "
            f"{mean(s.c for s in scored):>6.2f} "
            f"{mean(s.proc_time for s in scored):>6.2f}"
        )
        if has_wer:
            wer_vals = [s.asr_wer for s in scored if s.asr_wer is not None]
            avg_line += f" {mean(wer_vals):>5.2f}" if wer_vals else f" {'-':>5}"
        print(avg_line)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path, nargs="?", default=Path("results"),
                        help="Directory containing window_*.json files (recursive). Default: ./results")
    parser.add_argument("--judge-profile", default=None,
                        help="Profile name under pipelines/judge/<name>.json. Required.")
    parser.add_argument("--filter", default=None, help="substring match on window path")
    parser.add_argument("--audio", type=Path, default=None,
                        help="Source audio file used in the run. Enables ASR WER scoring. "
                             "Reference text auto-resolved to sibling <stem>.txt unless "
                             "--reference is given.")
    parser.add_argument("--reference", type=Path, default=None,
                        help="Ground-truth transcript (.txt). Overrides auto-resolution from --audio.")
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

    reference_text = ""
    total_duration = 0.0
    if args.audio is not None:
        if not args.audio.exists():
            print(f"audio not found: {args.audio}", file=sys.stderr)
            return 1
        try:
            total_duration = _audio_duration(args.audio)
        except Exception as e:
            print(f"ffprobe failed on {args.audio}: {e}", file=sys.stderr)
            return 1
        ref_path = args.reference or args.audio.with_suffix(".txt")
        if ref_path.exists():
            reference_text = ref_path.read_text()
            print(f"WER enabled: ref={ref_path} duration={total_duration:.1f}s", file=sys.stderr)
        else:
            print(f"reference not found: {ref_path}; WER disabled", file=sys.stderr)
    elif args.reference is not None:
        print("--reference requires --audio (need duration for proportional slice)", file=sys.stderr)
        return 1

    llm, sampling = _build_llm(cfg)

    scored: list[Scored] = []
    for w in windows:
        s = _score_one(llm, sampling, w, reference_text=reference_text, total_duration=total_duration)
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
                    "asr_wer": s.asr_wer,
                }
                for s in scored
            ],
            indent=2,
        ))

    return 0


if __name__ == "__main__":
    sys.exit(main())
