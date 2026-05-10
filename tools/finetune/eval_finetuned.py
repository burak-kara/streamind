#!/usr/bin/env python3
"""Evaluate a fine-tuned summarizer against the base model.

For each example in the held-out test split:
  1. Re-prompt both models through Ollama with the production summarize prompt.
  2. Score each candidate with the judge (B/K/L/C using the shared scorer).
  3. Print a per-example table + averages and a head-to-head delta.

The test split is the JSONL produced by prepare_rev16.py — its `_meta` block
already contains the teacher-vs-judge B/K from data prep, but we re-judge here
so we're measuring the *deployed* candidate, not the teacher's.

Usage:
    uv run python tools/finetune/eval_finetuned.py \
        --base-tag qwen3.5:4b \
        --finetuned-tag streamind-summarizer:rev16-r16 \
        --judge-model qwen3.5:9b-16k \
        --test-file tools/finetune/data/test.jsonl
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import ollama


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _load_scorer():
    spec = importlib.util.spec_from_file_location(
        "judge_scorer",
        REPO_ROOT / "plugins" / "nodes" / "sink" / "_judge_common" / "scorer.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["judge_scorer"] = mod
    spec.loader.exec_module(mod)
    return mod


_scorer = _load_scorer()


# Shared keyword post-processor — same one summarizer_llm runs at inference.
def _load_keywords():
    spec = importlib.util.spec_from_file_location(
        "summarizer_keywords",
        REPO_ROOT / "plugins" / "nodes" / "proc" / "_summarizer_common" / "keywords.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["summarizer_keywords"] = mod
    spec.loader.exec_module(mod)
    return mod


_kw = _load_keywords()


@dataclass
class Result:
    model_tag: str
    summary: str
    keywords: list[str]
    b: int
    k: int
    l: float
    c: float


def _parse_summary_response(content: str) -> tuple[str, list[str]]:
    content = content.strip()
    content = re.sub(r"<\|[^|]+\|>", "", content)
    content = re.sub(r"^```[a-z]*\n?", "", content)
    content = re.sub(r"\n?```$", "", content)
    obj = re.search(r"\{.*\}", content, re.DOTALL)
    if obj is None:
        raise ValueError(f"no JSON object: {content[:120]!r}")
    parsed = json.loads(obj.group(0))
    return str(parsed.get("summary", "")).strip(), [str(k).strip() for k in parsed.get("keywords", []) if str(k).strip()]


def _summarize(client: ollama.Client, tag: str, prompt: str, transcript: str,
               num_ctx: int, num_predict: int) -> tuple[str, list[str]]:
    """Mirror summarizer_llm.update() — same prompt, same JSON tolerance,
    same keyword post-processor — so the eval reflects deployed behaviour."""
    response = client.chat(
        model=tag,
        messages=[{"role": "user", "content": prompt}],
        options={"num_predict": num_predict, "num_ctx": num_ctx, "temperature": 0.2},
        think=False,
    )
    try:
        summary, keywords = _parse_summary_response(response.message.content)
    except (ValueError, json.JSONDecodeError) as e:
        logging.getLogger("eval").warning("[%s] parse failed: %s — empty fallback", tag, e)
        summary, keywords = "", []
    keywords = _kw.ensure_three_keywords(keywords, transcript)
    return summary, keywords


def _judge(client: ollama.Client, tag: str, transcript: str,
           summary: str, keywords: list[str], proc_time: float,
           num_ctx: int, num_predict: int):
    prompt = _scorer.build_prompt(transcript=transcript, summary=summary, keywords=keywords)
    response = client.chat(
        model=tag,
        messages=[{"role": "user", "content": prompt}],
        options={"num_predict": num_predict, "num_ctx": num_ctx, "temperature": 0.0},
        think=False,
    )
    b_breakdown, keyword_flags = _scorer.parse_judge_response(response.message.content)
    return _scorer.compute_score(b_breakdown, keyword_flags, proc_time)


def _extract_transcript(messages: list[dict]) -> str:
    """The user message contains the prompt with the transcript inlined.
    We need the raw transcript back for judging, so re-derive it by stripping
    the prompt template prefix.
    """
    user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
    marker = "Transcript:\n"
    idx = user_msg.find(marker)
    if idx == -1:
        return user_msg
    return user_msg[idx + len(marker):].strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test-file", type=Path, default=REPO_ROOT / "tools/finetune/data/test.jsonl")
    parser.add_argument("--base-tag", required=True, help="Ollama tag for the base summarizer.")
    parser.add_argument("--finetuned-tag", required=True, help="Ollama tag for the fine-tuned summarizer.")
    parser.add_argument("--judge-model", default="qwen3.5:9b-16k")
    parser.add_argument("--ollama-endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--num-predict", type=int, default=160)
    parser.add_argument("--judge-num-predict", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None, help="Cap examples (for fast iteration).")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("eval")

    if not args.test_file.exists():
        logger.error("test file missing: %s — run prepare_rev16.py first", args.test_file)
        return 1

    rows: list[dict] = []
    with args.test_file.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        logger.error("no examples in %s", args.test_file)
        return 1

    client = ollama.Client(host=args.ollama_endpoint)

    pairs: list[tuple[Result, Result]] = []
    for i, row in enumerate(rows):
        transcript = _extract_transcript(row["messages"])
        prompt = next(m["content"] for m in row["messages"] if m["role"] == "user")

        results: list[Result] = []
        for tag in (args.base_tag, args.finetuned_tag):
            try:
                summary, keywords = _summarize(client, tag, prompt, transcript,
                                               args.num_ctx, args.num_predict)
            except Exception as e:
                logger.error("[%s] summary call failed on row %d: %s", tag, i, e)
                results.append(Result(tag, "", [], 0, 0, 0.0, 0.0))
                continue
            try:
                # proc_time is irrelevant for offline eval — the latency reward
                # depends on real-time behaviour. We hold it constant at 0 so
                # both models get the same L bonus and B/K dominate the delta.
                score = _judge(client, args.judge_model, transcript, summary, keywords,
                               proc_time=0.0,
                               num_ctx=args.num_ctx, num_predict=args.judge_num_predict)
            except Exception as e:
                logger.error("[judge %s] failed on row %d/%s: %s", args.judge_model, i, tag, e)
                results.append(Result(tag, summary, keywords, 0, 0, 0.0, 0.0))
                continue
            results.append(Result(tag, summary, keywords, score.b, score.k, score.l, score.c))
        pairs.append((results[0], results[1]))
        logger.info(
            "ex %2d: base B=%2d K=%+d C=%5.2f | ft B=%2d K=%+d C=%5.2f | ΔC=%+5.2f",
            i,
            pairs[-1][0].b, pairs[-1][0].k, pairs[-1][0].c,
            pairs[-1][1].b, pairs[-1][1].k, pairs[-1][1].c,
            pairs[-1][1].c - pairs[-1][0].c,
        )

    base_avgs = {
        "B": mean(r[0].b for r in pairs),
        "K": mean(r[0].k for r in pairs),
        "C": mean(r[0].c for r in pairs),
    }
    ft_avgs = {
        "B": mean(r[1].b for r in pairs),
        "K": mean(r[1].k for r in pairs),
        "C": mean(r[1].c for r in pairs),
    }
    print()
    print(f"{'metric':>10}  {'base':>8}  {'finetuned':>10}  {'delta':>8}")
    print("-" * 44)
    for key in ("B", "K", "C"):
        print(f"{key:>10}  {base_avgs[key]:>8.2f}  {ft_avgs[key]:>10.2f}  {ft_avgs[key] - base_avgs[key]:>+8.2f}")

    wins = sum(1 for b, f in pairs if f.c > b.c)
    print(f"\nFT wins: {wins}/{len(pairs)} (ties broken to base)")

    if args.json_out:
        args.json_out.write_text(json.dumps({
            "test_file": str(args.test_file),
            "base_tag": args.base_tag,
            "finetuned_tag": args.finetuned_tag,
            "judge_model": args.judge_model,
            "averages": {"base": base_avgs, "finetuned": ft_avgs},
            "wins_for_finetuned": wins,
            "n": len(pairs),
            "pairs": [
                {
                    "base": {"summary": b.summary, "keywords": b.keywords, "B": b.b, "K": b.k, "L": b.l, "C": b.c},
                    "finetuned": {"summary": f.summary, "keywords": f.keywords, "B": f.b, "K": f.k, "L": f.l, "C": f.c},
                }
                for b, f in pairs
            ],
        }, indent=2))
        logger.info("wrote %s", args.json_out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
