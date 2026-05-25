#!/usr/bin/env python3
"""Distill a fine-tuning dataset from the rev16 podcast corpus.

For each episode directory under `datasets/rev16/<episode>/transcript.txt`, we:
  1. Slice the *gold* reference transcript into windows by word count
     (proxy for the 300s production window — avoids ASR noise in the labels).
  2. Ask a strong teacher (default `qwen3.5:9b-16k` via Ollama) for a
     `{summary, keywords}` JSON using the *production* prompt template.
  3. Score the candidate with an independent judge (default same model — flag
     `--judge-model` to change) using the shared B/K/L/C scorer.
  4. Keep only examples where `B >= --min-b` and `K == 6`.
  5. Split 80/10/10 by *episode* (no transcript window leaks across splits).

Outputs JSONL chat-format files compatible with TRL `SFTTrainer`:
    {"messages": [
        {"role": "user", "content": "<prompt with transcript filled in>"},
        {"role": "assistant", "content": "{\"summary\": ..., \"keywords\": [...]}"}
    ]}

Usage:
    uv run python tools/finetune/prepare_rev16.py \
        --teacher-model qwen3.5:9b-16k \
        --judge-model qwen3.5:9b-16k \
        --min-b 18 \
        --window-words 600
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import ollama


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
REV16_DIR = REPO_ROOT / "datasets" / "rev16"
PROMPT_TEMPLATE_FILE = (
    REPO_ROOT / "plugins" / "nodes" / "proc" / "_summarizer_vllm" / "summarize_prompt.txt"
)
DEFAULT_OUT_DIR = REPO_ROOT / "tools" / "finetune" / "data"


def _load_scorer():
    """Reuse the shared judge scorer so node, eval harness, and trainer agree."""
    spec = importlib.util.spec_from_file_location(
        "judge_scorer",
        REPO_ROOT / "plugins" / "nodes" / "sink" / "_judge_common" / "scorer.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["judge_scorer"] = mod
    spec.loader.exec_module(mod)
    return mod


_scorer = _load_scorer()


@dataclass
class Example:
    episode: str
    window_idx: int
    transcript: str
    summary: str
    keywords: list[str]
    b: int
    k: int
    b_breakdown: dict[str, int]
    keyword_flags: list[bool]
    teacher_model: str
    judge_model: str


@dataclass
class EpisodeStats:
    episode: str
    windows: int = 0
    kept: int = 0
    teacher_failed: int = 0
    judge_failed: int = 0
    below_threshold: int = 0
    keyword_failed: int = 0
    examples: list[Example] = field(default_factory=list)


def _read_prompt_template() -> str:
    if not PROMPT_TEMPLATE_FILE.exists():
        raise FileNotFoundError(
            f"prompt template not found: {PROMPT_TEMPLATE_FILE}. "
            "This script must use the production prompt so the model is "
            "trained against what it will see at inference time."
        )
    return PROMPT_TEMPLATE_FILE.read_text()


def _episode_key(p: Path) -> str:
    return p.parent.name


def _split_into_windows(transcript: str, window_words: int, stride_words: int | None) -> list[str]:
    """Split a transcript into roughly `window_words`-word chunks.

    rev16 transcripts are stored as a single line of prose (no timestamps), so
    a word-count window is a good proxy for time. With 16 kHz speech at ~150
    wpm, 600 words ≈ 240s — close enough to the 300s production window for
    the model to learn the right level of compression.
    """
    if stride_words is None or stride_words >= window_words:
        stride_words = window_words
    words = transcript.split()
    if not words:
        return []
    out: list[str] = []
    i = 0
    while i < len(words):
        chunk = words[i : i + window_words]
        if len(chunk) < window_words // 2:
            break  # Trailing fragment too short to be a meaningful window.
        out.append(" ".join(chunk))
        i += stride_words
    return out


def _parse_summary_response(content: str) -> tuple[str, list[str]]:
    """Mirror summarizer_llm.py's tolerant JSON extraction."""
    content = content.strip()
    content = re.sub(r"<\|[^|]+\|>", "", content)
    content = re.sub(r"^```[a-z]*\n?", "", content)
    content = re.sub(r"\n?```$", "", content)
    obj_match = re.search(r"\{.*\}", content, re.DOTALL)
    if obj_match is None:
        raise ValueError(f"no JSON object in teacher output: {content[:120]!r}")
    parsed = json.loads(obj_match.group(0))
    summary = str(parsed.get("summary", "")).strip()
    raw_kw = parsed.get("keywords", []) or []
    keywords = [str(k).strip() for k in raw_kw if str(k).strip()]
    return summary, keywords


def _ask_teacher(
    client: ollama.Client,
    model: str,
    prompt: str,
    num_ctx: int,
    num_predict: int,
) -> tuple[str, list[str]]:
    response = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={
            "num_predict": num_predict,
            "num_ctx": num_ctx,
            "temperature": 0.2,
        },
        think=False,
    )
    return _parse_summary_response(response.message.content)


def _ask_judge(
    client: ollama.Client,
    model: str,
    transcript: str,
    summary: str,
    keywords: list[str],
    num_ctx: int,
    num_predict: int,
):
    prompt = _scorer.build_prompt(transcript=transcript, summary=summary, keywords=keywords)
    response = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"num_predict": num_predict, "num_ctx": num_ctx, "temperature": 0.0},
        think=False,
    )
    return _scorer.parse_judge_response(response.message.content)


def _process_episode(
    transcript_path: Path,
    prompt_template: str,
    args: argparse.Namespace,
    teacher_client: ollama.Client,
    judge_client: ollama.Client,
    logger: logging.Logger,
) -> EpisodeStats:
    episode = _episode_key(transcript_path)
    stats = EpisodeStats(episode=episode)
    transcript = transcript_path.read_text().strip()
    windows = _split_into_windows(transcript, args.window_words, args.stride_words)
    stats.windows = len(windows)
    logger.info("[%s] %d windows", episode, len(windows))

    for w_idx, window in enumerate(windows):
        prompt = prompt_template.format(transcript=window)
        try:
            summary, keywords = _ask_teacher(
                teacher_client, args.teacher_model, prompt,
                args.teacher_num_ctx, args.teacher_num_predict,
            )
        except Exception as e:
            stats.teacher_failed += 1
            logger.warning("[%s w%d] teacher failed: %s", episode, w_idx, e)
            continue

        if len(keywords) != 3 or not summary:
            stats.teacher_failed += 1
            logger.info(
                "[%s w%d] teacher produced bad shape (summary=%dch, keywords=%d) — drop",
                episode, w_idx, len(summary), len(keywords),
            )
            continue

        try:
            b_breakdown, keyword_flags = _ask_judge(
                judge_client, args.judge_model, window, summary, keywords,
                args.judge_num_ctx, args.judge_num_predict,
            )
        except Exception as e:
            stats.judge_failed += 1
            logger.warning("[%s w%d] judge failed: %s", episode, w_idx, e)
            continue

        b = sum(b_breakdown.values())
        k = sum(2 if f else -2 for f in keyword_flags)

        if k != 6:
            stats.keyword_failed += 1
            logger.info("[%s w%d] keywords K=%d — drop", episode, w_idx, k)
            continue

        if b < args.min_b:
            stats.below_threshold += 1
            logger.info("[%s w%d] B=%d below %d — drop", episode, w_idx, b, args.min_b)
            continue

        stats.examples.append(Example(
            episode=episode, window_idx=w_idx,
            transcript=window, summary=summary, keywords=keywords,
            b=b, k=k, b_breakdown=b_breakdown, keyword_flags=keyword_flags,
            teacher_model=args.teacher_model, judge_model=args.judge_model,
        ))
        stats.kept += 1
        logger.info("[%s w%d] kept B=%d K=%d", episode, w_idx, b, k)

    return stats


def _example_to_chat(ex: Example, prompt_template: str) -> dict:
    user_content = prompt_template.format(transcript=ex.transcript)
    assistant_content = json.dumps(
        {"summary": ex.summary, "keywords": ex.keywords},
        ensure_ascii=False,
    )
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "_meta": {
            "episode": ex.episode,
            "window_idx": ex.window_idx,
            "B": ex.b,
            "K": ex.k,
            "B_breakdown": ex.b_breakdown,
            "keyword_flags": ex.keyword_flags,
            "teacher_model": ex.teacher_model,
            "judge_model": ex.judge_model,
        },
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _split_episodes(episodes: list[str], val_frac: float, test_frac: float, seed: int) -> tuple[list[str], list[str], list[str]]:
    """Split *by episode name* so transcript windows from one source can never
    appear in two splits — protects val/test from leakage.
    """
    rng = random.Random(seed)
    eps = sorted(episodes)
    rng.shuffle(eps)
    n = len(eps)
    n_test = max(1, int(round(n * test_frac))) if test_frac > 0 else 0
    n_val = max(1, int(round(n * val_frac))) if val_frac > 0 else 0
    if n_test + n_val >= n:
        # Pathological tiny corpus; degrade gracefully to 1/1/rest.
        n_test = 1 if test_frac > 0 else 0
        n_val = 1 if val_frac > 0 else 0
    test = eps[:n_test]
    val = eps[n_test : n_test + n_val]
    train = eps[n_test + n_val:]
    return train, val, test


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rev16-dir", type=Path, default=REV16_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--teacher-endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--teacher-model", default="qwen3.5:9b-16k")
    parser.add_argument("--teacher-num-ctx", type=int, default=4096)
    parser.add_argument("--teacher-num-predict", type=int, default=160)
    parser.add_argument("--judge-endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--judge-model", default="qwen3.5:9b-16k",
                        help="Use a different model than --teacher-model when possible to limit self-bias.")
    parser.add_argument("--judge-num-ctx", type=int, default=4096)
    parser.add_argument("--judge-num-predict", type=int, default=256)
    parser.add_argument("--window-words", type=int, default=600,
                        help="Words per training window (~600 ≈ 240s of speech at 150 wpm).")
    parser.add_argument("--stride-words", type=int, default=None,
                        help="Stride between windows; defaults to non-overlapping.")
    parser.add_argument("--min-b", type=int, default=18,
                        help="Discard examples below this B score (max 25).")
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.125)  # 2 of 16 by default
    parser.add_argument("--seed", type=int, default=20260510)
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="Process only N episodes (smoke-test mode).")
    parser.add_argument("--limit-windows-per-episode", type=int, default=None,
                        help="Cap windows per episode; useful for fast iteration.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("prepare_rev16")

    if not args.rev16_dir.exists():
        logger.error("rev16 dir not found: %s", args.rev16_dir)
        return 1

    txt_files = sorted(args.rev16_dir.glob("*/transcript.txt"))
    if args.max_episodes:
        txt_files = txt_files[: args.max_episodes]
    if not txt_files:
        logger.error("no .txt files in %s", args.rev16_dir)
        return 1

    prompt_template = _read_prompt_template()
    teacher_client = ollama.Client(host=args.teacher_endpoint)
    judge_client = ollama.Client(host=args.judge_endpoint) if args.judge_endpoint != args.teacher_endpoint else teacher_client

    if args.teacher_model == args.judge_model:
        logger.warning(
            "Teacher and judge are the same model (%s) — judge will be biased "
            "toward teacher's own outputs. Pass --judge-model to use a different model.",
            args.teacher_model,
        )

    all_stats: list[EpisodeStats] = []
    t0 = time.time()
    for txt in txt_files:
        stats = _process_episode(
            txt, prompt_template, args, teacher_client, judge_client, logger,
        )
        if args.limit_windows_per_episode:
            stats.examples = stats.examples[: args.limit_windows_per_episode]
        all_stats.append(stats)

    elapsed = time.time() - t0
    total_windows = sum(s.windows for s in all_stats)
    total_kept = sum(s.kept for s in all_stats)
    logger.info(
        "Processed %d episodes / %d windows in %.0fs — kept %d examples",
        len(all_stats), total_windows, elapsed, total_kept,
    )
    if total_kept == 0:
        logger.error(
            "No examples passed the filters. Lower --min-b, check teacher/judge "
            "models are pulled (`ollama list`), or run with --log-level DEBUG."
        )
        return 1

    train_eps, val_eps, test_eps = _split_episodes(
        [s.episode for s in all_stats if s.kept > 0],
        args.val_frac, args.test_frac, args.seed,
    )
    logger.info("Train episodes (%d): %s", len(train_eps), train_eps)
    logger.info("Val episodes (%d): %s", len(val_eps), val_eps)
    logger.info("Test episodes (%d): %s", len(test_eps), test_eps)

    rows_by_split: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for s in all_stats:
        if s.episode in train_eps:
            split = "train"
        elif s.episode in val_eps:
            split = "val"
        elif s.episode in test_eps:
            split = "test"
        else:
            continue
        for ex in s.examples:
            rows_by_split[split].append(_example_to_chat(ex, prompt_template))

    for split, rows in rows_by_split.items():
        out_path = args.out_dir / f"{split}.jsonl"
        _write_jsonl(out_path, rows)
        logger.info("Wrote %d examples → %s", len(rows), out_path)

    summary_path = args.out_dir / "prep_summary.json"
    summary_path.write_text(json.dumps({
        "elapsed_sec": elapsed,
        "teacher_model": args.teacher_model,
        "judge_model": args.judge_model,
        "min_b": args.min_b,
        "window_words": args.window_words,
        "totals": {
            "episodes": len(all_stats),
            "windows": total_windows,
            "kept": total_kept,
            "teacher_failed": sum(s.teacher_failed for s in all_stats),
            "judge_failed": sum(s.judge_failed for s in all_stats),
            "keyword_failed": sum(s.keyword_failed for s in all_stats),
            "below_threshold": sum(s.below_threshold for s in all_stats),
        },
        "splits": {
            "train": {"episodes": train_eps, "examples": len(rows_by_split["train"])},
            "val": {"episodes": val_eps, "examples": len(rows_by_split["val"])},
            "test": {"episodes": test_eps, "examples": len(rows_by_split["test"])},
        },
        "per_episode": [
            {
                "episode": s.episode,
                "windows": s.windows, "kept": s.kept,
                "teacher_failed": s.teacher_failed,
                "judge_failed": s.judge_failed,
                "keyword_failed": s.keyword_failed,
                "below_threshold": s.below_threshold,
            }
            for s in all_stats
        ],
    }, indent=2))
    logger.info("Wrote summary → %s", summary_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
