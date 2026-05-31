#!/usr/bin/env python3
"""Aggregate per-stamp leaderboards for one or more audio result folders.

Reads <audio_dir>/<stamp>/leaderboard.csv for every stamp, merges with a
`stamp` column, re-ranks by score (descending), and writes a single
<audio_dir>/leaderboard.csv.

Usage (from repo root):
    # Single audio dir:
    uv run python tools/postproc/rank_audio.py results/10_Creating_Your_Own_Lane_in_Pod

    # Multiple audio dirs:
    uv run python tools/postproc/rank_audio.py results/10_Creating_Your_Own_Lane_in_Pod results/11_Podcast_Tips_From_Berry_audio

    # All audio dirs under results/ at once:
    uv run python tools/postproc/rank_audio.py results/
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

_STAMP_RE = __import__("re").compile(r"^\d{8}_\d{6}$")


def _is_audio_dir(d: Path) -> bool:
    """True if d contains any stamp-named subdirs (YYYYMMDD_HHMMSS)."""
    return any(_STAMP_RE.match(child.name) for child in d.iterdir() if child.is_dir())


def rank_audio_dir(audio_dir: Path) -> None:
    stamp_csvs = sorted(audio_dir.glob("*/leaderboard.csv"))
    if not stamp_csvs:
        print(f"  {audio_dir.name}: no stamp leaderboard.csv files found — skipping", file=sys.stderr)
        return

    all_rows: list[dict] = []
    for csv_path in stamp_csvs:
        stamp = csv_path.parent.name
        with csv_path.open(newline="") as f:
            for row in csv.DictReader(f):
                all_rows.append({
                    "stamp":   stamp,
                    "profile": row["profile"],
                    "run":     row["run"],
                    "avg_bk":  row["avg_bk"],
                    "avg_l":   row["avg_l"],
                    "score":   row["score"],
                })

    all_rows.sort(
        key=lambda r: float(r["score"]) if r["score"] not in ("N/A", "") else 0.0,
        reverse=True,
    )

    out_path = audio_dir / "leaderboard.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["rank", "stamp", "profile", "run", "avg_bk", "avg_l", "score"]
        )
        writer.writeheader()
        for rank, row in enumerate(all_rows, 1):
            writer.writerow({"rank": rank, **row})

    rel = out_path.relative_to(REPO_ROOT)
    print(f"  wrote {rel}  ({len(all_rows)} entries from {len(stamp_csvs)} stamps)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "dirs", nargs="+", type=Path,
        help="Audio result dir(s), or the results/ root to process all audio dirs.",
    )
    args = parser.parse_args()

    for d in args.dirs:
        d = Path(d).resolve()
        if not d.is_dir():
            print(f"not a directory: {d}", file=sys.stderr)
            continue
        if _is_audio_dir(d):
            rank_audio_dir(d)
        else:
            # Treat as results/ root — iterate audio dirs one level down
            found = False
            for child in sorted(d.iterdir()):
                if child.is_dir() and _is_audio_dir(child):
                    rank_audio_dir(child)
                    found = True
            if not found:
                print(f"  {d}: no audio dirs found", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
