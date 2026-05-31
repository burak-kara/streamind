#!/usr/bin/env python3
"""Reprocess existing judge_scores_multi.json files with updated aggregate_scores/build_table.

Separates B+K (judge-derived) from L (objective latency) in all reports and
leaderboards. Safe to re-run: reads existing JSON, rewrites in place.

Usage (from repo root):
    # Rebuild specific stamp dirs:
    uv run python tools/postproc/rebuild_results.py results/ep10/20260530_204837

    # Rebuild ALL stamp dirs under results/ (auto-discover):
    uv run python tools/postproc/rebuild_results.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

_spec = importlib.util.spec_from_file_location(
    "eval_multi_judge", REPO_ROOT / "tools" / "eval_multi_judge.py"
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["eval_multi_judge"] = _mod
_spec.loader.exec_module(_mod)

aggregate_scores = _mod.aggregate_scores
build_table = _mod.build_table
JANUS_BONUS = _mod.JANUS_BONUS


def find_stamp_dirs(root: Path) -> list[Path]:
    """Auto-discover stamp dirs: dirs 4 levels above any judge_scores_multi.json.

    Layout: results/<audio>/<stamp>/<model>/<window>/<run>/judge_scores_multi.json
    """
    stamps: set[Path] = set()
    for p in root.rglob("judge_scores_multi.json"):
        stamps.add(p.parent.parent.parent.parent)
    return sorted(stamps)


def reconstruct_per_judge(old_agg: dict) -> dict[str, list[dict]]:
    per_judge: dict[str, list[dict]] = {j: [] for j in old_agg["judges"]}
    for w in old_agg["windows"]:
        for label, pj in w["per_judge"].items():
            per_judge[label].append({
                "path": w["path"],
                "B": pj.get("B"),
                "K": pj.get("K"),
                "L": pj.get("L"),
                "C": pj["C"],
            })
    return per_judge


def rebuild_run(run_dir: Path) -> dict:
    json_path = run_dir / "judge_scores_multi.json"
    old_agg = json.loads(json_path.read_text())
    new_agg = aggregate_scores(reconstruct_per_judge(old_agg), janus_bonus=JANUS_BONUS)
    json_path.write_text(json.dumps(new_agg, indent=2))
    (run_dir / "judge_report_multi.txt").write_text(build_table(new_agg, janus_bonus=JANUS_BONUS) + "\n")
    return new_agg


def _leaderboard_header(lb_path: Path) -> str:
    if not lb_path.exists():
        return ""
    lines = lb_path.read_text().splitlines()
    header: list[str] = []
    for line in lines:
        header.append(line)
        if line.startswith("===") and len(header) > 1:
            break
    return "\n".join(header)


def build_leaderboard(stamp_dir: Path) -> None:
    lb_path = stamp_dir / "leaderboard.txt"
    csv_path = stamp_dir / "leaderboard.csv"

    run_dirs = [p.parent for p in sorted(stamp_dir.rglob("judge_scores_multi.json"))]

    rows: list[tuple] = []
    for rd in run_dirs:
        c = json.loads((rd / "judge_scores_multi.json").read_text()).get("consensus", {})
        rel = rd.relative_to(stamp_dir).as_posix()
        rows.append((
            c.get("final_source_score") or 0.0,
            c.get("avg_bk") or 0.0,
            c.get("avg_l") or 0.0,
            "vllm-" + rel.split("/")[0],  # profile
            rd.name,                       # run label (e.g. run2)
            rel,
            rd,
        ))
    rows.sort(key=lambda r: r[0], reverse=True)

    # CSV
    csv_lines = ["rank,profile,run,avg_bk,avg_l,score"]
    for rank, (final, avg_bk, avg_l, profile, run_label, rel, _rd) in enumerate(rows, 1):
        csv_lines.append(
            f"{rank},{profile},{run_label},"
            f"{'N/A' if avg_bk == 0 else f'{avg_bk:.2f}'},"
            f"{'N/A' if avg_l == 0 else f'{avg_l:.2f}'},"
            f"{'N/A' if final == 0 else f'{final:.2f}'}"
        )
    csv_path.write_text("\n".join(csv_lines) + "\n")

    # Text leaderboard
    header = _leaderboard_header(lb_path)
    lb_lines: list[str] = [header, ""] if header else []
    lb_lines.append("## Leaderboard (final = avg B+K + avg L + Janus — higher is better)")
    lb_lines.append(f"  {'profile':<28} {'run':<8} {'avg_BK':>7} {'avg_L':>6} {'score':>7}")
    for final, avg_bk, avg_l, profile, run_label, _rel, _rd in rows:
        lb_lines.append(
            f"  {profile:<28} {run_label:<8}"
            f" {'   N/A ' if avg_bk == 0 else f'{avg_bk:7.2f}'}"
            f" {'  N/A ' if avg_l == 0 else f'{avg_l:6.2f}'}"
            f" {'   N/A ' if final == 0 else f'{final:7.2f}'}"
        )
    lb_lines.append("")

    for _, _, _, profile, _run, rel, rd in sorted(rows, key=lambda r: r[5]):
        lb_lines.append(f"## {profile}  ({rel})")
        report = rd / "judge_report_multi.txt"
        lb_lines.append(report.read_text().rstrip() if report.exists() else "  (no report)")
        lb_lines.append("")

    if lb_path.exists():
        for line in lb_path.read_text().splitlines():
            if line.startswith("Run log:"):
                lb_lines.append(line)
                break

    lb_path.write_text("\n".join(lb_lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "stamp_dirs", nargs="*", type=Path,
        help="Stamp dirs to rebuild (e.g. results/ep10/20260530_204837). "
             "Default: auto-discover all stamp dirs under results/.",
    )
    args = parser.parse_args()

    stamp_dirs = (
        [Path(p).resolve() for p in args.stamp_dirs]
        if args.stamp_dirs
        else find_stamp_dirs(REPO_ROOT / "results")
    )
    if not stamp_dirs:
        print("No stamp dirs found.", file=sys.stderr)
        return 1

    print("Rebuilding run reports...")
    for stamp_dir in stamp_dirs:
        for json_path in sorted(stamp_dir.rglob("judge_scores_multi.json")):
            rebuild_run(json_path.parent)
            print(f"  rebuilt {json_path.parent.relative_to(REPO_ROOT)}")

    print("\nRebuilding leaderboards...")
    for stamp_dir in stamp_dirs:
        build_leaderboard(stamp_dir)
        print(f"  rebuilt {stamp_dir.relative_to(REPO_ROOT)}/leaderboard.{{txt,csv}}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
