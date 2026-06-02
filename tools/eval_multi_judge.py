#!/usr/bin/env python3
"""Multi-judge offline evaluation orchestrator.

Scores pipeline output with several judge models from different families,
**one after another**, and reports all judges' scores plus a cross-judge
consensus in a single table/file. A single judge can bias every B/K/L score;
running an independent panel and measuring their spread turns that hidden bias
into an explicit signal.

Each judge runs as its own subprocess invocation of tools/eval_quality.py. A
single GPU (24 GB dev RTX 4090 / 32 GB prod RTX Pro 4500) cannot hold the panel
co-resident, and process exit fully reclaims CUDA VRAM before the next model
loads — far more robust than in-process vLLM load/unload, which leaks VRAM
across repeated loads.

Usage (on uni-lab, after the summarizer pipeline exits):

    uv run python tools/eval_multi_judge.py results/<run>/300/ \
      --judge-profiles vllm-mistral-small-24b-awq vllm-phi-4-awq vllm-gemma3-27b-it-int4-awq \
      --audio datasets/rev16/<episode>/audio.opus

Outputs (next to the source results):
  - judge_report_<profile>.txt   per-judge table (written by each child)
  - judge_report_multi.txt       combined table: C per judge + mean + stdev
  - judge_scores_multi.json       machine-readable per-window + per-judge + consensus
  - results/.../judge/<profile>/window_N.json   per-window judge JSON, namespaced
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean, pstdev

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Load the shared scoring contract (same importlib pattern as eval_quality.py;
# plugins/ is not an importable package). JANUS_BONUS lives there so the
# combined report's final source score cannot drift from the per-judge reports.
_scorer_spec = importlib.util.spec_from_file_location(
    "judge_scorer",
    _REPO_ROOT / "plugins" / "nodes" / "sink" / "_judge_common" / "scorer.py",
)
_scorer = importlib.util.module_from_spec(_scorer_spec)
assert _scorer_spec.loader is not None
sys.modules["judge_scorer"] = _scorer
_scorer_spec.loader.exec_module(_scorer)

JANUS_BONUS = _scorer.JANUS_BONUS


def aggregate_scores(
    per_judge: dict[str, list[dict]],
    janus_bonus: float = JANUS_BONUS,
) -> dict:
    """Combine per-judge window scores into a cross-judge view.

    `per_judge` maps a judge label to the list of window-score dicts emitted by
    eval_quality.py's --json-out (each has at least "path" and "C"; "B"/"K"/"L"
    are included when available). Pure function — no filesystem or GPU — so it
    is unit-testable on synthetic inputs.

    B and K come from LLM judges (subjective). L = 10·exp(-0.5·proc_time) when
    B≥10 (objective — proc_time is measured, not judged). Reporting them separately
    lets callers distinguish judge variance from measured latency.

    Returns:
      {
        "judges": [ordered judge labels],
        "windows": [
          {
            "path",
            "per_judge": {label: {B, K, BK, L, C}},
            "mean_c", "stdev_c",
            "mean_bk", "stdev_bk",
            "mean_l",
          }
        ],
        "judge_summary": {label: {"avg_c", "avg_bk", "final_source_score", "n"}},
        "consensus": {"avg_c", "avg_bk", "avg_l", "final_source_score", "n_windows"},
      }
    """
    labels = list(per_judge.keys())

    # Index each judge's rows by window path; preserve first-seen window order.
    by_label: dict[str, dict[str, dict]] = {}
    ordered_paths: list[str] = []
    seen: set[str] = set()
    for label in labels:
        rows = {}
        for row in per_judge[label]:
            path = row["path"]
            rows[path] = row
            if path not in seen:
                seen.add(path)
                ordered_paths.append(path)
        by_label[label] = rows

    windows: list[dict] = []
    for path in ordered_paths:
        per = {}
        cs: list[float] = []
        bks: list[float] = []
        ls_w: list[float] = []
        for label in labels:
            row = by_label[label].get(path)
            if row is None:
                continue  # judge skipped/failed this window — exclude from stats
            b = row.get("B")
            k = row.get("K")
            l = row.get("L")
            bk = float((b or 0) + (k or 0))
            per[label] = {
                "B": b,
                "K": k,
                "BK": bk,
                "L": l,
                "C": row["C"],
            }
            cs.append(float(row["C"]))
            bks.append(bk)
            if l is not None:
                ls_w.append(float(l))
        windows.append({
            "path": path,
            "per_judge": per,
            "mean_c": mean(cs) if cs else None,
            "stdev_c": pstdev(cs) if len(cs) > 1 else 0.0,
            "mean_bk": mean(bks) if bks else None,
            "stdev_bk": pstdev(bks) if len(bks) > 1 else 0.0,
            "mean_l": mean(ls_w) if ls_w else None,
        })

    judge_summary: dict[str, dict] = {}
    for label in labels:
        cs = [float(r["C"]) for r in per_judge[label]]
        bks = [float((r.get("B") or 0) + (r.get("K") or 0)) for r in per_judge[label]]
        avg_c = mean(cs) if cs else None
        avg_bk = mean(bks) if bks else None
        judge_summary[label] = {
            "avg_c": avg_c,
            "avg_bk": avg_bk,
            "final_source_score": (avg_c + janus_bonus) if avg_c is not None else None,
            "n": len(cs),
        }

    # Consensus = mean across each window's per-judge mean C, then averaged over
    # windows. Equivalent to averaging the per-judge avg_c when every judge
    # scored every window; robust to gaps because it averages what is present.
    win_means = [w["mean_c"] for w in windows if w["mean_c"] is not None]
    win_mean_bks = [w["mean_bk"] for w in windows if w["mean_bk"] is not None]
    win_mean_ls = [w["mean_l"] for w in windows if w["mean_l"] is not None]
    consensus_avg = mean(win_means) if win_means else None
    consensus_avg_bk = mean(win_mean_bks) if win_mean_bks else None
    consensus_avg_l = mean(win_mean_ls) if win_mean_ls else None
    consensus = {
        "avg_c": consensus_avg,
        "avg_bk": consensus_avg_bk,
        "avg_l": consensus_avg_l,
        "final_source_score": (consensus_avg + janus_bonus) if consensus_avg is not None else None,
        "n_windows": len(win_means),
    }

    return {
        "judges": labels,
        "windows": windows,
        "judge_summary": judge_summary,
        "consensus": consensus,
    }


def _short_path(path: str, width: int = 40) -> str:
    return path[-width:].rjust(width)


def _short_label(label: str, width: int = 12) -> str:
    """Readable column header: drop the vllm- prefix, keep the leading tokens."""
    return label.removeprefix("vllm-")[:width]


def build_table(agg: dict, janus_bonus: float = JANUS_BONUS) -> str:
    """Build the multi-judge report table.

    Judge columns show B+K (not C). L is objective (proc_time-derived) and
    appears as a single column shared across judges. Final source score =
    avg B+K + avg L + Janus, which equals the old avg C + Janus.
    """
    labels = agg["judges"]
    width = 40
    col = 12
    l_col = 7
    head = f"{'window':{width}}" + "".join(f" {_short_label(l, col):>{col}}" for l in labels)
    head += f" {'mean':>7} {'stdev':>6} {'L':>{l_col}}"
    lines = [head, "-" * len(head)]

    for w in agg["windows"]:
        row = f"{_short_path(w['path'], width)}"
        for l in labels:
            bk = w["per_judge"].get(l, {}).get("BK")
            row += f" {bk:>{col}.2f}" if bk is not None else f" {'-':>{col}}"
        row += f" {w['mean_bk']:>7.2f}" if w["mean_bk"] is not None else f" {'-':>7}"
        row += f" {w['stdev_bk']:>6.2f}" if w["mean_bk"] is not None else f" {'-':>6}"
        row += f" {w['mean_l']:>{l_col}.2f}" if w.get("mean_l") is not None else f" {'-':>{l_col}}"
        lines.append(row)
    lines.append("-" * len(head))

    # avg B+K row (per judge + consensus mean; no L here)
    avg_bk_row = f"{'avg B+K':{width}}"
    for l in labels:
        avg_bk = agg["judge_summary"][l].get("avg_bk")
        avg_bk_row += f" {avg_bk:>{col}.2f}" if avg_bk is not None else f" {'-':>{col}}"
    consensus_avg_bk = agg["consensus"].get("avg_bk")
    avg_bk_row += f" {consensus_avg_bk:>7.2f}" if consensus_avg_bk is not None else f" {'-':>7}"
    avg_bk_row += f" {'':>6} {'':>{l_col}}"
    lines.append(avg_bk_row)

    # avg L row (objective — same regardless of judge; shown in L column only)
    avg_l_row = f"{'avg L':{width}}"
    for _ in labels:
        avg_l_row += f" {'-':>{col}}"
    avg_l_row += f" {'-':>7} {'-':>6}"
    consensus_avg_l = agg["consensus"].get("avg_l")
    avg_l_row += f" {consensus_avg_l:>{l_col}.2f}" if consensus_avg_l is not None else f" {'-':>{l_col}}"
    lines.append(avg_l_row)

    # final source score = avg B+K (per judge) + avg L + Janus
    final_row = f"{'final source score (+Janus)':{width}}"
    avg_l_val = agg["consensus"].get("avg_l") or 0.0
    for l in labels:
        fss = agg["judge_summary"][l]["final_source_score"]
        final_row += f" {fss:>{col}.2f}" if fss is not None else f" {'-':>{col}}"
    cfss = agg["consensus"]["final_source_score"]
    final_row += f" {cfss:>7.2f}" if cfss is not None else f" {'-':>7}"
    final_row += f" {'':>6} {'':>{l_col}}"
    lines.append(final_row)

    lines.append("")
    lines.append(
        f"Janus bonus: +{janus_bonus:.0f} flat (per CHALLENGE.md). "
        "Final source score = avg B+K + avg L + Janus."
    )
    lines.append(
        "B+K from LLM judges (subjective); L = 10·exp(-0.5·proc_time) when B≥10 (objective). "
        "Low stdev = judges agree on B+K; high stdev = judge bias."
    )
    lines.append(
        "Consensus = mean across judges per window, averaged over windows."
    )
    return "\n".join(lines)


def _run_judge(
    results_dir: Path,
    profile: str,
    json_out: Path,
    report_out: Path,
    passthrough: list[str],
) -> bool:
    """Run one judge subprocess. Returns True on success."""
    cmd = [
        sys.executable,
        str(_REPO_ROOT / "tools" / "eval_quality.py"),
        str(results_dir),
        "--judge-profile", profile,
        "--judge-tag", profile,
        "--json-out", str(json_out),
        "--report-out", str(report_out),
        *passthrough,
    ]
    print(f"\n=== judge: {profile} ===", file=sys.stderr, flush=True)
    print(" ".join(cmd), file=sys.stderr, flush=True)
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        print(f"judge {profile} failed (exit {proc.returncode})", file=sys.stderr)
        return False
    return True


def _enrich_with_bounds(per_judge: dict[str, list[dict]]) -> None:
    """Attach from/to read from each source window file (best-effort).

    json-out lacks time bounds; the source window JSON has them. Mutates rows
    in place. Skipped silently if a file is unreadable (e.g. test fixtures).
    """
    for rows in per_judge.values():
        for row in rows:
            try:
                src = json.loads(Path(row["path"]).read_text())
                row.setdefault("from", src.get("from"))
                row.setdefault("to", src.get("to"))
            except Exception:
                pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_dir", type=Path,
                        help="Directory containing window_*.json files (recursive).")
    parser.add_argument("--judge-profiles", nargs="+", required=True,
                        help="Judge profile names under pipelines/judge/<name>.json, "
                             "run sequentially in the given order.")
    parser.add_argument("--audio", type=Path, default=None,
                        help="Source audio (passthrough to eval_quality.py; enables WER).")
    parser.add_argument("--reference", type=Path, default=None,
                        help="Ground-truth transcript .txt (passthrough).")
    parser.add_argument("--ground-truth", type=Path, default=None,
                        help="chunks.json with reference summaries/keywords (passthrough).")
    parser.add_argument("--filter", default=None,
                        help="Substring match on window path (passthrough).")
    parser.add_argument("--no-janus-bonus", action="store_true",
                        help="Exclude the +4 Janus bonus from final source scores.")
    parser.add_argument("--tmp-dir", type=Path, default=_REPO_ROOT / "tmp",
                        help="Directory for intermediate per-judge json-out files.")
    args = parser.parse_args()

    if not args.results_dir.exists():
        print(f"results dir not found: {args.results_dir}", file=sys.stderr)
        return 1

    # Build passthrough args once, forwarded verbatim to every child.
    passthrough: list[str] = []
    if args.audio is not None:
        passthrough += ["--audio", str(args.audio)]
    if args.reference is not None:
        passthrough += ["--reference", str(args.reference)]
    if args.ground_truth is not None:
        passthrough += ["--ground-truth", str(args.ground_truth)]
    if args.filter is not None:
        passthrough += ["--filter", args.filter]
    if args.no_janus_bonus:
        passthrough += ["--no-janus-bonus"]

    args.tmp_dir.mkdir(parents=True, exist_ok=True)
    janus_bonus = 0.0 if args.no_janus_bonus else JANUS_BONUS

    per_judge: dict[str, list[dict]] = {}
    for profile in args.judge_profiles:
        json_out = args.tmp_dir / f"judge_{profile}.json"
        report_out = args.results_dir / f"judge_report_{profile}.txt"
        ok = _run_judge(args.results_dir, profile, json_out, report_out, passthrough)
        if not ok:
            print(f"skipping {profile} in aggregation", file=sys.stderr)
            continue
        try:
            per_judge[profile] = json.loads(json_out.read_text())
        except Exception as e:
            print(f"could not read {json_out}: {e}", file=sys.stderr)

    if not per_judge:
        print("no judge produced scores; nothing to aggregate", file=sys.stderr)
        return 1

    _enrich_with_bounds(per_judge)
    agg = aggregate_scores(per_judge, janus_bonus=janus_bonus)

    table = build_table(agg, janus_bonus=janus_bonus)
    print("\n" + table)

    report_path = args.results_dir / "judge_report_multi.txt"
    report_path.write_text(table + "\n")
    print(f"\ncombined report saved: {report_path}", file=sys.stderr)

    scores_path = args.results_dir / "judge_scores_multi.json"
    scores_path.write_text(json.dumps(agg, indent=2))
    print(f"combined scores saved: {scores_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
