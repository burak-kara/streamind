#!/usr/bin/env python3
"""Assemble a complete Juturna pipeline config from base + summarizer profile.

Usage:
    python tools/assemble_config.py <window_seconds> <profile_name> <output_path>

Example:
    python tools/assemble_config.py 300 vllm-qwen3-8b ./tmp/config.json

Judge runs offline via tools/eval_quality.py — no judge profile is merged
into the live pipeline.
"""
import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("window", type=float, help="Window duration in seconds")
    parser.add_argument("profile_name", help="Summarizer profile name (without .json)")
    parser.add_argument("output_path", type=Path, help="Output config path")
    parser.add_argument("--asr", default=None, help="Override transcriber model_name (e.g. small.en)")
    parser.add_argument("--results-dir", default=None,
                        help="Override transmitter results_dir (e.g. ./results/<stamp> for a non-clobbering run root)")
    args = parser.parse_args()

    profile_path = Path(f"pipelines/summarizer/{args.profile_name}.json")
    if not profile_path.exists():
        available = sorted(p.stem for p in Path("pipelines/summarizer").glob("*.json"))
        print(f"Profile not found: {profile_path}")
        print(f"Available: {', '.join(available) if available else '(none)'}")
        sys.exit(1)

    config = json.loads(Path("pipelines/config-base.json").read_text())
    summarizer_node = json.loads(profile_path.read_text())

    for node in config["pipeline"]["nodes"]:
        if node["name"] == "aggregator":
            node["configuration"]["window_duration"] = args.window
        if args.asr and node["name"] == "transcriber":
            node["configuration"]["model_name"] = args.asr
        if args.results_dir and node["name"] == "transmitter":
            node["configuration"]["results_dir"] = args.results_dir

    nodes = config["pipeline"]["nodes"]
    tx_idx = next(i for i, n in enumerate(nodes) if n["name"] == "transmitter")
    nodes.insert(tx_idx, summarizer_node)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(config, indent=2))
    print(f"Assembled: {args.output_path}")


if __name__ == "__main__":
    main()
