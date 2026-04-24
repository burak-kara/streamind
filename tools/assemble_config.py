#!/usr/bin/env python3
"""Assemble a complete Juturna pipeline config from base + summarizer profile.

Usage:
    python tools/assemble_config.py <window_seconds> <profile_name> <output_path> [--judge <judge_profile>]

Examples:
    python tools/assemble_config.py 300 ollama-qwen3.5-9b ./tmp/config.json
    python tools/assemble_config.py 300 ollama-qwen3.5-9b ./tmp/config.json --judge ollama-qwen3.5-4b
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
    parser.add_argument("--judge", default=None, help="Optional judge profile name (without .json)")
    args = parser.parse_args()

    profile_path = Path(f"pipelines/summarizer/{args.profile_name}.json")
    if not profile_path.exists():
        available = sorted(p.stem for p in Path("pipelines/summarizer").glob("*.json"))
        print(f"Profile not found: {profile_path}")
        print(f"Available: {', '.join(available)}")
        sys.exit(1)

    config = json.loads(Path("pipelines/config-base.json").read_text())
    summarizer_node = json.loads(profile_path.read_text())

    for node in config["pipeline"]["nodes"]:
        if node["name"] == "aggregator":
            node["configuration"]["window_duration"] = args.window

    nodes = config["pipeline"]["nodes"]
    tx_idx = next(i for i, n in enumerate(nodes) if n["name"] == "transmitter")
    nodes.insert(tx_idx, summarizer_node)

    if args.judge:
        judge_path = Path(f"pipelines/judge/{args.judge}.json")
        if not judge_path.exists():
            available = sorted(p.stem for p in Path("pipelines/judge").glob("*.json"))
            print(f"Judge profile not found: {judge_path}")
            print(f"Available: {', '.join(available)}")
            sys.exit(1)
        judge_node = json.loads(judge_path.read_text())
        nodes.append(judge_node)
        # Fork: summarizer feeds both the live transmitter and the async judge.
        config["pipeline"]["links"].append({
            "from": summarizer_node["name"],
            "to": judge_node["name"],
        })

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(config, indent=2))
    print(f"Assembled: {args.output_path}")


if __name__ == "__main__":
    main()
