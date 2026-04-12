#!/usr/bin/env python3
"""Assemble a complete Juturna pipeline config from base + summarizer profile.

Usage:
    python tools/assemble_config.py <window_seconds> <profile_name> <output_path>

Example:
    python tools/assemble_config.py 300 ollama-qwen3.5-9b ./tmp/config-300s-ollama-qwen3.5-9b.json
"""
import json
import sys
from pathlib import Path


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)

    window = float(sys.argv[1])
    profile_name = sys.argv[2]
    output_path = Path(sys.argv[3])

    profile_path = Path(f"pipelines/summarizer/{profile_name}.json")
    if not profile_path.exists():
        available = sorted(p.stem for p in Path("pipelines/summarizer").glob("*.json"))
        print(f"Profile not found: {profile_path}")
        print(f"Available: {', '.join(available)}")
        sys.exit(1)

    config = json.loads(Path("pipelines/config-base.json").read_text())
    summarizer_node = json.loads(profile_path.read_text())

    for node in config["pipeline"]["nodes"]:
        if node["name"] == "aggregator":
            node["configuration"]["window_duration"] = window

    nodes = config["pipeline"]["nodes"]
    tx_idx = next(i for i, n in enumerate(nodes) if n["name"] == "transmitter")
    nodes.insert(tx_idx, summarizer_node)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(config, indent=2))
    print(f"Assembled: {output_path}")


if __name__ == "__main__":
    main()
