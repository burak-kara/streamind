#!/usr/bin/env python3
"""Re-render the user turn of each training example with the live prompt.

The distilled JSONLs (train/val/test) were generated against an older prompt
(`/no_think` + the Ollama-era template). The deployed vLLM summarizer uses
`plugins/nodes/proc/_summarizer_vllm/summarize_prompt.txt`. Training on one
prompt and serving another wastes the fine-tune, so this rewraps every
example's user message to the current template while keeping the teacher's
`assistant` answer and `_meta` untouched.

The transcript is recovered from the existing user turn (everything after the
last "Transcript:\n" marker), then re-wrapped. Idempotent: re-running against
already-wrapped data reproduces the same output.

Run after any change to summarize_prompt.txt.

Usage:
    uv run python tools/finetune/rewrap_prompts.py
    uv run python tools/finetune/rewrap_prompts.py --check   # report drift, write nothing
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PROMPT_FILE = REPO_ROOT / "plugins/nodes/proc/_summarizer_vllm/summarize_prompt.txt"
DATA_DIR = REPO_ROOT / "tools/finetune/data"
SPLITS = ("train", "val", "test")
MARKER = "Transcript:\n"


def _extract_transcript(user_content: str) -> str:
    idx = user_content.rfind(MARKER)
    if idx == -1:
        # No marker — treat the whole message as the transcript body. Shouldn't
        # happen with our data, but fail soft rather than silently truncate.
        return user_content.strip()
    return user_content[idx + len(MARKER):].strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--prompt-file", type=Path, default=PROMPT_FILE)
    parser.add_argument("--check", action="store_true",
                        help="Report how many rows would change; write nothing.")
    args = parser.parse_args()

    if not args.prompt_file.exists():
        print(f"prompt template not found: {args.prompt_file}", file=sys.stderr)
        return 1
    template = args.prompt_file.read_text()
    if "{transcript}" not in template:
        print(f"prompt template missing {{transcript}} placeholder: {args.prompt_file}", file=sys.stderr)
        return 1

    total_changed = 0
    for split in SPLITS:
        path = args.data_dir / f"{split}.jsonl"
        if not path.exists():
            print(f"skip {split}: {path} not found", file=sys.stderr)
            continue

        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        changed = 0
        out_lines = []
        for row in rows:
            msgs = row["messages"]
            user_idx = next(i for i, m in enumerate(msgs) if m["role"] == "user")
            transcript = _extract_transcript(msgs[user_idx]["content"])
            new_user = template.format(transcript=transcript)
            if new_user != msgs[user_idx]["content"]:
                changed += 1
            msgs[user_idx]["content"] = new_user
            out_lines.append(json.dumps(row, ensure_ascii=False))

        total_changed += changed
        if args.check:
            print(f"{split}: {changed}/{len(rows)} rows would change")
        else:
            path.write_text("\n".join(out_lines) + "\n")
            print(f"{split}: rewrapped {changed}/{len(rows)} rows -> {path}")

    if args.check and total_changed > 0:
        print(f"\n{total_changed} rows drift from the current prompt. Run without --check to fix.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
