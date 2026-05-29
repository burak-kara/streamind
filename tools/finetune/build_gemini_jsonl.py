"""Merge Gemini-labeled oneill outputs with the existing train.jsonl.

Reads:
  - results/qwen3.5-4b-ft/300/debug/window_*_transcript.txt  (transcripts)
  - tools/finetune/data/oneill_gemini_outputs.jsonl          (Gemini responses, one per window in order)
  - tools/finetune/data/train.jsonl                          (existing training data with old teacher labels)
  - plugins/nodes/proc/_summarizer_vllm/summarize_prompt.txt (production prompt template)

Writes:
  - tools/finetune/data/train_gemini.jsonl: new train set, oneill examples replaced with
    Gemini-labeled versions, other episodes retain old-teacher labels.
"""
import json
from pathlib import Path

window_dir = Path("results/qwen3.5-4b-ft/300/debug")
gemini_path = Path("tools/finetune/data/oneill_gemini_outputs.jsonl")
old_train = Path("tools/finetune/data/train.jsonl")
out = Path("tools/finetune/data/train_gemini.jsonl")

prompt = Path("plugins/nodes/proc/_summarizer_vllm/summarize_prompt.txt").read_text()

transcripts = sorted(
    window_dir.glob("window_*_transcript.txt"),
    key=lambda p: int(p.stem.split("_")[1])
)
gemini_responses = [json.loads(l) for l in gemini_path.read_text().splitlines() if l.strip()]

assert len(transcripts) == len(gemini_responses), \
    f"transcript count {len(transcripts)} != gemini response count {len(gemini_responses)}"

new_oneill = []
for tp, resp in zip(transcripts, gemini_responses):
    idx = int(tp.stem.split("_")[1])
    transcript = tp.read_text().strip()
    user_content = prompt.format(transcript=transcript)
    new_oneill.append({
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": json.dumps(resp, ensure_ascii=False)},
        ],
        "_meta": {
            "episode": "26_Episode_338_oneill_GEMINI",
            "window_idx": idx,
            "teacher_model": "gemini-2.5-pro-manual",
        },
    })

keep = []
for line in old_train.read_text().splitlines():
    if not line.strip():
        continue
    r = json.loads(line)
    if not r["_meta"]["episode"].startswith("26_Episode_338"):
        keep.append(r)

out_rows = keep + new_oneill
print(f"Kept from old train (non-oneill): {len(keep)}")
print(f"New oneill (Gemini-labeled): {len(new_oneill)}")
print(f"Total: {len(out_rows)}")
with out.open("w") as f:
    for r in out_rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"Wrote {out}")
