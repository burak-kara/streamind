"""Replace ALL assistant turns in train.jsonl with Gemini outputs.

Gemini outputs are in interleaved batch order; index_map.json tells us
which original train row each interleaved position corresponds to. Reorder
back to original train order, then write a new train file.
"""
import json
from pathlib import Path

src = Path("tools/finetune/data/train.jsonl")
gemini = Path("tools/finetune/data/all_gemini_outputs.jsonl")
index_map_path = Path("tools/finetune/data/gemini_batches/index_map.json")
out = Path("tools/finetune/data/train_full_gemini.jsonl")

train_rows = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
gemini_resps = [json.loads(l) for l in gemini.read_text().splitlines() if l.strip()]
index_map = json.loads(index_map_path.read_text())

assert len(train_rows) == len(gemini_resps) == len(index_map), \
    f"length mismatch: train={len(train_rows)} gemini={len(gemini_resps)} idx={len(index_map)}"

# index_map[k] = original_train_row_index for the k-th interleaved position
# So gemini_resps[k] should replace train_rows[index_map[k]]
ordered_responses = [None] * len(train_rows)
for k, orig_i in enumerate(index_map):
    ordered_responses[orig_i] = gemini_resps[k]

new_rows = []
for old, resp in zip(train_rows, ordered_responses):
    new_meta = dict(old.get("_meta") or {})
    new_meta["teacher_model"] = "gemini-2.5-pro-manual"
    for k in ("B", "K", "B_breakdown", "keyword_flags"):
        new_meta.pop(k, None)
    new_rows.append({
        "messages": [
            old["messages"][0],
            {"role": "assistant", "content": json.dumps(resp, ensure_ascii=False)},
        ],
        "_meta": new_meta,
    })

with out.open("w") as f:
    for r in new_rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"Wrote {len(new_rows)} rows -> {out}")
