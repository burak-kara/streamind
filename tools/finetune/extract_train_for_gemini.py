"""Dump train.jsonl into batched paste-ready files for manual Gemini relabel.

Interleaves examples across episodes so each batch has variety — prevents
Gemini from calibrating to one style for an entire turn.
"""
import json
from collections import defaultdict
from pathlib import Path

src = Path("tools/finetune/data/train.jsonl")
out_dir = Path("tools/finetune/data/gemini_batches")
out_dir.mkdir(parents=True, exist_ok=True)

BATCH = 16

rows = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
print(f"Total train rows: {len(rows)}")

# Group by episode, then round-robin pull
by_ep = defaultdict(list)
for i, r in enumerate(rows):
    ep = r["_meta"]["episode"]
    by_ep[ep].append((i, r))

# Round-robin into a flat list -> interleaved order
interleaved = []
queues = list(by_ep.values())
while any(queues):
    for q in queues:
        if q:
            interleaved.append(q.pop(0))

print(f"Interleaved order: {[ (i, r['_meta']['episode'][:30]) for i, r in interleaved[:8] ]}")

# Build batches in interleaved order, but write the ORIGINAL row index in headers
# so we can map Gemini outputs back to train.jsonl rows later.
index_map = []   # batch order index -> original row index

for batch_idx, start in enumerate(range(0, len(interleaved), BATCH)):
    batch = interleaved[start:start + BATCH]
    blocks = []
    for batch_pos, (orig_i, r) in enumerate(batch):
        global_pos = start + batch_pos
        index_map.append(orig_i)
        user_content = r["messages"][0]["content"]
        ep_short = r["_meta"]["episode"][:30]
        blocks.append(f"########## EXAMPLE {global_pos} (ep={ep_short}) ##########\n{user_content}\n")
    text = "\n".join(blocks)
    path = out_dir / f"batch_{batch_idx:02d}.txt"
    path.write_text(text)
    print(f"batch {batch_idx:02d}: {len(batch)} examples -> {path}")

# Save the index map so we can reorder Gemini outputs back to train order later
(out_dir / "index_map.json").write_text(json.dumps(index_map))
print(f"index_map written ({len(index_map)} entries)")
