#!/usr/bin/env python3
"""A blind batch: two brains' answers to the same messages, for Immy's yes or no without knowing which brain said
what (Immy, 2026-10-19: AI can't judge whether he sounds like an AI, so his yes and no pick the better brain). Both
brains lived the same made-up days (tools/train/ab_days.sh: same seeds, same messages), so each message Immy sent in
both gets both replies, next to each other in a random order; then a few of each brain's quiet thoughts, mixed in.
The page only ever sees id, kind, when, heard, reply (sync.py), so nothing on it says which brain. Which is which is
in KEY.json, kept on the Mac.

usage: ab_batch.py BATCH OUT.jsonl KEY.json A_ITEMS.jsonl B_ITEMS.jsonl   (A/B items: export.py on each brain's days,
       with PARTS=replies,thoughts KEEP_CUT=0)
env: THOUGHTS (default 20): quiet thoughts from each brain
"""
import json, os, random, sys


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def stream(x):
    """'B-2010/day-001' -> ('2010', 'day-001'): the same made-up day for both brains"""
    side_seed, day = x["day"].split("/")
    return side_seed.split("-", 1)[1], day


if __name__ == "__main__":
    batch, out, key_path, a_path, b_path = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
    if os.path.exists(out) or os.path.exists(key_path):
        raise SystemExit("exists: a batch's ids may already have answers; use a new batch number")
    rng = random.Random(batch)
    sides = {"A": jl(a_path), "B": jl(b_path)}
    replies = {s: {} for s in sides}
    for s, xs in sides.items():
        for x in xs:
            if x["kind"] == "reply" and not x.get("cut"):
                replies[s].setdefault((*stream(x), x["heard"]), x)
    groups = []
    for k in sorted(set(replies["A"]) & set(replies["B"])):
        pair = [("A", replies["A"][k]), ("B", replies["B"][k])]
        rng.shuffle(pair)
        groups.append(pair)
    n_thoughts = int(os.environ.get("THOUGHTS", "20"))
    for s, xs in sides.items():
        th = [x for x in xs if x["kind"] == "thought" and x.get("part") != "thinking" and not x.get("cut")]
        groups += [[(s, x)] for x in rng.sample(th, min(n_thoughts, len(th)))]
    rng.shuffle(groups)
    rows, key = [], {}
    for g in groups:
        for s, x in g:
            n = len(rows) + 1
            i = f"b{batch}-{n:04d}"
            rows.append(dict(x, id=i, batch=batch, n=n))
            key[i] = {"side": s, "day": x["day"], "step": x.get("step")}
    with open(out, "x") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(key_path, "x") as f:
        json.dump({"A": a_path, "B": b_path, "items": key}, f, indent=1)
    print(json.dumps({"items": len(rows), "reply_pairs": sum(len(g) == 2 for g in groups),
                      "thoughts": sum(len(g) == 1 for g in groups)}))
