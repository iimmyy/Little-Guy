#!/usr/bin/env python3
"""Which brain Immy picked in a blind batch (ab_batch.py): his yes and no, unblinded with the batch's key.
  - yes rate of each brain, on replies and on quiet thoughts
  - the pairs (both brains answering the same message): how often only A got a yes, only B, both, neither
  - a sign test on the pairs where he split them (one yes, one no): how likely a split this lopsided is by chance
usage: ab_verdict.py KEY.json LABELS.jsonl [LABELS2.jsonl ...]   (the latest answer per item wins)
"""
import json, math, sys


def sign_test(a, b):
    """two-sided: chance of a split at least this uneven if neither brain were better"""
    n, k = a + b, min(a, b)
    if n == 0:
        return None
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return round(min(1.0, 2 * p), 4)


if __name__ == "__main__":
    key = json.load(open(sys.argv[1]))["items"]
    latest = {}
    for f in sys.argv[2:]:
        for l in open(f):
            if l.strip():
                x = json.loads(l)
                if x["id"] in key and (x["id"] not in latest or x["t"] >= latest[x["id"]]["t"]):
                    latest[x["id"]] = x
    items = {}
    for f in [sys.argv[1].replace("key-", "items-").replace(".json", ".jsonl")]:
        for l in open(f):
            x = json.loads(l)
            items[x["id"]] = x
    rate = {s: {"reply": [0, 0], "thought": [0, 0]} for s in ("A", "B")}
    pairs = {}
    for i, lab in latest.items():
        side, it = key[i]["side"], items[i]
        r = rate[side][it["kind"]]
        r[0] += lab["label"] == "yes"; r[1] += 1
        if it["kind"] == "reply":
            same_day = it["day"].split("-", 1)[1]                   # "B-3100/day-001" -> "3100/day-001"
            pairs.setdefault((it["heard"], same_day), {})[side] = lab["label"]
    split = {"only A": 0, "only B": 0, "both yes": 0, "both no": 0}
    for p in pairs.values():
        if len(p) == 2:
            a, b = p["A"] == "yes", p["B"] == "yes"
            split["both yes" if a and b else "only A" if a else "only B" if b else "both no"] += 1
    out = {"judged": len(latest),
           "yes rate": {s: {k: f"{v[0]}/{v[1]}" for k, v in d.items()} for s, d in rate.items()},
           "pairs judged both": sum(split.values()), **split,
           "chance of a split this uneven": sign_test(split["only A"], split["only B"])}
    print(json.dumps(out, indent=1))
