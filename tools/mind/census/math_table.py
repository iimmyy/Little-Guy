#!/usr/bin/env python3
"""MATH-500 runs side by side on the problems every run has finished: right answers, speed vs untouched, cut-offs,
per level. A cut-off counts as the full 6000 words it burned. "12th grade" counts as the key's "12" (grader fix).

usage: math_table.py name=file.jsonl [name=file.jsonl ...]     (the first one is the baseline)
speed also gets a 90% range from 2000 resamples of the problems (paired: same problems for both runs), since a few
long answers swing the total.
"""
import json, os, random, sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../expert"))
ARGS, sys.argv[1:] = sys.argv[1:], []                     # grade.py grades its own arguments on import
from grade import right, answer_line

runs = {}
for a in ARGS:
    n, f = a.split("=", 1)
    runs[n] = {r["id"]: r for r in map(json.loads, open(f))}
W = lambda r: r["think_tokens"] + r.get("reply_tokens", 0) if r["finish"] == "stop" else 6000
ok = lambda r: bool(right(r, answer_line(r["reply"], True))) or (r["id"] == "test/prealgebra/1865.json" and "12" in answer_line(r["reply"], True))
base = next(iter(runs.values()))
ids = [i for i in base if all(i in R for R in runs.values())]
by = defaultdict(list)
for i in ids:
    by[base[i]["level"]].append(i)
print(f"{len(ids)} problems every run finished")
for n, R in runs.items():
    tot = sum(W(base[i]) for i in ids) / sum(W(R[i]) for i in ids)
    lv = " ".join(f"L{k} {sum(ok(R[i]) for i in v)}/{len(v)} {100 * (sum(W(R[i]) for i in v) / sum(W(base[i]) for i in v) - 1):+.0f}%"
                  for k, v in sorted(by.items()))
    rng, bs = random.Random(0), []
    for _ in range(2000):
        pick = [rng.choice(ids) for _ in ids]
        bs.append(sum(W(base[i]) for i in pick) / sum(W(R[i]) for i in pick))
    bs.sort()
    print(f"{n:12} right {sum(ok(R[i]) for i in ids):>3}  x{tot:.2f} ({bs[100]:.2f}-{bs[1899]:.2f})  cut {sum(R[i]['finish'] != 'stop' for i in ids)} | {lv}")
