#!/usr/bin/env python3
"""Paired comparison of two brains on the same generation questions (gen_bench.sh samples).

For each task: how many questions only A gets right, only B, and the exact two-sided sign-test (McNemar) p-value on
those disagreements. A small p means the gap is unlikely to be chance; with both brains right or wrong on most
questions, a few points of difference often isn't.
usage: gen_paired.py <gen out dir> <brain a> <brain b>
"""
import json, sys
from math import comb
from pathlib import Path

root, a, b = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
SCORE = {"gsm8k_cot": ("exact_match", "flexible-extract"), "humaneval_instruct": ("pass@1", None),
         "ifeval": ("prompt_level_strict_acc", None)}


def load(brain, task):
    fs = sorted((root / f"{brain}-short").glob(f"*/samples_{task}_*.jsonl"))
    if not fs:
        return None
    key, flt = SCORE[task]
    out = {}
    for line in open(fs[-1]):
        d = json.loads(line)
        if flt and d.get("filter") != flt:
            continue
        out[d["doc_id"]] = float(d[key]) >= 0.5
    return out


def sign_p(x, y):
    n, k = x + y, min(x, y)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


print(f"{a} vs {b}: questions right only for {a} / only for {b}, sign-test p")
for task in SCORE:
    ra, rb = load(a, task), load(b, task)
    if not ra or not rb:
        print(f"  {task:20} missing"); continue
    ids = ra.keys() & rb.keys()
    x = sum(ra[i] and not rb[i] for i in ids); y = sum(rb[i] and not ra[i] for i in ids)
    print(f"  {task:20} n={len(ids):5d}  {a} {sum(ra[i] for i in ids):5d} right, {b} {sum(rb[i] for i in ids):5d}"
          f"   only {a} {x:4d} / only {b} {y:4d}   p={sign_p(x, y):.3f}")
