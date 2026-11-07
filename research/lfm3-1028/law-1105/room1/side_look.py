#!/usr/bin/env python3
"""Print a seeded sample of side word rows for reading by hand. usage: side_look.py <rows> <n> <seed> [chars]"""
import collections, json, random, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
n, seed = int(sys.argv[2]), int(sys.argv[3])
chars = int(sys.argv[4]) if len(sys.argv) > 4 else 220
by = collections.defaultdict(dict)
for r in rows:
    by[r["step"]][r["arm"]] = r
steps = sorted(by)
random.Random(seed).shuffle(steps)
for s in steps[:n]:
    A = by[s]["A"]
    print(f"\n### step {s} problem {A['id']} wrote {A['written']!r} right {A['right']!r} tokens a={A['a']!r} b={A['b']!r} o={A['o']!r}")
    for arm in "ABC":
        print(f"[{arm}] " + by[s][arm]["text"][:chars].replace("\n", " / "))
