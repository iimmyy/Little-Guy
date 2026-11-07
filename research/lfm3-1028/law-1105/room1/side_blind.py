#!/usr/bin/env python3
"""Makes the blind reading sheet for GOAL-carry-v1. usage: side_blind.py <rows> <steps> <spent sample> <sheet out> <key out>"""
import collections, json, random, re, sys
rows_f, steps_f, spent_f, sheet, key = sys.argv[1:6]
by = collections.defaultdict(dict)
for l in open(rows_f):
    r = json.loads(l)
    by[r["step"]][r["arm"]] = r
pre = {}
for l in open(steps_f):
    j = json.loads(l)
    pre[j["step"]] = j["pre"]
spent = {int(x) for x in re.findall(r"### step (\d+)", open(spent_f).read())}
left = sorted(set(by) - spent)
rng = random.Random(1106)
pick = rng.sample(left, 60)
out, k = [], {}
for s in pick:
    A = by[s]["A"]
    other = A["written"].replace(A["b"], A["o"], 1)
    arms = list("ABC")
    rng.shuffle(arms)
    k[str(s)] = dict(zip("XYZ", arms))
    out.append(f"\n### {s} | before: ...{pre[s][-120:]!r} | written {A['written']} | right {A['right']} | other {other}")
    for lab, arm in zip("XYZ", arms):
        out.append(f"[{lab}] " + by[s][arm]["text"][:260].replace("\n", " / "))
open(sheet, "w").write("\n".join(out) + "\n")
json.dump(k, open(key, "w"))
print(f"left {len(left)}, picked {len(pick)}, spent {len(spent)}")
