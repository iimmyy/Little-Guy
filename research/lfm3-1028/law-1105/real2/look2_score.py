#!/usr/bin/env python3
"""Scores GOAL-look2-v1. usage: look2_score.py <rows> [more rows ...]"""
import collections, json, re, sys
NUM = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")
rows = []
for f in sys.argv[1:]:
    rows += [json.loads(l) for l in open(f)]
def close(x, v, ref):
    d = ref.replace(",", "")
    ulp = 10 ** (-len(d.split(".")[1])) if "." in d else 1
    return abs(x - v) <= 2 * ulp or abs(x - v) <= 0.0005 * abs(v)
def race(r, arm):
    text = r["reads"][arm]
    if text is None:
        return "none"
    pg = float(r["page"].replace(",", ""))
    for m in NUM.finditer(text):
        try:
            x = float(m.group().replace(",", ""))
        except ValueError:
            continue
        if close(x, r["true"], r["page"]):
            return "true"
        if close(x, pg, r["page"]):
            return "stays"
    return "neither"
T = [r for r in rows if r["set"] == "believed"]
C = [r for r in rows if r["set"] == "right"]
print(f"believed wrong: {len(T)} steps, {len({r['id'] for r in T})} problems; right controls: {len(C)}")
for name, S in (("believed wrong", T), ("right controls", C)):
    print(name)
    for arm in ("plain", "now", "hmm", "wait"):
        c = collections.Counter(race(r, arm) for r in S)
        print(f"  {arm:<6} lands on true {c['true']}, stays {c['stays']}, neither {c['neither']}" + (f" | {c['true']/len(S)*100:.1f}% true" if S else ""))
if len(T) < 20:
    print("fewer than 20 believed mistakes: this cannot answer")
n = max(len(T), 1)
w, nw = [sum(race(r, a) == "true" for r in T) / n for a in ("wait", "now")]
v = "A DOUBT WORD REACHES IT" if w >= .4 and w - nw >= .2 else "ANY SECOND LOOK REACHES IT" if w >= .4 and nw >= .4 else "THE SENTENCE HOLDS, HE NEEDS OUTSIDE INPUT" if w <= .15 and nw <= .15 else "IN BETWEEN"
print(f"\nBAR 1: wait {w*100:.1f}%, now {nw*100:.1f}%: {v}")
leaves = sum(race(r, "wait") != "stays" for r in T) / n
fa = sum(race(r, "wait") != "true" for r in C) / max(len(C), 1)
print(f"BAR 2: wait leaves his number on {leaves*100:.1f}% of believed mistakes, leaves it on {fa*100:.1f}% of right controls: {'ONE READ DOES BOTH' if leaves >= .5 and fa <= .03 else 'NOT YET'}")
print("\nbelieved mistakes, step by step: expr page | true | p | plain | now | hmm | wait")
for r in T:
    print(f"  {r['step']} {r['id'][:22]} {r['expr'].strip()} {r['page']} | {r['true']:.6g} | {r['minp']:.2f} | " + " | ".join(race(r, a) for a in ("plain", "now", "hmm", "wait")))
