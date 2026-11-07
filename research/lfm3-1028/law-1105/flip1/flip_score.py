#!/usr/bin/env python3
"""Score the final-answer flips against GOAL-flip-v1 (base = audited label). usage: flip_score.py <rows.jsonl> [corrections.json]
Corrections: {"id:arm": "right" or "wrong"}, from reading each flip by hand. Writes flips-to-read-v1.txt beside this script."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "science", "grades-v4"))
from keyed_grade import keyed_grade
CLOSE = "<channel|>"
rows = [json.loads(l) for l in open(sys.argv[1])]
fix = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else {}
def parts(r):
    t = r["text"] or ""
    return tuple(t.split(CLOSE, 1)) if CLOSE in t else (t, "")
def g(r):
    th, rep = parts(r)
    if not r["done"] or not rep:
        return "cut"
    key = f"{r['id']}:{r['arm']}"
    if key in fix:
        return fix[key]
    return keyed_grade("stop", rep, r["id"], r["answer"], r.get("scale"))
print("rows", len(rows))
res = {}
for s, want, flip in (("wrong", "wrong", "right"), ("right", "right", "wrong")):
    for arm in ("now", "wait"):
        x = [r for r in rows if r["set"] == s and r["arm"] == arm]
        gr = [g(r) for r in x]; n = len(x)
        f = sum(v == flip for v in gr); c = sum(v == "cut" for v in gr); o = sum(v not in (want, flip, "cut") for v in gr)
        th = sorted(len(parts(r)[0].split()) for r in x)
        res[(s, arm)] = f / max(n, 1)
        print(f"{s:5s} {arm:4s} flips {f}/{n} = {100*f/max(n,1):.1f}%  not finished {c}  other grade {o}  extra thinking words median {th[len(th)//2] if th else 0} max {th[-1] if th else 0}")
ww, wr, nw, nr = res[("wrong","wait")], res[("right","wait")], res[("wrong","now")], res[("right","now")]
if ww <= .05: v = "THE SENTENCE HOLDS AT THE FINAL ANSWER"
elif ww >= .15 and ww >= 2*wr and ww >= nw + .10: v = "AN EMPTY DOUBT WORD REACHES WHAT HE HAD"
elif ww >= .15 and ww < 2*wr: v = "A BLIND COIN"
else: v = "IN BETWEEN"
print("WAIT:", v)
print("NOW:", "AN EMPTY WORD LEAVES IT ALONE" if nw <= .05 and nr <= .05 else "AN EMPTY WORD MOVES THE ANSWER")
with open(os.path.join(HERE, "flips-to-read-v1.txt"), "w") as w:
    for r in rows:
        want, flip = ("wrong", "right") if r["set"] == "wrong" else ("right", "wrong")
        v = g(r)
        if v in (want, "cut"):
            continue
        th, rep = parts(r)
        w.write(f"=== {r['id']}:{r['arm']} set {r['set']} machine -> {v} key {r['answer']} scale {r.get('scale')}\nCONT ({len(th.split())} words): {th[:1800]!r}\n...END OF CONT: {th[-400:]!r}\nREPLY END: {rep[-400:]!r}\n\n")
