#!/usr/bin/env python3
"""GOAL-f5song-v1. usage: f5song_score.py <dir with new rows> <name> [corrections.json]. Corrections: {"run:id": "right" or "wrong"}"""
import json, os, sys
import numpy as np
BASE = os.environ["LITTLEGUY_ROOT"] + "/.run/results/mind/lfm3-1028/science"
sys.path.insert(0, BASE + "/grades-v4")
from keyed_grade import keyed_grade
d, name = sys.argv[1:3]
fix = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else {}
RUNS = ("g", "s1", "s2", "s3")
ids = [json.loads(l)["id"] for l in open(os.path.join(d, "sci-held-81.jsonl"))]
def load(path):
    return {r["id"]: r for r in map(json.loads, open(path))}
def G(r, run=None):
    k = f"{run}:{r['id']}"
    if run and k in fix:
        return fix[k]
    x = keyed_grade(r.get("finish"), r.get("reply"), r["id"], r["answer"], r.get("scale"))
    return "wrong" if x == "sign" else x
A = {u: load(f"{BASE}/final-f5-1104/f5-{u}.jsonl") for u in RUNS}
B = {u: load(os.path.join(d, f"{name}-{u}.jsonl")) for u in RUNS}
ok = [i for i in ids if all(i in A[u] and i in B[u] for u in RUNS)]
print("problems in both", len(ok), "of", len(ids), "runs", len(ok) * 4)
T = lambda r: (r.get("think_tokens") or 0) + (r.get("reply_tokens") or 0)
def table(S, label, run_fix):
    c = {"right": 0, "wrong": 0, "cut": 0}; tok = 0; by = []
    for u in RUNS:
        n = 0
        for i in ok:
            g = G(S[u][i], u if run_fix else None); c[g if g in c else "wrong"] += 1; tok += T(S[u][i]); n += g == "right"
        by.append(n)
    N = len(ok) * 4
    print(f"{label:<10} right {c['right']} ({100*c['right']/N:.1f}%) wrong {c['wrong']} ({100*c['wrong']/N:.1f}%) cut {c['cut']} ({100*c['cut']/N:.1f}%) tokens {tok} right by run {by}")
    return c, tok
ca, ta = table(A, "f5", False)
cb, tb = table(B, name, False)
if fix:
    table(B, name + " hand", True)
P = np.array(ok); rng = np.random.default_rng(1105)
def diff(kind):
    x = np.array([sum(G(B[u][i]) == kind for u in RUNS) - sum(G(A[u][i]) == kind for u in RUNS) for i in ok])
    v = [x[rng.integers(0, len(x), len(x))].sum() for _ in range(4000)]
    return int(x.sum()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
for k in ("right", "wrong", "cut"):
    print(f"change in {k}: {diff(k)}")
tk = np.array([sum(T(B[u][i]) for u in RUNS) - sum(T(A[u][i]) for u in RUNS) for i in ok]); v = [tk[rng.integers(0, len(tk), len(tk))].sum() / ta for _ in range(4000)]
print(f"tokens {100*(tb/ta-1):+.1f}% ({100*np.percentile(v,2.5):+.1f} to {100*np.percentile(v,97.5):+.1f})")
dr, dw = cb["right"] - ca["right"], cb["wrong"] - ca["wrong"]
if dr <= -16 or dw >= 16: t = "IT HURTS"
elif dr >= 16 and dw <= -16: t = "BOTH MOVE THE RIGHT WAY"
elif dr >= 16: t = "MORE RIGHT"
elif dw <= -16: t = "FEWER WRONG"
else: t = "NOTHING"
print("VERDICT:", t)
N = len(ok) * 4
print(f"target: right 80% {'reached' if cb['right']/N >= .8 else 'not reached'}, wrong under 10% {'reached' if cb['wrong']/N < .1 else 'not reached'}")
with open(os.path.join(d, f"wrongs-to-read-{name}-v1.txt"), "w") as w:
    for u in RUNS:
        for i in ok:
            r = B[u][i]
            if G(r) == "wrong":
                w.write(f"=== {u}:{i} key {r['answer']} scale {r.get('scale')} tokens {T(r)} f5 same run {G(A[u][i])}\nQ: {r['q'][:300]!r}\nEND: {(r.get('reply') or '')[-500:]!r}\n\n")
