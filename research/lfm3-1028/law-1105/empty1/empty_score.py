#!/usr/bin/env python3
"""Scores GOAL-empty-v1. usage: empty_score.py <rows.jsonl> [bar]"""
import json, sys
import numpy as np
rows = [json.loads(l) for l in open(sys.argv[1])]
print(f"steps {len(rows)}, problems {len({r['id'] for r in rows})}")
P = np.array([r["id"] for r in rows])
u = np.unique(P)
gp = {p: np.where(P == p)[0] for p in u}
rng = np.random.default_rng(1105)
M = lambda k: np.array([r["reads"][k][0] for r in rows])
def stat(x):
    v = [np.median(x[np.concatenate([gp[p] for p in rng.choice(u, len(u))])]) for _ in range(2000)]
    return float(np.median(x)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
print(f"plain reads, logit right - logit wrong: right on page median {np.median(M('plain:R')):+.2f}, wrong on page median {np.median(M('plain:W')):+.2f}; at wrong page he favours the right token on {(M('plain:W')>0).mean()*100:.0f}% of steps")
D = {}
for a in ("so", "now", "wait", "hmm", "actually", "yes", "right", "wrong"):
    D[a] = (stat(M(a + ":R") - M("plain:R")), stat(M(a + ":W") - M("plain:W")))
    r, w = D[a]
    print(f"{a:<9} d_R {r[0]:+.2f} ({r[1]:+.2f} to {r[2]:+.2f})   d_W {w[0]:+.2f} ({w[1]:+.2f} to {w[2]:+.2f})")
s = stat(M("side:W") - M("plain:W"))
print(f"side word, wrong page: d_W {s[0]:+.2f} ({s[1]:+.2f} to {s[2]:+.2f})  <- real input, no words")
if len(sys.argv) > 2:
    print()
    if len(rows) < 100:
        print("fewer than 100 steps: cannot answer")
    off = lambda x: x[1] > 0 or x[2] < 0
    e = [D["so"][0], D["so"][1], D["now"][0], D["now"][1]]
    if all(abs(x[0]) < 0.25 for x in e):
        print("EMPTY: FIRST LAW HOLDS FOR EMPTY WORDS")
    elif any(abs(x[0]) >= 0.5 and off(x) for x in e):
        print("EMPTY: EMPTY WORDS MOVE HIM")
    else:
        print("EMPTY: SMALL")
    for a in ("wait", "hmm", "actually"):
        r, w = D[a]
        if w[0] >= 0.5 and off(w) and r[0] >= -0.25:
            v = "LOOKS AGAIN"
        elif w[0] >= 0.5 and off(w) and r[0] <= -0.5 and off(r):
            v = "BLIND PUSH"
        elif abs(w[0]) < 0.25 and abs(r[0]) < 0.25:
            v = "NOTHING"
        else:
            v = "MIXED"
        print(f"{a}: {v}")


K = lambda a: (M(a + ":R") + M(a + ":W")) / 2
C = lambda a: (M(a + ":R") - M(a + ":W")) / 2
print("\nK what he knows, C how much he goes by the page")
print(f"plain: K median {np.median(K('plain')):+.2f}, C median {np.median(C('plain')):+.2f}")
for a in ("so", "now", "wait", "hmm", "actually", "yes", "right", "wrong"):
    dk, dc = stat(K(a) - K("plain")), stat(C(a) - C("plain"))
    print(f"{a:<9} change in K {dk[0]:+.2f} ({dk[1]:+.2f} to {dk[2]:+.2f})   change in C {dc[0]:+.2f} ({dc[1]:+.2f} to {dc[2]:+.2f})")
err = M("side:W") - K("plain")
print(f"prediction 1, side word reads K: median |error| {np.median(np.abs(err)):.2f} nats, median error {np.median(err):+.2f}, correlation {np.corrcoef(M('side:W'), K('plain'))[0,1]:.3f}; compare plain wrong page vs K correlation {np.corrcoef(M('plain:W'), K('plain'))[0,1]:.3f}")
