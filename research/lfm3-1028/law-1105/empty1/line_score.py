#!/usr/bin/env python3
"""usage: line_score.py <rows>"""
import json, sys
import numpy as np
rows = [json.loads(l) for l in open(sys.argv[1])]
P = np.array([r["id"] for r in rows]); u = np.unique(P); gp = {p: np.where(P == p)[0] for p in u}
rng = np.random.default_rng(1105)
M = lambda k: np.array([r["reads"][k] for r in rows])
def stat(x):
    v = [np.median(x[np.concatenate([gp[p] for p in rng.choice(u, len(u))])]) for _ in range(2000)]
    return float(np.median(x)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
print(f"steps {len(rows)}, problems {len(u)}")
K, C = (M("plain:R") + M("plain:W")) / 2, (M("plain:R") - M("plain:W")) / 2
ok = True
for phi in (0.25, 0.75):
    pred = K + (2 * phi - 1) * C
    e = M(f"phi{phi}:W") - pred
    r = np.corrcoef(M(f"phi{phi}:W") - K, (2 * phi - 1) * C)[0, 1]
    good = np.median(np.abs(e)) <= 1 and r >= 0.7
    ok &= good
    print(f"phi {phi}: median |error| {np.median(np.abs(e)):.2f}, median error {np.median(e):+.2f}, correlation of (m - K) with (2 phi - 1) C {r:.3f} -> {'met' if good else 'not met'}")
print("LINE:", "STRAIGHT LINE" if ok else "NOT A LINE")
r0 = np.corrcoef(K, M("m0"))[0, 1]
print(f"K vs m0: correlation {r0:.3f}; medians K {np.median(K):+.2f}, m0 {np.median(M('m0')):+.2f}; K > m0 on {(K>M('m0')).mean()*100:.0f}% of steps -> {'K IS WHAT HE KNOWS' if r0 >= 0.7 else 'K IS SOMETHING ELSE'}")
for name in ("wait", "yes"):
    a, b = M(f"mark{name}:R"), M(f"mark{name}:W")
    dk, dc = stat((a + b) / 2 - K), stat((a - b) / 2 - C)
    print(f"wordless {name}, phi 0.25: change in K {dk[0]:+.2f} ({dk[1]:+.2f} to {dk[2]:+.2f}), change in C {dc[0]:+.2f} ({dc[1]:+.2f} to {dc[2]:+.2f})")
