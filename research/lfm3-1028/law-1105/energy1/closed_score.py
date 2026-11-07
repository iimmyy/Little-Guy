#!/usr/bin/env python3
"""Scores GOAL-closed-v1. usage: closed_score.py <rows.jsonl> <first> <last>"""
import collections, json, statistics as st, sys

rows = [json.loads(l) for l in open(sys.argv[1])]
a, b = int(sys.argv[2]), int(sys.argv[3])
by = collections.defaultdict(list)
for r in rows:
    by[r["L"]].append(r)
N = max(by) + 1
med = lambda L, k: st.median(x[k] for x in by[L])
print("states", len(by[0]))
print(" L closed |  R     Rp   R<Rp  R/Rp |   Ac   shuf |  Ro    Rop  Ro>Rop |   Ao   shuf |  rr")
w1, w2, w3, w3r, w4 = [], [], [], [], []
for L in range(N):
    R = by[L]
    n = len(R)
    rc = [x["Ec1"] / x["Ec0"] for x in R]
    rp = [x["Ecp"] / x["Ec0"] for x in R]
    ro = [x["Eo1"] / max(x["Eo0"], 1e-9) for x in R]
    rop = [x["Eop"] / max(x["Eo0"], 1e-9) for x in R]
    below = sum(p < q for p, q in zip(rc, rp)) / n
    ratio = st.median(p / q for p, q in zip(rc, rp))
    above = sum(p > q for p, q in zip(ro, rop)) / n
    Ac, Acp = med(L, "Ac"), med(L, "Acp")
    if a <= L <= b:
        w1.append(below)
        w2.append(ratio)
        w3.append(Ac)
        w3r.append(Ac / max(abs(Acp), 1e-6))
        w4.append(above)
    print(f"{L:2d}  {med(L,'frac_closed'):.2f} | {st.median(rc):.3f} {st.median(rp):.3f} {below:.2f} {ratio:.3f} | {Ac:+.3f} {Acp:+.3f} |"
          f" {st.median(ro):.3f} {st.median(rop):.3f} {above:.2f} | {med(L,'Ao'):+.3f} {med(L,'Aop'):+.3f} | {med(L,'rr'):.3f}")
c1, c2, c3, c4 = min(w1) >= 0.9, max(w2) <= 0.95, min(w3) >= 0.2 and min(w3r) >= 5, min(w4) >= 0.9
print(f"C1 R<Rp worst L{a}-L{b}: {min(w1):.2f} -> {'PASS' if c1 else 'FAIL'}")
print(f"C2 R/Rp worst: {max(w2):.3f} -> {'PASS' if c2 else 'FAIL'}")
print(f"C3 Ac worst {min(w3):+.3f}, ratio to shuffled worst {min(w3r):.1f} -> {'PASS' if c3 else 'FAIL'}")
print(f"C4 (separate) Ro>Rop worst: {min(w4):.2f} -> {'PASS' if c4 else 'FAIL'}")
print("REAL" if c1 and c2 and c3 else "NOT REAL")
