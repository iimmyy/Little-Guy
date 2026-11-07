#!/usr/bin/env python3
"""Scores GOAL-energy-v1 from the rows. usage: energy_score.py <rows.jsonl> <flip layer>"""
import collections, json, statistics as st, sys

rows = [json.loads(l) for l in open(sys.argv[1])]
flip = int(sys.argv[2])
by = collections.defaultdict(list)
for r in rows:
    by[r["L"]].append(r)
N = max(by) + 1
med = lambda L, k: st.median(x[k] for x in by[L])
print(f"states {len(by[0])}  max rebuild error {max(r['rebuilt'] for r in rows):.4f}")
print(" L  fracW shareW |      A   shuf | dS<0  w:dS<0 s:dS<0 |    sf    tr    lam    rr")
for L in range(N):
    R = by[L]
    n = len(R)
    print(f"{L:2d}  {med(L,'frac_weak'):.2f}   {med(L,'share_weak'):.2f}  | {med(L,'A'):+.3f} {med(L,'Ap'):+.3f} |"
          f" {sum(x['S1']<x['S0'] for x in R)/n:.2f}   {sum(x['S1w']<x['S0w'] for x in R)/n:.2f}   {sum(x['S1s']<x['S0s'] for x in R)/n:.2f}  |"
          f" {med(L,'sf'):.3f} {med(L,'tr'):+.3f} {med(L,'lam'):+.3f} {med(L,'rr'):.3f}")
mid = [L for L in range(N) if round(N / 3) <= L <= flip - 3]
late = [L for L in range(flip + 1, N)]
last = N - 1
print("mid layers", mid, "late layers", late)
Amid = [med(L, "A") for L in mid]
Alate = [med(L, "A") for L in late]
cross = next((L for L in range(N) if med(L, "A") > 0 and all(med(q, "A") > 0 for q in range(L, N))), None)
p1 = all(a < 0 for a in Amid) and all(a > 0 for a in Alate) and cross is not None and abs(cross - flip) <= 1
print(f"P1 A negative mid, positive late, crossing within 1 of L{flip}: crossing at {cross} -> {'PASS' if p1 else 'FAIL'}")
dn = sum(x["S1"] < x["S0"] for x in by[last]) / len(by[last])
up = min(sum(x["S1"] > x["S0"] for x in by[L]) / len(by[L]) for L in mid)
p2 = dn >= 0.9 and up >= 0.9
print(f"P2 dS<0 at last layer {dn:.2f}; worst mid layer dS>0 {up:.2f} -> {'PASS' if p2 else 'FAIL'}")
Al, Ap = med(last, "A"), abs(med(last, "Ap"))
p3 = Al >= 0.20 and Al >= 5 * Ap
print(f"P3 last layer A {Al:+.3f}, shuffled {Ap:.3f} -> {'PASS' if p3 else 'FAIL'}")
print("GOAL:", "PASS" if p1 and p2 and p3 else "FAIL")
lams = [med(L, "lam") for L in range(N)]
lc = next((L for L in range(N) if lams[L] < 0 and all(v < 0 for v in lams[L:])), None)
print(f"P4 (description) lam mid {st.median(lams[L] for L in mid):+.3f}, last {lams[last]:+.3f}, stays negative from {lc}")
sfm, sfl = st.median(med(L, "sf") for L in mid), st.median(med(L, "sf") for L in range(flip, N))
print(f"P5 (description) sf mid {sfm:.3f}, L{flip}+ {sfl:.3f}")
for kind in ("close", "word"):
    R = [x for x in by[last] if x["kind"] == kind]
    print(f"last layer, {kind}: n {len(R)}, A {st.median(x['A'] for x in R):+.3f}, dS<0 {sum(x['S1']<x['S0'] for x in R)/max(1,len(R)):.2f}, lam {st.median(x['lam'] for x in R):+.3f}")
