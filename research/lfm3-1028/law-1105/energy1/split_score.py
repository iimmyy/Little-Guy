#!/usr/bin/env python3
"""Scores GOAL-split-v1. usage: split_score.py <rows.jsonl> <first> <last>"""
import collections, json, statistics as st, sys

rows = [json.loads(l) for l in open(sys.argv[1])]
a, b = int(sys.argv[2]), int(sys.argv[3])
by = collections.defaultdict(list)
for r in rows:
    by[r["L"]].append(r)
N = max(by) + 1
med = lambda L, k: st.median(x[k] for x in by[L])
print("states", len(by[0]))
print(" L open | level trace: open+str open+weak closed+str closed+weak | gate open closed | closed share of neg | Eo up  Ec down | cos(o,c) size o, c")
shares, Eo, Ec, cs = [], [], [], []
for L in range(N):
    R = by[L]
    n = len(R)
    neg = [sum(max(0, -x[k]) for k in ("t_lev_os", "t_lev_ow", "t_lev_cs", "t_lev_cw")) for x in R]
    cl = [max(0, -x["t_lev_cs"]) + max(0, -x["t_lev_cw"]) for x in R]
    sh = st.median(c / max(v, 1e-9) for c, v in zip(cl, neg))
    up = sum(x["Eo1"] > x["Eo0"] for x in R) / n
    dn = sum(x["Ec1"] < x["Ec0"] for x in R) / n
    if a <= L <= b:
        shares.append(sh)
        Eo.append(up)
        Ec.append(dn)
        cs.append(med(L, "cos_oc"))
    print(f"{L:2d} {med(L,'frac_open'):.2f} | {med(L,'t_lev_os'):+.3f} {med(L,'t_lev_ow'):+.3f} {med(L,'t_lev_cs'):+.3f} {med(L,'t_lev_cw'):+.3f} |"
          f" {med(L,'t_gat_o'):+.3f} {med(L,'t_gat_c'):+.3f} | {sh:.2f} | {up:.2f} {dn:.2f} | {med(L,'cos_oc'):+.3f} {med(L,'size_o'):.2f} {med(L,'size_c'):.2f}")
q1 = min(shares) >= 0.7
q2 = min(Eo) >= 0.9 and min(Ec) >= 0.9
q3 = max(cs) < -0.2
print(f"Q1 closed share of negative level trace, worst L{a}-L{b}: {min(shares):.2f} -> {'PASS' if q1 else 'FAIL'}")
print(f"Q2 open energy rises, worst {min(Eo):.2f}; closed energy falls, worst {min(Ec):.2f} -> {'PASS' if q2 else 'FAIL'}")
print(f"Q3 cos(open, closed) highest {max(cs):+.3f} -> {'PASS' if q3 else 'FAIL'}")
print("EXPLAINED" if q1 and q2 and q3 else "NOT EXPLAINED")
