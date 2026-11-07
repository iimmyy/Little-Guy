#!/usr/bin/env python3
"""Was edit_read.py's first result (AUROC 0.999) real? Immy 2026-11-03: check before throwing it out. Its reads took two paths:
clean and ordinary lines from one full forward, edits from a cut one. v2 reads all of them the cut way. So, v1's own lines, untouched:
- on v2's same-path reads, held-out problems. High here = the v1 line was genuinely good; near 0.5 = it was reading the path.
- the path itself: v1's clean reads (full path) against v2's clean reads (cut path), which should read alike if the path does not matter.
usage: edit_check.py <v1.npz> <v1-lines.npz> <v2.npz> <out.txt>
"""
import hashlib, sys
import numpy as np


def auroc(pos, neg):
    allv = np.concatenate([pos, neg]); r = np.empty(len(allv)); r[np.argsort(allv)] = np.arange(1, len(allv) + 1)
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


v1, ln, v2 = np.load(sys.argv[1]), np.load(sys.argv[2]), np.load(sys.argv[3])
held = lambda ids: np.array([int(hashlib.md5(i.encode()).hexdigest(), 16) % 3 == 2 for i in ids])
L = []
for form, alarm in (("abs", lambda r: r), ("spring", lambda r: r ** 2)):
    sc = lambda X: alarm(((X.astype(np.float64) - ln["mu"]) / ln["sd"]) @ ln[form])
    h1e, h1o, h2e, h2o = held(v1["pe"]), held(v1["po"]), held(v2["pe"]), held(v2["po"])
    a1c, a2c, a2e, a2o, a1o = sc(v1["Xc"][h1e]), sc(v2["Xc"][h2e]), sc(v2["Xe"][h2e]), sc(v2["Xo"][h2o]), sc(v1["Xo"][h1o])
    L.append(f"v1 {form} line:")
    L.append(f"  on same-path reads: AUROC edited vs clean {auroc(a2e, a2c):.3f}, vs ordinary {auroc(a2e, a2o):.3f}; edited higher in {np.mean(a2e > a2c):.0%} of pairs (n {len(a2e)})")
    for fa in (0.01, 0.02, 0.05, 0.1):
        th = np.quantile(a2o, 1 - fa)
        L.append(f"    alarm over the {fa:.0%} line of ordinary lines: catches {np.mean(a2e > th):.0%} of edits, fires on {np.mean(a2c > th):.0%} of clean copies")
    L.append(f"  the path: clean reads, full path median {np.median(a1c):.3f} vs cut path {np.median(a2c):.3f} (AUROC cut vs full {auroc(a2c, a1c):.3f}); ordinary full {np.median(a1o):.3f} vs cut {np.median(a2o):.3f} (AUROC {auroc(a2o, a1o):.3f})")
open(sys.argv[4], "w").write("\n".join(L) + "\n")
print("\n".join(L))
