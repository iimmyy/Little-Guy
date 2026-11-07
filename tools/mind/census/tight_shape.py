#!/usr/bin/env python3
"""Cause or effect? (Immy 2026-10-29: "the equation tightens when it's in flow state".) tightness.py found good runs
a little tighter on average. If he tightens BECAUSE he found his way, the gap shows up late (after the good run
has its answer and the bad run is lost). If tightness comes first, the gap is there from the opening sentences,
before anything about the outcome is decided. Reads the per-run series tightness.py saved and splits the thinking
into position windows; per window, per question: good runs' mean minus bad runs' mean, and how many questions have
the good runs tighter (lower = tighter for entropy and lock-in).

usage: tight_shape.py <tightness .pt or _partial.pt> [bins=0,100,250,500,1000,2000,3000,6000]
"""
import sys

import numpy as np
import torch

c = torch.load(sys.argv[1], weights_only=False)
opt = dict(a.split("=", 1) for a in sys.argv[2:])
EDGES = [int(x) for x in opt.get("bins", "0,100,250,500,1000,2000,3000,6000").split(",")]
res, SER = c["res"], c["series"]
qs = sorted({q for q, _ in res})

print(f"{len(qs)} questions, {len(res)} runs; good minus bad per position window (lower = good tighter)")
for meas in ("ent", "lock"):
    print(f"\n{'entropy (nats)' if meas == 'ent' else 'lock-in (share of depth)'}")
    print("  words        questions  good tighter   median good-bad   good mean / bad mean")
    for a, b in zip(EDGES, EDGES[1:]):
        d, gm, bm = [], [], []
        for q in qs:
            G, B = [], []
            for (qq, k), (g, _) in res.items():
                if qq != q:
                    continue
                s = SER[(qq, k)]
                m = (s["pos"] >= a) & (s["pos"] < b)
                if m.sum() < 3:                              # this run had not reached this window (or ended)
                    continue
                (G if g else B).append(float(s[meas][m].float().mean()))
            if G and B:
                d.append(np.mean(G) - np.mean(B)); gm += G; bm += B
        if d:
            print(f"  {a:5d}-{b:<5d}  {len(d):4d}       {sum(x < 0 for x in d):3d}/{len(d):<3d}        {np.median(d):+.4f}"
                  f"          {np.mean(gm):.4f} / {np.mean(bm):.4f}")
