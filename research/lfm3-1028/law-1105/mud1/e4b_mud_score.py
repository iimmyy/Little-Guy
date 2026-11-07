#!/usr/bin/env python3
"""GOAL-mud-v1, E4B reading alone. usage: e4b_mud_score.py <dir> <prefix>"""
import json, sys, os
import numpy as np
d, p = sys.argv[1:3]
R = [json.loads(l) for l in open(os.path.join(d, p + "-rows.jsonl"))]
z = np.load(os.path.join(d, p + "-votes.npz"), allow_pickle=True)
V, order = z["V"], [str(x) for x in z["order"]]
print("keys", list(z.keys()), "V", V.shape, "first labels", order[:7], "last", order[-4:])
s, t = np.abs(V.sum(1)), np.abs(V).sum(1)
M = t / np.maximum(s, 1e-3)
gap = np.array([r["gap"] for r in R]); miss = np.abs(s - gap)
print(f"{p}: {len(R)} words, {len({r['step'] for r in R})} pages, {len({r['id'] for r in R})} problems")
print(f"vote check median miss {np.median(miss):.3f} nats, p90 {np.percentile(miss,90):.3f}; stream_miss median {np.median([r['stream_miss'] for r in R]):.4f}, max {max(r['stream_miss'] for r in R):.4f}")
print(f"shouting median {np.median(t):.1f}, margin median {np.median(s):.2f}, M1 median {np.median(M):.2f} (p10 {np.percentile(M,10):.2f}, p90 {np.percentile(M,90):.1f})")
print(f"S median {np.median([r['S'] for r in R]):.3f} (p10 {np.percentile([r['S'] for r in R],10):.3f}, p90 {np.percentile([r['S'] for r in R],90):.3f})")
tot = np.abs(V).sum()
for nm, pre in (("attention","op"),("neurons","ffn"),("per-layer","per")):
    I = [i for i, n in enumerate(order) if n.startswith(pre)]
    sub = V[:, I]
    print(f"  {nm:<10} share of shouting {np.abs(sub).sum()/tot:.3f}, own M1 {np.median(np.abs(sub).sum(1)/np.maximum(np.abs(sub.sum(1)),1e-3)):.2f}, mean net vote {sub.sum(1).mean():+.2f}")
sz = np.array([r["sizes"] for r in R]); m = np.median(sz, 0)
print("write size at the end, median, per layer op/ffn/per, every 6th layer:")
lab = [n for n in order if n[:2] in ("op","ff","pe")]
idx = {n:i for i,n in enumerate(lab)}
print("sizes len", sz.shape, "labels", len(lab))
for L in (0,6,12,18,24,30,36,39,40,41):
    row=[]
    for pre in ("op","ffn","per"):
        n=f"{pre}{L}"
        row.append(f"{pre} {m[idx[n]]:.4f}" if n in idx and idx[n]<len(m) else f"{pre} -")
    print(f"  L{L:<2}", "  ".join(row))
sh = np.abs(V).mean(0)
ord2 = np.argsort(-sh)[:10]
print("loudest votes:", [(order[i], round(float(sh[i]),2)) for i in ord2])
print("verdict by bar: E4B HAS MUD TOO" if np.median(M) >= 3 else "verdict by bar: E4B IS CLEANER")
