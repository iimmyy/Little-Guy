#!/usr/bin/env python3
"""GOAL-mud-v1. usage: mud_score.py <dir> <base prefix> <new prefix>"""
import json, sys, os
import numpy as np
d, A, B = sys.argv[1:4]
def load(p):
    R = [json.loads(l) for l in open(os.path.join(d, p + "-rows.jsonl"))]
    z = np.load(os.path.join(d, p + "-votes.npz"), allow_pickle=True)
    return R, z["V"], [str(x) for x in z["order"]], [str(x) for x in z["types"]]
def reading(p):
    R, V, order, types = load(p)
    s, t = np.abs(V.sum(1)), np.abs(V).sum(1)
    M = t / np.maximum(s, 1e-3)
    gap = np.array([r["gap"] for r in R]); miss = np.abs(s - gap)
    print(f"\n{p}: {len(R)} words, {len({r['step'] for r in R})} pages, {len({r['id'] for r in R})} problems. vote check: median miss {np.median(miss):.3f} nats")
    print(f"  shouting median {np.median(t):.1f}, margin median {np.median(s):.2f}, M1 median {np.median(M):.2f} (p10 {np.percentile(M,10):.2f}, p90 {np.percentile(M,90):.1f})")
    print(f"  S stream survival median {np.median([r['S'] for r in R]):.3f}")
    conv = [i for i, n in enumerate(order) if n.startswith("op") and types[int(n[2:])] == "conv"]
    att = [i for i, n in enumerate(order) if n.startswith("op") and types[int(n[2:])] != "conv"]
    ffn = [i for i, n in enumerate(order) if n.startswith("ffn")]
    tot = np.abs(V).sum()
    for nm, I in (("conv", conv), ("attention", att), ("neurons", ffn)):
        sub = V[:, I]
        print(f"  {nm:<10} share of shouting {np.abs(sub).sum()/tot:.3f}, its own M1 median {np.median(np.abs(sub).sum(1)/np.maximum(np.abs(sub.sum(1)),1e-3)):.2f}, mean net vote {sub.sum(1).mean():+.2f}")
    C = np.array([r["cos"] for r in R])
    lab = order[1:]
    same = [i for i in range(len(lab) - 1) if lab[i].startswith("op")]
    nxt = [i for i in range(len(lab) - 1) if lab[i].startswith("ffn")]
    print(f"  cosine, operator to its own neurons {np.median(C[:, same]):+.3f}; neurons to next layer's operator {np.median(C[:, nxt]):+.3f}")
    ff = [i for i, n in enumerate(lab) if n.startswith("ffn")]
    return R, M
Ra, Ma = reading(A); Rb, Mb = reading(B)
ka = {(r["step"], r["p"]): i for i, r in enumerate(Ra)}; kb = {(r["step"], r["p"]): i for i, r in enumerate(Rb)}
both = sorted(set(ka) & set(kb))
print(f"\nsame words on both: {len(both)}; top word unchanged {np.mean([Ra[ka[k]]['top']==Rb[kb[k]]['top'] for k in both]):.3f}")
by = {}
for k in both:
    i = Ra[ka[k]]["id"]
    by.setdefault(i, []).append(np.log(Mb[kb[k]]) - np.log(Ma[ka[k]]))
x = np.array([np.mean(v) for v in by.values()])
rng = np.random.default_rng(1106)
b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(4000)]
lo, hi = np.percentile(b, [2.5, 97.5])
print(f"{B} minus {A}, log M1, {len(x)} problems: {x.mean():+.3f} ({lo:+.3f} to {hi:+.3f}), so M1 x{np.exp(x.mean()):.3f}")
print("VERDICT:", "THE METER HEARS THE COMPRESSOR" if hi < 0 else "THE COMPRESSOR IS LOUDER BY THIS METER" if lo > 0 else "THE METER DOESN'T HEAR IT")
