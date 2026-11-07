#!/usr/bin/env python3
"""Lean of each write after taking out the part of the write that is the same on every problem.
The fixed part comes from build spots only. usage: votes_bias_score.py <bias prefix> <votes prefix>"""
import json, sys
import numpy as np

bp, vp = sys.argv[1:3]
D = np.load(bp + "-x.npz")
V, Vb, order = D["V"].astype(np.float64), D["Vbias"].astype(np.float64), list(D["order"])
V0 = np.load(vp + "-x.npz")["V"].astype(np.float64)
rows = [json.loads(l) for l in open(vp + "-rows.jsonl")]
m = np.array([r["m"] for r in rows])
split = np.array([r["split"] for r in rows])
prob = np.array([r["id"] for r in rows])
tl = np.array([len(r["a"]) for r in rows])
NW = len(order)
print(f"check, votes read again vs first read: worst difference {np.abs(V-V0).max():.4f}")
Vs = V - Vb
print(f"share of the gap from the fixed part: sum of fixed votes, median {np.median(Vb.sum(1)):+.2f} nats, spread (std) {Vb.sum(1).std():.2f}; state part median {np.median(Vs.sum(1)):+.2f}, spread {Vs.sum(1).std():.2f}")


def fit(mask, Y):
    X = np.c_[np.ones(mask.sum()), m[mask], m[mask] ** 3]
    return np.linalg.lstsq(X, Y[mask], rcond=None)[0][0]


band = np.abs(m) <= 3
B, H = band & (split == "build"), band & (split == "held")
print(f"spots |m|<=3: build {int(B.sum())}, held {int(H.sum())}")
raw_b, raw_h = fit(B, V), fit(H, V)
fix_b, fix_h = fit(B, Vb), fit(H, Vb)
st_b, st_h = fit(B, Vs), fit(H, Vs)
u = np.unique(prob[H])
which = np.searchsorted(u, prob[H])
mh, Yh = m[H], Vs[H]
rng = np.random.default_rng(1105)
NF = 20000
exceed = np.zeros(NW)
for _ in range(NF):
    s = rng.choice((-1.0, 1.0), len(u))[which]
    X = np.c_[np.ones(len(mh)), s * mh, s * mh ** 3]
    a = np.linalg.solve(X.T @ X, X.T @ (Yh * s[:, None]))[0]
    exceed += np.abs(a) >= np.abs(st_h)
pv = (exceed + 1) / (NF + 1)
print("\nwrite    raw build  raw held | fixed build  fixed held | state build  state held  p held")
hits = []
for k in np.argsort(-np.abs(raw_b + raw_h)):
    flag = abs(st_h[k]) >= 0.25 and np.sign(st_h[k]) == np.sign(st_b[k]) and pv[k] < 0.001
    if flag:
        hits.append(k)
    print(f"{order[k]:<8} {raw_b[k]:>+8.3f}  {raw_h[k]:>+8.3f} | {fix_b[k]:>+10.3f}  {fix_h[k]:>+10.3f} | {st_b[k]:>+10.3f}  {st_h[k]:>+10.3f}  {pv[k]:.5f}{'  <-- meets the bar' if flag else ''}")
print(f"\nby the same per-write bar, state part only: {[order[k] for k in hits]}; held-out spots {int(H.sum())} (bar needs 100 to answer)")
print(f"total lean, state part, held: {st_h[st_h > 0].sum():+.3f} toward right, {st_h[st_h < 0].sum():+.3f} toward wrong")
for name, mask in (("right token has 3 digits", band & (tl == 3)), ("right token has 1 digit", band & (tl == 1)), ("not the token 0", band & np.array([r["a"] != "0" for r in rows]))):
    c = fit(mask, Vs)
    print(f"{name}: n {int(mask.sum())}; biggest state leans: " + ", ".join(f"{order[k]} {c[k]:+.2f}" for k in np.argsort(-np.abs(c))[:8]))
