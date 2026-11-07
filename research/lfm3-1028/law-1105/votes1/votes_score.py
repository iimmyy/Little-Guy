#!/usr/bin/env python3
"""Scores GOAL-votes-v1. usage: votes_score.py <right prefix> <slips prefix>"""
import json, sys
import numpy as np

prefix, sprefix = sys.argv[1:3]
D = np.load(prefix + "-x.npz")
V, S, R, order, all_m = D["V"].astype(np.float64), D["S"], D["R"], list(D["order"]), D["all_m"]
rows = [json.loads(l) for l in open(prefix + "-rows.jsonl")]
m = np.array([r["m"] for r in rows])
tally = np.array([r["tally"] for r in rows])
split = np.array([r["split"] for r in rows])
prob = np.array([r["id"] for r in rows])
seen = np.array([r["seen"] for r in rows])
NW = len(order)
print(f"decisive digit spots {len(all_m)}; share with right digit below 95% against his best wrong digit: {(all_m < np.log(19)).mean()*100:.1f}%; |m|<=6: {len(m)}; |m|<=3: {int((np.abs(m)<=3).sum())}; |m|<=1: {int((np.abs(m)<=1).sum())}")
print(f"identity, votes summed vs real logit gap: median miss {np.median(np.abs(m-tally)):.4f} nats, worst {np.abs(m-tally).max():.4f}")
cv = S.std(0) / S.mean(0)
print(f"write size, spread across states (std/mean): median {np.median(cv[1:])*100:.1f}%, worst {cv[1:].max()*100:.1f}% at {order[1+int(cv[1:].argmax())]}; word vector {cv[0]*100:.1f}%")
print("state rms by depth: " + " ".join(f"{order[i]}:{R[:,i].mean():.2f}" for i in (0, 4, 12, 24, 36, 48)))


def fit(mask, flips=None):
    X = np.c_[np.ones(mask.sum()), m[mask], m[mask] ** 3]
    return np.linalg.lstsq(X, V[mask], rcond=None)[0]


band = np.abs(m) <= 3
B, H = band & (split == "build"), band & (split == "held")
print(f"\nspots |m|<=3: build {int(B.sum())} ({len(set(prob[B]))} problems), held {int(H.sum())} ({len(set(prob[H]))} problems)")
if H.sum() < 100:
    print("fewer than 100 held-out spots: cannot answer")
cb, ch = fit(B), fit(H)
print(f"check, sum of alpha: build {cb[0].sum():+.4f}, held {ch[0].sum():+.4f}; sum of beta: build {cb[1].sum():.4f}, held {ch[1].sum():.4f}")

# sign flip by problem, held out
rng = np.random.default_rng(1105)
u = np.unique(prob[H])
which = np.searchsorted(u, prob[H])
mh, Vh = m[H], V[H]
NF = 20000
exceed = np.zeros(NW)
for _ in range(NF):
    s = rng.choice((-1.0, 1.0), len(u))[which]
    X = np.c_[np.ones(len(mh)), s * mh, s * mh ** 3]
    a = np.linalg.solve(X.T @ X, X.T @ (Vh * s[:, None]))[0]
    exceed += np.abs(a) >= np.abs(ch[0])
pv = (exceed + 1) / (NF + 1)
tight = np.abs(m) <= 1
signs = (V[tight] > 0).mean(0)
print("\nwrite     alpha build  alpha held  p held    share of the decision (beta)  leans right at |m|<=1")
hits = []
for k in range(NW):
    flag = abs(ch[0][k]) >= 0.25 and np.sign(ch[0][k]) == np.sign(cb[0][k]) and pv[k] < 0.001
    if flag:
        hits.append(k)
    print(f"{order[k]:<8} {cb[0][k]:>+10.3f}  {ch[0][k]:>+10.3f}  {pv[k]:.5f}  {ch[1][k]:>+24.3f}  {signs[k]*100:>18.0f}%{'  <-- meets the bar' if flag else ''}")
print(f"\nGOAL: {'IN THE VOTES' if hits and H.sum() >= 100 else 'NOT IN THE VOTES' if H.sum() >= 100 else 'CANNOT ANSWER'}")
print(f"total lean cancelled, held: {ch[0][ch[0] > 0].sum():.3f} nats toward right offset by {ch[0][ch[0] < 0].sum():.3f} toward wrong")
for name, mask in (("first time written", band & ~seen), ("already written earlier", band & seen)):
    c = fit(mask)
    print(f"\n{name}: n {int(mask.sum())}; biggest leans: " + ", ".join(f"{order[k]} {c[0][k]:+.2f}" for k in np.argsort(-np.abs(c[0]))[:8]))

Ds = np.load(sprefix + "-x.npz")
Vs = Ds["V"].astype(np.float64)
sr = [json.loads(l) for l in open(sprefix + "-rows.jsonl")]
print("\nhis 10 real slips, state before the first token that differs, a = right, b = what he wrote")
print("  step  right  wrote  gap(right-wrote)  p right  p wrote  top 3 writes voting right  top 3 voting wrong")
for i, r in enumerate(sr):
    o = np.argsort(Vs[i])
    print(f"  {r['step']:>4}  {r['a']:<5}  {r['b']:<5}  {r['m']:>+8.2f}  {r['pa']:>12.3f}  {r['pb']:>7.3f}  " + " ".join(f"{order[k]}{Vs[i][k]:+.1f}" for k in o[::-1][:3]) + "   " + " ".join(f"{order[k]}{Vs[i][k]:+.1f}" for k in o[:3]))
if hits:
    print("\nwrites meeting the bar, on the 10 slips, vote toward the right number: " + ", ".join(f"{order[k]} {(Vs[:,k]*np.sign(ch[0][k])>0).sum()} of 10" for k in hits))
