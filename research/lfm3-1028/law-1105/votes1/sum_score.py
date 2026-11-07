#!/usr/bin/env python3
"""Scores GOAL-sum-v1. Weights fitted on build problems, pooled right-vs-wrong pairs. usage: sum_score.py <prefix> <slips prefix>"""
import json, sys
import numpy as np

prefix, sp = sys.argv[1:3]
D = np.load(prefix + "-x.npz")
V3, order = D["V3"].astype(np.float64), list(D["order"])
rows = [json.loads(l) for l in open(prefix + "-rows.jsonl")]
split = np.array([r["split"] for r in rows])
prob = np.array([r["id"] for r in rows])
m1 = np.array([r["m"][0] for r in rows])
NW = len(order)
print(f"spots {len(rows)}, problems {len(set(prob))}, build {int((split=='build').sum())}, held {int((split=='held').sum())}; pairs with 2 wrong {int((~np.isnan(V3[:,1,0])).sum())}, 3 wrong {int((~np.isnan(V3[:,2,0])).sum())}")
V1 = V3[:, 0]
print(f"check, votes summed vs real gap: median miss {np.median(np.abs(V1.sum(1)-m1)):.4f} nats, worst {np.abs(V1.sum(1)-m1).max():.4f}")

# 1. the formula alpha_k = mu_k - beta_k D, on real numbers
T = V1.sum(1)
mu, Dd = V1.mean(0), T.mean()
beta = ((V1 - mu) * (T - Dd)[:, None]).mean(0) / T.var()
pred = mu - beta * Dd
band = np.abs(m1) <= 3
X = np.c_[np.ones(band.sum()), m1[band], m1[band] ** 3]
alpha = np.linalg.lstsq(X, V1[band], rcond=None)[0][0]
print(f"\nformula check, all spots: total signal D {Dd:.2f} nats. predicted vs measured lean across 49 writes: correlation {np.corrcoef(pred, alpha)[0,1]:+.3f}, slope {np.polyfit(pred, alpha, 1)[0]:.3f}")
print("  write   signal mu  share of signal  share of noise(beta)  predicted lean  measured lean")
for k in np.argsort(-np.abs(alpha))[:12]:
    print(f"  {order[k]:<7} {mu[k]:>+9.3f}  {mu[k]/Dd:>+15.3f}  {beta[k]:>+20.3f}  {pred[k]:>+14.3f}  {alpha[k]:>+13.3f}")
attn = [i for i, n in enumerate(order) if n.startswith("op") and int(n[2:]) >= 2 and int(n[2:]) % 2 == 0]
conv = [i for i, n in enumerate(order) if n.startswith("op") and i not in attn]
ffn = [i for i, n in enumerate(order) if n.startswith("ffn")]
for name, g in (("attention", attn), ("conv", conv), ("neuron", ffn)):
    print(f"  {name}: share of signal {mu[g].sum()/Dd:+.3f}, share of noise {beta[g].sum():+.3f}, predicted lean {pred[g].sum():+.2f}, measured {alpha[g].sum():+.2f}")

# 2. what the plain sum throws away
B, H = split == "build", split == "held"
pool = V3[B].reshape(-1, NW)
pool = pool[~np.isnan(pool[:, 0])]
w = np.linalg.solve(pool.T @ pool / len(pool), pool.mean(0))
w = w / w.mean()


def wins(weights, idx):
    s = np.nan_to_num(V3[idx] @ weights, nan=np.inf)
    return s.min(1) > 0


def d(weights, idx):
    s = V1[idx] @ weights
    return s.mean() / s.std()


one = np.ones(NW)
ws, wf = wins(one, H), wins(w, H)
res, lost = int((wf & ~ws).sum()), int((ws & ~wf).sum())
print(f"\nheld-out spots {int(H.sum())}: right beats all 3 best wrong, plain sum {ws.mean()*100:.2f}%, fitted weights {wf.mean()*100:.2f}%")
print(f"rescued {res}, lost {lost}; on build for comparison: plain {wins(one,B).mean()*100:.2f}%, fitted {wins(w,B).mean()*100:.2f}%")
rng = np.random.default_rng(1105)
idxs = np.where(H)[0]
pu = np.unique(prob[H])
gp = {p: idxs[prob[H] == p] for p in pu}
diffs = []
for _ in range(2000):
    ii = np.concatenate([gp[p] for p in rng.choice(pu, len(pu))])
    diffs.append(wins(w, ii).mean() - wins(one, ii).mean())
lo, hi = np.percentile(diffs, [2.5, 97.5])
b1, b2 = lo > 0, res >= 2 * max(lost, 1)
print(f"B1 difference {(wf.mean()-ws.mean())*100:+.2f} points, 95% {lo*100:+.2f} to {hi*100:+.2f} -> {'met' if b1 else 'not met'}")
print(f"B2 rescued at least twice lost -> {'met' if b2 else 'not met'}")
print(f"\nGOAL: {'THE SUM THROWS AWAY USABLE SIGNAL' if b1 and b2 else 'THE SUM IS NOT THE BOTTLENECK'}")
print(f"\nd-prime on held-out, plain {d(one,H):.3f}, fitted {d(w,H):.3f}")
sh = np.mean([wins(rng.permutation(w), H).mean() for _ in range(200)])
print(f"same weights shuffled across writes, held-out: {sh*100:.2f}%")
print(f"weights by type (mean over writes, plain = 1): attention {w[attn].mean():.2f}, conv {w[conv].mean():.2f}, neuron {w[ffn].mean():.2f}, word vector {w[0]:.2f}")
print("biggest weights: " + ", ".join(f"{order[k]} {w[k]:+.2f}" for k in np.argsort(-np.abs(w))[:12]))
Hb = H & band
print(f"shaky held-out spots |m|<=3, n {int(Hb.sum())}: plain {wins(one,Hb).mean()*100:.1f}%, fitted {wins(w,Hb).mean()*100:.1f}%")
Sx = np.load(sp + "-x.npz")["V"].astype(np.float64)
sr = [json.loads(l) for l in open(sp + "-rows.jsonl")]
print("\nhis 10 real slips, right vs what he wrote, counted only:")
print(f"  right ahead under plain sum {int((Sx.sum(1)>0).sum())} of 10; under fitted weights {int(((Sx@w)>0).sum())} of 10")
print("  " + " ".join(f"{r['step']}:{Sx[i].sum():+.1f}->{Sx[i]@w:+.1f}" for i, r in enumerate(sr)))
