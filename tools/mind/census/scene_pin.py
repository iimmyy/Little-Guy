#!/usr/bin/env python3
"""Check 3 for Immy's scene formula (2026-11-03): at the moment he STATES a number, does his stream hold where the
SCENE puts the answer (a*), separately from what he just said? The formula's error = pin(said) - a*.
A read just before he writes a number would only show what he is about to say. The fit test (fit_test.py) gives a
clean separation instead: each of 212 forks of his own right thinking appears three times with a different stated
number (a = his right answer A, b = wrong but same side, c = wrong other side, at A*r vs A/r). His stream after the
statement, every depth, is in fit-test-v1.npz.
  scene readout: one ridge line fit on all three versions with target log(A), the TRUE answer: the stated number
                 changes while the truth stays, so copying it does not work; the line has to find the scene's answer
  said readout:  one ridge line with target log(stated number)
Folds by problem (5), strength on inner folds. Reported per version: the scene readout's miss as a factor, and how
well the formula error |log(said) - scene readout| (using the true stated number, then the said readout) separates the
wrong statements (b, c) from the right one (a): AUROC, and the share of b / c flagged at the threshold that flags 10% of a.
usage: scene_pin.py <fit-test.npz> <out.txt>
"""
import math, sys
import numpy as np

z = np.load(sys.argv[1], allow_pickle=True)
X, A, B, C = z["X"], z["A"].astype(float), z["b"].astype(float), z["c"].astype(float)
n = len(A)
said = np.log(np.stack([A, B, C], 1))                     # (n, 3)
truth = np.log(np.stack([A, A, A], 1))
prob = np.repeat(np.arange(n), 3)
ALPHAS = (1e0, 1e1, 1e2, 1e3, 1e4, 1e5)
fold = np.random.RandomState(0).permutation(n) % 5
F = np.repeat(fold, 3)


def fit_pred(Xa, ya, Xb, al):
    mu, sd = Xa.mean(0), Xa.std(0) + 1e-6
    Aa, Bb = (Xa - mu) / sd, (Xb - mu) / sd
    m = ya.mean()
    c = np.linalg.solve(Aa @ Aa.T + al * np.eye(len(Aa)), ya - m)
    return (Bb @ Aa.T) @ c + m


def cv(Xf, y):
    pred = np.zeros(len(y))
    for f in range(5):
        tr, te = F != f, F == f
        best = None
        for al in ALPHAS:
            err = sum(((fit_pred(Xf[tr & (F != g)], y[tr & (F != g)], Xf[tr & (F == g)], al) - y[tr & (F == g)]) ** 2).sum() for g in set(F[tr]))
            if best is None or err < best[0]:
                best = (err, al)
        pred[te] = fit_pred(Xf[tr], y[tr], Xf[te], best[1])
    return pred


def auroc(pos, neg):
    allv = np.concatenate([pos, neg]); r = np.empty(len(allv)); r[np.argsort(allv)] = np.arange(1, len(allv) + 1)
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


L = [f"scene pin at the statement: {n} forks x 3 versions (a right, b wrong same side, c wrong other side), "
     f"wrong by x{np.exp(np.median(np.abs(np.log(z['r'])))):.2f} median"]
for li in range(0, X.shape[2], 2):
    Xf = X[:, :, li].reshape(n * 3, -1).astype(np.float64)
    sc = cv(Xf, truth.reshape(-1)).reshape(n, 3)
    sa = cv(Xf, said.reshape(-1)).reshape(n, 3)
    miss = np.abs(sc - truth)
    row = [f"index {li:2d}: scene readout miss (median) a x{math.exp(np.median(miss[:, 0])):.2f}, b x{math.exp(np.median(miss[:, 1])):.2f}, "
           f"c x{math.exp(np.median(miss[:, 2])):.2f}; within x1.25 {np.mean(miss < math.log(1.25)):.0%}"]
    for nm, s_ in (("true said", said), ("said readout", sa)):
        e = np.abs(s_ - sc)
        th = np.quantile(e[:, 0], 0.90)
        row.append(f"{nm}: AUROC a vs b {auroc(e[:, 1], e[:, 0]):.2f}, a vs c {auroc(e[:, 2], e[:, 0]):.2f}; flags b {np.mean(e[:, 1] > th):.0%} c {np.mean(e[:, 2] > th):.0%} (a 10%)")
    # v2: the error read directly (signed log(said / true) and its size), the formula's epsilon as one line
    err = (said - truth).reshape(-1)
    pe = cv(Xf, err).reshape(n, 3)
    pa = cv(Xf, np.abs(err)).reshape(n, 3)
    row.append(f"error read directly: signed |pred| AUROC a vs b {auroc(np.abs(pe[:, 1]), np.abs(pe[:, 0])):.2f} c {auroc(np.abs(pe[:, 2]), np.abs(pe[:, 0])):.2f}; "
               f"size AUROC b {auroc(pa[:, 1], pa[:, 0]):.2f} c {auroc(pa[:, 2], pa[:, 0]):.2f}; "
               f"sign right on b {np.mean(np.sign(pe[:, 1]) == np.sign(err.reshape(n, 3)[:, 1])):.0%} c {np.mean(np.sign(pe[:, 2]) == np.sign(err.reshape(n, 3)[:, 2])):.0%}")
    L.append("  " + " | ".join(row))
    print(L[-1], flush=True)
open(sys.argv[2], "w").write("\n".join(L) + "\n")
print("\n".join(L))
