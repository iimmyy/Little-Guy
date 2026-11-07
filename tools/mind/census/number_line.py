#!/usr/bin/env python3
"""Check 1 for Immy's scene formula (2026-11-03): is his number line LOG-shaped?
The formula puts every quantity on a pin and every relation as a step; multiplying is a step only on a log line. On the
number tokens of the picture set (picture_side.py capture: 1218 numbers from 846 human-written story problems, their
stream at every depth), read each number's value with a ridge line, two ways: target = value and target = log(value).
Folds by problem (5), ridge strength chosen on inner folds. Both readouts are scored the same way: rank agreement
(Spearman) of the held-out readout with the true value (blind to which scale was fit), plus R^2 in the fit's own scale.
If the log readout ranks better, his pins sit on a log line.
usage: number_line.py <picture-cap.npz> <out.txt>
"""
import sys
import numpy as np

z = np.load(sys.argv[1], allow_pickle=True)
X_all, val, grp = z["NX"], z["nown"].astype(np.float64), z["nprob"]
ok = val > 0
X_all, val, grp = X_all[ok], val[ok], grp[ok]
ALPHAS = (1e0, 1e1, 1e2, 1e3, 1e4, 1e5)
ug = np.unique(grp)
fold = {g: i % 5 for i, g in enumerate(np.random.RandomState(0).permutation(ug))}
F = np.array([fold[g] for g in grp])


def ranks(a):
    r = np.empty(len(a)); r[np.argsort(a)] = np.arange(len(a)); return r


def spearman(a, b):
    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


def fit_pred(Xa, ya, Xb, al):
    mu, sd = Xa.mean(0), Xa.std(0) + 1e-6
    A, B = (Xa - mu) / sd, (Xb - mu) / sd
    m = ya.mean()
    K = A @ A.T
    c = np.linalg.solve(K + al * np.eye(len(K)), ya - m)
    return (B @ A.T) @ c + m


def cv(X, y):
    pred = np.zeros(len(y))
    for f in range(5):
        tr, te = F != f, F == f
        best = None
        for al in ALPHAS:                              # inner folds = the other outer folds
            err = 0.0
            for g in set(F[tr]):
                a, b = tr & (F != g), tr & (F == g)
                err += ((fit_pred(X[a], y[a], X[b], al) - y[b]) ** 2).sum()
            if best is None or err < best[0]:
                best = (err, al)
        pred[te] = fit_pred(X[tr], y[tr], X[te], best[1])
    return pred


L = [f"number line shape: {len(val)} numbers from {len(ug)} problems, values {val.min():g}..{val.max():g} (median {np.median(val):g})"]
for li in range(0, X_all.shape[1], 2):
    X = X_all[:, li].astype(np.float64)
    pl, pg = cv(X, val), cv(X, np.log(val))
    r2 = lambda p, y: 1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    L.append(f"  index {li:2d}: rank agreement with the value: linear readout {spearman(pl, val):.3f}, log readout {spearman(pg, val):.3f}"
             f"   (R^2 in own scale: linear {r2(pl, val):.3f}, log {r2(pg, np.log(val)):.3f})")
    print(L[-1], flush=True)
open(sys.argv[2], "w").write("\n".join(L) + "\n")
print("\n".join(L))
