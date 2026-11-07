"""Small shared readers for the time-sense probes (difficulty_probe.py, expected_persist.py): rank agreement and a
ridge line with its strength picked by 5-fold rank agreement on the training rows only."""
import numpy as np


def rank(x):
    return np.asarray(x).argsort().argsort().astype(float)


def rho(a, b):
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def in_level(s, y, lv):
    return float(np.mean([rho(s[lv == k], y[lv == k]) for k in np.unique(lv) if (lv == k).sum() >= 5]))


def ridge(X, y, Xt, alphas=(1, 10, 100, 1e3, 1e4, 1e5)):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z, Zt = (X - mu) / sd, (Xt - mu) / sd
    ym = y.mean()
    folds = np.arange(len(y)) % 5
    best = None
    for al in alphas:                                     # pick the strength by 5-fold rank agreement on train only
        pr = np.zeros(len(y))
        for f in range(5):
            a, b = folds != f, folds == f
            w = np.linalg.solve(Z[a].T @ Z[a] + al * np.eye(Z.shape[1]), Z[a].T @ (y[a] - y[a].mean()))
            pr[b] = Z[b] @ w
        s = rho(pr, y)
        if best is None or s > best[0]:
            best = (s, al)
    w = np.linalg.solve(Z.T @ Z + best[1] * np.eye(Z.shape[1]), Z.T @ (y - ym))
    return Zt @ w + ym, best[1], best[0], (w / sd, ym - (mu / sd) @ w)
