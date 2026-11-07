#!/usr/bin/env python3
"""Does he know how long he has been thinking? Reading only. A straight line from the stream to words so far, at paragraph ends, on whole problems left out.
A hand-in by the clock would need this. Reported: fit quality, and how cleanly a line separates over-N-words from under-N-words.
usage: clock_read.py <cap.npz> <out.txt>
"""
import hashlib, sys
import numpy as np


def auc(y, s):
    y = np.asarray(y) > 0
    r = np.argsort(np.argsort(s)).astype(float) + 1
    n1, n0 = y.sum(), (~y).sum()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


z = np.load(sys.argv[1], allow_pickle=True)
kind, pos, gid = z["kind"], z["pos"], z["gid"]
fold = np.array([int(hashlib.md5(str(g).encode()).hexdigest(), 16) % 3 for g in gid])
L = [f"points {len(pos)}, words so far: median {np.median(pos):.0f}, 90% {np.quantile(pos, 0.9):.0f}"]
for Lk in sorted(int(k[1:]) for k in z.files if k.startswith("x")):
    X = z[f"x{Lk}"].astype(np.float64)
    for nm, s in (("paragraph ends", kind == 0), ("all kinds", np.ones(len(pos), bool))):
        y = np.log1p(pos)
        pred = np.zeros(len(y))
        for f in range(3):
            b, h = s & (fold != f), s & (fold == f)
            mu, sd = X[b].mean(0), X[b].std(0) + 1e-6
            Zb = (X[b] - mu) / sd
            c = np.linalg.solve(Zb.T @ Zb + 1e4 * np.eye(Zb.shape[1]), Zb.T @ (y[b] - y[b].mean()))
            pred[h] = ((X[h] - mu) / sd) @ c + y[b].mean()
        r2 = 1 - ((y[s] - pred[s]) ** 2).sum() / ((y[s] - y[s].mean()) ** 2).sum()
        err = np.median(np.abs(np.expm1(pred[s]) / np.maximum(pos[s], 1) - 1))
        msg = f"layer {Lk:2d} {nm:15s} n {s.sum():5d}: log(words so far) R2 {r2:.3f}, typical error {err:.0%}"
        for N in (1000, 1500, 2000):
            msg += f"; over {N} AUROC {auc(pos[s] > N, pred[s]):.3f}"
        L.append(msg)
open(sys.argv[2], "w").write("\n".join(L) + "\n")
print("\n".join(L))
