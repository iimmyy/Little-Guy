#!/usr/bin/env python3
"""Step 2 of the hand-in cell: can a straight line read "the answer has been on the page for a while", at the points where a commit starts?
Reading only. Built on 2/3 of the problems, scored on the other 1/3. Compared with a clock (words so far) so we know if it holds more than time.
usage: done_read.py <cap.npz> <out.txt>
"""
import hashlib, sys
import numpy as np


def auc(y, s):
    y = np.asarray(y) > 0
    r = np.argsort(np.argsort(s)).astype(float) + 1
    n1, n0 = y.sum(), (~y).sum()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


z = np.load(sys.argv[1], allow_pickle=True)
kind, since, pos, fin, gid = z["kind"], z["since"], z["pos"], z["fin"], z["gid"]
LAY = sorted(int(k[1:]) for k in z.files if k.startswith("x"))
build = np.array([int(hashlib.md5(str(g).encode()).hexdigest(), 16) % 3 != 2 for g in gid])
L = [f"{len(kind)} points; paragraph ends {(kind == 0).sum()}, openers {(kind == 1).sum()}, ordinary {(kind == 2).sum()}; held-out problems {len(set(gid[~build]))}"]
for k, nm in ((0, "paragraph ends"), (1, "opener words")):
    s = kind == k
    b = s & (since < 0)
    for lab, g in (("before the answer", b), ("after, finished runs", s & (since >= 0) & fin), ("after, cut-off runs", s & (since >= 0) & ~fin)):
        L.append(f"  {nm:15s} {lab:22s} n {g.sum():5d}  P(Thus) {z['p_thus'][g].mean():.3f}  P(any opener) {z['p_open'][g].mean():.3f}  P(final) {z['p_final'][g].mean():.3f}  P(answer) {z['p_answer'][g].mean():.3f}  P(end) {z['p_end'][g].mean():.4f}")


def ridge(X, y, w, al):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    A = Z * np.sqrt(w)[:, None]
    c = np.linalg.solve(A.T @ A + al * np.eye(Z.shape[1]), A.T @ (y * np.sqrt(w)))
    return lambda Q: ((Q - mu) / sd) @ c


for T in (0, 80, 300):
    for Lk in LAY:
        X = z[f"x{Lk}"].astype(np.float64)
        s = kind != 2
        y = (since >= T).astype(float)
        keep = s & (y.astype(bool) | (since < 0))                        # drop the zone between the answer and T
        b, h = keep & build, keep & ~build
        w = np.where(y[b] > 0, 0.5 / y[b].sum(), 0.5 / (1 - y[b]).sum()) * b.sum()
        best = None
        vb = np.array([int(hashlib.md5(str(g).encode()).hexdigest(), 16) % 4 == 0 for g in gid[b]])
        for al in (1e2, 1e3, 1e4, 1e5):
            f = ridge(X[b][~vb], y[b][~vb], w[~vb], al)
            a = auc(y[b][vb], f(X[b][vb]))
            if best is None or a > best[0]:
                best = (a, al)
        f = ridge(X[b], y[b], w, best[1])
        sc = f(X[h])
        clock = auc(y[h], pos[h])
        fh = fin[h]
        pos_h = y[h] > 0
        cut_auc = auc(y[h][~fh], sc[~fh]) if len(set(y[h][~fh])) == 2 else float("nan")
        fin_auc = auc(y[h][fh], sc[fh]) if len(set(y[h][fh])) == 2 else float("nan")
        ord_ = (kind == 2) & ~build
        th5 = np.quantile(sc[~pos_h], 0.95)
        L.append(f"T {T:3d} layer {Lk:2d} (alpha {best[1]:.0e}): AUROC {auc(y[h], sc):.3f} (clock alone {clock:.3f}; finished {fin_auc:.3f}, cut-off {cut_auc:.3f}); at 5% false alarms before the answer catches {np.mean(sc[pos_h] > th5):.0%}; ordinary words over that line {np.mean(f(X[ord_]) > th5):.1%}")
open(sys.argv[2], "w").write("\n".join(L) + "\n")
print("\n".join(L))
