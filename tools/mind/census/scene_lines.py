#!/usr/bin/env python3
"""Fit and check every term of Immy's formula on scene_mat.py material. Reading only. Each term is one straight line; 0 when nothing is wrong, 1 or more when it is.
  size    spring: how far the stated number is from the scene's answer, |log(said/true)|, a b c
  demand  spring: an earlier number from his own work stated as the answer, d vs a
  whole   wall: a non-whole on a countable. The control is the same non-whole on a divisible unit, which crosses nothing
  sign    wall: a negative count
  precision  curvature: how much the energy rises for a wrong number on this problem, from the right version's stream alone
Lines built on 2/3 of the problems, numbers reported on the other 1/3. Every term also reports how often it speaks at ordinary sentence ends and in the reply.
usage: scene_lines.py <mat.npz> <out lines.npz> <out.txt> [layer=16 alpha=1e3,1e4,1e5]
"""
import hashlib, sys
import numpy as np

args = [a for a in sys.argv[1:] if "=" not in a]
opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
matf, outz, outt = args[:3]
LK = int(opt.get("layer", 16))
ALPHAS = [float(x) for x in opt.get("alpha", "1e3,1e4,1e5").split(",")]
z = np.load(matf, allow_pickle=True)
X = z[f"x{LK}"].astype(np.float64)
ver, grp, pid, at, err = z["ver"], z["grp"], z["pid"], z["at"], z["err"]
V = {n: i for i, n in enumerate("abcdwn")}
H = lambda i: int(hashlib.md5(str(i).encode()).hexdigest(), 16)
bd = np.array([H(i) % 3 != 2 for i in pid])
fold = np.array([H(i) % 4 for i in pid])


def auc(p, n):
    if len(p) == 0 or len(n) == 0:
        return float("nan")
    s = np.concatenate([p, n])
    r = np.argsort(np.argsort(s)).astype(float) + 1
    return (r[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n))


def ridge(Xb, y, w, al):
    sc = Xb.std(0) + 1e-6
    Z = Xb / sc * np.sqrt(w)[:, None]
    return np.linalg.solve(Z.T @ Z + al * np.eye(Z.shape[1]), Z.T @ (y * np.sqrt(w))) / sc


def fit(sel, y, w=None):
    """no offset: the line lives on his own carriers (the file has no bias). Strength on whole problems left out of the build part."""
    idx = np.where(sel & bd)[0]
    w_ = np.ones(len(idx)) if w is None else w[idx]
    best = None
    for al in ALPHAS:
        e = 0
        for f in range(4):
            tr, te = fold[idx] != f, fold[idx] == f
            c = ridge(X[idx[tr]], y[idx[tr]], w_[tr], al)
            e += (w_[te] * (X[idx[te]] @ c - y[idx[te]]) ** 2).sum()
        if best is None or e < best[0]:
            best = (e, al)
    return ridge(X[idx], y[idx], w_, best[1]), best[1]


s0 = at == 0
neutral = (at == 2) | (at == 3)
L = [f"layer {LK}: points {len(ver)}, problems {len(set(pid))}, build {len(set(pid[bd]))}; count {len(set(pid[grp == 'count']))}, unit {len(set(pid[grp == 'unit']))}, other {len(set(pid[grp == 'other']))}"]
lines = {}


def report(name, c, pos_sets, neg_sets):
    sc = X @ c
    ho = ~bd
    nh = sc[ho & (at == 2)]
    rh = sc[ho & (at == 3)]
    th = np.quantile(nh, 0.98)
    msg = [f"{name}: ordinary median {np.median(nh):+.3f}, reply median {np.median(rh):+.3f}; 2% ordinary line at {th:+.3f}"]
    for nm, sel in pos_sets:
        msg.append(f"    {nm}: median {np.median(sc[ho & sel]):+.3f}, over the line {np.mean(sc[ho & sel] > th):.0%}, n {(ho & sel).sum()}")
    for pn, ps, nn, ns in neg_sets:
        msg.append(f"    AUROC {pn} vs {nn}: {auc(sc[ho & ps], sc[ho & ns]):.3f}")
    L.extend(msg)


isv = lambda n: ver == V[n]
# size spring
sel = (s0 & (isv("a") | isv("b") | isv("c"))) | neutral
y = np.where(s0, err, 0.0)
w = np.where(s0, 1.0, 0.5)
lines["size"], al = fit(sel, y, w)
L.append(f"size alpha {al:.0e}")
report("size spring", lines["size"], [("right", s0 & isv("a")), ("wrong b", s0 & isv("b")), ("wrong c", s0 & isv("c")), ("earlier number d", s0 & isv("d"))],
       [("b+c", s0 & (isv("b") | isv("c")), "right", s0 & isv("a"))])
# demand spring
sel = (s0 & (isv("a") | isv("d"))) | neutral
y = np.where(s0 & isv("d"), 1.0, 0.0)
lines["demand"], al = fit(sel, y, np.where(neutral, 0.5, 1.0))
report("demand spring", lines["demand"], [("right", s0 & isv("a")), ("earlier number d", s0 & isv("d")), ("wrong b", s0 & isv("b"))], [("d", s0 & isv("d"), "right", s0 & isv("a")), ("d", s0 & isv("d"), "b", s0 & isv("b"))])
# whole wall: the control is in the fit, so a decimal detector cannot pass
cnt, uni = grp == "count", grp == "unit"
sel = (s0 & (isv("a") | isv("b") | isv("w") | isv("n"))) | neutral
sel = sel & (cnt | uni | neutral)
y = np.where(s0 & isv("w") & cnt, 1.0, 0.0)
pos_w = y > 0
w_ = np.where(pos_w, 3.0, np.where(s0 & isv("w") & uni, 3.0, 1.0))
lines["whole"], al = fit(sel, y, w_)
report("whole wall", lines["whole"], [("right count", s0 & isv("a") & cnt), ("non-whole, countable (crossed)", s0 & isv("w") & cnt), ("non-whole, unit (not crossed)", s0 & isv("w") & uni), ("wrong whole", s0 & isv("b"))],
       [("crossed", s0 & isv("w") & cnt, "unit control", s0 & isv("w") & uni), ("crossed", s0 & isv("w") & cnt, "right", s0 & isv("a") & cnt)])
# sign wall
sel = (s0 & (isv("a") | isv("b") | isv("n"))) | neutral
y = np.where(s0 & isv("n"), 1.0, 0.0)
lines["sign"], al = fit(sel, y, np.where(y > 0, 3.0, 1.0))
report("sign wall", lines["sign"], [("right", s0 & isv("a")), ("negative", s0 & isv("n"))], [("negative", s0 & isv("n"), "right", s0 & isv("a"))])
# precision: on each held-out-by-fold problem, how far the energy climbs from the right number to wrong ones
E = (X @ lines["size"]) ** 2 + np.maximum(X @ lines["demand"], 0) ** 2 + np.maximum(X @ lines["whole"], 0) ** 2 + np.maximum(X @ lines["sign"], 0) ** 2
ids = sorted(set(pid[s0]))
curv = {}
for i in ids:
    m = pid == i
    a_, wr = E[m & s0 & isv("a")], E[m & s0 & (isv("b") | isv("c"))]
    if len(a_) and len(wr):
        curv[i] = float(wr.mean() - a_.mean())
vals = np.array(list(curv.values()))
L.append(f"energy: right median {np.median(E[~bd & s0 & isv('a')]):.3f}; b+c {np.median(E[~bd & s0 & (isv('b') | isv('c'))]):.3f}; demand d {np.median(E[~bd & s0 & isv('d')]):.3f}; crossed whole {np.median(E[~bd & s0 & isv('w') & cnt]):.3f}; negative {np.median(E[~bd & s0 & isv('n')]):.3f}; ordinary {np.median(E[~bd & (at == 2)]):.3f}; reply {np.median(E[~bd & (at == 3)]):.3f}")
for pn, ps in (("b+c", s0 & (isv("b") | isv("c"))), ("d", s0 & isv("d")), ("crossed", s0 & isv("w") & cnt), ("negative", s0 & isv("n"))):
    L.append(f"  energy AUROC {pn} vs right: {auc(E[~bd & ps], E[~bd & s0 & isv('a')]):.3f}")
L.append(f"curvature per problem: median {np.median(vals):.3f}, 10% {np.quantile(vals, 0.1):.3f}, 90% {np.quantile(vals, 0.9):.3f}, n {len(vals)}")
sel = s0 & (isv("a") | isv("b") | isv("c"))
y = np.array([curv.get(i, 0.0) for i in pid])
sel = sel & np.array([i in curv for i in pid])
lines["precision"], al = fit(sel, y)
pr = X @ lines["precision"]
ho = ~bd & s0 & isv("a") & np.array([i in curv for i in pid])
if ho.sum() > 5:
    r2 = 1 - ((y[ho] - pr[ho]) ** 2).sum() / ((y[ho] - y[bd & sel].mean()) ** 2).sum()
    L.append(f"precision line (alpha {al:.0e}): held-out from the right version's stream: R2 {r2:.3f}, correlation {np.corrcoef(y[ho], pr[ho])[0, 1]:.3f}, n {ho.sum()}")
L.append(f"precision at ordinary sentence ends median {np.median(pr[~bd & (at == 2)]):+.3f}, at right statements {np.median(pr[ho]):+.3f}")
np.savez(outz, layer=LK, **lines)
open(outt, "w").write("\n".join(L) + "\n")
print("\n".join(L))
