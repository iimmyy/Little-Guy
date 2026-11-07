#!/usr/bin/env python3
"""Can one run feel whether it agrees with his other runs? Whole problems held out. The bar was set before looking (Immy 2026-11-04): score >= 0.75 and catches >= half at 5% false alarms.
Lines are fit to his own agreement (no answer key) and scored on three things: off vs agree, wrong vs right, and pairs inside one problem. A key-fit line is shown beside it, plus shuffled labels and a plain-facts baseline.
usage: agree_fit.py <out.txt> <rows.jsonl> <x.npz> <labels.json> [tag=end layers=12,16,17,20,23]"""
import hashlib, json, sys, collections
import numpy as np
args = [a for a in sys.argv[1:] if "=" not in a]
opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
out, rf, xf, lf = args
TAG = opt.get("tag", "end"); LAYERS = [int(x) for x in opt.get("layers", "12,16,17,20,23").split(",")]
R = [json.loads(l) for l in open(rf)]; Z = np.load(xf); LAB = json.load(open(lf))
keep = np.array([r["tag"] == TAG and f"{r['run']}|{r['id']}" in LAB for r in R])
ix = np.where(keep)[0]; R2 = [R[i] for i in ix]
L2 = [LAB[f"{r['run']}|{r['id']}"] for r in R2]
state = np.array([l["state"] for l in L2]); grade = np.array([l["grade"] for l in L2]); tight = np.array([l["tight"] for l in L2])
pid = np.array([r["id"] for r in R2]); depth = np.array([r["tokens_before"] for r in R2], float)
FOLDS = 5
fold = np.array([int(hashlib.md5(p.encode()).hexdigest(), 16) % FOLDS for p in pid])


def auroc(p, n):
    if not len(p) or not len(n): return float("nan")
    o = np.sort(n); return float(np.mean((np.searchsorted(o, p, "left") + np.searchsorted(o, p, "right")) / (2 * len(n))))


def catch(p, n, fa=.05):
    return float(np.mean(p > np.quantile(n, 1 - fa))) if len(p) and len(n) else float("nan")


def oof(X, y, w):
    """out-of-fold line, whole problems held out; the penalty is chosen on the training folds only"""
    s = np.zeros(len(y)); sc = X.std(0) + 1e-6; Xs = X / sc
    for f in range(FOLDS):
        te = fold == f; tr = ~te & w
        best, bv = 1e3, -1
        ins = np.array([int(hashlib.md5((p + "i").encode()).hexdigest(), 16) % 4 == 0 for p in pid])
        A, B = tr & ~ins, tr & ins
        ym = y[A].mean()
        for al in (1e2, 1e3, 1e4, 1e5):
            b = np.linalg.solve(Xs[A].T @ Xs[A] + al * np.eye(X.shape[1]), Xs[A].T @ (y[A] - ym))
            q = Xs[B] @ b
            v = auroc(q[y[B] > .5], q[y[B] <= .5]) if len(set((y[B] > .5).tolist())) > 1 else -1
            if v > bv: best, bv = al, v
        ym = y[tr].mean()
        b = np.linalg.solve(Xs[tr].T @ Xs[tr] + best * np.eye(X.shape[1]), Xs[tr].T @ (y[tr] - ym))
        s[te] = Xs[te] @ b
    return s


def pairs(s, a, b):
    d = []
    for p in set(pid):
        A, B = s[(pid == p) & a], s[(pid == p) & b]
        if len(A) and len(B): d.append(auroc(A, B))
    return float(np.mean(d)) if d else float("nan"), len(d)


def boot(s, a, b, n=300):
    rng = np.random.default_rng(1); P = sorted(set(pid)); by = {p: np.where(pid == p)[0] for p in P}; o = []; c = []
    for _ in range(n):
        ii = np.concatenate([by[p] for p in rng.choice(P, len(P))])
        o.append(auroc(s[ii][a[ii]], s[ii][b[ii]])); c.append(catch(s[ii][a[ii]], s[ii][b[ii]]))
    return np.nanquantile(o, [.025, .975]), np.nanquantile(c, [.025, .975])


off, agr = state == "off", state == "agree"
wr, ri = grade == "wrong", grade == "right"
L = [f"{TAG}: {len(R2)} runs on {len(set(pid))} problems. agree {agr.sum()}, off {off.sum()}, tie {(state == 'tie').sum()}; right {ri.sum()}, wrong {wr.sum()}. Machine grade."]
rng = np.random.default_rng(3)
for lay in LAYERS:
    X = Z[f"L{lay}"][ix].astype(np.float64)
    use = off | agr
    s = oof(X, off.astype(float), use)
    ci, cc = boot(s, off, agr)
    p1, n1 = pairs(s, off, agr)
    L.append(f"layer {lay}, line fit to agreement (no key):")
    L.append(f"  off vs agree: score {auroc(s[off], s[agr]):.3f} ({ci[0]:.2f}-{ci[1]:.2f}), catches {catch(s[off], s[agr]):.0%} at 5% ({cc[0]:.0%}-{cc[1]:.0%}); same problem {p1:.3f} ({n1} problems)")
    ci, cc = boot(s, wr, ri); p2, n2 = pairs(s, wr, ri)
    L.append(f"  wrong vs right: score {auroc(s[wr], s[ri]):.3f} ({ci[0]:.2f}-{ci[1]:.2f}), catches {catch(s[wr], s[ri]):.0%} at 5% ({cc[0]:.0%}-{cc[1]:.0%}); same problem {p2:.3f} ({n2} problems)")
    s2 = oof(X, wr.astype(float), wr | ri); p3, n3 = pairs(s2, wr, ri)
    L.append(f"  beside it, a line fit to the answer key: wrong vs right {auroc(s2[wr], s2[ri]):.3f}, catches {catch(s2[wr], s2[ri]):.0%}; same problem {p3:.3f} ({n3} problems)")
    y0 = off.astype(float).copy(); idx = rng.permutation(np.where(use)[0]); y0[np.where(use)[0]] = off[idx].astype(float)
    s3 = oof(X, y0, use)
    L.append(f"  shuffled labels: {auroc(s3[(y0 > .5) & use], s3[(y0 <= .5) & use]):.3f}")
    s4 = oof(X, tight, np.ones(len(tight), bool))
    hi, lo = tight >= .75, tight <= .5
    L.append(f"  precision line, fit to how tightly his runs land: tight vs scattered {auroc(s4[hi], s4[lo]):.3f}, correlation {np.corrcoef(s4, tight)[0, 1]:.3f}; scattered {lo.sum()}, tight {hi.sum()}")
F = np.c_[np.log1p(depth)]
L.append(f"plain fact, how far into the thinking: off vs agree {auroc(F[off, 0], F[agr, 0]):.3f}; wrong vs right {auroc(F[wr, 0], F[ri, 0]):.3f}")
open(out, "w").write("\n".join(L) + "\n"); print("\n".join(L))
