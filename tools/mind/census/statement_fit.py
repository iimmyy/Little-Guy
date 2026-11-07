#!/usr/bin/env python3
"""Can his own right and wrong statements be told apart from inside him? Fits one line per layer on some problems, scores others, so nothing is
learned about the problem being scored. Every run of a problem stays on one side. Reading only.
The bar was set before looking (Immy 2026-11-04): score (AUROC) >= 0.75 and catches >= half of real mistakes at 5% false alarms.
Also shown: the same fit with shuffled labels (should sit at 0.5), a baseline from plain facts about the text (how far into the thinking, number size), an interval
from resampling whole problems, and the pairs inside one problem, where easy and hard problems cannot help.
usage: statement_fit.py <out.txt> <rows.jsonl> <x.npz> [<rows.jsonl> <x.npz> ...] [tag=final2 where=last at=eol layers=16,17 labels=<json {"run|id": label}> skip=id,id]
"""
import json, sys, hashlib
import numpy as np

args = [a for a in sys.argv[1:] if "=" not in a]
opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
out, parts = args[0], args[1:]
TAG, WHERE, AT = opt.get("tag", "final2"), opt.get("where"), opt.get("at")
LAYERS = [int(x) for x in opt.get("layers", "16,17").split(",")]
SKIP = set(filter(None, opt.get("skip", "").split(",")))
LABELS = json.load(open(opt["labels"])) if "labels" in opt else {}
FOLDS, ALPHAS = 5, (1e1, 1e2, 1e3, 1e4, 1e5)


def auroc(pos, neg):
    if not len(pos) or not len(neg):
        return float("nan")
    ordered_neg = np.sort(neg)
    less = np.searchsorted(ordered_neg, pos, side="left")
    less_equal = np.searchsorted(ordered_neg, pos, side="right")
    return float(np.mean((less + less_equal) / (2 * len(neg))))


def catch(pos, neg, fa=0.05):
    return float(np.mean(pos > np.quantile(neg, 1 - fa)))


rows, X = [], {L: [] for L in LAYERS}
for rf, xf in zip(parts[::2], parts[1::2]):
    R = [json.loads(l) for l in open(rf)]
    z = np.load(xf)
    keep = []
    for i, r in enumerate(R):
        lab = LABELS.get(f"{r['run']}|{r['id']}", r.get("lab"))
        if r["tag"] != TAG or lab not in ("right", "wrong") or r["id"][-6:] in SKIP:
            continue
        if (WHERE and r.get("where") != WHERE) or (AT and r.get("at") != AT):
            continue
        r["lab"] = lab
        rows.append(r); keep.append(i)
    for L in LAYERS:
        X[L].append(z[f"L{L}"][keep].astype(np.float64))
X = {L: np.concatenate(v) for L, v in X.items()}
y = np.array([r["lab"] == "wrong" for r in rows])
G = np.array([r["group"] for r in rows])
fold = np.array([int(hashlib.md5(g.encode()).hexdigest(), 16) % FOLDS for g in G])
T = [f"{TAG} where={WHERE} at={AT}: {len(rows)} statements, {int(y.sum())} wrong, {len(set(G[y]))} problems with a wrong, {len(set(G))} problems"]


def ridge(Xa, ya, Xb, al):
    mu, sd = Xa.mean(0), Xa.std(0) + 1e-6
    A, B = (Xa - mu) / sd, (Xb - mu) / sd
    w = np.where(ya, 0.5 / max(ya.sum(), 1), 0.5 / max((~ya).sum(), 1)) * len(ya)     # both classes weigh the same
    s = np.sqrt(w)[:, None]
    AW = A * s
    c = np.linalg.solve(AW @ AW.T + al * np.eye(len(A)), (ya.astype(float) - 0.5) * s[:, 0])
    return B @ (AW.T @ c)


def oof(XX, yy, inner=True):
    sc = np.zeros(len(yy))
    for f in range(FOLDS):
        te = fold == f
        tr = ~te
        if yy[tr].sum() < 3:
            sc[te] = 0.0
            continue
        best, bv = ALPHAS[2], -1
        if inner:
            gi = np.array([int(hashlib.md5((g + "i").encode()).hexdigest(), 16) % 4 == 0 for g in G[tr]])
            if yy[tr][gi].sum() and yy[tr][~gi].sum() >= 3:
                for al in ALPHAS:
                    v = ridge(XX[tr][~gi], yy[tr][~gi], XX[tr][gi], al)
                    a_ = auroc(v[yy[tr][gi]], v[~yy[tr][gi]])
                    if a_ > bv:
                        best, bv = al, a_
        s_ = ridge(XX[tr], yy[tr], XX[te], best)
        sc[te] = (s_ - s_.mean()) / (s_.std() + 1e-9)                                  # each fold on one scale, since each has its own line
    return sc


def boot(sc, yy, n=1000):
    rng = np.random.RandomState(0)
    ug = np.unique(G)
    idx = {g: np.where(G == g)[0] for g in ug}
    A, C = [], []
    for _ in range(n):
        s = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])
        if yy[s].sum() < 3:
            continue
        A.append(auroc(sc[s][yy[s]], sc[s][~yy[s]])); C.append(catch(sc[s][yy[s]], sc[s][~yy[s]]))
    return np.percentile(A, [2.5, 97.5]), np.percentile(C, [2.5, 97.5])


def pairs(sc, yy):
    w, t = 0.0, 0
    for g in np.unique(G):
        s = G == g
        if yy[s].any() and (~yy[s]).any():
            for a in sc[s][yy[s]]:
                for b in sc[s][~yy[s]]:
                    w += float(a > b) + 0.5 * float(a == b); t += 1
    return (w / t if t else float("nan")), t, len({g for g in np.unique(G) if yy[G == g].any() and (~yy[G == g]).any()})


for L in LAYERS:
    sc = oof(X[L], y)
    ai, ci = boot(sc, y)
    a_, c_ = auroc(sc[y], sc[~y]), catch(sc[y], sc[~y])
    pw, pn, pp = pairs(sc, y)
    rng = np.random.RandomState(1)
    ug = np.unique(G)                                                                  # shuffle whole problems' labels together, so the null keeps the grouping
    nulls = []
    for k in range(5):
        perm = dict(zip(ug, rng.permutation(ug)))
        fake = np.zeros(len(y), bool)
        for g in ug:
            src = y[G == perm[g]]
            s = np.where(G == g)[0]
            fake[s] = rng.choice(src, len(s)) if len(src) else False
        if fake.sum() >= 5:
            sn = oof(X[L], fake, inner=False)
            nulls.append(auroc(sn[fake], sn[~fake]))
    verdict = "PASS" if a_ >= 0.75 and c_ >= 0.5 else "fail"
    T.append(f"  layer {L}: score {a_:.3f} (95% {ai[0]:.2f}-{ai[1]:.2f}), catches {c_:.0%} at 5% false alarms (95% {ci[0]:.0%}-{ci[1]:.0%}) -> {verdict}; same problem, right vs wrong {pw:.2f} ({pn} pairs, {pp} problems); shuffled labels {np.mean(nulls) if nulls else float('nan'):.3f}")
d = np.array([r.get("tokens_before", 0) for r in rows], float)
vv = np.array([abs(float(r.get("v") or 0)) for r in rows])
B = np.stack([np.log1p(d), np.log10(vv + 1e-12)], 1)
sb = oof(B, y, inner=False)
T.append(f"  plain-facts baseline (depth, number size): score {auroc(sb[y], sb[~y]):.3f}, catches {catch(sb[y], sb[~y]):.0%} at 5% false alarms")
open(out, "w").write("\n".join(T) + "\n")
print("\n".join(T))
