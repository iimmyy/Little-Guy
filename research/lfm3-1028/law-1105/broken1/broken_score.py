#!/usr/bin/env python3
"""Scores GOAL-broken-v1. usage: broken_score.py <prefix>"""
import collections, json, sys
import numpy as np

prefix = sys.argv[1]
D = np.load(prefix + "-x.npz")
order = list(D["order"])
NW = len(order)
rows = [json.loads(l) for l in open(prefix + "-rows.jsonl")]
iu = np.triu_indices(NW)
ARMS = ("clean", "broken", "reformat")
KEY = {}
for i, r in enumerate(rows):
    KEY[(r["step"], r["arm"], r["k"])] = i
steps = sorted({r["step"] for r in rows})
info = {r["step"]: r for r in rows}
split = np.array([info[s]["split"] for s in steps])
prob = np.array([info[s]["id"] for s in steps])
surp = {a: np.array([rows[KEY[(s, a, 1)]]["num_surprise"] if (s, a, 1) in KEY else np.nan for s in steps]) for a in ARMS}
seen = np.array([info[s]["seen_before"] for s in steps])
print(f"steps {len(steps)} problems {len(set(prob))} build {int((split=='build').sum())} held {int((split=='held').sum())}")


def gram_at(k):
    """Gram matrices for one word position: steps x arms x NW x NW, nan where missing."""
    out = np.full((len(steps), 3, NW, NW), np.nan, dtype=np.float32)
    for si, s in enumerate(steps):
        for ai, a in enumerate(ARMS):
            i = KEY.get((s, a, k))
            if i is not None:
                g = np.zeros((NW, NW), dtype=np.float32)
                g[iu] = D["G"][i]
                out[si, ai] = g + g.T - np.diag(np.diag(g))
    return out


def reads(g):
    """All no-fit reads from Gram matrices; returns dict family -> (names, array steps x arms x candidates)."""
    S = np.cumsum(np.cumsum(g, 2), 3)
    idx = np.arange(1, NW)
    selfsq = np.diagonal(S, axis1=2, axis2=3)
    C = np.cumsum(np.diagonal(g, axis1=2, axis2=3), 2)
    total = ((C - selfsq) / (2 * C))[..., 1:]
    size = np.sqrt(selfsq)[..., 1:]
    held_dot = np.stack([S[..., i - 1, i] - S[..., i - 1, i - 1] for i in idx], 2)
    held_sz = np.sqrt(selfsq[..., :-1] * np.diagonal(g, axis1=2, axis2=3)[..., 1:])
    held = -held_dot / held_sz
    d = np.sqrt(np.diagonal(g, axis1=2, axis2=3))
    a, b = np.triu_indices(NW, 1)
    pair = -g[..., a, b] / (d[..., a] * d[..., b])
    return dict(total=([f"total@{order[i]}" for i in idx], total), size=([f"size@{order[i]}" for i in idx], size),
                held=([f"held@{order[i]}" for i in idx], held), pair=([f"pair:{order[x]}~{order[y]}" for x, y in zip(a, b)], pair))


def extra(k):
    ec = np.full((len(steps), 3, 24), np.nan)
    fo = np.full((len(steps), 3, 4), np.nan)
    for si, s in enumerate(steps):
        for ai, a in enumerate(ARMS):
            i = KEY.get((s, a, k))
            if i is not None:
                ec[si, ai] = D["EC"][i]
                fo[si, ai] = D["FO"][i]
    return dict(closed=([f"closed@L{L}" for L in range(24)], ec), first=(["entropy", "margin", "top_p", "next_surprise"], fo))


def all_reads(k):
    r = reads(gram_at(k))
    r.update(extra(k))
    return r


byk = {k: all_reads(k) for k in (-1, 0, 1, 2, 3, 5, 10, 20, 50)}
win = {f: (byk[1][f][0], np.nanmean(np.stack([byk[k][f][1] for k in (1, 2, 3)]), 0)) for f in byk[1]}


def same(b, r):
    ok = ~(np.isnan(b) | np.isnan(r))
    return ((b > r) + 0.5 * (b == r))[ok].mean() if ok.any() else np.nan


def auc(pos, neg):
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    allv = np.concatenate([pos, neg])
    o = allv.argsort()
    rk = np.empty(len(allv))
    rk[o] = np.arange(1, len(allv) + 1)
    return (rk[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def boot_low(b, r, groups, n=1000, seed=5):
    rng = np.random.default_rng(seed)
    u = np.unique(groups)
    pos = {g: np.where(groups == g)[0] for g in u}
    v = [same(b[idx], r[idx]) for idx in (np.concatenate([pos[g] for g in rng.choice(u, len(u))]) for _ in range(n))]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


B, H = split == "build", split == "held"
best = {}
for f, (names, arr) in win.items():
    sc = np.array([same(arr[B, 1, c], arr[B, 2, c]) for c in range(arr.shape[2])])
    c = int(np.nanargmax(np.abs(sc - 0.5)))
    sign = 1 if sc[c] >= 0.5 else -1
    best[f] = (names[c], c, sign, float(sc[c]))
fb = best["first"]
fb_held = same(fb[2] * win["first"][1][H, 1, fb[1]], fb[2] * win["first"][1][H, 2, fb[1]])
print(f"first-order baseline chosen on build: {fb[0]} sign {fb[2]:+d} build {fb[3]:.3f} held same-step {fb_held:.3f}")
any_pass, any_react = False, False
top = {}
for f in ("total", "size", "held", "pair", "closed"):
    name, c, sign, bsc = best[f]
    arr = sign * win[f][1][..., c]
    s1 = same(arr[H, 1], arr[H, 2])
    lo, hi = boot_low(arr[H, 1], arr[H, 2], prob[H])
    pool_neg = np.concatenate([arr[H, 0], arr[H, 2]])
    s2 = auc(arr[H, 1], pool_neg)
    th = np.nanpercentile(pool_neg, 95)
    catch = float(np.nanmean(arr[H, 1] > th))
    b1, b2, b3 = s1 >= 0.75 and lo > 0.60, s2 >= 0.75 and catch >= 0.5, s1 >= fb_held + 0.05
    rng = np.random.default_rng(9)
    flip = rng.random(H.sum()) < 0.5
    bb, rr = arr[H, 1].copy(), arr[H, 2].copy()
    bb[flip], rr[flip] = arr[H, 2][flip], arr[H, 1][flip]
    verdict = "PASS" if b1 and b2 and b3 else "REACTS, NO ALARM" if b1 and b3 else "FAIL"
    any_pass |= verdict == "PASS"
    any_react |= verdict == "REACTS, NO ALARM"
    top[f] = (c, sign, name)
    print(f"\n{f}: {name} sign {sign:+d} build {bsc:.3f}")
    print(f"  B1 same step broken > reformat {s1:.3f} (95% {lo:.3f}-{hi:.3f}); broken > clean {same(arr[H,1],arr[H,0]):.3f}; reformat > clean {same(arr[H,2],arr[H,0]):.3f}; shuffled {same(bb,rr):.3f}")
    print(f"  B2 pooled {s2:.3f}, catches {catch*100:.0f}% at 5% false alarms; B3 vs baseline {s1-fb_held:+.3f} -> {verdict}")
    print("  time course, same step broken > reformat: " + " ".join(f"w{k}:{same(sign*byk[k][f][1][H,1,c], sign*byk[k][f][1][H,2,c]):.2f}" for k in (0, 1, 2, 3, 5, 10, 20, 50)))
    for lab, mask in (("fresh", H & ~seen), ("already written", H & seen)):
        print(f"  {lab}: n {int(mask.sum())}, same step {same(arr[mask,1],arr[mask,2]):.3f}")
    near = H & (np.abs(surp["broken"] - surp["reformat"]) <= 2)
    print(f"  surprise within 2 bits: n {int(near.sum())}, same step {same(arr[near,1],arr[near,2]):.3f}")
print("\nGOAL:", "PASS" if any_pass else "REACTS, NO ALARM" if any_react else "FAIL")
print(f"\nsurprise at the number, bits: clean {np.nanmedian(surp['clean']):.1f}, broken {np.nanmedian(surp['broken']):.1f}, reformat {np.nanmedian(surp['reformat']):.1f}")
pre = byk[-1]["total"][1]
diff = np.nanmax(np.abs(pre[:, 1] - pre[:, 0]) / np.maximum(np.abs(pre[:, 0]), 1e-6), axis=0)
print(f"check, word before the number, total read relative difference, broken vs clean: median over layers {np.median(diff):.5f}, worst {np.max(diff):.5f}")
print("top 12 pair candidates on build:")
names, arr = win["pair"]
sc = np.array([same(arr[B, 1, c], arr[B, 2, c]) for c in range(arr.shape[2])])
for c in np.argsort(-np.abs(sc - 0.5))[:12]:
    print(f"  {names[c]} build {sc[c]:.3f} held {same(arr[H,1,c],arr[H,2,c]):.3f}")
