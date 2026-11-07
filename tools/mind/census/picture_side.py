#!/usr/bin/env python3
"""Does he already hold the stick-figure picture? (Immy 2026-11-02: his picture has no numbers in it, only relations:
"stuffed, food left over, so fewer than 12". Numbers come last. Untouched model, reading only.)

B  the picture's verdict before thinking: at the question mark and at the first thinking word (last prompt token),
   is the answer's SIDE readable: below every number in the story / between them / above every number?
   Keyword control: the same reader on a bag of words of the text (does the question's wording alone give it away?).
C  use on arrival: at each number word after the first, is "bigger or smaller than the number before it" readable,
   layer by layer (before the layer-14 grab or after)? Control: a reader that sees only the number's own size.
Readers: ridge, folds by problem, ridge strength picked inside the training folds only. Chance and a label-shuffle
null (best layer over all layers, so picking the best layer is paid for) are printed next to every result.

usage: picture_side.py capture <problems.jsonl> <out.npz> [hf= dev=cuda] [extra=<jsonl of extra items to score>]
       picture_side.py read <capture.npz> <out.txt> [shuffles=200 shuffles_c= layer=12 old=<first set.jsonl>]
"""
import json, math, os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from situation_probe import TAIL, question_only

NUM = re.compile(r"(?<![\d.])\d[\d,]*(?:\.\d+)?(?![\d])")


def side(text, answer):
    g = [float(x.replace(",", "")) for x in NUM.findall(text)]
    if not g:
        return None
    a = float(answer)
    return 0 if a < min(g) else 2 if a > max(g) else 1        # below all / between (or equal) / above all


def capture(src, out, opt):
    import torch
    from patients import _lfm3
    dev = opt.get("dev", "cuda")
    p = _lfm3(dev, opt.get("hf"))
    tok, m = p.tok, p.model
    cap = []
    for l in m.model.layers:
        l.register_forward_hook(lambda mod, a, o: cap.append((o[0] if isinstance(o, tuple) else o)[0]))
    m.model.embed_tokens.register_forward_hook(lambda mod, a, o: cap.insert(0, o[0]))
    rows = [(r, "set") for r in map(json.loads, open(src))]
    if opt.get("extra"):
        rows += [({**r, "text": r.get("text") or r["q"]}, "extra") for r in map(json.loads, open(opt["extra"]))]
    QS, FT, NX, meta, nmeta = [], [], [], [], []
    for pi, (r, grp) in enumerate(rows):
        lab = side(r["text"], r["answer"])
        if lab is None:
            continue
        chat = tok.apply_chat_template([{"role": "user", "content": r["text"] + TAIL}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=True)
        enc = tok(chat, add_special_tokens=False, return_offsets_mapping=True)
        ids, offs = enc.input_ids, enc.offset_mapping
        t0 = chat.find(r["text"])
        at = lambda c: max(i for i, (s, e) in enumerate(offs) if s <= c and e > s)     # token holding char c
        qm = r["text"].rfind("?")
        q_tok = at(t0 + (qm if qm >= 0 else len(r["text"]) - 1))
        cap.clear()
        with torch.no_grad():
            m.model(input_ids=torch.tensor([ids], device=dev))
        H = torch.stack(cap).float().cpu().numpy()                    # layer index x word x width
        QS.append(H[:, q_tok].astype(np.float16)); FT.append(H[:, -1].astype(np.float16))
        meta.append((pi, r["id"], grp, lab, r["text"]))
        prev = None
        for mm in NUM.finditer(r["text"]):
            v = float(mm[0].replace(",", ""))
            if prev is not None and v != prev and grp == "set":
                NX.append(H[:, at(t0 + mm.end() - 1)].astype(np.float16))
                nmeta.append((pi, int(v > prev), math.log(v + 1), math.log(prev + 1)))
            prev = v
        print(f"{pi + 1}/{len(rows)} {r['id']} side {lab} number pairs so far {len(NX)}", flush=True)
    np.savez(out, QS=np.stack(QS), FT=np.stack(FT), NX=np.stack(NX),
             prob=np.array([x[0] for x in meta]), ids=np.array([x[1] for x in meta]), grp=np.array([x[2] for x in meta]),
             lab=np.array([x[3] for x in meta]), text=np.array([x[4] for x in meta]),
             nprob=np.array([x[0] for x in nmeta]), nlab=np.array([x[1] for x in nmeta]),
             nown=np.array([x[2] for x in nmeta]), nprev=np.array([x[3] for x in nmeta]))
    print(f"saved {len(QS)} problems, {len(NX)} number pairs to {out}")


def fit_pred(Xa, Ya, Xb, al):
    """dual ridge on standardized features; Y one-hot (or one column), centered"""
    mu, sd = Xa.mean(0), Xa.std(0) + 1e-6
    A, B = (Xa - mu) / sd, (Xb - mu) / sd
    K = A @ A.T
    W = np.linalg.solve(K + al * np.eye(len(A)), Ya - Ya.mean(0))
    return B @ A.T @ W + Ya.mean(0)


ALPHAS = (1e0, 1e1, 1e2, 1e3, 1e4, 1e5)


def cv_acc_many(X, ys, groups, n_cls, folds=4):
    """cv_acc for many label vectors on the same X at once (the true labels and the shuffles): every split's
    standardised kernel is decomposed once (K = Q diag(lam) Q^T), then each label vector and each ridge strength is a
    cheap product. Same folds, same strength choice, same predictions as calling cv_acc on each y."""
    ug = np.unique(groups)
    fold_of = {g: i % folds for i, g in enumerate(ug)}
    f = np.array([fold_of[g] for g in groups])
    Ys = [np.eye(n_cls)[y] for y in ys]

    def split(tr, te):
        Xa, Xb = X[tr], X[te]
        mu, sd = Xa.mean(0), Xa.std(0) + 1e-6
        A, B = (Xa - mu) / sd, (Xb - mu) / sd
        lam, Q = np.linalg.eigh(A @ A.T)
        KQ = (B @ A.T) @ Q
        def pred(Y, al):
            m = Y[tr].mean(0)
            return KQ @ ((Q.T @ (Y[tr] - m)) / (lam + al)[:, None]) + m
        return pred

    preds = [np.zeros(len(y), int) for y in ys]
    for k in range(folds):
        tr, te = f != k, f == k
        inner = [(split(tr & (f != j), f == j), f == j) for j in range(folds) if j != k]
        final = split(tr, te)
        for yi, (y, Y) in enumerate(zip(ys, Ys)):
            best, best_al = -1, ALPHAS[0]
            for al in ALPHAS:
                ok = sum((pf(Y, al).argmax(1) == y[ite]).sum() for pf, ite in inner)
                if ok > best:
                    best, best_al = ok, al
            preds[yi][te] = final(Y, best_al).argmax(1)
    return [((p == y).mean(), p) for p, y in zip(preds, ys)]


def cv_acc(X, y, groups, n_cls, folds=4):
    """outer folds by problem; ridge strength chosen by inner folds on the training part only"""
    return cv_acc_many(X, [y], groups, n_cls, folds)[0]


def bag_of_words(texts):
    vocab = sorted({w for t in texts for w in re.findall(r"[a-z]+", t.lower())})
    ix = {w: i for i, w in enumerate(vocab)}
    X = np.zeros((len(texts), len(vocab)))
    for i, t in enumerate(texts):
        for w in re.findall(r"[a-z]+", t.lower()):
            X[i, ix[w]] += 1
    return X


def number_feats(texts):
    """what the given numbers alone could tell: how many, their sizes in order, spread, does one divide another"""
    F = []
    for t in texts:
        g = [float(x.replace(",", "")) for x in NUM.findall(t)]
        lg = [math.log1p(v) for v in g]
        srt = sorted(lg) + [0.0] * 4
        F.append([len(g), len(set(g)), min(lg), max(lg), np.mean(lg), lg[0], lg[-1], *srt[:4], max(lg) - min(lg),
                  float(any(b and a % b == 0 and a != b for a in g for b in g)), float(all(v == int(v) for v in g))])
    return np.array(F)


def read(cap_f, out, opt):
    Z = np.load(cap_f)
    sh = int(opt.get("shuffles", 200))
    rng = np.random.default_rng(0)
    L = [f"picture side, {cap_f.rsplit('/', 1)[-1]}"]
    s = Z["grp"] == "set"
    lab, prob = Z["lab"][s], Z["prob"][s]
    cnt = np.bincount(lab, minlength=3)
    chance = cnt.max() / cnt.sum()
    L.append(f"B: answer side, {s.sum()} problems: below all {cnt[0]}, between {cnt[1]}, above all {cnt[2]}; "
             f"chance (always say the biggest group) {chance:.3f}")
    texts = list(Z["text"][s])
    new_only = ~np.isin(Z["ids"][s], [json.loads(l)["id"] for l in open(opt["old"])]) if opt.get("old") else None
    controls = {"words, whole text": bag_of_words(texts), "words, question sentence": bag_of_words([question_only(t) for t in texts]),
                "numbers only": number_feats(texts)}
    controls["words + numbers"] = np.concatenate([controls["words, whole text"], controls["numbers only"]], 1)
    cpred = {}
    for k, Xc in controls.items():
        a, cpred[k] = cv_acc(Xc, lab, prob, 3)
        L.append(f"  control, {k:26s}: {a:.3f}" + (f"  (new problems only {(cpred[k] == lab)[new_only].mean():.3f})" if new_only is not None else ""))
    best_c = max(cpred, key=lambda k: (cpred[k] == lab).mean())
    nlay = Z["QS"].shape[1]
    pre = int(opt.get("layer", 12))           # chosen on the first 93 (v1) before this set was read
    for site in ("QS", "FT"):
        X = Z[site][s].astype(np.float32)
        shuf = [rng.permutation(lab) for _ in range(sh)]
        allr = [cv_acc_many(X[:, li], [lab] + shuf, prob, 3) for li in range(nlay)]
        res = [r[0] for r in allr]
        accs = [r[0] for r in res]
        null = [max(allr[li][1 + si][0] for li in range(nlay)) for si in range(sh)]
        bl = int(np.argmax(accs))
        name = "question mark" if site == "QS" else "first thinking word"
        L.append(f"  at the {name}: best layer index {bl} acc {accs[bl]:.3f}; shuffled-label best-of-layers null "
                 f"95th pct {np.percentile(null, 95):.3f} (p {np.mean(np.array(null) >= accs[bl]):.3f})")
        L.append("    by layer index: " + " ".join(f"{li}:{a:.2f}" for li, a in enumerate(accs)))
        pm = res[pre][1]
        for tag, sel in [("all", np.ones(len(lab), bool))] + ([("new only", new_only)] if new_only is not None else []):
            d = (pm == lab)[sel].astype(float) - (cpred[best_c] == lab)[sel].astype(float)
            bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
            L.append(f"    pre-chosen layer {pre}, {tag} ({sel.sum()}): model {(pm == lab)[sel].mean():.3f} vs best control "
                     f"'{best_c}' {(cpred[best_c] == lab)[sel].mean():.3f}; gap {d.mean():+.3f} (95% {np.percentile(bs, 2.5):+.3f} to {np.percentile(bs, 97.5):+.3f})")
        rec = [((pm == lab) & (lab == c)).sum() / max((lab == c).sum(), 1) for c in range(3)]
        L.append(f"    layer {pre} right per side: below {rec[0]:.2f}, between {rec[1]:.2f}, above {rec[2]:.2f}")
        e = Z["grp"] == "extra"
        if e.any():
            Y = np.eye(3)[lab]
            for li in (pre, 15):
                pr = fit_pred(X[:, li], Y, Z[site][e][:, li].astype(np.float32), 1e3)
                for i, row in zip(np.where(e)[0], pr):
                    L.append(f"    extra {Z['ids'][i]} (true side {Z['lab'][i]}) layer {li}: read side {int(row.argmax())} "
                             f"scores below/between/above {row[0]:.2f}/{row[1]:.2f}/{row[2]:.2f}")
    sh = int(opt.get("shuffles_c", sh))
    nl, npb = Z["nlab"], Z["nprob"]
    c = np.bincount(nl, minlength=2)
    L.append(f"C: number pairs {len(nl)} (bigger {c[1]}, smaller {c[0]}); chance {c.max() / c.sum():.3f}")
    own = cv_acc(Z["nown"][:, None], nl, npb, 2)[0]
    both = cv_acc(np.stack([Z["nown"], Z["nprev"]], 1), nl, npb, 2)[0]
    L.append(f"  control, own size only: {own:.3f}   (ceiling, own size + the earlier number's size given: {both:.3f})")
    X = Z["NX"].astype(np.float32)
    shuf = [rng.permutation(nl) for _ in range(sh)]
    allr = [cv_acc_many(X[:, li], [nl] + shuf, npb, 2) for li in range(nlay)]
    accs = [allr[li][0][0] for li in range(nlay)]
    null = [max(allr[li][1 + si][0] for li in range(nlay)) for si in range(sh)]
    bl = int(np.argmax(accs))
    L.append(f"  at the number word: best layer index {bl} acc {accs[bl]:.3f}; shuffled null 95th pct "
             f"{np.percentile(null, 95):.3f} (p {np.mean(np.array(null) >= accs[bl]):.3f})")
    L.append("    by layer index: " + " ".join(f"{li}:{a:.2f}" for li, a in enumerate(accs)))
    open(out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    cmd, args = sys.argv[1], [a for a in sys.argv[2:] if "=" not in a]
    opt = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)
    {"capture": lambda: capture(args[0], args[1], opt), "read": lambda: read(args[0], args[1], opt)}[cmd]()
