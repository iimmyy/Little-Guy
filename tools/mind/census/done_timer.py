#!/usr/bin/env python3
"""Step 1 of the done signal (Immy 2026-11-03: "I'm done"; urgency gating, Cisek: commit when evidence x urgency
crosses a bound). Reading only, nothing changed. Does c5's stream already hold the URGENCY half: how long he has been
going since he first stated the right answer?
Material: his runs on the night TRAIN split (build split, never the test 100), whole-number answers, where the right
answer is stated somewhere in the thinking (first statement found like fit_test.fork_of). Read at paragraph ends
(the token holding a blank line): up to `per` before the first right statement and up to `per` after it, per run.
Target: log(1 + words since the first right statement), 0 before it. Ridge readout from the stream at a few depths,
folds by problem, strength on inner folds.
Controls (what a timer-free model also knows): log(1 + words so far in the thinking), i.e. plain position; the stream's
readout ADDED to position. The real question: does the stream know the time since the answer beyond the clock?
Also: does the readout separate the long tail of a loop (after > 600 words) from a normal wrap-up (after < 200)?
usage: done_timer.py <runs.jsonl> [...] <out.npz> <out.txt> [hf=<model> per=12 layers=8,12,14,16,17,20 dev=cuda]
"""
import json, math, os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from situation_probe import TAIL
from natural_error import whole

args = [a for a in sys.argv[1:] if "=" not in a]
opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
runs, out_npz, out_txt = args[:-2], args[-2], args[-1]
PER, DEV = int(opt.get("per", 12)), opt.get("dev", "cuda")
LAYERS = [int(x) for x in opt.get("layers", "8,12,14,16,17,20").split(",")]
ALPHAS = (1e1, 1e2, 1e3, 1e4, 1e5)


def first_statement(think, A):
    m = re.search(rf"(?:=|\bis\b|\bare\b|\bget\b|\bgives\b|\bequals\b)\s*\$?\\?\(?\s*{A}(?![\d.,/])", think)
    return m.end() if m else None


def capture():
    import torch
    from patients import _lfm3
    p_ = _lfm3(DEV, opt.get("hf"))
    tok, m = p_.tok, p_.model
    cap = {}
    for L in LAYERS:
        m.model.layers[L].register_forward_hook(lambda mod, a, o, L=L: cap.__setitem__(L, (o[0] if isinstance(o, tuple) else o)[0]))
    rng = np.random.RandomState(0)
    X, Y, P, G, K, AFT = [], [], [], [], [], []
    for f in runs:
        for t in map(json.loads, open(f)):
            A = whole(str(t["answer"]))
            if A is None or A < 2:
                continue
            th = t["thinking"]
            st = first_statement(th, A)
            if st is None:
                continue
            kind = 1 if t["finish"] != "stop" else 0           # 1 = cut off, 0 = handed in
            head = tok.apply_chat_template([{"role": "user", "content": t["q"] + TAIL}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=True) + "\n"
            enc = tok(head + th, add_special_tokens=False, return_offsets_mapping=True)
            ids, offs = enc.input_ids, enc.offset_mapping
            h0 = len(head)
            ends = [i for i, (a, b) in enumerate(offs) if a >= h0 and "\n\n" in (head + th)[a:b]]
            if not ends:
                continue
            words_at = lambda c: len(re.findall(r"\S+", th[:max(0, c - h0)]))
            w_st = words_at(h0 + st)
            pre = [i for i in ends if offs[i][0] < h0 + st]
            post = [i for i in ends if offs[i][0] >= h0 + st]
            pick = sorted(rng.choice(pre, min(PER, len(pre)), replace=False).tolist()) if pre else []
            pick += sorted(rng.choice(post, min(PER, len(post)), replace=False).tolist()) if post else []
            if not pick:
                continue
            with torch.no_grad():
                m.model(input_ids=torch.tensor([ids], device=DEV))
            for i in pick:
                w = words_at(offs[i][1])
                aft = max(0, w - w_st) if offs[i][0] >= h0 + st else 0
                X.append(np.stack([cap[L][i].float().cpu().numpy() for L in LAYERS]).astype(np.float16))
                Y.append(math.log1p(aft)); P.append(math.log1p(w)); G.append(t["id"]); K.append(kind); AFT.append(aft)
            print(f"{len(X)} points", flush=True) if len(X) % 200 < len(pick) else None
    np.savez(out_npz, X=np.stack(X), Y=np.array(Y), P=np.array(P), G=np.array(G), K=np.array(K), AFT=np.array(AFT), layers=np.array(LAYERS))


def fit_pred(Xa, ya, Xb, al):
    mu, sd = Xa.mean(0), Xa.std(0) + 1e-6
    Aa, Bb = (Xa - mu) / sd, (Xb - mu) / sd
    m = ya.mean()
    c = np.linalg.solve(Aa @ Aa.T + al * np.eye(len(Aa)), ya - m)
    return (Bb @ Aa.T) @ c + m


def cv(Xf, y, F):
    pred = np.zeros(len(y))
    for f in range(5):
        tr, te = F != f, F == f
        best = None
        for al in ALPHAS:
            err = sum(((fit_pred(Xf[tr & (F != g)], y[tr & (F != g)], Xf[tr & (F == g)], al) - y[tr & (F == g)]) ** 2).sum() for g in range(5) if g != f)
            if best is None or err < best[0]:
                best = (err, al)
        pred[te] = fit_pred(Xf[tr], y[tr], Xf[te], best[1])
    return pred


def auroc(pos, neg):
    allv = np.concatenate([pos, neg]); r = np.empty(len(allv)); r[np.argsort(allv)] = np.arange(1, len(allv) + 1)
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def read():
    z = np.load(out_npz)
    X, Y, P, G, K, AFT, LS = z["X"], z["Y"], z["P"], z["G"], z["K"], z["AFT"], z["layers"]
    ug = np.unique(G)
    fold = {g: i % 5 for i, g in enumerate(np.random.RandomState(0).permutation(ug))}
    F = np.array([fold[g] for g in G])
    r2 = lambda p: 1 - ((p - Y) ** 2).sum() / ((Y - Y.mean()) ** 2).sum()
    post = AFT > 0
    L = [f"done timer: {len(Y)} paragraph ends from {len(ug)} problems ({(K == 1).sum()} points in cut-off runs); "
         f"after the first right statement {post.sum()} (loop tail > 600 words: {(AFT > 600).sum()}, wrap-up < 200: {(post & (AFT < 200)).sum()})"]
    pp = cv(P[:, None], Y, F)
    L.append(f"  control, position alone: R^2 {r2(pp):.3f}")
    for li, Lr in enumerate(LS):
        Xf = X[:, li].astype(np.float64)
        ps = cv(Xf, Y, F)
        pc = cv(np.concatenate([Xf, P[:, None] * 10], 1), Y, F)
        sep = auroc(ps[AFT > 600], ps[post & (AFT < 200)]) if (AFT > 600).any() and (post & (AFT < 200)).any() else float("nan")
        res = Y - pp
        rs = cv(Xf, res, F)
        L.append(f"  layer {Lr:2d}: stream R^2 {r2(ps):.3f}, stream + position {r2(pc):.3f}; past-the-clock (stream reads what position "
                 f"misses) R^2 {1 - ((rs - res) ** 2).sum() / ((res - res.mean()) ** 2).sum():.3f}; "
                 f"answer-found bit AUROC {auroc(ps[post], ps[~post]):.2f}; loop tail vs wrap-up AUROC {sep:.2f}")
        print(L[-1], flush=True)
    open(out_txt, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    if not os.path.exists(out_npz):
        capture()
    read()
