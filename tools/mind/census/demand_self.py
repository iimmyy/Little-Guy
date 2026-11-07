#!/usr/bin/env python3
"""The demand spring from his own work (Immy's tickle point again). The line learned on an earlier number planted as the answer reads backwards on his real mistakes (0.30) and fires on 38% of right statements.
His own runs already hold the material with nothing planted: every line where he states a whole number he worked out himself is either the quantity the question asks for or something else. So the spring is
"what I just stated is not what was asked". Numbers before and after his first right statement both count as "not asked", so depth alone can't do the job. Build problems only, whole problems held out.
usage: demand_self.py cap <hf> <out.npz> <runs.jsonl> [...] [skip=<problems.jsonl> per=3]
       demand_self.py fit <cap.npz> <mat.npz> <lines in.npz> <lines out.npz> <out.txt> [layer=16]
"""
import hashlib, json, os, random, re, sys
import numpy as np

H = lambda i: int(hashlib.md5(str(i).encode()).hexdigest(), 16)
STATED = re.compile(r"(?:=|\bis\b|\bare\b|\bget\b|\bgives\b|\bequals\b|\btotal of\b)\s*\$?\\?\(?\s*(\d{1,6})(?![\d.,/])")
mode = sys.argv[1]
if mode == "cap":
    import torch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from patients import _lfm3
    args = [a for a in sys.argv[2:] if "=" not in a]
    opt = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)
    hf, outz, runs = args[0], args[1], args[2:]
    PER = int(opt.get("per", 3))
    skip = {json.loads(l)["id"] for l in open(opt["skip"])} if "skip" in opt else set()
    p_ = _lfm3("cuda", hf)
    tok, m = p_.tok, p_.model
    LAY = (16, 17)
    got = {}
    for L_ in LAY:
        m.model.layers[L_].feed_forward.register_forward_pre_hook(lambda mod, a, L_=L_: got.__setitem__(L_, a[0][0].float()))
    X = {L_: [] for L_ in LAY}
    Y, P, W, S = [], [], [], []
    rng = random.Random(0)
    nrun = 0
    for f in runs:
        for t in map(json.loads, open(f)):
            s = str(t["answer"]).strip().replace(",", "")
            th = t["thinking"]
            if t["id"] in skip or not th or not re.fullmatch(r"\d+", s) or int(s) < 2:
                continue
            A = int(s)
            right, other = [], []
            first = None
            for mm in STATED.finditer(th):
                if re.match(r"[a-zA-Z^*(\[{_]|\s*[-+*/^\\]", th[mm.end(1):mm.end(1) + 4]):       # "12a", "2^3", "12 + 3": part of an expression. "12 apples" is a stated value.
                    continue
                e = re.search(r"[.!?\n]", th[mm.end(1):])
                if not e:
                    continue
                v, cut = int(mm.group(1)), mm.end(1) + e.end()
                if v == A:
                    first = first if first is not None else cut
                    right.append(cut)
                elif v >= 2:
                    other.append(cut)
            if not right or not other:                                      # a run that holds both, so every comparison is inside one problem's voice
                continue
            pick = [(c, 0) for c in rng.sample(right, min(PER, len(right)))] + [(c, 1) for c in rng.sample(other, min(PER, len(other)))]
            head = tok.apply_chat_template([{"role": "user", "content": t["q"]}], tokenize=False, add_generation_prompt=True, enable_thinking=True) + "\n"
            enc = tok(head + th[:max(c for c, _ in pick)], add_special_tokens=False, return_offsets_mapping=True)
            offs = enc.offset_mapping
            ends = np.array([b for a, b in offs])
            with torch.no_grad():
                m.model(input_ids=torch.tensor([enc.input_ids], device="cuda"))
            for c, y in pick:
                k = int(np.searchsorted(ends, len(head) + c, side="right")) - 1     # the last word at or before the sentence end, read in one forward pass
                for L_ in LAY:
                    X[L_].append(got[L_][k].cpu().numpy().astype(np.float32))
                Y.append(y); P.append(t.get("src_id", t["id"])); W.append(c < first); S.append(th[max(0, c - 70):c])
            nrun += 1
            if nrun % 50 == 0:
                print(nrun, "runs", len(Y), "reads", flush=True)
    np.savez(outz, **{f"x{L_}": np.stack(X[L_]) for L_ in LAY}, y=np.array(Y), pid=np.array(P), before=np.array(W), snip=np.array(S))
    print("CAP_DONE", nrun, "runs", len(Y), "reads, not asked", sum(Y))
elif mode == "fit":
    capf, matf, linf, outz, outt = sys.argv[2:7]
    opt = dict(a.split("=", 1) for a in sys.argv[7:] if "=" in a)
    LK = int(opt.get("layer", 16))
    c, z = np.load(capf, allow_pickle=True), np.load(matf, allow_pickle=True)
    Xs, ys, ps, bf = c[f"x{LK}"].astype(np.float64), c["y"].astype(bool), c["pid"], c["before"]
    Xm, at, pm = z[f"x{LK}"].astype(np.float64), z["at"], z["pid"]
    keep = (at == 2) | (at == 3)                                            # ordinary words and the reply stay at zero, like every other line
    X = np.concatenate([Xs, Xm[keep]])
    y = np.concatenate([ys.astype(float), np.zeros(keep.sum())])
    P = np.concatenate([ps, pm[keep]])
    k_ = np.concatenate([np.where(ys, 0, 1), np.full(keep.sum(), 2)])
    w = np.array([{0: 1.0 / max((k_ == 0).sum(), 1), 1: 1.0 / max((k_ == 1).sum(), 1), 2: 0.3 / max((k_ == 2).sum(), 1)}[i] for i in k_]) * len(k_)
    bd = np.array([H(i) % 3 != 2 for i in P])
    fold = np.array([H(i) % 4 for i in P])

    def ridge(Xb, yb, wb, al):
        sc = Xb.std(0) + 1e-6
        Z = Xb / sc * np.sqrt(wb)[:, None]
        return np.linalg.solve(Z.T @ Z + al * np.eye(Z.shape[1]), Z.T @ (yb * np.sqrt(wb))) / sc

    def auc(a, b):
        s_ = np.concatenate([a, b])
        r = np.argsort(np.argsort(s_)).astype(float) + 1
        return (r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))

    idx = np.where(bd)[0]
    best = None
    for al in (1e3, 1e4, 1e5, 1e6):
        e = 0
        for f in range(4):
            tr, te = fold[idx] != f, fold[idx] == f
            co = ridge(X[idx[tr]], y[idx[tr]], w[idx[tr]], al)
            e += (w[idx[te]] * (X[idx[te]] @ co - y[idx[te]]) ** 2).sum()
        if best is None or e < best[0]:
            best = (e, al)
    co = ridge(X[idx], y[idx], w[idx], best[1])
    sc = X @ co
    ho = ~bd
    a0, a1, a2 = ho & (k_ == 0), ho & (k_ == 1), ho & (k_ == 2)
    n_s = len(ys)
    hs = ho[:n_s]
    L = [f"own demand line, layer {LK}, alpha {best[1]:.0e}: not asked {(k_ == 0).sum()}, asked {(k_ == 1).sum()}; held out {a0.sum()}, {a1.sum()}",
         f"  not asked median {np.median(sc[a0]):+.3f}, asked {np.median(sc[a1]):+.3f}, ordinary {np.median(sc[a2]):+.3f}; AUROC not asked vs asked {auc(sc[a0], sc[a1]):.3f}",
         f"  depth check: numbers after his first right statement vs right statements {auc(sc[:n_s][hs & ys & ~bf], sc[:n_s][hs & ~ys]):.3f}; numbers before it vs right statements {auc(sc[:n_s][hs & ys & bf], sc[:n_s][hs & ~ys]):.3f}"]
    base = dict(np.load(linf))
    base["demand"] = co
    np.savez(outz, **base)
    open(outt, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
