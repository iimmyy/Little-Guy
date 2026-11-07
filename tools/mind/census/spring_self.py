#!/usr/bin/env python3
"""His own wrong numbers (Immy's tickle point, for the springs). Lines learned on swapped numbers do not hear his own errors (0.30 on real trap mistakes). The wall heard them once it was learned from his own mouth (0.91).
Same move for the springs: get wrong numbers he works out himself, on build material only.
  prepare: word problems with one irrelevant sentence added (a number that has nothing to do with the question). Same key. He sometimes uses it.
  cap: read his stream at the line where he first states his final answer, wrong or right, plus his wrong mid-run answers.
  fit: a line for "this stated number is one of his own wrong ones", with whole problems held out.
usage: spring_self.py prepare <problems.jsonl> <out.jsonl>
       spring_self.py cap <hf> <out.npz> <runs.jsonl> [...]
       spring_self.py fit <cap.npz> <mat.npz> <lines in.npz> <lines out.npz> <out.txt> [layer=16]
"""
import hashlib, json, os, random, re, sys
import numpy as np

ASK = re.compile(r"[Hh]ow many ([a-zA-Z-]+)")
H = lambda i: int(hashlib.md5(str(i).encode()).hexdigest(), 16)
NAMES = ("Her neighbor", "A friend", "His cousin", "Their teacher", "The shop next door", "Someone down the street")
mode = sys.argv[1]
if mode == "prepare":
    src, out = sys.argv[2:4]
    rng = random.Random(1)
    n = 0
    with open(out, "w") as fo:
        for t in map(json.loads, open(src)):
            m = ASK.search(t["q"])
            if not m or not str(t.get("kind", "")).startswith("mathworld") or not re.fullmatch(r"\d+", str(t["answer"]).strip()):
                continue
            k = t["q"].rfind(". ", 0, m.start())
            if k < 0:
                continue
            nums = [int(x) for x in re.findall(r"\b\d{1,4}\b", t["q"])]
            v = rng.choice(nums) + rng.randrange(2, 12) if nums else rng.randrange(3, 40)
            s = f" {rng.choice(NAMES)} has {v} {m.group(1).lower()}."
            q = t["q"][:k + 1] + s + t["q"][k + 1:]
            fo.write(json.dumps({"id": t["id"] + "~d", "q": q, "answer": t["answer"], "cat": "spring-self", "src_id": t["id"]}) + "\n")
            n += 1
    print("wrote", n)
elif mode == "cap":
    import torch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from patients import _lfm3
    hf, outz, runs = sys.argv[2], sys.argv[3], sys.argv[4:]
    p_ = _lfm3("cuda", hf)
    tok, m = p_.tok, p_.model
    LAY = (16, 17)
    got = {}
    for L_ in LAY:
        m.model.layers[L_].feed_forward.register_forward_pre_hook(lambda mod, a, L_=L_: got.__setitem__(L_, a[0][0, -1].float().cpu().numpy()))
    X = {L_: [] for L_ in LAY}
    Y, P, S, ER = [], [], [], []
    rng = random.Random(0)
    for f in runs:
        for t in map(json.loads, open(f)):
            s = str(t["answer"]).strip().replace(",", "")
            if t["finish"] != "stop" or not t["thinking"] or not re.fullmatch(r"\d+", s):
                continue
            b = re.findall(r"\\boxed\{\s*\$?(\d[\d,]*)\s*\}", t["reply"]) or re.findall(r"\\boxed\{\s*\$?(\d[\d,]*)\s*\}", t["thinking"])
            if not b:
                continue
            got_v = int(b[-1].replace(",", ""))
            A = int(s)
            wrong = got_v != A
            if not wrong and rng.random() > 0.2:             # a sample of the right ones is plenty
                continue
            ms = list(re.finditer(rf"(?:=|\bis\b|\bare\b|\bget\b|\bgives\b|\bequals\b|\btotal of\b|answer:?)\s*\$?\\?\(?\s*(?:\\boxed\{{)?({got_v})(?![\d.,/])", t["thinking"]))
            if not ms:
                continue
            e = re.search(r"[.!?\n]", t["thinking"][ms[0].end(1):])
            cut = ms[0].end(1) + (e.end() if e else 0)
            head = tok.apply_chat_template([{"role": "user", "content": t["q"]}], tokenize=False, add_generation_prompt=True, enable_thinking=True) + "\n"
            with torch.no_grad():
                m.model(input_ids=torch.tensor([tok(head + t["thinking"][:cut], add_special_tokens=False).input_ids], device="cuda"))
            for L_ in LAY:
                X[L_].append(got[L_].astype(np.float32))
            Y.append(wrong); P.append(t.get("src_id", t["id"])); S.append(t["thinking"][max(0, ms[0].start() - 50):cut][-90:]); ER.append(abs(np.log(max(got_v, 1) / max(A, 1))))
    np.savez(outz, **{f"x{L_}": np.stack(X[L_]) for L_ in LAY}, y=np.array(Y), pid=np.array(P), snip=np.array(S), err=np.array(ER))
    print("CAP_DONE", len(Y), "wrong", sum(Y))
elif mode == "fit":
    capf, matf, linf, outz, outt = sys.argv[2:7]
    opt = dict(a.split("=", 1) for a in sys.argv[7:] if "=" in a)
    LK = int(opt.get("layer", 16))
    c, z = np.load(capf, allow_pickle=True), np.load(matf, allow_pickle=True)
    Xs, ys, ps = c[f"x{LK}"].astype(np.float64), c["y"], c["pid"]
    Xm, at, pm = z[f"x{LK}"].astype(np.float64), z["at"], z["pid"]
    keep = (at == 2) | (at == 3)
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
    L = [f"own spring line, layer {LK}, alpha {best[1]:.0e}: his own wrong {(k_ == 0).sum()}, own right {(k_ == 1).sum()}; held out wrong {(ho & (k_ == 0)).sum()}, right {(ho & (k_ == 1)).sum()}"]
    if (ho & (k_ == 0)).sum():
        L.append(f"  own wrong median {np.median(sc[ho & (k_ == 0)]):+.3f}, own right {np.median(sc[ho & (k_ == 1)]):+.3f}, ordinary {np.median(sc[ho & (k_ == 2)]):+.3f}; AUROC wrong vs right {auc(sc[ho & (k_ == 0)], sc[ho & (k_ == 1)]):.3f}")
    base = dict(np.load(linf))
    base["size"] = co
    np.savez(outz, **base)
    open(outt, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
