#!/usr/bin/env python3
"""His own wall crossings (Immy's tickle point, 2026-11-03: you cannot surprise yourself). A planted "12.5" surprises him; a 3.5 he worked out does not: on trap problems the whole-number wall read +0.83 on planted and +0.09 on his own.
So get crossings from his own mouth, with nothing planted in his thinking. Build-material word problems with the numbers nudged so a division no longer comes out even: "140 stones in bags of 14" becomes "143 stones in bags of 14".
  prepare: write the nudged problems (countable and divisible-unit questions, split by id, never a test set).
  cap: after running them, read his stream at the line where he first states a non-whole result, at layers 16 and 17. A line fit on these can only hear a wall that does not rely on surprise.
usage: wall_self.py prepare <problems.jsonl> <out.jsonl> [reps=1 unit_reps=3]
       wall_self.py cap <hf> <runs.jsonl> <out.npz>
       wall_self.py fit <cap.npz> <mat.npz> <out lines.npz> <out.txt> [layer=16]
"""
import hashlib, json, os, random, re, sys
import numpy as np

UNITS = {"hours", "minutes", "seconds", "days", "weeks", "months", "years", "miles", "kilometers", "km", "meters", "feet", "inches", "yards", "centimeters", "cm", "dollars", "cents", "cups", "liters", "litres",
         "gallons", "ounces", "pounds", "kilograms", "kg", "grams", "tons", "acres", "percent", "degrees", "more", "fewer", "less", "times", "square", "cubic", "ml", "quarts", "pints", "mph", "lbs", "oz"}
ASK = re.compile(r"[Hh]ow many ([a-zA-Z-]+)")
NUM = re.compile(r"(?<![\d.,])\b(\d{1,5})\b(?![\d.,])")
ST = re.compile(r"(?:=|\bis\b|\bare\b|\bget\b|\bgives\b|\bequals\b|≈|about|gets?)\s*\$?\\?\(?\s*(\d+\.\d+|\d+\s+\d+/\d+|\d+\s*\\d?frac\{\d+\}\{\d+\}|\\d?frac\{\d+\}\{\d+\})(?![\d/])")
H = lambda i: int(hashlib.md5(str(i).encode()).hexdigest(), 16)


def group(q):
    m = ASK.search(q)
    return ("unit" if m.group(1).lower() in UNITS else "count") if m else None


mode = sys.argv[1]
if mode == "prepare":
    src, out = sys.argv[2:4]
    opt = dict(a.split("=", 1) for a in sys.argv[4:] if "=" in a)
    R, RU = int(opt.get("reps", 1)), int(opt.get("unit_reps", 3))
    rng = random.Random(0)
    n = 0
    with open(out, "w") as fo:
        for t in map(json.loads, open(src)):
            g = group(t["q"])
            ms = list(NUM.finditer(t["q"]))
            if g is None or len(ms) < 2 or not str(t.get("kind", "")).startswith("mathworld"):
                continue
            for k in range(RU if g == "unit" else R):
                big = max(ms, key=lambda x: int(x.group(1)))
                others = [int(x.group(1)) for x in ms if x is not big and int(x.group(1)) > 1]
                v = int(big.group(1))
                if not others or v < 5:
                    continue
                d = rng.choice(others)
                w = v + 1 + rng.randrange(3)
                while w % d == 0:
                    w += 1
                q = t["q"][:big.start()] + str(w) + t["q"][big.end():]
                fo.write(json.dumps({"id": f"{t['id']}~{k}", "q": q, "answer": "", "cat": "wall-self", "grp": g, "src_id": t["id"]}) + "\n")
                n += 1
    print("wrote", n)
elif mode == "cap":
    import torch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from patients import _lfm3
    hf, runs, outz = sys.argv[2:5]
    p_ = _lfm3("cuda", hf)
    tok, m = p_.tok, p_.model
    LAY = (16, 17)
    got = {}
    for L_ in LAY:
        m.model.layers[L_].feed_forward.register_forward_pre_hook(lambda mod, a, L_=L_: got.__setitem__(L_, a[0][0, -1].float().cpu().numpy()))
    X = {L_: [] for L_ in LAY}
    G, P, S, FIN = [], [], [], []
    for t in map(json.loads, open(runs)):
        mm = ST.search(t["thinking"] or "")
        g = group(t["q"])
        if not mm or g is None:
            continue
        e = re.search(r"[.!?\n]", t["thinking"][mm.end(1):])
        cut = mm.end(1) + (e.end() if e else 0)
        head = tok.apply_chat_template([{"role": "user", "content": t["q"]}], tokenize=False, add_generation_prompt=True, enable_thinking=True) + "\n"
        with torch.no_grad():
            m.model(input_ids=torch.tensor([tok(head + t["thinking"][:cut], add_special_tokens=False).input_ids], device="cuda"))
        for L_ in LAY:
            X[L_].append(got[L_].astype(np.float32))
        G.append(g); P.append(t.get("src_id", t["id"].split("~")[0])); S.append(t["thinking"][max(0, mm.start() - 50):cut][-90:]); FIN.append(t["finish"] == "stop")
    np.savez(outz, **{f"x{L_}": np.stack(X[L_]) for L_ in LAY}, grp=np.array(G), pid=np.array(P), snip=np.array(S), fin=np.array(FIN))
    print("CAP_DONE", len(G), {g: G.count(g) for g in set(G)})
elif mode == "fit":
    capf, matf, outz, outt = sys.argv[2:6]
    opt = dict(a.split("=", 1) for a in sys.argv[6:] if "=" in a)
    LK = int(opt.get("layer", 16))
    c, z = np.load(capf, allow_pickle=True), np.load(matf, allow_pickle=True)
    Xs, gs, ps = c[f"x{LK}"].astype(np.float64), c["grp"], c["pid"]
    Xm, ver, at, pm, gm = z[f"x{LK}"].astype(np.float64), z["ver"], z["at"], z["pid"], z["grp"]
    keep = (at == 2) | (at == 3) | ((at == 0) & np.isin(ver, [0, 1, 2, 3]))      # his right number, wrong whole numbers, ordinary, reply: all zero
    X = np.concatenate([Xs, Xm[keep]])
    y = np.concatenate([(gs == "count").astype(float), np.zeros(keep.sum())])
    P = np.concatenate([ps, pm[keep]])
    src = np.concatenate([np.where(gs == "count", 0, 1), np.where(at[keep] == 0, 2, 3)])
    w = np.array([{0: 3.0, 1: 3.0, 2: 1.0, 3: 0.5}[k] for k in src])
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
    for al in (1e3, 1e4, 1e5):
        e = 0
        for f in range(4):
            tr, te = fold[idx] != f, fold[idx] == f
            co = ridge(X[idx[tr]], y[idx[tr]], w[idx[tr]], al)
            e += (w[idx[te]] * (X[idx[te]] @ co - y[idx[te]]) ** 2).sum()
        if best is None or e < best[0]:
            best = (e, al)
    co = ridge(X[idx], y[idx], w[idx], best[1])
    sc = X @ co
    L = [f"own wall line, layer {LK}, alpha {best[1]:.0e}: his own non-whole statements, countable {(src == 0).sum()}, unit {(src == 1).sum()}; held out:"]
    ho = ~bd
    for nm, k in (("own non-whole, countable", 0), ("own non-whole, unit", 1), ("stated whole numbers", 2), ("ordinary + reply", 3)):
        L.append(f"  {nm:26s} n {(ho & (src == k)).sum():4d} median {np.median(sc[ho & (src == k)]):+.3f}")
    L.append(f"  AUROC own crossing vs own unit control {auc(sc[ho & (src == 0)], sc[ho & (src == 1)]):.3f}; vs stated whole {auc(sc[ho & (src == 0)], sc[ho & (src == 2)]):.3f}; vs ordinary {auc(sc[ho & (src == 0)], sc[ho & (src == 3)]):.3f}")
    # same layer's other lines are kept; only the whole wall is replaced
    base = dict(np.load(opt["lines"])) if "lines" in opt else {}
    base["whole"] = co
    np.savez(outz, **base)
    open(outt, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
