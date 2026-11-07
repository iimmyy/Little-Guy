#!/usr/bin/env python3
"""Does the model build a picture of ANY problem while it reads, not just geometry? (Immy 2026-11-02: "when I start
reading I instantly build a mental image... if it becomes distorted that's my check".) Test 1, measuring only.

Word problems with a sentence-by-sentence scene map (who has how many of what) give the true amount bound to every
owner + thing stated in the story. The untouched model reads each problem (same prompt as our runs) and its notepad
(residual stream after every layer) is kept at a cue added where the thinking starts, read at its last word:
  cue:   "Count of <thing> (<owner>):" after the full problem
  blind: the same cue with the story cut down to its question sentence (it can only guess from the words: the control)
  swap:  the same cue on the story with the stated numbers rotated between the things (same numbers, new owners).
         If the notepad binds amounts to things, the read follows the number to its new owner.
  answer: "Answer:" after the full problem, before any thinking (does the picture already hold the answer?)
Amounts are read as log(amount).

usage: situation_probe.py capture <problems.jsonl> <out.npz> [hf=] [dev=mps] [limit=]
       situation_probe.py read <capture.npz> <out.txt>
       situation_probe.py codes <capture.npz> <codes.npz> [hf=] [dev=mps]
       situation_probe.py read_code <capture.npz> <codes.npz> <out.txt> [rank=k] [layers=14,15]
       situation_probe.py heads <problems.jsonl> <codes.npz> <out.txt> [layer=14] [sets=15,8,13,1;all;mlp] [mode=mean] [hf=] [dev=mps]
"""
import json, math, os, re, sys
import numpy as np

TAIL = "\n\nEnd with a line: Answer: <your answer>"


def stated(r):
    """[(cue, amount)] for every amount written as digits in the story: amounts a thing holds ("Count of <thing>
    (<owner>):") and numbers on links ("Number linking <a> and <b>:"), one cue wording for both sources. First time
    each cue appears; a number already used by an earlier cue is not used again (one fact per written number)."""
    out, seen, used = [], set(), set()
    def add(cue, v):
        try:
            a = float(str(v).replace(",", ""))
        except ValueError:
            return
        if a <= 0 or a != int(a) or cue in seen or int(a) in used or not re.search(rf"(?<![\d.,]){int(a)}(?![\d])", r["text"]):
            return
        seen.add(cue); used.add(int(a))
        out.append((cue, int(a)))
    for s in r["sentences"]:
        for t in s["things"]:
            if str(t["thing"]).strip():
                add(f"Count of {t['thing'].strip()} ({t['owner'].strip()}):", t["amount"])
        for l in s.get("links", []):
            if str(l.get("from", "")).strip():
                add(f"Number linking {str(l['from']).strip()} and {str(l['to']).strip()}:", l.get("value", ""))
    return out


def swapped(r, things):
    """story with the stated amounts rotated one place between things (only if each number is written once and they differ)"""
    nums = [a for _, a in things]
    if len(set(nums)) < 2 or any(len(re.findall(rf"(?<![\d.,]){a}(?![\d])", r["text"])) != 1 for a in nums):
        return None, None
    rot = nums[1:] + nums[:1]
    to = dict(zip(map(str, nums), map(str, rot)))
    text = re.sub(r"(?<![\d.,])\d+(?![\d])", lambda mm: to.get(mm[0], mm[0]), r["text"])   # one pass: no re-replacing
    return text, rot


def question_only(text):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return parts[-1]


def capture(src, out, opt):
    import torch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from patients import _lfm3
    rows = list(map(json.loads, open(src)))[:int(opt.get("limit", 10 ** 9))]
    dev = opt.get("dev", "mps")
    p = _lfm3(dev, opt.get("hf"))
    tok, m = p.tok, p.model
    cap = []
    hooks = [l.register_forward_hook(lambda mod, a, o: cap.append((o[0] if isinstance(o, tuple) else o)[0]))
             for l in m.model.layers]
    m.model.embed_tokens.register_forward_hook(lambda mod, a, o: cap.insert(0, o[0]))
    prompt = lambda q: tok(tok.apply_chat_template([{"role": "user", "content": q + TAIL}], tokenize=False,
                                                   add_generation_prompt=True, enable_thinking=True),
                           add_special_tokens=False).input_ids
    def last(ids):
        cap.clear()
        with torch.no_grad():
            m.model(input_ids=torch.tensor([ids], device=dev))
        return torch.stack(cap)[:, -1].float().cpu().numpy().astype(np.float16)
    X, meta = [], []                       # meta: problem, owner, thing, kind, log amount
    for pi, r in enumerate(rows):
        th = stated(r)
        full, blind = prompt(r["text"]), prompt(question_only(r["text"]))
        sw_text, rot = swapped(r, th)
        sw = prompt(sw_text) if sw_text else None
        for j, (c, a) in enumerate(th):
            cue = tok(c, add_special_tokens=False).input_ids
            for base, kind, amt in ((full, "cue", a), (blind, "blind", a)) + (((sw, "swap", rot[j]),) if sw else ()):
                X.append(last(base + cue))
                meta.append((pi, c, kind, math.log(amt)))
        try:
            ans = float(r["answer"])
        except (KeyError, TypeError, ValueError):
            ans = None
        X.append(last(full + tok("Answer:", add_special_tokens=False).input_ids))
        meta.append((pi, "Answer:", "answer", math.log(ans) if ans and ans > 0 else np.nan))
        print(f"{pi + 1}/{len(rows)} {r['id']} amounts {len(th)} swap {'yes' if sw else 'no'}", flush=True)
    for h in hooks:
        h.remove()
    np.savez(out, X=np.stack(X), prob=np.array([mm[0] for mm in meta]), cue=np.array([mm[1] for mm in meta]),
             kind=np.array([mm[2] for mm in meta]), y=np.array([mm[3] for mm in meta], float), ids=np.array([r["id"] for r in rows]),
             source=np.array([r.get("source", "") for r in rows]))
    print(f"saved {len(X)} sites to {out}")


def ridge_fit(Xa, ya, Xb, al):
    """dual-form ridge (as scene_probe's ridge_cv) fitted on a, read on b"""
    from scene_probe import scale
    mu, sd = Xa.mean(0), scale(Xa)
    A, B = (Xa - mu) / sd, (Xb - mu) / sd
    lam, Q = np.linalg.eigh(A @ A.T)
    d = np.clip(lam, 0, None) + al
    return (B @ A.T) @ Q @ ((Q.T @ (ya - ya.mean(0))) / (d if ya.ndim == 1 else d[:, None])) + ya.mean(0)


def read(src, out):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from scene_probe import ridge_cv
    z = np.load(src)
    kind, prob, y, src_of = z["kind"], z["prob"], z["y"], z["source"][z["prob"]]
    L = [f"situation test 1 reader, {src}: held-out by problem (5 folds), amounts as log"]
    def pick(P, sel):
        """for each site: is the read closest to its own amount among the problem's stated amounts (chance 1/k)"""
        ok = n = ch = 0
        for i in np.where(sel)[0]:
            same = np.where((prob == prob[i]) & (kind == "cue"))[0]
            cand = np.unique(np.round(y[same], 6))
            if len(cand) < 2:
                continue
            n += 1; ch += 1 / len(cand)
            ok += cand[np.argmin(abs(cand - P[i]))] == round(y[i], 6)
        return ok / max(n, 1), ch / max(n, 1), n
    def r2(P, sel):
        return 1 - ((P[sel] - y[sel]) ** 2).sum() / ((y[sel] - y[sel].mean()) ** 2).sum()
    alpha_cue = {}
    for layer in range(z["X"].shape[1]):
        X = z["X"][:, layer].astype(np.float32)
        s = kind == "cue"                       # one alpha per layer for the swap reader, picked on cue sites only
        sc = {al: ((ridge_cv(X[s], y[s, None], prob[s], alphas=(al,))[:, 0] - y[s]) ** 2).sum() for al in (1e2, 1e3, 1e4)}
        alpha_cue[layer] = min(sc, key=sc.get)
        P = np.full(len(y), np.nan)
        for k in ("cue", "blind"):              # each kind read by a reader fitted on that kind (as in geometry)
            s = kind == k
            P[s] = ridge_cv(X[s], y[s, None], prob[s])[:, 0]
        # swap: the reader fitted on cue sites of the other folds' problems, applied to the rotated story of the
        # held-out problems (it never saw a rotated story, so following the number means the notepad moved it)
        s_c, s_w = kind == "cue", kind == "swap"
        fold = prob % 5
        for f in range(5):
            a, b = s_c & (fold != f), s_w & (fold == f)
            if b.any():
                P[b] = ridge_fit(X[a], y[a], X[b], alpha_cue[layer])
        line = f"  index {layer:2d}"
        for k in ("cue", "blind", "swap"):
            pa, ch, n = pick(P, kind == k)
            line += f" | {k} pick {pa:.2f} (chance {ch:.2f}, n {n}) R2 {r2(P, kind == k):+.2f}"
        for s in ("mathworld", "gpt-4.1"):
            pa, _, n = pick(P, (kind == "cue") & (src_of == s))
            line += f" | cue {s[:4]} {pa:.2f}"
        L.append(line)
        print(line, flush=True)
    open(out, "w").write("\n".join(L) + "\n")


def codes(cap_f, out, opt):
    """his own word for each number: the notepad at the number in a plain sentence ("The number is 35."), every
    layer. Grading in these units needs no idea of how he stores numbers (Immy: don't time a drive in wheel turns)."""
    import torch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from patients import _lfm3
    z = np.load(cap_f)
    nums = sorted({int(round(math.exp(v))) for v in z["y"][np.isin(z["kind"], ["cue", "swap"])]})
    dev = opt.get("dev", "mps")
    p = _lfm3(dev, opt.get("hf"))
    tok, m = p.tok, p.model
    cap = []
    for l in m.model.layers:
        l.register_forward_hook(lambda mod, a, o: cap.append((o[0] if isinstance(o, tuple) else o)[0]))
    m.model.embed_tokens.register_forward_hook(lambda mod, a, o: cap.insert(0, o[0]))
    C = []
    for n in nums:
        ids = tok(f"The number is {n}", add_special_tokens=False).input_ids
        cap.clear()
        with torch.no_grad():
            m.model(input_ids=torch.tensor([ids], device=dev))
        C.append(torch.stack(cap)[:, -1].float().cpu().numpy())
    np.savez(out, nums=np.array(nums), C=np.stack(C).astype(np.float16))
    print(f"saved codes for {len(nums)} numbers to {out}")


def fit_rank(Xa, Ya, Xb, al, k):
    """ridge read squeezed through k channels: the top-k directions of what it predicts on its own training problems
    (k = 0: no squeeze). Small k that still reads well = the picture fits on a small screen."""
    Pb = ridge_fit(Xa, Ya, Xb, al)
    if not k:
        return Pb
    Pa = ridge_fit(Xa, Ya, Xa, al)
    mu = Pa.mean(0)
    V = np.linalg.svd(Pa - mu, full_matrices=False)[2][:k]
    return (Pb - mu) @ V.T @ V + mu


def read_code(cap_f, code_f, out, opt=None):
    """read each cue as a point in his number code, then pick the closest written number of that problem"""
    opt = opt or {}
    rank = int(opt.get("rank", 0))
    layers = [int(x) for x in opt["layers"].split(",")] if "layers" in opt else None
    z, c = np.load(cap_f), np.load(code_f)
    kind, prob = z["kind"], z["prob"]
    num = np.array([int(round(math.exp(v))) if np.isfinite(v) else -1 for v in z["y"]])
    idx = {n: i for i, n in enumerate(c["nums"])}
    src_of = z["source"][prob]
    L = [f"situation test 1 graded in his own number code ({code_f}): pick = nearest written number, held out by problem"
         + (f"; screen squeezed to {rank} channels" if rank else "")]
    fold = prob % 5
    for layer in layers or range(z["X"].shape[1]):
        X = z["X"][:, layer].astype(np.float32)
        C = c["C"][:, layer].astype(np.float32)
        C = C - C.mean(0)
        C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-8
        P = np.full((len(num), C.shape[1]), np.nan, np.float32)
        for k in ("cue", "blind"):               # reader fitted on that kind's other-fold problems
            s = kind == k
            for f in range(5):
                a, b = s & (fold != f), s & (fold == f)
                P[b] = fit_rank(X[a], C[[idx[n] for n in num[a]]], X[b], 1e3, rank)
        s_c, s_w = kind == "cue", kind == "swap"  # swap: the cue reader (never saw a rotated story)
        for f in range(5):
            a, b = s_c & (fold != f), s_w & (fold == f)
            if b.any():
                P[b] = fit_rank(X[a], C[[idx[n] for n in num[a]]], X[b], 1e3, rank)
        line = f"  index {layer:2d}"
        for k, extra in (("cue", None), ("blind", None), ("swap", None), ("cue", "mathworld"), ("cue", "gpt-4.1")):
            ok = n = ch = 0
            for i in np.where((kind == k) & ((src_of == extra) if extra else True))[0]:
                cand = sorted(set(num[(prob == prob[i]) & (kind == "cue")]))
                if len(cand) < 2:
                    continue
                sims = [float(P[i] @ C[idx[v]]) for v in cand]
                n += 1; ch += 1 / len(cand); ok += cand[int(np.argmax(sims))] == num[i]
            line += f" | {(extra or k)[:9]} {ok / max(n, 1):.2f}" + (f" (chance {ch / max(n, 1):.2f}, n {n})" if not extra else "")
        L.append(line)
        print(line, flush=True)
    open(out, "w").write("\n".join(L) + "\n")


def heads(src, code_f, out, opt):
    """which head grabs the picture? Switch off one head of a layer (all positions) and re-read the cue sites. The
    reader is fitted on clean cue sites of the other folds' problems and applied to the held-out ones with that head
    off, so a drop means the picture came through that head. Read at index layer+1 and at the last index."""
    import torch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from patients import _lfm3
    rows = list(map(json.loads, open(src)))
    LAYER = int(opt.get("layer", 14))
    dev = opt.get("dev", "mps")
    p = _lfm3(dev, opt.get("hf"))
    tok, m = p.tok, p.model
    att = m.model.layers[LAYER].self_attn
    nh = m.config.num_attention_heads
    hd = getattr(att, "head_dim", m.config.hidden_size // nh)
    off = {"h": None}                      # None, a tuple of heads, or "mlp" (the layer's lookup table)
    # mode=mean: a switched-off part writes its own average (over every word of every problem) instead of zero,
    # so the notepad stays in its usual range and only the problem-specific part is removed
    MEAN = opt.get("mode") == "mean"
    avg = {"head": None, "mlp": None, "n": 0, "acc": False}
    def zero(mod, a):
        if avg["acc"]:
            x = a[0][0].float()
            avg["head"] = x.sum(0) if avg["head"] is None else avg["head"] + x.sum(0)
            avg["n"] += x.shape[0]
        if not isinstance(off["h"], tuple):
            return None
        x = a[0].clone()
        for h in off["h"]:
            x[..., h * hd:(h + 1) * hd] = (avg["head"][h * hd:(h + 1) * hd] / avg["n"]).to(x.dtype) if MEAN else 0
        return (x,)
    def mlp_off(mod, a, o):
        if avg["acc"]:
            v = o[0].float().sum(0)
            avg["mlp"] = v if avg["mlp"] is None else avg["mlp"] + v
        if off["h"] != "mlp":
            return None
        return (avg["mlp"] / avg["n"]).to(o.dtype).expand_as(o).clone() if MEAN else o * 0
    (getattr(att, "o_proj", None) or att.out_proj).register_forward_pre_hook(zero)
    m.model.layers[LAYER].feed_forward.register_forward_hook(mlp_off)
    cap = []
    for l in m.model.layers:
        l.register_forward_hook(lambda mod, a, o: cap.append((o[0] if isinstance(o, tuple) else o)[0]))
    m.model.embed_tokens.register_forward_hook(lambda mod, a, o: cap.insert(0, o[0]))
    at = [LAYER + 1, len(m.model.layers)]
    name = lambda h: "none" if h is None else h if h == "mlp" else "all heads" if len(h) == nh else ",".join(map(str, h))
    sites = []                             # (problem, ids, amount)
    for pi, r in enumerate(rows):
        full = tok(tok.apply_chat_template([{"role": "user", "content": r["text"] + TAIL}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=True), add_special_tokens=False).input_ids
        for c, a in stated(r):
            sites.append((pi, full + tok(c, add_special_tokens=False).input_ids, a))
    # sets=15,8,13,1;all;mlp  (default: each head alone)
    conds = [None] + ([("mlp" if g == "mlp" else tuple(range(nh)) if g == "all" else tuple(int(x) for x in g.split(",")))
                       for g in opt["sets"].split(";")] if "sets" in opt else [(h,) for h in range(nh)])
    X = np.zeros((len(conds), len(sites), len(at), m.config.hidden_size), np.float16)
    if MEAN:                               # one clean pass to collect the averages
        avg["acc"] = True
        for _, ids, _ in sites:
            with torch.no_grad():
                m.model(input_ids=torch.tensor([ids], device=dev))
        avg["acc"] = False
        cap.clear()
    for ci, h in enumerate(conds):
        off["h"] = h
        for si, (_, ids, _) in enumerate(sites):
            cap.clear()
            with torch.no_grad():
                m.model(input_ids=torch.tensor([ids], device=dev))
            X[ci, si] = torch.stack([cap[i][-1] for i in at]).float().cpu().numpy()
        print(f"off {name(h)} done", flush=True)
    prob = np.array([s[0] for s in sites]); num = np.array([s[2] for s in sites])
    c = np.load(code_f); idx = {n: i for i, n in enumerate(c["nums"])}
    np.savez(out.replace(".txt", ".npz"), X=X, prob=prob, num=num, at=np.array(at))
    fold = prob % 5
    L = [f"head hunt, layer {LAYER} ({nh} heads, switched off at every position to {'their average' if MEAN else 'zero'}), {len(sites)} cue sites, "
         f"{len(rows)} problems; pick = nearest written number in his number code"]
    for ai, a_ in enumerate(at):
        C = c["C"][:, a_].astype(np.float32); C = C - C.mean(0); C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-8
        T = C[[idx[n] for n in num]]
        base = None
        for ci, h in enumerate(conds):
            P = np.zeros_like(T)
            for f in range(5):
                tr, te = fold != f, fold == f
                P[te] = ridge_fit(X[0, tr, ai].astype(np.float32), T[tr], X[ci, te, ai].astype(np.float32), 1e3)
            ok = n = 0
            for i in range(len(sites)):
                cand = sorted(set(num[prob == prob[i]]))
                if len(cand) < 2:
                    continue
                n += 1; ok += cand[int(np.argmax([P[i] @ C[idx[v]] for v in cand]))] == num[i]
            acc = ok / n
            base = acc if h is None else base
            L.append(f"  index {a_:2d} off {name(h):>12}: pick {acc:.3f}" + ("" if h is None else f"  change {acc - base:+.3f}"))
    open(out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    if sys.argv[1] == "heads":
        heads(sys.argv[2], sys.argv[3], sys.argv[4], dict(a.split("=", 1) for a in sys.argv[5:]))
        sys.exit()
    if sys.argv[1] == "capture":
        capture(sys.argv[2], sys.argv[3], dict(a.split("=", 1) for a in sys.argv[4:]))
    elif sys.argv[1] == "codes":
        codes(sys.argv[2], sys.argv[3], dict(a.split("=", 1) for a in sys.argv[4:]))
    elif sys.argv[1] == "read_code":
        read_code(sys.argv[2], sys.argv[3], sys.argv[4], dict(a.split("=", 1) for a in sys.argv[5:]))
    else:
        read(sys.argv[2], sys.argv[3])
