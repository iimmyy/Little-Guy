#!/usr/bin/env python3
"""Does the model build a scene while it reads? (Immy 2026-11-02: "gather points to make a relative scene FIRST as you're
comprehending it", read the question backwards, a visualizer bolted on.) Step 1 is measuring only.

Problems with drawing code ([asy]) give the true place of every labelled point. The untouched model reads each problem
(the same prompt as our runs) and its notepad (the residual stream after every layer) is kept at:
  label: the end of the label("$A$", ...) statement that places a point (the coordinates are in the last few words)
  pre:   a mention of the point's letter in the prose BEFORE the drawing (it cannot know the place yet: the control)
  post:  a mention AFTER the drawing (rare: MATH puts the drawing last, after the ask)
  gather: the words between the end of the question and the start of the thinking (saved for step 4)
  cue:   asking "where is D?": "Point $D$" added where the thinking starts, read at the letter (one read per point)
  blind: the same cue with the drawing cut out of the question (it can only guess: the control for cue)
Places are relative: each problem's points minus their centre, over their spread.

usage: scene_probe.py capture <problems.jsonl> <sealed.jsonl> <out.npz> [hf=] [dev=mps] [limit=]
       scene_probe.py read <capture.npz> <out.txt>
       scene_probe.py polarity <capture.npz> <out.txt> [kind=cue] [at=<layer>]
sealed.jsonl: problems never to be used (matched by exact text and left out).
"""
import json, os, re, sys
import numpy as np

LAB = re.compile(r'label\(\s*"\$?([A-Z])\$?"\s*,\s*(?:\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)|([A-Z][A-Za-z0-9_]*)\b)')
PAIR = re.compile(r'\b([A-Z][A-Za-z0-9_]*)\s*=\s*\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)')
TAIL = "\n\nEnd with a line: Answer: <your answer>"


def parse(q):
    """{letter: ((x, y), label_end_char)} for labels placed by numbers or by a numerically defined pair"""
    a, b = q.find("[asy]"), q.find("[/asy]")
    if a < 0 or b < a:
        return {}, a, b
    var = {m[1]: (float(m[2]), float(m[3])) for m in PAIR.finditer(q, a, b)}
    pts = {}
    for m in LAB.finditer(q, a, b):
        xy = (float(m[2]), float(m[3])) if m[2] else var.get(m[4])
        end = q.find(";", m.end(), b)
        if xy and m[1] not in pts and end > 0:
            pts[m[1]] = (xy, end - 1)
    if len({xy for xy, _ in pts.values()}) < 3:
        return {}, a, b
    return pts, a, b


def relative(pts):
    xy = np.array([p for p, _ in pts.values()], float)
    xy -= xy.mean(0)
    return xy / np.sqrt((xy ** 2).sum(1).mean())


def capture(src, sealed_f, out, opt):
    import torch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from patients import _lfm3
    sealed = {json.loads(l)["q"] for l in open(sealed_f)}
    rows = [r for r in map(json.loads, open(src)) if r["q"] not in sealed]
    probs = [(r, *parse(r["q"])) for r in rows]
    probs = [p for p in probs if p[1]][:int(opt.get("limit", 10 ** 9))]
    print(f"{len(rows)} problems not sealed, {len(probs)} with >= 3 placed points", flush=True)
    dev = opt.get("dev", "mps")
    p = _lfm3(dev, opt.get("hf"))
    tok, m = p.tok, p.model
    cap = []
    hooks = [l.register_forward_hook(lambda mod, a, o: cap.append((o[0] if isinstance(o, tuple) else o)[0]))
             for l in m.model.layers]
    m.model.embed_tokens.register_forward_hook(lambda mod, a, o: cap.insert(0, o[0]))
    X, meta = [], []                       # X: (layers+1, d) per site; meta: problem, letter, kind, x, y
    for pi, (r, pts, a, b) in enumerate(probs):
        text = tok.apply_chat_template([{"role": "user", "content": r["q"] + TAIL}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=True)
        q0 = text.find(r["q"])
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        offs = enc.offset_mapping
        def tok_at(c):                     # token covering char c of the question
            c += q0
            return next(i for i, (s, e) in enumerate(offs) if s <= c < e)
        rel = dict(zip(pts, relative(pts)))
        sites = []
        for L, (_, end) in pts.items():
            sites.append((tok_at(end), L, "label"))
            for mm in re.finditer(rf"(?<![A-Za-z]){L}(?![A-Za-z])", r["q"]):
                if a <= mm.start() < b + 6:
                    continue
                t = tok_at(mm.start())
                if re.sub(r"[^A-Za-z]", "", tok.decode([enc.input_ids[t]])) == L:
                    sites.append((t, L, "pre" if mm.start() < a else "post"))
        qend = tok_at(len(r["q"]) - 1)
        for t in range(qend, len(enc.input_ids)):
            sites.append((t, "", "gather"))
        def run(ids):
            cap.clear()
            with torch.no_grad():
                m.model(input_ids=torch.tensor([ids], device=dev))
            return torch.stack(cap)        # (25, T, d)
        def keep(H, t, L, kind, pos):
            X.append(H[:, t].float().cpu().numpy().astype(np.float16))
            x, y = rel[L] if L else (np.nan, np.nan)
            meta.append((pi, L, kind, x, y, pos))
        H = run(enc.input_ids)
        for t, L, kind in sites:
            keep(H, t, L, kind, t - qend)
        blind_q = r["q"][:a] + r["q"][b + 6:]
        blind = tok(tok.apply_chat_template([{"role": "user", "content": blind_q + TAIL}], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=True),
                    add_special_tokens=False).input_ids
        for L in pts:
            cue = tok(f"Point ${L}$", add_special_tokens=False).input_ids
            ct = max(i for i, c in enumerate(cue) if L in tok.decode([c]))
            for base, kind in ((enc.input_ids, "cue"), (blind, "blind")):
                keep(run(base + cue), len(base) + ct, L, kind, ct)
        print(f"{pi + 1}/{len(probs)} {r['id']} points {len(pts)} sites {len(sites) + 2 * len(pts)}", flush=True)
    for h in hooks:
        h.remove()
    np.savez(out, X=np.stack(X), prob=np.array([mm[0] for mm in meta]), letter=np.array([mm[1] for mm in meta]),
             kind=np.array([mm[2] for mm in meta]), xy=np.array([mm[3:5] for mm in meta], float),
             gather_pos=np.array([mm[5] for mm in meta]), ids=np.array([pr[0]["id"] for pr in probs]))
    print(f"saved {len(X)} sites to {out}")


def scale(A):
    """per-number spread, floored at 1% of the typical spread (near-silent numbers are not blown up into noise)"""
    sd = A.std(0)
    return np.maximum(sd, 0.01 * np.median(sd) + 1e-8)


def ridge_cv(X, Y, groups, alphas=(1e1, 1e2, 1e3, 1e4, 1e5), k=5, seed=0):
    """grouped k-fold ridge (alpha chosen inside the training folds); returns out-of-fold predictions"""
    ug = np.unique(groups)
    rng = np.random.default_rng(seed)
    fold = dict(zip(rng.permutation(ug), np.arange(len(ug)) % k))
    f = np.array([fold[g] for g in groups])
    P = np.zeros_like(Y)
    def fit(Xa, Ya, Xb, al):
        mu, sd = Xa.mean(0), scale(Xa)
        A, B = (Xa - mu) / sd, (Xb - mu) / sd
        lam, Q = np.linalg.eigh(A @ A.T)                      # dual form: sites < numbers per notepad
        QtY, BAQ = Q.T @ (Ya - Ya.mean(0)), (B @ A.T) @ Q
        return [BAQ @ (QtY / (np.clip(lam, 0, None) + a)[:, None]) + Ya.mean(0) for a in al]
    for i in range(k):
        tr, te = f != i, f == i
        inner = f[tr] % 2 == (i % 2)                          # a split inside the training folds picks alpha
        if inner.all() or (~inner).all():
            inner = np.arange(tr.sum()) % 2 == 0
        sc = [((pr - Y[tr][~inner]) ** 2).sum() for pr in fit(X[tr][inner], Y[tr][inner], X[tr][~inner], alphas)]
        P[te] = fit(X[tr], Y[tr], X[te], [alphas[int(np.argmin(sc))]])[0]
    return P


def scores(P, Y, prob):
    r2 = 1 - ((P - Y) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum()
    ok = n = 0                                                # relations: is A left of / above B, same problem
    for g in np.unique(prob):
        idx = np.where(prob == g)[0]
        for i in idx:
            for j in idx:
                for d in (0, 1):
                    if i < j and abs(Y[i, d] - Y[j, d]) > 0.2:
                        n += 1
                        ok += np.sign(P[i, d] - P[j, d]) == np.sign(Y[i, d] - Y[j, d])
    return r2, ok / max(1, n), n


def read(src, out):
    Z = np.load(src)
    X, prob, kind, letter, Y = Z["X"], Z["prob"], Z["kind"], Z["letter"], Z["xy"]
    L = [f"scene read-out (relative places, ridge, 5-fold by problem); {len(Z['ids'])} problems; layer 0 = embeddings",
         "R2 = share of the place explained; rel = left/right + above/below between two points of one problem (chance 0.50)"]
    for kd in ("label", "cue", "blind", "pre", "post"):
        s = kind == kd
        if s.sum() < 40:
            L.append(f"{kd}: only {s.sum()} sites, skipped")
            continue
        oh = (letter[s][:, None] == np.unique(letter[s])[None]).astype(float)
        b = scores(ridge_cv(oh, Y[s], prob[s]), Y[s], prob[s])
        sh = scores(ridge_cv(X[s, 12].astype(float), np.random.default_rng(1).permutation(Y[s]), prob[s]), Y[s], prob[s])
        L.append(f"== {kd}: {s.sum()} sites on {len(np.unique(prob[s]))} problems.  letter-only guess: R2 {b[0]:+.2f} rel {b[1]:.2f}"
                 f" (pairs {b[2]});  shuffled places, layer 12: R2 {sh[0]:+.2f} rel {sh[1]:.2f}")
        for layer in range(X.shape[1]):
            r2, rel, _ = scores(ridge_cv(X[s, layer].astype(float), Y[s], prob[s]), Y[s], prob[s])
            L.append(f"   layer {layer:2d}: R2 {r2:+.2f}  rel {rel:.2f}")
        print("\n".join(L[-X.shape[1] - 1:]), flush=True)
    open(out, "w").write("\n".join(L) + "\n")


def polarity(src, out, kind="cue", at=None):
    """Reverse polarity (Immy): each layer only ADDS to the notepad, so erasing the scene means writing its negative.
    The read-out direction is fit at layer `at` (default: the best cue layer) on 4 of 5 problem folds; on the held-out
    fold every layer's own write (notepad after it minus before it) is put through that same read-out. A layer that
    writes the scene has its share line up with the true place (+); one that cancels it lines up against it (-)."""
    Z = np.load(src)
    s = Z["kind"] == kind
    X, Y, prob = Z["X"][s].astype(float), Z["xy"][s], Z["prob"][s]
    if at is None:
        at = int(np.argmax([scores(ridge_cv(X[:, l], Y, prob), Y, prob)[0] for l in range(X.shape[1])]))
    ug = np.unique(prob)
    fold = dict(zip(np.random.default_rng(0).permutation(ug), np.arange(len(ug)) % 5))
    f = np.array([fold[g] for g in prob])
    share = np.zeros((X.shape[1], len(Y), 2))                 # layer write through the read-out
    for i in range(5):
        tr, te = f != i, f == i
        A = X[tr, at]
        mu, sd = A.mean(0), scale(A)
        An = (A - mu) / sd
        beta = An.T @ np.linalg.solve(An @ An.T + 1e3 * np.eye(len(An)), Y[tr] - Y[tr].mean(0))
        for l in range(X.shape[1]):
            w = X[te, l] - (X[te, l - 1] if l else 0)
            share[l, te] = (w / sd) @ beta
    L = [f"polarity of each layer's write along the scene read-out ({kind} reads, read-out fit at layer {at}, held-out folds)",
         "agree = correlation of the layer's share with the true place (+ writes the scene, - writes it backwards); "
         "size = typical share / typical place"]
    run = np.zeros_like(share[0])
    for l in range(X.shape[1]):
        run += share[l]
        c = np.mean([np.corrcoef(share[l, :, d], Y[:, d])[0, 1] for d in (0, 1)])
        cr = np.mean([np.corrcoef(run[:, d], Y[:, d])[0, 1] for d in (0, 1)])
        L.append(f"   layer {l:2d}: agree {c:+.2f}  size {np.abs(share[l]).mean() / np.abs(Y).mean():.2f}   "
                 f"running total agree {cr:+.2f}")
    open(out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    opt = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)
    args = [a for a in sys.argv[2:] if "=" not in a]
    {"capture": lambda: capture(*args[:3], opt), "read": lambda: read(*args[:2]),
     "polarity": lambda: polarity(*args[:2], opt.get("kind", "cue"), int(opt["at"]) if "at" in opt else None)}[sys.argv[1]]()
