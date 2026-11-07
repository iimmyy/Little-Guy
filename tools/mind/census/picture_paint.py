#!/usr/bin/env python3
"""What paints over the picture? (Immy 2026-11-02, check A after B.) The answer-side verdict (below / between / above
the story's numbers) reads best around layer index 12-14 at the question mark and fades from index 15 on, in the floors
that make the words. Untouched model, reading and switching off only.

hunt    Fit the verdict reader at index `at` (folds by problem; each problem is scored with a reader that never saw
        it). Then split every later floor's write at the site into its parts: each attention head (its o_proj slice,
        passed through the floor's own after-norm with the shared scale), the conv mixer, the MLP. Push of a part =
        its write along "true side minus the other sides" of the reader. Negative = it writes against the verdict.
        Also: the reader's margin on the stream itself, index by index (where does it fall?).
ablate  Switch named parts off at the site only (replace their write with their average over problems) and re-read
        the verdict at every later index with fresh readers (same folds as picture_side.py). If the read comes back,
        that part paints over it; if not, it only looked like it.

usage: picture_paint.py hunt   <set.jsonl> <out.txt> [site=QS|FT at=14 from=12 limit= dev=cuda]
       picture_paint.py ablate <set.jsonl> <out.txt> parts=14.h15,14.mlp,15.conv [site=QS at=14 limit= dev=cuda]
"""
import json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from situation_probe import TAIL
from picture_side import side, cv_acc, ALPHAS, fit_pred

FOLDS = 4


def setup(src, opt):
    from patients import _lfm3
    dev = opt.get("dev", "cuda")
    p = _lfm3(dev, opt.get("hf"))
    tok, m = p.tok, p.model
    items = []
    for r in list(map(json.loads, open(src)))[:int(opt.get("limit", 10 ** 9))]:
        lab = side(r["text"], r["answer"])
        if lab is None:
            continue
        chat = tok.apply_chat_template([{"role": "user", "content": r["text"] + TAIL}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=True)
        enc = tok(chat, add_special_tokens=False, return_offsets_mapping=True)
        ids, offs = enc.input_ids, enc.offset_mapping
        if opt.get("site", "QS") == "QS":
            t0, qm = chat.find(r["text"]), r["text"].rfind("?")
            c = t0 + (qm if qm >= 0 else len(r["text"]) - 1)
            pos = max(i for i, (s, e) in enumerate(offs) if s <= c and e > s)
        else:
            pos = len(ids) - 1
        items.append((r["id"], ids, pos, lab))
    return m, dev, items


def stream(m, dev, items, edit=None):
    """residual at the site, every index (0 = embeddings); edit(pos) installs/removes switch-off hooks"""
    cap, out = [], []
    hs = [l.register_forward_hook(lambda mod, a, o: cap.append((o[0] if isinstance(o, tuple) else o)[0]))
          for l in m.model.layers]
    hs.append(m.model.embed_tokens.register_forward_hook(lambda mod, a, o: cap.insert(0, o[0])))
    for _, ids, pos, _ in items:
        cap.clear()
        undo = edit(pos) if edit else []
        with torch.no_grad():
            m.model(input_ids=torch.tensor([ids], device=dev))
        for h in undo:
            h.remove()
        out.append(torch.stack([c[pos] for c in cap]).float().cpu().numpy())
    for h in hs:
        h.remove()
    return np.stack(out)


def fold_of(n):
    return np.arange(n) % FOLDS          # problems are their own groups; same split as picture_side's cv_acc


def readers(X, y):
    """per held-out problem: raw-space reader weights (d x 3) and bias, from a fold that never saw it"""
    f, Y = fold_of(len(y)), np.eye(3)[y]
    W, B = np.zeros((len(y), X.shape[1], 3), np.float32), np.zeros((len(y), 3), np.float32)
    for k in range(FOLDS):
        tr = f != k
        best, al_b = -1, ALPHAS[0]
        for al in ALPHAS:
            ok = sum((fit_pred(X[tr & (f != j)], Y[tr & (f != j)], X[f == j], al).argmax(1) == y[f == j]).sum()
                     for j in range(FOLDS) if j != k)
            if ok > best:
                best, al_b = ok, al
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        A = (X[tr] - mu) / sd
        coef = np.linalg.solve(A @ A.T + al_b * np.eye(len(A)), Y[tr] - Y[tr].mean(0))
        w = (A.T @ coef) / sd[:, None]
        W[f == k], B[f == k] = w, Y[tr].mean(0) - mu @ w
    return W, B


def hunt(src, out, opt):
    m, dev, items = setup(src, opt)
    at, frm = int(opt.get("at", 14)), int(opt.get("from", 12))
    y = np.array([it[3] for it in items])
    S = stream(m, dev, items)
    W, B = readers(S[:, at], y)
    # "true side minus the mean of the other sides", per problem
    V = np.stack([W[i][:, y[i]] - W[i][:, [c for c in range(3) if c != y[i]]].mean(1) for i in range(len(y))])
    L = [f"paint hunt, {len(y)} problems, site {opt.get('site', 'QS')}, verdict reader at index {at}"]
    L.append("stream, read by the index-%d reader (held out): margin and accuracy by index" % at)
    for li in range(at, S.shape[1]):
        sc = np.einsum("nd,ndc->nc", S[:, li], W) + B
        mg = (V * S[:, li]).sum(1)
        L.append(f"  index {li:2d}: margin {mg.mean():+.3f}  acc {(sc.argmax(1) == y).mean():.3f}")
    # parts of each floor >= frm, at the site
    lay = m.model.layers
    parts, push, check = [], [], []
    got = {}
    hs = []
    for i in range(frm, len(lay)):
        l = lay[i]
        if l.is_attention_layer:
            hs.append(l.self_attn.o_proj.register_forward_pre_hook(lambda mod, a, i=i: got.__setitem__(("a", i), a[0][0])))
        hs.append(l.post_operator_norm.register_forward_hook(lambda mod, a, o, i=i: got.__setitem__(("op", i), (a[0][0], o[0]))))
        hs.append(l.post_ffn_norm.register_forward_hook(lambda mod, a, o, i=i: got.__setitem__(("mlp", i), o[0])))
    for n, (_, ids, pos, _) in enumerate(items):
        got.clear()
        with torch.no_grad():
            m.model(input_ids=torch.tensor([ids], device=dev))
        row, names = [], []
        for i in range(frm, len(lay)):
            l = lay[i]
            raw, normed = got[("op", i)]
            raw, normed = raw[pos].float(), normed[pos].float()
            if l.is_attention_layer:
                a, Wo, hd = got[("a", i)][pos].float(), l.self_attn.o_proj.weight.float(), l.self_attn.head_dim
                scale = l.post_operator_norm.weight.float() * torch.rsqrt(raw.pow(2).mean() + l.post_operator_norm.variance_epsilon)
                whole = scale * (a @ Wo.T)                     # the heads' parts must add up to the floor's real write
                check.append(float((whole - normed).norm() / normed.norm()))
                for h in range(a.shape[0] // hd):
                    sl = slice(h * hd, (h + 1) * hd)
                    row.append(float((scale * (a[sl] @ Wo[:, sl].T)).cpu().numpy() @ V[n]))
                    names.append(f"{i}.h{h}")
            else:
                row.append(float(normed.cpu().numpy() @ V[n]))
                names.append(f"{i}.conv")
            row.append(float(got[("mlp", i)][pos].float().cpu().numpy() @ V[n]))
            names.append(f"{i}.mlp")
        push.append(row)
        parts = names
    for h in hs:
        h.remove()
    P = np.array(push)
    L.append(f"check: heads' parts vs the floor's real write, worst relative gap {max(check):.4f} (bf16 rounding ~0.01)")
    mean, se = P.mean(0), P.std(0) / np.sqrt(len(P))
    L.append(f"floor totals (sum of parts' push; negative = writes against the verdict):")
    for i in range(frm, len(lay)):
        k = [j for j, nm in enumerate(parts) if nm.split(".")[0] == str(i)]
        L.append(f"  floor {i:2d} ({'attn' if lay[i].is_attention_layer else 'conv'}): {P[:, k].sum(1).mean():+.3f}")
    order = np.argsort(mean)
    L.append("parts that push hardest AGAINST the verdict (mean push, +-1 se, share of problems pushed against):")
    for j in order[:15]:
        L.append(f"  {parts[j]:9s} {mean[j]:+.4f} +-{se[j]:.4f}  against in {(P[:, j] < 0).mean():.2f}")
    L.append("parts that push hardest FOR it:")
    for j in order[::-1][:8]:
        L.append(f"  {parts[j]:9s} {mean[j]:+.4f} +-{se[j]:.4f}")
    open(out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def ablate(src, out, opt):
    m, dev, items = setup(src, opt)
    at = int(opt.get("at", 14))
    y = np.array([it[3] for it in items])
    lay = m.model.layers
    want = [p.split(".") for p in opt["parts"].split(",")]
    # 1. each part's average write input at the site
    acc, got, hs = {}, {}, []
    for i, kind in want:
        i = int(i); l = lay[i]
        if kind.startswith("h"):
            hs.append(l.self_attn.o_proj.register_forward_pre_hook(lambda mod, a, i=i: got.__setitem__(("a", i), a[0][0])))
        elif kind == "mlp":
            hs.append(l.post_ffn_norm.register_forward_hook(lambda mod, a, o, i=i: got.__setitem__(("mlp", i), o[0])))
        else:
            hs.append(l.post_operator_norm.register_forward_hook(lambda mod, a, o, i=i: got.__setitem__(("op", i), o[0])))
    for _, ids, pos, _ in items:
        got.clear()
        with torch.no_grad():
            m.model(input_ids=torch.tensor([ids], device=dev))
        for k, v in got.items():
            acc[k] = acc.get(k, 0) + v[pos].float() / len(items)
    for h in hs:
        h.remove()

    def edit(pos):
        hs = []
        for i, kind in want:
            i = int(i); l = lay[i]
            if kind.startswith("h"):
                hd, h = l.self_attn.head_dim, int(kind[1:])
                sl = slice(h * hd, (h + 1) * hd)
                def pre(mod, a, sl=sl, i=i):
                    x = a[0].clone(); x[0, pos, sl] = acc[("a", i)][sl].to(x.dtype); return (x,)
                hs.append(l.self_attn.o_proj.register_forward_pre_hook(pre))
            else:
                norm, key = (l.post_ffn_norm, ("mlp", i)) if kind == "mlp" else (l.post_operator_norm, ("op", i))
                def post(mod, a, o, key=key):
                    o = o.clone(); o[0, pos] = acc[key].to(o.dtype); return o
                hs.append(norm.register_forward_hook(post))
        return hs

    S0 = stream(m, dev, items)
    S1 = stream(m, dev, items, edit)
    groups = np.arange(len(y))
    L = [f"paint ablate, {len(y)} problems, site {opt.get('site', 'QS')}, parts switched off at the site: {opt['parts']}",
         "verdict read with fresh readers at each index (folds by problem), untouched -> parts off:"]
    for li in range(at - 2, S0.shape[1]):
        a0, a1 = cv_acc(S0[:, li], y, groups, 3)[0], cv_acc(S1[:, li], y, groups, 3)[0]
        L.append(f"  index {li:2d}: {a0:.3f} -> {a1:.3f}  ({a1 - a0:+.3f})")
    open(out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    cmd, args = sys.argv[1], [a for a in sys.argv[2:] if "=" not in a]
    opt = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)
    {"hunt": hunt, "ablate": ablate}[cmd](args[0], args[1], opt)
