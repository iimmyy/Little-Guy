#!/usr/bin/env python3
"""Imager piece 1, the gather fetch (Immy 2026-11-02): the head that grabs the picture when asked (layer 14 head 15,
situation_probe.py heads) also grabs it at the end of every sentence while reading. No training.

What head 15 lacks at sentence ends (head_look.py): its write-gate is already half open there, but its query does not
seek the numbers. The edit adds HIS OWN "seek the amount" query direction at sentence ends only:
  seek     d = mean query when asked for an amount minus mean query on ordinary story words (head 15, before its norm)
  trigger  t(x) = s . x, a straight line on the attention input that is ~1 at sentence ends and ~0 on other words
           (s from the sentence-end vs other-word means, other-word mean projected out)
  edit     q_proj rows of head 15  +=  a * d_hat s^T   (rank 1; a = the gap between asked and sentence-end seek)
Keys, values, gate and every other head untouched. Built on half the problems (fold 0-2), checked on the other half.

usage: gather_fetch.py fit   <problems.jsonl> <edit.npz> [layer=14 head=15 hf= dev=mps]
       gather_fetch.py fit_bind <problems.jsonl> <edit.npz> [layer=14 head=15 rank=8 alpha=10 hf= dev=mps]  (v2: binding)
       gather_fetch.py check <problems.jsonl> <edit.npz> <thinking.jsonl> <out.txt> [scale=1 hf= dev=mps]
"""
import json, math, os, re, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from patients import _lfm3
from situation_probe import stated, TAIL, ridge_fit

END = re.compile(r"[.?!](?=\s|$)")


def prompt(tok, text):
    t = tok.apply_chat_template([{"role": "user", "content": text + TAIL}], tokenize=False, add_generation_prompt=True,
                                enable_thinking=True)
    enc = tok(t, add_special_tokens=False, return_offsets_mapping=True)
    return enc.input_ids, enc.offset_mapping, t.find(text)


def story_sites(r, offs, q0):
    """token index of every sentence end, of every other story word, and each sentence's written numbers"""
    covering = lambda a, b: [i for i, (s, e) in enumerate(offs) if s < q0 + b and e > q0 + a]
    ends, nums, start = [], [], 0
    for mm in END.finditer(r["text"]):
        t = covering(mm.start(), mm.start() + 1)
        if t:
            ends.append(t[-1])
            nums.append([int(x[0]) for x in re.finditer(r"(?<![\d.,])\d+(?![\d])", r["text"][start:mm.start()])])
        start = mm.start() + 1
    story = covering(0, len(r["text"]))
    return ends, [t for t in story if t not in ends], nums


class Model:
    def __init__(self, opt, layer, head):
        self.p = _lfm3(opt.get("dev", "mps"), opt.get("hf"))
        self.tok, self.m, self.dev = self.p.tok, self.p.model, opt.get("dev", "mps")
        self.att = self.m.model.layers[layer].self_attn
        self.hd, self.head, self.layer = self.att.head_dim, head, layer
        self.got = {}
        self.att.register_forward_pre_hook(lambda mod, a, kw: self.got.__setitem__("x", (a[0] if a else kw["hidden_states"])[0].float()),
                                           with_kwargs=True)
        sl = slice(head * self.hd, (head + 1) * self.hd)
        self.att.q_proj.register_forward_hook(lambda mod, a, o: self.got.__setitem__("q", o[0, :, sl].float()))
        self.cap = []
        for l in self.m.model.layers:
            l.register_forward_hook(lambda mod, a, o: self.cap.append((o[0] if isinstance(o, tuple) else o)[0]))
        self.m.model.embed_tokens.register_forward_hook(lambda mod, a, o: self.cap.insert(0, o[0]))

    def run(self, ids, eager=False):
        self.cap.clear()
        self.m.config._attn_implementation = "eager" if eager else "sdpa"
        w = {}
        h = self.att.register_forward_hook(lambda mod, a, o: w.__setitem__("w", o[1][0, self.head].float())) if eager else None
        with torch.no_grad():
            out = self.m(input_ids=torch.tensor([ids], device=self.dev))
        if h:
            h.remove()
        return out.logits[0].float(), w.get("w")


def fit(src, out, opt):
    layer, head = int(opt.get("layer", 14)), int(opt.get("head", 15))
    M = Model(opt, layer, head)
    rows = [r for i, r in enumerate(map(json.loads, open(src))) if i % 5 < 3]
    xe, xo, qa, qo, qe = [], [], [], [], []
    for r in rows:
        ids, offs, q0 = prompt(M.tok, r["text"])
        ends, other, _ = story_sites(r, offs, q0)
        M.run(ids)
        x, q = M.got["x"].cpu().numpy(), M.got["q"].cpu().numpy()
        xe += list(x[ends]); xo += list(x[other]); qe += list(q[ends]); qo += list(q[other])
        for c, a in stated(r):
            M.run(ids + M.tok(c, add_special_tokens=False).input_ids)
            qa.append(M.got["q"][-1].cpu().numpy())
    xe, xo, qa, qo, qe = map(np.array, (xe, xo, qa, qo, qe))
    d = qa.mean(0) - qo.mean(0)
    dh = d / np.linalg.norm(d)
    gap = float(qa.mean(0) @ dh - qe.mean(0) @ dh)
    mo, me = xo.mean(0), xe.mean(0)
    s = me - mo
    s = s - (s @ mo) / (mo @ mo) * mo                          # zero on the average ordinary word
    s = s / (s @ me)                                           # one on the average sentence end
    te, to = xe @ s, xo @ s
    np.savez(out, d_hat=dh, s=s, a=gap, layer=layer, head=head)
    print(f"fit on {len(rows)} problems: asked {len(qa)}, sentence ends {len(xe)}, other words {len(xo)}")
    print(f"seek gap along d (asked minus sentence end) {gap:.3f}; asked {qa.mean(0) @ dh:.3f} end {qe.mean(0) @ dh:.3f} other {qo.mean(0) @ dh:.3f}")
    print(f"trigger t on fit sites: sentence ends median {np.median(te):.3f} (10% {np.percentile(te, 10):.3f}); "
          f"other words median {np.median(to):.3f} (90% {np.percentile(to, 90):.3f}, 99% {np.percentile(to, 99):.3f})")


def fit_bind(src, out, opt):
    """v2: at a sentence end, ask what a question about that sentence's amount asks (binding), zero on other words"""
    layer, head, rank = int(opt.get("layer", 14)), int(opt.get("head", 15)), int(opt.get("rank", 8))
    M = Model(opt, layer, head)
    rows = [r for i, r in enumerate(map(json.loads, open(src))) if i % 5 < 3]
    Xp, Yp, Xo = [], [], []
    for r in rows:
        ids, offs, q0 = prompt(M.tok, r["text"])
        ends, other, nums = story_sites(r, offs, q0)
        M.run(ids)
        x, q = M.got["x"].cpu().numpy(), M.got["q"].cpu().numpy()
        Xo += list(x[other])
        asked = {}
        for c, a in stated(r):
            M.run(ids + M.tok(c, add_special_tokens=False).input_ids)
            asked.setdefault(a, []).append(M.got["q"][-1].cpu().numpy())
        for t, ns in zip(ends, nums):
            qs = [v for n in ns for v in asked.get(n, [])]
            if qs:
                Xp.append(x[t]); Yp.append(np.mean(qs, 0) - q[t])
    Xp, Yp, Xo = map(np.array, (Xp, Yp, Xo))
    w = np.sqrt(len(Xo) / len(Xp))                              # sentence ends weigh as much as all other words together
    X = np.vstack([Xp * w, Xo]); Y = np.vstack([Yp * w, np.zeros((len(Xo), Yp.shape[1]))])
    al = float(opt.get("alpha", 10.0))
    A = np.linalg.solve(X.T @ X + al * np.eye(X.shape[1]), X.T @ Y).T   # (128, d): added query = A x (no centering: a weight edit has no bias)
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    A = (U[:, :rank] * S[:rank]) @ Vt[:rank]
    fp, fo = Xp @ A.T, Xo @ A.T
    cosp = np.mean([a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9) for a, b in zip(fp, Yp)])
    np.savez(out, A=A, layer=layer, head=head, rank=rank)
    print(f"fit on {len(rows)} problems: sentence ends with an asked amount {len(Xp)}, other words {len(Xo)}, rank {rank}, alpha {al}")
    print(f"  sentence ends: target size {np.linalg.norm(Yp, axis=1).mean():.3f}, made {np.linalg.norm(fp, axis=1).mean():.3f}, cos to target {cosp:.3f}")
    print(f"  other words: added query size median {np.median(np.linalg.norm(fo, axis=1)):.3f}, 99% {np.percentile(np.linalg.norm(fo, axis=1), 99):.3f}")


def check(src, edit_f, think_f, out, opt):
    E = np.load(edit_f)
    layer, head, scale = int(E["layer"]), int(E["head"]), float(opt.get("scale", 1))
    M = Model(opt, layer, head)
    tok = M.tok
    rows = [r for i, r in enumerate(map(json.loads, open(src))) if i % 5 >= 3]
    W = M.att.q_proj.weight
    sl = slice(head * M.hd, (head + 1) * M.hd)
    W0 = W.data[sl].clone()
    delta = torch.tensor(scale * (E["A"] if "A" in E else float(E["a"]) * np.outer(E["d_hat"], E["s"])), dtype=W.dtype, device=W.device)
    s_t = torch.tensor(E["s"] if "s" in E else np.zeros(W.shape[1]), device=W.device, dtype=torch.float32)
    lines = [f"gather fetch check: layer {layer} head {head}, scale {scale}, {len(rows)} held-out problems (fold 3-4)"]
    res = {}
    for cond in ("untouched", "edited"):
        W.data[sl] = W0 + (delta if cond == "edited" else 0)
        own, trig_end, trig_oth, sites, gath, leak = [], [], [], [], [], []
        for pi, r in enumerate(rows):
            ids, offs, q0 = prompt(tok, r["text"])
            ends, other, nums = story_sites(r, offs, q0)
            _, w = M.run(ids, eager=True)
            x = M.got["x"]
            tt = (x @ s_t).cpu().numpy()
            trig_end += list(tt[ends]); trig_oth += list(tt[other])
            leak += [tok.decode([ids[i]]) for i in other if tt[i] > 0.5]
            numtok = {}
            for mm in re.finditer(r"(?<![\d.,])\d+(?![\d])", r["text"]):
                numtok.setdefault(int(mm[0]), []).extend(i for i, (s, e) in enumerate(offs) if s < q0 + mm.end() and e > q0 + mm.start())
            for t, ns in zip(ends, nums):
                tk = [i for n in ns for i in numtok.get(n, [])]
                if tk:
                    own.append(float(w[t, tk].sum()))
            H = torch.stack([M.cap[i] for i in (layer + 1, len(M.m.model.layers))])[:, :, :].float().cpu().numpy()
            allnums = sorted(numtok)
            order = [int(mm[0]) for mm in re.finditer(r"(?<![\d.,])\d+(?![\d])", r["text"])]
            if len(allnums) >= 2:
                gath.append((pi, H[:, -1], order, allnums))
            for t, ns in zip(ends, nums):
                if ns and len(allnums) >= 2:
                    sites.append((pi, H[:, t], ns, allnums))
        res[cond] = (own, trig_end, trig_oth, sites, gath)
        if cond == "untouched":
            from collections import Counter
            lines.append(f"  trigger leaks (other words over 0.5): {len(leak)} of {len(trig_oth)}, most common {Counter(leak).most_common(8)}")
        lines.append(f"  {cond}: head {head} attention at sentence ends on that sentence's numbers mean {np.mean(own):.3f} "
                     f"(median {np.median(own):.3f}, n {len(own)})")
    _, te, to, _, _ = res["untouched"]
    lines.append(f"  trigger on held-out story: sentence ends median {np.median(te):.3f}; other words median {np.median(to):.3f}, "
                 f"99% {np.percentile(to, 99):.3f}")
    # picture written at the sentence end, nobody asking: read the sentence's number in his own number code
    W.data[sl] = W0
    nums = sorted({n for c in res.values() for s in c[3] + c[4] for n in s[3]})
    C = []
    for n in nums:
        M.run(tok(f"The number is {n}", add_special_tokens=False).input_ids)
        C.append(torch.stack([M.cap[i][-1] for i in (layer + 1, len(M.m.model.layers))]).float().cpu().numpy())
    C = np.stack(C)
    idx = {n: i for i, n in enumerate(nums)}
    for ai, name in enumerate((f"index {layer + 1}", "last index")):
        Cn = C[:, ai] - C[:, ai].mean(0); Cn /= np.linalg.norm(Cn, axis=1, keepdims=True) + 1e-8
        for cond in ("untouched", "edited"):
            S = res[cond][3]
            X = np.array([s[1][ai] for s in S]); prob = np.array([s[0] for s in S])
            T = Cn[[idx[s[2][0]] for s in S]]
            P = np.zeros_like(T)
            for f in range(4):
                tr, te_ = prob % 4 != f, prob % 4 == f
                P[te_] = ridge_fit(X[tr], T[tr], X[te_], 1e3)
            ok = ch = 0
            for i, s in enumerate(S):
                best = s[3][int(np.argmax([P[i] @ Cn[idx[v]] for v in s[3]]))]
                ok += best in s[2]; ch += len(set(s[2]) & set(s[3])) / len(s[3])
            lines.append(f"  picture at sentence ends ({name}), {cond}: pick {ok / len(S):.3f} (chance {ch / len(S):.3f}, n {len(S)})")
            # the whole stick figure at the gather spot (end of the prompt, nobody asking): slot k = the k-th written number
            G = res[cond][4]
            ok = ch = n = 0
            for k in range(3):
                Gk = [g for g in G if len(g[2]) > k]
                if len(Gk) < 10:
                    continue
                X = np.array([g[1][ai] for g in Gk]); prob = np.array([g[0] for g in Gk])
                T = Cn[[idx[g[2][k]] for g in Gk]]
                P = np.zeros_like(T)
                for f in range(4):
                    tr, te_ = prob % 4 != f, prob % 4 == f
                    P[te_] = ridge_fit(X[tr], T[tr], X[te_], 1e3)
                for i, g in enumerate(Gk):
                    ok += g[3][int(np.argmax([P[i] @ Cn[idx[v]] for v in g[3]]))] == g[2][k]; ch += 1 / len(g[3]); n += 1
            lines.append(f"  whole picture at the end of the prompt ({name}), {cond}: slot pick {ok / max(n, 1):.3f} (chance {ch / max(n, 1):.3f}, n {n})")
    # drift: next-word odds on his own thinking, edited vs untouched
    T_ = [r for r in map(json.loads, open(think_f)) if r["id"].startswith("sit/mw/")][:40]
    kls = []
    for r in T_:
        ids = tok(tok.apply_chat_template([{"role": "user", "content": r["q"] + TAIL}], tokenize=False, add_generation_prompt=True,
                                          enable_thinking=True) + r["thinking"][:3000], add_special_tokens=False).input_ids[:1500]
        W.data[sl] = W0
        a, _ = M.run(ids)
        W.data[sl] = W0 + delta
        b, _ = M.run(ids)
        la, lb = torch.log_softmax(a, -1), torch.log_softmax(b, -1)
        kls.append((la.exp() * (la - lb)).sum(-1).cpu())
    k = torch.cat(kls)
    W.data[sl] = W0
    lines.append(f"  drift on his own thinking ({len(T_)} answers, {len(k)} words): KL mean {float(k.mean()):.4f}, "
                 f"99% {float(k.quantile(.99)):.3f}, max {float(k.max()):.3f}")
    open(out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    o = dict(a.split("=", 1) for a in sys.argv[4:] if "=" in a)
    if sys.argv[1] == "fit":
        fit(sys.argv[2], sys.argv[3], o)
    elif sys.argv[1] == "fit_bind":
        fit_bind(sys.argv[2], sys.argv[3], o)
    else:
        check(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], dict(a.split("=", 1) for a in sys.argv[6:]))
