#!/usr/bin/env python3
"""What does one head WRITE? (gather fetch diagnosis, 2026-11-02) The head's own output (its slice through o_proj,
after its gate) at sentence ends and at cue words, read in his own number code; untouched vs with the gather edit.
If the head's write carries the sentence's numbers once edited, the fetch works and the question is who reads it later.

usage: head_write.py <problems.jsonl> <edit.npz> <out.txt> [scale=1 folds=3,4 hf= dev=mps]
"""
import json, os, re, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gather_fetch import Model, prompt, story_sites
from situation_probe import stated, ridge_fit

src, edit_f, out = sys.argv[1:4]
opt = dict(a.split("=", 1) for a in sys.argv[4:])
E = np.load(edit_f)
layer, head, scale = int(E["layer"]), int(E["head"]), float(opt.get("scale", 1))
M = Model(opt, layer, head)
tok, att, hd = M.tok, M.att, M.hd
sl = slice(head * hd, (head + 1) * hd)
wr = {}
att.o_proj.register_forward_pre_hook(lambda mod, a: wr.__setitem__("o", (a[0][0, :, sl].float() @ mod.weight[:, sl].float().T)))
W = att.q_proj.weight
W0 = W.data[sl].clone()
delta = torch.tensor(scale * (E["A"] if "A" in E else float(E["a"]) * np.outer(E["d_hat"], E["s"])), dtype=W.dtype, device=W.device)
rows = list(map(json.loads, open(src)))
if "folds" in opt:                                   # e.g. folds=3,4: only problems the edit was not fitted on
    keep = {int(x) for x in opt["folds"].split(",")}
    rows = [r for i, r in enumerate(rows) if i % 5 in keep]
code = {}
def C(n):
    if n not in code:
        W.data[sl] = W0
        M.run(tok(f"The number is {n}", add_special_tokens=False).input_ids)
        code[n] = wr["o"][-1].cpu().numpy()          # head 15's own write on the number's plain sentence
    return code[n]
res = {}
for cond in ("untouched", "edited"):
    per, cue = [], []
    for pi, r in enumerate(rows):
        W.data[sl] = W0 + (delta if cond == "edited" else 0)
        ids, offs, q0 = prompt(tok, r["text"])
        ends, _, nums = story_sites(r, offs, q0)
        M.run(ids)
        o = wr["o"].cpu().numpy()
        allnums = sorted({int(x[0]) for x in re.finditer(r"(?<![\d.,])\d+(?![\d])", r["text"])})
        for t, ns in zip(ends, nums):
            if ns and len(allnums) >= 2:
                per.append((pi, o[t], ns, allnums))
        if cond == "untouched":
            for c, a in stated(r):
                M.run(ids + tok(c, add_special_tokens=False).input_ids)
                if len(allnums) >= 2:
                    cue.append((pi, wr["o"][-1].cpu().numpy(), [a], allnums))
    res[cond] = per
    if cue:
        res["asked (untouched)"] = cue
W.data[sl] = W0
L = [f"head {head} (layer {layer}) own write, read in number code (reader fitted on head-{head} writes of 'The number is N'), "
     f"{len(rows)} problems, edit scale {scale}"]
for name, S in res.items():
    nums = sorted({n for s in S for n in s[3]})
    Cm = np.stack([C(n) for n in nums]); mu = Cm.mean(0)
    Cn = Cm - mu; Cn /= np.linalg.norm(Cn, axis=1, keepdims=True) + 1e-8
    idx = {n: i for i, n in enumerate(nums)}
    X = np.array([s[1] for s in S]); prob = np.array([s[0] for s in S]); T = Cn[[idx[s[2][0]] for s in S]]
    P = np.zeros_like(T)
    for f in range(5):
        tr, te = prob % 5 != f, prob % 5 == f
        P[te] = ridge_fit(X[tr], T[tr], X[te], 1e1)
    ok = ch = 0
    for i, s in enumerate(S):
        ok += s[3][int(np.argmax([P[i] @ Cn[idx[v]] for v in s[3]]))] in s[2]; ch += len(set(s[2]) & set(s[3])) / len(s[3])
    size = float(np.linalg.norm(X, axis=1).mean())
    L.append(f"  {name:18s} sites {len(S):3d}: pick {ok / len(S):.3f} (chance {ch / len(S):.3f}); write size {size:.2f}")
open(out, "w").write("\n".join(L) + "\n"); print("\n".join(L))
