#!/usr/bin/env python3
"""Surgical quiet lane = sidechain duck on the note (Immy 2026-11-02: "way softer, more surgical, more planned").
The hammer (quiet_lane.py) barred every layer from k directions at every word: 7% of words changed at k=8.
Here three dials instead:
  where:    head 15's own note directions: the top r directions of its writes at sentence ends (binding edit on),
            found on folds 0-2
  when:     only at sentence-end words (where the note sits), only layers after the note is written (15..23)
  how much: those layers' writes along the note directions turned DOWN by g (25/50/75%), never muted elsewhere
Conditions: untouched, binding edit alone, binding edit + duck g. Checked on folds 3-4: the picture at sentence ends
and at the prompt end in the WHOLE notepad (his number code), and drift on his own thinking (sentence ends found the
same way in the thinking).

usage: note_duck.py <problems.jsonl> <bind_edit.npz> <thinking.jsonl> <out.txt> [r=8 gs=0.25,0.5,0.75 shrink=0.1 hf= dev=mps]
"""
import json, os, re, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gather_fetch import Model, prompt, story_sites, END
from situation_probe import ridge_fit, TAIL

src, edit_f, think_f, out = sys.argv[1:5]
opt = dict(a.split("=", 1) for a in sys.argv[5:])
r_dim, gs = int(opt.get("r", 8)), [float(g) for g in opt.get("gs", "0.25,0.5,0.75").split(",")]
E = np.load(edit_f)
layer, head = int(E["layer"]), int(E["head"])
M = Model(opt, layer, head)
tok, att, hd, dev = M.tok, M.att, M.hd, M.dev
sl = slice(head * hd, (head + 1) * hd)
Wq = att.q_proj.weight
W0 = Wq.data[sl].clone()
A = torch.tensor(E["A"], dtype=Wq.dtype, device=Wq.device)
wr = {}
att.o_proj.register_forward_pre_hook(lambda mod, a: wr.__setitem__("o", a[0][0, :, sl].float() @ mod.weight[:, sl].float().T))
duck = {"g": 0.0, "pos": None, "U": None, "inp": None}
later = list(range(layer + 1, len(M.m.model.layers)))
def pre(mod, a, kw):
    duck["inp"] = a[0] if a else kw["hidden_states"]
def post(mod, a, kw, o):
    if not duck["g"] or duck["pos"] is None:
        return None
    h = o[0] if isinstance(o, tuple) else o
    w = (h - duck["inp"])[0, duck["pos"]].float()
    h = h.clone()
    h[0, duck["pos"]] = (h[0, duck["pos"]].float() - duck["g"] * (w @ duck["U"]) @ duck["U"].T).to(h.dtype)
    return (h, *o[1:]) if isinstance(o, tuple) else h
for i in later:
    M.m.model.layers[i].register_forward_pre_hook(pre, with_kwargs=True)
    M.m.model.layers[i].register_forward_hook(post, with_kwargs=True)

rows = list(map(json.loads, open(src)))
fitrows, testrows = [r for i, r in enumerate(rows) if i % 5 < 3], [r for i, r in enumerate(rows) if i % 5 >= 3]
# 1. where: head 15's note directions, binding edit on, sentence ends of the fit problems
Wq.data[sl] = W0 + A
notes = []
for r in fitrows:
    ids, offs, q0 = prompt(tok, r["text"])
    ends, _, _ = story_sites(r, offs, q0)
    M.run(ids)
    notes += list(wr["o"][ends].cpu().numpy())
notes = np.array(notes)
_, S, Vt = np.linalg.svd(notes - notes.mean(0), full_matrices=False)
U = torch.tensor(Vt[:r_dim].T, dtype=torch.float32, device=dev)
duck["U"] = U
L = [f"note duck: layer {layer} head {head} note directions r {r_dim} (share of note variance {float((S[:r_dim] ** 2).sum() / (S ** 2).sum()):.3f}), "
     f"ducked in layers {later[0]}-{later[-1]} at sentence ends only; {len(testrows)} held-out problems"]

def ends_in(ids_text_offs, text, q0):
    offs = ids_text_offs
    return [next((i for i, (s, e) in enumerate(offs) if s <= q0 + mm.start() < e), None) for mm in END.finditer(text)]

conds = [("untouched", False, 0.0), ("binding", True, 0.0)] + [(f"binding+duck {g:.2f}", True, g) for g in gs]
res = {}
for name, bind, g in conds:
    Wq.data[sl] = W0 + (A if bind else 0)
    duck["g"] = g
    sites, gath = [], []
    for pi, r in enumerate(testrows):
        ids, offs, q0 = prompt(tok, r["text"])
        ends, _, nums = story_sites(r, offs, q0)
        duck["pos"] = ends
        M.run(ids)
        H = torch.stack([M.cap[i] for i in (layer + 1, len(M.m.model.layers))]).float().cpu().numpy()
        allnums = sorted({int(x[0]) for x in re.finditer(r"(?<![\d.,])\d+(?![\d])", r["text"])})
        order = [int(x[0]) for x in re.finditer(r"(?<![\d.,])\d+(?![\d])", r["text"])]
        if len(allnums) >= 2:
            gath.append((pi, H[:, -1], order, allnums))
            for t, ns in zip(ends, nums):
                if ns:
                    sites.append((pi, H[:, t], ns, allnums))
    res[name] = (sites, gath)
    print(f"{name} done", flush=True)
duck["g"], duck["pos"] = 0.0, None
Wq.data[sl] = W0
nums = sorted({n for s, gth in res.values() for x in s + gth for n in x[3]})
C = []
for n in nums:
    M.run(tok(f"The number is {n}", add_special_tokens=False).input_ids)
    C.append(torch.stack([M.cap[i][-1] for i in (layer + 1, len(M.m.model.layers))]).float().cpu().numpy())
C = np.stack(C); idx = {n: i for i, n in enumerate(nums)}

Unp = U.cpu().numpy()
# figure-8 mic (Immy): learn the crowd from thousands of unlabeled words of his own thinking (untouched model), then
# listen along the note directions with the nulls on the crowd: w = (crowd covariance + shrink)^-1 . note direction
Wq.data[sl] = W0; duck["g"] = 0.0
crowd = [[], []]
for r in [r for r in map(json.loads, open(think_f)) if r["id"].startswith("sit/")][:60]:
    ids = tok(tok.apply_chat_template([{"role": "user", "content": r["q"] + TAIL}], tokenize=False, add_generation_prompt=True,
                                      enable_thinking=True) + r["thinking"][:3000], add_special_tokens=False).input_ids[:1200]
    M.run(ids)
    for ai, i in enumerate((layer + 1, len(M.m.model.layers))):
        crowd[ai].append(M.cap[i].float().cpu().numpy())   # every word (cap holds words x d)
NULLW = []
for ai in range(2):
    Z = np.concatenate(crowd[ai]); Z = Z - Z.mean(0)
    Sg = Z.T @ Z / len(Z)
    Sg += float(opt.get("shrink", 0.1)) * np.trace(Sg) / Sg.shape[0] * np.eye(Sg.shape[0])
    NULLW.append(np.linalg.solve(Sg, Unp))
L.append(f"  crowd for the figure-8 read: {sum(len(z) for z in crowd[0])} unlabeled words of his thinking, shrink {opt.get('shrink', 0.1)}")
SCREEN = {"on": False}
def pick(S, ai, Cn, target):
    X = np.array([s[1][ai] for s in S])
    if SCREEN["on"] == "null":             # figure-8: note directions, nulls on the crowd
        X = X @ NULLW[ai]
    elif SCREEN["on"]:                     # read through the 8 note directions only (the screen)
        X = X @ Unp
    prob = np.array([s[0] for s in S]); T = Cn[[idx[target(s)] for s in S]]
    P = np.zeros_like(T)
    for f in range(4):
        tr, te = prob % 4 != f, prob % 4 == f
        P[te] = ridge_fit(X[tr], T[tr], X[te], 1e-2 if SCREEN["on"] else 1e3)
    return P
for scr, ai, where in [(False, 0, f"index {layer + 1}"), (False, 1, "last index"), (True, 0, f"screen {layer + 1}"), (True, 1, "screen last"),
                      ("null", 0, f"fig8 {layer + 1}"), ("null", 1, "fig8 last")]:
    SCREEN["on"] = scr
    Cn = C[:, ai] - C[:, ai].mean(0); Cn /= np.linalg.norm(Cn, axis=1, keepdims=True) + 1e-8
    for name, _, _ in conds:
        S, G = res[name]
        P = pick(S, ai, Cn, lambda s: s[2][0])
        ok = sum(s[3][int(np.argmax([P[i] @ Cn[idx[v]] for v in s[3]]))] in s[2] for i, s in enumerate(S)) / len(S)
        ch = np.mean([len(set(s[2]) & set(s[3])) / len(s[3]) for s in S])
        okg = n = 0
        for k in range(3):
            Gk = [x for x in G if len(x[2]) > k]
            Pk = pick(Gk, ai, Cn, lambda x, k=k: x[2][k])
            okg += sum(x[3][int(np.argmax([Pk[i] @ Cn[idx[v]] for v in x[3]]))] == x[2][k] for i, x in enumerate(Gk)); n += len(Gk)
        L.append(f"  {where:9s} {name:20s}: picture at sentence ends {ok:.3f} (chance {ch:.3f}, n {len(S)}); whole picture at prompt end {okg / n:.3f} (n {n})")
# drift on his own thinking, each condition vs untouched, duck at the thinking's own sentence ends too
T_ = [r for r in map(json.loads, open(think_f)) if r["id"].startswith("sit/mw/")][:40]
seqs = []
for r in T_:
    text = tok.apply_chat_template([{"role": "user", "content": r["q"] + TAIL}], tokenize=False, add_generation_prompt=True,
                                   enable_thinking=True) + r["thinking"][:3000]
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc.input_ids[:1200], enc.offset_mapping[:1200]
    pos = sorted({i for mm in END.finditer(text) for i, (s, e) in enumerate(offs) if s <= mm.start() < e})
    seqs.append((ids, pos))
base = []
for ids, pos in seqs:
    lg, _ = M.run(ids); base.append(torch.log_softmax(lg, -1))
for name, bind, g in conds[1:]:
    Wq.data[sl] = W0 + (A if bind else 0)
    duck["g"] = g
    kl, flip, n = 0.0, 0, 0
    for (ids, pos), b in zip(seqs, base):
        duck["pos"] = pos
        lg, _ = M.run(ids); a = torch.log_softmax(lg, -1)
        kl += float((b.exp() * (b - a)).sum(-1).sum()); flip += int((a.argmax(-1) != b.argmax(-1)).sum()); n += a.shape[0]
    L.append(f"  drift {name:20s}: KL mean {kl / n:.4f}, top word changed {100 * flip / n:.2f}%")
duck["g"], duck["pos"] = 0.0, None
Wq.data[sl] = W0
open(out, "w").write("\n".join(L) + "\n")
print("\n".join(L))
