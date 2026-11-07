#!/usr/bin/env python3
"""Quiet lane (Immy 2026-11-02): reserve a section of the notepad where NO layer may write, and see how much the model
suffers. If little, the imager can keep its notes there instead of in pencil on a busy page.

Every layer's write (its output minus its input) has its part along k chosen directions removed, so those directions
keep exactly what the word itself put there and nothing more. Two kinds of section:
  quiet:  the k directions the layers write least (smallest variance of the summed writes on his own thinking)
  random: k random directions (the control)
Cost = how much the next-word odds on his own thinking change (KL vs untouched) and how often his top word changes.
Directions are chosen on half the answers and the cost is measured on the other half.

usage: quiet_lane.py <thinking.jsonl> <out.txt> [ks=8,32,128] [n=40] [hf=] [dev=mps]
"""
import json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from patients import _lfm3
from situation_probe import TAIL

src, out = sys.argv[1], sys.argv[2]
opt = dict(a.split("=", 1) for a in sys.argv[3:])
ks = [int(k) for k in opt.get("ks", "8,32,128").split(",")]
dev, N = opt.get("dev", "mps"), int(opt.get("n", 40))
p = _lfm3(dev, opt.get("hf"))
tok, m = p.tok, p.model
layers = m.model.layers
rows = [r for r in map(json.loads, open(src)) if r.get("thinking")][: 2 * N]
seqs = [tok(tok.apply_chat_template([{"role": "user", "content": r["q"] + TAIL}], tokenize=False, add_generation_prompt=True,
                                    enable_thinking=True) + r["thinking"][:3000], add_special_tokens=False).input_ids[:1200] for r in rows]
pick, test = seqs[:N], seqs[N:]

state = {"P": None, "inp": None, "acc": None}
def pre(mod, a, kw):
    state["inp"] = (a[0] if a else kw["hidden_states"])
def post(mod, a, kw, o):
    h = o[0] if isinstance(o, tuple) else o
    w = (h - state["inp"]).float()
    if state["acc"] is not None:
        state["acc"].append(w[0])
    if state["P"] is None:
        return None
    h2 = (h.float() - w @ state["P"]).to(h.dtype)          # remove the write along the reserved directions
    return (h2, *o[1:]) if isinstance(o, tuple) else h2
for l in layers:
    l.register_forward_pre_hook(pre, with_kwargs=True)
    l.register_forward_hook(post, with_kwargs=True)

def logits(ids):
    with torch.no_grad():
        return torch.log_softmax(m(input_ids=torch.tensor([ids], device=dev)).logits[0].float(), -1)

# 1. where do the layers write least? summed writes of all layers on the pick half
d = m.config.hidden_size
cov = torch.zeros(d, d, device=dev)
cnt = 0
for ids in pick:
    state["acc"] = []
    logits(ids)
    w = torch.stack(state["acc"]).sum(0)                    # (T, d): everything the layers wrote at each word
    cov += w.T @ w; cnt += w.shape[0]
state["acc"] = None
ev, V = torch.linalg.eigh((cov / cnt).cpu().double())
V = V.float()
base = [logits(ids) for ids in test]
L = [f"quiet lane: every layer barred from writing along k directions; cost on {len(test)} held-out answers of his own thinking "
     f"({sum(len(s) for s in test)} words); directions picked on {len(pick)} other answers",
     f"  write variance: smallest eigenvalue {ev[0]:.2e}, median {ev[d // 2]:.2e}, largest {ev[-1]:.2e}"]
g = torch.Generator().manual_seed(1102)
for k in ks:
    for kind in ("quiet", "random"):
        U = V[:, :k] if kind == "quiet" else torch.linalg.qr(torch.randn(d, k, generator=g))[0]
        state["P"] = (U @ U.T).to(dev)
        kl, flip, n = 0.0, 0, 0
        for ids, b in zip(test, base):
            a = logits(ids)
            kl += float((b.exp() * (b - a)).sum(-1).sum()); flip += int((a.argmax(-1) != b.argmax(-1)).sum()); n += a.shape[0]
        state["P"] = None
        share = float(ev[:k].sum() / ev.sum()) if kind == "quiet" else float("nan")
        L.append(f"  k {k:4d} {kind:6s}: KL mean {kl / n:.4f}, top word changed {100 * flip / n:.2f}% of words"
                 + (f"  (these directions carry {100 * share:.4f}% of all writing)" if kind == "quiet" else ""))
        print(L[-1], flush=True)
open(out, "w").write("\n".join(L) + "\n")
print("\n".join(L))
