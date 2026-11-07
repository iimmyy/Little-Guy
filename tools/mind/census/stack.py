#!/usr/bin/env python3
"""STACK test (Immy: "every trick added up makes a big difference"): early-layer neuron routing + input zeroing
together. Built from input_sparsity.py; the router part is route_e2e.py's perfect pick (an upper bound).

Lateral inhibition on the inputs (2026-10-26): before an MLP reads the stream, zero its weakest entries, so the
gate/up weight columns that would read them never need to be fetched (TEAL-style, training-free).

Two ways:
  plain    zero the s% smallest-magnitude entries of the MLP input for each word
  block    zero whole 32-number blocks (the size the 4-bit file stores weights in, so the saving is real on disk)
Drift = KL from the untouched model and top-guess agreement, wiki / chat / math, same test words as route_e2e.py.

usage: input_sparsity.py <out folder> <math traces jsonl>      writes input_sparsity.txt
"""
import json, os, sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from e4b_mac import load

out, traces = sys.argv[1], sys.argv[2]
BENCH = os.path.join(os.path.dirname(__file__), "../../../.run/bench")
model, tok = load()
layers = model.model.layers
wiki = tok(open(f"{BENCH}/wiki.test.raw").read()[:200000], add_special_tokens=False).input_ids
chat = tok(open(f"{BENCH}/chat_heldout.txt").read(), add_special_tokens=False).input_ids
rows = [json.loads(l) for l in open(traces)]


def mathwin(r):
    p = tok.apply_chat_template([{"role": "user", "content": r["q"]}], tokenize=False, add_generation_prompt=True,
                                enable_thinking=True)
    return tok(p + "<|channel>thought\n" + r["thinking"] + "<channel|>" + r["reply"], add_special_tokens=False).input_ids[:768]


test = [([tok.bos_token_id] + wiki[i * 511:(i + 1) * 511], "wiki") for i in range(16, 19)] + \
       [([tok.bos_token_id] + chat[i * 511:(i + 1) * 511], "chat") for i in range(32, 37)] + \
       [(mathwin(r), "math") for r in rows[3::42]]
cfg = {"mode": None, "s": 0.0, "route": set(), "keep": 1.0}
N = layers[0].mlp.intermediate_size
dnorm = [l.mlp.down_proj.weight.float().norm(dim=0) for l in layers]


def pre_down(i):
    def h(mod, args):
        if i not in cfg["route"]:
            return None
        p = args[0]
        size = p.float().abs() * dnorm[i]
        return (p * torch.zeros_like(p).scatter_(-1, size.topk(int(cfg["keep"] * N), dim=-1).indices, 1),)
    return h


for i, l in enumerate(layers):
    l.mlp.down_proj.register_forward_pre_hook(pre_down(i))
energy = {"plain": [], "block": []}


def sparsify(x):
    if cfg["mode"] is None:
        return x
    s = cfg["s"]
    if cfg["mode"] == "plain":
        k = int(s * x.shape[-1])
        thr = x.abs().float().kthvalue(k, dim=-1, keepdim=True).values
        return x * (x.abs() > thr)
    B = x.shape[-1] // 32
    e = x.float().view(*x.shape[:-1], B, 32).pow(2).sum(-1)
    k = int(s * B)
    thr = e.kthvalue(k, dim=-1, keepdim=True).values
    return (x.view(*x.shape[:-1], B, 32) * (e > thr).unsqueeze(-1)).view_as(x)


def pre(mod, args):
    return (sparsify(args[0]),)


for l in layers:
    l.mlp.gate_proj.register_forward_pre_hook(pre)
    l.mlp.up_proj.register_forward_pre_hook(pre)


def logp(w):
    return model(input_ids=torch.tensor([w], device="mps")).logits[0, 1:].float().log_softmax(-1)


CONF = [("route only", 0.0), ("plain", 0.15), ("plain", 0.25), ("stack 15", 0.15), ("stack 25", 0.25)]
kl = {(m, s, t): [] for m, s in CONF for t in ("wiki", "chat", "math")}
same = {c: [] for c in CONF}
with torch.no_grad():
    for k, (w, t) in enumerate(test):
        cfg["mode"], cfg["route"] = None, set()
        b = logp(w)
        for m, s in CONF:
            cfg["mode"] = None if m == "route only" else "plain"
            cfg["s"] = s
            cfg["route"], cfg["keep"] = (set(range(7)), 0.2) if m.startswith(("stack", "route")) else (set(), 1.0)
            lp = logp(w)
            kl[(m, s, t)].append((b.exp() * (b - lp)).sum(-1).mean().item())
            same[(m, s)].append((lp.argmax(-1) == b.argmax(-1)).float().mean().item())
            del lp
        del b
        torch.mps.empty_cache()
        print(f"  {k + 1}/{len(test)}", flush=True)
lines = ["STACK: layers 0-6 run only their best 20% of neurons + weakest MLP inputs zeroed in all layers", "",
         f"{'mode':6} {'zeroed':>7} {'KL wiki':>8} {'KL chat':>8} {'KL math':>8} {'same top':>9} {'MLP bytes saved':>16}"]
for m, s in CONF:
    lines.append(f"{m:6} {s:>7.0%} " + " ".join(f"{np.mean(kl[(m, s, t)]):>8.4f}" for t in ("wiki", "chat", "math"))
                 + f" {np.mean(same[(m, s)]) * 100:>8.1f}% {(s * 2 / 3 + (7 / 42 * 0.8 / 3 if m != 'plain' else 0)) * 100:>15.1f}%")
lines += ["", "(for scale: Google's own 4-bit release drifts 0.037 wiki / 0.027 chat from the 16-bit)"]
open(os.path.join(out, "stack.txt"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
print("STACK_DONE")
