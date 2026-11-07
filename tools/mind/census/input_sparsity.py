#!/usr/bin/env python3
"""Lateral inhibition on the inputs (2026-10-26): before an MLP reads the stream, zero its weakest entries, so the
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
cfg = {"mode": None, "s": 0.0}
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


CONF = [(m, s) for m in ("plain", "block") for s in (0.25, 0.5, 0.7)]
kl = {(m, s, t): [] for m, s in CONF for t in ("wiki", "chat", "math")}
same = {c: [] for c in CONF}
with torch.no_grad():
    for k, (w, t) in enumerate(test):
        cfg["mode"] = None
        b = logp(w)
        for m, s in CONF:
            cfg["mode"], cfg["s"] = m, s
            lp = logp(w)
            kl[(m, s, t)].append((b.exp() * (b - lp)).sum(-1).mean().item())
            same[(m, s)].append((lp.argmax(-1) == b.argmax(-1)).float().mean().item())
            del lp
        del b
        torch.mps.empty_cache()
        print(f"  {k + 1}/{len(test)}", flush=True)
lines = ["Zeroing the weakest MLP inputs (all 42 layers, gate + up), end to end", "",
         f"{'mode':6} {'zeroed':>7} {'KL wiki':>8} {'KL chat':>8} {'KL math':>8} {'same top':>9} {'MLP bytes saved':>16}"]
for m, s in CONF:
    lines.append(f"{m:6} {s:>7.0%} " + " ".join(f"{np.mean(kl[(m, s, t)]):>8.4f}" for t in ("wiki", "chat", "math"))
                 + f" {np.mean(same[(m, s)]) * 100:>8.1f}% {s * 2 / 3 * 100:>15.1f}%")
lines += ["", "(for scale: Google's own 4-bit release drifts 0.037 wiki / 0.027 chat from the 16-bit)"]
open(os.path.join(out, "input_sparsity.txt"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
print("SPARSE_DONE")
