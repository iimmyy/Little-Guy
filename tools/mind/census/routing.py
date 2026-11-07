#!/usr/bin/env python3
"""Routing between neurons without training (Immy, 2026-10-26: "what if we built a better routing system between each
neuron ... how can someone take any model and apply the math and see gains").

For every word and every layer: how few of the 10,240 neurons are really needed, and can we know which ones without
computing all of them?
  oracle   keep the m neurons with the biggest output for THIS word (needs all of them computed: the ceiling)
  habit    keep the m neurons that were biggest for the PREVIOUS word (no extra math at all: neurons that just fired
           fire again, like a brain's ongoing activity)
  random   keep m random neurons (the floor)
error = how much of the layer's MLP output vector is lost: |out - out_kept| / |out|.  Measure only, nothing changed.

usage: routing.py <out folder> [math traces jsonl]      writes routing.txt
"""
import json, os, sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from e4b_mac import load

out = sys.argv[1]
traces = sys.argv[2] if len(sys.argv) > 2 else None
BENCH = os.path.join(os.path.dirname(__file__), "../../../.run/bench")
model, tok = load()
layers = model.model.layers
L, N = len(layers), layers[0].mlp.intermediate_size
BUD = [0.05, 0.1, 0.2, 0.4]

wins, src = [], []
wiki = tok(open(f"{BENCH}/wiki.test.raw").read()[:200000], add_special_tokens=False).input_ids
for i in range(16, 20):
    wins.append([tok.bos_token_id] + wiki[i * 511:(i + 1) * 511]); src.append("wiki")
chat = tok(open(f"{BENCH}/chat_heldout.txt").read(), add_special_tokens=False).input_ids
for i in range(32, 40):
    wins.append([tok.bos_token_id] + chat[i * 511:(i + 1) * 511]); src.append("chat")
if traces:
    for r in [json.loads(l) for l in open(traces)][3::21]:
        p = tok.apply_chat_template([{"role": "user", "content": r["q"]}], tokenize=False, add_generation_prompt=True,
                                    enable_thinking=True)
        wins.append(tok(p + "<|channel>thought\n" + r["thinking"] + "<channel|>" + r["reply"],
                        add_special_tokens=False).input_ids[:768]); src.append("math")

err = {(s, kind, b): [] for s in dict.fromkeys(src) for kind in ("oracle", "habit", "random") for b in BUD}
err_layer = np.zeros((3, len(BUD), L)); cnt = np.zeros(L)
cur = {"src": None}
dnorm = [l.mlp.down_proj.weight.float().norm(dim=0) for l in layers]
g_store = [None] * L
gen = torch.Generator(device="cpu").manual_seed(0)


def gate_hook(i):
    def h(mod, args, o):
        g_store[i] = o
    return h


def up_hook(i):
    def h(mod, args, o):
        p = (F.gelu(g_store[i][0, 1:].float(), approximate="tanh") * o[0, 1:].float())        # T x N
        Wd = layers[i].mlp.down_proj.weight.float()                                          # 2560 x N
        full = p @ Wd.T
        fn = full.norm(dim=1).clamp_min(1e-12)
        size = p.abs() * dnorm[i]
        T = p.shape[0]
        for bj, b in enumerate(BUD):
            m = int(b * N)
            top = size.topk(m, dim=1).indices
            keep = {"oracle": top[1:], "habit": top[:-1],
                    "random": torch.stack([torch.randperm(N, generator=gen)[:m] for _ in range(T - 1)]).to(p.device)}
            for kj, (kind, idx) in enumerate(keep.items()):
                mask = torch.zeros(T - 1, N, device=p.device).scatter_(1, idx, 1.0)
                e = ((p[1:] * mask) @ Wd.T - full[1:]).norm(dim=1) / fn[1:]
                err[(cur["src"], kind, b)].append(e.mean().item())
                err_layer[kj, bj, i] += e.mean().item()
        cnt[i] += 1
    return h


for i, l in enumerate(layers):
    l.mlp.gate_proj.register_forward_hook(gate_hook(i))
    l.mlp.up_proj.register_forward_hook(up_hook(i))
with torch.no_grad():
    for k, (w, s) in enumerate(zip(wins, src)):
        cur["src"] = s
        model.model(input_ids=torch.tensor([w], device="mps"))
        torch.mps.empty_cache()
        print(f"  {k + 1}/{len(wins)}", flush=True)

lines = ["Routing without training: fraction of each layer's MLP output LOST when only m% of neurons run", "",
         f"{'text':6} {'keep':>5} {'oracle':>8} {'habit':>8} {'random':>8}"]
for s in dict.fromkeys(src):
    for b in BUD:
        lines.append(f"{s:6} {b:>5.0%} " + " ".join(f"{np.mean(err[(s, k, b)]):>8.3f}" for k in ("oracle", "habit", "random")))
lines += ["", "by layer (all text), keep 20%:  layer: oracle / habit"]
bj = BUD.index(0.2)
lines += [f"  {i:>2}: {err_layer[0, bj, i] / cnt[i]:.3f} / {err_layer[1, bj, i] / cnt[i]:.3f}" for i in range(L)]
open(os.path.join(out, "routing.txt"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
print("ROUTING_DONE")
