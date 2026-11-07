#!/usr/bin/env python3
"""End to end: if a PERFECT router ran only the top m% of neurons (by this word's own output) in some layers, how far
does the whole model drift? (If the perfect pick breaks him, no cheap predictor can work.)

Drift = KL from the untouched model's next-word distribution, and how often its top guess is unchanged, on the same
test words as routing.py. Also the expected byte saving: the skipped neurons' gate/up/down rows are never read.

usage: route_e2e.py <out folder> <math traces jsonl>       writes route_e2e.txt
"""
import json, os, sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from e4b_mac import load

out, traces = sys.argv[1], sys.argv[2]
BENCH = os.path.join(os.path.dirname(__file__), "../../../.run/bench")
model, tok = load()
layers = model.model.layers
N = layers[0].mlp.intermediate_size
cap = model.config.final_logit_softcapping
wiki = tok(open(f"{BENCH}/wiki.test.raw").read()[:200000], add_special_tokens=False).input_ids
chat = tok(open(f"{BENCH}/chat_heldout.txt").read(), add_special_tokens=False).input_ids
rows = [json.loads(l) for l in open(traces)]


def mathwin(r):
    p = tok.apply_chat_template([{"role": "user", "content": r["q"]}], tokenize=False, add_generation_prompt=True,
                                enable_thinking=True)
    return tok(p + "<|channel>thought\n" + r["thinking"] + "<channel|>" + r["reply"], add_special_tokens=False).input_ids[:768]


test = [([tok.bos_token_id] + wiki[i * 511:(i + 1) * 511], "wiki") for i in range(16, 20)] + \
       [([tok.bos_token_id] + chat[i * 511:(i + 1) * 511], "chat") for i in range(32, 40)] + \
       [(mathwin(r), "math") for r in rows[3::21]]
dnorm = [l.mlp.down_proj.weight.float().norm(dim=0) for l in layers]
cfg = {"layers": set(), "keep": 1.0}


def pre_down(i):
    def h(mod, args):
        if i not in cfg["layers"]:
            return None
        p = args[0]
        m = int(cfg["keep"] * N)
        size = p.float().abs() * dnorm[i]
        mask = torch.zeros_like(p).scatter_(-1, size.topk(m, dim=-1).indices, 1)
        return (p * mask,)
    return h


for i, l in enumerate(layers):
    l.mlp.down_proj.register_forward_pre_hook(pre_down(i))


def logp(w):
    return model(input_ids=torch.tensor([w], device="mps")).logits[0, 1:].float().log_softmax(-1)   # HF applies the cap


SETS = {"0-6": set(range(7)), "0,1,5": {0, 1, 5}, "0-12": set(range(13)), "all 42": set(range(42))}
CONF = [(name, S, keep) for name, S in SETS.items() for keep in (0.05, 0.1, 0.2, 0.4)]
kl = {(c[0], c[2], s): [] for c in CONF for s in ("wiki", "chat", "math")}
same = {(c[0], c[2]): [] for c in CONF}
with torch.no_grad():
    for k, (w, s) in enumerate(test):                       # one window at a time: the reference never piles up
        cfg["layers"] = set()
        b = logp(w)
        for name, S, keep in CONF:
            cfg["layers"], cfg["keep"] = S, keep
            lp = logp(w)
            kl[(name, keep, s)].append((b.exp() * (b - lp)).sum(-1).mean().item())
            same[(name, keep)].append((lp.argmax(-1) == b.argmax(-1)).float().mean().item())
            del lp
        del b
        torch.mps.empty_cache()
        print(f"  {k + 1}/{len(test)}", flush=True)
lines = ["Perfect router, end to end: drift from the untouched model", "",
         f"{'layers':8} {'keep':>5} {'KL wiki':>8} {'KL chat':>8} {'KL math':>8} {'same top':>9} {'MLP bytes saved':>16}"]
for name, S, keep in CONF:
    lines.append(f"{name:8} {keep:>5.0%} " + " ".join(f"{np.mean(kl[(name, keep, s)]):>8.4f}" for s in ("wiki", "chat", "math"))
                 + f" {np.mean(same[(name, keep)]) * 100:>8.1f}% {len(S) / 42 * (1 - keep) * 100:>15.1f}%")
open(os.path.join(out, "route_e2e.txt"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
print("E2E_DONE")
