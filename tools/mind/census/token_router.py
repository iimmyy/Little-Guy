#!/usr/bin/env python3
"""Word-keyed router for E4B's early layers (2026-10-26). routing.py found layers 0-6 extremely sparse (the best 20%
of neurons carry ~95-99% of the output) but "habit" (last word's neurons) fails. The brain contrast: early layers are
like the sensory periphery, labeled lines per stimulus, so the active neurons may follow the WORD more than context.

Two training-free tables, tested on words in context:
  seen      for each word, its average neuron sizes over the census text (needs calibration text)
  alone     for each word, its neuron sizes when it is run through the model on its own, after just <bos> (needs NO
            text at all: the table comes from the weights, so it works on any model)
Keep the table's top m% for that word; error = fraction of the layer's MLP output lost, as in routing.py.

usage: token_router.py <out folder> <math traces jsonl>      writes token_router.txt
"""
import json, os, sys
from collections import defaultdict

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
LS = list(range(0, 13))
BUD = [0.02, 0.05, 0.1, 0.2]
dnorm = {i: layers[i].mlp.down_proj.weight.float().norm(dim=0) for i in LS}
wiki = tok(open(f"{BENCH}/wiki.test.raw").read()[:200000], add_special_tokens=False).input_ids
chat = tok(open(f"{BENCH}/chat_heldout.txt").read(), add_special_tokens=False).input_ids
rows = [json.loads(l) for l in open(traces)]


def mathwin(r):
    p = tok.apply_chat_template([{"role": "user", "content": r["q"]}], tokenize=False, add_generation_prompt=True,
                                enable_thinking=True)
    return tok(p + "<|channel>thought\n" + r["thinking"] + "<channel|>" + r["reply"], add_special_tokens=False).input_ids[:768]


train = [[tok.bos_token_id] + wiki[i * 511:(i + 1) * 511] for i in range(16)] + \
        [[tok.bos_token_id] + chat[i * 511:(i + 1) * 511] for i in range(32)] + [mathwin(r) for r in rows[::7]]
test = [([tok.bos_token_id] + wiki[i * 511:(i + 1) * 511], "wiki") for i in range(16, 20)] + \
       [([tok.bos_token_id] + chat[i * 511:(i + 1) * 511], "chat") for i in range(32, 40)] + \
       [(mathwin(r), "math") for r in rows[3::21]]

cap = {}
g_store = {}


def gate_hook(i):
    def h(mod, args, o):
        g_store[i] = o
    return h


def up_hook(i):
    def h(mod, args, o):
        p = F.gelu(g_store[i][0].float(), approximate="tanh") * o[0].float()
        cap[i] = p
    return h


for i in LS:
    layers[i].mlp.gate_proj.register_forward_hook(gate_hook(i))
    layers[i].mlp.up_proj.register_forward_hook(up_hook(i))


def forward(ids):
    """run only through layer max(LS): stop early by raising after it"""
    class Stop(Exception):
        pass

    def stopper(mod, args, o):
        raise Stop
    hk = layers[max(LS)].register_forward_hook(stopper)
    try:
        model.model(input_ids=torch.tensor([ids], device="mps"))
    except Stop:
        pass
    hk.remove()


# table 1: average sizes per word over the census text (fp16, CPU) ---------------------------------------------------
seen_sum, seen_n = {}, defaultdict(int)
with torch.no_grad():
    for w in train:
        forward(w)
        S = torch.stack([(cap[i].abs() * dnorm[i]) for i in LS], 1).half().cpu()          # T x L x N
        for t, tid in enumerate(w):
            if t == 0:
                continue
            if tid in seen_sum:
                seen_sum[tid] += S[t].float()
            else:
                seen_sum[tid] = S[t].float()
            seen_n[tid] += 1
        torch.mps.empty_cache()
print(f"seen table: {len(seen_sum)} distinct words", flush=True)

# table 2: each test word alone after <bos> (no text needed) -----------------------------------------------------
need = sorted({tid for w, _ in test for tid in w[1:]})
alone = {}
with torch.no_grad():
    for c0 in range(0, len(need), 256):
        chunk = need[c0:c0 + 256]
        ids = torch.tensor([[tok.bos_token_id, t] for t in chunk], device="mps")

        class Stop(Exception):
            pass
        caps = {}

        def up_b(i):
            def h(mod, args, o):
                caps[i] = (F.gelu(g_store[i][:, 1].float(), approximate="tanh") * o[:, 1].float()).abs() * dnorm[i]
            return h
        hs = [layers[i].mlp.up_proj.register_forward_hook(up_b(i)) for i in LS]

        def stopper(mod, args, o):
            raise Stop
        hk = layers[max(LS)].register_forward_hook(stopper)
        try:
            model.model(input_ids=ids)
        except Stop:
            pass
        for h in hs + [hk]:
            h.remove()
        S = torch.stack([caps[i] for i in LS], 1).half().cpu()
        for j, t in enumerate(chunk):
            alone[t] = S[j]
print(f"alone table: {len(alone)} words", flush=True)

# test --------------------------------------------------------------------------------------------------------------
res = defaultdict(list)
with torch.no_grad():
    for w, s in test:
        forward(w)
        T = len(w) - 1
        for li, i in enumerate(LS):
            p = cap[i][1:]
            Wd = layers[i].mlp.down_proj.weight.float()
            full = p @ Wd.T
            fn = full.norm(dim=1).clamp_min(1e-12)
            size = p.abs() * dnorm[i]
            seen_rows = torch.stack([(seen_sum[t][li] / seen_n[t]) if t in seen_sum else torch.full((N,), 1.0)
                                     for t in w[1:]]).to(p.device)
            alone_rows = torch.stack([alone[t][li].float() for t in w[1:]]).to(p.device)
            for b in BUD:
                m = int(b * N)
                for kind, score in (("oracle", size), ("seen", seen_rows), ("alone", alone_rows)):
                    mask = torch.zeros(T, N, device=p.device).scatter_(1, score.topk(m, dim=1).indices, 1.0)
                    e = ((p * mask) @ Wd.T - full).norm(dim=1) / fn
                    res[(i, b, kind)].append(e.mean().item())
                    res[(s, b, kind, "early")].append(e.mean().item()) if i <= 6 else None
        torch.mps.empty_cache()

lines = ["Word-keyed router, early layers: fraction of each layer's MLP output LOST when only m% of neurons run", "",
         "by layer:   keep   oracle    seen   alone"]
for i in LS:
    for b in BUD:
        lines.append(f"  layer {i:>2}  {b:>4.0%}  " + "  ".join(f"{np.mean(res[(i, b, k)]):.3f}" for k in ("oracle", "seen", "alone")))
lines += ["", "layers 0-6 together, by text:"]
for s in ("wiki", "chat", "math"):
    for b in BUD:
        lines.append(f"  {s:5} {b:>4.0%}  " + "  ".join(f"{np.mean(res[(s, b, k, 'early')]):.3f}" for k in ("oracle", "seen", "alone")))
open(os.path.join(out, "token_router.txt"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
print("TOKEN_ROUTER_DONE")
