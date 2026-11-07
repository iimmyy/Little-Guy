#!/usr/bin/env python3
"""Swap a whole layer's MLP (10,240 neurons, 78.6M weights) for ONE straight-line equation out ~ A x + b (2560 x 2560,
6.6M weights). Immy's "simplify the equation", at the scale of a layer; the brain contrast: relay areas.

1. fit A, b per layer by least squares on the census words (x = what the MLP reads, after its input norm; out = what it
   writes, before its output norm)
2. on different words: how much of each layer's MLP output the line explains (R^2)
3. end to end: swap the k best layers' MLPs for their lines (k = 1, 2, 4, 6, 8) and measure drift as before

usage: mlp_linear.py <out folder> <math traces jsonl>      writes mlp_linear.txt and mlp_linear_maps.pt
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
L, D = len(layers), model.config.hidden_size
wiki = tok(open(f"{BENCH}/wiki.test.raw").read()[:200000], add_special_tokens=False).input_ids
chat = tok(open(f"{BENCH}/chat_heldout.txt").read(), add_special_tokens=False).input_ids
rows = [json.loads(l) for l in open(traces)]


def mathwin(r):
    p = tok.apply_chat_template([{"role": "user", "content": r["q"]}], tokenize=False, add_generation_prompt=True,
                                enable_thinking=True)
    return tok(p + "<|channel>thought\n" + r["thinking"] + "<channel|>" + r["reply"], add_special_tokens=False).input_ids[:768]


train = [[tok.bos_token_id] + wiki[i * 511:(i + 1) * 511] for i in range(16)] + \
        [[tok.bos_token_id] + chat[i * 511:(i + 1) * 511] for i in range(32)] + [mathwin(r) for r in rows[::7]]
test = [([tok.bos_token_id] + wiki[i * 511:(i + 1) * 511], "wiki") for i in range(16, 19)] + \
       [([tok.bos_token_id] + chat[i * 511:(i + 1) * 511], "chat") for i in range(32, 37)] + \
       [(mathwin(r), "math") for r in rows[3::42]]

grab = {}
mode = {"fit": False, "swap": {}, "maps": None}


def mlp_hook(i):
    def h(mod, args, o):
        if mode["fit"]:
            grab[i] = (args[0][0, 1:].float(), o[0, 1:].float())
        if i in mode["swap"]:
            A, b = mode["swap"][i]
            return (args[0].float() @ A + b).to(o.dtype)
        return None
    return h


for i, l in enumerate(layers):
    l.mlp.register_forward_hook(mlp_hook(i))

# 1. fit ------------------------------------------------------------------------------------------------------------
XtX = torch.zeros(L, D + 1, D + 1, dtype=torch.float64)
XtY = torch.zeros(L, D + 1, D, dtype=torch.float64)
YtY = torch.zeros(L, dtype=torch.float64); Ysum = torch.zeros(L, D, dtype=torch.float64); n = 0
mode["fit"] = True
with torch.no_grad():
    for k, w in enumerate(train):
        model.model(input_ids=torch.tensor([w], device="mps"))
        for i in range(L):
            x, y = grab[i]
            x = torch.cat([x, torch.ones(x.shape[0], 1, device="mps")], 1).cpu().double()
            y = y.cpu().double()
            XtX[i] += x.T @ x; XtY[i] += x.T @ y; YtY[i] += (y * y).sum(); Ysum[i] += y.sum(0)
        n += len(w) - 1
        grab.clear()
        torch.mps.empty_cache()
        if k % 10 == 9:
            print(f"  fit {k + 1}/{len(train)}", flush=True)
maps = []
for i in range(L):
    lam = 1e-4 * XtX[i].diagonal()[:D].mean()
    reg = lam * torch.eye(D + 1, dtype=torch.float64); reg[D, D] = 0
    Wb = torch.linalg.solve(XtX[i] + reg, XtY[i])
    maps.append((Wb[:D].float(), Wb[D].float()))
torch.save(maps, os.path.join(out, "mlp_linear_maps.pt"))
del XtX, XtY

# 2. R^2 on test words ----------------------------------------------------------------------------------------------
res = np.zeros(L); tot = np.zeros(L)
with torch.no_grad():
    for w, s in test:
        model.model(input_ids=torch.tensor([w], device="mps"))
        for i in range(L):
            x, y = grab[i]
            A, b = maps[i]
            pred = x @ A.to("mps") + b.to("mps")
            res[i] += ((y - pred) ** 2).sum().item()
            tot[i] += ((y - y.mean(0)) ** 2).sum().item()
        grab.clear()
        torch.mps.empty_cache()
mode["fit"] = False
r2 = 1 - res / tot
order = list(np.argsort(-r2))
lines = ["Whole-MLP straight-line swap", "", "R^2 per layer on test words (1 = the line explains the MLP fully):"]
lines += ["  " + " ".join(f"{i}:{r2[i]:.2f}" for i in range(L))]
lines += ["  best layers: " + ", ".join(f"{i} ({r2[i]:.3f})" for i in order[:10]), ""]
print("\n".join(lines), flush=True)


# 3. end to end -----------------------------------------------------------------------------------------------------
def logp(w):
    return model(input_ids=torch.tensor([w], device="mps")).logits[0, 1:].float().log_softmax(-1)


KS = [1, 2, 4, 6, 8]
kl = {(k, t): [] for k in KS for t in ("wiki", "chat", "math")}
same = {k: [] for k in KS}
dev_maps = {i: (maps[i][0].to("mps"), maps[i][1].to("mps")) for i in order[:max(KS)]}
with torch.no_grad():
    for j, (w, t) in enumerate(test):
        mode["swap"] = {}
        b = logp(w)
        for k in KS:
            mode["swap"] = {i: dev_maps[i] for i in order[:k]}
            lp = logp(w)
            kl[(k, t)].append((b.exp() * (b - lp)).sum(-1).mean().item())
            same[k].append((lp.argmax(-1) == b.argmax(-1)).float().mean().item())
            del lp
        del b
        torch.mps.empty_cache()
        print(f"  e2e {j + 1}/{len(test)}", flush=True)
mlp_w = 3 * D * layers[0].mlp.intermediate_size
lines += [f"{'swapped':>8} {'layers':24} {'KL wiki':>8} {'KL chat':>8} {'KL math':>8} {'same top':>9} {'MLP bytes saved':>16}"]
for k in KS:
    lines.append(f"{k:>8} {','.join(map(str, sorted(order[:k]))):24} " +
                 " ".join(f"{np.mean(kl[(k, t)]):>8.4f}" for t in ("wiki", "chat", "math")) +
                 f" {np.mean(same[k]) * 100:>8.1f}% {k * (mlp_w - D * D) / (L * mlp_w) * 100:>15.1f}%")
lines += ["", "(for scale: Google's own 4-bit release drifts 0.037 wiki / 0.027 chat from the 16-bit)"]
open(os.path.join(out, "mlp_linear.txt"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines[-len(KS) - 3:]))
print("LINEAR_DONE")
