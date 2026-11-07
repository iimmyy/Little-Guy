#!/usr/bin/env python3
"""Change a set of neurons (Immy 2026-10-29: "we don't punish, we change a set of neurons"). The neurons that
logic_cluster.py found busier when his logic goes well get a little more gain, the ones busier when it goes badly a
little less: each neuron's output column (w2) is scaled by 1 + gamma (good side) or 1 - gamma (bad side). A pure
weight edit, like a dose change on one receptor population; nothing is added, nothing listens for words.

Only neurons that hold up on BOTH halves of the questions are used (same sign, |effect| >= min on each half).

usage: make_gain.py <hunt .pt from logic_cluster.py> <out folder> [hf=~/.littleguy/hf/lfm3-1.7b n=200 gamma=0.25
                    min=0.3 side=both|good|bad]
"""
import json, os, shutil, sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from patients import _lfm3

hunt, out = sys.argv[1:3]
opt = dict(a.split("=", 1) for a in sys.argv[3:])
HF = os.path.expanduser(opt.get("hf", "~/.littleguy/hf/lfm3-1.7b"))
N, G, MIN, SIDE = int(opt.get("n", 200)), float(opt.get("gamma", 0.25)), float(opt.get("min", 0.3)), opt.get("side", "both")

h = torch.load(hunt, weights_only=False)
eA, eB, eAll = h["eA"], h["eB"], h["eAll"]
ok = (torch.sign(eA) == torch.sign(eB)) & (eA.abs() >= MIN) & (eB.abs() >= MIN)
if SIDE == "good":
    ok &= eAll > 0
elif SIDE == "bad":
    ok &= eAll < 0
cand = torch.nonzero(ok)
cand = cand[torch.argsort(-eAll[ok].abs())][:N]
print(f"{int(ok.sum())} neurons hold up on both halves; using the {len(cand)} strongest (side {SIDE}), gamma {G}")

p = _lfm3("cpu", HF)
lay = p.model.model.layers
placed = []
with torch.no_grad():
    for l, j in cand.tolist():
        s = 1 + G if eAll[l, j] > 0 else 1 - G
        w = lay[l].feed_forward.w2.weight
        w[:, j] = (w[:, j].float() * s).to(w.dtype)
        placed.append([l, j, round(s, 3), round(float(eAll[l, j]), 3)])
per_layer = {}
for l, j, s, e in placed:
    per_layer[l] = per_layer.get(l, 0) + 1
print("per layer:", dict(sorted(per_layer.items())))
os.makedirs(out, exist_ok=True)
p.model.save_pretrained(out)
p.tok.save_pretrained(out)
for fn in os.listdir(HF):
    if fn.endswith((".py", ".jinja", ".json")) and not os.path.exists(os.path.join(out, fn)):
        shutil.copy(os.path.join(HF, fn), out)
json.dump({"hunt": hunt, "gamma": G, "min": MIN, "side": SIDE, "neurons": placed}, open(os.path.join(out, "gain.json"), "w"), indent=1)
print("GAIN_DONE")
