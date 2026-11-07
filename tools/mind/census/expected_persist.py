#!/usr/bin/env python3
"""Does the expected time stay in the stream while the patient thinks? Step 2 of Immy's time sense: the overtime
neuron fires in the middle of the thinking, so "how long this should take" has to still be readable there, not only on
the question's last word (difficulty_probe.py).

At thinking word t (t = 0, 16, 32, ...), only answers still thinking at t, so word position itself says nothing about
which of them run long (all are at the same t):
  carried   the question-trained line (difficulty_dirs.pt, built on other problems) read unchanged at word t: can
            one fixed neuron keep using it?
  refit     a fresh line built at word t on half these answers, scored on the other half: is it there at all?
  rho = rank agreement with the real thinking length (1 = perfect order, 0 = none).

usage: expected_persist.py <patient> <answers jsonl with thinking> <difficulty_dirs.pt> <out folder>
                           [layers=10,20 dev=mps]
"""
import json, os, sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from patients import load
from probe_math import rho, ridge

name, src, dirs, out = sys.argv[1:5]
opt = dict(a.split("=", 1) for a in sys.argv[5:])
LAYS, DEV = [int(x) for x in opt.get("layers", "10,20").split(",")], opt.get("dev", "mps")
TS = [0, 16, 32, 64, 128, 256, 512]
TAIL = "\n\nEnd with a line: Answer: <your answer>"
OPEN = {"e4b": "<|channel>thought\n", "lfm3": "\n"}[name]
MLP = {"e4b": lambda l: l.mlp, "lfm3": lambda l: l.feed_forward}[name]
D = torch.load(dirs, weights_only=False)["layers"]
p = load(name, DEV)
tok, layers = p.tok, p.model.model.layers
rows = [r for r in map(json.loads, open(src)) if r["thinking"]]
cap, pos = {}, [None]
for i in LAYS:
    MLP(layers[i]).register_forward_pre_hook(lambda m, a, i=i: cap.__setitem__(i, a[0][0, pos[0]].float().cpu()))

F = {i: {t: [] for t in TS} for i in LAYS}
Y = {t: [] for t in TS}
with torch.no_grad():
    for n, r in enumerate(rows):
        pr = tok(tok.apply_chat_template([{"role": "user", "content": r["q"] + TAIL}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=True), add_special_tokens=False).input_ids
        th = tok(OPEN + r["thinking"], add_special_tokens=False).input_ids[:max(TS) + 1]
        T = r["think_tokens"] if r["finish"] == "stop" else 6000
        ts = [t for t in TS if t < min(T, len(th))]
        pos[0] = torch.tensor([len(pr) - 1 + t for t in ts])  # t = 0 is the question's last word itself
        cap.clear()
        p.model.model(input_ids=torch.tensor([pr + th], device=DEV))
        for i in LAYS:
            for k, t in enumerate(ts):
                F[i][t].append(cap[i][k].numpy())
        for t in ts:
            Y[t].append(np.log(T))
        if (n + 1) % 25 == 0:
            print(f"  {n + 1}/{len(rows)}", flush=True)

lines = [f"Expected time while thinking ({name}, {len(rows)} answers): at word t, only answers still thinking at t",
         "layer  word  answers  carried  refit"]
for i in LAYS:
    w, b = D[i]["w"].numpy(), D[i]["b"]
    for t in TS:
        X, y = np.array(F[i][t]), np.array(Y[t])
        if len(y) < 20:
            continue
        a, c = np.arange(len(y)) % 2 == 0, np.arange(len(y)) % 2 == 1
        pr, _, _, _ = ridge(X[a], y[a], X[c])
        lines.append(f"{i:5d}  {t:4d}  {len(y):7d}   {rho(X @ w + b, y):+.2f}    {rho(pr, y[c]):+.2f}")
print("\n".join(lines))
open(os.path.join(out, "expected_persist.txt"), "w").write("\n".join(lines) + "\n")
print("PERSIST_DONE")
