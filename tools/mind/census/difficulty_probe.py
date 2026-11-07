#!/usr/bin/env python3
"""Does the patient know, before it starts thinking, how long it will think? Step 1 of Immy's time sense (2026-10-29:
"all he should know is his own time": the clock is a feeling of overtime, so it needs an expected time to be over).

At the exact input of each layer's MLP (what a new neuron there would see), on the last word of the question (the
moment before the thinking starts): a straight-line reader (ridge regression) for y = log(thinking words) of the
patient's own plain answer to that question (a cut-off counts as 6000). Built on one set of problems, scored on a
different set, so it has to transfer to questions it never saw:
  rho        rank agreement with the real thinking length on the test set (1 = perfect order, 0 = none)
  in-level   the same, inside each MATH level and averaged: does it know more than the dataset's own difficulty label?
Baselines on the same test set: the MATH level itself, and the question's length in words.

usage: difficulty_probe.py <patient> <train problems> <train answers> <test problems> <test answers> <out folder>
                           [every=2 dev=mps]
answers: a plain run (gen_patient.py or engine_run.py rows; ids match the problems file, think_tokens + finish)
"""
import json, os, sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from patients import load
from probe_math import in_level, rho, ridge

name, trq, tra, teq, tea, out = sys.argv[1:7]
opt = dict(a.split("=", 1) for a in sys.argv[7:])
EVERY, DEV = int(opt.get("every", 2)), opt.get("dev", "mps")
os.makedirs(out, exist_ok=True)
TAIL = "\n\nEnd with a line: Answer: <your answer>"
MLP = {"e4b": lambda l: l.mlp, "lfm3": lambda l: l.feed_forward}[name]
p = load(name, DEV)
tok, layers = p.tok, p.model.model.layers
LAY = list(range(0, len(layers), EVERY))
cap = {}
for i in LAY:
    MLP(layers[i]).register_forward_pre_hook(lambda m, a, i=i: cap.__setitem__(i, a[0][0, -1].float().cpu()))


def rows(qf, af):
    qs = {r["id"]: r for r in map(json.loads, open(qf))}
    out = []
    for a in map(json.loads, open(af)):
        q = qs[a["id"]]
        out.append({"q": q["q"], "level": q["level"], "y": np.log(a["think_tokens"] if a["finish"] == "stop" else 6000) if a["think_tokens"] else np.log(6000)})
    return out


def read(R):
    F = {i: [] for i in LAY}
    qlen = []
    with torch.no_grad():
        for n, r in enumerate(R):
            pr = tok(tok.apply_chat_template([{"role": "user", "content": r["q"] + TAIL}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=True), add_special_tokens=False).input_ids
            cap.clear()
            p.model.model(input_ids=torch.tensor([pr], device=DEV))
            for i in LAY:
                F[i].append(cap[i])
            qlen.append(np.log(len(tok(r["q"], add_special_tokens=False).input_ids)))
            if (n + 1) % 25 == 0:
                print(f"  {n + 1}/{len(R)}", flush=True)
    return {i: torch.stack(v).numpy() for i, v in F.items()}, np.array(qlen)


Rtr, Rte = rows(trq, tra), rows(teq, tea)
ytr, yte = np.array([r["y"] for r in Rtr]), np.array([r["y"] for r in Rte])
lte = np.array([r["level"] for r in Rte])
print(f"train {len(Rtr)}, test {len(Rte)}", flush=True)
Ftr, qtr = read(Rtr)
Fte, qte = read(Rte)
lines = [f"Expected time inside {name}: does the question's last word already say how long the thinking will run?",
         f"train {len(Rtr)} problems, test {len(Rte)} other problems; y = log(own plain thinking words), cut-off = 6000",
         f"baselines on test: MATH level rho {rho(lte, yte):+.2f}; question length rho {rho(qte, yte):+.2f} "
         f"(in-level {in_level(qte, yte, lte):+.2f})",
         "layer   rho   in-level   train-cv   ridge"]
save = {}
for i in LAY:
    pr, al, cv, (w, b) = ridge(Ftr[i], ytr, Fte[i])
    lines.append(f"{i:5d}  {rho(pr, yte):+.2f}   {in_level(pr, yte, lte):+.2f}      {cv:+.2f}    {al:g}")
    save[i] = {"w": torch.tensor(w, dtype=torch.float32), "b": float(b), "rho": rho(pr, yte),
               "in_level": in_level(pr, yte, lte), "pred": pr.tolist()}
print("\n".join(lines))
open(os.path.join(out, "difficulty_probe.txt"), "w").write("\n".join(lines) + "\n")
torch.save({"layers": save, "y_test": yte.tolist(), "level_test": lte.tolist()}, os.path.join(out, "difficulty_dirs.pt"))
print("DIFFICULTY_DONE")
