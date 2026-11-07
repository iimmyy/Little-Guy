#!/usr/bin/env python3
"""Everyday common sense for any patient (patients.py): the first N HellaSwag questions, pick the sensible ending of 4,
each ending scored by its average log-chance per word (zero-shot, same scoring as pam.py hs=N). Writes which questions
it got right, so two runs can be paired question by question.

usage: hs_patient.py <patient> <out.json> [n=1000 dev=cuda]
"""
import json, os, sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from patients import load

name, out = sys.argv[1:3]
opt = dict(a.split("=", 1) for a in sys.argv[3:])
N, DEV = int(opt.get("n", 1000)), opt.get("dev", "cuda")
BENCH = os.path.join(os.path.dirname(__file__), "../../../.run/bench")
p = load(name, DEV)
tok = p.tok
bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
hl = open(f"{BENCH}/hellaswag_val_full.txt").read().split("\n")
qs = [(hl[k], int(hl[k + 1]), hl[k + 2:k + 6]) for k in range(0, 6 * N, 6)]
right = []
with torch.no_grad():
    for ctx, lab, ends in qs:
        c = bos + tok(ctx, add_special_tokens=False).input_ids
        sc = []
        for e in ends:
            w = c + tok(" " + e, add_special_tokens=False).input_ids
            h = p.body(torch.tensor([w], device=DEV))[0, len(c) - 1:-1]          # the predictions for the ending words
            lp = p.logits(h).log_softmax(-1)
            sc.append(lp.gather(-1, torch.tensor(w[len(c):], device=DEV)[:, None]).mean().item())
        right.append(int(np.argmax(sc)) == lab)
json.dump({"patient": name, "n": N, "right": right, "score": float(np.mean(right))}, open(out, "w"))
print(f"{name} HellaSwag, {N} questions: {100 * np.mean(right):.1f}% right")
print("HS_DONE")
