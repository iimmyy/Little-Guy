#!/usr/bin/env python3
"""At the night forks of natural_error.py (his own runs, untouched thinking), is c5's key over its threshold, and how
likely is a doubt opener next? Kinds: right handed in, right then kept checking until cut off. Per model (c5, c5s ...):
the key's gate input (> 0 = over threshold, the doubt crowd ducked), and the chance the next word opens doubt.
If the key is silent where he doubts a right answer, the trust side has to lift the key there; if it already fires and
doubt still wins, the lever is past the key (the crowd).
usage: key_at_doubt.py <out.txt> <runs.jsonl> [...] name=<model folder> [name=<model folder> ...]
"""
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from patients import _lfm3
from situation_probe import TAIL
from natural_error import natural_forks, KINDS

DOUBT_IDS = [1938, 2375, 3911, 7731, 8952]
out = sys.argv[1]
runs = [a for a in sys.argv[2:] if "=" not in a]
models = [a.split("=", 1) for a in sys.argv[2:] if "=" in a]
items = [x for x in natural_forks(runs) if x[7] != 1]
kind = np.array([x[7] for x in items])
L = [f"c5's key at the night forks: " + ", ".join(f"{KINDS[k]} {int((kind == k).sum())}" for k in (0, 2))]
for name, path in models:
    p_ = _lfm3("cuda", path)
    tok, m = p_.tok, p_.model
    ff = m.model.layers[17].feed_forward
    got = {}
    h = ff.w1.register_forward_hook(lambda mod, a, o: got.__setitem__("g", o[0, -1, 1076].float()))
    G, D = [], []
    for (iid, text, think, A, W, (s0, s1, cut), run, kd) in items:
        head = tok.apply_chat_template([{"role": "user", "content": text + TAIL}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=True) + "\n"
        ids = tok(head + think[:cut], add_special_tokens=False).input_ids
        with torch.no_grad():
            lg = m(input_ids=torch.tensor([ids], device="cuda")).logits[0, -1].float()
        G.append(float(got["g"])); D.append(float(torch.softmax(lg, -1)[DOUBT_IDS].sum()))
    h.remove()
    G, D = np.array(G), np.array(D)
    for k in (0, 2):
        s = kind == k
        L.append(f"  {name}: {KINDS[k]}: key over threshold {np.mean(G[s] > 0):.0%} (gate input median {np.median(G[s]):+.2f}), "
                 f"chance next word opens doubt median {np.median(D[s]):.3f} mean {np.mean(D[s]):.3f}")
        print(L[-1], flush=True)
    del m
    torch.cuda.empty_cache()
open(out, "w").write("\n".join(L) + "\n")
print("\n".join(L))
