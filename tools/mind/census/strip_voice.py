#!/usr/bin/env python3
"""Keep a baked feeling, drop its voice: the output rows the urgency receptors rewrote (the close word and
"Alternatively") go back to the untouched rows (the tied embedding), so the time sense (carrier, overtime neuron,
chorus) stays in the stream with nothing acting on it yet. Step 1 of the combined build (Immy 2026-10-29).

usage: strip_voice.py <baked folder with urgency.json> <out folder>
"""
import json, os, shutil, sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from patients import _lfm3

src, out = sys.argv[1:3]
info = json.load(open(os.path.join(src, "urgency.json")))
p = _lfm3("cpu", src)
m = p.model
with torch.no_grad():
    for tid in (info["end"], info["alt"]):
        before = float((m.lm_head.weight[tid].float() - m.model.embed_tokens.weight[tid].float()).norm())
        m.lm_head.weight[tid] = m.model.embed_tokens.weight[tid]
        print(f"row {tid} ({p.tok.decode([tid])!r}): moved back by {before:.3f}")
    other = (m.lm_head.weight.float() - m.model.embed_tokens.weight.float()).norm(dim=1)
    print(f"rows still different from the untouched ones: {int((other > 1e-4).sum())}")
os.makedirs(out, exist_ok=True)
m.save_pretrained(out)
p.tok.save_pretrained(out)
for fn in os.listdir(src):
    if fn.endswith((".py", ".jinja", ".pt", ".json")) and not os.path.exists(os.path.join(out, fn)):
        shutil.copy(os.path.join(src, fn), out)
json.dump({**info, "voice": "stripped"}, open(os.path.join(out, "urgency.json"), "w"), indent=1)
print("STRIP_DONE")
