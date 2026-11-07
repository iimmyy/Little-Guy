#!/usr/bin/env python3
"""Control for the sidechain compressor: a copy of a compressor build with its key neuron put back exactly as it was in
the untouched model (gate row, up row, down column), so nothing writes the private wire. The crowd's ducking edits stay.
If the build still helps, the gain came from the crowd edits reading the stream on their own, not from the trigger.

usage: restore_key.py <untouched hf folder> <compressor build folder> <out folder>        (LFM3 save_pretrained builds)
"""
import json, os, shutil, sys
import torch
from safetensors import safe_open
from safetensors.torch import save_file

base, build, out = sys.argv[1:4]
c = json.load(open(os.path.join(build, "pam.json")))["compressor"]
L, j = c["key_layer"], c["key_neuron"]
pre = f"model.layers.{L}.feed_forward."
os.makedirs(out, exist_ok=True)
with safe_open(os.path.join(build, "model.safetensors"), "pt") as fb, safe_open(os.path.join(base, "model.safetensors"), "pt") as fa:
    meta = fb.metadata()
    full = {k: fb.get_tensor(k) for k in fb.keys()}
    for nm in ("w1", "w3"):                                           # gate and up: the key's row
        full[pre + nm + ".weight"][j] = fa.get_tensor(pre + nm + ".weight")[j]
    full[pre + "w2.weight"][:, j] = fa.get_tensor(pre + "w2.weight")[:, j]   # down: the key's column
    for nm in ("w1", "w3", "w2"):
        a, b = fa.get_tensor(pre + nm + ".weight"), full[pre + nm + ".weight"]
        print(f"layer {L} {nm}: {int((a != b).any(1 if nm != 'w2' else 0).sum())} neurons still differ from untouched")
save_file(full, os.path.join(out, "model.safetensors"), metadata=meta)
for fn in os.listdir(build):
    if fn != "model.safetensors" and os.path.isfile(os.path.join(build, fn)):
        shutil.copy(os.path.join(build, fn), out)
pj = json.load(open(os.path.join(out, "pam.json")))
pj["compressor"]["key_restored"] = True
json.dump(pj, open(os.path.join(out, "pam.json"), "w"), indent=1)
print(f"key layer {L} #{j} restored; saved {out}")
