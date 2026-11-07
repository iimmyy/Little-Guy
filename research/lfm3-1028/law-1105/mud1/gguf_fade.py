#!/usr/bin/env python3
"""Copies a GGUF and scales post-norm weights in place: the faders. Nothing else in the file changes, so it stays a stock model file.
usage: gguf_fade.py <in.gguf> <out.gguf> <rule> [<rule> ...]
rule: <layers>:<parts>=<gain>. layers: all, or 0-7, or 2,5,9. parts: conv, attn, op (either), ffn. Later rules multiply on top.
Writes <out>.fade.json with every gain applied and sha256 of the input's first 1 MB for tracing."""
import os
import hashlib, json, re, shutil, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~") + "/work/llama.cpp-lfm3/gguf-py")
from gguf import GGUFReader
src, dst, rules = sys.argv[1], sys.argv[2], sys.argv[3:]
r = GGUFReader(src, "r")
T = {t.name: t for t in r.tensors}
nb = int(r.fields["lfm3.block_count"].parts[-1][0])
conv = {i for i in range(nb) if f"blk.{i}.shortconv.conv.weight" in T}
gain = {}
for i in range(nb):
    gain[f"blk.{i}.post_attention_norm.weight"] = 1.0
    gain[f"blk.{i}.post_ffw_norm.weight"] = 1.0
for rule in rules:
    m = re.fullmatch(r"([^:]+):(conv|attn|op|ffn)=([0-9.]+)", rule)
    if not m:
        sys.exit(f"bad rule {rule}")
    ls, part, g = m.group(1), m.group(2), float(m.group(3))
    if ls == "all":
        L = range(nb)
    elif "-" in ls:
        a, b = map(int, ls.split("-")); L = range(a, b + 1)
    else:
        L = [int(x) for x in ls.split(",")]
    for i in L:
        if part == "ffn":
            gain[f"blk.{i}.post_ffw_norm.weight"] *= g
        elif part == "op" or (part == "conv") == (i in conv):
            gain[f"blk.{i}.post_attention_norm.weight"] *= g
shutil.copyfile(src, dst)
done = 0
with open(dst, "r+b") as f:
    for name, g in gain.items():
        if g == 1.0:
            continue
        t = T[name]
        if t.tensor_type.name != "F32":
            sys.exit(f"{name} is {t.tensor_type.name}, expected F32")
        f.seek(int(t.data_offset))
        a = np.frombuffer(f.read(int(t.n_bytes)), dtype="<f4")
        f.seek(int(t.data_offset))
        f.write((a * g).astype("<f4").tobytes())
        done += 1
h = hashlib.sha256(open(src, "rb").read(1 << 20)).hexdigest()
json.dump(dict(src=src, rules=rules, faders_changed=done, conv_layers=sorted(conv), gains={k: v for k, v in gain.items() if v != 1.0}, src_head_sha256=h), open(dst + ".fade.json", "w"), indent=1)
print(f"{dst}: {done} faders changed, rules {rules}")
