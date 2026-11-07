#!/usr/bin/env python3
"""Builds a plain LFM3 GGUF: optional fix (take the opposing directions out of the writers' output matrices) and optional neuron cut (same count in every layer, the stock engine needs that).
usage: gguf_fix.py <in.gguf> <out.gguf> [fix=<eig.pt> gains=<gains.json> r=64 writers=all] [keep=<keep.npz>]"""
import os
import json, sys
import numpy as np, torch
sys.path.insert(0, os.path.expanduser("~") + "/work/llama.cpp-lfm3/gguf-py")
import gguf
from gguf import GGUFReader, GGUFWriter
args = [a for a in sys.argv[1:] if "=" not in a]; opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
src, dst = args
r = GGUFReader(src, "r"); T = {t.name: t for t in r.tensors}
ARCH = str(r.fields["general.architecture"].contents()); nb = int(r.fields[f"{ARCH}.block_count"].parts[-1][0])
def load(t):
    a = np.array(t.data)
    if a.dtype == np.uint8:
        a = a.view("<u2") if t.tensor_type.name == "BF16" else a.view("<f4")
    if t.tensor_type.name == "BF16":
        u = a.astype("<u2").astype("<u4") << 16
        return u.view("<f4").reshape(tuple(int(x) for x in t.shape[::-1]))
    return a.astype("<f4").reshape(tuple(int(x) for x in t.shape[::-1]))
def to_bf16(x):
    u = np.asarray(x, dtype="<f4").view("<u4")
    return ((u.astype("<u8") + 0x8000) >> 16).astype("<u2")
new = {}
log = dict(src=src, opt=opt)
if "fix" in opt:
    E = torch.load(opt["fix"]); U = E["U"]; order = E["order"]; R = int(opt.get("r", 64)); which = opt.get("writers", "all")
    gains = json.load(open(opt["gains"]))["gain"]
    for k, name in enumerate(order[:-1]):
        if which != "all" and not name.startswith(which):
            continue
        L = int(name.lstrip("opfn"))
        if name.startswith("ffn"):
            mn, pn = f"blk.{L}.ffn_down.weight", f"blk.{L}.post_ffw_norm.weight"
        else:
            mn = f"blk.{L}.shortconv.out_proj.weight" if f"blk.{L}.shortconv.out_proj.weight" in T else f"blk.{L}.attn_output.weight"
            pn = f"blk.{L}.post_attention_norm.weight"
        M = torch.from_numpy(load(T[mn]).copy()).double(); D = torch.from_numpy(load(T[pn]).copy()).double(); P = U[k][:, :R].double()
        DM = D[:, None] * M
        DM = DM - P @ (P.T @ DM)
        new[mn] = DM.float().numpy()
        new[pn] = (torch.ones_like(D) * gains[name]).float().numpy()
keep = None
plan_w2 = {}
OFF = [int(x) for x in opt["off"].split(",")] if "off" in opt else []
if "plan" in opt:
    PL = torch.load(opt["plan"]); mode = opt.get("mode", "rebuild")
    keep = {L: np.sort(PL[L]["energy_keep" if mode == "plain" else "keep"].numpy()) for L in range(nb)}
    if mode == "rebuild":
        plan_w2 = {f"blk.{L}.ffn_down.weight": PL[L]["w2"].float().numpy() for L in range(nb)}
        for L in range(nb):
            if PL[L].get("op") is not None:
                new[f"blk.{L}.{PL[L]['opname']}.weight"] = PL[L]["op"].float().numpy()
    n = {len(v) for v in keep.values()}
    if len(n) != 1:
        sys.exit(f"stock engine needs the same neuron count in every layer, got {n}")
    keepn = n.pop(); log["neurons"] = keepn
elif "keep" in opt:
    K = np.load(opt["keep"]); keep = {L: np.sort(K[f"L{L}"]) for L in range(nb)}
    n = {len(v) for v in keep.values()}
    if len(n) != 1:
        sys.exit(f"stock engine needs the same neuron count in every layer, got {n}")
    keepn = n.pop(); log["neurons"] = keepn
w = GGUFWriter(dst, arch=ARCH, endianess=r.endianess)
for f in r.fields.values():
    if f.name in ("GGUF.version", "GGUF.tensor_count", "GGUF.kv_count", "general.architecture"):
        continue
    val = f.contents(); tp = f.types[0]; sub = f.types[-1] if tp == gguf.GGUFValueType.ARRAY else None
    if keep is not None and f.name == f"{ARCH}.feed_forward_length":
        val = keepn
    if OFF and f.name == f"{ARCH}.feed_forward_length":
        full = int(f.contents()) if not isinstance(f.contents(), list) else None
        val = [32 if L in OFF else (full if full is not None else f.contents()[L]) for L in range(nb)]
        tp = gguf.GGUFValueType.ARRAY; sub = gguf.GGUFValueType.UINT32; log["off"] = OFF
    w.add_key_value(f.name, val, tp, sub_type=sub)
out = {}
for t in r.tensors:
    name = t.name
    change = name in new
    parts = name.split(".")
    cut = keep is not None and parts[0] == "blk" and parts[2] in ("ffn_gate", "ffn_up", "ffn_down")
    off = parts[0] == "blk" and int(parts[1]) in OFF and parts[2] in ("ffn_gate", "ffn_up", "ffn_down")
    if off:
        x = load(t); x = x[:, :32] * 0 if parts[2] == "ffn_down" else x[:32, :]
        out[name] = (to_bf16(x).reshape(-1), tuple(x.shape), t.tensor_type) if t.tensor_type.name == "BF16" else (x.astype("<f4").reshape(-1), tuple(x.shape), t.tensor_type); continue
    if not change and not cut:
        a = t.data   # memory-mapped view, written straight from disk (np.array here copied every tensor into RAM, which killed the 15 GB E4B build)
        if t.tensor_type.name == "BF16" and a.dtype == np.uint8:
            a = a.view("<u2")
        out[name] = (a, None if t.tensor_type.name in ("BF16", "F32") else "bytes", t.tensor_type); continue
    x = new[name] if change else load(t)
    if cut:
        L = int(parts[1])
        if name in plan_w2:
            x = plan_w2[name]   # already only the kept neurons, in the same ascending order
        else:
            x = x[:, keep[L]] if parts[2] == "ffn_down" else x[keep[L], :]
    if t.tensor_type.name == "BF16":
        out[name] = (to_bf16(x).reshape(-1), tuple(x.shape), t.tensor_type)
    else:
        out[name] = (x.astype("<f4").reshape(-1), tuple(x.shape), t.tensor_type)
for t in r.tensors:
    a, sh, ty = out[t.name]
    if sh == "bytes":
        w.add_tensor(t.name, a, raw_dtype=ty); continue
    shape = tuple(int(s) for s in t.shape[::-1]) if sh is None else sh
    a = a.reshape(shape)
    if ty.name == "F32":
        a = a.astype("<f4")
    w.add_tensor(t.name, a, raw_shape=list(shape), raw_dtype=ty)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
json.dump(log, open(dst + ".fix.json", "w"), indent=1)
print(f"GGUF_FIX_DONE {dst}: {len(new)} matrices or norms changed, neurons {log.get('neurons', 'all')}", flush=True)
