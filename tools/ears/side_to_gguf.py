#!/usr/bin/env python3
"""Puts the trained side path (widen.py's side.pt) into a copy of his ears file (an mmproj .gguf), for llama.cpp:
  clip.audio.wide_bands = 39, and tensors a.wide.conv1d.{0,1}.weight, a.wide.conv1d.{0,1}.norm.weight, a.wide.proj.weight
(read by the fork's gemma4a graph and preprocessor). Everything else is copied as it is.
A side path trained with --where sep (the high end on its own page) also gets clip.audio.wide_pool, its hidden layer
a.wide.mid.{weight,bias} and a.wide.proj.bias. The late path (rounds 2-3) has no port and is refused.
usage: side_to_gguf.py EARS.gguf SIDE.pt OUT.gguf      (needs the llama.cpp tree's gguf-py on PYTHONPATH, and torch)
"""
import sys

import numpy as np
import torch
import gguf


def main(src, side_pt, out):
    ck = torch.load(side_pt, map_location="cpu")
    sd = {k: v.float().numpy() for k, v in ck["side"].items()}
    where = ck.get("where", "early")
    if where not in ("early", "sep"):
        raise SystemExit(f"a {where} side path has no llama.cpp port")
    new = {
        "a.wide.conv1d.0.weight": sd["l0.0.weight"],          # [128, 1, 3, 3]  (out, in, time, freq)
        "a.wide.conv1d.0.norm.weight": sd["l0.1.weight"],
        "a.wide.conv1d.1.weight": sd["l1.0.weight"],          # [32, 128, 3, 3]
        "a.wide.conv1d.1.norm.weight": sd["l1.1.weight"],
        "a.wide.proj.weight": sd["out.weight"],               # early [1024, 320]; sep [1536, mid or 320]
    }
    if where == "sep":
        if "mid.weight" in sd:
            new["a.wide.mid.weight"] = sd["mid.weight"]          # [mid, 320]
            new["a.wide.mid.bias"] = sd["mid.bias"]
        new["a.wide.proj.bias"] = sd["out.bias"]
    n_wide = int(round(len(ck["edges_hz"]) - 2))
    r = gguf.GGUFReader(src)
    arch = r.fields[gguf.Keys.General.ARCHITECTURE]
    w = gguf.GGUFWriter(out, arch=bytes(arch.parts[arch.data[0]]).decode())
    for f in r.fields.values():
        if f.name == gguf.Keys.General.ARCHITECTURE or f.name.startswith("GGUF."):
            continue
        t = f.types[0]
        w.add_key_value(f.name, f.contents(), t, sub_type=f.types[-1] if t == gguf.GGUFValueType.ARRAY else None)
    w.add_uint32("clip.audio.wide_bands", n_wide)
    if where == "sep":
        w.add_uint32("clip.audio.wide_pool", int(ck.get("pool", 4)))
    for t in r.tensors:
        w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    for name, a in new.items():
        a = np.ascontiguousarray(a, dtype=np.float32)
        w.add_tensor_info(name, a.shape, a.dtype, a.nbytes, gguf.GGMLQuantizationType.F32)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    for t in r.tensors:
        w.write_tensor_data(t.data)
    for a in new.values():
        w.write_tensor_data(np.ascontiguousarray(a, dtype=np.float32))
    w.close()
    print(f"wrote {out}: {len(r.tensors)} tensors copied, {len(new)} added, wide_bands={n_wide}, {where}, step {ck.get('step')}")


if __name__ == "__main__":
    main(*sys.argv[1:])
