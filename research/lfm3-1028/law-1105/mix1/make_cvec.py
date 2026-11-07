#!/usr/bin/env python3
"""Write a knob as an engine control vector. input of layer L = output of layer L-1. usage: make_cvec.py <dirs.pt> <word> <L> <out.gguf>"""
import sys
import numpy as np, torch, gguf
dirs, word, L, out = sys.argv[1:5]; L = int(L)
v = torch.load(dirs)["d"][word][L].numpy().astype(np.float32)
w = gguf.GGUFWriter(out, "controlvector")
w.add_uint32("controlvector.block_count", 1)
w.add_tensor(f"direction.{L - 1}", v)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print(out, word, "layer output", L - 1, "size", float(np.linalg.norm(v)))
