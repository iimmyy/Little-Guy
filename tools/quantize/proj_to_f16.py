"""Rewrite Gemma's per_layer_model_proj from BF16 to F16 inside a GGUF, in place.
llama-quantize never touches this tensor, so it keeps the converter's BF16, and the board's CPU has no BF16
instructions: reading (prefill) ran ~25% slower than Google's F16 file. Same 2 bytes per number, so no offsets move.
Usage: python proj_to_f16.py <gguf-py dir> <model.gguf>
"""
import sys
import numpy as np
sys.path.insert(0, sys.argv[1])
from gguf import GGUFReader, GGMLQuantizationType

r = GGUFReader(sys.argv[2], "r+")
t = next(x for x in r.tensors if x.name == "per_layer_model_proj.weight")
assert t.tensor_type == GGMLQuantizationType.BF16, t.tensor_type
f32 = (np.ascontiguousarray(t.data).view(np.uint16).astype(np.uint32) << 16).view(np.float32)
t.data[...] = f32.astype(np.float16).view(np.uint8).reshape(t.data.shape)
t.field.parts[4][0] = GGMLQuantizationType.F16
r.data.flush()
print("per_layer_model_proj.weight: BF16 -> F16")
