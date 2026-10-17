"""Rung 2 check: how compressible are a GGUF's Q4_0 codes without changing a single bit of the model?

Q4_0 stores each block of 32 weights as one f16 scale + 32 four-bit codes (18 bytes). Counts every code per matrix
and reports its Shannon entropy (the floor for lossless coding of codes on their own), plus the same with a separate
code table per matrix row-group, and how many bytes per word that would save.

usage: python code_entropy.py <model.gguf>
"""
import sys
import numpy as np
from gguf import GGUFReader, GGMLQuantizationType


def entropy(counts):
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


r = GGUFReader(sys.argv[1])
tot_bytes = tot_codes = 0
tot_bits = 0.0
by_type = {}
for t in r.tensors:
    if t.tensor_type != GGMLQuantizationType.Q4_0:
        continue
    raw = np.asarray(t.data).reshape(-1, 18)
    q = raw[:, 2:]
    codes = np.concatenate([q & 0x0F, q >> 4], axis=1).ravel()
    c = np.bincount(codes, minlength=16).astype(np.float64)
    h = entropy(c)
    kind = t.name.split(".")[-2] if t.name.startswith("blk.") else t.name
    b = by_type.setdefault(kind, [0, 0, 0.0, np.zeros(16)])
    b[0] += raw.nbytes; b[1] += codes.size; b[2] += h * codes.size; b[3] += c
    tot_bytes += raw.nbytes; tot_codes += codes.size; tot_bits += h * codes.size
print(f"{'matrix':14} {'bytes':>9} | {'entropy bits/code':>17} | {'code share 0..15 (%)':s}")
for k, (nb, nc, bits, c) in sorted(by_type.items()):
    print(f"{k:14} {nb / 1e6:7.0f}MB | {bits / nc:17.3f} | " + " ".join(f"{100 * x / c.sum():4.1f}" for x in c))
h = tot_bits / tot_codes
scale_bytes = tot_codes / 32 * 2
saved = tot_codes * (4 - h) / 8
print(f"\nall Q4_0: {tot_bytes / 1e9:.3f} GB, {tot_codes / 1e9:.2f} G codes, mean entropy {h:.3f} bits/code "
      f"(4 = no gain). Lossless floor saves {saved / 1e6:.0f} MB = {100 * saved / tot_bytes:.1f}% of the Q4_0 bytes "
      f"(scales {scale_bytes / 1e6:.0f} MB kept as is).")
