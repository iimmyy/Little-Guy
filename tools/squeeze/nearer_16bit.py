#!/usr/bin/env python3
"""Part by part: which of two 4-bit files is nearer the 16-bit original?

For every weight tensor in both files and the original, read an even sample of rows, unpack both 4-bit versions and
measure the error against the original (relative RMS = RMS(error) / RMS(original)). Prints the tensors with the biggest
gaps, a per-kind summary and a whole-brain total (squared error scaled by each tensor's size).
usage: nearer_16bit.py <a.gguf> <b.gguf> <original bf16.gguf> <name a> <name b> [rows per tensor, default 256]
"""
import sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".run/lcpp-src/gguf-py"))
import numpy as np
import gguf
from gguf.quants import dequantize

pa, pb, pr, na, nb = sys.argv[1:6]
nrows = int(sys.argv[6]) if len(sys.argv) > 6 else 256
T = gguf.GGMLQuantizationType
A = {t.name: t for t in gguf.GGUFReader(pa).tensors}
B = {t.name: t for t in gguf.GGUFReader(pb).tensors}
R = {t.name: t for t in gguf.GGUFReader(pr).tensors}


def rows(t, idx):
    """the sampled rows of tensor t as float32 (n, cols)"""
    d = np.asarray(t.data)
    if t.tensor_type == T.BF16:                         # the reader hands BF16 over as raw bytes
        d = d.view(np.uint16)
    d = d.reshape(-1, d.shape[-1])[idx]
    if t.tensor_type == T.BF16:
        return (d.astype(np.uint32) << 16).view(np.float32)
    if t.tensor_type in (T.F32, T.F16):
        return d.astype(np.float32)
    return dequantize(d, t.tensor_type).astype(np.float32)


out, kinds = [], defaultdict(lambda: [0.0, 0.0, 0])
tot = [0.0, 0.0]
for name, r in R.items():
    if name not in A or name not in B or len(r.shape) < 2:
        continue
    a, b = A[name], B[name]
    if a.tensor_type == T.F32 and b.tensor_type == T.F32:
        continue
    n = int(np.prod([int(x) for x in r.shape[1:]]))       # rows (ggml shape is [cols, rows, ...])
    idx = np.linspace(0, n - 1, min(n, nrows)).astype(int)
    ref = rows(r, idx)
    ea = ((rows(a, idx) - ref) ** 2).mean()
    eb = ((rows(b, idx) - ref) ** 2).mean()
    p = float((ref ** 2).mean()) or 1e-30
    size = int(np.prod([int(x) for x in r.shape]))
    tot[0] += ea * size; tot[1] += eb * size
    kind = name.split(".")[-2] if name.startswith("blk.") else name.rsplit(".", 1)[0]
    k = kinds[kind]; k[0] += ea * size; k[1] += eb * size; k[2] += 1
    out.append((name, a.tensor_type.name, b.tensor_type.name, np.sqrt(ea / p), np.sqrt(eb / p)))

closer = sum(1 for o in out if o[3] < o[4])
print(f"{len(out)} tensors compared ({nrows} rows each); {na} nearer the original in {closer}, {nb} in {len(out) - closer - sum(1 for o in out if o[3] == o[4])}, tied {sum(1 for o in out if o[3] == o[4])}")
print(f"whole brain, squared error vs original: {na} {tot[0]:.4e}  {nb} {tot[1]:.4e}  ratio {na}/{nb} {tot[0] / max(tot[1], 1e-30):.2f}")
print(f"by kind (count, squared error {na} / {nb})")
for k, (ea, eb, c) in sorted(kinds.items(), key=lambda kv: -max(kv[1][0], kv[1][1])):
    print(f"  {k:24} {c:3d}  {ea:.3e} / {eb:.3e}   ratio {ea / max(eb, 1e-30):7.2f}")
print(f"biggest gaps (relative RMS error vs original; format {na} / {nb})")
for o in sorted(out, key=lambda o: -abs(np.log((o[3] + 1e-12) / (o[4] + 1e-12))))[:12]:
    print(f"  {o[0]:36} {o[1]:5} / {o[2]:5}  {o[3]:.4%} / {o[4]:.4%}")
