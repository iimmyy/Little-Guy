"""Q4_0 on the model's own 4-bit grid: for every block of 32, try each grid step a Q4_0 block can hold (f16 step =
+-max|x|/k for k = 1..8 and its 2 nearest f16 neighbours each way) and keep the one closest to the 16-bit weights.
QAT models (Gemma 4 QAT) were trained onto such a grid, so this recovers it instead of estimating a scale; on E4B
~79% of blocks come back bit for bit (in bf16) and the rest within ~0.1% RMS, for word tables as for block matrices.

Writes a small GGUF with only the requested tensors (Q4_0); swap them into a full file with swap_blocks.py --any-type.

usage: grid_q4_0.py <bf16.gguf> <out.gguf> <tensor name> [<tensor name> ...]
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".run/lcpp-src/gguf-py"))
import gguf


def grid_q4_0(x):
    """x: float64 [n_blocks, 32] -> (d f16 [n_blocks], q int [n_blocks, 32] in -8..7, squared error [n_blocks])"""
    amax = x.abs().amax(1, keepdim=True)
    best_err = torch.full((len(x),), float("inf"), dtype=torch.float64)
    best_d = torch.zeros(len(x), dtype=torch.float16)
    best_q = torch.zeros(x.shape, dtype=torch.int8)
    for k in range(1, 9):
        base = (amax / k).to(torch.float16).view(torch.int16)
        for off in (-2, -1, 0, 1, 2):
            for sign in (1.0, -1.0):
                d16 = (base + off).view(torch.float16) * sign
                d = d16.double()
                safe = torch.where(d != 0, d, torch.ones_like(d))
                q = torch.round(x / safe).clamp(-8, 7)
                err = ((d * q - x) ** 2).sum(1)
                better = err < best_err
                best_err = torch.where(better, err, best_err)
                best_d = torch.where(better, d16[:, 0], best_d)
                best_q = torch.where(better[:, None], q.to(torch.int8), best_q)
    return best_d, best_q, best_err


def pack(d, q):
    """ggml block_q4_0: f16 d, then 16 bytes; byte j = (q[j] + 8) | (q[j + 16] + 8) << 4"""
    u = (q.to(torch.int16) + 8).to(torch.uint8)
    qs = u[:, :16] | (u[:, 16:] << 4)
    return np.concatenate([d.numpy().view(np.uint8).reshape(-1, 2), qs.numpy()], axis=1)


src, out, names = sys.argv[1], sys.argv[2], sys.argv[3:]
r = gguf.GGUFReader(src)
by = {t.name: t for t in r.tensors}
w = gguf.GGUFWriter(out, r.fields[gguf.Keys.General.ARCHITECTURE].contents(), endianess=r.endianess)
packed = []
for name in names:
    t = by[name]
    assert t.tensor_type == gguf.GGMLQuantizationType.BF16, f"{name}: expected BF16, got {t.tensor_type.name}"
    raw = np.asarray(t.data).view(np.uint16)
    n_per_row = int(t.shape[0])
    rows = raw.reshape(-1, n_per_row)
    blocks, sq, energy, exact = [], 0.0, 0.0, 0
    for i in range(0, len(rows), 4096):
        x16 = torch.from_numpy(rows[i:i + 4096].astype(np.int32) << 16).view(torch.float32)
        x = x16.double().reshape(-1, 32)
        d, q, err = grid_q4_0(x)
        rec = (d.double()[:, None] * q.double()).to(torch.bfloat16).float()
        exact += int((rec == x.float()).all(1).sum())
        sq += float(err.sum()); energy += float((x ** 2).sum())
        blocks.append(pack(d, q))
    data = np.concatenate(blocks).reshape(len(rows), -1)
    n_blocks = data.shape[0] * data.shape[1] // 18
    print(f"{name}: {n_blocks} blocks, bit-exact (bf16) {100 * exact / n_blocks:.2f}%, leftover RMS {np.sqrt(sq / energy):.2e}, "
          f"{data.nbytes / 1e6:.1f} MB", flush=True)
    packed.append((name, data))
for name, data in packed:
    w.add_tensor(name, data, raw_shape=data.shape, raw_dtype=gguf.GGMLQuantizationType.Q4_0)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file()
w.close()
print("wrote", out)
