#!/usr/bin/env python3
"""Detour (Immy, 2026-10-18): can "repackaging" the stencils (a rotation) let them survive fewer bits?

Read-only, room by room, on real text. For each room's three stencil sets (gate, up: input 2,560; down: input 10,240)
we round the weights to b bits in blocks of 32 (symmetric, best of 32 scales per block by squared error), with and
without a rotation of the input side first. The rotation is a block Hadamard with random signs (512 wide for gate/up,
2048 wide for down): it blends each block of inputs so that no single weight sticks out, then the room's input gets the
same blend, so the exact math is unchanged before rounding. We report the error of the room's output against the exact
room, like where_to_look.py, next to the error today's 4-bit file (e4b-grid4) makes in the same room.
usage: rotate_look.py <hf dir> <text> <grid4 gguf> <out dir> [tokens, default 512]
"""
import json, re, sys, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".run/lcpp-src/gguf-py"))
import gguf
from gguf.quants import dequantize

hf, text, grid, out = sys.argv[1], sys.argv[2], sys.argv[3], Path(sys.argv[4])
ntok = int(sys.argv[5]) if len(sys.argv) > 5 else 512
out.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(6)
torch.manual_seed(0)
t0 = time.time()
tok = AutoTokenizer.from_pretrained(hf)
model = AutoModelForCausalLM.from_pretrained(hf, dtype=torch.bfloat16).eval()
mlps = sorted(((int(re.search(r"layers\.(\d+)\.mlp$", n).group(1)), m) for n, m in model.named_modules()
               if "language_model" in n and re.search(r"layers\.\d+\.mlp$", n) and hasattr(m, "gate_proj")),
              key=lambda x: x[0])
assert len(mlps) == 42, f"expected the 42 text rooms, found {len(mlps)}"
X = {}
for i, m in mlps:
    m.register_forward_pre_hook(lambda mod, args, i=i: X.__setitem__(i, args[0][0].detach().float().clone()))
ids = tok(open(text).read(), add_special_tokens=False).input_ids[: ntok - 1]
with torch.no_grad():
    model(input_ids=torch.tensor([[tok.bos_token_id] + ids]))
print(f"captured {len(ids) + 1} words x {len(X)} rooms in {time.time() - t0:.0f}s", flush=True)
G = {t.name: t for t in gguf.GGUFReader(grid).tensors}


def q4file(name, shape):
    t = G[name]
    return torch.from_numpy(dequantize(np.asarray(t.data), t.tensor_type).astype(np.float32).reshape(shape))


def hadamard(n):
    h = torch.ones(1, 1)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / h.shape[0] ** 0.5


def rotation(dim, width):
    """block-diagonal Hadamard with random signs: orthogonal, so rotating weights and inputs alike changes nothing"""
    H = hadamard(width)
    s = (torch.randint(0, 2, (dim,)) * 2 - 1).float()
    return lambda W: (W.view(W.shape[0], dim // width, width) @ H).view(W.shape[0], dim) * s, \
           lambda x: (x.view(x.shape[0], dim // width, width) @ H).view(x.shape[0], dim) * s


def rtn(W, bits):
    """symmetric rounding in blocks of 32 along the input side, best of 32 scales per block (squared error)"""
    lo, hi = -2 ** (bits - 1), 2 ** (bits - 1) - 1
    B = W.reshape(-1, 32)
    m = B.abs().amax(1, keepdim=True).clamp_min(1e-12)
    best, berr = None, None
    for k in torch.linspace(max(hi - 1.0, 0.75), -lo + 0.5, 32):     # (first run started at 0 for 2-bit: d = inf, nan)
        d = m / k
        q = (B / d).round().clamp(lo, hi) * d
        e = ((q - B) ** 2).sum(1, keepdim=True)
        if best is None:
            best, berr = q, e
        else:
            better = e < berr
            best = torch.where(better, q, best); berr = torch.minimum(e, berr)
    return best.view_as(W)


act = lambda z: torch.nn.functional.gelu(z, approximate="tanh")
rows = []
for i, m in mlps:
    Wg, Wu, Wd = (p.weight.detach().float() for p in (m.gate_proj, m.up_proj, m.down_proj))
    x = X[i]
    y = (act(x @ Wg.T) * (x @ Wu.T)) @ Wd.T
    ny = y.norm(dim=1)
    rel = lambda yy: float(((y - yy).norm(dim=1) / ny).median())
    r = {"room": i}
    r["bar_4bit_grid"] = rel((act(x @ q4file(f"blk.{i}.ffn_gate.weight", Wg.shape).T) * (x @ q4file(f"blk.{i}.ffn_up.weight", Wu.shape).T))
                             @ q4file(f"blk.{i}.ffn_down.weight", Wd.shape).T)
    rin_w, rin_x = rotation(2560, 512)
    rmid_w, rmid_x = rotation(10240, 2048)
    xr = rin_x(x)
    Wg_r, Wu_r, Wd_r = rin_w(Wg), rin_w(Wu), rmid_w(Wd)
    chk = rel(rmid_x(act(xr @ Wg_r.T) * (xr @ Wu_r.T)) @ Wd_r.T)       # rotated, unrounded: must equal the exact room
    assert chk < 1e-4, f"rotation changed the exact math in room {i}: {chk}"
    for bits in (4, 3, 2):
        r[f"plain_{bits}"] = rel((act(x @ rtn(Wg, bits).T) * (x @ rtn(Wu, bits).T)) @ rtn(Wd, bits).T)
        a = act(xr @ rtn(Wg_r, bits).T) * (xr @ rtn(Wu_r, bits).T)
        r[f"rot_{bits}"] = rel(rmid_x(a) @ rtn(Wd_r, bits).T)
    rows.append(r)
    print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)

json.dump(rows, open(out / "rotate_look.json", "w"), indent=1)
med = lambda k: float(np.median([r[k] for r in rows]))
print(f"\nSUMMARY (median over {len(rows)} rooms; error of the room's output, relative)")
print(f"  bar: today's 4-bit grid file {med('bar_4bit_grid'):.4f}")
for bits in (4, 3, 2):
    print(f"  {bits}-bit   plain {med(f'plain_{bits}'):.4f}   rotated {med(f'rot_{bits}'):.4f}   "
          f"(rotation cuts error {med(f'plain_{bits}') / med(f'rot_{bits}'):.2f}x)")
print(f"done in {time.time() - t0:.0f}s")
