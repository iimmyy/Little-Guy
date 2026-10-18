#!/usr/bin/env python3
"""Step 0 of "tell it where to look" (Immy, 2026-10-18): read-only measurements of the 16-bit brain's own rooms.

For every room (layer) we capture the thought entering its stencil set (the FFN input) on real text, then ask:
  1. silence    : skip the quietest share f of the 10,240 stencils for each word, chosen perfectly (oracle). How big
                  is the error in the room's output? (the ceiling: no guide can do better)
  2. hologram   : a small low-rank copy ("projector") of the room's first two stencil sets, rank r. Let it pick the
                  stencils to keep, then do the exact math on those. Same error question, at r = 64 / 128 / 256.
  3. lookahead  : the projector of room L fed the PREVIOUS room's thought (decides one room early)
  4. shelves    : shelves of 8 stencils (the skip kernel's group); share of whole shelves that go dark per word,
                  in today's order vs re-sorted by how often each stencil is needed (re-sorting is lossless)
Bar: the error the current 4-bit file (e4b-grid4) already makes in the same room's output.
Error = |room output - exact room output| / |exact room output|, median over words.
usage: where_to_look.py <hf dir> <text> <grid4 gguf> <out dir> [tokens, default 1024]
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
ntok = int(sys.argv[5]) if len(sys.argv) > 5 else 1024
out.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(6)
t0 = time.time()
tok = AutoTokenizer.from_pretrained(hf)
model = AutoModelForCausalLM.from_pretrained(hf, dtype=torch.bfloat16)
model.eval()
mlps = sorted(((int(re.search(r"layers\.(\d+)\.mlp$", n).group(1)), m) for n, m in model.named_modules()
               if "language_model" in n and re.search(r"layers\.\d+\.mlp$", n) and hasattr(m, "gate_proj")), key=lambda x: x[0])
assert len(mlps) == 42, f"expected the 42 text rooms, found {len(mlps)}"
print(f"{len(mlps)} rooms found, loaded in {time.time() - t0:.0f}s", flush=True)

X = {}
for i, m in mlps:
    m.register_forward_pre_hook(lambda mod, args, i=i: X.__setitem__(i, args[0][0].float().clone()))
ids = tok(open(text).read(), add_special_tokens=False).input_ids[: ntok - 1]
ids = torch.tensor([[tok.bos_token_id] + ids])
with torch.no_grad():
    model(input_ids=ids)
print(f"captured {ids.shape[1]} words x {len(X)} rooms in {time.time() - t0:.0f}s", flush=True)

G = {t.name: t for t in gguf.GGUFReader(grid).tensors}
def q4(name, shape):
    t = G[name]
    return torch.from_numpy(dequantize(np.asarray(t.data), t.tensor_type).astype(np.float32).reshape(shape))

F = [0.5, 0.7, 0.8, 0.9, 0.95]
R = [64, 128, 256]
act = torch.nn.functional.gelu
rows = []
prev = None
for i, m in mlps:
    Wg, Wu, Wd = (p.weight.float() for p in (m.gate_proj, m.up_proj, m.down_proj))
    x = X[i]
    a = act(x @ Wg.T, approximate="tanh") * (x @ Wu.T)                  # 10,240 stencil readings per word
    y = a @ Wd.T
    ny = y.norm(dim=1)
    if i == mlps[0][0]:                                                 # my math must equal Gemma's own room
        with torch.no_grad():
            ref = m(x[None].to(torch.bfloat16))[0].float()
        chk = float(((y - ref).norm(dim=1) / ny).median())
        print(f"check: my room math vs Gemma's room 0: {chk:.4f} (bf16 noise ~0.005)", flush=True)
        if chk > 0.02:
            sys.exit("room math does not match the model; stopping")
    dn = Wd.norm(dim=0)                                                 # how loud each stencil can be in the output
    c = a.abs() * dn
    def err(keep):                                                      # keep: words x stencils, True = use it
        return float(((y - (a * keep) @ Wd.T).norm(dim=1) / ny).median())
    def top(score, f):
        k = int(round(score.shape[1] * (1 - f)))
        idx = score.topk(k, dim=1).indices
        return torch.zeros_like(score, dtype=torch.bool).scatter_(1, idx, True)
    r = {"room": i}
    yq = (act(x @ q4(f"blk.{i}.ffn_gate.weight", Wg.shape).T, approximate="tanh") * (x @ q4(f"blk.{i}.ffn_up.weight", Wu.shape).T)) @ q4(f"blk.{i}.ffn_down.weight", Wd.shape).T
    r["bar_4bit"] = float(((y - yq).norm(dim=1) / ny).median())
    for f in F:
        r[f"oracle_{f}"] = err(top(c, f))
    holos = {}
    for rk in R:
        Ug, Sg, Vg = torch.svd_lowrank(Wg, q=rk, niter=4)
        Uu, Su, Vu = torch.svd_lowrank(Wu, q=rk, niter=4)
        holos[rk] = (Ug * Sg, Vg, Uu * Su, Vu)
        def guess(xx, h=holos[rk]):
            return (act((xx @ h[1]) @ h[0].T, approximate="tanh") * ((xx @ h[3]) @ h[2].T)).abs() * dn
        g = guess(x)
        for f in (0.5, 0.8, 0.9):
            r[f"holo{rk}_{f}"] = err(top(g, f))
        if rk == 128 and prev is not None:
            gp = guess(prev)
            for f in (0.5, 0.8):
                r[f"look128_{f}"] = err(top(gp, f))
    keep = top(c, 0.8)
    need = keep.float().mean(0)                                         # how often each stencil is needed
    order = need.argsort()
    for name, kk in (("shelf_dark_orig", keep), ("shelf_dark_sorted", keep[:, order])):
        r[name] = float((~kk.view(kk.shape[0], -1, 8).any(dim=2)).float().mean())
    rows.append(r)
    prev = x
    print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    del Wg, Wu, Wd, a, y, c, holos

json.dump(rows, open(out / "where_to_look.json", "w"), indent=1)
med = lambda k: float(np.median([r[k] for r in rows if k in r]))
print(f"\nSUMMARY (median over {len(rows)} rooms; error of the room's output, relative)")
print(f"  bar: today's 4-bit file already makes {med('bar_4bit'):.4f}")
print("  skip share        " + "  ".join(f"{f:>6}" for f in F))
print("  perfect guide     " + "  ".join(f"{med(f'oracle_{f}'):6.4f}" for f in F))
for rk in R:
    print(f"  hologram r={rk:<4}  " + "  ".join(f"{med(f'holo{rk}_{f}'):6.4f}" if f'holo{rk}_{f}' in rows[0] else "     -" for f in F))
print("  lookahead r=128   " + "  ".join(f"{med(f'look128_{f}'):6.4f}" if f'look128_{f}' in rows[1] else "     -" for f in F))
print(f"  shelves of 8 fully dark at 80% skip: today's order {med('shelf_dark_orig'):.3f}, sorted {med('shelf_dark_sorted'):.3f}")
print(f"  hologram size per room (gate+up, 8-bit): r=64 {64 * 12800 * 2 / 1e6:.1f} MB, r=128 {128 * 12800 * 2 / 1e6:.1f} MB, r=256 {256 * 12800 * 2 / 1e6:.1f} MB (room ~53 MB)")
print(f"done in {time.time() - t0:.0f}s")
