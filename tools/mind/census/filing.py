#!/usr/bin/env python3
"""By-day filing (Immy, 2026-10-27): store each 4-bit weight matrix by INPUT instead of by output row, so a zeroed
input (or a skipped neuron, for down) means one whole drawer is skipped. The pills (4-bit codes) and labels (one
scale per 32 pills) are copied untouched, so the math is the Q4_0 file's math and he stays on Google's grid.
Immy's addition, "separate the narc and regular Rx": inputs that are almost always needed live in a safe that is read
every time in the normal row order; only the rest go in drawers.

1. exactness: real Q4_0 tensors from e4b-grid4, real inputs; dense dequant vs drawer-by-drawer sum
2. bytes per word on real text for the mild / strong / extreme settings (tricks.py), per matrix:
     today    the Q4_0 file as it is: zeroed inputs save nothing; skipped neurons save only their up rows
     by-day   drawers; a label is read if any of its 32 pills is needed
     + safe   inputs kept >= SAFE_AT of the time on the fit half go in the safe (labels copied there); judged on
              the other half
     ideal    what the "on paper" numbers assumed (every skipped pill and its share of label free)
   In the routed layers each up matrix takes whichever layout reads less on average (kept rows vs input drawers).

usage: filing.py <out folder> <math traces jsonl>      writes filing.txt
"""
import json, os, sys

import gguf
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from e4b_mac import load, weights
from tricks import Tricks

out, traces = sys.argv[1], sys.argv[2]
GRID = os.environ.get("E4B_GRID") or weights("fresh/grid/e4b-grid4.gguf")
BENCH = os.path.join(os.path.dirname(__file__), "../../../.run/bench")
SAFE_AT = 0.95
SETTINGS = {"mild": (range(7), 0.2, 0.15), "strong": (range(13), 0.2, 0.25), "extreme": (range(13), 0.1, 0.4)}

rd = gguf.GGUFReader(GRID)
TENS = {t.name: t for t in rd.tensors}


def q4_0(name):
    """pills (R x K, -8..7) and labels (R x K/32) of one Q4_0 matrix with K inputs and R outputs"""
    t = TENS[name]
    K, R = int(t.shape[0]), int(t.shape[1])
    raw = np.asarray(t.data).reshape(-1, 18)
    d = raw[:, :2].copy().view(np.float16)[:, 0].astype(np.float64)
    qs = raw[:, 2:]
    q = np.concatenate([qs & 15, qs >> 4], 1).astype(np.int8) - 8
    return q.reshape(R, K), d.reshape(R, K // 32)


# bytes the runner reads per word outside the MLPs: attention, norms, the output head (the whole dictionary), the
# per-layer projection; the per-layer table and the input dictionary cost one row each
rest = sum(int(t.n_bytes) for n, t in TENS.items()
           if n not in ("per_layer_token_embd.weight", "token_embd.weight")
           and not n.endswith(("ffn_gate.weight", "ffn_up.weight", "ffn_down.weight")))
rest += int(TENS["token_embd.weight"].n_bytes)
mlp_full = sum(int(TENS[f"blk.{i}.ffn_{m}.weight"].n_bytes) for i in range(42) for m in ("gate", "up", "down"))

model, tok = load()
layers = model.model.layers
L, D, N = len(layers), model.config.hidden_size, layers[0].mlp.intermediate_size
T = Tricks(model)
wiki = tok(open(f"{BENCH}/wiki.test.raw").read()[:200000], add_special_tokens=False).input_ids
chat = tok(open(f"{BENCH}/chat_heldout.txt").read(), add_special_tokens=False).input_ids
rows = [json.loads(l) for l in open(traces)]


def mathwin(r):
    p = tok.apply_chat_template([{"role": "user", "content": r["q"]}], tokenize=False, add_generation_prompt=True,
                                enable_thinking=True)
    return tok(p + "<|channel>thought\n" + r["thinking"] + "<channel|>" + r["reply"], add_special_tokens=False).input_ids[:600]


fit = [[tok.bos_token_id] + wiki[20 * 511:21 * 511], [tok.bos_token_id] + chat[40 * 511:41 * 511], mathwin(rows[5])]
ev = [[tok.bos_token_id] + wiki[21 * 511:22 * 511], [tok.bos_token_id] + chat[41 * 511:42 * 511], mathwin(rows[155])]

# capture what each MLP really receives after the tricks (these hooks run after Tricks' own)
cap = {}
for i, l in enumerate(layers):
    l.mlp.gate_proj.register_forward_pre_hook(lambda m, a, i=i: cap.__setitem__(("x", i), a[0][0, 1:].float().cpu().numpy()))
    l.mlp.down_proj.register_forward_pre_hook(lambda m, a, i=i: cap.__setitem__(("p", i), a[0][0, 1:].float().cpu().numpy()))


def runs(windows):
    for w in windows:
        cap.clear()
        with torch.no_grad():
            model.model(input_ids=torch.tensor([w], device="mps"))
        torch.mps.empty_cache()
        yield cap


lines = []

# 1. exactness ------------------------------------------------------------------------------------------------------
T.set(route=range(13), keep=0.2, zero=0.25)
c = next(runs([ev[1][:200]]))
for name, key in (("blk.3.ffn_gate.weight", "x"), ("blk.3.ffn_down.weight", "p")):
    q, d = q4_0(name)
    v = c[(key, 3)][-1].astype(np.float64)                          # one real word, after the tricks
    dense = (q * np.repeat(d, 32, 1)) @ v
    drawers = q.T.copy()                                            # by-day filing: drawer j = every row's pill j
    js = np.nonzero(v)[0]
    byday = ((drawers[js] * v[js, None]) * d[:, js // 32].T).sum(0)
    lines.append(f"exactness {name}: {len(js)}/{len(v)} drawers opened, max |dense - by-day| = "
                 f"{np.abs(dense - byday).max():.2e} (output size {np.abs(dense).max():.2f}; float64, only the adding "
                 f"order differs)")
    print(lines[-1], flush=True)


# 2. bytes per word --------------------------------------------------------------------------------------------------
def blocks(m):
    return m.reshape(*m.shape[:-1], -1, 32).any(-1).sum(-1)


def matrix_bytes(R, K, m, S):
    """per word (arrays over words): today, by-day, by-day+safe, ideal, drawer openings (by-day, +safe)"""
    full = R * K * 0.5 + R * K // 32 * 2
    byday = R * m.sum(-1) * 0.5 + R * blocks(m) * 2
    reg = m & ~S
    safe = R * S.sum() * 0.5 + R * blocks(S[None])[0] * 2 + R * reg.sum(-1) * 0.5 + R * blocks(reg) * 2
    return [np.full(len(m), full), byday, safe, full * m.mean(-1), m.sum(-1), 1 + reg.sum(-1)]


def row_bytes(R, K, kept, mx):
    """up in a routed layer filed as today (by row): only the kept neurons' rows are read; zeroed inputs save nothing"""
    full = R * K * 0.5 + R * K // 32 * 2
    rows = full * kept / R
    return [rows, rows, rows, full * kept / R * mx.mean(-1), 0 * kept, 0 * kept]


for sname, (route, keep, zero) in SETTINGS.items():
    T.set(route=route, keep=keep, zero=zero)
    cnt = {}
    nw = 0
    for c in runs(fit):
        for i in range(L):
            for key in ("x", "p"):
                cnt[(key, i)] = cnt.get((key, i), 0) + (c[(key, i)] != 0).sum(0)
        nw += len(c[("x", 0)])
    S = {k: v / nw >= SAFE_AT for k, v in cnt.items()}
    tot = {}                                                        # (layer, matrix) -> summed per-word numbers
    nw = 0
    for c in runs(ev):
        for i in range(L):
            mx, mp = c[("x", i)] != 0, c[("p", i)] != 0
            per = {"gate": matrix_bytes(N, D, mx, S[("x", i)]), "up": matrix_bytes(N, D, mx, S[("x", i)]),
                   "down": matrix_bytes(D, N, mp, S[("p", i)])}
            if i in route:
                per["up rows"] = row_bytes(N, D, mp.sum(-1), mx)
            for m, v in per.items():
                acc = tot.setdefault((i, m), [0.0] * 6)
                for j in range(6):
                    acc[j] += float(np.sum(v[j]))
        nw += len(c[("x", 0)])
    for i in route:                                                 # one layout per routed up matrix
        dr, rw = tot[(i, "up")], tot.pop((i, "up rows"))
        tot[(i, "up")] = [rw[0], *(min(a, b) for a, b in zip(dr[1:3], rw[1:3])), rw[3],
                          *(a if dr[1] < rw[1] else b for a, b in zip(dr[4:], rw[4:]))]
    avg = {k: [a / nw for a in v] for k, v in tot.items()}
    mlp = [sum(v[j] for v in avg.values()) for j in range(4)]
    words = rest + mlp_full
    lines += ["", f"== {sname}: route layers {min(route)}-{max(route)} keep {keep:.0%}, zero {zero:.0%} of inputs "
              f"({nw} eval words; safe = inputs kept >= {SAFE_AT:.0%} on {sum(len(w) - 1 for w in fit)} fit words)",
              f"  {'filing':10} {'MLP MB/word':>12} {'MLP read':>9} {'all MB/word':>12} {'speed vs grid4 today':>21}"]
    for j, nm in enumerate(("today", "by-day", "+ safe", "ideal")):
        lines.append(f"  {nm:10} {mlp[j] / 1e6:>12.0f} {mlp[j] / mlp_full:>9.1%} {(rest + mlp[j]) / 1e6:>12.0f} "
                     f"{words / (rest + mlp[j]):>20.2f}x")
    sx = np.mean([S[("x", i)].mean() for i in range(L)])
    sp = np.mean([S[("p", i)].mean() for i in route])
    o1 = sum(v[4] for v in avg.values())
    o2 = sum(v[5] for v in avg.values())
    lines.append(f"  safe holds {sx:.1%} of the bloodstream inputs and {sp:.1%} of the neurons in routed layers; "
                 f"drawer openings per word {o1:,.0f} -> {o2:,.0f}")
    print("\n".join(lines[-8:]), flush=True)

lines += ["", f"grid4 today, no tricks: MLP {mlp_full / 1e6:.0f} MB + rest {rest / 1e6:.0f} MB = "
          f"{(mlp_full + rest) / 1e6:.0f} MB per word"]
open(os.path.join(out, "filing.txt"), "w").write("\n".join(lines) + "\n")
print("FILING_DONE")
