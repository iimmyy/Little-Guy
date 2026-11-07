#!/usr/bin/env python3
"""Neuron census of E4B (Immy, 2026-10-26: "simplify the neurons themselves ... one complex and one simple is still
better than two complex"). Measure only, nothing is changed.

Every MLP neuron i in every layer is two equations multiplied, then sent out:
    out += down_i * ( gelu(gate_i . x)  x  (up_i . x) )        a = gelu(gate . x) (the curve),  u = up . x (the line)
For each of the 42 x 10,240 neurons, over real text (WikiText, held-out chat, E4B's own math thinking):
  share     its output energy E[(a u)^2] |down_i|^2 as a fraction of its layer's total (a dead neuron ~ 0)
  freeze    how much of its output is lost if the curve a is frozen to its average:  E[((a - mean a) u)^2] / E[(a u)^2]
            (small = the gate is basically a constant, the neuron is ONE straight line and can merge with others)
  straight  same, if the curve is replaced by its best straight line a ~ alpha g + beta (the gelu barely bends)
  off       fraction of words where its output is under 5% of its typical size (a switch that is mostly off)
Two passes over the same words: pass 1 the averages and line fits, pass 2 the errors and the off counts.

usage: census.py <out folder> [math traces jsonl]        writes census.npz and census.txt
"""
import json, os, sys, time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from e4b_mac import load

out = sys.argv[1]
traces = sys.argv[2] if len(sys.argv) > 2 else None
os.makedirs(out, exist_ok=True)
BENCH = os.path.join(os.path.dirname(__file__), "../../../.run/bench")
model, tok = load()
layers = model.model.layers
L, N = len(layers), layers[0].mlp.intermediate_size

# words ----------------------------------------------------------------------------------------------------------
wins, src = [], []
wiki = tok(open(f"{BENCH}/wiki.test.raw").read()[:200000], add_special_tokens=False).input_ids
for i in range(16):
    wins.append([tok.bos_token_id] + wiki[i * 511:(i + 1) * 511]); src.append("wiki")
chat = tok(open(f"{BENCH}/chat_heldout.txt").read(), add_special_tokens=False).input_ids
for i in range(32):
    wins.append([tok.bos_token_id] + chat[i * 511:(i + 1) * 511]); src.append("chat")
if traces:
    rows = [json.loads(l) for l in open(traces)][::7]
    for r in rows:
        p = tok.apply_chat_template([{"role": "user", "content": r["q"]}], tokenize=False, add_generation_prompt=True,
                                    enable_thinking=True)
        text = p + "<|channel>thought\n" + r["thinking"] + "<channel|>" + r["reply"]
        wins.append(tok(text, add_special_tokens=False).input_ids[:768]); src.append("math")
ntok = sum(len(w) - 1 for w in wins)
print(f"{len(wins)} windows, {ntok} words ({', '.join(f'{s} {src.count(s)}' for s in dict.fromkeys(src))})", flush=True)

# hooks ----------------------------------------------------------------------------------------------------------
Z = lambda: torch.zeros(L, N, device="mps", dtype=torch.float64 if False else torch.float32)
S = {k: Z() for k in ("n", "a", "a2", "g", "g2", "ag", "p2", "u2", "dev_f", "dev_s", "off")}
state = {"pass": 1, "g": [None] * L}


def gate_hook(L_):
    def h(mod, args, outp):
        state["g"][L_] = outp
    return h


def up_hook(L_):
    def h(mod, args, outp):
        g = state["g"][L_][0, 1:].float()                  # skip the first word (attention sink)
        u = outp[0, 1:].float()
        a = F.gelu(g, approximate="tanh")
        p = a * u
        if state["pass"] == 1:
            S["n"][L_] += g.shape[0]
            S["a"][L_] += a.sum(0); S["a2"][L_] += (a * a).sum(0)
            S["g"][L_] += g.sum(0); S["g2"][L_] += (g * g).sum(0); S["ag"][L_] += (a * g).sum(0)
            S["p2"][L_] += (p * p).sum(0); S["u2"][L_] += (u * u).sum(0)
        else:
            S["dev_f"][L_] += (((a - FIT["abar"][L_]) * u) ** 2).sum(0)
            S["dev_s"][L_] += (((a - FIT["alpha"][L_] * g - FIT["beta"][L_]) * u) ** 2).sum(0)
            S["off"][L_] += (p.abs() < 0.05 * FIT["prms"][L_]).float().sum(0)
    return h


for i, layer in enumerate(layers):
    layer.mlp.gate_proj.register_forward_hook(gate_hook(i))
    layer.mlp.up_proj.register_forward_hook(up_hook(i))


def run(tag):
    t = time.time()
    with torch.no_grad():
        for k, w in enumerate(wins):
            model.model(input_ids=torch.tensor([w], device="mps"))
            if k % 20 == 19:
                print(f"  {tag} {k + 1}/{len(wins)} ({time.time() - t:.0f} s)", flush=True)


run("pass 1")
n = S["n"]
FIT = {"abar": S["a"] / n, "prms": (S["p2"] / n).sqrt()}
vg = S["g2"] / n - (S["g"] / n) ** 2
FIT["alpha"] = (S["ag"] / n - FIT["abar"] * S["g"] / n) / vg.clamp_min(1e-12)
FIT["beta"] = FIT["abar"] - FIT["alpha"] * S["g"] / n
state["pass"] = 2
run("pass 2")

# results --------------------------------------------------------------------------------------------------------
dn = torch.stack([l.mlp.down_proj.weight.float().norm(dim=0) for l in layers])          # |down_i|, L x N
energy = S["p2"] / n * dn ** 2
R = {"share": energy / energy.sum(1, keepdim=True), "freeze": S["dev_f"] / S["p2"].clamp_min(1e-30),
     "straight": S["dev_s"] / S["p2"].clamp_min(1e-30), "off": S["off"] / n,
     "abar": FIT["abar"], "alpha": FIT["alpha"], "beta": FIT["beta"], "prms": FIT["prms"], "down_norm": dn}
R = {k: v.cpu().numpy() for k, v in R.items()}
np.savez_compressed(os.path.join(out, "census.npz"), **R, words=ntok)

share, freeze, straight, off = R["share"], R["freeze"], R["straight"], R["off"]
dead = share < 1e-6
const = ~dead & (freeze < 0.05)
line = ~dead & ~const & (straight < 0.05)
switch = ~dead & ~const & ~line & (off > 0.9)
cx = ~(dead | const | line | switch)
lines = [f"E4B neuron census, {ntok} words, {L} layers x {N} neurons = {L * N:,}", "",
         "kinds (in this order, first match wins):",
         "  dead      output energy < 1e-6 of its layer (the layer average is ~1e-4)",
         "  constant  freezing the curve to its average keeps >95% of the output exactly (ONE straight line)",
         "  straight  the curve's best straight line keeps >95% (gelu barely bends: line x line)",
         "  switch    under 5% of its typical output on >90% of words (mostly off)",
         "  complex   everything else", "",
         f"{'layer':>5} {'dead':>7} {'const':>7} {'straight':>9} {'switch':>7} {'complex':>8}   top-1% neurons' share of the layer"]
for i in range(L):
    top = np.sort(share[i])[::-1][:N // 100].sum()
    lines.append(f"{i:>5} {dead[i].sum():>7} {const[i].sum():>7} {line[i].sum():>9} {switch[i].sum():>7} {cx[i].sum():>8}   {top:.2f}")
tot = lambda m: f"{m.sum():,} ({m.mean() * 100:.1f}%)"
lines += ["", f"ALL  dead {tot(dead)}, constant {tot(const)}, straight {tot(line)}, switch {tot(switch)}, complex {tot(cx)}"]
for q in (0.5, 0.9, 0.99):
    lines.append(f"  median-ish: {int(q * 100)}th percentile of freeze {np.quantile(freeze, q):.3f}, straight "
                 f"{np.quantile(straight, q):.3f}, off {np.quantile(off, q):.3f}")
open(os.path.join(out, "census.txt"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
print("CENSUS_DONE")
