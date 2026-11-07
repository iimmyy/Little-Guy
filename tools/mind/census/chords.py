#!/usr/bin/env python3
"""Chords (Immy, 2026-10-27: "what is math closely related to?" -> music/physics): the bloodstream is 2,560 numbers
wide, a 2,560-key piano. How many "chords" (principal directions of the uncentered second moment) does E4B really
play at each layer? If few, the whole model could be rotated onto them and the bloodstream made narrower: every matrix
that reads or writes it shrinks (a smaller MODEL, SliceGPT-style, training-free).

1. spectrum: at the input of every layer (and the final norm), how many directions hold 99 / 99.9 / 99.99% of the energy
2. upper bound for slicing: drift when every layer's input is projected onto its top-k directions (k = 90/80/70% wide)

usage: chords.py <out folder> <math traces jsonl>      writes chords.txt, chords_basis.pt (eigenvectors, k=2560)
"""
import json, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from e4b_mac import load

out, traces = sys.argv[1], sys.argv[2]
BENCH = os.path.join(os.path.dirname(__file__), "../../../.run/bench")
model, tok = load()
layers = model.model.layers
L, D = len(layers), model.config.hidden_size
wiki = tok(open(f"{BENCH}/wiki.test.raw").read()[:200000], add_special_tokens=False).input_ids
chat = tok(open(f"{BENCH}/chat_heldout.txt").read(), add_special_tokens=False).input_ids
rows = [json.loads(l) for l in open(traces)]


def mathwin(r):
    p = tok.apply_chat_template([{"role": "user", "content": r["q"]}], tokenize=False, add_generation_prompt=True,
                                enable_thinking=True)
    return tok(p + "<|channel>thought\n" + r["thinking"] + "<channel|>" + r["reply"], add_special_tokens=False).input_ids[:768]


train = [[tok.bos_token_id] + wiki[i * 511:(i + 1) * 511] for i in range(0, 16, 2)] + \
        [[tok.bos_token_id] + chat[i * 511:(i + 1) * 511] for i in range(0, 32, 2)] + [mathwin(r) for r in rows[::14]]
test = [([tok.bos_token_id] + wiki[i * 511:(i + 1) * 511], "wiki") for i in range(16, 19)] + \
       [([tok.bos_token_id] + chat[i * 511:(i + 1) * 511], "chat") for i in range(32, 37)] + \
       [(mathwin(r), "math") for r in rows[3::42]]

SITES = list(layers) + [model.model.norm]                       # input of every layer + the final norm
mom = [torch.zeros(D, D, dtype=torch.float64) for _ in SITES]
st = {"mode": "stats", "P": None}
nwords = [0]


def pre(j):
    def h(mod, args, kwargs):
        hs = args[0] if args else kwargs["hidden_states"]
        if st["mode"] == "stats":
            x = hs[0, 1:].float()                                # skip the first word (attention sink)
            mom[j] += (x.T @ x).cpu().double()
            if j == 0:
                nwords[0] += x.shape[0]
            return None
        if st["P"] is None:
            return None
        P = st["P"][j]
        y = ((hs.float() @ P) @ P.T).to(hs.dtype)
        y[:, :1] = hs[:, :1]                                     # the first word stays untouched (it is the sink)
        if args:
            return (y,) + args[1:], kwargs
        kwargs["hidden_states"] = y
        return args, kwargs
    return h


for j, m in enumerate(SITES):
    m.register_forward_pre_hook(pre(j), with_kwargs=True)

t0 = time.time()
with torch.no_grad():
    for w in train:
        model.model(input_ids=torch.tensor([w], device="mps"))
        torch.mps.empty_cache()
print(f"moments on {nwords[0]} words ({time.time() - t0:.0f} s)", flush=True)

V, need = [], []
for j in range(len(SITES)):
    ev, vec = torch.linalg.eigh(mom[j])
    ev, vec = ev.flip(0).clamp_min(0), vec.flip(1)
    c = (ev.cumsum(0) / ev.sum()).numpy()
    need.append([int(np.searchsorted(c, q)) + 1 for q in (0.99, 0.999, 0.9999)] +
                [float(ev[0] / ev.sum())])
    V.append(vec.float())
    mom[j] = None
torch.save(V, os.path.join(out, "chords_basis.pt"))

st["mode"] = "run"


def logp(w):
    return model(input_ids=torch.tensor([w], device="mps")).logits[0, 1:].float().log_softmax(-1)


KS = (2304, 2048, 1792)
kl = {(k, t): [] for k in KS for t in ("wiki", "chat", "math")}
with torch.no_grad():
    for n, (w, t) in enumerate(test):
        st["P"] = None
        b = logp(w)
        for k in KS:
            st["P"] = [v[:, :k].to("mps") for v in V]
            lp = logp(w)
            kl[(k, t)].append((b.exp() * (b - lp)).sum(-1).mean().item())
            del lp
            st["P"] = None
            torch.mps.empty_cache()
        del b
        print(f"  eval {n + 1}/{len(test)}", flush=True)

lines = ["Chords: how many directions of the 2,560-wide bloodstream E4B really uses (uncentered energy, first word "
         f"skipped; {nwords[0]} words of wiki/chat/math)", "",
         f"{'site':>10} {'99%':>6} {'99.9%':>6} {'99.99%':>7} {'top chord share':>16}"]
for j, (a, b, c, top) in enumerate(need):
    lines.append(f"{('layer ' + str(j)) if j < L else 'final':>10} {a:>6} {b:>6} {c:>7} {top:>15.1%}")
lines += ["", "upper bound for a narrower bloodstream: every layer's input projected onto its top-k chords "
          "(drift from the untouched model)", f"{'width':>12} {'KL wiki':>8} {'KL chat':>8} {'KL math':>8}"]
for k in KS:
    lines.append(f"{k:>5} ({k / D:.0%}) " + " ".join(f"{np.mean(kl[(k, t)]):>8.4f}" for t in ("wiki", "chat", "math")))
lines += ["", "for scale: Google's own 4-bit release drifts 0.037 wiki / 0.027 chat from the 16-bit"]
open(os.path.join(out, "chords.txt"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
print("CHORDS_DONE")
