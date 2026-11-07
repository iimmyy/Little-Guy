#!/usr/bin/env python3
"""Mastering the bloodstream (Immy, 2026-10-27: "what about dBTP?" -> "try it"). The bloodstream has a few giant
numbers (one is ~62x steadier than it is big: a DC offset) that set the headroom for everything else.

At the input of every layer (and the final norm), applied to the giants only (each layer's 4 biggest numbers by
average energy on sample text):
  dc        lock each giant to its average (DC offset made exactly constant)
  limit q   true-peak style soft limiter, ceiling = q x the giant's average size: c tanh(x / c)
  crush 8   the whole bloodstream rounded to 8 bits per word (scale = that word's peak); last time: destroyed
  limit + crush   does taming the peaks free enough headroom for 8 bits?
Scores like probe.py: KL from the untouched model, dNLL on the real next word (< 0 = predicts real text better).

usage: peaks.py <out folder> <math traces jsonl>      writes peaks.txt
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
    return tok(p + "<|channel>thought\n" + r["thinking"] + "<channel|>" + r["reply"], add_special_tokens=False).input_ids[:600]


fit = [[tok.bos_token_id] + wiki[i * 511:(i + 1) * 511] for i in (2, 4)] + \
      [[tok.bos_token_id] + chat[i * 511:(i + 1) * 511] for i in (2, 4, 6)] + [mathwin(r) for r in rows[10::150]]
test = [([tok.bos_token_id] + wiki[i * 511:(i + 1) * 511], "wiki") for i in (20, 21)] + \
       [([tok.bos_token_id] + chat[i * 511:(i + 1) * 511], "chat") for i in (40, 41, 42)] + \
       [(mathwin(r), "math") for r in rows[5::150]]

SITES = list(layers) + [model.model.norm]
S = len(SITES)
st = {"mode": "stats", "fn": None}
e1 = [torch.zeros(D, dtype=torch.float64) for _ in range(S)]
e2 = [torch.zeros(D, dtype=torch.float64) for _ in range(S)]
nw = [0]


def pre(j):
    def h(mod, args, kwargs):
        hs = args[0] if args else kwargs["hidden_states"]
        if st["mode"] == "stats":
            x = hs[0, 1:].float()
            e1[j] += x.sum(0).cpu().double(); e2[j] += x.pow(2).sum(0).cpu().double()
            if j == 0:
                nw[0] += x.shape[0]
            return None
        if st["fn"] is None:
            return None
        y = hs.clone()
        y[:, 1:] = st["fn"](j, hs[:, 1:].float()).to(hs.dtype)  # the first word (attention sink) stays untouched
        if args:
            return (y,) + args[1:], kwargs
        kwargs["hidden_states"] = y
        return args, kwargs
    return h


for j, m in enumerate(SITES):
    m.register_forward_pre_hook(pre(j), with_kwargs=True)


def run(w):
    lp = model(input_ids=torch.tensor([w], device="mps")).logits[0, :-1].float().log_softmax(-1)
    tgt = torch.tensor(w[1:], device="mps")
    return lp, -lp.gather(1, tgt[:, None]).squeeze(1)


with torch.no_grad():
    for w in fit:
        model.model(input_ids=torch.tensor([w], device="mps"))
        torch.mps.empty_cache()
    st["mode"] = "run"
    BASE = []
    for w, s in test:
        lp, nll = run(w)
        BASE.append((lp.to(torch.bfloat16).cpu(), nll.cpu()))
        del lp
        torch.mps.empty_cache()

GI, MEAN, SIZE = [], [], []
for j in range(S):
    m, rms = e1[j] / nw[0], (e2[j] / nw[0]).sqrt()
    g = rms.argsort(descending=True)[:4]
    GI.append(g.to("mps")); MEAN.append(m[g].float().to("mps")); SIZE.append(rms[g].float().to("mps"))


def dc(j, x):
    x[..., GI[j]] = MEAN[j].expand_as(x[..., GI[j]])
    return x


def limit(q):
    def f(j, x):
        c = q * SIZE[j]
        x[..., GI[j]] = c * torch.tanh(x[..., GI[j]] / c)
        return x
    return f


def crush(j, x, bits=8):
    s = x.abs().amax(-1, keepdim=True) / (2 ** (bits - 1) - 1)
    return (x / s).round().clamp(-(2 ** (bits - 1)), 2 ** (bits - 1) - 1) * s


def both(q):
    lf = limit(q)

    def f(j, x):
        return crush(j, lf(j, x))
    return f


IDEAS = [("dc: giants locked to their average", dc), ("limit 0.9", limit(0.9)), ("limit 0.7", limit(0.7)),
         ("limit 0.5", limit(0.5)), ("crush 8 bits (control)", crush), ("limit 0.9 + crush 8", both(0.9)),
         ("limit 0.5 + crush 8", both(0.5))]
lines = [f"Mastering the bloodstream: the 4 giants per layer ({nw[0]} fit words), {sum(len(w) - 1 for w, _ in test)} "
         "test words", "", f"{'idea':38} {'KL wiki':>8} {'KL chat':>8} {'KL math':>8}   dNLL wiki chat math"]
with torch.no_grad():
    for name, fn in IDEAS:
        st["fn"] = fn
        kl, dn = {}, {}
        for (w, s), (blp, bnll) in zip(test, BASE):
            lp, nll = run(w)
            b = blp.to("mps").float()
            kl.setdefault(s, []).append((b.exp() * (b - lp)).sum(-1).mean().item())
            dn.setdefault(s, []).append((nll.cpu() - bnll).mean().item())
            del lp, b
            torch.mps.empty_cache()
        lines.append(f"{name:38} " + " ".join(f"{np.mean(kl[s]):>8.4f}" for s in ("wiki", "chat", "math")) +
                     "   " + " ".join(f"{np.mean(dn[s]):+.3f}" for s in ("wiki", "chat", "math")))
        print(lines[-1], flush=True)
st["fn"] = None
gl = ["", "giants (index: average / size) at layers 0, 6, 12, 24, 41, final:"]
for j in (0, 6, 12, 24, 41, S - 1):
    gl.append(f"  {j if j < L else 'final'}: " + "  ".join(f"#{int(g)} {float(m):+.1f}/{float(z):.1f}"
                                                          for g, m, z in zip(GI[j], MEAN[j], SIZE[j])))
lines += gl + ["", "for scale: Google's own 4-bit release drifts 0.037 wiki / 0.027 chat from the 16-bit"]
open(os.path.join(out, "peaks.txt"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
print("PEAKS_DONE")
