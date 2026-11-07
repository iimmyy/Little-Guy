#!/usr/bin/env python3
"""E4B-lean, step 1 (2026-10-27, Immy: a changed MODEL, not a run-time trick): permanently delete the weakest neurons
in layers 0-12 (the same ones for every word, so the file really gets smaller) and build what they added back in as a
straight line. Measure only: nothing is saved as a brain file yet.

The line, two ways:
  part    a new straight-line part per layer, out += x A + b (exact least squares; a new tensor, needs engine support)
  pairs   the line written AS ORDINARY GEMMA NEURONS, so the file stays plain Gemma: GELU(a) - GELU(-a) = a exactly
          (also for the tanh version), so a neuron pair with gate rows +w/-w, up row = 1/c on one bloodstream number
          that stays nearly constant (c), and down columns +d/-d adds d (w . x)(x_k / c). Rank r of the line costs
          2r neurons. Works only as well as x_k really is constant: measured here, and the eval runs the pairs as
          they would really behave (times x_k / c), not the ideal line.

Importance of a neuron = its average output energy on sample text, E[p_i^2] |down_i|^2.

usage: lean.py <out folder> <math traces jsonl>      writes lean.txt
"""
import json, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from e4b_mac import load

out, traces = sys.argv[1], sys.argv[2]
BENCH = os.path.join(os.path.dirname(__file__), "../../../.run/bench")
LAYERS = list(range(13))
KEEPS = (0.7, 0.5)
RANKS = (128, 256)
model, tok = load()
layers = model.model.layers
L, D, N = len(layers), model.config.hidden_size, layers[0].mlp.intermediate_size
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
dn2 = [l.mlp.down_proj.weight.float().pow(2).sum(0) for l in layers]

st = {"mode": "stats", "cfg": None, "x": {}}
energy = {i: torch.zeros(N, dtype=torch.float64) for i in LAYERS}
xs = {i: [torch.zeros(D, dtype=torch.float64), torch.zeros(D, dtype=torch.float64)] for i in LAYERS}
xtx = {i: torch.zeros(D + 1, D + 1, dtype=torch.float64) for i in LAYERS}
xtr = {}
masks = {}
count = [0]


def aug(x):
    return torch.cat([x, torch.ones(x.shape[0], 1, device=x.device)], 1).cpu().double()


def mlp_pre(i):
    def h(mod, args):
        st["x"][i] = args[0]
    return h


def down_pre(i):
    def h(mod, args):
        if i not in LAYERS:
            return None
        p = args[0]
        x = st["x"][i][0, 1:].float()
        if st["mode"] == "stats":
            energy[i] += (p[0, 1:].float().pow(2).sum(0) * dn2[i]).cpu().double()
            xs[i][0] += x.sum(0).cpu().double(); xs[i][1] += x.pow(2).sum(0).cpu().double()
            xa = aug(x)
            xtx[i] += xa.T @ xa
            if i == 0:
                count[0] += x.shape[0]
            return None
        if st["mode"] == "fit":
            xa = aug(x)
            for k in KEEPS:
                r = ((p * (1 - masks[(k, i)])).float() @ mod.weight.float().T)[0, 1:].cpu().double()
                a = xtr.setdefault((k, i), torch.zeros(D + 1, D, dtype=torch.float64))
                a += xa.T @ r
            return None
        c = st["cfg"]
        if c and i in LAYERS:
            return (p * masks[(c["keep"], i)],)
        return None
    return h


def mlp_post(i):
    def h(mod, args, o):
        c = st["cfg"]
        if st["mode"] != "run" or not c or i not in LAYERS or c["line"] is None:
            return None
        x = args[0].float()
        if c["line"] == "part":
            A, b = PART[(c["keep"], i)]
            return o + (x @ A + b).to(o.dtype)
        W, Dm, b, kk, cc = PAIRS[(c["keep"], i, c["rank"])]
        s = x[..., kk:kk + 1] / cc                                  # what the "constant" input really is, over c
        return o + (((x @ W) * s) @ Dm + b * s * s).to(o.dtype)
    return h


for i, l in enumerate(layers):
    l.mlp.register_forward_pre_hook(mlp_pre(i))
    l.mlp.down_proj.register_forward_pre_hook(down_pre(i))
    l.mlp.register_forward_hook(mlp_post(i))


def passes(mode):
    st["mode"] = mode
    with torch.no_grad():
        for w in train:
            model.model(input_ids=torch.tensor([w], device="mps"))
            torch.mps.empty_cache()


t0 = time.time()
passes("stats")
for i in LAYERS:
    order = energy[i].argsort(descending=True)
    for k in KEEPS:
        m = torch.zeros(N); m[order[:int(k * N)]] = 1
        masks[(k, i)] = m.to("mps", torch.bfloat16)
print(f"stats on {count[0]} words ({time.time() - t0:.0f} s)", flush=True)
passes("fit")
print(f"fit ({time.time() - t0:.0f} s)", flush=True)

PART, PAIRS, const = {}, {}, {}
for i in LAYERS:
    mean = xs[i][0] / count[0]
    sd = (xs[i][1] / count[0] - mean ** 2).clamp_min(0).sqrt()
    ratio = mean.abs() / sd.clamp_min(1e-9)
    kk = int(ratio.argmax())
    const[i] = (kk, float(mean[kk]), float(ratio[kk]))
    reg = 1e-4 * xtx[i].diagonal()[:D].mean() * torch.eye(D + 1, dtype=torch.float64); reg[D, D] = 0
    for k in KEEPS:
        Wb = torch.linalg.solve(xtx[i] + reg, xtr[(k, i)])           # (D+1) x D
        PART[(k, i)] = (Wb[:D].float().to("mps"), Wb[D].float().to("mps"))
        # reduced-rank line: keep the r output directions the fitted line uses most on this text
        C = Wb.T @ xtx[i] @ Wb / count[0]
        ev, V = torch.linalg.eigh(C)
        for r in RANKS:
            P = V[:, -r:]                                             # D x r output directions
            W = Wb[:D] @ P                                            # input -> strand strengths
            PAIRS[(k, i, r)] = (W.float().to("mps"), P.T.float().to("mps"), Wb[D].float().to("mps"), kk,
                                torch.tensor(const[i][1], device="mps"))
print(f"lines solved ({time.time() - t0:.0f} s)", flush=True)

st["mode"] = "run"


def logp(w):
    return model(input_ids=torch.tensor([w], device="mps")).logits[0, 1:].float().log_softmax(-1)


CONF = [(f"keep {k:.0%}" + {None: "", "part": " + line part"}.get(ln, f" + line as {r * 2} neurons"), k, ln, r)
        for k in KEEPS for ln, r in [(None, 0), ("part", 0)] + [("pairs", r) for r in RANKS]]
kl = {(c[0], t): [] for c in CONF for t in ("wiki", "chat", "math")}
with torch.no_grad():
    for j, (w, t) in enumerate(test):
        st["cfg"] = None
        b = logp(w)
        for name, k, ln, r in CONF:
            st["cfg"] = {"keep": k, "line": ln, "rank": r}
            lp = logp(w)
            kl[(name, t)].append((b.exp() * (b - lp)).sum(-1).mean().item())
            del lp
        del b
        torch.mps.empty_cache()
        print(f"  eval {j + 1}/{len(test)}", flush=True)

q4 = 0.5625                                                           # bytes per weight in the 4-bit file
neuron_bytes = 3 * D * q4
grid4 = 4.22e9
lines = ["E4B-lean step 1: weakest neurons in layers 0-12 deleted for good (same ones for every word) + their average "
         "effect built back in as a straight line", "",
         f"{'config':34} {'KL wiki':>8} {'KL chat':>8} {'KL math':>8} {'file vs grid4':>14}"]
for name, k, ln, r in CONF:
    gone = len(LAYERS) * (N - int(k * N)) * neuron_bytes
    add = len(LAYERS) * (D * (D + 1) * q4 if ln == "part" else 2 * r * neuron_bytes if ln == "pairs" else 0)
    lines.append(f"{name:34} " + " ".join(f"{np.mean(kl[(name, t)]):>8.4f}" for t in ("wiki", "chat", "math")) +
                 f" {-(gone - add) / grid4:>+13.1%}")
lines += ["", "nearly-constant bloodstream number per layer (the pilot light the neuron pairs lean on): "
          "layer: index, average, average / spread"]
lines += ["  " + "  ".join(f"{i}: #{const[i][0]} {const[i][1]:+.2f} x{const[i][2]:.1f}" for i in LAYERS[s:s + 5])
          for s in range(0, len(LAYERS), 5)]
lines += ["", "for scale: Google's own 4-bit release drifts 0.037 wiki / 0.027 chat from the 16-bit; "
          "the run-time gate pick at 0-12 20% + line drifted 0.031 / 0.022 / 0.0024"]
open(os.path.join(out, "lean.txt"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
print("LEAN_DONE")
