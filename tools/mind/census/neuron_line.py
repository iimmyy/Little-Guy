#!/usr/bin/env python3
"""Single neurons + background line (2026-10-26): like tonic_phasic.py, but the kept units are single neurons, not
packs of 32 (possible in 4-bit if down_proj is stored one row per neuron; only down then leaves the QAT grid).

Tonic + phasic MLP (2026-10-26). Phasic: run only the strongest packs of 32 neurons exactly (packs = the 4-bit file's
storage unit along down_proj, so skipping them really skips reads). Tonic: one straight line predicts what all the
skipped weak packs would have added together (fitted by least squares from the MLP's input).

Perfect pick of packs (by this word's own outputs; a gate-first kernel would see the same after reading the gates).
Configs, end to end drift as before:   packs only   vs   packs + background line

usage: tonic_phasic.py <out folder> <math traces jsonl>      writes tonic_phasic.txt
"""
import json, os, sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from e4b_mac import load

out, traces = sys.argv[1], sys.argv[2]
BENCH = os.path.join(os.path.dirname(__file__), "../../../.run/bench")
model, tok = load()
layers = model.model.layers
L, D, N = len(layers), model.config.hidden_size, layers[0].mlp.intermediate_size
P = N // 32
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

# which (layers, keep) get a fitted background line
FITS = {("0-12", 0.2): set(range(13)), ("all", 0.4): set(range(L))}
dn = [l.mlp.down_proj.weight.float().norm(dim=0) for l in layers]
st = {"x": [None] * L, "mode": "off", "fit": None, "cfg": None}


def packmask(p, i, keep):
    size = p.float().abs() * dn[i]
    top = size.topk(int(keep * N), dim=-1).indices
    return torch.zeros_like(p).scatter_(-1, top, 1)


def mlp_pre(i):
    def h(mod, args):
        st["x"][i] = args[0]
    return h


acc = {}


def down_pre(i):
    def h(mod, args):
        p = args[0]
        if st["mode"] == "fit":
            for key, S in FITS.items():
                if i in S:
                    m = packmask(p, i, key[1])
                    r = ((p * (1 - m)).float() @ mod.weight.float().T)[0, 1:].cpu().double()
                    x = st["x"][i][0, 1:].float()
                    x = torch.cat([x, torch.ones(x.shape[0], 1, device=x.device)], 1).cpu().double()
                    a = acc.setdefault((key, i), [torch.zeros(D + 1, D + 1, dtype=torch.float64),
                                                  torch.zeros(D + 1, D, dtype=torch.float64)])
                    a[0] += x.T @ x; a[1] += x.T @ r
            return None
        c = st["cfg"]
        if c and i in c["layers"]:
            return (p * packmask(p, i, c["keep"]),)
        return None
    return h


def mlp_post(i):
    def h(mod, args, o):
        c = st["cfg"]
        if st["mode"] == "run" and c and c.get("line") and i in c["layers"]:
            A, b = c["line"][i]
            return o + (args[0].float() @ A + b).to(o.dtype)
        return None
    return h


for i, l in enumerate(layers):
    l.mlp.register_forward_pre_hook(mlp_pre(i))
    l.mlp.down_proj.register_forward_pre_hook(down_pre(i))
    l.mlp.register_forward_hook(mlp_post(i))

st["mode"] = "fit"
with torch.no_grad():
    for k, w in enumerate(train):
        model.model(input_ids=torch.tensor([w], device="mps"))
        torch.mps.empty_cache()
        if k % 5 == 4:
            print(f"  fit {k + 1}/{len(train)}", flush=True)
lines_fit = {}
for (key, i), (xtx, xtr) in acc.items():
    reg = 1e-4 * xtx.diagonal()[:D].mean() * torch.eye(D + 1, dtype=torch.float64); reg[D, D] = 0
    Wb = torch.linalg.solve(xtx + reg, xtr).float()
    lines_fit.setdefault(key, {})[i] = (Wb[:D].to("mps"), Wb[D].to("mps"))
acc.clear()
st["mode"] = "run"


def logp(w):
    return model(input_ids=torch.tensor([w], device="mps")).logits[0, 1:].float().log_softmax(-1)


CONF = [("0-12 neurons 20%", set(range(13)), 0.2, None), ("0-12 neurons 20% + line", set(range(13)), 0.2, ("0-12", 0.2)),
        ("all neurons 40%", set(range(L)), 0.4, None), ("all neurons 40% + line", set(range(L)), 0.4, ("all", 0.4))]
kl = {(c[0], t): [] for c in CONF for t in ("wiki", "chat", "math")}
same = {c[0]: [] for c in CONF}
with torch.no_grad():
    for j, (w, t) in enumerate(test):
        st["cfg"] = None
        b = logp(w)
        for name, S, keep, fk in CONF:
            st["cfg"] = {"layers": S, "keep": keep, "line": lines_fit[fk] if fk else None}
            lp = logp(w)
            kl[(name, t)].append((b.exp() * (b - lp)).sum(-1).mean().item())
            same[name].append((lp.argmax(-1) == b.argmax(-1)).float().mean().item())
            del lp
        del b
        torch.mps.empty_cache()
        print(f"  e2e {j + 1}/{len(test)}", flush=True)
mlp_w = 3 * D * N
lines = ["Single neurons + background line: strongest neurons exact, a least-squares line for the skipped rest", "",
         f"{'config':24} {'KL wiki':>8} {'KL chat':>8} {'KL math':>8} {'same top':>9} {'MLP bytes saved (gate-first)':>29}"]
for name, S, keep, fk in CONF:
    saved = len(S) / L * ((1 - keep) * 2 / 3 - (D * D / mlp_w if fk else 0))
    lines.append(f"{name:24} " + " ".join(f"{np.mean(kl[(name, t)]):>8.4f}" for t in ("wiki", "chat", "math")) +
                 f" {np.mean(same[name]) * 100:>8.1f}% {saved * 100:>28.1f}%")
lines += ["", "(for scale: Google's own 4-bit release drifts 0.037 wiki / 0.027 chat from the 16-bit)"]
open(os.path.join(out, "neuron_line.txt"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
print("NLINE_DONE")
