#!/usr/bin/env python3
"""Stations (Immy's Ramipril picture, 2026-10-26): reorder each layer's neurons so the ones that fire together share a
pack of 32, then route whole packs (the 4-bit file's unit along down_proj) with a background line for the rest.

Reordering neurons is exact in 16-bit (same rows of gate/up, same columns of down, just moved). In the 4-bit file the
gate/up rows keep their grid; only down needs re-storing (measured separately).

1. co-firing: for sample words, which neurons are in this word's top 20% (by output size)
2. balanced clustering: 320 stations x 32 neurons per layer, spherical k-means on those co-firing profiles with
   exactly 32 per station (greedy capacity assignment)
3. same end-to-end test as tonic_phasic.py, packs = stations: perfect pick of stations, with and without a fitted line

usage: stations.py <out folder> <math traces jsonl>      writes stations.txt and station_perms.pt
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
dn = [l.mlp.down_proj.weight.float().norm(dim=0) for l in layers]
st = {"x": [None] * L, "mode": "profile", "cfg": None}
prof = [[] for _ in range(L)]
PROFILE_TOKENS = 4000


def mlp_pre(i):
    def h(mod, args):
        st["x"][i] = args[0]
    return h


acc = {}
FITS = {("0-12", 0.2): set(range(13)), ("all", 0.4): set(range(L))}
perm = [None] * L
inv = [None] * L


def stmask(p, i, keep):
    """mask of the kept stations, in the model's own neuron order"""
    s = (p.float().abs() * dn[i]).pow(2)[..., perm[i]].view(*p.shape[:-1], P, 32).sum(-1)
    top = s.topk(int(keep * P), dim=-1).indices
    m = torch.zeros(*p.shape[:-1], P, device=p.device).scatter_(-1, top, 1).repeat_interleave(32, -1)
    return m[..., inv[i]].to(p.dtype)


def down_pre(i):
    def h(mod, args):
        p = args[0]
        if st["mode"] == "profile":
            size = p[0, 1:].float().abs() * dn[i]
            top = size.topk(int(0.2 * N), dim=-1).indices
            prof[i].append(torch.zeros_like(size).scatter_(-1, top, 1).half().cpu())
            return None
        if st["mode"] == "fit":
            for key, S in FITS.items():
                if i in S:
                    m = stmask(p, i, key[1])
                    r = ((p * (1 - m)).float() @ mod.weight.float().T)[0, 1:].cpu().double()
                    x = st["x"][i][0, 1:].float()
                    x = torch.cat([x, torch.ones(x.shape[0], 1, device=x.device)], 1).cpu().double()
                    a = acc.setdefault((key, i), [torch.zeros(D + 1, D + 1, dtype=torch.float64),
                                                  torch.zeros(D + 1, D, dtype=torch.float64)])
                    a[0] += x.T @ x; a[1] += x.T @ r
            return None
        c = st["cfg"]
        if c and i in c["layers"]:
            return (p * stmask(p, i, c["keep"]),)
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

# 1. co-firing profiles --------------------------------------------------------------------------------------------
with torch.no_grad():
    got = 0
    for w in train:
        model.model(input_ids=torch.tensor([w], device="mps"))
        got += len(w) - 1
        torch.mps.empty_cache()
        if got >= PROFILE_TOKENS:
            break
print(f"profiles from {got} words", flush=True)


# 2. balanced clustering --------------------------------------------------------------------------------------------
def balanced(X, k, cap, iters=12, seed=0):
    """X: n x T (rows = neurons' co-firing profiles), exactly `cap` per cluster"""
    n = X.shape[0]
    X = X / X.norm(dim=1, keepdim=True).clamp_min(1e-6)
    g = torch.Generator(device="cpu").manual_seed(seed)
    C = X[torch.randperm(n, generator=g)[:k].to(X.device)].clone()
    for _ in range(iters):
        S = X @ C.T                                                            # n x k
        lab = torch.full((n,), -1, dtype=torch.long, device=X.device)
        room = torch.full((k,), cap, dtype=torch.long, device=X.device)
        # rounds: every unplaced neuron asks its best station that still has room; a station keeps its best askers
        while (lab < 0).any():
            free = (lab < 0).nonzero().squeeze(1)
            s = S[free].masked_fill(room[None, :] <= 0, -1e9)
            want = s.argmax(1)
            score = s.gather(1, want[:, None]).squeeze(1)
            # rank askers inside each station by score, accept up to that station's room
            o1 = score.argsort(descending=True, stable=True)                        # best first ...
            o = o1[want[o1].argsort(stable=True)]                                   # ... grouped by station
            w_o = want[o]
            first = torch.ones_like(w_o, dtype=torch.bool); first[1:] = w_o[1:] != w_o[:-1]
            idx = torch.arange(len(o), device=X.device)
            start = torch.cummax(torch.where(first, idx, torch.zeros_like(idx)), 0).values
            rank = idx - start
            ok = rank < room[w_o]
            lab[free[o[ok]]] = w_o[ok]
            room -= torch.bincount(w_o[ok], minlength=k)
        C = torch.zeros_like(C).index_add_(0, lab, X)
        C = C / C.norm(dim=1, keepdim=True).clamp_min(1e-6)
    return lab


for i in range(L):
    X = torch.cat(prof[i], 0).T.float().to("mps")                              # N x T
    lab = balanced(X, P, 32)
    perm[i] = lab.argsort(stable=True)
    inv[i] = perm[i].argsort()
    prof[i] = None
    if i % 6 == 5:
        print(f"  clustered layer {i}", flush=True)
torch.save([p.cpu() for p in perm], os.path.join(out, "station_perms.pt"))

# 3. fit the background lines, then end to end --------------------------------------------------------------------
st["mode"] = "fit"
with torch.no_grad():
    for k, w in enumerate(train):
        model.model(input_ids=torch.tensor([w], device="mps"))
        torch.mps.empty_cache()
        if k % 10 == 9:
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


CONF = [("0-6 stations 20%", set(range(7)), 0.2, None), ("0-12 stations 20%", set(range(13)), 0.2, None),
        ("0-12 stations 20% + line", set(range(13)), 0.2, ("0-12", 0.2)), ("all stations 40%", set(range(L)), 0.4, None),
        ("all stations 40% + line", set(range(L)), 0.4, ("all", 0.4))]
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
lines = ["Stations: neurons reordered so co-firing ones share packs of 32; perfect pick of stations (+ background line)",
         "", f"{'config':26} {'KL wiki':>8} {'KL chat':>8} {'KL math':>8} {'same top':>9} {'MLP bytes saved (gate-first)':>29}"]
for name, S, keep, fk in CONF:
    saved = len(S) / L * ((1 - keep) * 2 / 3 - (D * D / mlp_w if fk else 0))
    lines.append(f"{name:26} " + " ".join(f"{np.mean(kl[(name, t)]):>8.4f}" for t in ("wiki", "chat", "math")) +
                 f" {np.mean(same[name]) * 100:>8.1f}% {saved * 100:>28.1f}%")
lines += ["", "same test with packs in the model's own random order (tonic_phasic.txt):",
          "  0-6 packs 20%: 0.1093 / 0.0847 / 0.0053;  0-12 + line: 0.1388 / 0.0889 / 0.0085;  all 40% + line: 0.7299 / 0.7079 / 0.1870",
          "single neurons (route_e2e.txt): 0-6 at 20%: 0.0118 / 0.0060 / 0.0008;  all 42 at 40%: 0.0773 / 0.0681 / 0.0484",
          "(for scale: Google's own 4-bit release drifts 0.037 wiki / 0.027 chat from the 16-bit)"]
open(os.path.join(out, "stations.txt"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
print("STATIONS_DONE")
