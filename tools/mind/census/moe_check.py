#!/usr/bin/env python3
"""Dense E4B re-filed as a mixture of experts (2026-10-27): the standard MODEL format for "different neurons for
different words" (Gemma 4 has its own MoE layer: a dense block + experts + a router). Deciding question, training-free:
does E4B survive working in TEAMS instead of single neurons, and can a router built by plain least squares pick them?

Layers 0-12. Teams = balanced clusters of neurons that fire together (stations.py's method), 32 or 256 neurons each.
  perfect   the teams with the biggest real output for this word (upper bound)
  router    a straight-line router fitted by least squares from the MLP's input to each team's log output energy
            (closed form, no gradient training), as Gemma 4's router would hold it
  + line    plus the least-squares background line for the skipped teams (could live in the dense block as
            neuron pairs, see lean.py)

usage: moe_check.py <out folder> <math traces jsonl>      writes moe_check.txt
"""
import json, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from e4b_mac import load

out, traces = sys.argv[1], sys.argv[2]
BENCH = os.path.join(os.path.dirname(__file__), "../../../.run/bench")
LAYERS = list(range(13))
SIZES = (32, 256)
KEEPS = (0.2, 0.3)
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
dn = [l.mlp.down_proj.weight.float().norm(dim=0) for l in layers]
st = {"mode": "profile", "cfg": None, "x": {}}
prof = {i: [] for i in LAYERS}
PROFILE_TOKENS = 4000
team = {}                                                   # (size, layer) -> team id of each neuron (N,)
xtx = {i: torch.zeros(D + 1, D + 1, dtype=torch.float64) for i in LAYERS}
xty = {}                                                    # (size, layer) -> (D+1) x teams
xtr = {}                                                    # (size, keep, layer) -> (D+1) x D
ROUTER, LINE = {}, {}


def aug(x):
    return torch.cat([x, torch.ones(x.shape[0], 1, device=x.device)], 1).cpu().double()


def team_energy(p, i, size):
    e = (p.float() * dn[i]).pow(2)
    return torch.zeros(*e.shape[:-1], N // size, device=e.device).index_add_(-1, team[(size, i)], e)


def router_scores(x, i, size):
    W = ROUTER[(size, i)]
    return x.float() @ W[:D] + W[D]


def team_mask(scores, size, keep):
    top = scores.topk(int(keep * scores.shape[-1]), dim=-1).indices
    return torch.zeros_like(scores).scatter_(-1, top, 1)


def neuron_mask(tm, i, size):
    return tm[..., team[(size, i)]]


def mlp_pre(i):
    def h(mod, args):
        st["x"][i] = args[0]
    return h


def down_pre(i):
    def h(mod, args):
        if i not in LAYERS:
            return None
        p = args[0]
        mode = st["mode"]
        if mode == "profile":
            size = p[0, 1:].float().abs() * dn[i]
            top = size.topk(int(0.2 * N), dim=-1).indices
            prof[i].append(torch.zeros_like(size).scatter_(-1, top, 1).half().cpu())
            return None
        x = st["x"][i][0, 1:]
        if mode == "router":
            xa = aug(x.float())
            xtx[i] += xa.T @ xa
            for s in SIZES:
                y = team_energy(p[0, 1:], i, s).clamp_min(1e-12).log().cpu().double()
                xty[(s, i)] = xty.get((s, i), 0) + xa.T @ y
            return None
        if mode == "line":
            xa = aug(x.float())
            for s in SIZES:
                sc = router_scores(x, i, s)
                for k in KEEPS:
                    m = neuron_mask(team_mask(sc, s, k), i, s)
                    r = ((p[0, 1:] * (1 - m.to(p.dtype))).float() @ mod.weight.float().T).cpu().double()
                    xtr[(s, k, i)] = xtr.get((s, k, i), 0) + xa.T @ r
            return None
        c = st["cfg"]
        if not c:
            return None
        s, k = c["size"], c["keep"]
        sc = team_energy(p, i, s) if c["pick"] == "perfect" else router_scores(st["x"][i], i, s)
        return (p * neuron_mask(team_mask(sc, s, k), i, s).to(p.dtype),)
    return h


def mlp_post(i):
    def h(mod, args, o):
        c = st["cfg"]
        if st["mode"] != "run" or not c or not c.get("line") or i not in LAYERS:
            return None
        A, b = LINE[(c["size"], c["keep"], i)]
        return o + (args[0].float() @ A + b).to(o.dtype)
    return h


for i, l in enumerate(layers):
    l.mlp.register_forward_pre_hook(mlp_pre(i))
    l.mlp.down_proj.register_forward_pre_hook(down_pre(i))
    l.mlp.register_forward_hook(mlp_post(i))


def passes(mode, limit=None):
    st["mode"] = mode
    got = 0
    with torch.no_grad():
        for w in train:
            model.model(input_ids=torch.tensor([w], device="mps"))
            torch.mps.empty_cache()
            got += len(w) - 1
            if limit and got >= limit:
                break
    return got


def balanced(X, k, cap, iters=12, seed=0):
    """stations.py: X n x T co-firing profiles, exactly `cap` neurons per team"""
    n = X.shape[0]
    X = X / X.norm(dim=1, keepdim=True).clamp_min(1e-6)
    g = torch.Generator(device="cpu").manual_seed(seed)
    C = X[torch.randperm(n, generator=g)[:k].to(X.device)].clone()
    for _ in range(iters):
        S = X @ C.T
        lab = torch.full((n,), -1, dtype=torch.long, device=X.device)
        room = torch.full((k,), cap, dtype=torch.long, device=X.device)
        while (lab < 0).any():
            free = (lab < 0).nonzero().squeeze(1)
            s = S[free].masked_fill(room[None, :] <= 0, -1e9)
            want = s.argmax(1)
            score = s.gather(1, want[:, None]).squeeze(1)
            o1 = score.argsort(descending=True, stable=True)
            o = o1[want[o1].argsort(stable=True)]
            w_o = want[o]
            first = torch.ones_like(w_o, dtype=torch.bool); first[1:] = w_o[1:] != w_o[:-1]
            idx = torch.arange(len(o), device=X.device)
            start = torch.cummax(torch.where(first, idx, torch.zeros_like(idx)), 0).values
            ok = (idx - start) < room[w_o]
            lab[free[o[ok]]] = w_o[ok]
            room -= torch.bincount(w_o[ok], minlength=k)
        C = torch.zeros_like(C).index_add_(0, lab, X)
        C = C / C.norm(dim=1, keepdim=True).clamp_min(1e-6)
    return lab


t0 = time.time()
got = passes("profile", PROFILE_TOKENS)
print(f"profiles from {got} words ({time.time() - t0:.0f} s)", flush=True)
for i in LAYERS:
    X = torch.cat(prof[i], 0).T.float().to("mps")
    for s in SIZES:
        team[(s, i)] = balanced(X, N // s, s)
    prof[i] = None
print(f"teams built ({time.time() - t0:.0f} s)", flush=True)


def solve(a, b):
    reg = 1e-4 * a.diagonal()[:D].mean() * torch.eye(D + 1, dtype=torch.float64); reg[D, D] = 0
    return torch.linalg.solve(a + reg, b).float().to("mps")


passes("router")
for (s, i), y in xty.items():
    ROUTER[(s, i)] = solve(xtx[i], y)
print(f"routers fitted ({time.time() - t0:.0f} s)", flush=True)
passes("line")
for (s, k, i), r in xtr.items():
    Wb = solve(xtx[i], r)
    LINE[(s, k, i)] = (Wb[:D], Wb[D])
xtr.clear()
print(f"lines fitted ({time.time() - t0:.0f} s)", flush=True)

st["mode"] = "run"


def logp(w):
    return model(input_ids=torch.tensor([w], device="mps")).logits[0, 1:].float().log_softmax(-1)


CONF = [(f"teams of {s}, keep {k:.0%}, {pk}{' + line' if ln else ''}", s, k, pk, ln)
        for s in SIZES for k in KEEPS for pk, ln in (("perfect", False), ("router", False), ("router", True))]
kl = {(c[0], t): [] for c in CONF for t in ("wiki", "chat", "math")}
hit = {c[0]: [] for c in CONF}
with torch.no_grad():
    for j, (w, t) in enumerate(test):
        st["cfg"] = None
        b = logp(w)
        for name, s, k, pk, ln in CONF:
            st["cfg"] = {"size": s, "keep": k, "pick": pk, "line": ln}
            lp = logp(w)
            kl[(name, t)].append((b.exp() * (b - lp)).sum(-1).mean().item())
            del lp
        st["cfg"] = None
        del b
        torch.mps.empty_cache()
        print(f"  eval {j + 1}/{len(test)}", flush=True)

lines = ["Dense E4B as teams (mixture of experts), layers 0-12, training-free", "",
         f"{'config':44} {'KL wiki':>8} {'KL chat':>8} {'KL math':>8}"]
for name, *_ in CONF:
    lines.append(f"{name:44} " + " ".join(f"{np.mean(kl[(name, t)]):>8.4f}" for t in ("wiki", "chat", "math")))
lines += ["", "single neurons for comparison: perfect pick 0-12 20% + line 0.020 / 0.012 / 0.0014 (neuron_line.txt); "
          "gate pick 0.031 / 0.022 / 0.0024 (gate_line.txt)",
          "for scale: Google's own 4-bit release drifts 0.037 wiki / 0.027 chat from the 16-bit"]
open(os.path.join(out, "moe_check.txt"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
print("MOE_DONE")
