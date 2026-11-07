#!/usr/bin/env python3
"""Weird-idea lab, round 4: FREEZE a layer (FL Studio "freeze track"; brain: reflexes). A layer's MLP answer for each
word is computed once with no context (<bos> + word) and stored as one table row; a least-squares straight line on the
MLP input adds the context correction:  out ~ table[word] + x A + b.  Reads per word for that layer: one row + the
line, instead of all 78.6M MLP weights. Harness as probe.py, different windows. (Immy, 2026-10-27: "come up with the most random ideas and just test them ... trying things you
would think no one has tried"). Load E4B once, throw many dumb-sounding changes at it, ~10 s each.

Every idea gets two scores on the same small set (2 wiki + 3 chat + 2 math windows, ~3,600 words):
  KL     how much it changes him (drift from the untouched model)
  dNLL   whether he predicts the REAL next word better (< 0) or worse (> 0), nats per word
plus idea-specific stats (how often a cache was used, how many layers' reads were saved).

usage: probe.py <out folder> <math traces jsonl> [idea names ...]       appends to probe.txt
"""
import json, os, sys, time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from e4b_mac import load

out, traces = sys.argv[1], sys.argv[2]
want = set(sys.argv[3:])
BENCH = os.path.join(os.path.dirname(__file__), "../../../.run/bench")
model, tok = load()
lm, layers = model.model, model.model.layers
L = len(layers)
wiki = tok(open(f"{BENCH}/wiki.test.raw").read()[:200000], add_special_tokens=False).input_ids
chat = tok(open(f"{BENCH}/chat_heldout.txt").read(), add_special_tokens=False).input_ids
rows = [json.loads(l) for l in open(traces)]


def mathwin(r):
    p = tok.apply_chat_template([{"role": "user", "content": r["q"]}], tokenize=False, add_generation_prompt=True,
                                enable_thinking=True)
    return tok(p + "<|channel>thought\n" + r["thinking"] + "<channel|>" + r["reply"], add_special_tokens=False).input_ids[:600]


test = [([tok.bos_token_id] + wiki[i * 511:(i + 1) * 511], "wiki") for i in (22, 23, 24, 25)] + \
       [([tok.bos_token_id] + chat[i * 511:(i + 1) * 511], "chat") for i in (43, 44, 45, 46)] + \
       [(mathwin(r), "math") for r in rows[7::75]]


TEMP = {"t": 1.0}


def run(w):
    lp = (model(input_ids=torch.tensor([w], device="mps")).logits[0, :-1].float() / TEMP["t"]).log_softmax(-1)
    tgt = torch.tensor(w[1:], device="mps")
    return lp, -lp.gather(1, tgt[:, None]).squeeze(1)


with torch.no_grad():
    BASE = []
    for w, s in test:
        lp, nll = run(w)
        BASE.append((lp.to(torch.bfloat16).cpu(), nll.cpu()))
        torch.mps.empty_cache()

hooks, STATS = [], {}


def clear():
    for h in hooks:
        h.remove()
    hooks.clear()
    STATS.clear()


def score(name):
    kl, dn = {}, {}
    t = time.time()
    with torch.no_grad():
        for (w, s), (blp, bnll) in zip(test, BASE):
            lp, nll = run(w)
            b = blp.to("mps").float()
            kl.setdefault(s, []).append((b.exp() * (b - lp)).sum(-1).mean().item())
            dn.setdefault(s, []).append((nll.cpu() - bnll).mean().item())
            del lp
            torch.mps.empty_cache()
    extra = " ".join(f"{k}={np.mean(v):.3f}" if isinstance(v, list) else f"{k}={v}" for k, v in STATS.items())
    line = (f"{name:38} KL " + " ".join(f"{np.mean(kl[s]):7.4f}" for s in ("wiki", "chat", "math")) +
            "   dNLL " + " ".join(f"{np.mean(dn[s]):+7.4f}" for s in ("wiki", "chat", "math")) + f"   {extra}")
    print(line + f"  ({time.time() - t:.0f}s)", flush=True)
    with open(os.path.join(out, "probe.txt"), "a") as f:
        f.write(line + "\n")
    clear()





FREEZE = [0, 1, 2, 3]
mlp_in, mlp_out = {}, {}


def cap_hook(i):
    def h(mod, args, o):
        mlp_in[i], mlp_out[i] = args[0], o
    return h


caps = [layers[i].mlp.register_forward_hook(cap_hook(i)) for i in FREEZE]
fit_w = [[tok.bos_token_id] + wiki[i * 511:(i + 1) * 511] for i in range(0, 16, 2)] + \
        [[tok.bos_token_id] + chat[i * 511:(i + 1) * 511] for i in range(0, 32, 2)] + [mathwin(r) for r in rows[::14]]
need = sorted({t for w in fit_w for t in w} | {t for w, _ in test for t in w})
D = model.config.hidden_size
row = torch.full((model.config.vocab_size,), -1, dtype=torch.long); row[need] = torch.arange(len(need))
table = {i: torch.zeros(len(need), D, dtype=torch.bfloat16) for i in FREEZE}   # CPU, only the words in use
with torch.no_grad():
    for c0 in range(0, len(need), 256):
        ids = torch.tensor([[tok.bos_token_id, t] for t in need[c0:c0 + 256]], device="mps")
        lm(input_ids=ids)
        for i in FREEZE:
            table[i][row[need[c0:c0 + 256]]] = mlp_out[i][:, 1].to(torch.bfloat16).cpu()
    print(f"alone tables: {len(need)} words x {len(FREEZE)} layers", flush=True)
    XtX = {i: torch.zeros(D + 1, D + 1, dtype=torch.float64) for i in FREEZE}
    XtY = {i: torch.zeros(D + 1, D, dtype=torch.float64) for i in FREEZE}
    R2 = {i: [0.0, 0.0, 0.0] for i in FREEZE}                     # table-only residual, total, line residual (later)
    for w in fit_w:
        lm(input_ids=torch.tensor([w], device="mps"))
        for i in FREEZE:
            x = mlp_in[i][0, 1:].float()
            y = mlp_out[i][0, 1:].float() - table[i][row[w[1:]]].to("mps").float()
            x1 = torch.cat([x, torch.ones(x.shape[0], 1, device="mps")], 1).cpu().double()
            XtX[i] += x1.T @ x1; XtY[i] += x1.T @ y.cpu().double()
        torch.mps.empty_cache()
LINE = {}
for i in FREEZE:
    reg = 1e-4 * XtX[i].diagonal()[:D].mean() * torch.eye(D + 1, dtype=torch.float64); reg[D, D] = 0
    Wb = torch.linalg.solve(XtX[i] + reg, XtY[i]).float()
    LINE[i] = (Wb[:D].to("mps"), Wb[D].to("mps"))
for c in caps:
    c.remove()
cur = {"ids": None}


def freeze(layer_set, use_line=True):
    def mk(i):
        def h(mod, args, o):
            ids = cur["ids"]
            y = table[i][row[ids]].to("mps").to(o.dtype)
            if use_line:
                A, b = LINE[i]
                y = y + (args[0][0].float() @ A + b).to(o.dtype)
            return y[None]
        return h
    for i in layer_set:
        hooks.append(layers[i].mlp.register_forward_hook(mk(i)))
    STATS["reads_saved"] = f"{len(layer_set) * 2.1:.1f}% of all"


_run = run


def run(w):
    cur["ids"] = torch.tensor(w)
    return _run(w)


IDEAS = {
    "freeze layer 0 (table only)": lambda: freeze([0], False),
    "freeze layer 0 (table + line)": lambda: freeze([0]),
    "freeze layer 1 (table + line)": lambda: freeze([1]),
    "freeze layer 2 (table + line)": lambda: freeze([2]),
    "freeze layer 3 (table + line)": lambda: freeze([3]),
    "freeze layers 0-1 (table + line)": lambda: freeze([0, 1]),
    "freeze layers 0-3 (table + line)": lambda: freeze([0, 1, 2, 3]),
}
with open(os.path.join(out, "probe.txt"), "a") as f:
    f.write(f"\n# round 4 freeze (fresh text) {time.strftime('%Y-%m-%d %H:%M')}\n")
for name, fn in IDEAS.items():
    fn()
    score(name)
print("PROBE4_DONE")
