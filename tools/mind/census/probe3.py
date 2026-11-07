#!/usr/bin/env python3
"""Weird-idea lab, round 3: is "late MLPs 5% quieter predicts real text better" real (fresh text it never saw) and
is it more than temperature? Harness as probe.py, different windows. (Immy, 2026-10-27: "come up with the most random ideas and just test them ... trying things you
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




def gains(spec, ga=None):
    """spec: {layer range: gain} on the MLP contribution; ga: same for attention"""
    for rng, g in spec.items():
        for i in rng:
            hooks.append(layers[i].post_feedforward_layernorm.register_forward_hook(lambda m, a, o, g=g: o * g))
    for rng, g in (ga or {}).items():
        for i in rng:
            hooks.append(layers[i].post_attention_layernorm.register_forward_hook(lambda m, a, o, g=g: o * g))


def temp(t):
    TEMP["t"] = t


LATE = range(24, 42)
IDEAS = {
    "temperature 1.03": lambda: temp(1.03),
    "temperature 1.05": lambda: temp(1.05),
    "temperature 1.08": lambda: temp(1.08),
    "late MLP x0.97": lambda: gains({LATE: 0.97}),
    "late MLP x0.95": lambda: gains({LATE: 0.95}),
    "late MLP x0.93": lambda: gains({LATE: 0.93}),
    "layers 24-29 MLP x0.9": lambda: gains({range(24, 30): 0.9}),
    "layers 30-35 MLP x0.9": lambda: gains({range(30, 36): 0.9}),
    "layers 36-41 MLP x0.9": lambda: gains({range(36, 42): 0.9}),
    "layer 41 MLP x0.8": lambda: gains({range(41, 42): 0.8}),
    "late MLP x0.95 + temperature 1.03": lambda: (gains({LATE: 0.95}), temp(1.03)),
    "late MLP x0.95 + temperature 0.97": lambda: (gains({LATE: 0.95}), temp(0.97)),
}
with open(os.path.join(out, "probe.txt"), "a") as f:
    f.write(f"\n# round 3 (FRESH text) {time.strftime('%Y-%m-%d %H:%M')}  {sum(len(w) - 1 for w, _ in test)} words\n")
for name, fn in IDEAS.items():
    fn()
    score(name)
    TEMP["t"] = 1.0
print("PROBE3_DONE")
