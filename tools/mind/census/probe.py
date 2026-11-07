#!/usr/bin/env python3
"""Weird-idea lab (Immy, 2026-10-27: "come up with the most random ideas and just test them ... trying things you
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


test = [([tok.bos_token_id] + wiki[i * 511:(i + 1) * 511], "wiki") for i in (20, 21)] + \
       [([tok.bos_token_id] + chat[i * 511:(i + 1) * 511], "chat") for i in (40, 41, 42)] + \
       [(mathwin(r), "math") for r in rows[5::150]]


def run(w):
    lp = model(input_ids=torch.tensor([w], device="mps")).logits[0, :-1].float().log_softmax(-1)
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


# ---- ideas ----------------------------------------------------------------------------------------------------------
def habituation(layer_set, thr, window=64):
    """repetition suppression: if this word's MLP input is almost the same as one of the last `window` words'
    (cosine > thr), reuse that word's MLP output and skip the MLP for this word"""
    STATS["reuse"] = []

    def mk(i):
        def h(mod, args, o):
            x = args[0][0].float()                                     # T x D
            xn = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            S = xn @ xn.T
            T = S.shape[0]
            idx = torch.arange(T, device=S.device)
            ok = (idx[None, :] < idx[:, None]) & (idx[None, :] >= idx[:, None] - window) & (idx[None, :] > 0)
            S = S.masked_fill(~ok, -2)
            best, j = S.max(1)
            use = best > thr
            STATS["reuse"].append(use.float().mean().item())
            o2 = o.clone()
            o2[0, use] = o[0, j[use]]
            return o2
        return h
    for i in layer_set:
        hooks.append(layers[i].mlp.register_forward_hook(mk(i)))


def gain(layer_set, g, part="mlp"):
    """volume knob on a layer's contribution to the bloodstream (after its output norm)"""
    for i in layer_set:
        mod = layers[i].post_feedforward_layernorm if part == "mlp" else layers[i].post_attention_layernorm
        hooks.append(mod.register_forward_hook(lambda m, a, o: o * g))


def noise(layer_set, amp, seed=0):
    """stochastic resonance: a little Gaussian noise on the bloodstream entering these layers"""
    gen = torch.Generator(device="mps").manual_seed(seed)

    def h(mod, args):
        x = args[0]
        n = torch.randn(x.shape, generator=gen, device=x.device, dtype=torch.float32)
        return (x + (amp * x.float().norm(dim=-1, keepdim=True) / x.shape[-1] ** 0.5 * n).to(x.dtype),) + args[1:]
    for i in layer_set:
        hooks.append(layers[i].register_forward_pre_hook(h))


def twice(i):
    """think twice: run layer i again on its own output (only layers 24-41: they write no memory)"""
    def h(mod, args, kwargs, o):
        hooks_off = [hk for hk in []]
        return mod.__class__.forward(mod, o, args[1], **kwargs)
    hooks.append(layers[i].register_forward_hook(h, with_kwargs=True))


def skip(i):
    def h(mod, args, kwargs, o):
        return args[0]
    hooks.append(layers[i].register_forward_hook(h, with_kwargs=True))


SAVED = {}


def merge(i, j):
    """average two neighbour layers' weights into layer i, skip layer j (both 24-41, same attention type)"""
    pi, pj = dict(layers[i].named_parameters()), dict(layers[j].named_parameters())
    SAVED[i] = {k: v.detach().cpu().clone() for k, v in pi.items()}
    with torch.no_grad():
        for k in pi:
            pi[k].copy_(((pi[k].float() + pj[k].float()) / 2).to(pi[k].dtype))
    skip(j)
    STATS["layers_saved"] = 1


def unmerge():
    for i, d in SAVED.items():
        with torch.no_grad():
            for k, v in layers[i].named_parameters():
                v.copy_(d[k].to(v.device))
    SAVED.clear()


def crush(bits):
    """the bloodstream rounded to `bits` bits per number (per word scale) at every layer's input"""
    def h(mod, args):
        x = args[0].float()
        s = x.abs().amax(-1, keepdim=True) / (2 ** (bits - 1) - 1)
        return ((x / s).round().clamp(-(2 ** (bits - 1)), 2 ** (bits - 1) - 1) * s).to(args[0].dtype),
    for l in layers:
        hooks.append(l.register_forward_pre_hook(lambda m, a: h(m, a) + a[1:]))


def swap_gate_up(layer_set):
    def mk(i):
        def h(mod, args, o):
            x = args[0]
            return mod.down_proj(mod.act_fn(mod.up_proj(x)) * mod.gate_proj(x))
        return h
    for i in layer_set:
        hooks.append(layers[i].mlp.register_forward_hook(mk(i)))


def flip(i):
    def h(mod, args, kwargs, o):
        return 2 * args[0] - o
    hooks.append(layers[i].register_forward_hook(h, with_kwargs=True))


def lag(layer_set):
    """each of these layers' MLP answers with the PREVIOUS word's output (one-word delay)"""
    def h(mod, args, o):
        o2 = o.clone()
        o2[0, 1:] = o[0, :-1]
        return o2
    for i in layer_set:
        hooks.append(layers[i].mlp.register_forward_hook(h))


LATE, MID, EARLY, ALL = range(24, 42), range(13, 24), range(0, 13), range(42)
IDEAS = {
    "habituation late 0.98": lambda: habituation(LATE, 0.98),
    "habituation late 0.95": lambda: habituation(LATE, 0.95),
    "habituation late 0.90": lambda: habituation(LATE, 0.90),
    "habituation all 0.95": lambda: habituation(ALL, 0.95),
    "gain mlp late 0.9": lambda: gain(LATE, 0.9),
    "gain mlp late 1.1": lambda: gain(LATE, 1.1),
    "gain mlp mid 0.9": lambda: gain(MID, 0.9),
    "gain mlp mid 1.1": lambda: gain(MID, 1.1),
    "gain mlp early 1.1": lambda: gain(EARLY, 1.1),
    "gain attn late 1.1": lambda: gain(LATE, 1.1, "attn"),
    "gain attn late 0.9": lambda: gain(LATE, 0.9, "attn"),
    "noise mid 0.02": lambda: noise(MID, 0.02),
    "noise mid 0.05": lambda: noise(MID, 0.05),
    "noise late 0.05": lambda: noise(LATE, 0.05),
    "think twice 30": lambda: twice(30),
    "think twice 36": lambda: twice(36),
    "think twice 40": lambda: twice(40),
    "merge 26+27": lambda: merge(26, 27),
    "merge 32+33": lambda: merge(32, 33),
    "merge 38+39": lambda: merge(38, 39),
    "skip 27 (control for merge)": lambda: skip(27),
    "skip 33 (control for merge)": lambda: skip(33),
    "crush bloodstream 8 bits": lambda: crush(8),
    "crush bloodstream 4 bits": lambda: crush(4),
    "crush bloodstream 2 bits": lambda: crush(2),
    "swap gate/up layer 30": lambda: swap_gate_up([30]),
    "flip sign layer 30": lambda: flip(30),
    "lag late MLPs one word": lambda: lag(LATE),
    "lag layer 40 MLP one word": lambda: lag([40]),
}
with open(os.path.join(out, "probe.txt"), "a") as f:
    f.write(f"\n# {time.strftime('%Y-%m-%d %H:%M')}  {sum(len(w) - 1 for w, _ in test)} words; "
            "KL wiki chat math, dNLL wiki chat math (negative = predicts real text BETTER)\n")
for name, fn in IDEAS.items():
    if want and not any(wd in name for wd in want):
        continue
    fn()
    score(name)
    unmerge()
print("PROBE_DONE")
