#!/usr/bin/env python3
"""Weird-idea lab, round 2 (bytes): bits by importance, English-only dictionary, bloodstream x-ray, volume sweep.
Harness as probe.py. (Immy, 2026-10-27: "come up with the most random ideas and just test them ... trying things you
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



from safetensors import safe_open
from e4b_mac import FILE

SHARE = torch.tensor(np.load(os.path.join(out, "census.npz"))["share"], device="mps")      # L x N


def fake_quant_rows(W, bits, group=32):
    """round each row in groups of `group` to `bits` bits (asymmetric min/max per group); 1 bit = sign x mean |w|"""
    shp = W.shape
    G = W.float().reshape(-1, group)
    if bits == 1:
        q = G.sign() * G.abs().mean(1, keepdim=True)
    else:
        lo, hi = G.amin(1, keepdim=True), G.amax(1, keepdim=True)
        s = (hi - lo).clamp_min(1e-8) / (2 ** bits - 1)
        q = ((G - lo) / s).round().clamp(0, 2 ** bits - 1) * s + lo
    return q.reshape(shp).to(W.dtype)


def restore_mlp():
    with safe_open(FILE, "pt") as f, torch.no_grad():
        for i, l in enumerate(layers):
            for nm in ("gate_proj", "up_proj", "down_proj"):
                getattr(l.mlp, nm).weight.copy_(f.get_tensor(f"model.language_model.layers.{i}.mlp.{nm}.weight").to("mps"))


def bits_by_importance(frac, bits):
    """the weakest `frac` of each layer's neurons (census output share) stored at `bits` bits"""
    with torch.no_grad():
        for i, l in enumerate(layers):
            weak = SHARE[i].argsort()[: int(frac * SHARE.shape[1])]
            g, u, d = l.mlp.gate_proj.weight, l.mlp.up_proj.weight, l.mlp.down_proj.weight
            g[weak] = fake_quant_rows(g[weak], bits)
            u[weak] = fake_quant_rows(u[weak], bits)
            d[:, weak] = fake_quant_rows(d[:, weak].T.contiguous(), bits).T
    STATS["MLP_bytes_saved"] = f"{frac * (4.5 - (bits + 0.5)) / 4.5 * 100:.0f}%"
    return "mlp"


def all_at(bits):
    """control: ALL neurons at `bits` bits (round-to-nearest, no QAT grid) to see what the 4-bit rounding itself costs"""
    return bits_by_importance(1.0, bits)


# English-only dictionary: keep the K words E4B most often puts in its own top-10 on a separate English text sample
with torch.no_grad():
    cnt = torch.zeros(model.config.vocab_size, device="mps")
    sample = [[tok.bos_token_id] + wiki[i * 511:(i + 1) * 511] for i in range(0, 16)] + \
             [[tok.bos_token_id] + chat[i * 511:(i + 1) * 511] for i in range(0, 32)] + [mathwin(r) for r in rows[::10]]
    for w in sample:
        z = model(input_ids=torch.tensor([w], device="mps")).logits[0]
        cnt.index_add_(0, z.topk(10, dim=-1).indices.flatten(), torch.ones(z.shape[0] * 10, device="mps"))
        cnt.index_add_(0, torch.tensor(w, device="mps"), torch.ones(len(w), device="mps"))
        torch.mps.empty_cache()
    seen = int((cnt > 0).sum())
    rank = cnt.argsort(descending=True)
print(f"dictionary: {seen} of {model.config.vocab_size} words ever used or in a top-10 on the sample", flush=True)


def english_head(K):
    keep = torch.zeros(model.config.vocab_size, dtype=torch.bool, device="mps")
    keep[rank[:K]] = True

    def h(mod, args, o):
        return o.masked_fill(~keep, -1e4)
    hooks.append(model.lm_head.register_forward_hook(h))
    STATS["head_reads_saved"] = f"{(1 - K / model.config.vocab_size) * 100:.0f}%"


def xray():
    """not a change: report how lopsided the bloodstream is (biggest number vs the typical one, per word)"""
    STATS["max/median"] = []
    STATS["top3 dims share of energy"] = []
    STATS["giant dims"] = set()

    def h(mod, args):
        x = args[0][0, 1:].float().abs()
        STATS["max/median"].append((x.max(-1).values / x.median(-1).values).mean().item())
        e = x.pow(2)
        STATS["top3 dims share of energy"].append((e.topk(3, -1).values.sum(-1) / e.sum(-1)).mean().item())
        STATS["giant dims"].update(x.argmax(-1).unique().tolist()[:5])
    for l in layers[::6]:
        hooks.append(l.register_forward_pre_hook(h))


def gains(gm_early, gm_mid, gm_late, ga_late=1.0):
    for rng, g in ((range(0, 13), gm_early), (range(13, 24), gm_mid), (range(24, 42), gm_late)):
        if g != 1.0:
            for i in rng:
                hooks.append(layers[i].post_feedforward_layernorm.register_forward_hook(lambda m, a, o, g=g: o * g))
    if ga_late != 1.0:
        for i in range(24, 42):
            hooks.append(layers[i].post_attention_layernorm.register_forward_hook(lambda m, a, o: o * ga_late))


IDEAS = {
    "xray bloodstream (no change)": xray,
    "control: all neurons 4-bit RTN": lambda: all_at(4),
    "weakest 25% neurons at 2 bits": lambda: bits_by_importance(0.25, 2),
    "weakest 50% neurons at 2 bits": lambda: bits_by_importance(0.5, 2),
    "weakest 75% neurons at 2 bits": lambda: bits_by_importance(0.75, 2),
    "weakest 50% neurons at 1 bit": lambda: bits_by_importance(0.5, 1),
    "weakest 25% neurons at 1 bit": lambda: bits_by_importance(0.25, 1),
    "english head 16k": lambda: english_head(16384),
    "english head 32k": lambda: english_head(32768),
    "english head 64k": lambda: english_head(65536),
    "gains late .9": lambda: gains(1, 1, 0.9),
    "gains late .95": lambda: gains(1, 1, 0.95),
    "gains late .85": lambda: gains(1, 1, 0.85),
    "gains mid 1.05 late .95": lambda: gains(1, 1.05, 0.95),
    "gains early 1.05 mid 1.05 late .9": lambda: gains(1.05, 1.05, 0.9),
    "gains late .9 attn late 1.05": lambda: gains(1, 1, 0.9, 1.05),
}
with open(os.path.join(out, "probe.txt"), "a") as f:
    f.write(f"\n# round 2 {time.strftime('%Y-%m-%d %H:%M')}  {sum(len(w) - 1 for w, _ in test)} words\n")
for name, fn in IDEAS.items():
    if want and not any(wd in name for wd in want):
        continue
    touched = fn()
    score(name) if name != "xray bloodstream (no change)" else None
    if name.startswith("xray"):
        with torch.no_grad():
            for w, _ in test:
                model.model(input_ids=torch.tensor([w], device="mps"))
        line = f"{name:38} " + " ".join(f"{k}={np.mean(v):.3f}" if isinstance(v, list) else f"{k}={sorted(v)}" for k, v in STATS.items())
        print(line, flush=True)
        open(os.path.join(out, "probe.txt"), "a").write(line + "\n")
        clear()
    if touched == "mlp":
        restore_mlp()
print("PROBE2_DONE")
