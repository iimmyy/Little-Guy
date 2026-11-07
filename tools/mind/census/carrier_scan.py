#!/usr/bin/env python3
"""Measurements for the carrier head, before any edit (step A of the installed time sense, NIGHT-PLAN-2026-10-29).
The carrier: one attention head in a full-attention layer that, from every thinking word, looks straight at `<think>`
(where the patient's own expected time is sharp, difficulty_probe.py) and carries that reading forward on a channel
nobody else uses. What it needs, measured here at layer L:
  heads     cost of giving up each head (KL on thinking words when its output is zeroed) and its look-back share
  key       how cleanly a straight line at the attention input picks out `<think>` from every other word (the key
            switch fires only there; the rest of the head's key group must not feel it)
  value     the least-used direction of the value group (the private channel), and how much the group uses it
  gate      the attention input's giant dimension (a constant to hold the head's sigmoid gate open)
  norm      spread of the layer's summed attention output size on thinking words (the sandwich norm divides the
            carried value by it: multiplicative noise)
  expected  the expected time read at the attention input on `<think>`: total thinking words and words until the
            answer first appears in the thinking (the time it should take), half to build, half to score

usage: carrier_scan.py <answers jsonl with thinking> <out folder> [layer=10 n=60 dev=mps]
"""
import json, os, random, sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../expert"))
ARGS, sys.argv[1:] = sys.argv[1:], []                     # grade.py grades its own arguments on import
from grade import answer_line
from patients import load
from probe_math import rho, ridge

src, out = ARGS[:2]
opt = dict(a.split("=", 1) for a in ARGS[2:])
L, N, DEV = int(opt.get("layer", 10)), int(opt.get("n", 60)), opt.get("dev", "mps")
os.makedirs(out, exist_ok=True)
TAIL = "\n\nEnd with a line: Answer: <your answer>"
p = load("lfm3", DEV)
p.model.config._attn_implementation = "eager"
tok, m = p.tok, p.model
ly = m.model.layers[L]
at = ly.self_attn
H, KV, HD = at.config.num_attention_heads, at.config.num_key_value_heads, at.head_dim
rows = [r for r in map(json.loads, open(src)) if r["thinking"]]
rng = random.Random(0)


def solved_at(r, th_ids):
    """Thinking word where the reply's own final answer first appears in the thinking (or the end if it never does)."""
    a = (answer_line(r["reply"], True) or "").strip().strip("$ ")
    k = r["thinking"].find(a) if a else -1
    if k < 0:
        return len(th_ids)
    return len(tok("\n" + r["thinking"][:k + len(a)], add_special_tokens=False).input_ids)


cap = {}
hk = [ly.operator_norm.register_forward_hook(lambda mod, a, o: cap.__setitem__("x", o[0].float())),
      at.register_forward_hook(lambda mod, a, o: cap.__setitem__("w", o[1][0].float())),
      ly.post_operator_norm.register_forward_pre_hook(lambda mod, a: cap.__setitem__("ao", a[0][0].float()))]
Xt, Xo, Ao, last, look = [], [], [], [], []
Ysolve, Ytot = [], []
seqs = []
with torch.no_grad():
    for n, r in enumerate(rows):
        pr = tok(tok.apply_chat_template([{"role": "user", "content": r["q"] + TAIL}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=True), add_special_tokens=False).input_ids
        th = tok("\n" + r["thinking"], add_special_tokens=False).input_ids
        T = r["think_tokens"] if r["finish"] == "stop" else 6000
        Ysolve.append(np.log(max(8, min(solved_at(r, th), T)))); Ytot.append(np.log(T))
        ids = pr + th[:700]
        cap.clear()
        m.model(input_ids=torch.tensor([ids], device=DEV))
        P = len(pr) - 1                                    # `<think>`
        Xt.append(cap["x"][P].cpu())
        if n < N:
            q = sorted(rng.sample(range(P + 1, len(ids)), min(48, len(ids) - P - 1)))
            Xo.append(cap["x"][q].cpu()); Ao.append(cap["ao"][q].cpu())
            look.append(cap["w"][:, P + 32:, P].mean(1).cpu())
            seqs.append(ids)
        if (n + 1) % 25 == 0:
            print(f"  read {n + 1}/{len(rows)}", flush=True)
for h in hk:
    h.remove()
Xt, Xo, Ao = torch.stack(Xt), torch.cat(Xo), torch.cat(Ao)
lines = [f"carrier scan, LFM3 layer {L} ({H} heads, {KV} key groups, head size {HD}), {len(rows)} answers ({N} for the stream)"]

# heads: the cost of zeroing each one, on thinking words (KL of the next-word odds, base vs head removed)
zero = [None]
def drop(mod, a):
    if zero[0] is None:
        return None
    x = a[0].clone(); x[..., zero[0] * HD:(zero[0] + 1) * HD] = 0
    return (x,)
hz = at.o_proj.register_forward_pre_hook(drop)
kl = torch.zeros(H)
with torch.no_grad():
    for ids in seqs[:10]:
        zero[0] = None
        base = torch.log_softmax(m(input_ids=torch.tensor([ids], device=DEV)).logits[0, -500:].float(), -1)
        for h in range(H):
            zero[0] = h
            lp = torch.log_softmax(m(input_ids=torch.tensor([ids], device=DEV)).logits[0, -500:].float(), -1)
            kl[h] += float((base.exp() * (base - lp)).sum(-1).mean()) / 10
hz.remove()
lk = torch.stack(look).mean(0)
lines.append("head  group  KL if zeroed  look-back share on <think> (thinking words 32+)")
for h in range(H):
    lines.append(f"{h:4d}  {h // (H // KV):5d}  {float(kl[h]):.5f}       {float(lk[h]):.3f}")

# key: a straight line that fires on `<think>` only (Fisher, ridge), its separation from every other word
Cx = torch.cov(Xo.T); Cx = Cx + 0.05 * Cx.diagonal().mean() * torch.eye(Cx.shape[0])
dk = torch.linalg.solve(Cx, Xt.mean(0) - Xo.mean(0))
st, so = Xt @ dk, Xo @ dk
lines.append(f"<think> detector: <think> {float(st.mean()):.2f} +- {float(st.std()):.2f}, other words "
             f"{float(so.mean()):.2f} +- {float(so.std()):.2f}, separation {(float(st.min()) - float(so.mean())) / float(so.std()):.0f} "
             f"spreads (worst <think> vs typical word), max other {(float(so.max()) - float(so.mean())) / float(so.std()):.1f}")

# value: least-used direction of each key group's value
for g in range(KV):
    Wv = at.v_proj.weight.float().cpu()[g * HD:(g + 1) * HD]
    V = Xo @ Wv.T
    ev, U = torch.linalg.eigh(torch.cov(V.T))
    lines.append(f"value group {g}: least-used direction carries {float(ev[0] / ev.sum()):.2e} of the variance "
                 f"(median direction {float(ev.median() / ev.sum()):.2e})")
mu, sd = Xo.mean(0), Xo.std(0)
gd = int((mu.abs() / (sd + 1e-6)).argmax())
lines.append(f"attention input giant dimension {gd}: {float(mu[gd]):.3f} +- {float(sd[gd]):.3f} ({float(sd[gd] / mu[gd].abs()):.1%})")
rms = Ao.pow(2).mean(-1).sqrt()
lines.append(f"summed attention output size on thinking words: median {float(rms.median()):.3f}, spread "
             f"{float(rms.std() / rms.mean()):.1%} (10-90%: {float(rms.quantile(.1)):.3f}-{float(rms.quantile(.9)):.3f})")

# expected time at the attention input on `<think>`
Ys, Yt = np.array(Ysolve), np.array(Ytot)
a, b = np.arange(len(Ys)) % 2 == 0, np.arange(len(Ys)) % 2 == 1
X = Xt.numpy()
for nm, y in [("words until the answer appears", Ys), ("total thinking words", Yt)]:
    pr_, al, cv, _ = ridge(X[a], y[a], X[b])
    lines.append(f"expected time at the attention input on <think>, {nm}: rho {rho(pr_, y[b]):+.2f} (built on {a.sum()}, ridge {al:g})")
lines.append(f"answer appears at median {int(np.exp(np.median(Ys)))} words vs thinking ends at {int(np.exp(np.median(Yt)))} "
             f"(share of thinking after the answer, median {float(np.median(1 - np.exp(Ys - Yt))):.0%})")
print("\n".join(lines))
open(os.path.join(out, f"carrier_scan_l{L}.txt"), "w").write("\n".join(lines) + "\n")
torch.save({"Xt": Xt, "Ysolve": Ys, "Ytot": Yt, "kl": kl, "look": lk, "think_dir": dk, "giant": gd}, os.path.join(out, f"carrier_scan_l{L}.pt"))
print("SCAN_DONE")
