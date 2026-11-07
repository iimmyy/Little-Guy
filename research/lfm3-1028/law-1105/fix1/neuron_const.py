#!/usr/bin/env python3
"""How much of each neuron's output is the same every word? For every layer and neuron: mean and variance of its activity on held pages, and its write size.
Ceiling for a cut once constants are handled: how few neurons hold 90/95/99% of the varying energy, against the same for total energy. Energies are added as if neurons don't overlap, so this is a rough ceiling.
usage: neuron_const.py <hf> <read.pt> <out.pt> <report.txt> [pages=42 words=256]"""
import json, sys
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
args = [a for a in sys.argv[1:] if "=" not in a]; opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
hf, readf, out, rep = args; PAGES, WORDS = int(opt.get("pages", 42)), int(opt.get("words", 256))
DEV = "cuda"
tok = AutoTokenizer.from_pretrained(hf, trust_remote_code=True)
m = AutoModelForCausalLM.from_pretrained(hf, dtype=torch.bfloat16, trust_remote_code=True).to(DEV).eval().requires_grad_(False)
LS = m.model.layers; NL = len(LS); IDX = {}; NEU = {}; RMS = {}; G = {}
for L, b in enumerate(LS):
    b.feed_forward.w2.register_forward_pre_hook(lambda mod, a, L=L: NEU.__setitem__(L, a[0][0][IDX["i"]].float()))
    b.post_ffn_norm.register_forward_pre_hook(lambda mod, a, L=L: RMS.__setitem__(L, a[0][0][IDX["i"]].float().pow(2).mean(-1, keepdim=True).sqrt()))
    b.feed_forward.w1.register_forward_hook(lambda mod, a, o, L=L: G.__setitem__(L, o[0][IDX["i"]].float()))
held = json.load(open(readf + ".held.json"))["held"]
head = lambda q: tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=True) + "\n"
s1 = {L: 0 for L in range(NL)}; s2 = {L: 0 for L in range(NL)}; g1 = {L: 0 for L in range(NL)}; g2 = {L: 0 for L in range(NL)}; cnt = 0
for j in held[:PAGES]:
    h = head(j["q"]); ids = tok(h + j["pre"], add_special_tokens=False).input_ids[:4000]; first = len(tok(h, add_special_tokens=False).input_ids)
    if len(ids) - first < 40:
        continue
    rng = np.random.default_rng(len(ids)); IDX["i"] = torch.tensor(sorted(rng.choice(np.arange(first, len(ids) - 1), size=min(WORDS, len(ids) - 1 - first), replace=False).tolist()), device=DEV)
    with torch.inference_mode():
        m.model(input_ids=torch.tensor([ids], device=DEV))
    for L in range(NL):
        v = NEU[L] / RMS[L]  # the write is D*w2*h/rms, so h/rms is what each neuron really sends
        s1[L] = s1[L] + v.double().sum(0); s2[L] = s2[L] + v.double().pow(2).sum(0)
        gg = torch.nn.functional.silu(G[L]); g1[L] = g1[L] + gg.double().sum(0); g2[L] = g2[L] + gg.double().pow(2).sum(0)
    cnt += len(IDX["i"])
lines = [f"words {cnt}", "L  | const share of a typical neuron (median, p90) | share of neurons over 50%, 90% constant | gate spread/mean median | total energy that is constant | neurons for 90/95/99% varying energy | neurons for 90/95/99% total energy"]
keep = {}
for L, b in enumerate(LS):
    mu = s1[L] / cnt; e2 = s2[L] / cnt; var = (e2 - mu.pow(2)).clamp(min=0)
    D = b.post_ffn_norm.weight.float().double(); w2n = (b.feed_forward.w2.weight.float().double() * D[:, None]).pow(2).sum(0)
    tot, vary = e2 * w2n, var * w2n
    const = (mu.pow(2) / e2.clamp(min=1e-18))
    gm = g1[L] / cnt; gs = (g2[L] / cnt - gm.pow(2)).clamp(min=0).sqrt()
    def need(x):
        c = torch.cumsum(x.sort(descending=True).values, 0) / x.sum()
        return [int((c < t).sum()) + 1 for t in (.9, .95, .99)]
    lines.append(f"{L:2d} | {float(const.median()):.3f} {float(const.quantile(.9)):.3f} | {float((const>.5).double().mean()):.3f} {float((const>.9).double().mean()):.3f} | {float((gs/gm.abs().clamp(min=1e-9)).median()):.2f} | {float(1-vary.sum()/tot.sum()):.3f} | " + " ".join(map(str, need(vary))) + " | " + " ".join(map(str, need(tot))))
    keep[L] = dict(mu=mu.float().cpu(), e2=e2.float().cpu(), w2n=w2n.float().cpu())
torch.save(keep, out)
open(rep, "w").write("\n".join(lines) + "\n"); print("\n".join(lines)); print("NEURON_CONST_DONE")
