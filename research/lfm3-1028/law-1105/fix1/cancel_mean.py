#!/usr/bin/env python3
"""Is the cancelling a constant? Reads each write's average vector on held pages: its share of the write's energy, cos between neighbours' averages, and neighbour cos after taking the averages out.
usage: cancel_mean.py <hf> <read.pt> <report.txt> [pages=40 words=128]"""
import json, sys
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
args = [a for a in sys.argv[1:] if "=" not in a]; opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
hf, readf, rep = args; PAGES, WORDS = int(opt.get("pages", 40)), int(opt.get("words", 128))
DEV = "cuda"
tok = AutoTokenizer.from_pretrained(hf, trust_remote_code=True)
m = AutoModelForCausalLM.from_pretrained(hf, dtype=torch.bfloat16, trust_remote_code=True).to(DEV).eval().requires_grad_(False)
LS = m.model.layers; NL = len(LS); got = {}; IDX = {}
order = [n for L in range(NL) for n in (f"op{L}", f"ffn{L}")]
def save(name):
    def f(mod, a, o):
        got[name] = o[0][IDX["i"]].float()
    return f
for L, b in enumerate(LS):
    b.post_operator_norm.register_forward_hook(save(f"op{L}")); b.post_ffn_norm.register_forward_hook(save(f"ffn{L}"))
    b.operator_norm.register_forward_pre_hook(lambda mod, a, L=L: got.__setitem__(f"x{L}", a[0][0][IDX["i"]].float()))
held = json.load(open(readf + ".held.json"))["held"]
head = lambda q: tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=True) + "\n"
s = {n: 0 for n in order}; e = {n: 0.0 for n in order}; dot = [0.0] * (len(order) - 1); xs = {L: 0 for L in range(NL)}; xe = {L: 0.0 for L in range(NL)}; cnt = 0
for j in held[:PAGES]:
    h = head(j["q"]); ids = tok(h + j["pre"], add_special_tokens=False).input_ids[:4000]; first = len(tok(h, add_special_tokens=False).input_ids)
    if len(ids) - first < 40:
        continue
    rng = np.random.default_rng(len(ids)); IDX["i"] = torch.tensor(sorted(rng.choice(np.arange(first, len(ids) - 1), size=min(WORDS, len(ids) - 1 - first), replace=False).tolist()), device=DEV)
    with torch.inference_mode():
        m.model(input_ids=torch.tensor([ids], device=DEV))
    for k, n in enumerate(order):
        s[n] = s[n] + got[n].double().sum(0); e[n] += float(got[n].pow(2).sum())
        if k + 1 < len(order):
            dot[k] += float((got[n] * got[order[k + 1]]).sum())
    for L in range(NL):
        xs[L] = xs[L] + got[f"x{L}"].double().sum(0); xe[L] += float(got[f"x{L}"].pow(2).sum())
    cnt += len(IDX["i"])
mean = {n: s[n] / cnt for n in order}
lines = [f"words {cnt}", "write  size  mean share  | cos of means to next | cos to next as is | cos to next, means out | stream mean share"]
for k, n in enumerate(order):
    E = e[n] / cnt; ms = float(mean[n].pow(2).sum()) / E
    L = int(n.lstrip("opfn"))
    sx = float((xs[L] / cnt).pow(2).sum()) / (xe[L] / cnt)
    if k + 1 < len(order):
        nx = order[k + 1]; En = e[nx] / cnt
        cm = float(torch.nn.functional.cosine_similarity(mean[n], mean[nx], dim=0)); raw = (dot[k] / cnt) / (E * En) ** .5
        cen = ((dot[k] / cnt) - float(mean[n] @ mean[nx])) / ((E - float(mean[n].pow(2).sum())) * (En - float(mean[nx].pow(2).sum()))) ** .5
        lines.append(f"{n:>6} {E**.5:6.2f} {ms:6.3f}   | {cm:+.3f} | {raw:+.3f} | {cen:+.3f} | {sx:.3f}")
    else:
        lines.append(f"{n:>6} {E**.5:6.2f} {ms:6.3f}   |   -   |   -   |   -   | {sx:.3f}")
open(rep, "w").write("\n".join(lines) + "\n"); print("\n".join(lines)); print("CANCEL_MEAN_DONE")
