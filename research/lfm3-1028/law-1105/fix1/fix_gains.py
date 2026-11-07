#!/usr/bin/env python3
"""Every LFM3 write is D*y/rms(y). The file build folds D into the output matrix (y2 = (I-P) D y) and leaves a flat post norm a, because some D are near zero and can't be inverted.
This reads, per write, the average rms(y2)/rms(y) on held pages, which is the flat post norm that keeps the remaining write the same size, and how much it wobbles per word. usage: fix_gains.py <hf> <read.pt> <eig.pt> <r> <writers: all|op|ffn> <out.json> [pages=20]"""
import json, sys
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
args = [a for a in sys.argv[1:] if "=" not in a]; opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
hf, readf, eigf, r, which, out = args; r = int(r); PAGES = int(opt.get("pages", 20))
DEV = "cuda"
tok = AutoTokenizer.from_pretrained(hf, trust_remote_code=True)
m = AutoModelForCausalLM.from_pretrained(hf, dtype=torch.bfloat16, trust_remote_code=True).to(DEV).eval().requires_grad_(False)
E = torch.load(eigf); U = E["U"].to(DEV); order = E["order"]
held = json.load(open(readf + ".held.json"))["held"]
head = lambda q: tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=True) + "\n"
sum_, cnt, sq = {}, {}, {}
mods = {}
for L, b in enumerate(m.model.layers):
    mods[f"op{L}"] = b.post_operator_norm; mods[f"ffn{L}"] = b.post_ffn_norm
def hook(name, k, a_):
    P = U[k][:, :r]
    def f(mod, a):
        y = a[0][0].float(); Dy = y * a_
        y2 = Dy - (Dy @ P) @ P.T
        q = (y2.pow(2).mean(-1) / y.pow(2).mean(-1).clamp(min=1e-12)).sqrt()
        sum_[name] = sum_.get(name, 0.0) + float(q.sum()); cnt[name] = cnt.get(name, 0) + len(q); sq[name] = sq.get(name, 0.0) + float(q.pow(2).sum())
    return f
mins = {}
for k, name in enumerate(order[:-1]):
    if which != "all" and not name.startswith(which):
        continue
    a_ = mods[name].weight.float().detach()
    mins[name] = float(a_.abs().min())
    mods[name].register_forward_pre_hook(hook(name, k, a_))
for j in held[:PAGES]:
    ids = tok(head(j["q"]) + j["pre"], add_special_tokens=False).input_ids[:4000]
    with torch.inference_mode():
        m.model(input_ids=torch.tensor([ids], device=DEV))
g = {k: sum_[k] / cnt[k] for k in sum_}
wob = {k: max(sq[k] / cnt[k] - g[k] ** 2, 0) ** .5 / g[k] for k in g}
json.dump(dict(r=r, which=which, gain=g, wobble=wob, min_abs_D=mins), open(out, "w"), indent=1)
print("FIX_GAINS_DONE gains", {k: round(v, 3) for k, v in list(g.items())[:6]}, "wobble median", round(sorted(wob.values())[len(wob) // 2], 3), "max", round(max(wob.values()), 3), flush=True)
