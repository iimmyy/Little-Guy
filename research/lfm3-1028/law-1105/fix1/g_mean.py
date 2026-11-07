#!/usr/bin/env python3
"""Any model: is the cancelling between neighbouring writes a constant? Same reading as cancel_mean.py. E4B writes: attention, neurons, per-layer input; the stream is scaled by a layer scalar after each layer (cosines don't care).
usage: g_mean.py <arch> <hf> <steps.jsonl> <report.txt> [pages=40 words=128]"""
import sys
import numpy as np, torch
import g_arch
args = [a for a in sys.argv[1:] if "=" not in a]; opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
arch, hf, stepsf, rep = args; PAGES, WORDS = int(opt.get("pages", 40)), int(opt.get("words", 128))
DEV = "cuda"
m, tok, body, head, cap = g_arch.load(arch, hf); LS = body.layers; NL = len(LS); got = {}; IDX = {}
order = []
for L, b in enumerate(LS):
    mods = [("op", b.post_operator_norm if arch == "lfm3" else b.post_attention_layernorm), ("ffn", g_arch.post(arch, b))]
    if arch != "lfm3" and getattr(b, "post_per_layer_input_norm", None) is not None:
        mods.append(("per", b.post_per_layer_input_norm))
    for nm, mod in mods:
        mod.register_forward_hook(lambda mod_, a, o, k=f"{nm}{L}": got.__setitem__(k, o[0][IDX["i"]].float())); order.append(f"{nm}{L}")
fit, held = g_arch.pages(arch, stepsf, ["screen-sci-40.jsonl", "screen-math-40.jsonl"])
s = {n: 0 for n in order}; e = {n: 0.0 for n in order}; dot = [0.0] * (len(order) - 1); cnt = 0
for q, text, _ in held[:PAGES]:
    h = head(q); ids = tok(h + text, add_special_tokens=False).input_ids[:4000]; first = len(tok(h, add_special_tokens=False).input_ids)
    if len(ids) - first < 40:
        continue
    rng = np.random.default_rng(len(ids)); IDX["i"] = torch.tensor(sorted(rng.choice(np.arange(first, len(ids) - 1), size=min(WORDS, len(ids) - 1 - first), replace=False).tolist()), device=DEV)
    with torch.inference_mode():
        body(input_ids=torch.tensor([ids], device=DEV))
    for k, n in enumerate(order):
        s[n] = s[n] + got[n].double().sum(0); e[n] += float(got[n].pow(2).sum())
        if k + 1 < len(order):
            dot[k] += float((got[n] * got[order[k + 1]]).sum())
    cnt += len(IDX["i"])
mean = {n: s[n] / cnt for n in order}
lines = [f"arch {arch} words {cnt}", "write  size  mean share | cos of means to next | cos to next as is | cos to next, means out"]
for k, n in enumerate(order[:-1]):
    nx = order[k + 1]; E = e[n] / cnt; En = e[nx] / cnt; mn, mx = mean[n], mean[nx]
    cm = float(torch.nn.functional.cosine_similarity(mn, mx, dim=0)); raw = (dot[k] / cnt) / (E * En) ** .5
    cen = ((dot[k] / cnt) - float(mn @ mx)) / max((E - float(mn.pow(2).sum())) * (En - float(mx.pow(2).sum())), 1e-30) ** .5
    lines.append(f"{n:>6}>{nx:<6} {E**.5:7.3f} {float(mn.pow(2).sum())/E:6.3f} | {cm:+.3f} | {raw:+.3f} | {cen:+.3f}")
open(rep, "w").write("\n".join(lines) + "\n"); print("\n".join(lines)); print("G_MEAN_DONE", flush=True)
