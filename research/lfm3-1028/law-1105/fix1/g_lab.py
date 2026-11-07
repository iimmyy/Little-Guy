#!/usr/bin/env python3
"""Any model: drift of cut plans on held pages (rebuild vs plain by energy). usage: g_lab.py <arch> <hf> <steps.jsonl> <plans dir> <out.json> [sizes=30,50 pages=30 words=64]"""
import json, sys, time
import numpy as np, torch
import g_arch
args = [a for a in sys.argv[1:] if "=" not in a]; opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
arch, hf, stepsf, plans, out = args; SIZES = [int(x) for x in opt.get("sizes", "30,50").split(",")]; PAGES, WORDS = int(opt.get("pages", 30)), int(opt.get("words", 64))
DEV = "cuda"
m, tok, body, head, cap = g_arch.load(arch, hf); LS = body.layers
fit, held = g_arch.pages(arch, stepsf, ["screen-sci-40.jsonl", "screen-math-40.jsonl"])
pg = []
for q, text, _ in held[:PAGES]:
    h = head(q); ids = tok(h + text, add_special_tokens=False).input_ids[:4000]; first = len(tok(h, add_special_tokens=False).input_ids)
    if len(ids) - first < 40:
        continue
    rng = np.random.default_rng(len(ids)); pg.append((torch.tensor([ids], device=DEV), torch.tensor(sorted(rng.choice(np.arange(first, len(ids) - 1), size=min(WORDS, len(ids) - 1 - first), replace=False).tolist()), device=DEV)))
ORIG = {L: g_arch.w2(arch, b).weight.data.clone() for L, b in enumerate(LS)}
def run():
    with torch.inference_mode():
        return [g_arch.logits(m, body, cap, ids, pos).cpu() for ids, pos in pg]
base = run(); n = sum(len(x) for x in base); res = {}
for s in SIZES:
    P = torch.load(f"{plans}/cutplan-{s}.pt")
    for arm in ("rebuild", "plain"):
        for L, b in enumerate(LS):
            w = ORIG[L].clone()
            if arm == "rebuild":
                K = P[L]["keep"].to(DEV); w[:] = 0; w[:, K] = P[L]["w2"].to(DEV).to(w.dtype)
            else:
                mask = torch.ones(w.shape[1], dtype=torch.bool, device=DEV); mask[P[L]["energy_keep"].to(DEV)] = False; w[:, mask] = 0
            g_arch.w2(arch, b).weight.data.copy_(w)
        a = run()
        kl = sum(float((x.exp() * (x - y)).sum()) for x, y in zip(base, a)) / n; same = sum(int((x.argmax(-1) == y.argmax(-1)).sum()) for x, y in zip(base, a)) / n
        res[f"{arm}-{s}"] = dict(kl=kl, same=same); print(arm, s, round(kl, 4), round(same, 4), flush=True); json.dump(res, open(out, "w"), indent=1)
print("G_LAB_DONE", flush=True)
