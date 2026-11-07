#!/usr/bin/env python3
"""Any model: Immy's lesion test, by drift on held pages. Remove one whole layer (skip it), or only its operator write, or only its neuron write.
usage: g_lesion.py <arch> <hf> <steps.jsonl> <out.json> [pages=25 words=64 modes=layer,op,ffn fill=zero|mean fitpages=40]
fill=mean swaps a removed write for its own average write (fit pages, every position after the first) instead of zero."""
import json, sys, time
import numpy as np, torch
import g_arch
args = [a for a in sys.argv[1:] if "=" not in a]; opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
arch, hf, stepsf, out = args; PAGES, WORDS = int(opt.get("pages", 25)), int(opt.get("words", 64)); MODES = opt.get("modes", "layer,op,ffn").split(",")
DEV = "cuda"
m, tok, body, head, cap = g_arch.load(arch, hf); LS = body.layers; NL = len(LS)
fit, held = g_arch.pages(arch, stepsf, ["screen-sci-40.jsonl", "screen-math-40.jsonl"])
pg = []
for q, text, _ in held[:PAGES]:
    h = head(q); ids = tok(h + text, add_special_tokens=False).input_ids[:4000]; first = len(tok(h, add_special_tokens=False).input_ids)
    if len(ids) - first < 40:
        continue
    rng = np.random.default_rng(len(ids)); pg.append((torch.tensor([ids], device=DEV), torch.tensor(sorted(rng.choice(np.arange(first, len(ids) - 1), size=min(WORDS, len(ids) - 1 - first), replace=False).tolist()), device=DEV)))
def run():
    with torch.inference_mode():
        return [g_arch.logits(m, body, cap, ids, pos).cpu() for ids, pos in pg]
FILL = opt.get("fill", "zero"); MEAN = {}
def postmod(tag, b):
    return g_arch.post(arch, b) if tag == "ffn" else (b.post_operator_norm if arch == "lfm3" else b.post_attention_layernorm)
if FILL == "mean":
    acc = {}; cnt = [0]
    def grab(key):
        def f(mod, a, o):
            acc[key] = acc.get(key, 0) + o[0, 1:].double().sum(0)
        return f
    hs = [postmod(t, b).register_forward_hook(grab((t, L))) for L, b in enumerate(LS) for t in ("op", "ffn")]
    with torch.inference_mode():
        for q, text, _ in fit[:int(opt.get("fitpages", 40))]:
            ids = tok(head(q) + text, add_special_tokens=False).input_ids[:2000]
            body(input_ids=torch.tensor([ids], device=DEV)); cnt[0] += len(ids) - 1
    [h.remove() for h in hs]
    MEAN = {k: (v / cnt[0]).to(torch.bfloat16) for k, v in acc.items()}
    print("mean writes from", cnt[0], "positions", flush=True)
def fill(tag, L, o):
    return MEAN[(tag, L)].to(o.dtype).expand_as(o).clone() if FILL == "mean" else torch.zeros_like(o)
base = run(); n = sum(len(x) for x in base); res = {}; t0 = time.time()
ACT = {}
def layer_hook(L):
    def f(mod, a, kw, o):
        if ACT.get("layer") == L:
            x = a[0] if a else kw["hidden_states"]
            return (x,) + tuple(o[1:]) if isinstance(o, tuple) else x
        return o
    return f
def zero_hook(tag, L):
    def f(mod, a, o):
        if ACT.get(tag) == L:
            return fill(tag, L, o)
        return o
    return f
for L, b in enumerate(LS):
    b.register_forward_hook(layer_hook(L), with_kwargs=True)
    (b.post_operator_norm if arch == "lfm3" else b.post_attention_layernorm).register_forward_hook(zero_hook("op", L))
    g_arch.post(arch, b).register_forward_hook(zero_hook("ffn", L))
if "combo" in opt:
    tag, ls = opt["combo"].split(":"); ls = [int(x) for x in ls.split(",")]
    tot = sum(p.numel() for nm, p in body.named_parameters() if p.dim() == 2 and "embed" not in nm) + m.lm_head.weight.numel()
    neu = sum(g_arch.w2(arch, LS[L]).weight.numel() * 3 for L in ls); rec = dict(combo=opt["combo"], share=neu / tot, total=tot)
    for how in (["zero", "mean"] if FILL == "mean" else ["zero"]):
        hooks = [postmod(tag, LS[L]).register_forward_hook((lambda L_: lambda mod_, a, o: torch.zeros_like(o) if how == "zero" else MEAN[(tag, L_)].to(o.dtype).expand_as(o).clone())(L)) for L in ls]
        a = run(); [h.remove() for h in hooks]
        kl = sum(float((x.exp() * (x - y)).sum()) for x, y in zip(base, a)) / n; same = sum(int((x.argmax(-1) == y.argmax(-1)).sum()) for x, y in zip(base, a)) / n
        rec[how] = dict(kl=kl, same=same)
        print("COMBO", opt["combo"], how, "kl", round(kl, 4), "same", round(same, 4), "math share removed", round(neu / tot, 4), "of", tot, flush=True)
    json.dump(rec, open(out, "w"), indent=1); MODES = []
for mode in MODES:
    for L in range(NL):
        ACT.clear(); ACT[mode] = L; a = run(); ACT.clear()
        kl = sum(float((x.exp() * (x - y)).sum()) for x, y in zip(base, a)) / n; same = sum(int((x.argmax(-1) == y.argmax(-1)).sum()) for x, y in zip(base, a)) / n
        res[f"{mode}{L}"] = dict(kl=kl, same=same); print(mode, L, round(kl, 4), round(same, 4), round(time.time() - t0), "s", flush=True)
        json.dump(res, open(out, "w"), indent=1)
print("G_LESION_DONE", flush=True)
