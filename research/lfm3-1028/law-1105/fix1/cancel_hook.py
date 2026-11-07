#!/usr/bin/env python3
"""Step 2, LFM3 family. Hooks only, nothing saved to a model. Removes opposing directions from writes on held pages.
Part A: one pair at a time, r=64, does the later write's part along those directions shrink? Part B: many pairs at once, word drift, cancelling, neuron quiet.
usage: cancel_hook.py <hf> <read.pt> <eig.pt> <out.json> [pagesA=25 pagesB=40 words=64]"""
import json, sys, time
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
args = [a for a in sys.argv[1:] if "=" not in a]; opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
hf, readf, eigf, out = args
PA, PB, WORDS = int(opt.get("pagesA", 25)), int(opt.get("pagesB", 40)), int(opt.get("words", 64))
DEV = "cuda"
tok = AutoTokenizer.from_pretrained(hf, trust_remote_code=True)
m = AutoModelForCausalLM.from_pretrained(hf, dtype=torch.bfloat16, trust_remote_code=True).to(DEV).eval().requires_grad_(False)
LS = m.model.layers; NL = len(LS)
HD = m.lm_head.weight.float()
E = torch.load(eigf); U = E["U"].to(DEV); order = E["order"]; KK = len(order)
held = json.load(open(readf + ".held.json"))["held"]
head = lambda q: tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=True) + "\n"
pages = []
for j in held:
    h = head(j["q"]); ids = tok(h + j["pre"], add_special_tokens=False).input_ids[:4000]; first = len(tok(h, add_special_tokens=False).input_ids)
    if len(ids) - first < 40:
        continue
    rng = np.random.default_rng(len(ids)); pos = sorted(rng.choice(np.arange(first, len(ids) - 1), size=min(WORDS, len(ids) - 1 - first), replace=False).tolist())
    pages.append((torch.tensor([ids], device=DEV), torch.tensor(pos, device=DEV)))
print("held pages", len(pages), flush=True)
mods = {}
for L, b in enumerate(LS):
    mods[f"op{L}"] = b.post_operator_norm; mods[f"ffn{L}"] = b.post_ffn_norm
RM, got, IDX, NEU, rmsy = {}, {}, {}, {}, {}
def hook(name):
    def f(mod, a, o):
        v = o
        if name in RM:
            P = RM[name]; x = v[0].float(); x = x - (x @ P) @ P.T; v = x.to(o.dtype).unsqueeze(0)
        got[name] = v[0][IDX["i"]].float()
        return v
    return f
for n, mod in mods.items():
    mod.register_forward_hook(hook(n))
def neuh(L):
    def f(mod, a):
        NEU[L] = a[0][0][IDX["i"]].float()
    return f
def rmsh(L):
    def f(mod, a):
        rmsy[L] = a[0][0][IDX["i"]].float().pow(2).mean(-1, keepdim=True)
    return f
for L, b in enumerate(LS):
    b.feed_forward.w2.register_forward_pre_hook(neuh(L)); b.post_ffn_norm.register_forward_pre_hook(rmsh(L))
def run(ids, pos):
    IDX["i"] = pos
    with torch.inference_mode():
        hs = m.model(input_ids=ids).last_hidden_state[0][pos].float()
        lp = torch.log_softmax(hs @ HD.T, -1)
    return lp
def S_of():
    w = torch.stack([got[n] for n in order], 1)
    return float((w.sum(1).pow(2).sum(-1) / w.pow(2).sum(-1).sum(-1)).mean())
res = dict(A=[], B={}); t0 = time.time()
base_lp = []
print("Part A", flush=True)
for k in range(KK - 1):
    P = U[k][:, :64]; r0, r1, kl, same, n = 0.0, 0.0, 0.0, 0, 0
    for ids, pos in pages[:PA]:
        RM.clear(); lp0 = run(ids, pos); p0 = got[order[k + 1]] @ P
        RM[order[k]] = P; lp1 = run(ids, pos); p1 = got[order[k + 1]] @ P
        r0 += float(p0.pow(2).sum()); r1 += float(p1.pow(2).sum())
        kl += float((lp0.exp() * (lp0 - lp1)).sum(-1).sum()); same += int((lp0.argmax(-1) == lp1.argmax(-1)).sum()); n += len(pos)
    res["A"].append(dict(k=k, pair=order[k] + ">" + order[k + 1], later_ratio=(r1 / max(r0, 1e-12)) ** .5, kl=kl / n, same=same / n))
    print(res["A"][-1], round(time.time() - t0), "s", flush=True)
    json.dump(res, open(out, "w"), indent=1)
RM.clear()
print("Part B", flush=True)
conds = {}
allk = list(range(KK - 1)); opk = [k for k in allk if order[k].startswith("op")]; ffk = [k for k in allk if order[k].startswith("ffn")]
for r in (16, 64, 256):
    conds[f"all-r{r}"] = (allk, r, False)
conds["op>ffn-r64"] = (opk, 64, False); conds["ffn>op-r64"] = (ffk, 64, False); conds["op>ffn-r256"] = (opk, 256, False)
conds["all-r64-both"] = (allk, 64, True)
for name, (ks, r, both) in conds.items():
    kl, same, n, S0, S1 = 0.0, 0, 0, 0.0, 0.0
    e0 = {L: 0 for L in range(NL)}; e1 = {L: 0 for L in range(NL)}
    for ids, pos in pages[:PB]:
        RM.clear(); lp0 = run(ids, pos); S0 += S_of()
        for L in range(NL):
            e0[L] = e0[L] + (NEU[L].pow(2) / rmsy[L]).sum(0)
        for k in ks:
            RM[order[k]] = U[k][:, :r]
        if both:
            # also takes each pair's directions out of the later write, on top of its own
            ext = {}
            for k in ks:
                nm = order[k + 1]; ext.setdefault(nm, []).append(U[k][:, :r])
            for nm, L_ in ext.items():
                Q = torch.cat(L_ + ([RM[nm]] if nm in RM else []), 1); Q, _ = torch.linalg.qr(Q.double()); RM[nm] = Q.float()
        lp1 = run(ids, pos); S1 += S_of()
        for L in range(NL):
            e1[L] = e1[L] + (NEU[L].pow(2) / rmsy[L]).sum(0)
        kl += float((lp0.exp() * (lp0 - lp1)).sum(-1).sum()); same += int((lp0.argmax(-1) == lp1.argmax(-1)).sum()); n += len(pos)
    RM.clear()
    q = {}
    for L in range(NL):
        ratio = (e1[L] / e0[L].clamp(min=1e-9)); q[L] = dict(under50=float((ratio < .5).float().mean()), under25=float((ratio < .25).float().mean()), median=float(ratio.median()))
    res["B"][name] = dict(kl=kl / n, same=same / n, S0=S0 / min(PB, len(pages)), S1=S1 / min(PB, len(pages)), quiet=q)
    print(name, {a: b for a, b in res["B"][name].items() if a != "quiet"}, "under50 by layer", [round(q[L]["under50"], 3) for L in range(NL)], round(time.time() - t0), "s", flush=True)
    json.dump(res, open(out, "w"), indent=1)
print("CANCEL_HOOK_DONE", round(time.time() - t0), "s", flush=True)
