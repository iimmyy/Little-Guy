#!/usr/bin/env python3
"""Drift of neuron cuts on held pages. Applies a plan in memory by zeroing dropped neurons' writes, so no file is built. Arms: rebuild (re-solved writes), plain (kept by energy, writes untouched), plan-plain (the rebuild's neuron choice, writes untouched).
usage: cut_lab.py <hf> <read.pt> <plans dir> <out.json> [sizes=10,20,30,50 pages=42 words=64]"""
import json, sys, time
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
args = [a for a in sys.argv[1:] if "=" not in a]; opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
hf, readf, plans, out = args; SIZES = [int(x) for x in opt.get("sizes", "10,20,30,50").split(",")]; PAGES, WORDS = int(opt.get("pages", 42)), int(opt.get("words", 64))
DEV = "cuda"
tok = AutoTokenizer.from_pretrained(hf, trust_remote_code=True)
m = AutoModelForCausalLM.from_pretrained(hf, dtype=torch.bfloat16, trust_remote_code=True).to(DEV).eval().requires_grad_(False)
if "state" in opt:
    print(m.load_state_dict(torch.load(opt["state"]), strict=False), flush=True)
LS = m.model.layers; HD = m.lm_head.weight.float(); ORIG = {L: b.feed_forward.w2.weight.data.clone() for L, b in enumerate(LS)}
held = json.load(open(readf + ".held.json"))["held"]
head = lambda q: tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=True) + "\n"
pages = []
for j in held[:PAGES]:
    h = head(j["q"]); ids = tok(h + j["pre"], add_special_tokens=False).input_ids[:4000]; first = len(tok(h, add_special_tokens=False).input_ids)
    if len(ids) - first < 40:
        continue
    rng = np.random.default_rng(len(ids)); pos = sorted(rng.choice(np.arange(first, len(ids) - 1), size=min(WORDS, len(ids) - 1 - first), replace=False).tolist())
    pages.append((torch.tensor([ids], device=DEV), torch.tensor(pos, device=DEV)))
def lp():
    r = []
    for ids, pos in pages:
        with torch.inference_mode():
            hs = m.model(input_ids=ids).last_hidden_state[0][pos].float()
            r.append(torch.log_softmax(hs @ HD.T, -1).cpu())
    return r
MU = torch.load(opt.get("gram", plans.replace("plans-plain", "gram-plain.pt")), mmap=True)["mu"]
base = lp(); n = sum(len(x) for x in base); res = {}; t0 = time.time()
def score(name, a):
    kl = sum(float((b.exp() * (b - c)).sum()) for b, c in zip(base, a)) / n
    same = sum(int((b.argmax(-1) == c.argmax(-1)).sum()) for b, c in zip(base, a)) / n
    res[name] = dict(kl=kl, same=same); print(name, round(kl, 4), round(same, 4), round(time.time() - t0), "s", flush=True)
    json.dump(res, open(out, "w"), indent=1)
for s in SIZES:
    P = torch.load(f"{plans}/cutplan-{s}.pt")
    for arm in ("rebuild", "plain", "plan-plain", "plain+dc", "plan-plain+dc"):
        for L, b in enumerate(LS):
            w = ORIG[L].clone()
            if arm == "rebuild":
                K = P[L]["keep"].to(DEV); w[:] = 0; w[:, K] = P[L]["w2"].to(DEV).to(w.dtype)
            else:
                K = (P[L]["energy_keep"] if arm.startswith("plain") else P[L]["keep"]).to(DEV)
                mask = torch.ones(w.shape[1], dtype=torch.bool, device=DEV); mask[K] = False
                if arm.endswith("+dc"):
                    # keep the constant: a one-vector change to the kept writes so the average write matches the uncut one
                    mu = MU[L].to(DEV).float(); wf = w.float(); c = wf @ mu; muK = mu.clone(); muK[mask] = 0
                    wf[:, mask] = 0; wf = wf + torch.outer(c - wf @ muK, muK) / muK.pow(2).sum(); w = wf.to(w.dtype)
                else:
                    w[:, mask] = 0
            b.feed_forward.w2.weight.data.copy_(w)
        score(f"{arm}-{s}", lp())
print("CUT_LAB_DONE", flush=True)
