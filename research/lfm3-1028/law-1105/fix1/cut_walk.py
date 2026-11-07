#!/usr/bin/env python3
"""Break, then rebuild layer by layer. Starts from a cut plan (neurons removed, kept writes re-solved on the uncut model's inputs), then walks up in blocks:
each block's kept neurons, and its operator's output matrix, are re-solved by least squares so that, given the CUT model's own inputs, they write what the uncut model wrote.
No gradient steps. usage: cut_walk.py <hf> <steps.jsonl> <read.pt> <plan.pt> <out.pt> [pages=80 words=400 maxtok=2500 blocks=4 ridge=1e-4 ops=1]"""
import json, sys, time, random
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
args = [a for a in sys.argv[1:] if "=" not in a]; opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
hf, stepsf, readf, planf, out = args
PAGES, WORDS, MAXTOK, NB, RIDGE, OPS = int(opt.get("pages", 80)), int(opt.get("words", 400)), int(opt.get("maxtok", 2500)), int(opt.get("blocks", 4)), float(opt.get("ridge", 1e-4)), int(opt.get("ops", 1))
DEV = "cuda"
tok = AutoTokenizer.from_pretrained(hf, trust_remote_code=True)
O = AutoModelForCausalLM.from_pretrained(hf, dtype=torch.bfloat16, trust_remote_code=True).to(DEV).eval().requires_grad_(False)
C = AutoModelForCausalLM.from_pretrained(hf, dtype=torch.bfloat16, trust_remote_code=True).to(DEV).eval().requires_grad_(False)
NL = len(C.model.layers); PL = torch.load(planf)
KEEP = {L: PL[L]["keep"].to(DEV) for L in range(NL)}
for L, b in enumerate(C.model.layers):
    w = b.feed_forward.w2.weight.data; w.zero_(); w[:, KEEP[L]] = PL[L]["w2"].to(DEV).to(w.dtype)
def outmat(b):
    return b.self_attn.o_proj if hasattr(b, "self_attn") else b.conv.out_proj
IDX, T = {}, {}
for L, b in enumerate(O.model.layers):
    b.post_ffn_norm.register_forward_pre_hook(lambda m_, a, L=L: T.__setitem__(("of", L), a[0][0][IDX["i"]].float()))
    b.post_operator_norm.register_forward_pre_hook(lambda m_, a, L=L: T.__setitem__(("oo", L), a[0][0][IDX["i"]].float()))
for L, b in enumerate(C.model.layers):
    b.feed_forward.w2.register_forward_pre_hook(lambda m_, a, L=L: T.__setitem__(("ch", L), a[0][0][IDX["i"]][:, KEEP[L]].float()))
    b.post_ffn_norm.register_forward_pre_hook(lambda m_, a, L=L: T.__setitem__(("cf", L), a[0][0][IDX["i"]].float().pow(2).mean(-1, keepdim=True).sqrt()))
    outmat(b).register_forward_pre_hook(lambda m_, a, L=L: T.__setitem__(("cz", L), a[0][0][IDX["i"]].float()))
    b.post_operator_norm.register_forward_pre_hook(lambda m_, a, L=L: T.__setitem__(("co", L), a[0][0][IDX["i"]].float().pow(2).mean(-1, keepdim=True).sqrt()))
heldq = [j["q"] for j in json.load(open(readf + ".held.json"))["held"]]
screen = set()
for f in ("screen-sci-40.jsonl", "screen-math-40.jsonl"):
    screen |= {json.loads(l)["q"] for l in open(f)}
byq = {}
for l in open(stepsf):
    j = json.loads(l)
    if j["q"] in screen or j["q"] in set(heldq):
        continue
    if j["q"] not in byq or len(j["pre"]) > len(byq[j["q"]]["pre"]):
        byq[j["q"]] = j
J = sorted(byq.values(), key=lambda j: j["id"]); random.Random(1108).shuffle(J)
head = lambda q: tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=True) + "\n"
def page(j, seed, words, maxtok):
    h = head(j["q"]); ids = tok(h + j["pre"], add_special_tokens=False).input_ids[:maxtok]; first = len(tok(h, add_special_tokens=False).input_ids)
    if len(ids) - first < 40:
        return None
    rng = np.random.default_rng(seed); pos = sorted(set(range(first)) | set(rng.choice(np.arange(first, len(ids) - 1), size=min(words, len(ids) - 1 - first), replace=False).tolist()))
    return torch.tensor([ids], device=DEV), torch.tensor(pos, device=DEV)
fit = [p for p in (page(j, i, WORDS, MAXTOK) for i, j in enumerate(J[:PAGES])) if p is not None]
HD = O.lm_head.weight.float()
def held_eval(tag):
    kl, same, n = 0.0, 0, 0
    for j in json.load(open(readf + ".held.json"))["held"]:
        p = page(j, len(j["pre"]), 64, 4000)
        if p is None:
            continue
        ids, pos = p; pos = pos[pos >= len(tok(head(j["q"]), add_special_tokens=False).input_ids)]; IDX["i"] = pos
        with torch.inference_mode():
            a_ = torch.log_softmax(O.model(input_ids=ids).last_hidden_state[0][pos].float() @ HD.T, -1)
            c_ = torch.log_softmax(C.model(input_ids=ids).last_hidden_state[0][pos].float() @ HD.T, -1)
        kl += float((a_.exp() * (a_ - c_)).sum()); same += int((a_.argmax(-1) == c_.argmax(-1)).sum()); n += len(pos)
    print("HELD", tag, round(kl / n, 4), round(same / n, 4), flush=True)
held_eval("static")
blocks = np.array_split(np.arange(NL), NB); t0 = time.time()
for bl in blocks:
    acc = {}
    for ids, pos in fit:
        IDX["i"] = pos
        with torch.inference_mode():
            O.model(input_ids=ids); C.model(input_ids=ids)
            for L in bl:
                L = int(L)
                v = T[("ch", L)] / T[("cf", L)]; u = T[("of", L)] / T[("of", L)].pow(2).mean(-1, keepdim=True).sqrt()
                g, x = acc.get(("f", L), (0, 0)); acc[("f", L)] = (g + (v.T @ v).double(), x + (u.T @ v).double())
                if OPS:
                    z = T[("cz", L)] / T[("co", L)]; uo = T[("oo", L)] / T[("oo", L)].pow(2).mean(-1, keepdim=True).sqrt()
                    g, x = acc.get(("o", L), (0, 0)); acc[("o", L)] = (g + (z.T @ z).double(), x + (uo.T @ z).double())
    for (kind, L), (g, x) in acc.items():
        lam = RIDGE * g.diag().mean(); W = torch.linalg.solve(g + lam * torch.eye(g.shape[0], device=DEV, dtype=g.dtype), x.T).T
        b = C.model.layers[L]
        if kind == "f":
            w = b.feed_forward.w2.weight.data; w[:, KEEP[L]] = W.to(w.dtype)
        else:
            outmat(b).weight.data.copy_(W.to(outmat(b).weight.dtype))
    print("block", list(map(int, bl)), "solved", round(time.time() - t0), "s", flush=True)
    held_eval("after block " + str(int(bl[-1])))
res = {L: dict(keep=PL[L]["keep"], w2=C.model.layers[L].feed_forward.w2.weight.data[:, KEEP[L]].cpu().clone(), op=outmat(C.model.layers[L]).weight.data.cpu().clone() if OPS else None,
               opname=("attn_output" if hasattr(C.model.layers[L], "self_attn") else "shortconv.out_proj")) for L in range(NL)}
torch.save(res, out); print("WALK_SAVED", flush=True)

print("CUT_WALK_DONE", round(time.time() - t0), "s", flush=True)
