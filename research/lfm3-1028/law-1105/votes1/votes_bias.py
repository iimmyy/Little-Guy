#!/usr/bin/env python3
"""Splits each vote into the part that is the same for every problem and the part that depends on this state. Reading only.
usage: votes_bias.py <hf> <steps.jsonl> <rows.jsonl> <out prefix>"""
import json, sys, time
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

hf, steps, rowsf, prefix = sys.argv[1:5]
DEV = "cuda"
tok = AutoTokenizer.from_pretrained(hf, trust_remote_code=True)
m = AutoModelForCausalLM.from_pretrained(hf, dtype=torch.bfloat16, trust_remote_code=True).to(DEV).eval().requires_grad_(False)
LS = m.model.layers
N = len(LS)
U = m.lm_head.weight.float()
c = m.model.embedding_norm.weight.float()
eps = m.model.embedding_norm.variance_epsilon
IDX, got = {}, {}


def save(name):
    def f(mod, a, o=None):
        v = a[0] if o is None else o
        got[name] = v[0][IDX["i"]].float()
    return f


LS[0].operator_norm.register_forward_pre_hook(save("x0"))
m.model.embedding_norm.register_forward_pre_hook(save("xN"))
for L, b in enumerate(LS):
    b.post_operator_norm.register_forward_hook(save(f"op{L}"))
    b.post_ffn_norm.register_forward_hook(save(f"ffn{L}"))
order = ["x0"] + [n_ for L in range(N) for n_ in (f"op{L}", f"ffn{L}")]
ID = {}
for i in range(U.shape[0]):
    s = tok.decode([i])
    if s.isdigit() and s.isascii():
        ID[s] = i
rows = [json.loads(l) for l in open(rowsf)]
want = {r["step"] for r in rows}
J = {}
for l in open(steps):
    j = json.loads(l)
    if j["step"] in want:
        J[j["step"]] = j
print("spots", len(rows), "steps", len(J), flush=True)
W_all, D_all, R_all, V_all = {}, {}, {}, {}
t0 = time.time()
by = {}
for n, r in enumerate(rows):
    by.setdefault(r["step"], []).append((n, r))
for c_, (s, lst) in enumerate(by.items()):
    j = J[s]
    h = tok.apply_chat_template([{"role": "user", "content": j["q"]}], tokenize=False, add_generation_prompt=True, enable_thinking=True) + "\n"
    ids = tok(h + j["pre"] + j["number"], add_special_tokens=False).input_ids
    IDX["i"] = torch.tensor([r["q"] - 1 for _, r in lst], device=DEV)
    with torch.inference_mode():
        m.model(input_ids=torch.tensor([ids], device=DEV))
        W = torch.stack([got[n_] for n_ in order], 1)
        rr = torch.sqrt(got["xN"].pow(2).mean(-1, keepdim=True) + eps)
    for q, (n, r) in enumerate(lst):
        d = c * (U[ID[r["a"]]] - U[ID[r["b"]]])
        W_all[n], D_all[n], R_all[n] = W[q].cpu(), d.cpu(), float(rr[q])
    if c_ % 100 == 0:
        print(c_, len(by), round(time.time() - t0), flush=True)
n = len(rows)
W = torch.stack([W_all[i] for i in range(n)])
D = torch.stack([D_all[i] for i in range(n)])
R = torch.tensor([R_all[i] for i in range(n)])
build = torch.tensor([r["split"] == "build" for r in rows])
mean_w = W[build].mean(0)
Vbias = (mean_w[None] @ D[:, :, None])[:, :, 0] / R[:, None]
V = (W @ D[:, :, None])[:, :, 0] / R[:, None]
np.savez(prefix + "-x.npz", V=V.numpy(), Vbias=Vbias.numpy(), order=np.array(order), usize=np.array([[float(U[ID[r["a"]]].norm()), float(U[ID[r["b"]]].norm())] for r in rows], dtype=np.float32))
print("BIAS_DONE spots", n, round(time.time() - t0), flush=True)
