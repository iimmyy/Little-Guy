#!/usr/bin/env python3
"""Is closed-gate energy a real descent, or just the norm? Reading only.
usage: closed_read.py <hf> <jobs.jsonl> <out.jsonl> [runs]"""
import json, random, sys, time
import numpy as np, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

hf, jobs, out = sys.argv[1:4]
RUNS = int(sys.argv[4]) if len(sys.argv) > 4 else 40
DEV = "cuda"
tok = AutoTokenizer.from_pretrained(hf, trust_remote_code=True)
m = AutoModelForCausalLM.from_pretrained(hf, dtype=torch.bfloat16, trust_remote_code=True).to(DEV).eval().requires_grad_(False)
LS = m.model.layers
N = len(LS)
got = {L: {} for L in range(N)}
W = {}
for L, b in enumerate(LS):
    ff = b.feed_forward
    W[L] = dict(w1=ff.w1.weight.float(), w2=ff.w2.weight.float(), w3=ff.w3.weight.float(),
                s1=b.ffn_norm.weight.float(), s2=b.post_ffn_norm.weight.float(),
                eps1=float(b.ffn_norm.variance_epsilon), eps2=float(b.post_ffn_norm.variance_epsilon))
    b.ffn_norm.register_forward_pre_hook(lambda mod, a, L=L: got[L].__setitem__("x", a[0][0].float()))
    ff.register_forward_pre_hook(lambda mod, a, L=L: got[L].__setitem__("xt", a[0][0].float()))
    b.post_ffn_norm.register_forward_hook(lambda mod, a, o, L=L: got[L].__setitem__("d", o[0].float()))


def rms_norm(v, s, eps):
    r = (v.pow(2).mean(-1, keepdim=True) + eps).sqrt()
    return s * v / r


random.seed(0)
J0 = [json.loads(l) for l in open(jobs)]
random.shuffle(J0)
J0 = J0[:RUNS]
t0 = time.time()
with open(out, "w") as fo:
    for k, j in enumerate(J0):
        head = tok.apply_chat_template([{"role": "user", "content": j["q"]}], tokenize=False, add_generation_prompt=True, enable_thinking=True) + "\n"
        close = max(p["c"] for p in j["pos"])
        th = j["thinking"][:close]
        enc = tok(head + th, add_special_tokens=False, return_offsets_mapping=True)
        ends = np.array([b for a, b in enc.offset_mapping])
        n0 = len(head)
        first = int(np.searchsorted(ends, n0, side="right"))
        last = int(np.searchsorted(ends, n0 + close, side="right")) - 1
        rng = random.Random(1000 + k)
        picks = [("close", last)] + [("word", rng.randrange(first, last)) for _ in range(2)] if last > first + 2 else [("close", last)]
        with torch.inference_mode():
            m.model(input_ids=torch.tensor([enc.input_ids], device=DEV))
            for kind, i in picks:
                for L in range(N):
                    w, g = W[L], got[L]
                    x, xt, dd = g["x"][i], g["xt"][i], g["d"][i]
                    gt, e = w["w1"] @ xt, w["w3"] @ xt
                    mm = F.silu(gt)
                    op = gt > 0
                    a = mm * e
                    torch.manual_seed(L * 1000 + i)
                    dp = rms_norm(w["w2"][:, torch.randperm(w["w2"].shape[1], device=DEV)] @ a, w["s2"], w["eps2"])
                    xt1 = rms_norm(x + dd, w["s1"], w["eps1"])
                    xtp = rms_norm(x + dp, w["s1"], w["eps1"])
                    e1, ep = w["w3"] @ xt1, w["w3"] @ xtp
                    en = lambda v, mask: float((mm[mask].abs() * v[mask] ** 2).sum()) / 2
                    r1 = (x.pow(2).mean() + w["eps1"]).sqrt()
                    step = w["s1"] * dd / r1
                    stepp = w["s1"] * dp / r1
                    gc = w["w3"][~op].T @ (mm[~op].abs() * e[~op])
                    go = w["w3"][op].T @ (mm[op].abs() * e[op])
                    cos = lambda p, q: float(F.cosine_similarity(p, q, dim=0))
                    fo.write(json.dumps(dict(
                        run=j["run"], id=j["id"], kind=kind, tok=i, L=L, frac_closed=float((~op).float().mean()),
                        Ec0=en(e, ~op), Ec1=en(e1, ~op), Ecp=en(ep, ~op), Eo0=en(e, op), Eo1=en(e1, op), Eop=en(ep, op),
                        Ac=cos(step, -gc), Acp=cos(stepp, -gc), Ao=cos(step, go), Aop=cos(stepp, go),
                        rr=float((x + dd).pow(2).mean().sqrt() / x.pow(2).mean().sqrt()),
                    )) + "\n")
        fo.flush()
print("CLOSED_DONE", round(time.time() - t0), flush=True)
