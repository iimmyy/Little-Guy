#!/usr/bin/env python3
"""Why did S not descend? Splits each layer by gate sign and weight sign. Reading only.
usage: gate_split.py <hf> <jobs.jsonl> <out.jsonl> [runs]"""
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
    W[L]["weak"] = F.cosine_similarity(W[L]["w2"].T, W[L]["w3"], dim=1) < 0
    W[L]["cl"] = ((W[L]["s1"] * W[L]["s2"])[None] * W[L]["w2"].T * W[L]["w3"]).sum(1)
    W[L]["cg"] = ((W[L]["s1"] * W[L]["s2"])[None] * W[L]["w2"].T * W[L]["w1"]).sum(1)
    b.ffn_norm.register_forward_pre_hook(lambda mod, a, L=L: got[L].__setitem__("x", a[0][0].float()))
    ff.register_forward_pre_hook(lambda mod, a, L=L: got[L].__setitem__("xt", a[0][0].float()))
    b.post_ffn_norm.register_forward_hook(lambda mod, a, o, L=L: got[L].__setitem__("d", o[0].float()))


def rms_norm(v, s, eps):
    r = (v.pow(2).mean(-1, keepdim=True) + eps).sqrt()
    return s * v / r, r


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
                    sg = torch.sigmoid(gt)
                    mm, mp = gt * sg, sg + gt * sg * (1 - sg)
                    op = gt > 0
                    wk = w["weak"]
                    a = mm * e
                    f = w["w2"] @ a
                    r1 = (x.pow(2).mean() + w["eps1"]).sqrt()
                    r2 = (f.pow(2).mean() + w["eps2"]).sqrt()
                    d = x.shape[0]
                    scale = 1.0 / (r1 * r2 * d)
                    lev = mm * w["cl"] * scale
                    gat = mp * e * w["cg"] * scale
                    fo_ = w["w2"][:, op] @ a[op]
                    fc_ = w["w2"][:, ~op] @ a[~op]
                    xt2 = rms_norm(x + dd, w["s1"], w["eps1"])[0]
                    e2 = w["w3"] @ xt2
                    row = dict(run=j["run"], id=j["id"], kind=kind, tok=i, L=L,
                               frac_open=float(op.float().mean()),
                               t_lev_os=float(lev[op & ~wk].sum()), t_lev_ow=float(lev[op & wk].sum()),
                               t_lev_cs=float(lev[~op & ~wk].sum()), t_lev_cw=float(lev[~op & wk].sum()),
                               t_gat_o=float(gat[op].sum()), t_gat_c=float(gat[~op].sum()),
                               Eo0=float((mm[op] * e[op] ** 2).sum()), Eo1=float((mm[op] * e2[op] ** 2).sum()),
                               Ec0=float((mm[~op].abs() * e[~op] ** 2).sum()), Ec1=float((mm[~op].abs() * e2[~op] ** 2).sum()),
                               cos_oc=float(F.cosine_similarity(fo_, fc_, dim=0)),
                               size_o=float(fo_.norm() / f.norm()), size_c=float(fc_.norm() / f.norm()))
                    fo.write(json.dumps(row) + "\n")
        fo.flush()
        print(k, RUNS, round(time.time() - t0), flush=True)
print("SPLIT_DONE", round(time.time() - t0), flush=True)
