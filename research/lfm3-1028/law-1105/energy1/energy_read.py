#!/usr/bin/env python3
"""Is S = 1/2 m e^2 a real energy in the running model? Reading only.
usage: energy_read.py <hf> <jobs.jsonl> <out.jsonl> [runs]"""
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
    W[L] = dict(
        w1=ff.w1.weight.float(), w2=ff.w2.weight.float(), w3=ff.w3.weight.float(),
        s1=b.ffn_norm.weight.float(), s2=b.post_ffn_norm.weight.float(),
        eps1=float(b.ffn_norm.variance_epsilon), eps2=float(b.post_ffn_norm.variance_epsilon),
    )
    W[L]["weak"] = F.cosine_similarity(W[L]["w2"].T, W[L]["w3"], dim=1) < 0
    b.ffn_norm.register_forward_pre_hook(lambda mod, a, L=L: got[L].__setitem__("x", a[0][0].float()))
    ff.register_forward_pre_hook(lambda mod, a, L=L: got[L].__setitem__("xt", a[0][0].float()))
    b.post_ffn_norm.register_forward_pre_hook(lambda mod, a, L=L: got[L].__setitem__("f", a[0][0].float()))
    b.post_ffn_norm.register_forward_hook(lambda mod, a, o, L=L: got[L].__setitem__("d", o[0].float()))


def rms_norm(v, s, eps):
    r = (v.pow(2).mean(-1, keepdim=True) + eps).sqrt()
    return s * v / r, r


def jacobian(w, x, xt, dtype=torch.float32):
    """Jacobian of update d(x) = post(F(pre(x))) with respect to x."""
    w1, w2, w3 = [w[k].to(dtype) for k in ("w1", "w2", "w3")]
    s1, s2 = w["s1"].to(dtype), w["s2"].to(dtype)
    x, xt = x.to(dtype), xt.to(dtype)
    g, e = w1 @ xt, w3 @ xt
    sg = torch.sigmoid(g)
    mm, mp = g * sg, sg + g * sg * (1 - sg)
    f = w2 @ (mm * e)
    JF = (w2 * mm) @ w3 + (w2 * (mp * e)) @ w1
    d = x.shape[0]
    r1 = (x.pow(2).mean() + w["eps1"]).sqrt()
    r2 = (f.pow(2).mean() + w["eps2"]).sqrt()
    J = JF - f[:, None] * (f @ JF)[None, :] / (d * r2 ** 2)
    J = (s2 / r2)[:, None] * J
    J2 = J * (s1 / r1)[None, :]
    J2 = J2 - (J2 @ x)[:, None] * x[None, :] / (d * r1 ** 2)
    return J2


def fd_check(L, x):
    """Hand Jacobian against a finite difference, double precision."""
    w = W[L]
    x = x.double()
    t = lambda k: w[k].double()

    def upd(v):
        xt, _ = rms_norm(v, t("s1"), w["eps1"])
        f = t("w2") @ (F.silu(t("w1") @ xt) * (t("w3") @ xt))
        return rms_norm(f, t("s2"), w["eps2"])[0]

    xt, _ = rms_norm(x, t("s1"), w["eps1"])
    J = jacobian(w, x, xt, torch.float64)
    torch.manual_seed(0)
    v = torch.randn_like(x)
    v = v / v.norm() * x.norm() * 1e-5
    num = (upd(x + v) - upd(x - v)) / 2
    return float((num - J @ v).norm() / num.norm())


random.seed(0)
J0 = [json.loads(l) for l in open(jobs)]
random.shuffle(J0)
J0 = J0[:RUNS]
t0 = time.time()
checked = False
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
                    x, xt, f0, dd = g["x"][i], g["xt"][i], g["f"][i], g["d"][i]
                    gt, e = w["w1"] @ xt, w["w3"] @ xt
                    mm = F.silu(gt)
                    f = w["w2"] @ (mm * e)
                    rebuilt = float((f - f0).norm() / f0.norm())
                    if not checked:
                        print("rebuild check L", L, round(rebuilt, 5), "fd check", round(fd_check(L, x), 6), flush=True)
                    grad = w["w3"].T @ (mm * e)
                    step = w["s1"] * dd / (x.pow(2).mean() + w["eps1"]).sqrt()
                    A = float(F.cosine_similarity(step, -grad, dim=0))
                    torch.manual_seed(L)
                    fp = w["w2"][:, torch.randperm(w["w2"].shape[1], device=DEV)] @ (mm * e)
                    dp = rms_norm(fp, w["s2"], w["eps2"])[0]
                    Ap = float(F.cosine_similarity(w["s1"] * dp / (x.pow(2).mean() + w["eps1"]).sqrt(), -grad, dim=0))
                    x2 = x + dd
                    xt2 = rms_norm(x2, w["s1"], w["eps1"])[0]
                    e2 = w["w3"] @ xt2
                    wk = w["weak"]
                    S0, S1 = 0.5 * mm * e ** 2, 0.5 * mm * e2 ** 2
                    fire = (mm * e).abs() * w["w2"].norm(dim=0)
                    J = jacobian(w, x, xt)
                    sym = 0.5 * (J + J.T)
                    sf = float(sym.pow(2).sum() / J.pow(2).sum())
                    tr = float(torch.trace(J)) / x.shape[0]
                    sgn, lad = torch.linalg.slogdet(torch.eye(J.shape[0], device=DEV) + J)
                    rr = float(x2.pow(2).mean().sqrt() / x.pow(2).mean().sqrt())
                    fo.write(json.dumps(dict(
                        run=j["run"], id=j["id"], kind=kind, tok=i, L=L, rebuilt=rebuilt, A=A, Ap=Ap,
                        S0=float(S0.sum()), S1=float(S1.sum()),
                        S0w=float(S0[wk].sum()), S1w=float(S1[wk].sum()),
                        S0s=float(S0[~wk].sum()), S1s=float(S1[~wk].sum()),
                        sf=sf, tr=tr, sgn=float(sgn), lad=float(lad) / x.shape[0], rr=rr,
                        lam=float(lad) / x.shape[0] - float(np.log(rr)),
                        share_weak=float(fire[wk].sum() / fire.sum()), frac_weak=float(wk.float().mean()),
                    )) + "\n")
                if not checked:
                    checked = True
        fo.flush()
        print(k, RUNS, round(time.time() - t0), flush=True)
print("ENERGY_DONE", round(time.time() - t0), flush=True)
