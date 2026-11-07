#!/usr/bin/env python3
"""Plans neuron cuts from the gram alone, with the same count in every layer. Removes in batches the neurons the rest can best cover (saliency |D w2_i|^2 / P_ii, P = inverse gram), then re-solves the kept neurons' writes by least squares.
Also ranks by plain energy for the no-rebuild arm. usage: cut_solve.py <hf> <gram.pt> <out_dir> <report.txt> [sizes=0.1,0.2,0.3,0.5 ridge=1e-5 batch=0.02]"""
import json, struct, sys
import numpy as np, torch
args = [a for a in sys.argv[1:] if "=" not in a]; opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
hf, gramf, outdir, rep = args; SIZES = [float(x) for x in opt.get("sizes", "0.1,0.2,0.3,0.5").split(",")]; RIDGE = float(opt.get("ridge", 1e-5)); BF = float(opt.get("batch", 0.02))
DEV = "cuda"
_f = open(hf + "/model.safetensors", "rb"); _L = struct.unpack("<Q", _f.read(8))[0]; _H = json.loads(_f.read(_L)); _B = 8 + _L
ST = torch.load(opt["state"]) if "state" in opt else {}
def get(k):
    if k in ST:
        return ST[k].float()
    a, b = _H[k]["data_offsets"]; _f.seek(_B + a); u = np.frombuffer(_f.read(b - a), dtype="<u2").astype("<u4") << 16
    return torch.from_numpy(u.view("<f4").reshape(_H[k]["shape"]).copy())
d = torch.load(gramf, mmap=True); Gall, MU = d["G"], d["mu"]
NL, N = Gall.shape[0], Gall.shape[1]
res = {s: {} for s in SIZES}; lines = ["L | missed share of the write at each size, rebuilt then plain energy cut | same, varying part only"]
for L in range(NL):
    G = Gall[L].to(DEV).double(); G = (G + G.T) / 2; D = get(f"model.layers.{L}.post_ffn_norm.weight").to(DEV).double(); W = get(f"model.layers.{L}.feed_forward.w2.weight").to(DEV).double()
    mu = MU[L].to(DEV).double(); C = G - torch.outer(mu, mu)
    lam = RIDGE * G.diag().mean(); DW = D[:, None] * W
    tot = float(((DW @ G) * DW).sum()); totv = float(((DW @ C) * DW).sum())
    P = torch.linalg.inv(G + lam * torch.eye(N, device=DEV, dtype=G.dtype))
    alive = torch.arange(N, device=DEV); Weff = DW.clone(); bs = int(BF * N)
    miss = {}; missv = {}; pe = {}
    for s in sorted(SIZES):
        target = N - int(round(s * N))
        while len(alive) > target:
            k = min(bs, len(alive) - target)
            sal = Weff.pow(2).sum(0) / P.diag().clamp(min=1e-18)
            drop = sal.topk(k, largest=False).indices
            mask = torch.ones(len(alive), dtype=torch.bool, device=DEV); mask[drop] = False
            keep = mask.nonzero()[:, 0]
            Pdd = P[drop][:, drop]; Pkd = P[keep][:, drop]
            T = torch.linalg.solve(Pdd, Pkd.T)   # kept coefficients that cover the dropped neurons
            Weff = Weff[:, keep] - Weff[:, drop] @ T
            P = P[keep][:, keep] - Pkd @ T
            alive = alive[keep]
        A = alive
        Err = DW.clone(); Err[:, A] = Err[:, A] - Weff
        mi = float(((Err @ G) * Err).sum()) / tot; miv = float(((Err @ C) * Err).sum()) / totv
        en = DW.pow(2).sum(0) * G.diag(); top = en.topk(target).indices
        Err2 = DW.clone(); Err2[:, top] = 0
        ep = float(((Err2 @ G) * Err2).sum()) / tot
        DWk = Weff; size2 = float(((DWk @ G[A][:, A]) * DWk).sum())
        res[s][L] = dict(keep=A.cpu(), w2=(Weff / D[:, None]).to(torch.bfloat16).cpu(), gain=(size2 / tot) ** .5, energy_keep=top.sort().values.cpu())
        miss[s] = mi; missv[s] = miv; pe[s] = ep
    lines.append(f"{L:2d} | " + " ".join(f"{miss[s]:.3f}/{pe[s]:.3f}" for s in SIZES) + " | " + " ".join(f"{missv[s]:.3f}" for s in SIZES))
    print(lines[-1], flush=True)
for s in SIZES:
    torch.save(res[s], f"{outdir}/cutplan-{int(s*100)}.pt")
open(rep, "w").write("\n".join(lines) + "\n"); print("CUT_SOLVE_DONE", flush=True)
