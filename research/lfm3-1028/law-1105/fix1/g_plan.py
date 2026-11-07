#!/usr/bin/env python3
"""Any model: reads each layer's neuron gram in blocks and plans cuts with rebuilt writes (same as cut_solve, without keeping every gram in memory).
usage: g_plan.py <arch> <hf> <steps.jsonl> <out_dir> <report.txt> [sizes=0.3,0.5 block=3 pages=100 words=600 maxtok=4000 ridge=1e-5]"""
import sys, time
import numpy as np, torch
import g_arch
args = [a for a in sys.argv[1:] if "=" not in a]; opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
arch, hf, stepsf, outdir, rep = args
SIZES = [float(x) for x in opt.get("sizes", "0.3,0.5").split(",")]; BL = int(opt.get("block", 3)); PAGES = int(opt.get("pages", 100)); WORDS = int(opt.get("words", 600)); MAXTOK = int(opt.get("maxtok", 4000)); RIDGE = float(opt.get("ridge", 1e-5))
DEV = "cuda"
m, tok, body, head, cap = g_arch.load(arch, hf)
LS = body.layers; NL = len(LS); IDX = {}; NEU = {}; RMS = {}
for L, b in enumerate(LS):
    g_arch.w2(arch, b).register_forward_pre_hook(lambda mod, a, L=L: NEU.__setitem__(L, a[0][0][IDX["i"]].float()))
    g_arch.post(arch, b).register_forward_pre_hook(lambda mod, a, L=L: RMS.__setitem__(L, a[0][0][IDX["i"]].float().pow(2).mean(-1, keepdim=True).sqrt()))
fit, held = g_arch.pages(arch, stepsf, ["screen-sci-40.jsonl", "screen-math-40.jsonl"])
P = []
for i, (q, text, _) in enumerate(fit[:PAGES]):
    h = head(q); ids = tok(h + text, add_special_tokens=False).input_ids[:MAXTOK]; first = len(tok(h, add_special_tokens=False).input_ids)
    if len(ids) - first < 40:
        continue
    rng = np.random.default_rng(i); P.append((torch.tensor([ids], device=DEV), torch.tensor(sorted(set(range(first)) | set(rng.choice(np.arange(first, len(ids) - 1), size=min(WORDS, len(ids) - 1 - first), replace=False).tolist())), device=DEV)))
print("arch", arch, "layers", NL, "fit pages", len(P), "held pages", len(held), flush=True)
res = {s: {} for s in SIZES}; lines = ["L | missed share rebuilt/plain at sizes " + " ".join(map(str, SIZES))]; t0 = time.time(); cnt = 0
for start in range(0, NL, BL):
    blk = list(range(start, min(NL, start + BL))); n = g_arch.w2(arch, LS[0]).in_features
    G = {L: torch.zeros(n, n, dtype=torch.float64, device=DEV) for L in blk}; cnt = 0
    for ids, pos in P:
        IDX["i"] = pos
        with torch.inference_mode():
            body(input_ids=ids)
            for L in blk:
                v = NEU[L] / RMS[L]; G[L] += (v.T @ v).double()
        cnt += len(pos)
    for L in blk:
        Gm = G.pop(L) / cnt; Gm = (Gm + Gm.T) / 2; N = Gm.shape[0]
        b = LS[L]; D = g_arch.post(arch, b).weight.detach().double(); W = g_arch.w2(arch, b).weight.detach().double(); DW = D[:, None] * W
        tot = float(((DW @ Gm) * DW).sum()); lam = RIDGE * Gm.diag().mean()
        Pm = torch.linalg.inv(Gm + lam * torch.eye(N, device=DEV, dtype=Gm.dtype)); alive = torch.arange(N, device=DEV); Weff = DW.clone(); bs = int(0.02 * N); msg = []
        for s in sorted(SIZES):
            target = N - int(round(s * N))
            while len(alive) > target:
                k = min(bs, len(alive) - target); sal = Weff.pow(2).sum(0) / Pm.diag().clamp(min=1e-18); drop = sal.topk(k, largest=False).indices
                mask = torch.ones(len(alive), dtype=torch.bool, device=DEV); mask[drop] = False; keep = mask.nonzero()[:, 0]
                Pdd = Pm[drop][:, drop]; Pkd = Pm[keep][:, drop]; T = torch.linalg.solve(Pdd, Pkd.T)
                Weff = Weff[:, keep] - Weff[:, drop] @ T; Pm = Pm[keep][:, keep] - Pkd @ T; alive = alive[keep]
            Err = DW.clone(); Err[:, alive] -= Weff; mi = float(((Err @ Gm) * Err).sum()) / tot
            en = DW.pow(2).sum(0) * Gm.diag(); top = en.topk(target).indices; E2 = DW.clone(); E2[:, top] = 0; ep = float(((E2 @ Gm) * E2).sum()) / tot
            res[s][L] = dict(keep=alive.cpu(), w2=(Weff / D[:, None]).to(torch.bfloat16).cpu(), energy_keep=top.sort().values.cpu()); msg.append(f"{mi:.3f}/{ep:.3f}")
        del Pm, Gm
        lines.append(f"{L:2d} | " + " ".join(msg)); print(lines[-1], round(time.time() - t0), "s", flush=True)
for s in SIZES:
    torch.save(res[s], f"{outdir}/cutplan-{int(s*100)}.pt")
open(rep, "w").write("\n".join(lines) + "\n"); print("G_PLAN_DONE", flush=True)
