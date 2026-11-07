#!/usr/bin/env python3
"""Step 1 report. For each neighbouring pair, the directions where the two writes oppose: eigenvectors of -(C+C')/2. usage: cancel_eig.py <read.pt> <out.pt> <report.txt>"""
import sys, torch
src, out, rep = sys.argv[1:]
d = torch.load(src); A, C, order, types = d["A"], d["C"], d["order"], d["types"]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
R = [8, 16, 32, 64, 128, 256]
U, L = [], []
lines = [f"pages {d['pages']} words {d['n']}", "pair  cos  | write sizes | neg energy | share of opposing energy in top r " + " ".join(map(str, R)) + " | corr of top 16 | size of earlier write inside top 64"]
for k in range(C.shape[0]):
    Ck, Aa, Ab = C[k].to(DEV).double(), A[k].to(DEV).double(), A[k + 1].to(DEV).double()
    S = -(Ck + Ck.T) / 2
    e, v = torch.linalg.eigh(S)
    e, v = e.flip(0), v.flip(1)
    pos = e.clamp(min=0).sum()
    cos = float(torch.trace(Ck) / (torch.trace(Aa) * torch.trace(Ab)).sqrt())
    share = [float(e[:r].clamp(min=0).sum() / pos) for r in R]
    v16 = v[:, :16]
    corr = (e[:16] / ((v16.T @ Aa @ v16).diag() * (v16.T @ Ab @ v16).diag()).sqrt()).mean()
    v64 = v[:, :64]
    inside = float((v64.T @ Aa @ v64).diag().sum() / torch.trace(Aa))
    lines.append(f"{order[k]:>6}>{order[k+1]:<6} {cos:+.3f} | {float(torch.trace(Aa)).__pow__(.5):6.2f} {float(torch.trace(Ab)).__pow__(.5):6.2f} | {float(pos):8.3f} | " + " ".join(f"{s:.2f}" for s in share) + f" | {float(corr):+.2f} | {inside:.3f}")
    U.append(v[:, :512].float().cpu()); L.append(e[:512].float().cpu())
torch.save(dict(U=torch.stack(U), L=torch.stack(L), order=order, types=types), out)
open(rep, "w").write("\n".join(lines) + "\n")
print("\n".join(lines)); print("CANCEL_EIG_DONE")
