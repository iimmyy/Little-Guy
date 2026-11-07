#!/usr/bin/env python3
"""Misfit crowd (2026-11-03 morning, a test candidate for Immy to judge: is a separate crowd still "one alarm"?).
c5m2b: the misfit detector in layer 17 pushes c5's wire 397 down, selective (right answers 0%), but c5's crowd is a
small lever (doubt at misfits 0.062 -> 0.076) and one neuron's write tops out at ~7 raw (post-FFN norm). c5w2 (second
wire from layer 9) is louder but its read is noisy and corrupts digits (drift 3.5x c5).
This keeps c5m2b exactly and gives the misfit its OWN doubt crowd, picked the way c5 picked its crowd but at misfit
forks: in layers 18-23, the engaged neurons (gate > 0 there) with the strongest doubt vote at misfit forks (each
neuron's push toward hand-in minus doubt words, through the final norm; pam_belief.py), not in c5's crowd, until they
hold `target` points. Each crowd layer learns a read of the detector's firing (1 at misfit-fork height, 0 silent; the
same fit c5's crowd uses for its key) and each picked neuron's gate gets +gain * (gate_at_hot + shift) * read: at full
read its gate is lifted by the mirror of c5's duck. c5's key, wire and crowd, and the detector, are untouched.
Built on 2/3 of the forks by problem (the same split as pam_misfit.py), the held-out 1/3 reports.
usage: pam_misfit_crowd.py <c5m2b folder> <problems.jsonl> <thinking.jsonl> <out folder> [target=4 gain=1 ord=20 dev=cuda]
"""
import json, os, random, shutil, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from patients import _lfm3
from situation_probe import TAIL
from fit_test import forks

HAND_IDS, DOUBT_IDS = [49, 1672, 2083, 8175, 21009], [1938, 2375, 3911, 7731, 8952]
base, src, think_f, out = sys.argv[1:5]
opt = dict(a.split("=", 1) for a in sys.argv[5:])
DEV, TARGET, GAIN, NORD = opt.get("dev", "cuda"), float(opt.get("target", 4)), float(opt.get("gain", 1)), int(opt.get("ord", 20))
META = json.load(open(os.path.join(base, "pam.json")))
C5CROWD = {(c["layer"], c["neuron"]) for c in META["receptors"]}
SHIFT, MB = float(META["shift"]), META["misfit_brake"]
KL_, BN = MB["layer"], MB["neuron"]
p = _lfm3(DEV, base)
tok, m = p.tok, p.model
lay = m.model.layers
CL = list(range(KL_ + 1, len(lay)))
FINAL = m.model.embedding_norm
dvec = FINAL.weight.float() * (m.lm_head.weight[HAND_IDS].float().mean(0) - m.lm_head.weight[DOUBT_IDS].float().mean(0))
WD = {L: (lay[L].post_ffn_norm.weight.float()[:, None] * lay[L].feed_forward.w2.weight.float() * dvec[:, None]).sum(0) for L in CL}

got = {}
lay[KL_].feed_forward.register_forward_pre_hook(lambda mod, a: got.__setitem__("xk", a[0][0].float()))
for L in CL:
    lay[L].feed_forward.register_forward_pre_hook(lambda mod, a, L=L: got.__setitem__(("x", L), a[0][0].float()))
    lay[L].post_ffn_norm.register_forward_pre_hook(lambda mod, a, L=L: got.__setitem__(("y", L), a[0][0].float()))
FINAL.register_forward_pre_hook(lambda mod, a: got.__setitem__("fin", a[0][0].float()))
ffk = lay[KL_].feed_forward

items = forks(src, think_f)
rng = random.Random(0)
seqs, V, G, DET = [], [], [], []
XC = {L: [] for L in CL}
VOTE = {L: [] for L in CL}
GATE = {L: [] for L in CL}
for k, (iid, text, think, A, g, (s0, s1, cut), (b, c, r)) in enumerate(items):
    head = tok.apply_chat_template([{"role": "user", "content": text + TAIL}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=True) + "\n"
    n_head = len(tok(head, add_special_tokens=False).input_ids)
    for vi, v in enumerate((A, b, c)):
        ids = tok(head + think[:s0] + str(v) + think[s1:cut], add_special_tokens=False).input_ids
        with torch.no_grad():
            m.model(input_ids=torch.tensor([ids], device=DEV))
            poss = [(len(ids) - 1, vi)]
            if vi == 0:
                rngp = list(range(n_head, len(ids) - 2))
                poss += [(j, 3) for j in rng.sample(rngp, min(NORD, len(rngp)))]
            xk = got["xk"]
            det = F.silu(xk @ ffk.w1.weight[BN].float()) * (xk @ ffk.w3.weight[BN].float())
            rf = got["fin"].pow(2).mean(-1, keepdim=True).add(1e-5).sqrt()
            for pos, vv in poss:
                seqs.append((ids, pos, vv, k)); V.append(vv); G.append(k); DET.append(float(det[pos]))
                for L in CL:
                    x = got[("x", L)][pos]
                    XC[L].append(x.cpu())
                    if vv < 3:                                             # per-neuron captures only at the forks
                        gt_ = x @ lay[L].feed_forward.w1.weight.float().T
                        act = F.silu(gt_) * (x @ lay[L].feed_forward.w3.weight.float().T)
                        ry = got[("y", L)][pos].pow(2).mean().add(1e-5).sqrt()
                        VOTE[L].append((act * WD[L] / ry / rf[pos]).half().cpu()); GATE[L].append(gt_.half().cpu())
    if k % 100 == 0:
        print(f"capture {k}/{len(items)}", flush=True)
V, G, DET = np.array(V), np.array(G), np.array(DET)
build = G % 3 != 2
FV = V < 3
names = ["a right", "b wrong, fits", "c wrong, misfit", "ordinary words"]
L_ = [f"misfit crowd on {os.path.basename(base.rstrip('/'))}: detector layer {KL_} #{BN}; crowd layers {CL[0]}-{CL[-1]}, "
      f"target {TARGET} points, gain {GAIN}; {len(items)} forks (build {len(set(G[build]))}, held out {len(set(G[~build]))})"]
dhot = float(np.median(DET[build & (V == 2)]))
L_.append(f"  detector activation median: " + ", ".join(f"{names[v]} {np.median(DET[V == v]):+.3f}" for v in range(4)) + f" (hot {dhot:+.3f})")

# ---- pick the crowd: strongest doubt votes at build-half misfit forks, engaged there, not c5's
fmask = (build & (V == 2))[FV]
cand = []
for L in CL:
    vt = torch.stack(VOTE[L]).float()[torch.tensor(fmask)].mean(0)
    gh = torch.stack(GATE[L]).float()[torch.tensor(fmask)].median(0).values
    for j in torch.argsort(vt)[:2000].tolist():
        if gh[j] > 0 and vt[j] < 0 and (L, j) not in C5CROWD:
            cand.append((float(vt[j]), L, j, float(gh[j])))
cand.sort()
pick, held = [], 0.0
for c_ in cand:
    if held <= -TARGET:
        break
    pick.append(c_); held += c_[0]
L_.append(f"  crowd: {len(pick)} neurons holding {held:+.2f} points of doubt vote at misfit forks; by layer " +
          ", ".join(f"{L}: {sum(1 for c_ in pick if c_[1] == L)}" for L in CL))


def ridge0(Xa, y, alphas=(1, 10, 100, 1e3, 1e4)):
    sc = np.sqrt((Xa ** 2).mean(0)) + 1e-6
    Z = Xa / sc
    f = np.arange(len(y)) % 5
    best = None
    for al in alphas:
        err = sum(((Z[f == i] @ np.linalg.solve(Z[f != i].T @ Z[f != i] + al * np.eye(Z.shape[1]), Z[f != i].T @ y[f != i]) - y[f == i]) ** 2).sum()
                  for i in range(5))
        if best is None or err < best[0]:
            best = (err, al)
    return np.linalg.solve(Z.T @ Z + best[1] * np.eye(Z.shape[1]), Z.T @ y) / sc


# ---- each crowd layer's read of the detector (fit on the build half, checked held out)
kt = np.clip(DET / dhot, 0, None)
RD = {}
for L in sorted({c_[1] for c_ in pick}):
    Xc = torch.stack(XC[L]).double().numpy()
    RD[L] = ridge0(Xc[build], kt[build])
    rd = Xc @ RD[L]
    L_.append(f"  layer {L}: read (held out) median " + ", ".join(f"{names[v][:1]} {np.median(rd[~build & (V == v)]):+.2f}" for v in range(4))
              + f"; ordinary 99% |read| {np.quantile(np.abs(rd[~build & (V == 3)]), .99):.2f}")
print("\n".join(L_), flush=True)

orig = {(L, j): lay[L].feed_forward.w1.weight[j].clone() for _, L, j, _ in pick}


def set_crowd(on):
    with torch.no_grad():
        for v_, L, j, gh in pick:
            W1 = lay[L].feed_forward.w1.weight
            row = orig[(L, j)].float()
            if on:
                row = row + GAIN * (gh + SHIFT) * torch.tensor(RD[L], dtype=torch.float32, device=DEV)
            W1[j] = row.to(W1.dtype)


def logits_at(part):
    res = {v: [] for v in range(4)}
    for (ids, pos, vv, k) in seqs:
        if (k % 3 != 2) == part:
            with torch.no_grad():
                res[vv].append(m(input_ids=torch.tensor([ids[:pos + 1]], device=DEV)).logits[0, pos].float())
    return res


pd = lambda lgs: float(np.mean([float(torch.softmax(l, -1)[DOUBT_IDS].sum()) for l in lgs]))
for part in (True, False):
    tag = "build" if part else "held out"
    set_crowd(False); befo = logits_at(part)
    set_crowd(True); after = logits_at(part)
    kl = [float((torch.log_softmax(a, -1).exp() * (torch.log_softmax(a, -1) - torch.log_softmax(b, -1))).sum())
          for a, b in zip(befo[3], after[3])]
    L_.append(f"  {tag}: chance the next word opens doubt, c5m2b -> + misfit crowd: " +
              ", ".join(f"{names[v]} {pd(befo[v]):.3f} -> {pd(after[v]):.3f}" for v in range(4)) +
              f"; ordinary words drift KL mean {np.mean(kl):.4f}, 99% {np.quantile(kl, .99):.3f}")
    print(L_[-1], flush=True)
os.makedirs(out, exist_ok=True)
m.save_pretrained(out)
tok.save_pretrained(out)
for f in os.listdir(base):
    if f.endswith((".py", ".jinja", ".json", ".txt")) and not f.endswith(".index.json") and not os.path.exists(os.path.join(out, f)):
        shutil.copy(os.path.join(base, f), out)
META["misfit_crowd"] = {"target": TARGET, "gain": GAIN, "shift": SHIFT,
                        "neurons": [{"layer": L, "neuron": j, "vote": round(v_, 4), "gate_at_hot": round(gh, 2)} for v_, L, j, gh in pick]}
json.dump(META, open(os.path.join(out, "pam.json"), "w"), indent=1)
open(os.path.join(out, "pam_misfit_crowd_report.txt"), "w").write("\n".join(L_) + "\n")
print("\n".join(L_))
print("SAVED", out)
