#!/usr/bin/env python3
"""The running-out-of-time cell (Immy 2026-11-03: edge devices have small context windows; much more gain). Bolt-on, no training, no new neuron, same file shape.
Forced hand-ins on his own runs: closing the thinking at 3000 tokens gains 17 right answers and costs almost none, because long runs that never finish
get an answer at all. And a straight line reads how long he has been going (R2 ~0.85, whole problems left out). So one quiet neuron in the last layer
reads that clock at paragraph ends and leans on "</think>" once the words pass N: a collapsing decision bound (Cisek's urgency), written into the file so no engine
flag is needed. Under N it stays silent. Fit on the build split, reported on the other 1/3.
Nf=<words>: Immy's verdict, in place of one bound. Forced (the answer is on the page): the bound is Nf. Floating (no answer yet): keep working, the bound is N. mat=<scene_mat.npz>: breaks (a wrong number, an earlier number, a crossed wall at the next paragraph end) gets a gate well under zero.
usage: clock_cell.py <base folder> <cap.npz> <out folder> <runs.jsonl> [...] [layer=23 N=1900 Nf= slope=8 want=0.6 dev=cuda skip=eval.jsonl mat=]
N is in whitespace words (1900 is about 3000 tokens). slope: gate rise per unit of log(words), so 8 means +8*log(2)=+5.5 at twice N.
"""
import hashlib, json, os, random, shutil, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from patients import _lfm3

args = [a for a in sys.argv[1:] if "=" not in a]
opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
base, capf, out, runs = args[0], args[1], args[2], args[3:]
LB, NW, SLOPE, WANT, DEV = int(opt.get("layer", 23)), float(opt.get("N", 1900)), float(opt.get("slope", 8)), float(opt.get("want", 0.6)), opt.get("dev", "cuda")
skip = {json.loads(l)["id"] for l in open(opt["skip"])} if "skip" in opt else set()
isb = lambda g: int(hashlib.md5(str(g).encode()).hexdigest(), 16) % 3 != 2
z = np.load(capf, allow_pickle=True)
kind, pos, gid, since = z["kind"], z["pos"], z["gid"], z["since"]
Xa = z[f"x{LB}"].astype(np.float64)
NF = float(opt["Nf"]) if "Nf" in opt else NW
brk_ = np.zeros(len(pos), bool)
if "mat" in opt:
    zm = np.load(opt["mat"], allow_pickle=True)
    b_ = (zm["at"] == 1) & np.isin(zm["ver"], [1, 3, 4])
    Xa = np.concatenate([Xa, zm[f"x{LB}"][b_].astype(np.float64)])
    kind = np.concatenate([kind, np.full(b_.sum(), 0)])
    pos = np.concatenate([pos, np.full(b_.sum(), 500)])
    since = np.concatenate([since, np.full(b_.sum(), -1)])
    gid = np.concatenate([gid, zm["pid"][b_]])
    brk_ = np.concatenate([brk_, np.ones(b_.sum(), bool)])
bd = np.array([isb(g) for g in gid])


def ridge0(X, y, w, groups, alphas=(1e3, 1e4, 1e5, 1e6)):
    sc = X.std(0) + 1e-6
    Z = X / sc * np.sqrt(w)[:, None]
    Y = y * np.sqrt(w)
    f = np.array([int(hashlib.md5(str(g).encode()).hexdigest(), 16) % 4 for g in groups])
    best = None
    for al in alphas:
        err = sum(((Z[f == i] @ np.linalg.solve(Z[f != i].T @ Z[f != i] + al * np.eye(Z.shape[1]), Z[f != i].T @ Y[f != i]) - Y[f == i]) ** 2).sum() for i in range(4))
        if best is None or err < best[0]:
            best = (err, al)
    return np.linalg.solve(Z.T @ Z + best[1] * np.eye(Z.shape[1]), Z.T @ Y) / sc


# target gate: paragraph ends SLOPE*log(words/N), clamped low; every other word well under zero at any length
bound = np.where(since >= 0, NF, NW)
tg = np.where(kind == 0, np.maximum(SLOPE * np.log(np.maximum(pos, 1) / bound), -12.0), -14.0)
tg = np.where(brk_, -8.0, tg)
w = np.where(kind == 0, 1.0, 1.0)
far = (pos > 0.5 * NW)                                       # the points near the bound count more than the crowd of early ones
w = w * (1 + 3 * far) * np.where(kind == 3, 3.0, 1.0)      # the reply has to stay silent: past the bound a loud neuron writes "</think>" there (u1, read by hand)
gate = ridge0(Xa[bd], tg[bd], w[bd], gid[bd])
level = ridge0(Xa, np.ones(len(Xa)), np.ones(len(Xa)), gid)
g = Xa @ gate
L = [f"clock cell on {os.path.basename(base.rstrip('/'))}: layer {LB}, floating bound {NW:.0f} words, forced bound {NF:.0f} words, slope {SLOPE}, broken points {brk_.sum()}"]
hg = g[~bd & (kind == 0)]
hs, hp = since[~bd & (kind == 0)], pos[~bd & (kind == 0)]
for lab, se in (("answer up, past forced bound", (hs >= 0) & (hp > NF)), ("no answer, past forced bound, under floating bound", (hs < 0) & (hp > NF) & (hp < NW)), ("no answer, past floating bound", (hs < 0) & (hp > NW))):
    if se.any():
        L.append(f"  held out, {lab}: n {se.sum()}, gate median {np.median(hg[se]):+.2f}, over 0 {np.mean(hg[se] > 0):.0%}")
if brk_.any():
    gb = g[~bd & brk_]
    L.append(f"  held out, broken scenes at the next paragraph end: n {len(gb)}, gate median {np.median(gb):+.2f}, over 0 {np.mean(gb > 0):.0%}")
for part, sel in (("build", bd), ("held out", ~bd)):
    P = sel & (kind == 0)
    for lo, hi in ((0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0), (1.0, 1.5), (1.5, 9)):
        s = P & (pos >= lo * NW) & (pos < hi * NW)
        if s.any():
            L.append(f"  {part}: paragraph ends at {lo:.2f}-{hi:.2f} N, n {s.sum():4d}: gate median {np.median(g[s]):+.2f}, over 0 {np.mean(g[s] > 0):.0%}")
    s = sel & (kind != 0) & (kind != 3)
    L.append(f"  {part}: other thinking words n {s.sum()}: gate median {np.median(g[s]):+.2f}, over 0 {np.mean(g[s] > 0):.2%}; level median {np.median(Xa[sel] @ level):.2f}")
    s = sel & (kind == 3)
    L.append(f"  {part}: reply words n {s.sum()}: gate median {np.median(g[s]):+.2f}, over 0 {np.mean(g[s] > 0):.2%}; past N {np.mean(g[s & (pos > NW)] > 0):.2%} of {(s & (pos > NW)).sum()}")

p_ = _lfm3(DEV, base)
tok, m = p_.tok, p_.model
ff = m.model.layers[LB].feed_forward
END = tok("</think>", add_special_tokens=False).input_ids[0]
meta = json.load(open(os.path.join(base, "pam.json"))) if os.path.exists(os.path.join(base, "pam.json")) else {}
used = {int(v) for v in meta.get("handin_cell", {}).get("neurons", {}).values()} if LB == meta.get("handin_cell", {}).get("layer") else set()
with torch.no_grad():
    Xt = torch.tensor(Xa, dtype=torch.float32, device=DEV)
    act = F.silu(Xt @ ff.w1.weight.float().T) * (Xt @ ff.w3.weight.float().T)
    ws = act.abs() * ff.w2.weight.float().norm(dim=0)[None]
    loud = ws.kthvalue(max(1, int(0.999 * ws.shape[0])), dim=0).values
    NB = next(int(i) for i in loud.argsort() if int(i) not in used)
    del Xt, act, ws
L.append(f"  neuron: layer {LB} #{NB} (old 99.9% write {float(loud[NB]):.2e}, median {float(loud.median()):.2e})")

# points from his runs for direction, strength and report
rng = random.Random(2)
PTS = []
for f in runs:
    for t in map(json.loads, open(f)):
        th = t["thinking"]
        if t["id"] in skip or not th or len(th.split()) < 20:
            continue
        head = tok.apply_chat_template([{"role": "user", "content": t["q"]}], tokenize=False, add_generation_prompt=True, enable_thinking=True) + "\n"
        enc = tok(head + th, add_special_tokens=False, return_offsets_mapping=True)
        ids, off = enc.input_ids, enc.offset_mapping
        full, h0 = head + th, len(head)
        for i in range(1, len(ids) - 1):
            a, b = off[i]
            if a < h0 or "\n\n" not in full[a:b]:
                continue
            PTS.append((ids, i, len(th[:b - h0].split()), isb(t["id"])))
rng.shuffle(PTS)
pick = lambda lo, hi, build, n: [pt for pt in PTS if pt[3] == build and lo * NW <= pt[2] < hi * NW][:n]


def pend(pt):
    with torch.no_grad():
        hs = m.model(input_ids=torch.tensor([pt[0][: pt[1] + 1]], device=DEV)).last_hidden_state[0, -1]
        return float(torch.softmax(m.lm_head(hs).float(), -1)[END])


delta = {}


def hook(mod, a, o):
    x = o[0] if isinstance(o, tuple) else o
    d = torch.zeros_like(x[0, -1], requires_grad=True)
    delta["d"] = d
    x = x.clone()
    x[0, -1] = x[0, -1] + d
    return (x,) + tuple(o[1:]) if isinstance(o, tuple) else x


h = ff.register_forward_hook(hook)
tot = 0
for pt in pick(1.0, 9, True, 40):
    hs = m.model(input_ids=torch.tensor([pt[0][: pt[1] + 1]], device=DEV)).last_hidden_state[0, -1]
    lp = torch.log_softmax(m.lm_head(hs).float(), -1)[END]
    g_, = torch.autograd.grad(lp, delta["d"])
    tot = tot + g_.detach() / (g_.norm() + 1e-9)
h.remove()
d = (tot / tot.norm()).float()
orig = (ff.w1.weight[NB].clone(), ff.w3.weight[NB].clone(), ff.w2.weight[:, NB].clone())


def setn(scale):
    with torch.no_grad():
        ff.w1.weight[NB] = torch.tensor(gate, dtype=ff.w1.weight.dtype, device=DEV)
        ff.w3.weight[NB] = torch.tensor(level, dtype=ff.w1.weight.dtype, device=DEV)
        ff.w2.weight[:, NB] = (scale * d).to(ff.w2.weight.dtype)


# strength: at a fifth over N, on the build half, his chance of closing should reach WANT
pts = pick(1.1, 1.5, True, 40)
s = 2.0
for it in range(12):
    setn(s)
    pe = float(np.mean([pend(pt) for pt in pts]))
    L.append(f"  scale {s:.3g}: chance of closing at 1.1-1.5 N {pe:.3f} (asked {WANT})")
    print(L[-1], flush=True)
    if abs(pe - WANT) < 0.07:
        break
    s *= 2.0 if pe < WANT / 3 else float(np.clip(1 + (WANT - pe), 0.6, 1.5))
    if s > 3000:
        break
for lo, hi in ((0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0), (1.0, 1.25), (1.25, 1.5), (1.5, 2), (2, 9)):
    ptsh = pick(lo, hi, False, 40)
    if ptsh:
        L.append(f"  held out, paragraph ends at {lo:.2f}-{hi:.2f} N, n {len(ptsh)}: chance of closing {np.mean([pend(pt) for pt in ptsh]):.3f}")
os.makedirs(out, exist_ok=True)
m.save_pretrained(out)
tok.save_pretrained(out)
for f_ in os.listdir(base):
    if f_.endswith((".py", ".jinja", ".json", ".txt")) and not f_.endswith(".index.json") and not os.path.exists(os.path.join(out, f_)):
        shutil.copy(os.path.join(base, f_), out)
meta["clock_cell"] = {"layer": LB, "neuron": NB, "N": NW, "Nf": NF, "slope": SLOPE, "scale": s}
json.dump(meta, open(os.path.join(out, "pam.json"), "w"), indent=1)
open(os.path.join(out, "clock_report.txt"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
print("SAVED", out)
