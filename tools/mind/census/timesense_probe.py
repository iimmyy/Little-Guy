#!/usr/bin/env python3
"""The parts of a time sense a model already has, measured in one pass (step E of NIGHT-PLAN-2026-10-29: is the
installed time sense universal, or an LFM3 accident?). For any patient, on its own plain answers:
  expected   on the word where the thinking opens: does a straight line already say how long this question will take
             (total thinking words, and words until the answer first appears)? Built on half, scored on the other
             half; baseline = the question's length
  value      the same reading off each attention layer's value on that word, per key group: is the estimate stored
             where attention could fetch it (what the carrier routes)?
  persist    the opening-word line read unchanged at thinking word t (only answers still thinking at t): does the
             model keep the estimate while thinking, or drop it (what the carrier fixes)?
  clock      a straight line for log(words thinking so far) on thinking words, held-out miss by length (Vierordt check:
             long stretches read short?)

usage: timesense_probe.py <patient> <answers jsonl with thinking> <out folder> [every=4 dev=mps cut=3000]
"""
import json, os, sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../expert"))
ARGS, sys.argv[1:] = sys.argv[1:], []                     # grade.py grades its own arguments on import
from grade import answer_line
from patients import load
from probe_math import rho, ridge

name, src, out = ARGS[:3]
opt = dict(a.split("=", 1) for a in ARGS[3:])
EVERY, DEV, CUT = int(opt.get("every", 4)), opt.get("dev", "mps"), int(opt.get("cut", 3000))
os.makedirs(out, exist_ok=True)
TAIL = "\n\nEnd with a line: Answer: <your answer>"
OPEN = {"e4b": "<|channel>thought\n", "lfm3": "\n"}[name]
MLP = {"e4b": lambda l: l.mlp, "lfm3": lambda l: l.feed_forward}[name]
ATT = {"e4b": lambda l: l.self_attn, "lfm3": lambda l: getattr(l, "self_attn", None)}[name]
p = load(name, DEV)
tok, lay = p.tok, p.model.model.layers
LAY = list(range(0, len(lay), EVERY))
rows = [r for r in map(json.loads, open(src)) if r["thinking"]]
cap = {}
for i in LAY:
    MLP(lay[i]).register_forward_pre_hook(lambda m, a, i=i: cap.__setitem__(("x", i), a[0][0].float().cpu()))
    at = ATT(lay[i])
    if at is not None and hasattr(at, "v_proj"):
        at.v_proj.register_forward_hook(lambda m, a, o, i=i: cap.__setitem__(("v", i), o[0].float().cpu()))
KV = getattr(p.model.config, "num_key_value_heads", None) or p.model.config.text_config.num_key_value_heads


def solve(r, th):
    T = r["think_tokens"] if r["finish"] == "stop" else 10 ** 5
    a = (answer_line(r["reply"], True) or "").strip().strip("$ ") if r["finish"] == "stop" else ""
    k = r["thinking"].find(a) if a else -1
    s = len(th) if k < 0 else len(tok(OPEN + r["thinking"][:k + len(a)], add_special_tokens=False).input_ids)
    return np.log(max(8, min(s, T))), np.log(min(T, 10 ** 5) if r["finish"] == "stop" else max(len(th), 6000))


Xp = {i: [] for i in LAY}; Vp = {}
Xt = {i: {t: [] for t in (16, 64, 256, 1024)} for i in LAY}; Yt = {t: [] for t in (16, 64, 256, 1024)}
CL = {i: [] for i in LAY}; CT = []; CA = []
Ys, Yall, ql = [], [], []
with torch.no_grad():
    for n, r in enumerate(rows):
        pr = tok(tok.apply_chat_template([{"role": "user", "content": r["q"] + TAIL}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=True), add_special_tokens=False).input_ids
        op = tok(OPEN, add_special_tokens=False).input_ids if name == "e4b" else []
        th = tok(r["thinking"] if name == "e4b" else OPEN + r["thinking"], add_special_tokens=False).input_ids
        ids = pr + op + th[:CUT]
        P = len(pr) + len(op) - 1                          # the word where the thinking opens
        cap.clear()
        p.model.model(input_ids=torch.tensor([ids], device=DEV))
        s_, t_ = solve(r, th)
        Ys.append(s_); Yall.append(t_); ql.append(np.log(len(tok(r["q"], add_special_tokens=False).input_ids)))
        n_th = min(len(th), CUT)
        ts = sorted(set(int(x) for x in np.exp(np.random.RandomState(n).uniform(0, np.log(max(2, n_th - 1)), 40))))
        for i in LAY:
            x = cap[("x", i)]
            Xp[i].append(x[P])
            for t in Xt[i]:
                if t < n_th:
                    Xt[i][t].append(x[P + t])
            CL[i].append(x[[P + t for t in ts]])
            if ("v", i) in cap:
                Vp.setdefault(i, []).append(cap[("v", i)][P])
        for t in Yt:
            if t < n_th:
                Yt[t].append(t_)
        CT += ts; CA += [n] * len(ts)
        if (n + 1) % 25 == 0:
            print(f"  {n + 1}/{len(rows)}", flush=True)

Ys, Yall, ql = np.array(Ys), np.array(Yall), np.array(ql)
a, b = np.arange(len(Ys)) % 2 == 0, np.arange(len(Ys)) % 2 == 1
lines = [f"time-sense parts inside {name}: {len(rows)} answers ({int(sum(r['finish'] != 'stop' for r in rows))} ran out of words), "
         f"built on half, scored on the other half",
         f"baseline (question length): total rho {rho(ql[b], Yall[b]):+.2f}, until-answer rho {rho(ql[b], Ys[b]):+.2f}",
         "layer  expected-total  expected-until-answer  persist (opening line at word 16 / 64 / 256 / 1024)  clock miss  long-read (2k+ words)"]
best_val = []
for i in LAY:
    X = torch.stack(Xp[i]).numpy()
    pt, _, _, (w, bb) = ridge(X[a], Yall[a], X[b])
    ps, _, _, _ = ridge(X[a], Ys[a], X[b])
    per = []
    for t in (16, 64, 256, 1024):
        Xq = torch.stack(Xt[i][t]).numpy() if Xt[i][t] else None
        per.append(f"{rho(Xq @ w + bb, Yt[t]):+.2f}" if Xq is not None and len(Yt[t]) >= 10 else "  -  ")
    C = torch.cat(CL[i]); T_ = torch.tensor(CT).float(); A_ = torch.tensor(CA)
    bm = A_ % 2 == 0
    y = torch.log(T_.clamp(min=1))
    pr_, _, _, _ = ridge(C[bm].numpy(), y[bm].numpy(), C[~bm].numpy())
    miss = float(np.exp(np.median(np.abs(pr_ - y[~bm].numpy()))))
    lng = (T_[~bm] >= 2000).numpy()
    lr = f"x{float(np.exp(np.median(pr_[lng] - y[~bm].numpy()[lng]))):.2f}" if lng.sum() >= 10 else "  -  "
    lines.append(f"{i:5d}     {rho(pt, Yall[b]):+.2f}            {rho(ps, Ys[b]):+.2f}              {'  '.join(per)}                  x{miss:.2f}       {lr}")
lines.append("value on the opening word (per key group), total thinking rho:")
for i, V in Vp.items():
    V = torch.stack(V).numpy()
    hd = V.shape[1] // KV
    g = []
    for k in range(KV):
        pv, _, _, _ = ridge(V[a][:, k * hd:(k + 1) * hd], Yall[a], V[b][:, k * hd:(k + 1) * hd])
        g.append(f"{rho(pv, Yall[b]):+.2f}")
    lines.append(f"  layer {i:3d}: " + " ".join(g))
print("\n".join(lines))
open(os.path.join(out, f"timesense_{name}.txt"), "w").write("\n".join(lines) + "\n")
print("TIMESENSE_DONE")
