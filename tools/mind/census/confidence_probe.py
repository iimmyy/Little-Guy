#!/usr/bin/env python3
"""Does the model know when it is right? (Immy 2026-10-29: the stopping signal is confidence; a confident mind waves
off a doubt cheaply instead of redoing the work.) On the model's own plain answers, find every commit moment in the
thinking ("Thus ... = X", "answer is X", "\\boxed{X}"), grade X against the truth on its own (never the final reply),
and read the stream on the word that ends X:
  probe        a straight line for "X is right", built on the train answers, scored on the test answers (AUROC)
  baselines    how far into the thinking (log words), and consistency (how many earlier moments already said X),
               the strongest cheap signal a timer-free rule could use
  after-right  of moments where X is right, how much thinking still follows (the doubt spiral the gate would cut)

usage: confidence_probe.py <patient> <train answers> <test answers> <out folder> [every=2 dev=mps cut=6000 per=10]
"""
import json, os, re, sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../expert"))
ARGS, sys.argv[1:] = sys.argv[1:], []                     # grade.py grades its own arguments on import
from grade import right
from patients import load
from probe_math import ridge

name, src_a, src_b, out = ARGS[:4]
opt = dict(a.split("=", 1) for a in ARGS[4:])
EVERY, DEV, CUT, PER = int(opt.get("every", 2)), opt.get("dev", "mps"), int(opt.get("cut", 6000)), int(opt.get("per", 10))
os.makedirs(out, exist_ok=True)
TAIL = "\n\nEnd with a line: Answer: <your answer>"
OPEN = {"e4b": "<|channel>thought\n", "lfm3": "\n"}[name]
MLP = {"e4b": lambda l: l.mlp, "lfm3": lambda l: l.feed_forward}[name]
COMMIT = re.compile(r"\\boxed\{([^{}]{1,40})\}|(?:answer is|answer =|answer:)\s*\$?([^\n$]{1,40}?)\$?(?=[.\n]|$)"
                    r"|(?:Thus|So|Therefore|Hence)[^.\n]{0,60}?=\s*\$?([^\n$=]{1,40}?)\$?(?=\.\s|\n|$)")

p = load(name, DEV)
tok, lay = p.tok, p.model.model.layers
LAY = list(range(0, len(lay), EVERY))
cap = {}
for i in LAY:
    MLP(lay[i]).register_forward_pre_hook(lambda m, a, i=i: cap.__setitem__(i, a[0][0].float().cpu()))


def moments(r):
    t, seen, ms = r["thinking"], {}, []
    for m in COMMIT.finditer(t):
        x = next(g for g in m.groups() if g is not None).strip().rstrip(".").strip()
        if not x or len(x) > 40:
            continue
        key = re.sub(r"\s+", "", x)
        ok = bool(right({**r, "reply": x}, x))            # graded on X alone: the final reply never leaks in
        g = next(j for j in (1, 2, 3) if m.group(j) is not None)
        ms.append(dict(end=m.end(g), start=m.start(g), x=x, ok=ok, prior=seen.get(key, 0), explicit=g < 3))
        seen[key] = seen.get(key, 0) + 1
    if len(ms) > PER:                                     # spread over the answer, keep first and last
        ms = [ms[int(k)] for k in np.linspace(0, len(ms) - 1, PER)]
    return ms


def collect(path):
    X = {i: [] for i in LAY}; Y, POS, CON, AFTER, RID, EXP, LP = [], [], [], [], [], [], []
    rows = [r for r in map(json.loads, open(path)) if r["thinking"]]
    with torch.no_grad():
        for n, r in enumerate(rows):
            ms = moments(r)
            if not ms:
                continue
            pr = tok(tok.apply_chat_template([{"role": "user", "content": r["q"] + TAIL}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=True),
                     add_special_tokens=False).input_ids
            enc = tok(OPEN + r["thinking"], add_special_tokens=False, return_offsets_mapping=True)
            th, off = enc.input_ids[:CUT], enc.offset_mapping[:CUT]
            ends = np.array([e for _, e in off])
            keep = []
            for m in ms:
                k = int(np.searchsorted(ends, m["end"] + len(OPEN)))
                k0 = int(np.searchsorted(ends, m["start"] + len(OPEN) + 1))
                if k < len(th) and k0 >= 1:
                    keep.append((k, k0, m))
            if not keep:
                continue
            cap.clear()
            hid = p.model.model(input_ids=torch.tensor([pr + th], device=DEV)).last_hidden_state[0]
            for k, k0, m in keep:
                # expectation (Immy): the chance he gave his own X while writing it, mean log-prob over its words
                at = torch.arange(len(pr) + k0, len(pr) + k + 1, device=DEV)
                lp = torch.log_softmax(p.model.lm_head(hid[at - 1]).float(), -1)
                tgt = torch.tensor([pr + th][0], device=DEV)[at]
                LP.append(float(lp.gather(1, tgt[:, None]).mean()))
                EXP.append(m["explicit"])
                for i in LAY:
                    X[i].append(cap[i][len(pr) + k].half())
                Y.append(m["ok"]); POS.append(np.log(k + 1)); CON.append(m["prior"])
                AFTER.append(len(enc.input_ids) - k); RID.append(n)
            if (n + 1) % 20 == 0:
                print(f"  {os.path.basename(path)} {n + 1}/{len(rows)}: {len(Y)} moments", flush=True)
    return (X, np.array(Y, float), np.array(POS), np.array(CON, float), np.array(AFTER), np.array(RID),
            np.array(EXP, bool), np.array(LP))


def auroc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, bool)
    if y.all() or not y.any():
        return float("nan")
    r = np.argsort(np.argsort(s)) + 1
    return (r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())


Xa, Ya, Pa, Ca, Aa, Ra, Ea, La = collect(src_a)
Xb, Yb, Pb, Cb, Ab, Rb, Eb, Lb = collect(src_b)
if opt.get("only") == "explicit":       # control: only explicit answer claims ("answer is X", boxed), no in-between steps
    sa, sb = Ea, Eb
    Xa = {i: [v for v, k in zip(X_, sa) if k] for i, X_ in Xa.items()}; Xb = {i: [v for v, k in zip(X_, sb) if k] for i, X_ in Xb.items()}
    Ya, Pa, Ca, Aa, Ra, La = Ya[sa], Pa[sa], Ca[sa], Aa[sa], Ra[sa], La[sa]
    Yb, Pb, Cb, Ab, Rb, Lb = Yb[sb], Pb[sb], Cb[sb], Ab[sb], Rb[sb], Lb[sb]
    name = name + "-explicit"
lines = [f"does {name} know when it is right? commit moments in its own thinking, X graded alone",
         f"train: {len(Ya)} moments from {len(set(Ra))} answers ({Ya.mean():.0%} right); "
         f"test: {len(Yb)} moments from {len(set(Rb))} answers ({Yb.mean():.0%} right)",
         f"baselines on test, AUROC (0.5 = coin, 1 = knows): how far in {auroc(-Pb, Yb):.2f} (earlier = right) / "
         f"{auroc(Pb, Yb):.2f} (later = right); consistency (earlier moments said the same X) {auroc(Cb, Yb):.2f}; "
         f"expectation (chance he gave X) {auroc(Lb, Yb):.2f}",
         f"thinking still to come after a RIGHT moment (test): median {int(np.median(Ab[Yb == 1]))} words, "
         f"total {int(Ab[Yb == 1].sum())}",
         "layer  probe AUROC (test)  probe + consistency"]
best = (0, None)
for i in LAY:
    A = torch.stack(Xa[i]).float().numpy(); B = torch.stack(Xb[i]).float().numpy()
    pb, alpha, cv, _ = ridge(A, Ya, B)
    pa2, _, _, _ = ridge(np.c_[A, Ca, Pa], Ya, np.c_[B, Cb, Pb])
    au = auroc(pb, Yb)
    lines.append(f"{i:5d}     {au:.2f}                {auroc(pa2, Yb):.2f}      (alpha {alpha:g})")
    if au > best[0]:
        best = (au, i, pb)
au, i, pb = best
hi = pb >= np.quantile(pb, 0.75)
lines.append(f"best layer {i}: of the test moments the line rates top quarter, {Yb[hi].mean():.0%} are right "
             f"(all moments {Yb.mean():.0%}); bottom quarter {Yb[pb <= np.quantile(pb, 0.25)].mean():.0%}")
print("\n".join(lines))
open(os.path.join(out, f"confidence_{name}.txt"), "w").write("\n".join(lines) + "\n")
np.savez(os.path.join(out, f"confidence_{name}_scores.npz"), pb=pb, Yb=Yb, Pb=Pb, Cb=Cb, Ab=Ab, Rb=Rb, Lb=Lb, layer=i)
print("CONFIDENCE_DONE")
