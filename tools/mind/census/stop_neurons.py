#!/usr/bin/env python3
"""What makes him stop? (Immy 2026-10-29: "look at his neurons when he gives his answer, what makes him stop there".)
Two kinds of moment in the untouched model's own thinking, both with the right answer in hand:
  stop       the last word of thinking in a finished, right answer (the next word he wrote was the close word)
  doubt      the word ending a sentence that states the right answer, where the next thing he wrote was a doubt
             ("But wait", "Let's double-check", ...)
On each, every MLP neuron's activation (input to w2) is read, with its push on the close word: activation x (its
output column, through the post-FFN norm weight, the final norm weight and the close word's output row). The norms'
per-word scale is left out, so pushes are comparable within a word, roughly across words.
  per layer  how well the layer's neurons tell stop from doubt (straight line, built on train, scored on test)
  neurons    the ones whose push on the close word differs most between stop and doubt (train), with the same
             difference on test: the stop circuit he already has, if there is one
  close      the chance he gave the close word at each kind of moment

usage: stop_neurons.py <patient> <train answers> <test answers> <out folder> [dev=mps cut=7000 top=25]
"""
import json, os, re, sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../expert"))
ARGS, sys.argv[1:] = sys.argv[1:], []                     # grade.py grades its own arguments on import
from grade import right, answer_line
from patients import load
from probe_math import ridge

name, src_a, src_b, out = ARGS[:4]
opt = dict(a.split("=", 1) for a in ARGS[4:])
DEV, CUT, TOP = opt.get("dev", "mps"), int(opt.get("cut", 7000)), int(opt.get("top", 25))
os.makedirs(out, exist_ok=True)
TAIL = "\n\nEnd with a line: Answer: <your answer>"
OPEN, CLOSE = "\n", "</think>"
CLAIM = re.compile(r"\\boxed\{([^{}]{1,40})\}|(?:answer is|answer =|answer:)\s*\$?([^\n$]{1,40}?)\$?(?=[.\n]|$)", re.I)
HANDOFF = re.compile(r"^\s*(I'll (produce|write|output|give|present|answer)|We need to (end|output|produce|write|present)|Now (we need to )?(output|produce|write)|So (the )?final (output|line)|Thus,? final\.?$|Thus,? I'll|End with|Final answer|So final answer|Thus final answer)", re.I)
FORK = opt.get("fork", "0") == "1"   # same moment both sides: right answer just stated, next sentence = handoff vs doubt
DOUBT = re.compile(r"^\s*(But|Wait|However|Hmm|Let me|Let's|Actually|Alternatively|Double|Check|Hold on)\b", re.I)

p = load(name, DEV)
tok, m_ = p.tok, p.model
lay = m_.model.layers
L = len(lay)
cid = tok.convert_tokens_to_ids(CLOSE)
g_final = m_.model.embedding_norm.weight.float()
u_close = m_.lm_head.weight[cid].float() * g_final         # what the close word listens for in the final stream
PUSH = []                                                  # per layer: each neuron's push on the close word per unit
for l in lay:
    w2 = l.feed_forward.w2.weight.float()                   # hidden x inter
    g = getattr(l, "post_ffn_norm", None) or getattr(l, "post_feedforward_layernorm", None)
    col = w2 * g.weight.float()[:, None] if g is not None else w2
    PUSH.append((u_close @ col).cpu())
cap = {}
for i, l in enumerate(lay):
    l.feed_forward.w2.register_forward_pre_hook(lambda mod, a, i=i: cap.__setitem__(i, a[0][0]))


def moments(r, think):
    ms = []
    if not FORK and r["finish"] == "stop" and right(r, answer_line(r["reply"], True)):
        ms.append(("stop", len(think.rstrip()) - 1))
    for m in CLAIM.finditer(think):
        x = next(g for g in m.groups() if g is not None).strip().rstrip(".").strip()
        if not x or not right({**r, "reply": x}, x):
            continue
        e = m.end()
        while e < len(think) and not (think[e] == "\n" or think[e] == "." and (e + 1 == len(think) or think[e + 1] in " \n")):
            e += 1
        if e < len(think.rstrip()) - 2 and DOUBT.match(think[e + 1:e + 60]):
            ms.append(("doubt", e))
        elif FORK and e < len(think.rstrip()) - 2 and HANDOFF.match(think[e + 1:e + 60]):
            ms.append(("stop", e))                       # "stop" here = he turns to handing in the answer
    return ms


def collect(path):
    A = {i: [] for i in range(L)}; kind, pc, rid = [], [], []
    rows = [r for r in map(json.loads, open(path)) if r["thinking"]]
    with torch.no_grad():
        for n, r in enumerate(rows):
            think = OPEN + r["thinking"]
            ms = moments(r, think)
            if not ms:
                continue
            pr = tok(tok.apply_chat_template([{"role": "user", "content": r["q"] + TAIL}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=True),
                     add_special_tokens=False).input_ids
            enc = tok(think, add_special_tokens=False, return_offsets_mapping=True)
            ids, ends = enc.input_ids[:CUT], np.array([e for _, e in enc.offset_mapping[:CUT]])
            keep = [(k_, int(np.searchsorted(ends, c + 1))) for k_, c in ms]
            keep = [(k_, t) for k_, t in keep if t < len(ids)]
            if not keep:
                continue
            cap.clear()
            hid = m_.model(input_ids=torch.tensor([pr + ids], device=DEV)).last_hidden_state[0]
            at = torch.tensor([len(pr) + t for _, t in keep], device=DEV)
            lp = torch.log_softmax(m_.lm_head(hid[at]).float(), -1)[:, cid].exp().cpu().numpy()
            for i in range(L):
                a = cap[i][at].float().cpu()
                A[i].append(a.half())
            kind += [k_ for k_, _ in keep]; pc += list(lp); rid += [n] * len(keep)
            if (n + 1) % 20 == 0:
                print(f"  {os.path.basename(path)} {n + 1}/{len(rows)}: stop {kind.count('stop')} doubt {kind.count('doubt')}", flush=True)
    return {i: torch.cat(v).float() for i, v in A.items()}, np.array(kind), np.array(pc), np.array(rid)


def auroc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, bool)
    r = np.argsort(np.argsort(s)) + 1
    return (r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())


Aa, ka, pa, ra = collect(src_a)
Ab, kb, pb, rb = collect(src_b)
ya, yb = ka == "stop", kb == "stop"
lines = [f"what makes {name} stop: right answer in hand, then STOP vs DOUBT",
         f"train: stop {ya.sum()} doubt {(~ya).sum()}; test: stop {yb.sum()} doubt {(~yb).sum()}",
         f"chance he gave the close word (test): at stop median {np.median(pb[yb]):.3f}, at doubt median {np.median(pb[~yb]):.4f}",
         "layer  neurons tell stop from doubt (AUROC test)   total push on close word: stop / doubt (test, median)"]
cand = []
for i in range(L):
    sc, _, _, _ = ridge(Aa[i].numpy(), ya.astype(float), Ab[i].numpy())
    ca, cb = Aa[i] * PUSH[i], Ab[i] * PUSH[i]               # each neuron's push on the close word at each moment
    lines.append(f"{i:5d}     {auroc(sc, yb):.2f}                                  "
                 f"{float(cb[yb].sum(1).median()):+7.2f} / {float(cb[~yb].sum(1).median()):+7.2f}")
    da = ca[ya].mean(0) - ca[~ya].mean(0)                    # how much more it pushes the close word at stop (train)
    sd = ca.std(0) + 1e-6
    db = cb[yb].mean(0) - cb[~yb].mean(0)
    for j in torch.argsort(-da.abs())[:TOP].tolist():
        cand.append((float(da[j]), float(db[j]), i, j, float(da[j] / sd[j]),
                     float(Aa[i][ya, j].mean()), float(Aa[i][~ya, j].mean())))
cand.sort(key=lambda c: -abs(c[0]))
lines.append(f"top neurons by extra push on the close word at stop (train), same number on test:")
lines.append("  layer neuron   push diff train / test   (in spreads)   activation stop / doubt (train)")
for da, db, i, j, z, s_, d_ in cand[:40]:
    lines.append(f"  {i:4d} {j:6d}     {da:+7.3f} / {db:+7.3f}      ({z:+.1f})        {s_:+.3f} / {d_:+.3f}")
print("\n".join(lines))
tag = name + ("-fork" if FORK else "")
open(os.path.join(out, f"stop_neurons_{tag}.txt"), "w").write("\n".join(lines) + "\n")
torch.save({"cand": cand, "PUSH": PUSH}, os.path.join(out, f"stop_neurons_{tag}.pt"))
print("STOP_NEURONS_DONE")
