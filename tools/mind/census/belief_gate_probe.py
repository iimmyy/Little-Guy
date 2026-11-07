#!/usr/bin/env python3
"""Step 1 of the belief gate (Immy 2026-10-29: "if I believe I have the answer, that's when I stop"; make the belief
so loud he can't ignore it). Read the stream where the choice is made: on the word that ends the sentence carrying an
explicit answer claim ("the answer is X." / "\\boxed{X}"), the next word is either a doubt ("But wait") or the stop.
  belief      a straight line for "X is right" at that word, built on the train answers, scored on the test answers
  inside      a straight line for "still thinking" vs the same kind of word in the reply (the gate must be shut there,
              or the stop word gets pushed inside the answer: last night's leak). Reported as the gap between the
              two groups in spreads, and the overlap (share of reply words that look like thinking)
  next        what he actually wrote next after a RIGHT claim (doubt / stop / other): how often the doubt wins today

usage: belief_gate_probe.py <patient> <train answers> <test answers> <out folder> [dev=mps cut=7000]
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
DEV, CUT = opt.get("dev", "mps"), int(opt.get("cut", 7000))
os.makedirs(out, exist_ok=True)
TAIL = "\n\nEnd with a line: Answer: <your answer>"
OPEN, CLOSE = {"lfm3": ("\n", "</think>")}[name]
MLP = {"lfm3": lambda l: l.feed_forward}[name]
CLAIM = re.compile(r"\\boxed\{([^{}]{1,40})\}|(?:answer is|answer =|answer:)\s*\$?([^\n$]{1,40}?)\$?(?=[.\n]|$)", re.I)
DOUBT = re.compile(r"^\s*(But|Wait|However|Hmm|Let me|Let's|Actually|Alternatively|Double|Check|Hold on)\b", re.I)

p = load(name, DEV)
tok, lay = p.tok, p.model.model.layers
LAY = list(range(0, len(lay), 2))
cap = {}
for i in LAY:
    MLP(lay[i]).register_forward_pre_hook(lambda m, a, i=i: cap.__setitem__(i, a[0][0].float().cpu()))


def claims(r, text, base, inside):
    ms = []
    for m in CLAIM.finditer(text):
        x = next(g for g in m.groups() if g is not None).strip().rstrip(".").strip()
        if not x:
            continue
        e = m.end()
        while e < len(text) and not (text[e] == "\n" or text[e] == "." and (e + 1 == len(text) or text[e + 1] in " \n")):
            e += 1                                        # the word that ends the sentence (not a decimal point)
        nxt = text[e + 1:e + 60] if e < len(text) else ""
        kind = "stop" if inside and e >= len(text.rstrip()) - 2 else "doubt" if DOUBT.match(nxt) else "other"
        ms.append(dict(pos=base + min(e, len(text) - 1), ok=bool(right({**r, "reply": x}, x)), inside=inside, next=kind,
                       words=text[max(0, e - 80):e + 1]))
    return ms


def collect(path):
    X = {i: [] for i in LAY}; meta = []
    rows = [r for r in map(json.loads, open(path)) if r["thinking"]]
    with torch.no_grad():
        for n, r in enumerate(rows):
            pr = tok(tok.apply_chat_template([{"role": "user", "content": r["q"] + TAIL}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=True),
                     add_special_tokens=False).input_ids
            think = OPEN + r["thinking"]
            text = think + CLOSE + r["reply"]
            ms = claims(r, think, 0, True) + (claims(r, r["reply"], len(think) + len(CLOSE), False) if r["finish"] == "stop" else [])
            if not ms:
                continue
            enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
            ids, ends = enc.input_ids[:CUT], np.array([e for _, e in enc.offset_mapping[:CUT]])
            keep = [(int(np.searchsorted(ends, m["pos"] + 1)), m) for m in ms]
            keep = [(k, m) for k, m in keep if k < len(ids)]
            if not keep:
                continue
            cap.clear()
            p.model.model(input_ids=torch.tensor([pr + ids], device=DEV))
            for k, m in keep:
                for i in LAY:
                    X[i].append(cap[i][len(pr) + k].half())
                meta.append({**m, "row": n})
            if (n + 1) % 20 == 0:
                print(f"  {os.path.basename(path)} {n + 1}/{len(rows)}: {len(meta)} claims", flush=True)
    return X, meta


def auroc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, bool)
    if y.all() or not y.any():
        return float("nan")
    r = np.argsort(np.argsort(s)) + 1
    return (r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())


Xa, Ma = collect(src_a)
Xb, Mb = collect(src_b)
ia = np.array([m["inside"] for m in Ma]); ib = np.array([m["inside"] for m in Mb])
ya = np.array([m["ok"] for m in Ma], float); yb = np.array([m["ok"] for m in Mb], float)
nx = [m["next"] for m in Mb if m["inside"] and m["ok"]]


def grams(ms):                                            # control: the wording alone (letter triples of the last 80
    G = np.zeros((len(ms), 4096), np.float32)              # characters), so "he knows" is not just "the words show it"
    for n, m in enumerate(ms):
        w = m["words"].lower()
        for j in range(len(w) - 2):
            G[n, hash(w[j:j + 3]) % 4096] += 1
    return G
pw, _, _, _ = ridge(grams(Ma)[ia], ya[ia], grams(Mb)[ib])
lines = [f"belief gate, step 1 ({name}): claims at the word that ends the claim sentence",
         f"train: {int(ia.sum())} thinking claims ({ya[ia].mean():.0%} right), {int((~ia).sum())} reply claims; "
         f"test: {int(ib.sum())} thinking claims ({yb[ib].mean():.0%} right), {int((~ib).sum())} reply claims",
         f"after a RIGHT thinking claim he wrote next (test): doubt {nx.count('doubt')}, stop {nx.count('stop')}, "
         f"other {nx.count('other')}",
         f"wording alone (letter triples of the claim sentence): belief AUROC {auroc(pw, yb[ib]):.2f}",
         "layer  belief AUROC (thinking claims, test)  inside: gap in spreads / reply words that look like thinking"]
for i in LAY:
    A = torch.stack(Xa[i]).float().numpy(); B = torch.stack(Xb[i]).float().numpy()
    pb, _, _, _ = ridge(A[ia], ya[ia], B[ib])
    pi, _, _, _ = ridge(A, ia.astype(float), B)
    sd = np.sqrt((pi[ib].var() + pi[~ib].var()) / 2)
    gap = (pi[ib].mean() - pi[~ib].mean()) / sd
    over = float((pi[~ib] >= np.quantile(pi[ib], 0.05)).mean())
    lines.append(f"{i:5d}     {auroc(pb, yb[ib]):.2f}                                 {gap:5.1f} / {over:.0%}")
print("\n".join(lines))
open(os.path.join(out, f"belief_gate_{name}.txt"), "w").write("\n".join(lines) + "\n")
json.dump({"train": Ma, "test": Mb}, open(os.path.join(out, f"belief_gate_{name}_meta.json"), "w"))
print("BELIEF_GATE_DONE")
