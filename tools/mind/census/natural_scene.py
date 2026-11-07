#!/usr/bin/env python3
"""Does the finished formula hear his REAL mistakes? (The one test that matters: planted swaps are not his own errors.) Reading only. Every term of the formula, on the untouched stream at layer 16.
Material: trap sets, thinking on, finished runs, minus items read by hand as a wrong key or ambiguous. Each run read at the last line of his thinking that states his final answer, and, as a control for the
kind of line, right runs read the same way. Reported overall, per set, and per line kind (a commit line vs a working line), because the kind of line moves the read.
usage: natural_scene.py <hf> <lines16.npz> <out.txt> <brake folder> <run names, comma separated> [dev=cuda per_right=200]
"""
import glob, json, os, re, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from patients import _lfm3
from natural_error import auroc

args = [a for a in sys.argv[1:] if "=" not in a]
opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
hf, linesf, out, B, names = args[:5]
WHERE = opt.get("where", "last")                     # where=first: the line where he first writes that number, close to where the mistake is made
DEV = opt.get("dev", "cuda")
EXCLUDE = {"nluds-0778", "nluds-1542", "nluds-1608", "nluds-1610", "nluds-0808", "nluds-1606", "nluds-2265", "nluds-2269"}
ln = np.load(linesf)
TERMS = ("size", "demand", "whole", "sign")
NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
COMMIT = re.compile(r"(final|answer|boxed|I.ll|We.ll|produce)", re.I)


def final(reply):
    k = reply.rfind("\\boxed{")
    if k >= 0:
        i, d = k + 7, 1
        while i < len(reply) and d:
            d += {"{": 1, "}": -1}.get(reply[i], 0)
            i += 1
        return reply[k + 7:i - 1]
    ms = NUM.findall(reply)
    return ms[-1] if ms else None


def value(s):
    if s is None:
        return None
    s = s.replace(",", "").replace("\\dfrac", "\\frac").replace("\\,", "").replace("$", "")
    mx = re.search(r"(\d+)\s*\\frac\{(\d+)\}\{(\d+)\}", s)
    if mx:
        return int(mx.group(1)) + int(mx.group(2)) / int(mx.group(3)), [f"{mx.group(2)}/{mx.group(3)}", f"\\frac{{{mx.group(2)}}}{{{mx.group(3)}}}"]
    fr = re.search(r"\\frac\{(\d+)\}\{(\d+)\}", s) or re.search(r"(\d+)\s*/\s*(\d+)", s)
    if fr:
        return int(fr.group(1)) / int(fr.group(2)), [f"{fr.group(1)}/{fr.group(2)}", f"\\frac{{{fr.group(1)}}}{{{fr.group(2)}}}"]
    m_ = NUM.search(s)
    return (float(m_.group()), [m_.group()]) if m_ else None


items = []
for nm in names.split(","):
    for f in sorted(glob.glob(os.path.join(B, f"brake-*-{nm}.jsonl"))):
        for t in map(json.loads, open(f)):
            if t["finish"] != "stop" or not t["thinking"] or t["id"] in EXCLUDE:
                continue
            got = value(final(t["reply"]))
            try:
                A = float(str(t["answer"]).replace(",", ""))
            except ValueError:
                continue
            if got is None:
                continue
            v, forms = got
            ends = []
            for fm in forms:
                for mm in re.finditer(rf"(?<![\d.]){re.escape(fm)}(?![\d.])", t["thinking"]):
                    nl = t["thinking"].find("\n", mm.end())
                    ends.append(nl if nl > 0 else len(t["thinking"]))
            if ends:
                items.append((t, max(ends) if WHERE == "last" else min(ends), abs(v - A) < 1e-6, os.path.basename(f).split("-")[1], nm, v))
wrong = [x for x in items if not x[2]]
right = [x for x in items if x[2]]
rng = np.random.RandomState(0)
right = [right[i] for i in rng.permutation(len(right))[:int(opt.get("per_right", 200))]]
p_ = _lfm3(DEV, hf)
tok, m = p_.tok, p_.model
got = {}
m.model.layers[16].feed_forward.register_forward_pre_hook(lambda mod, a: got.__setitem__("x", a[0][0, -1].float().cpu().numpy()))


def rd(it):
    t, end = it[0], it[1]
    head = tok.apply_chat_template([{"role": "user", "content": t["q"]}], tokenize=False, add_generation_prompt=True, enable_thinking=True) + "\n"
    ids = tok(head + t["thinking"][:end], add_special_tokens=False).input_ids
    with torch.no_grad():
        m.model(input_ids=torch.tensor([ids], device=DEV))
    return got["x"].astype(np.float64)


Xw, Xr = np.stack([rd(x) for x in wrong]), np.stack([rd(x) for x in right])
sc = lambda X: {tn: X @ ln[tn] for tn in TERMS}
Sw, Sr = sc(Xw), sc(Xr)
Ew = sum(np.maximum(Sw[tn], 0) ** 2 for tn in TERMS)
Er = sum(np.maximum(Sr[tn], 0) ** 2 for tn in TERMS)
L = [f"real mistakes {len(wrong)} vs right {len(right)}, read at the {WHERE} line of his thinking stating his final answer"]
for fa in (0.02, 0.05, 0.1, 0.2):
    th_ = np.quantile(Er, 1 - fa)
    L.append(f"  energy line that {fa:.0%} of right answers cross ({th_:.3f}): catches {np.mean(Ew > th_):.0%} of real mistakes")
kind = lambda it: "commit" if COMMIT.search(it[0]["thinking"][max(0, it[1] - 120):it[1]]) else "working"
for tn in TERMS:
    L.append(f"  {tn:7s} real median {np.median(Sw[tn]):+.3f}, right {np.median(Sr[tn]):+.3f}, AUROC {auroc(Sw[tn], Sr[tn]):.3f}")
L.append(f"  energy  real median {np.median(Ew):.3f}, right {np.median(Er):.3f}, AUROC {auroc(Ew, Er):.3f}")
for st in ("tricks", "herring", "walls"):
    a = np.array([x[3] == st for x in wrong]); b = np.array([x[3] == st for x in right])
    if a.sum() and b.sum():
        L.append(f"  {st:8s} wrong {a.sum():3d} right {b.sum():3d}: energy AUROC {auroc(Ew[a], Er[b]):.3f}; " + ", ".join(f"{tn} {auroc(Sw[tn][a], Sr[tn][b]):.3f}" for tn in TERMS))
for kd in ("commit", "working"):
    a = np.array([kind(x) == kd for x in wrong]); b = np.array([kind(x) == kd for x in right])
    if a.sum() and b.sum():
        L.append(f"  {kd:8s} lines, wrong {a.sum():3d} right {b.sum():3d}: energy AUROC {auroc(Ew[a], Er[b]):.3f}; " + ", ".join(f"{tn} {auroc(Sw[tn][a], Sr[tn][b]):.3f}" for tn in TERMS))
nonwhole = np.array([abs(x[5] - round(x[5])) > 1e-6 for x in wrong])
if nonwhole.sum():
    L.append(f"  non-whole hand-ins {nonwhole.sum()}: whole wall real median {np.median(Sw['whole'][nonwhole]):+.3f}, right {np.median(Sr['whole']):+.3f}, AUROC {auroc(Sw['whole'][nonwhole], Sr['whole']):.3f}")
if (~nonwhole).sum():
    L.append(f"  whole-number mistakes {(~nonwhole).sum()}: demand AUROC {auroc(Sw['demand'][~nonwhole], Sr['demand']):.3f}, size AUROC {auroc(Sw['size'][~nonwhole], Sr['size']):.3f}, energy {auroc(Ew[~nonwhole], Er):.3f}")
order = np.argsort(-Ew)
for i in order:
    x = wrong[i]
    L.append(f"    E {Ew[i]:.2f} " + " ".join(f"{tn} {Sw[tn][i]:+.2f}" for tn in TERMS) + f" {x[4]:18s} {x[0]['id']:12s} key {str(x[0]['answer'])[:8]} said {x[5]:.4g} | {x[0]['thinking'][max(0, x[1] - 70):x[1]]!r}")
open(out, "w").write("\n".join(L) + "\n")
print("\n".join(L))
