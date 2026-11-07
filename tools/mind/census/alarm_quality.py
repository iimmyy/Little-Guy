#!/usr/bin/env python3
"""Before adding a second part to the formula (Immy 2026-11-03: when the formula does not equal the expected value, a new
term pushes past the untouched model): how reliable is "does not equal" in long thinking? The formula's own line in c5s
(layer 17, w1[1076] minus c5's row, scaled back by steep * gamma: + fits, - the stated number is off), read at:
- planted edits, the same line clean and edited (planted_edit.py jobs);
- every line end of the clean thinking (false alarms: each line that crosses would get a push).
For thresholds set by a false-alarm share on clean edit spots: how many edits it catches, and alarms per 100 ordinary lines.
usage: alarm_quality.py <jobs.jsonl> <out.txt> c5=<folder> c5s=<folder> [dev=cuda]
"""
import json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from patients import _lfm3
from natural_error import auroc

args = [a for a in sys.argv[1:] if "=" not in a]
jobs_f, out = args
o = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
DEV = o.get("dev", "cuda")
KL, GN = 17, 1076
p0 = _lfm3("cpu", o["c5"]); row0 = p0.model.model.layers[KL].feed_forward.w1.weight[GN].float().clone(); del p0
p = _lfm3(DEV, o["c5s"])
tok, m = p.tok, p.model
meta = json.load(open(os.path.join(o["c5s"], "pam.json")))
gamma, steep = meta["picture_fit"]["gamma"], meta["compressor"]["steep"]
ff = m.model.layers[KL].feed_forward
line = (ff.w1.weight[GN].float().to(DEV) - row0.to(DEV)) / (steep * gamma)
got = {}
ff.register_forward_pre_hook(lambda mod, a: got.__setitem__("x", a[0][0].float()))
by = {}
for j in map(json.loads, open(jobs_f)):
    if j["cond"] in ("clean", "own"):
        by.setdefault(j["pid"], {})[j["cond"]] = j
spots = [v for v in by.values() if len(v) == 2]
cl, ed, ord_ = [], [], []
for k, v in enumerate(spots):
    enc = tok(v["clean"]["prompt"], add_special_tokens=False, return_offsets_mapping=True)
    with torch.no_grad():
        m.model(input_ids=torch.tensor([enc.input_ids], device=DEV))
    r = (got["x"] @ line).cpu().numpy()
    cl.append(float(r[-1]))
    txt = v["clean"]["prompt"]
    start = txt.find("<think>")
    ord_ += [float(r[i]) for i, (a, b) in enumerate(enc.offset_mapping[:-1]) if a > start and "\n" in txt[a:b]]
    ids = tok(v["own"]["prompt"], add_special_tokens=False).input_ids
    with torch.no_grad():
        m.model(input_ids=torch.tensor([ids], device=DEV))
    ed.append(float((got["x"][-1] @ line).cpu()))
    if k % 20 == 0:
        print(f"{k}/{len(spots)}", flush=True)
cl, ed, ord_ = np.array(cl), np.array(ed), np.array(ord_)
depth = np.array([v["own"]["depth"] for v in spots])
L = [f"the formula's line in c5s at {len(spots)} planted edits and {len(ord_)} ordinary line ends of clean thinking (+ fits, - off; story fit test held out: right +0.022, fits -0.371, misfit -0.469)",
     f"  median: clean spot {np.median(cl):+.3f}, edited {np.median(ed):+.3f}, ordinary lines {np.median(ord_):+.3f}; edited lower in {np.mean(ed < cl):.0%} of pairs",
     f"  AUROC edited vs clean {auroc(-ed, -cl):.2f}; deep (300+ words) {auroc(-ed[depth >= 300], -cl[depth >= 300]):.2f}, shallow {auroc(-ed[depth < 300], -cl[depth < 300]):.2f}"]
for fa in (0.05, 0.1, 0.2, 0.3):
    th = np.quantile(cl, fa)
    L.append(f"  alarm if the line is under {th:+.3f} ({fa:.0%} of clean spots): catches {np.mean(ed < th):.0%} of edits (deep {np.mean(ed[depth >= 300] < th):.0%}), fires on {np.mean(ord_ < th) * 100:.1f} of 100 ordinary lines")
open(out, "w").write("\n".join(L) + "\n")
print("\n".join(L))
