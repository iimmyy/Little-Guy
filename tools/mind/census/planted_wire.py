#!/usr/bin/env python3
"""Is the principle wired? (Immy 2026-11-03: "make sure that our principle even works, that it's all wired really well.")
At each planted edit (planted_edit.py jobs), the same text clean and edited, the last word of the line, link by link:
1. Picture: the swap-trained scene error readout (fit test, index 14) on the untouched model's stream. Does it break?
2. Key: c5's key (layer 17 #1076) over threshold, in c5 and c5s. In c5s it should fall on the edit (trust drops).
3. Next word: the chance he opens doubt, and the chance he opens a hand-in, in each model.
Link 4 (the final answer) comes from planted_edit.py's rows. Split by depth (words before the edit) and by whether the
line computes the number in plain sight. The readout was learned on story problems with a format line; here it is applied
unchanged to long bare thinking, so link 1 is itself a test of how far the picture travels.
usage: planted_wire.py <jobs.jsonl> <fit-test.npz> <out.txt> plain=<folder> c5=<folder> c5s=<folder> [dev=cuda]
"""
import json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from patients import _lfm3
from natural_error import fit_pred, auroc

DOUBT = [1938, 2375, 3911, 7731, 8952]
HAND = [49, 1672, 2083, 8175, 21009]
KL, BN, IDX = 17, 1076, 14
args = [a for a in sys.argv[1:] if "=" not in a]
jobs_f, fit_f, out = args
pairs = [a.split("=", 1) for a in sys.argv[1:] if "=" in a]
DEV = dict(pairs).get("dev", "cuda")
models = [(n, p) for n, p in pairs if n != "dev"]
by = {}
for j in map(json.loads, open(jobs_f)):
    if j["cond"] in ("clean", "own"):
        by.setdefault(j["pid"], {})[j["cond"]] = j
spots = [v for v in by.values() if len(v) == 2]
depth = np.array([v["own"]["depth"] for v in spots])
shown = np.array([bool(v["own"]["shown"]) for v in spots])
GROUPS = [("all", np.ones(len(spots), bool)), ("shallow <300", depth < 300), ("deep 300+", depth >= 300),
          ("shown on the line", shown), ("not shown", ~shown)]
L = [f"wiring at {len(spots)} planted edits (depth median {int(np.median(depth))} words; {int((depth >= 300).sum())} deep, {int(shown.sum())} shown on the line)"]
res = {}
Xstream = {}
for name, path in models:
    p_ = _lfm3(DEV, path)
    tok, m = p_.tok, p_.model
    got = {}
    h1 = m.model.layers[KL].feed_forward.w1.register_forward_hook(lambda mod, a, o: got.__setitem__("g", float(o[0, -1, BN].float())))
    h2 = m.model.layers[IDX - 1].register_forward_hook(lambda mod, a, o: got.__setitem__("x", (o[0] if isinstance(o, tuple) else o)[0, -1].float().cpu().numpy()))
    R = {c: {"g": [], "d": [], "h": [], "x": []} for c in ("clean", "own")}
    for k, v in enumerate(spots):
        for c in ("clean", "own"):
            ids = tok(v[c]["prompt"], add_special_tokens=False).input_ids
            with torch.no_grad():
                hs = m.model(input_ids=torch.tensor([ids], device=DEV)).last_hidden_state[0, -1]
                pr = torch.softmax(m.lm_head(hs).float(), -1)
            R[c]["g"].append(got["g"]); R[c]["x"].append(got["x"])
            R[c]["d"].append(float(pr[DOUBT].sum())); R[c]["h"].append(float(pr[HAND].sum()))
        if k % 20 == 0:
            print(f"{name} {k}/{len(spots)}", flush=True)
    h1.remove(); h2.remove()
    res[name] = {c: {kk: np.array(vv) for kk, vv in d.items()} for c, d in R.items()}
    del m
    torch.cuda.empty_cache()

fz = np.load(fit_f, allow_pickle=True)
XF, A, B, C = fz["X"], fz["A"].astype(float), fz["b"].astype(float), fz["c"].astype(float)
n = len(A)
err = np.abs(np.log(np.stack([A, B, C], 1) / A[:, None])).reshape(-1)
Xf = XF[:, :, IDX].reshape(n * 3, -1).astype(np.float64)
base = models[0][0]
pc = fit_pred(Xf, err, res[base]["clean"]["x"].astype(np.float64), 1e3)
pe = fit_pred(Xf, err, res[base]["own"]["x"].astype(np.float64), 1e3)
L.append(f"link 1, picture (readout on {base}: 0 = fits, ~0.7 = a swap's typical error; story fit test: right 0.19, swaps 0.66):")
for lab, s in GROUPS:
    if s.any():
        L.append(f"  {lab:18s} n {int(s.sum()):3d}  clean {np.median(pc[s]):.3f}  edited {np.median(pe[s]):.3f}  edited reads higher in {np.mean(pe[s] > pc[s]):.0%}  AUROC {auroc(pe[s], pc[s]):.2f}")
L.append("link 2, key over threshold (clean -> edited), trust should fall on the edit:")
for name, _ in models:
    for lab, s in GROUPS:
        if s.any():
            c_, e_ = res[name]["clean"]["g"][s], res[name]["own"]["g"][s]
            L.append(f"  {name:6s} {lab:18s} n {int(s.sum()):3d}  {np.mean(c_ > 0):.0%} -> {np.mean(e_ > 0):.0%}  (gate input median {np.median(c_):+.2f} -> {np.median(e_):+.2f})")
L.append("link 3, next word, chance of doubt and of a hand-in (clean -> edited):")
for name, _ in models:
    for lab, s in GROUPS:
        if s.any():
            r = res[name]
            L.append(f"  {name:6s} {lab:18s} n {int(s.sum()):3d}  doubt {np.mean(r['clean']['d'][s]):.3f} -> {np.mean(r['own']['d'][s]):.3f}  hand-in {np.mean(r['clean']['h'][s]):.3f} -> {np.mean(r['own']['h'][s]):.3f}")
open(out, "w").write("\n".join(L) + "\n")
print("\n".join(L))
