#!/usr/bin/env python3
"""Picture-fit key (Immy 2026-11-02 night: "what made me trust it was ... it fits the picture"; one alarm system).
Bolt-on on top of c5. c5's key (layer 17 MLP neuron #1076) reads belief ("the right answer is on the page") and, over its
threshold, raises the private wire that ducks the doubt crowd in layers 18-23. This adds a second input to the SAME key:
  belief'(x) = belief(x) + gamma * fit(x)
fit(x) is a line read at the key's own input (layer 17, after its ffn norm) at the fork after he states his answer:
target 0 where the stated number fits the picture (the right answer; a wrong one on the same side of the story's numbers)
and -1 where it lands on the other side (does not fit), 0 on ordinary thinking words. So a misfit holds the key under its
threshold (doubt is let through: "wait, something's not right"); a fit leaves c5 as it was. Only the key's gate row and
up row change (rank one each); the wire and the crowd are untouched. No training: one ridge line from his own states.
gamma (auto): the smallest that puts the median misfit fork of the build half under the threshold with margin m.

Built on 2/3 of the forks (by problem); the other 1/3 is the held-out check of the key. The dinner problem is never in it.

target=formula (Immy's scene formula, 2026-11-03; scene_pin.py: the error |log(said / true)| is read from his stream at
AUROC ~0.9 for wrong-but-fits AND misfits): the line's target at a fork is tau - |log(stated / right answer)| (tau=0.2,
about x1.22): + tau at the right answer (the scene holds -> trust a little more), negative the further a stated number
is off (it breaks -> the key is held under its threshold), 0 at ordinary words. Both polarities act only through c5's
key (ducking or not ducking): nothing is lifted. gamma auto puts the median WRONG fork (b and c) under the threshold.
usage: pam_fit.py <c5 folder> <problems.jsonl> <thinking.jsonl> <out folder> [margin=0.1 gamma=auto ord=20 balance=0|1 target=side|formula tau=0.2 dev=cuda]
"""
import json, os, random, shutil, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from patients import _lfm3
from situation_probe import TAIL
from fit_test import forks

c5, src, think_f, out = sys.argv[1:5]
opt = dict(a.split("=", 1) for a in sys.argv[5:])
DEV, MARGIN, NORD = opt.get("dev", "cuda"), float(opt.get("margin", 0.1)), int(opt.get("ord", 20))
FORMULA, TAU = opt.get("target", "side") == "formula", float(opt.get("tau", 0.2))
P = json.load(open(os.path.join(c5, "pam.json")))["compressor"]
GK, GN, CTH, STEEP = P["key_layer"], P["key_neuron"], P["threshold"], P["steep"]
p = _lfm3(DEV, c5)
tok, m = p.tok, p.model
ff = m.model.layers[GK].feed_forward
x_at = {}
ff.w1.register_forward_hook(lambda mod, a, o: x_at.__setitem__("x", a[0][0].float()))     # the key's own input
items = forks(src, think_f)
rng = random.Random(0)
X, Y, V, G = [], [], [], []                     # V: 0 a / 1 b / 2 c / 3 ordinary; G: problem index
for k, (iid, text, think, A, g, (s0, s1, cut), (b, c, r)) in enumerate(items):
    head = tok.apply_chat_template([{"role": "user", "content": text + TAIL}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=True) + "\n"
    for vi, v in enumerate((A, b, c)):
        ids = tok(head + think[:s0] + str(v) + think[s1:cut], add_special_tokens=False).input_ids
        with torch.no_grad():
            m.model(input_ids=torch.tensor([ids], device=DEV))
        yv = (TAU - abs(np.log(v / A))) if FORMULA else (-1.0 if vi == 2 else 0.0)
        X.append(x_at["x"][-1].cpu()); Y.append(yv); V.append(vi); G.append(k)
        if vi == 0:                             # ordinary thinking words before the fork, unswapped
            n_head = len(tok(head, add_special_tokens=False).input_ids)
            pos = list(range(n_head, len(ids) - 2))
            for j in rng.sample(pos, min(NORD, len(pos))):
                X.append(x_at["x"][j].cpu()); Y.append(0.0); V.append(3); G.append(k)
    if k % 100 == 0:
        print(f"{k}/{len(items)}", flush=True)
X, Y, V, G = torch.stack(X).numpy().astype(np.float64), np.array(Y), np.array(V), np.array(G)
build = G % 3 != 2                              # 2/3 of problems build, 1/3 held out
w1, w3 = ff.w1.weight, ff.w3.weight
gate0, up0 = w1[GN].float().cpu().numpy(), w3[GN].float().cpu().numpy()
belief = X @ up0                                 # c5's own belief read at these words (its up row)
gin = X @ gate0                                  # its gate input: steep * (belief - threshold); fires when > 0


def ridge0(Xa, y, alphas=(1, 10, 100, 1e3, 1e4), w=None):
    """no intercept, scale-normalised, strength by 5 folds (same as c5's key fit); w = row weights"""
    sc = np.sqrt((Xa ** 2).mean(0)) + 1e-6
    Z = Xa / sc
    if w is not None:                              # weighted least squares = rows scaled by sqrt(weight)
        sw = np.sqrt(w / w.mean())
        Z, y = Z * sw[:, None], y * sw
    f = np.arange(len(y)) % 5
    best = None
    for al in alphas:
        err = sum(((Z[f == i] @ np.linalg.solve(Z[f != i].T @ Z[f != i] + al * np.eye(Z.shape[1]), Z[f != i].T @ y[f != i]) - y[f == i]) ** 2).sum()
                  for i in range(5))
        if best is None or err < best[0]:
            best = (err, al)
    return np.linalg.solve(Z.T @ Z + best[1] * np.eye(Z.shape[1]), Z.T @ y) / sc


# balance=1 (c5f2): each group (right, fits, misfit, ordinary words) carries the same total weight, so the line has to
# tell misfit from fit and leave right answers and ordinary words at 0 (c5f: ordinary words outnumbered forks 20:1 and
# the line learned "a swapped number", firing on fits as much as misfits)
BAL = opt.get("balance") == "1"
wts = np.array([1.0 / (V[build] == v).sum() for v in V[build]]) if BAL else None
wf = ridge0(X[build], Y[build], w=wts)
fit = X @ wf
names = ["a right", "b wrong, fits", "c wrong, misfit", "ordinary words"]
L = [f"picture-fit key on c5 (layer {GK} #{GN}, threshold {CTH}); {len(items)} forks, build {len(set(G[build]))} problems, "
     f"held out {len(set(G[~build]))}"]
for part, sel in (("build", build), ("held out", ~build)):
    L.append(f"  {part}: fit line median per group: " + ", ".join(f"{names[v]} {np.median(fit[sel & (V == v)]):+.3f}" for v in range(4)))
    L.append(f"  {part}: c5 belief median per group: " + ", ".join(f"{names[v]} {np.median(belief[sel & (V == v)]):+.3f}" for v in range(4))
             + f"; key over threshold: " + ", ".join(f"{names[v][:1]} {(gin[sel & (V == v)] > 0).mean():.0%}" for v in range(4)))
gm = opt.get("gamma", "auto")
if gm == "auto":
    # median misfit fork's gate input pushed to -steep * margin (belief margin under the threshold)
    wrong = (V == 1) | (V == 2) if FORMULA else (V == 2)
    gc, fc = gin[build & wrong], fit[build & wrong]
    gamma = max(0.0, float(np.median(gc) / STEEP + MARGIN) / max(float(-np.median(fc)), 1e-6))
else:
    gamma = float(gm)
ng = gin + STEEP * gamma * fit
L.append(f"gamma {gamma:.3f} ({gm})")
for part, sel in (("build", build), ("held out", ~build)):
    L.append(f"  {part}: key over threshold, c5 -> c5 + fit: " + ", ".join(
        f"{names[v]} {(gin[sel & (V == v)] > 0).mean():.0%} -> {(ng[sel & (V == v)] > 0).mean():.0%}" for v in range(4)))
with torch.no_grad():
    w3[GN] = torch.tensor(up0 + gamma * wf, dtype=w3.dtype, device=w3.device)
    w1[GN] = torch.tensor(gate0 + STEEP * gamma * wf, dtype=w1.dtype, device=w1.device)
os.makedirs(out, exist_ok=True)
m.save_pretrained(out)
tok.save_pretrained(out)
for f in os.listdir(c5):
    if f.endswith((".py", ".jinja", ".json", ".txt")) and not f.endswith(".index.json") and not os.path.exists(os.path.join(out, f)):
        shutil.copy(os.path.join(c5, f), out)
meta = json.load(open(os.path.join(c5, "pam.json")))
meta["picture_fit"] = {"target": "formula" if FORMULA else "side", "tau": TAU, "balance": BAL, "gamma": gamma, "margin": MARGIN, "forks": len(items), "rows": "layer %d w1[%d] and w3[%d]" % (GK, GN, GN)}
json.dump(meta, open(os.path.join(out, "pam.json"), "w"), indent=1)
open(os.path.join(out, "pam_fit_report.txt"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
print("SAVED", out)
