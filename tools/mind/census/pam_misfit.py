#!/usr/bin/env python3
"""Misfit brake on c5's wire (Immy 2026-11-02 night: trust = the answer fits the picture; a misfit is "wait, something's
not right"; one alarm system).
Fit test (fit_test.py): whether a stated number fits his picture is readable at the fork, clearest at index 9 (0.74 vs
surface 0.53), but at c5's key (layer 17) a line mostly sees "a wrong number" (pam_fit.py, c5f/c5f2). So the brake sits
where the misfit is clearest: one quiet existing MLP neuron in layer L (default 9) is rewired
  gate = steep * (misfit(x) - th), up = misfit(x), down = -k on c5's private wire
misfit(x) is a ridge line at that neuron's own input, groups weighted equally: 1 at misfit forks (c), 0 at right (a),
wrong-but-fits (b) and ordinary thinking words. Over threshold it writes a negative onto the wire that c5's key raises
at forks, so the doubt crowd is not ducked: the "wait" comes through. Below threshold SiLU keeps it silent. Layers
between L and 17 barely read the wire (c5 chose the least-read dimension). c5's key, wire and crowd are untouched.
k is calibrated (stepped through the sandwich norm, like c5's write) so the wire drop at misfit forks equals c5's
rise there. th is the build half's 90th percentile of misfit(x) at fits and right answers (false alarms ~10% by design).
Built on 2/3 of the forks by problem; the held-out 1/3 reports firing rates and the wire. The dinner problem is never in it.

Layer 9 is right before layer 10, which resets 397 (lane_carry.py, 2026-11-03): c5m2 at layer 9 broke; use layer 11+.
usage: pam_misfit.py <c5 folder> <problems.jsonl> <thinking.jsonl> <out folder> [layer=9 steep=30 ord=20 drop= dev=cuda]
drop=<raw wire drop> (c5m2): instead of only cancelling c5's rise, push the wire the other way by that much.
"""
import json, os, random, shutil, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from patients import _lfm3
from situation_probe import TAIL
from fit_test import forks

DOUBT_IDS = [1938, 2375, 3911, 7731, 8952]       # c5's doubt openers (pam.json)


def ridge0(Xa, y, alphas=(1, 10, 100, 1e3, 1e4), w=None):
    """no intercept (these models have no constant input; the fit finds its own offset carriers), scale-normalised,
    strength by 5 folds, optional row weights (same as pam_fit.py / c5's key)"""
    sc = np.sqrt((Xa ** 2).mean(0)) + 1e-6
    Z = Xa / sc
    if w is not None:
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

c5, src, think_f, out = sys.argv[1:5]
opt = dict(a.split("=", 1) for a in sys.argv[5:])
DEV, LB, STEEP, NORD = opt.get("dev", "cuda"), int(opt.get("layer", 9)), float(opt.get("steep", 30)), int(opt.get("ord", 20))
P = json.load(open(os.path.join(c5, "pam.json")))["compressor"]
GK, WIRE = P["key_layer"], P["wire"]
p = _lfm3(DEV, c5)
tok, m = p.tok, p.model
lay = m.model.layers
ff = lay[LB].feed_forward
got = {}
ff.w1.register_forward_hook(lambda mod, a, o: got.__setitem__("x", a[0][0].float()))          # the brake's own input
ff.register_forward_hook(lambda mod, a, o: got.__setitem__("mlp", o[0].float()))
lay[GK].register_forward_hook(lambda mod, a, o: got.__setitem__("wire", (o[0] if isinstance(o, tuple) else o)[0, :, WIRE].float()))
items = forks(src, think_f)
rng = random.Random(0)
seqs, X, V, G, POS = [], [], [], [], []
for k, (iid, text, think, A, g, (s0, s1, cut), (b, c, r)) in enumerate(items):
    head = tok.apply_chat_template([{"role": "user", "content": text + TAIL}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=True) + "\n"
    n_head = len(tok(head, add_special_tokens=False).input_ids)
    for vi, v in enumerate((A, b, c)):
        ids = tok(head + think[:s0] + str(v) + think[s1:cut], add_special_tokens=False).input_ids
        with torch.no_grad():
            m.model(input_ids=torch.tensor([ids], device=DEV))
        seqs.append((ids, len(ids) - 1, vi, k))
        X.append(got["x"][-1].cpu()); V.append(vi); G.append(k)
        if vi == 0:
            pos = list(range(n_head, len(ids) - 2))
            for j in rng.sample(pos, min(NORD, len(pos))):
                X.append(got["x"][j].cpu()); V.append(3); G.append(k)
                seqs.append((ids, j, 3, k))
    if k % 100 == 0:
        print(f"capture {k}/{len(items)}", flush=True)
X, V, G = torch.stack(X).numpy().astype(np.float64), np.array(V), np.array(G)
build = G % 3 != 2
Y = (V == 2).astype(float)
w = np.array([1.0 / (V[build] == v).sum() for v in V[build]])
line = ridge0(X[build], Y[build], w=w)
mis = X @ line
th = float(np.quantile(mis[build & ((V == 0) | (V == 1))], 0.90))
names = ["a right", "b wrong, fits", "c wrong, misfit", "ordinary words"]
L = [f"misfit brake on c5's wire {WIRE}: layer {LB}, {len(items)} forks (build {len(set(G[build]))} problems, held out {len(set(G[~build]))})"]
for part, sel in (("build", build), ("held out", ~build)):
    L.append(f"  {part}: misfit line median " + ", ".join(f"{names[v]} {np.median(mis[sel & (V == v)]):+.3f}" for v in range(4))
             + "; over threshold " + ", ".join(f"{names[v][:1]} {(mis[sel & (V == v)] > th).mean():.0%}" for v in range(4)))
# the quietest neuron of layer LB over these words (its write size at the 99.9th percentile)
with torch.no_grad():
    Xt = torch.tensor(X, dtype=torch.float32, device=DEV)
    act = F.silu(Xt @ ff.w1.weight.float().T) * (Xt @ ff.w3.weight.float().T)
    ws = act.abs() * ff.w2.weight.float().norm(dim=0)[None]          # torch.quantile caps at 16M elements; kthvalue does not
    loud = ws.kthvalue(max(1, int(0.999 * ws.shape[0])), dim=0).values
    BN = int(loud.argmin())
    del Xt, act
L.append(f"  brake neuron: layer {LB} #{BN} (old 99.9% write {float(loud[BN]):.2e}, median {float(loud.median()):.2e}); threshold {th:+.3f}")
gate_line = ridge0(X[build], mis[build] - th, w=w)       # gate = steep * (misfit - th), offset from his own carriers
lt = torch.tensor(line, dtype=ff.w1.weight.dtype, device=DEV)
gt = torch.tensor(STEEP * gate_line, dtype=ff.w1.weight.dtype, device=DEV)
orig = (ff.w1.weight[BN].clone(), ff.w3.weight[BN].clone(), ff.w2.weight[:, BN].clone())
post = lay[LB].post_ffn_norm.weight.float()


def set_brake(scale):
    with torch.no_grad():
        ff.w1.weight[BN] = gt
        ff.w3.weight[BN] = lt
        d = torch.zeros_like(ff.w2.weight[:, BN]); d[WIRE] = -scale / float(post[WIRE])
        ff.w2.weight[:, BN] = d


def wire_at(sel_v, part, doubt=False):
    """c5's wire after its key layer at these words (and, doubt=True, the chance the next word opens doubt)"""
    vals, dts = [], []
    for (ids, pos, vi, k) in seqs:
        if vi != sel_v or (k % 3 != 2) != part:
            continue
        with torch.no_grad():
            lg = m(input_ids=torch.tensor([ids], device=DEV)).logits[0, pos].float()
        vals.append(float(got["wire"][pos]))
        dts.append(float(torch.softmax(lg, -1)[DOUBT_IDS].sum()))
    return (np.array(vals), np.array(dts)) if doubt else np.array(vals)


# how the real gate input behaves at the words:
with torch.no_grad():
    set_brake(1.0)
    gin = X @ ff.w1.weight[BN].float().cpu().numpy()
L.append(f"  gate input median: " + ", ".join(f"{names[v]} {np.median(gin[build & (V == v)]):+.2f}" for v in range(4))
         + "; fires " + ", ".join(f"{names[v][:1]} {(gin[build & (V == v)] > 0).mean():.0%}" for v in range(4)))
# calibrate the write: the wire drop at misfit forks should equal c5's rise there (c5's wire minus the untouched value
# is not available here, so: c5's wire at misfit forks minus its wire at ordinary words)
with torch.no_grad():
    ff.w1.weight[BN], ff.w3.weight[BN], ff.w2.weight[:, BN] = orig
base_c, base_o = wire_at(2, True), wire_at(3, True)
target = float(np.median(base_c) - np.median(base_o))
L.append(f"  c5's wire (build half): misfit forks median {np.median(base_c):+.3f}, ordinary words {np.median(base_o):+.3f}; c5's rise here {target:+.3f}")
if opt.get("drop"):
    # drop=<raw> (c5m2, Immy's one alarm, two polarities): the misfit push mirrors c5's own duck, the same height the
    # other way (c5: +4.1 x the wire's usual size 0.858 = +3.5 raw at its forks), so the crowd is turned UP: "wait"
    target = float(opt["drop"])
    L.append(f"  misfit push set to mirror c5's duck: wire drop {target:.3f} raw at misfit forks")
s_ = 1.0
for it in range(8):
    set_brake(s_)
    drop = float(np.median(base_c - wire_at(2, True)))
    L.append(f"  write step {it}: scale {s_:.4g}, wire drop at misfit forks {drop:+.3f} (asked {target:.3f})")
    print(L[-1], flush=True)
    if target <= 0 or abs(drop / target - 1) < 0.05:
        break
    if drop <= 0 or s_ > 1e4:
        # the wire is not carrying the push (c5m2 at layer 9: layer 10 resets 397, lane_carry.py); stop, never scale on
        sys.exit(f"push does not reach the key layer (drop {drop:+.3f} at scale {s_:.3g}); pick a layer after the reset")
    s_ *= min(target / drop, 10.0)
for part in (True, False):
    tag = "build" if part else "held out"
    after = [wire_at(v, part, doubt=True) for v in range(3)]
    with torch.no_grad():
        cur = (ff.w1.weight[BN].clone(), ff.w3.weight[BN].clone(), ff.w2.weight[:, BN].clone())
        ff.w1.weight[BN], ff.w3.weight[BN], ff.w2.weight[:, BN] = orig
    befo = [wire_at(v, part, doubt=True) for v in range(3)]
    with torch.no_grad():
        ff.w1.weight[BN], ff.w3.weight[BN], ff.w2.weight[:, BN] = cur
    L.append(f"  {tag}: c5's wire median, c5 -> c5 + brake: " + ", ".join(
        f"{names[v]} {np.median(befo[v][0]):+.2f} -> {np.median(after[v][0]):+.2f}" for v in range(3)))
    L.append(f"  {tag}: chance the next word opens doubt, c5 -> c5 + brake: " + ", ".join(
        f"{names[v]} {befo[v][1].mean():.3f} -> {after[v][1].mean():.3f}" for v in range(3)))
os.makedirs(out, exist_ok=True)
m.save_pretrained(out)
tok.save_pretrained(out)
for f in os.listdir(c5):
    if f.endswith((".py", ".jinja", ".json", ".txt")) and not f.endswith(".index.json") and not os.path.exists(os.path.join(out, f)):
        shutil.copy(os.path.join(c5, f), out)
meta = json.load(open(os.path.join(c5, "pam.json")))
meta["misfit_brake"] = {"drop": opt.get("drop"), "layer": LB, "neuron": BN, "threshold": th, "steep": STEEP, "wire": WIRE, "scale": s_}
json.dump(meta, open(os.path.join(out, "pam.json"), "w"), indent=1)
open(os.path.join(out, "pam_misfit_report.txt"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
print("SAVED", out)
