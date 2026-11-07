#!/usr/bin/env python3
"""Where in E4B does the common sense that LFM3 lacks live? The diagnosis before a transplant (Immy, 2026-10-28:
"take the things from E4B that are clearly working for it and that LFM is missing").

Questions: the HellaSwag items E4B gets right and LFM3 gets wrong (the gap), and as many items both get right, picked
so E4B's confidence on them matches the gap items one for one (the control). Matching matters: any damage hurts a
shaky answer more than a sure one, so without it every room would look like it holds the gap.

Three looks, all on the same items and scoring as hs_patient.py (mean log-chance per ending word):
  lens    read E4B's pick after every room (final norm + output head on the stream there): where does it settle?
  knock   skip a window of W rooms (each one's output = its input): which rooms does the gap need, beyond the control?
  ple     the same windows with only their per-layer word notes zeroed: how much does that tissue lean on Gemma's own
          vocabulary? (A graft in LFM3 can't get those notes for free.)
A skipped room still runs underneath, so rooms 22-23 keep writing the keys/values that rooms 24-41 reuse (KV sharing):
knocking them out removes their effect on the stream, not on the later rooms' memory.

usage: rooms.py <e4b right.json> <lfm3 right.json> <out folder> [w=4 stride=2 dev=cuda max=0]
"""
import json, os, sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from patients import load

e4b_js, lfm_js, out = sys.argv[1:4]
opt = dict(a.split("=", 1) for a in sys.argv[4:])
W, STRIDE, DEV, MAX = int(opt.get("w", 4)), int(opt.get("stride", 2)), opt.get("dev", "cuda"), int(opt.get("max", 0))
os.makedirs(out, exist_ok=True)
BENCH = os.path.join(os.path.dirname(__file__), "../../../.run/bench")
ra, rb = json.load(open(e4b_js))["right"], json.load(open(lfm_js))["right"]
n = min(len(ra), len(rb))
gap = [i for i in range(n) if ra[i] and not rb[i]]
both = [i for i in range(n) if ra[i] and rb[i]]
if MAX:
    gap = gap[:MAX]

p = load("e4b", DEV)
tok, lm = p.tok, p.model.model
layers = lm.layers
L = len(layers)
hl = open(f"{BENCH}/hellaswag_val_full.txt").read().split("\n")
Q = {k: (hl[6 * k], int(hl[6 * k + 1]), hl[6 * k + 2:6 * k + 6]) for k in range(n)}


def batch(k):
    """the four endings of question k as one right-padded batch (causal: padding after an ending can't reach it)"""
    ctx, lab, ends = Q[k]
    c = [tok.bos_token_id] + tok(ctx, add_special_tokens=False).input_ids
    es = [tok(" " + e, add_special_tokens=False).input_ids for e in ends]
    T = len(c) + max(map(len, es))
    ids = torch.full((4, T), tok.pad_token_id, dtype=torch.long)
    for j, e in enumerate(es):
        ids[j, :len(c) + len(e)] = torch.tensor(c + e)
    return ids.to(DEV), len(c), es, lab


def scores(h, c, es):
    """mean log-chance per ending word, from the stream h (4, T, D) with the final norm already applied"""
    out = []
    for j, e in enumerate(es):
        lp = p.logits(h[j, c - 1:c - 1 + len(e)]).log_softmax(-1)
        out.append(lp.gather(-1, torch.tensor(e, device=DEV)[:, None]).mean().item())
    return np.array(out)


def margin(s, lab):
    return float(s[lab] - np.delete(s, lab).max())


hooks = []


def skip(rooms, what):
    """what = "knock": the room passes its input through; "ple": the room runs without its per-layer word notes"""
    for i in rooms:
        if what == "knock":
            hooks.append(layers[i].register_forward_hook(lambda m, a, o: a[0]))
        else:
            hooks.append(layers[i].register_forward_pre_hook(lambda m, a: (a[0], torch.zeros_like(a[1])) + a[2:]))


def clear():
    while hooks:
        hooks.pop().remove()


with torch.no_grad():
    # baseline on the gap and on every both-right item; the control is chosen from these by E4B's margin
    base = {}
    for k in gap + both:
        ids, c, es, lab = batch(k)
        base[k] = margin(scores(p.body(ids), c, es), lab)
    ctrl = []
    for k in sorted(gap, key=lambda k: base[k]):                          # nearest unused both-right margin
        ctrl.append(min((x for x in both if x not in ctrl), key=lambda x: abs(base[x] - base[k])))
    print(f"gap {len(gap)} items, control {len(ctrl)}; E4B margin gap {np.mean([base[k] for k in gap]):.3f} "
          f"control {np.mean([base[k] for k in ctrl]):.3f}", flush=True)
    res = {"w": W, "stride": STRIDE, "gap": gap, "control": ctrl, "base": base, "lens": {}, "knock": [], "ple": []}

    # lens: hidden_states = embeddings, then each room's output before the final norm
    for k in gap + ctrl:
        ids, c, es, lab = batch(k)
        hs = lm(input_ids=ids, output_hidden_states=True).hidden_states
        res["lens"][k] = [margin(scores(lm.norm(hs[i + 1]), c, es), lab) for i in range(L - 1)] + [base[k]]

    for what in ("knock", "ple"):
        for r0 in range(0, L - W + 1, STRIDE):
            rooms = list(range(r0, r0 + W))
            skip(rooms, what)
            m = {}
            for k in gap + ctrl:
                ids, c, es, lab = batch(k)
                m[k] = margin(scores(p.body(ids), c, es), lab)
            clear()
            g = np.mean([m[k] > 0 for k in gap]); cc = np.mean([m[k] > 0 for k in ctrl])
            res[what].append({"rooms": rooms, "gap_right": float(g), "ctrl_right": float(cc),
                              "gap_margin": float(np.mean([m[k] - base[k] for k in gap])),
                              "ctrl_margin": float(np.mean([m[k] - base[k] for k in ctrl]))})
            print(f"  {what} {r0:2}-{r0 + W - 1:2}: gap {100 * g:5.1f}% right, control {100 * cc:5.1f}%", flush=True)
            json.dump(res, open(f"{out}/rooms.json", "w"))

lines = [f"E4B rooms on HellaSwag: {len(gap)} gap items (E4B right, LFM3 wrong), {len(ctrl)} margin-matched controls", "",
         "lens: share already picking the right ending after each room", "  room  gap  control"]
for i in range(L):
    lines.append(f"  {i:4} {100 * np.mean([res['lens'][k][i] > 0 for k in gap]):4.0f} "
                 f"{100 * np.mean([res['lens'][k][i] > 0 for k in ctrl]):6.0f}")
for what in ("knock", "ple"):
    lines += ["", f"{what} (window {W}): right after, gap vs control (both 100% before); margin change",
              "  rooms   gap  control   gap-ctrl   margin gap / control"]
    for r in res[what]:
        lines.append(f"  {r['rooms'][0]:2}-{r['rooms'][-1]:2} {100 * r['gap_right']:5.1f} {100 * r['ctrl_right']:7.1f} "
                     f"{100 * (r['gap_right'] - r['ctrl_right']):9.1f}   {r['gap_margin']:+.3f} / {r['ctrl_margin']:+.3f}")
open(f"{out}/rooms.txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
print("ROOMS_DONE")
