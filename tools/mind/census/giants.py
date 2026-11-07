#!/usr/bin/env python3
"""First exam of the dosing pipeline: does this model carry a giant, and where?

E4B carries one huge number in its bloodstream (#611 early, #1434 middle, #2302 late) that holds most of the loudness
each norm reads; the PAM works through it. Before dosing any other model, check it has one. Per receptor site (each norm
that reads the stream), over held-out wiki and chat text (the first word skipped: it is the attention sink):
  giant    the loudest of the hidden numbers there, by root mean square
  share    how much of the loudness (mean square) that one number carries, on average over words
  x med    how many times louder it is than the median number
  2nd      the runner-up's share, so a giant stands out from a crowd of big numbers

usage: giants.py <patient> <out folder> [dev=mps nwin=8]      patients: see patients.py
writes giants.txt and giants.json
"""
import json, os, sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from patients import load

name, out = sys.argv[1:3]
opt = dict(a.split("=", 1) for a in sys.argv[3:])
DEV, NWIN = opt.get("dev", "mps"), int(opt.get("nwin", 8))
BENCH = os.path.join(os.path.dirname(__file__), "../../../.run/bench")
os.makedirs(out, exist_ok=True)
p = load(name, DEV)
tok = p.tok
bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
wiki = tok(open(f"{BENCH}/wiki.test.raw").read()[:200000], add_special_tokens=False).input_ids
chat = tok(open(f"{BENCH}/chat_heldout.txt").read(), add_special_tokens=False).input_ids
wins = [bos + wiki[i * 511:(i + 1) * 511] for i in range(0, 2 * NWIN, 2)] + \
       [bos + chat[i * 511:(i + 1) * 511] for i in range(0, 2 * NWIN, 2)]

D = p.model.config.hidden_size
acc = {(i, k): {"s2": torch.zeros(D, dtype=torch.float64), "share": torch.zeros(D, dtype=torch.float64), "n": 0}
       for i, k, _ in p.norms}


def hook(i, k):
    def h(mod, args):
        x = args[0][0, 1:].float()
        ms = x.pow(2).mean(-1, keepdim=True)
        a = acc[(i, k)]
        a["s2"] += x.pow(2).sum(0).cpu().double()
        a["share"] += (x.pow(2) / D / ms).sum(0).cpu().double()
        a["n"] += x.shape[0]
    return h


hs = [n.register_forward_pre_hook(hook(i, k)) for i, k, n in p.norms]
with torch.no_grad():
    for w in wins:
        p.body(torch.tensor([w], device=DEV))
for h in hs:
    h.remove()

rows = {}
for i, k, _ in p.norms:
    a = acc[(i, k)]
    rms = (a["s2"] / a["n"]).sqrt()
    share = a["share"] / a["n"]
    top = torch.topk(rms, 2).indices.tolist()
    rows[(i, k)] = dict(giant=top[0], share=float(share[top[0]]), xmed=float(rms[top[0]] / rms.median()),
                        second=top[1], share2=float(share[top[1]]))

L = len(p.layer_types)
lines = [f"{name}: giant census over {len(wins)} windows ({NWIN} wiki, {NWIN} chat), hidden size {D}", "",
         "share = the loudest number's part of the loudness that norm reads; x med = times louder than the median number",
         "", f"{'layer':>5} {'type':>18} | {'operator norm':^34} | {'feed-forward norm':^34}",
         f"{'':>5} {'':>18} | {'giant':>6} {'share':>6} {'x med':>6} {'2nd':>6} {'share':>6} | "
         f"{'giant':>6} {'share':>6} {'x med':>6} {'2nd':>6} {'share':>6}"]
for i in range(L):
    op = next(r for (j, k), r in rows.items() if j == i and k != "mlp")
    ff = rows[(i, "mlp")]
    cell = lambda r: f"{'#' + str(r['giant']):>6} {r['share']:>6.2f} {r['xmed']:>6.0f} {'#' + str(r['second']):>6} {r['share2']:>6.2f}"
    lines.append(f"{i:>5} {p.layer_types[i]:>18} | {cell(op)} | {cell(ff)}")
open(os.path.join(out, "giants.txt"), "w").write("\n".join(lines) + "\n")
json.dump({f"{i}-{k}": r for (i, k), r in rows.items()}, open(os.path.join(out, "giants.json"), "w"), indent=1)
print("\n".join(lines))
print("GIANTS_DONE")
