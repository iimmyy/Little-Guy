#!/usr/bin/env python3
"""A model's vitals for dosing the PAM (Immy, 2026-10-28: "like a doctor looking at symptoms").

Dose a is a number on our knob. What a model feels is how much its gain actually swings at the receptor (the norm),
and that depends on the model: how much of the loudness the giant carries, how much the giant wobbles, and whether the
wobble means anything. Measure those per layer, convert every dose into real gain swing, and the best dose on E4B
becomes a portable number ("about X% swing") to dose any other model by, like mg per kg.

Per layer, at the two norms that read the bloodstream (same giant and resting level as pam.py):
  share    how much of the loudness (mean square) the giant carries                       -> leverage, receptor density
  wobble   the giant's wobble around rest, relative to its resting level                  -> the natural signal's size
  sens     share x relative wobble: the gain swing one unit of dose gives (first order)   -> dose to effect conversion
  meaning  correlation of the giant's wobble with his odds of closing the thinking (math) -> does the signal carry news
Per dose: the real gain swing at every norm with the PAM on (spread of the multiplier over words, in %).

usage: vitals.py <out folder> <math traces jsonl> [doses=0.5,0.75,1,1.25,2 dev=mps nmath=20]
writes vitals.txt and vitals.json
"""
import json, os, sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from e4b_mac import load

out, traces = sys.argv[1:3]
opt = dict(a.split("=", 1) for a in sys.argv[3:])
DOSES = [float(d) for d in opt.get("doses", "0.5,0.75,1,1.25,2").split(",")]
DEV, NMATH = opt.get("dev", "mps"), int(opt.get("nmath", 20))
BENCH = os.path.join(os.path.dirname(__file__), "../../../.run/bench")
os.makedirs(out, exist_ok=True)
model, tok = load(DEV)
layers = model.model.layers
L, D = len(layers), model.config.hidden_size
END = tok.convert_tokens_to_ids("<channel|>")
wiki = tok(open(f"{BENCH}/wiki.test.raw").read()[:200000], add_special_tokens=False).input_ids
chat = tok(open(f"{BENCH}/chat_heldout.txt").read(), add_special_tokens=False).input_ids
rows = [json.loads(l) for l in open(traces)]


def mathwin(r, span=False):
    p = tok.apply_chat_template([{"role": "user", "content": r["q"]}], tokenize=False, add_generation_prompt=True,
                                enable_thinking=True)
    ids = tok(p + "<|channel>thought\n" + r["thinking"] + "<channel|>" + r["reply"], add_special_tokens=False).input_ids[:768]
    if not span:
        return ids
    s = len(tok(p + "<|channel>thought\n", add_special_tokens=False).input_ids)
    return ids, s, ids.index(END) if END in ids else len(ids)


# the same windows pam.py found the giants on, so giants and resting levels match the runs exactly
fitwin = [[tok.bos_token_id] + wiki[i * 511:(i + 1) * 511] for i in range(0, 16, 2)] + \
         [[tok.bos_token_id] + chat[i * 511:(i + 1) * 511] for i in range(0, 32, 2)] + [mathwin(r) for r in rows[::14]]
s1 = [torch.zeros(D, dtype=torch.float64) for _ in range(L)]
s2 = [torch.zeros(D, dtype=torch.float64) for _ in range(L)]
cnt = [0]


def stats_hook(i):
    def h(mod, args):
        x = args[0][0, 1:].float()
        s1[i] += x.sum(0).cpu().double(); s2[i] += x.pow(2).sum(0).cpu().double()
        if i == 0:
            cnt[0] += x.shape[0]
    return h


hs = [layers[i].input_layernorm.register_forward_pre_hook(stats_hook(i)) for i in range(L)]
with torch.no_grad():
    for w in fitwin:
        model.model(input_ids=torch.tensor([w], device=DEV))
for h in hs:
    h.remove()
GI = [int((s2[i] / cnt[0]).sqrt().argmax()) for i in range(L)]
REST = [float(s1[i][GI[i]] / cnt[0]) for i in range(L)]

# measuring windows: held-out text plus his own math thinking (rows the giant search never saw)
mrows = [r for k, r in enumerate(rows) if k % 14][:NMATH]
mwins = [mathwin(r, True) for r in mrows]
wins = [([tok.bos_token_id] + wiki[i * 511:(i + 1) * 511], "wiki", None) for i in range(16, 19)] + \
       [([tok.bos_token_id] + chat[i * 511:(i + 1) * 511], "chat", None) for i in range(32, 37)] + \
       [(w, "math", (s, e)) for w, s, e in mwins]

NORMS = [(i, "attn", l.input_layernorm) for i, l in enumerate(layers)] + \
        [(i, "mlp", l.pre_feedforward_layernorm) for i, l in enumerate(layers)]
st = {"a": 0.0, "rec": None}


def norm_hook(i, kind):
    def h(mod, args, o):
        x = args[0][0].float()
        g = x[:, GI[i]]
        ms = x.pow(2).mean(-1) + mod.eps
        a = st["a"]
        g2 = REST[i] + (1 + a) * (g - REST[i])
        ms2 = (ms + (g2.pow(2) - g.pow(2)) / D).clamp_min(mod.eps)
        logm = 0.5 * (ms / ms2).log()
        st["rec"].setdefault((i, kind), []).append({"share": (g.pow(2) / D / ms).cpu(), "dev": ((g - REST[i]) / abs(REST[i])).cpu(),
                                                     "logm": logm.cpu()})
        return None if a == 0 else (o.float() * logm.exp()[None, :, None]).to(o.dtype)
    return h


for i, kind, n in NORMS:
    n.register_forward_hook(norm_hook(i, kind))


def endlogp(ids):
    h = model.model(input_ids=torch.tensor([ids], device=DEV)).last_hidden_state[0]
    cap, res = model.config.final_logit_softcapping, []
    for c0 in range(0, h.shape[0], 128):
        lg = model.lm_head(h[c0:c0 + 128]).float()
        if cap:
            lg = cap * torch.tanh(lg / cap)
        res.append((lg[:, END] - lg.logsumexp(-1)).cpu())
    return torch.cat(res)


runs = {}
with torch.no_grad():
    for a in [0.0] + DOSES:
        st["a"], st["rec"] = a, {}
        ends = []
        for w, t, sp in wins:
            if a == 0 and t == "math":
                ends.append(endlogp(w))                                      # his odds of closing the thinking, per word
            else:
                model.model(input_ids=torch.tensor([w], device=DEV))
            if DEV == "mps":
                torch.mps.empty_cache()
        runs[a] = (st["rec"], ends)
        print(f"  dose {a:+.2f} measured", flush=True)

rec0, ends = runs[0.0]
mi = [k for k, (_, t, _) in enumerate(wins) if t == "math"]
vit = {}
for i, kind, _ in NORMS:
    R = rec0[(i, kind)]
    share = torch.cat([r["share"][1:] for r in R]).mean().item()
    dev = torch.cat([r["dev"][1:] for r in R])
    sens = torch.cat([(r["share"] * r["dev"])[1:] for r in R]).std().item()   # first-order swing per unit dose
    xs, ys = [], []
    for j, k in enumerate(mi):                                                # thinking words only; END odds at the next word
        s, e = wins[k][2]
        xs.append(R[k]["dev"][s:e]); ys.append(ends[j][s:e])
    x, y = torch.cat(xs).numpy(), torch.cat(ys).numpy()
    meaning = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 else 0.0
    swing = {}
    for a in DOSES:
        Ra = runs[a][0][(i, kind)]
        lm = torch.cat([r["logm"][1:] for r in Ra])
        swing[a] = (lm.std().item() * 100, lm.mean().item() * 100)
    vit[(i, kind)] = dict(share=share, wobble=dev.std().item(), sens=sens * 100, meaning=meaning, swing=swing)

lines = [f"E4B vitals for the PAM ({len(wins)} windows: 3 wiki, 5 chat, {len(mwins)} math thinking traces)", "",
         "per layer, attention norm | MLP norm. share = giant's part of the loudness; wobble = giant's spread around rest",
         "(fraction of rest); sens = gain swing per unit dose, %; meaning = corr(giant wobble, odds of closing the thinking)", "",
         f"{'layer':>5} {'giant':>6} {'rest':>6} | {'share':>6} {'wobble':>6} {'sens%':>6} {'mean':>6} | {'share':>6} {'wobble':>6} {'sens%':>6} {'mean':>6}"]
for i in range(L):
    A, M = vit[(i, "attn")], vit[(i, "mlp")]
    lines.append(f"{i:>5} {'#' + str(GI[i]):>6} {REST[i]:>+6.0f} | {A['share']:>6.2f} {A['wobble']:>6.3f} {A['sens']:>6.2f} {A['meaning']:>+6.2f} | "
                 f"{M['share']:>6.2f} {M['wobble']:>6.3f} {M['sens']:>6.2f} {M['meaning']:>+6.2f}")
lines += ["", "real gain swing at the receptor per dose (spread of the multiplier over words, %, all norms)",
          f"{'dose':>6} {'mean':>6} {'median':>7} {'max':>6} {'max layer':>10} {'avg shift':>10}"]
for a in DOSES:
    sw = [(v["swing"][a][0], k) for k, v in vit.items()]
    vals = np.array([s for s, _ in sw])
    top = max(sw)
    lines.append(f"{a:>+6.2f} {vals.mean():>6.2f} {np.median(vals):>7.2f} {top[0]:>6.2f} {str(top[1][0]) + ' ' + top[1][1]:>10} "
                 f"{np.mean([v['swing'][a][1] for v in vit.values()]):>+10.2f}")
open(os.path.join(out, "vitals.txt"), "w").write("\n".join(lines) + "\n")
json.dump({f"{i}-{k}": {**v, "swing": {str(a): s for a, s in v["swing"].items()}, "giant": GI[i], "rest": REST[i]}
           for (i, k), v in vit.items()}, open(os.path.join(out, "vitals.json"), "w"), indent=1)
print("\n".join(lines))
print("VITALS_DONE")
