#!/usr/bin/env python3
"""What does the PAM's "I have it" meter read in the middle of arithmetic? (2026-10-31: p3 still pops on digits, e.g.
'9*137' -> '17', after the two loudest receptors were dropped, so the cause is not one receptor.)
The PAM shifts each receptor's gate by -beta_j * (belief(x) - theta). The shift is the w1 row change times the input:
    shift_j(x) = (w1_build - w1_untouched)_j . x        meter_j(x) = shift_j(x) / -beta_j  (= belief - theta)
All receptors in one layer share one belief line, so the meter is one number per layer per word. Per word of the weigh-in
text (weigh_in.py's .pt), in the build:
  meter       belief - theta per receptor layer (0 = undecided, the intended reading off the forks)
  engaged     share of the receptors whose own gate is open (> 0) there: the PAM only acts on engaged receptors
  push        size of what the PAM changed in the receptor layers' MLP writes, after the auto-level (summed over layers)
by the kind of word being chosen next: a digit inside a calculation, a sentence opener, anything else.

usage: pam_meter.py <untouched hf folder> <build folder> <weigh-in .pt> <out prefix> [dev=cuda]
"""
import json, os, re, sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from patients import _lfm3

base, build, wpt, out = sys.argv[1:5]
opt = dict(a.split("=", 1) for a in sys.argv[5:])
DEV = opt.get("dev", "cuda")
W = torch.load(wpt, weights_only=False)
pam = json.load(open(os.path.join(build, "pam.json")))
pa, pb = _lfm3(DEV, base), _lfm3(DEV, build)
tok = pb.tok
RL = sorted({r["layer"] for r in pam["receptors"]})
REC = {L: [r for r in pam["receptors"] if r["layer"] == L] for L in RL}
with torch.no_grad():
    D = {L: (pb.model.model.layers[L].feed_forward.w1.weight[[r["neuron"] for r in REC[L]]].float()
             - pa.model.model.layers[L].feed_forward.w1.weight[[r["neuron"] for r in REC[L]]].float()) for L in RL}
BETA = {L: torch.tensor([r["beta"] for r in REC[L]], device=DEV) for L in RL}
del pa

# the kind of word chosen next (weigh-in word t predicts token t+1)
DIG = re.compile(r"^\s?\d+$")
MATH = re.compile(r"[\d=*/+\-^×·(]\s*$")


def kind(seq, t):
    nxt, ctx = tok.decode(seq[t + 1:t + 2]), tok.decode(seq[max(0, t - 6):t + 1])
    if DIG.match(nxt) and MATH.search(ctx):
        return "digit in a calculation"
    if re.search(r"(\n|[.!?])\s*$", ctx) and re.match(r"^\s?[A-Z]", nxt):
        return "sentence opener"
    return "other"


by_ans = {}
for k, (n, t) in enumerate(W["word"]):
    by_ans.setdefault(n, []).append((k, t))
NW = len(W["word"])
METER = torch.zeros(NW, len(RL)); ENG = torch.zeros(NW, len(RL)); PUSH = torch.zeros(NW)
KIND = [""] * NW
lay = pb.model.model.layers
with torch.no_grad():
    for n, ws in sorted(by_ans.items()):
        seq = W["seqs"][n]
        ks = torch.tensor([k for k, _ in ws]); ts = torch.tensor([t for _, t in ws], device=DEV)
        cap, hs = {}, []
        for L in RL:
            hs.append(lay[L].feed_forward.register_forward_pre_hook(lambda mod, a, L=L: cap.__setitem__(("x", L), a[0][0, ts].float())))
            hs.append(lay[L].post_ffn_norm.register_forward_pre_hook(lambda mod, a, L=L: cap.__setitem__(("y", L), a[0][0, ts].float())))
        pb.model.model(input_ids=torch.tensor([seq], device=DEV))
        for h in hs:
            h.remove()
        push = torch.zeros(len(ws), device=DEV)
        for i, L in enumerate(RL):
            ff, x = lay[L].feed_forward, cap[("x", L)]
            ix = [r["neuron"] for r in REC[L]]
            g = x @ ff.w1.weight[ix].float().T                     # build gate (shift included)
            s = x @ D[L].T                                         # the PAM's shift
            u = x @ ff.w3.weight[ix].float().T
            METER[ks, i] = (s / -BETA[L][None]).median(1).values.cpu()
            ENG[ks, i] = ((g - s) > 0).float().mean(1).cpu()
            da = (F.silu(g) - F.silu(g - s)) * u                   # what the PAM changed in each receptor's firing
            wr = da @ (ff.w2.weight[:, ix].float() * lay[L].post_ffn_norm.weight.float()[:, None]).T
            push += wr.norm(dim=-1) / cap[("y", L)].pow(2).mean(-1).add(1e-5).sqrt() / x.shape[-1] ** .5
        PUSH[ks] = push.cpu()
        for k, t in ws:
            KIND[k] = kind(seq, t)
        print(f"  {n + 1}/{len(by_ans)} answers", flush=True)

KL = W["kl"]
q = lambda x, p: float(torch.quantile(x.float(), p)) if len(x) else float("nan")
lines = [f"PAM meter for {build}: receptor layers {RL}, {NW} words of the weigh-in text",
         "meter = belief - theta at the receptors' input (0 = undecided); engaged = receptors with their own gate open;",
         "push = size of the PAM's change to the receptor layers' writes (after the auto-level)", ""]
for kd in ("digit in a calculation", "sentence opener", "other"):
    m = torch.tensor([k == kd for k in KIND])
    mt = METER[m]
    lines.append(f"{kd}: {int(m.sum())} words, drift KL mean {float(KL[m].mean()):.4f}, 99% {q(KL[m], .99):.3f}")
    for i, L in enumerate(RL):
        lines.append(f"  layer {L}: meter median {q(mt[:, i], .5):+.2f} (1% {q(mt[:, i], .01):+.2f}, 99% {q(mt[:, i], .99):+.2f}), "
                     f"|meter| > 0.3 at {float((mt[:, i].abs() > .3).float().mean()):.0%}; engaged {float(ENG[m, i].mean()):.0%}")
    lines.append(f"  push median {q(PUSH[m], .5):.4f}, 99% {q(PUSH[m], .99):.4f}")
lines.append("")
lines.append(f"correlation over all words, log push vs log KL: "
             f"{float(torch.corrcoef(torch.stack([PUSH.clamp_min(1e-6).log(), KL.clamp_min(1e-6).log()]))[0, 1]):.3f}")
lines += ["", "top 15 pops: KL, kind, meter per layer, engaged per layer, push"]
for w in torch.argsort(-KL)[:15].tolist():
    n, t = W["word"][w]
    ctx = tok.decode(W["seqs"][n][max(0, t - 8):t + 1]).replace("\n", " / ")
    lines.append(f"  KL {float(KL[w]):5.2f}  {KIND[w]:22}  meter " + " ".join(f"{float(v):+.2f}" for v in METER[w]) +
                 "  engaged " + " ".join(f"{float(v):.0%}" for v in ENG[w]) + f"  push {float(PUSH[w]):.4f}"
                 f"   ...{ctx!r} -> {tok.decode(W['seqs'][n][t + 1:t + 2])!r}")
torch.save({"meter": METER, "engaged": ENG, "push": PUSH, "kind": KIND, "layers": RL}, out + ".pt")
open(out + ".txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
print("METER_DONE")
