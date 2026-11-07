#!/usr/bin/env python3
"""Root cause of a compressor build's drift (2026-11-01: c1 drifted KL 0.34 mean in ordinary thinking, 99% 9.5).
For every scored word of some practice answers: drift (KL untouched -> build), the key neuron's firing, and how hard the
build pushed the ducked doubt crowd's gates there (sum over ducked neurons of |gate change|). Sorts drift into
  key firing   the compressor acting (intended at answer moments)
  key silent   a side effect: then is it the crowd being pushed anyway (sidechain hiss) or something else?
and prints the biggest drift words with their text.

usage: compressor_rca.py <untouched hf folder> <build folder> <answers jsonl> <out prefix> [n=20 dev=cuda]
"""
import json, os, sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from patients import _lfm3

base, build, ans, out = sys.argv[1:5]
opt = dict(a.split("=", 1) for a in sys.argv[5:])
N, DEV = int(opt.get("n", 20)), opt.get("dev", "cuda")
pj = json.load(open(os.path.join(build, "pam.json")))
C = pj["compressor"]
GK, GN = C["key_layer"], C["key_neuron"]
pa, pb = _lfm3(DEV, base), _lfm3(DEV, build)
tok = pb.tok
la, lb = pa.model.model.layers, pb.model.model.layers
CL = sorted({r["layer"] for r in pj["receptors"]})
IX = {L: [r["neuron"] for r in pj["receptors"] if r["layer"] == L] for L in CL}
with torch.no_grad():
    DG = {L: (lb[L].feed_forward.w1.weight[IX[L]].float() - la[L].feed_forward.w1.weight[IX[L]].float()) for L in CL}
TAIL = "\n\nEnd with a line: Answer: <your answer>"
rows = [r for r in map(json.loads, open(ans)) if r["thinking"]][:N]
KL, KEY, PUSH, WHERE = [], [], [], []
with torch.no_grad():
    for n, r in enumerate(rows):
        pr = tok(tok.apply_chat_template([{"role": "user", "content": r["q"] + TAIL}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=True), add_special_tokens=False).input_ids
        th = tok("\n" + r["thinking"], add_special_tokens=False).input_ids[:3000]
        ids = torch.tensor([pr + th], device=DEV)
        a = torch.log_softmax(pa.model(input_ids=ids).logits[0, len(pr):].float(), -1)
        cap = {}
        hs = [lb[GK].feed_forward.register_forward_pre_hook(lambda mod, x: cap.__setitem__("k", x[0][0, len(pr):].float()))]
        for L in CL:
            hs.append(lb[L].feed_forward.register_forward_pre_hook(lambda mod, x, L=L: cap.__setitem__(L, x[0][0, len(pr):].float())))
        b = torch.log_softmax(pb.model(input_ids=ids).logits[0, len(pr):].float(), -1)
        for h in hs:
            h.remove()
        ff = lb[GK].feed_forward
        key = F.silu(cap["k"] @ ff.w1.weight[GN].float()) * (cap["k"] @ ff.w3.weight[GN].float())
        push = sum((cap[L] @ DG[L].T).abs().sum(-1) for L in CL)
        KL.append((a.exp() * (a - b)).sum(-1).cpu()); KEY.append(key.cpu()); PUSH.append(push.cpu())
        WHERE += [(n, len(pr), t) for t in range(len(th))]
        print(f"  {n + 1}/{len(rows)}", flush=True)
KL, KEY, PUSH = torch.cat(KL), torch.cat(KEY), torch.cat(PUSH)
hot = float(torch.quantile(KEY, .999))
fire = KEY > 0.1 * hot
q = lambda x, p: float(torch.quantile(x, p)) if len(x) else float("nan")
L_ = [f"compressor RCA for {build}: {len(KL)} thinking words of {len(rows)} answers; key 99.9% firing {hot:.3f}",
      f"key firing (> 10% of that): {int(fire.sum())} words, KL mean {float(KL[fire].mean()):.4f}, 99% {q(KL[fire], .99):.3f}; "
      f"share of all drift {float(KL[fire].sum() / KL.sum()):.0%}",
      f"key silent: {int((~fire).sum())} words, KL mean {float(KL[~fire].mean()):.4f}, 99% {q(KL[~fire], .99):.3f}",
      f"  crowd push where silent: median {q(PUSH[~fire], .5):.3f}, 99% {q(PUSH[~fire], .99):.3f}; where firing: median {q(PUSH[fire], .5):.3f}",
      f"  correlation (silent words) log KL vs crowd push: "
      f"{float(torch.corrcoef(torch.stack([KL[~fire].clamp_min(1e-6).log(), PUSH[~fire]]))[0, 1]):.3f}"]
for lo_, hi_ in ((0, 0.5), (0.5, 1), (1, 2), (2, 4), (4, 1e9)):
    s = ~fire & (PUSH >= lo_) & (PUSH < hi_)
    if s.any():
        L_.append(f"  silent words with crowd push {lo_}-{hi_}: {int(s.sum())} words, KL mean {float(KL[s].mean()):.4f}, "
                  f"share of silent drift {float(KL[s].sum() / KL[~fire].sum()):.0%}")
L_.append("top 20 drift words: KL, key, crowd push, text")
for w in torch.argsort(-KL)[:20].tolist():
    n, p0, t = WHERE[w]
    th = tok("\n" + rows[n]["thinking"], add_special_tokens=False).input_ids
    L_.append(f"  KL {float(KL[w]):5.2f} key {float(KEY[w]):+.3f} push {float(PUSH[w]):5.2f}  ...{tok.decode(th[max(0, t - 12):t + 1])!r}"
              f" -> {tok.decode(th[t + 1:t + 2])!r}")
open(out + ".txt", "w").write("\n".join(L_) + "\n")
torch.save({"kl": KL, "key": KEY, "push": PUSH, "where": WHERE}, out + ".pt")
print("\n".join(L_))
print("RCA_DONE")
