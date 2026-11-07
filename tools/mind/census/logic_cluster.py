#!/usr/bin/env python3
"""Which neurons are busy when his logic goes well? (Immy 2026-10-29: "there's a certain neuron cluster in him that
causes his logic to become better"; "we don't punish, we change a set of neurons".)

Same question, several sampled answers from the untouched model: some solve it, some get stuck or go wrong. So the
question's difficulty is the same on both sides and only the way he thinks differs. The untouched model reads each
answer back; per MLP neuron, the mean activation over the EARLY thinking (words 0..W, before the outcome shows) is
compared between the good and the bad runs of the same question:
    effect(neuron) = mean over questions of [mean good runs - mean bad runs] / pooled spread
Questions are split in two halves: neurons are ranked on half A and checked on half B (sign agreement, and how well
a cluster score built on A tells good from bad runs of the same question in B).

usage: logic_cluster.py <answers jsonl[,more]> <out prefix> [hf=~/.littleguy/hf/lfm3-1.7b early=400 top=200 dev=mps]
"""
import json, os, random, re, sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../expert"))
ARGS, sys.argv[1:] = sys.argv[1:], []                     # grade.py grades its own arguments on import
from grade import right, right_reply
from patients import _lfm3

srcs, out = ARGS[0].split(","), ARGS[1]
opt = dict(a.split("=", 1) for a in ARGS[2:])
HF = os.path.expanduser(opt.get("hf", "~/.littleguy/hf/lfm3-1.7b"))
W, TOP, DEV = int(opt.get("early", 400)), int(opt.get("top", 200)), opt.get("dev", "mps")
TAIL = "\n\nEnd with a line: Answer: <your answer>"
CLAIM = re.compile(r"\\boxed\{([^{}]{1,40})\}|(?:answer is|answer =|answer:|answer)\s*:?\s*\$?([^\n$]{1,40}?)\$?(?=[.\n]|$)", re.I)

runs = {}
for s in srcs:
    for l in open(s):
        r = json.loads(l)
        good = r["finish"] == "stop" and bool(r["reply"]) and right_reply(r, r["reply"])
        if opt.get("label") == "reached":                  # good = his logic got there: the right answer is stated
            good = good or any(right({**r, "reply": x}, x) for m in CLAIM.finditer(r["thinking"] or "")
                               for x in [next(g for g in m.groups() if g)])   # somewhere, even if he never stopped
        if len((r["thinking"] or "").split()) > 50:
            runs.setdefault(r["id"], []).append((good, r))
qs = sorted(i for i, rs in runs.items() if any(g for g, _ in rs) and not all(g for g, _ in rs))[:int(opt.get("maxq", 10**6))]
print(f"{len(runs)} questions; {len(qs)} with both good and bad runs "
      f"({sum(len(runs[i]) for i in qs)} runs)", flush=True)

p = _lfm3(DEV, HF)
tok, lay = p.tok, p.model.model.layers
L = len(lay)
cap = {}
for i, l in enumerate(lay):
    l.feed_forward.w2.register_forward_pre_hook(lambda mod, a, i=i: cap.__setitem__(i, a[0][0]))

M = {}                                                     # (question, run index) -> layers x neurons mean activation
if os.path.exists(out + "_raw.pt"):                        # already read once: redo only the statistics
    M = torch.load(out + "_raw.pt", weights_only=False)["M"]
with torch.no_grad():
    for n, q in enumerate(qs):
        for k, (g, r) in enumerate(runs[q]):
            if (q, k) in M:
                continue
            pr = tok(tok.apply_chat_template([{"role": "user", "content": r["q"] + TAIL}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=True), add_special_tokens=False).input_ids
            th = tok("\n" + r["thinking"], add_special_tokens=False).input_ids[:W]
            cap.clear()
            p.model.model(input_ids=torch.tensor([pr + th], device=DEV))
            M[(q, k)] = torch.stack([cap[i][len(pr):].float().mean(0) for i in range(L)]).cpu()
        if any((q, k) not in M for k in range(len(runs[q]))) or n % 10 == 9:
            print(f"  {n + 1}/{len(qs)} questions read", flush=True)
torch.save({"M": M, "good": {(q, k): g for q in qs for k, (g, _) in enumerate(runs[q])}, "qs": qs}, out + "_raw.pt")


def effect(qset):
    # each run minus its question's mean (the question itself drops out), then one spread over all those runs:
    # a per-question spread from 2-5 runs is too noisy to divide by
    diffs, cen = [], []
    for q in qset:
        X = torch.stack([M[(q, k)] for k in range(len(runs[q]))])
        g = torch.tensor([g for g, _ in runs[q]])
        cen.append(X - X.mean(0))
        diffs.append(X[g].mean(0) - X[~g].mean(0))
    return torch.stack(diffs).mean(0) / (torch.cat(cen).std(0) + 1e-4), torch.stack(diffs)


random.seed(0)
order = qs[:]; random.shuffle(order)
A, B = order[: len(order) // 2], order[len(order) // 2:]
eA, _ = effect(A); eB, dB = effect(B); eAll, _ = effect(qs)
flat = eA.flatten()
top = torch.argsort(-flat.abs())[:TOP]
agree = float((torch.sign(flat[top]) == torch.sign(eB.flatten()[top])).float().mean())
rand = torch.randint(0, flat.numel(), (TOP,))
agree_r = float((torch.sign(flat[rand]) == torch.sign(eB.flatten()[rand])).float().mean())
# cluster score: the top-A neurons, signed by their A effect, standardized per neuron; within each B question,
# does it rank the good runs above the bad ones?
mu = torch.stack(list(M.values())).flatten(1)[:, top].mean(0); sd = torch.stack(list(M.values())).flatten(1)[:, top].std(0) + 1e-4
w = torch.sign(flat[top])
wins = ties = tot = 0
for q in B:
    sc = [(g, float((((M[(q, k)].flatten()[top] - mu) / sd) * w).mean())) for k, (g, _) in enumerate(runs[q])]
    for g1, s1 in sc:
        for g2, s2 in sc:
            if g1 and not g2:
                tot += 1; wins += s1 > s2; ties += s1 == s2
lines = [f"logic cluster hunt: {len(qs)} questions with good and bad runs (half A {len(A)}, half B {len(B)}), early {W} words",
         f"top {TOP} neurons on A: same sign on B {agree:.2f} (random neurons {agree_r:.2f})",
         f"cluster score (A's neurons) ranks a good run above a bad run of the same B question: {(wins + 0.5 * ties) / max(tot, 1):.2f} ({tot} pairs; 0.50 = chance)",
         "", "top neurons over all questions (effect = good minus bad, in spreads; + means busier when his logic goes well):"]
fa = eAll.flatten()
for j in torch.argsort(-fa.abs())[:40].tolist():
    li, ni = divmod(j, eAll.shape[1])
    lines.append(f"  layer {li:2d} neuron {ni:5d}  effect {float(fa[j]):+.2f}   (A {float(eA.flatten()[j]):+.2f}, B {float(eB.flatten()[j]):+.2f})")
per_layer = eAll.abs().topk(20, dim=1).values.mean(1)
lines += ["", "where the differences live (mean |effect| of each layer's 20 strongest neurons):",
          "  " + "  ".join(f"L{i}:{float(v):.2f}" for i, v in enumerate(per_layer))]
print("\n".join(lines))
open(out + ".txt", "w").write("\n".join(lines) + "\n")
torch.save({"eA": eA, "eB": eB, "eAll": eAll, "A": A, "B": B, "top": top}, out + ".pt")
print("LOGIC_DONE")
