#!/usr/bin/env python3
"""Does his thinking "tighten" when it goes well? (Immy 2026-10-29: flow state; "the model internals are tightening,
things get tighter and closer to being grabbed, the equation tightens".) A mode, not a neuron cluster: a few numbers
per run, measured over the EARLY thinking (words 0..W, before the outcome shows), compared between good and bad runs
of the SAME question (sampled answers from the untouched model):
  entropy     how scattered his next-word choice is (nats; lower = more focused)
  lock-in     how deep he decides: the first layer from which every later layer's draft (logit lens: that layer's
              stream through the final norm and output rows) already has the final top word; as a share of the depth
  mid gap     KL from the final distribution to layer 12's draft (how far the middle is from the decision)
Per question: mean over good runs minus mean over bad runs; then the sign across questions.

usage: tightness.py <answers jsonl[,more]> <out prefix> [hf=~/.littleguy/hf/lfm3-1.7b early=400 label=right|reached
                    mid=12 dev=mps]
"""
import json, os, re, sys

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
W, MID, DEV, LABEL = int(opt.get("early", 400)), int(opt.get("mid", 12)), opt.get("dev", "mps"), opt.get("label", "right")
AT = opt.get("at", "all")                                  # all words, or only sentence openings (at=sentence)
TAIL = "\n\nEnd with a line: Answer: <your answer>"
CLAIM = re.compile(r"\\boxed\{([^{}]{1,40})\}|(?:answer is|answer =|answer:|answer)\s*:?\s*\$?([^\n$]{1,40}?)\$?(?=[.\n]|$)", re.I)

runs = {}
for s in srcs:
    for l in open(s):
        r = json.loads(l)
        good = r["finish"] == "stop" and bool(r["reply"]) and right_reply(r, r["reply"])
        if LABEL == "reached":
            good = good or any(right({**r, "reply": x}, x) for m in CLAIM.finditer(r["thinking"] or "")
                               for x in [next(g for g in m.groups() if g)])
        if len((r["thinking"] or "").split()) > 50:
            runs.setdefault(r["id"], []).append((good, r))
qs = sorted(i for i, rs in runs.items() if any(g for g, _ in rs) and not all(g for g, _ in rs))
print(f"{len(qs)} questions with good and bad runs ({sum(len(runs[q]) for q in qs)} runs), label {LABEL}", flush=True)

p = _lfm3(DEV, HF)
tok, m = p.tok, p.model
norm, head = m.model.embedding_norm, m.lm_head
NAMES = ["entropy", "lock-in", "mid gap"]
res, SER = {}, {}                                          # per-run means; per-run series over the thinking
PART = out + "_partial.pt"                                 # saved after every question, so a timeout keeps the work
if os.path.exists(PART):
    c = torch.load(PART, weights_only=False)
    res, SER = c["res"], c["series"]
    res = {(q, k): (runs[q][k][0], v) for (q, k), (_, v) in res.items() if q in runs and k < len(runs[q])}
    print(f"resuming: {len(res)} runs already read", flush=True)
with torch.no_grad():
    for n, q in enumerate(qs):
        for k, (g, r) in enumerate(runs[q]):
            if (q, k) in res:
                continue
            pr = tok(tok.apply_chat_template([{"role": "user", "content": r["q"] + TAIL}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=True), add_special_tokens=False).input_ids
            enc = tok("\n" + r["thinking"], add_special_tokens=False, return_offsets_mapping=True)
            th = enc.input_ids[:W]
            txt = "\n" + r["thinking"]
            # decision points: the word before a new sentence starts (after a newline, or ". " / "? " / "! ")
            dec = [t for t in range(1, len(th)) if (lambda a: txt[max(0, a - 2):a].rstrip(" ").endswith(("\n", ".", "?", "!")))(enc.offset_mapping[t][0])]
            o = m(input_ids=torch.tensor([pr + th], device=DEV), output_hidden_states=True)
            sl = slice(len(pr) - 1, len(pr) + len(th) - 1)              # positions that pick the thinking words
            if AT == "sentence":                                          # only where he picks how the next sentence opens
                sl = torch.tensor([len(pr) + t - 1 for t in dec], device=DEV)
            lf = torch.log_softmax(o.logits[0, sl].float(), -1)
            ent = float(-(lf.exp() * lf).sum(-1).mean())
            top = lf.argmax(-1)
            hs = o.hidden_states[1:]                                      # after each layer (the last one is pre-norm)
            agree = []
            for h in hs:
                agree.append((head(norm(h[0, sl])).argmax(-1) == top))
            A = torch.stack(agree)                                        # layers x words
            ok_from = torch.flip(torch.cumprod(torch.flip(A.int(), [0]), 0), [0])   # 1 where this and all later agree
            lock = float((ok_from.shape[0] - ok_from.sum(0)).float().mean() / ok_from.shape[0])
            lm = torch.log_softmax(head(norm(hs[MID][0, sl])).float(), -1)
            gap = float((lf.exp() * (lf - lm)).sum(-1).mean())
            res[(q, k)] = (g, [ent, lock, gap])
            SER[(q, k)] = {"ent": (-(lf.exp() * lf).sum(-1)).cpu(), "lock": ((ok_from.shape[0] - ok_from.sum(0)).float() / ok_from.shape[0]).cpu(),
                           "pos": (sl if torch.is_tensor(sl) else torch.arange(sl.start, sl.stop)).cpu() - len(pr), "len": len(th)}
        print(f"  {n + 1}/{len(qs)}: " + "  ".join(f"{'G' if g else 'b'} " + "/".join(f"{v:.2f}" for v in res[(q, k)][1])
                                                     for k, (g, _) in enumerate(runs[q])), flush=True)
        torch.save({"res": res, "series": SER}, PART)
torch.save({"res": res, "series": SER, "qs": qs, "names": NAMES}, out + ".pt")

lines = [f"tightness at {AT} words, first {W} words, {len(qs)} questions with good and bad runs (label {LABEL})",
         "measure      good - bad per question (median)   questions where good is TIGHTER   good mean / bad mean"]
for i, nm in enumerate(NAMES):
    d = []
    gm, bm = [], []
    for q in qs:
        G = [v[i] for (qq, k), (g, v) in res.items() if qq == q and g]
        B = [v[i] for (qq, k), (g, v) in res.items() if qq == q and not g]
        d.append(np.mean(G) - np.mean(B)); gm += G; bm += B
    tighter = sum(x < 0 for x in d)                                       # lower = tighter for all three
    lines.append(f"  {nm:9s}  {np.median(d):+.3f}                          {tighter}/{len(d)}"
                 f"                            {np.mean(gm):.3f} / {np.mean(bm):.3f}")
print("\n".join(lines))
open(out + ".txt", "w").write("\n".join(lines) + "\n")
print("TIGHT_DONE")
