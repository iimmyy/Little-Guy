#!/usr/bin/env python3
"""Does the patient carry its own clock? Step 1 of the baked timer (test 4: closing the thinking at word 128 beat the
overthinking detector; brake.py warm=128). A baked timer is one MLP neuron: it can only read the stream in front of
its layer's MLP, one word at a time, no counter. So it works only if that stream already says "I am thinking, and I
have been for more than W words" along one straight line.

Per layer, at the exact input of the layer's MLP (after its norm, what a new neuron in that MLP would see), on
thinking words and reply words of real answers: a straight-line reader (logistic regression) for
  y = 1  inside the thinking, at or after word W
  y = 0  inside the thinking before word W, or in the reply (it must stay quiet there, or it writes the closing word
         into the answer)
built on half the answers, scored on the other half:
  auc       chance a random y=1 word scores above a random y=0 word
  fire      per held-out answer, the first thinking word where the reader crosses its set point: median and the
            middle half (the hook timer fires at exactly W)
  early     share of thinking words before W/2 where it is already over the set point
  reply     share of reply words over the set point (should be ~0)
Saved per layer for the bake (clock_dirs.pt): direction, offset, set point, scores.

usage: clock_probe.py <patient> <answers jsonl> <out folder> [w=128 n=60 every=2 per=96 dev=cuda]
"""
import json, os, random, sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from patients import load

name, src, out = sys.argv[1:4]
opt = dict(a.split("=", 1) for a in sys.argv[4:])
W, NANS, EVERY, PER, DEV = (int(opt.get("w", 128)), int(opt.get("n", 60)), int(opt.get("every", 2)),
                            int(opt.get("per", 96)), opt.get("dev", "cuda"))
os.makedirs(out, exist_ok=True)
TAIL = "\n\nEnd with a line: Answer: <your answer>"
OPEN, CLOSE = {"e4b": ("<|channel>thought\n", "<channel|>"), "lfm3": ("\n", "\n</think>\n\n")}[name]
MLP = {"e4b": lambda l: l.mlp, "lfm3": lambda l: l.feed_forward}[name]
p = load(name, DEV)
tok, layers = p.tok, p.model.model.layers
LAY = list(range(0, len(layers), EVERY))
rows = [r for r in map(json.loads, open(src)) if r["finish"] == "stop" and r["thinking"]][:NANS]
rng = random.Random(0)

cap, pos = {}, [None]
for i in LAY:
    MLP(layers[i]).register_forward_pre_hook(lambda m, a, i=i: cap.__setitem__(i, a[0][0, pos[0]].float().cpu()))

feats = {i: [] for i in LAY}
meta = []                                                  # per sample: (answer, thinking word index or -1, y)
with torch.no_grad():
    for n, r in enumerate(rows):
        pr = tok(tok.apply_chat_template([{"role": "user", "content": r["q"] + TAIL}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=True), add_special_tokens=False).input_ids
        op, th = tok(OPEN, add_special_tokens=False).input_ids, tok(r["thinking"], add_special_tokens=False).input_ids
        cl, rp = tok(CLOSE, add_special_tokens=False).input_ids, tok(r["reply"], add_special_tokens=False).input_ids[:400]
        T, base = len(th), len(pr) + len(op)
        # thinking words: all of them up to 2W (where the set point is decided), then a sample of the rest
        tk = list(range(min(T, 2 * W))) + sorted(rng.sample(range(2 * W, T), min(PER, max(0, T - 2 * W))))
        rk = sorted(rng.sample(range(len(rp)), min(PER // 3, len(rp))))
        ids = pr + op + th + cl + rp
        pos[0] = torch.tensor([base + t for t in tk] + [base + T + len(cl) + k for k in rk])
        cap.clear()
        p.model.model(input_ids=torch.tensor([ids], device=DEV))
        for i in LAY:
            feats[i].append(cap[i].half())
        meta += [(n, t, int(t >= W)) for t in tk] + [(n, -1, 0) for _ in rk]
        print(f"  {n + 1}/{len(rows)}: {T} thinking words", flush=True)

A = np.array([m[0] for m in meta]); Tw = np.array([m[1] for m in meta]); Y = torch.tensor([m[2] for m in meta]).float()
tr = torch.tensor(A % 2 == 0); te = ~tr


def auc(s, y):
    s, y = np.asarray(s), np.asarray(y)
    o = s.argsort().argsort().astype(float)
    n1 = y.sum(); n0 = len(y) - n1
    return float((o[y == 1].sum() - n1 * (n1 - 1) / 2) / max(1, n1 * n0))


lines = [f"Clock inside {name}: can one straight line read 'thinking, past word {W}'? {len(rows)} answers, built on half,",
         "scored on the other half (fire = first thinking word over the set point; the hook timer fires at exactly W)", "",
         f"{'layer':>5} {'auc':>6} {'fire median':>12} {'middle half':>14} {'early':>6} {'reply':>6}"]
save = {}
for i in LAY:
    X = torch.cat(feats[i]).float()
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-4
    Z = (X - mu) / sd
    w = torch.zeros(X.shape[1], requires_grad=True); b = torch.zeros(1, requires_grad=True)
    opt_ = torch.optim.LBFGS([w, b], max_iter=200, line_search_fn="strong_wolfe")

    def step():
        opt_.zero_grad()
        z = Z[tr] @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, Y[tr]) + 1e-3 * w.pow(2).sum()
        loss.backward()
        return loss
    opt_.step(step)
    with torch.no_grad():
        s = (Z @ w + b).numpy()
    # set point: on the training answers, the score that makes the median first crossing land at word W
    def fires(mask, thr):
        out_ = []
        for a in np.unique(A[mask]):
            k = (A == a) & (Tw >= 0)
            t, sc = Tw[k], s[k]
            o = np.argsort(t); t, sc = t[o], sc[o]
            hit = np.nonzero(sc > thr)[0]
            out_.append(t[hit[0]] if len(hit) else 10 ** 6)
        return np.array(out_)
    cands = np.quantile(s[tr.numpy()], np.linspace(0.05, 0.995, 120))
    thr = min(cands, key=lambda c: abs(np.median(fires(tr.numpy(), c)) - W))
    f = fires(te.numpy(), thr)
    ten = te.numpy()
    early = float((s[ten & (Tw >= 0) & (Tw < W // 2)] > thr).mean())
    reply = float((s[ten & (Tw < 0)] > thr).mean())
    a = auc(s[ten], Y.numpy()[ten])
    q1, q3 = np.quantile(f, [0.25, 0.75])
    lines.append(f"{i:>5} {a:>6.3f} {np.median(f):>12.0f} {f'{q1:.0f}-{q3:.0f}':>14} {early:>6.2f} {reply:>6.2f}")
    save[i] = {"w": (w.detach() / sd), "b": float(b) - float((w.detach() / sd) @ mu), "thr": float(thr), "auc": a,
               "fire_median": float(np.median(f)), "early": early, "reply": reply}
    print(lines[-1], flush=True)
torch.save(save, os.path.join(out, "clock_dirs.pt"))
open(os.path.join(out, "clock_probe.txt"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
print("CLOCK_DONE")
