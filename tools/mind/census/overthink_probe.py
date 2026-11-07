#!/usr/bin/env python3
"""Can the patient's own stream tell when it is overthinking? Step 4 of the gated brake (Immy, 2026-10-28).

Labels from overthink_label.py: thinking words after the point where the forced answer was already right and stayed
right = overthinking; words before the last wrong cut = working; the stretch in between is left out (not known).
Only answers whose final answer was right and that settled before the end.

Per layer, the stream at those words: a straight-line detector (difference of the two classes' means, the simplest
probe) built on half the problems and scored on the other half (never on its own problems):
  auc       chance a random overthinking word scores above a random working word (0.5 = blind, 1 = perfect)
  auc_late  the same, after removing what the word's position in the thinking alone predicts (later words are more
            often overthinking; a detector that only counts time is not reading the state)
  auc_bin   words compared only with words at the same point of their thinking (10 bins), detector built the same way
  cos_brief how much the detector points along the "Be brief." change (brief_patient.py), +1 same way, -1 opposite
Saved: per layer the detector direction, its threshold scale, and the scores, for the gate (overthink_dirs.pt).

usage: overthink_probe.py <patient> <labels .graded.jsonl> <answers jsonl> <brief_dirs.pt> <out folder> [per=64 dev=cuda]
"""
import json, os, random, sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from patients import load

name, lab_path, ans_path, brief_path, out = sys.argv[1:6]
opt = dict(a.split("=", 1) for a in sys.argv[6:])
PER, DEV = int(opt.get("per", 64)), opt.get("dev", "cuda")
os.makedirs(out, exist_ok=True)
TAIL = "\n\nEnd with a line: Answer: <your answer>"
OPEN, CLOSE = {"e4b": ("<|channel>thought\n", "<channel|>"), "lfm3": ("\n", "\n</think>\n\n")}[name]
p = load(name, DEV)
tok, L = p.tok, len(p.model.model.layers)
ans = {r["id"]: r for r in map(json.loads, open(ans_path))}
labs = [r for r in map(json.loads, open(lab_path)) if r["solved_cut"] is not None and r["forced_right"][-1]]
brief = torch.load(brief_path)
rng = random.Random(0)

feats = {i: [] for i in range(L)}          # per layer: (problem, word position share, label, stream)
meta = []
with torch.no_grad():
    for n, lb in enumerate(labs):
        r = ans[lb["id"]]
        pr = tok(tok.apply_chat_template([{"role": "user", "content": r["q"] + TAIL}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=True), add_special_tokens=False).input_ids
        op = tok(OPEN, add_special_tokens=False).input_ids
        th = tok(r["thinking"], add_special_tokens=False).input_ids
        T, j = len(th), lb["solved_cut"]
        known = lb["cuts"][j]                                   # had the answer by here
        unsure = lb["cuts"][j - 1] if j > 0 else 0              # still wrong at this cut
        over = list(range(known, T)); work = list(range(0, unsure))
        if not over:
            continue
        pick = [(t, 1) for t in rng.sample(over, min(PER, len(over)))] + \
               [(t, 0) for t in rng.sample(work, min(PER, len(work)))]
        ids = torch.tensor([pr + op + th[:max(t for t, _ in pick) + 1]], device=DEV)
        hs = p.model.model(input_ids=ids, output_hidden_states=True).hidden_states
        base = len(pr) + len(op)
        pos = torch.tensor([base + t for t, _ in pick], device=DEV)
        for i in range(L):
            h = hs[i + 1][0, pos].float().cpu()                 # the stream after layer i
            for (t, y), v in zip(pick, h):
                feats[i].append((n, t / T, y, v))
        meta.append({"id": lb["id"], "over": len(over), "work": len(work), "T": T})
        print(f"  {n + 1}/{len(labs)} {lb['id']}: {len(work)} working, {len(over)} overthinking words", flush=True)


def auc(s, y):
    s, y = np.asarray(s), np.asarray(y)
    o = s.argsort().argsort().astype(float)
    n1 = y.sum(); n0 = len(y) - n1
    return float((o[y == 1].sum() - n1 * (n1 - 1) / 2) / max(1, n1 * n0))


probs = sorted({f[0] for f in feats[0]})
half = set(probs[::2])
te0 = [f for f in feats[0] if f[0] not in half]
lines = [f"Overthinking inside {name}: {len(meta)} answers, detector built on half the problems, scored on the other",
         f"(for scale: the word's position in its thinking alone gives auc {auc([f[1] for f in te0], [f[2] for f in te0]):.3f})",
         "", f"{'layer':>5} {'auc':>6} {'auc_late':>9} {'auc_bin':>8} {'cos_brief':>10}"]
save = {}
for i in range(L):
    F = feats[i]
    tr = [f for f in F if f[0] in half]; te = [f for f in F if f[0] not in half]
    X1 = torch.stack([f[3] for f in tr if f[2] == 1]); X0 = torch.stack([f[3] for f in tr if f[2] == 0])
    d = X1.mean(0) - X0.mean(0)
    w = d / d.norm()
    s_te = [float(f[3] @ w) for f in te]; y_te = [f[2] for f in te]
    # remove what position alone predicts: fit score ~ position on the training half, score the leftovers
    s_tr = np.array([float(f[3] @ w) for f in tr]); t_tr = np.array([f[1] for f in tr])
    k, c = np.polyfit(t_tr, s_tr, 1)
    s_late = [s - (k * f[1] + c) for s, f in zip(s_te, te)]
    # fairer: compare words at the same point of their thinking (10 bins by position share). Built from the class
    # difference inside each bin (so time can't be what it learns), scored inside each bin on the held-out problems
    bins = lambda f: min(9, int(f[1] * 10))
    dd, nb = 0, 0
    for k_ in range(10):
        a1 = [f[3] for f in tr if f[2] == 1 and bins(f) == k_]; a0 = [f[3] for f in tr if f[2] == 0 and bins(f) == k_]
        if len(a1) >= 5 and len(a0) >= 5:
            dd = dd + torch.stack(a1).mean(0) - torch.stack(a0).mean(0); nb += 1
    wb = dd / dd.norm() if nb else w
    num = den = 0.0
    for k_ in range(10):
        e = [f for f in te if bins(f) == k_]
        y_ = [f[2] for f in e]
        if 0 < sum(y_) < len(y_):
            pairs = sum(y_) * (len(y_) - sum(y_))
            num += auc([float(f[3] @ wb) for f in e], y_) * pairs; den += pairs
    a_bin = num / den if den else float("nan")
    bd = brief[i]["mean"]
    cb = float(torch.nn.functional.cosine_similarity(w, bd, dim=0)) if bd.norm() > 0 else 0.0
    a, al = auc(s_te, y_te), auc(s_late, y_te)
    lines.append(f"{i:>5} {a:>6.3f} {al:>9.3f} {a_bin:>8.3f} {cb:>+10.3f}")
    mid = float((X1 @ w).mean() + (X0 @ w).mean()) / 2
    m1, m0 = float((X1 @ wb).mean()), float((X0 @ wb).mean())         # the gate's set point: between the two classes
    save[i] = {"dir": w, "dir_bin": wb, "auc_bin": a_bin, "mid_bin": (m1 + m0) / 2, "over_bin": m1, "work_bin": m0,
               "sd_bin": float(torch.cat([X1 @ wb, X0 @ wb]).std()), "mid": mid, "spread": float(torch.cat([X1 @ w, X0 @ w]).std()), "auc": a, "auc_late": al,
               "cos_brief": cb}
torch.save(save, os.path.join(out, "overthink_dirs.pt"))
json.dump(meta, open(os.path.join(out, "overthink_meta.json"), "w"))
open(os.path.join(out, "overthink_probe.txt"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
print("PROBE_DONE")
