#!/usr/bin/env python3
"""Does he know he is lost? (Immy 2026-10-30, after the wrong-turn count: 241 of 242 failures run out of words, and
being lost burns 43% of all his thinking.) At the same word P, compare runs of the SAME question that have not yet
stated the right answer:
  lost        never states the right answer anywhere in the run (failure A)
  on track    states it later than P (and mostly ends right)
Runs that already stated it before P are left out (that is the known "I have it" signal, confidence_probe.py).
Feature: each layer's stream averaged over the last 64 thinking words before P. A ridge line is fit on 4/5 of the
questions and scored on the other 1/5 (questions never shared), so it cannot learn which question is hard.
Scores: AUROC over all held-out runs, and the honest one, pairs within a question (on track scored above lost).
Cheap controls: layer 0 (sees little more than the words), doubt-word rate so far, repeated-line share so far.

usage: lost_probe.py <answers jsonl[,more]> <out prefix> [hf=~/.littleguy/hf/lfm3-1.7b at=250,500,1000,1500,2000,3000
                     win=64 fix=8 dev=mps nq=]
The read is saved after every question (<out>_feats.pt) and resumed on a rerun; the scores are cheap.
"""
import json, os, re, sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../expert"))
ARGS, sys.argv[1:] = sys.argv[1:], []                     # grade.py grades its own arguments on import
from grade import right, right_reply

srcs, out = ARGS[0].split(","), ARGS[1]
opt = dict(a.split("=", 1) for a in ARGS[2:])
HF = os.path.expanduser(opt.get("hf", "~/.littleguy/hf/lfm3-1.7b"))
AT = [int(x) for x in opt.get("at", "250,500,1000,1500,2000,3000").split(",")]
WIN, DEV = int(opt.get("win", 64)), opt.get("dev", "mps")
FIX = int(opt.get("fix", 8))                               # layer named before looking: the confidence probe's best
TAIL = "\n\nEnd with a line: Answer: <your answer>"
CLAIM = re.compile(r"\\boxed\{([^{}]{1,40})\}|(?:answer is|answer =|answer:|answer)\s*:?\s*\$?([^\n$]{1,40}?)\$?(?=[.\n]|$)", re.I)
DOUBT = re.compile(r"\b(wait|but|alternatively|hmm|actually|however|maybe|perhaps|let me re|let's re|double-check|recheck)\b", re.I)


def first_right_claim(r):
    """Character where the run first states the right answer in its thinking (None if never)."""
    for mm in CLAIM.finditer(r["thinking"] or ""):
        x = next(g for g in mm.groups() if g)
        if right({**r, "reply": x}, x):
            return mm.start()
    return None


runs = {}
for s in srcs:
    for l in open(s):
        r = json.loads(l)
        if len((r["thinking"] or "").split()) > 50:
            good = r["finish"] == "stop" and bool(r["reply"]) and right_reply(r, r["reply"])
            runs.setdefault(r["id"], []).append({"good": good, "claim": first_right_claim(r), "r": r})
# a question counts when it has a lost run and a run that finds the answer
qs = sorted(q for q, rs in runs.items() if any(x["claim"] is None and not x["good"] for x in rs)
            and any(x["claim"] is not None or x["good"] for x in rs))
qs = qs[:int(opt["nq"])] if "nq" in opt else qs      # nq=: a few questions, for a smoke test
print(f"{len(qs)} questions with a lost run and a run that finds it ({sum(len(runs[q]) for q in qs)} runs)", flush=True)

FEATS = out + "_feats.pt"
F = torch.load(FEATS, weights_only=False) if os.path.exists(FEATS) else {}
if F:
    print(f"resuming: {len(F)} runs already read", flush=True)
todo = [(q, k) for q in qs for k in range(len(runs[q])) if (q, k) not in F]
if todo:
    from patients import _lfm3
    p = _lfm3(DEV, HF)
    tok, m = p.tok, p.model
    with torch.no_grad():
        for n, q in enumerate(qs):
            for k, x in enumerate(runs[q]):
                if (q, k) in F:
                    continue
                r = x["r"]
                pr = tok(tok.apply_chat_template([{"role": "user", "content": r["q"] + TAIL}], tokenize=False,
                                                 add_generation_prompt=True, enable_thinking=True), add_special_tokens=False).input_ids
                txt = "\n" + r["thinking"]
                enc = tok(txt, add_special_tokens=False, return_offsets_mapping=True)
                pts = [P for P in AT if P < len(enc.input_ids)]   # still thinking at P
                rec = {"n_tok": len(enc.input_ids), "at": {}}
                if pts:
                    th = enc.input_ids[:max(pts)]
                    o = m(input_ids=torch.tensor([pr + th], device=DEV), output_hidden_states=True)
                    hs = torch.stack(o.hidden_states)[:, 0]          # layers+1, words, width
                    for P in pts:
                        ch = enc.offset_mapping[P][0]               # character where word P starts
                        seen = txt[:ch]
                        lines = [z.strip() for z in seen.split("\n") if len(z.strip()) > 10]
                        rec["at"][P] = {
                            "h": hs[:, len(pr) + P - WIN:len(pr) + P].float().mean(1).half().cpu(),
                            "char": ch,
                            "doubt": len(DOUBT.findall(seen)) / max(1, len(seen.split())) * 100,
                            "repeat": 1 - len(set(lines)) / max(1, len(lines)),
                        }
                    del o, hs
                F[(q, k)] = rec
            torch.save(F, FEATS)
            print(f"  {n + 1}/{len(qs)} {q}", flush=True)
    if DEV == "mps":
        torch.mps.empty_cache()


def auroc(s, y):
    """Chance that a random positive (on track) scores above a random negative (lost); ties count half."""
    s, y = np.asarray(s), np.asarray(y)
    a, b = s[y == 1], s[y == 0]
    if not len(a) or not len(b):
        return float("nan")
    return float(((a[:, None] > b[None]).sum() + 0.5 * (a[:, None] == b[None]).sum()) / (len(a) * len(b)))


def ridge_dual(X, y, Xt, alphas=(1e1, 1e2, 1e3, 1e4, 1e5)):
    """Ridge line in its n x n form (few runs, wide stream), strength picked by 5-fold AUROC on the train rows."""
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z, Zt = (X - mu) / sd, (Xt - mu) / sd
    folds = np.arange(len(y)) % 5

    def fit(A, ya, al):
        K = A @ A.T
        c = np.linalg.solve(K + al * np.eye(len(ya)), ya - ya.mean())
        return A.T @ c, ya.mean()
    best = None
    for al in alphas:
        pr = np.zeros(len(y))
        for f in range(5):
            a, b = folds != f, folds == f
            w, b0 = fit(Z[a], y[a], al)
            pr[b] = Z[b] @ w + b0
        s = auroc(pr, y)
        if best is None or s > best[0]:
            best = (s, al)
    w, b0 = fit(Z, y, best[1])
    return Zt @ w + b0


def pairs_within(s, y, g):
    """Within each question: share of (on track, lost) pairs where on track scores higher; pooled over questions,
    plus the number of questions where on track wins on average."""
    hit = tot = 0
    wins = qn = 0
    for q in np.unique(g):
        a, b = s[(g == q) & (y == 1)], s[(g == q) & (y == 0)]
        if len(a) and len(b):
            c = (a[:, None] > b[None]).sum() + 0.5 * (a[:, None] == b[None]).sum()
            hit += c; tot += len(a) * len(b)
            wins += c / (len(a) * len(b)) > 0.5; qn += 1
    return (hit / tot if tot else float("nan")), int(wins), qn


lines = [f"Does he know he is lost? {len(qs)} questions; lost = never states the right answer, on track = states it "
         f"after word P. Held out by question (5 folds). AUROC: 0.5 = chance, 1 = always tells them apart.", ""]
gq = {q: i % 5 for i, q in enumerate(np.random.RandomState(0).permutation(qs))}
for P in AT:
    rows = []
    for q in qs:
        for k, x in enumerate(runs[q]):
            a = F[(q, k)]["at"].get(P)
            if a is None:
                continue                                    # finished before P
            if x["claim"] is not None and x["claim"] < a["char"]:
                continue                                    # already has it at P
            lost = x["claim"] is None and not x["good"]
            rows.append((q, int(not lost), a))
    keep = {q for q in qs if {y for qq, y, _ in rows if qq == q} == {0, 1}}   # both kinds still present at P
    rows = [z for z in rows if z[0] in keep]
    if len(keep) < 6:
        lines.append(f"word {P}: only {len(keep)} questions still have both kinds, skipped")
        continue
    g = np.array([z[0] for z in rows]); y = np.array([z[1] for z in rows])
    fold = np.array([gq[q] for q in g])
    H = torch.stack([z[2]["h"] for z in rows]).float().numpy()          # runs, layers+1, width
    def held_out(yy):
        out = {}
        for L in range(H.shape[1]):
            s = np.zeros(len(yy))
            for f in range(5):
                tr, te = fold != f, fold == f
                if te.any():
                    s[te] = ridge_dual(H[tr, L], yy[tr], H[te, L])
            out[L] = (auroc(s, yy), *pairs_within(s, yy, g))
        return out
    res = held_out(y)
    best = max(range(1, H.shape[1]), key=lambda L: res[L][0])
    # luck control: the same pipeline with the labels shuffled inside each question (no real signal left)
    sh = y.copy(); rs = np.random.RandomState(P)
    for q in np.unique(g):
        sh[g == q] = rs.permutation(sh[g == q])
    luck = held_out(sh)
    luck_best = max(luck[L][0] for L in range(1, H.shape[1]))
    cheap = {nm: -np.array([z[2][nm] for z in rows]) for nm in ("doubt", "repeat")}   # more doubt/repeats = lost
    lines.append(f"word {P}: {len(keep)} questions, {int(y.sum())} on track / {int((1 - y).sum())} lost runs")
    lines.append(f"  best layer {best:2d}  AUROC {res[best][0]:.2f}   within-question pairs {res[best][1]:.2f}"
                 f" ({res[best][2]}/{res[best][3]} questions)")
    lines.append(f"  layer {FIX} (fixed)   AUROC {res[FIX][0]:.2f}   within-question pairs {res[FIX][1]:.2f}"
                 f" ({res[FIX][2]}/{res[FIX][3]})   <- chosen in advance (confidence probe's layer)")
    lines.append(f"  luck: best layer with shuffled labels AUROC {luck_best:.2f} (what picking the best of 24 gives by chance)")
    lines.append(f"  layer 0          AUROC {res[0][0]:.2f}   within-question pairs {res[0][1]:.2f} ({res[0][2]}/{res[0][3]})")
    for nm, s in cheap.items():
        pw = pairs_within(s, y, g)
        lines.append(f"  {nm + ' rate':16} AUROC {auroc(s, y):.2f}   within-question pairs {pw[0]:.2f} ({pw[1]}/{pw[2]})")
    lines.append("  by layer (AUROC): " + " ".join(f"{L}:{res[L][0]:.2f}" for L in range(H.shape[1])))
    lines.append("")
open(out + ".txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
print("LOST_DONE")
