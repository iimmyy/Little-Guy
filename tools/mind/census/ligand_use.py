#!/usr/bin/env python3
"""Which signals does he have but barely use? (Immy 2026-10-31: a principle must bolt on like a PAM: boost a ligand the
model already makes, at the receptors it already reaches.) At every sentence start in his thinking:
  ligands   three things a straight line can read off his page (layer stream at that word), each fit on 4/5 of the
            questions and read on the other 1/5:
              belief      the right answer is already stated (yes/no)
              clock       how far in he is (log word count)
              ease        how short this question's thinking usually runs (minus log mean thinking words of its runs)
  behavior  his wrap-up drive: score of the hand-in openers minus score of the doubt openers (the belief gate's word
            lists), i.e. how much he leans toward "I'll write the answer" over "But / Wait".
clarity = how well his page carries the ligand (held out); use = how much the wrap-up drive follows his own read of it
(rank correlation inside each question, then averaged). All three should push the drive up. Clear but unused = a target.

usage: ligand_use.py <answers jsonl[,more]> <out prefix> [hf=~/.littleguy/hf/lfm3-1.7b layers=4,8,12,16,20 per=40
                     maxw=3000 gate=<belief.json> dev=mps]
"""
import json, os, re, sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../expert"))
ARGS, sys.argv[1:] = sys.argv[1:], []                     # grade.py grades its own arguments on import
from grade import right, right_reply
from probe_math import rank, rho

srcs, out = ARGS[0].split(","), ARGS[1]
opt = dict(a.split("=", 1) for a in ARGS[2:])
HF = os.path.expanduser(opt.get("hf", "~/.littleguy/hf/lfm3-1.7b"))
LAYERS = [int(x) for x in opt.get("layers", "4,8,12,16,20").split(",")]
PER, MAXW, DEV = int(opt.get("per", 40)), int(opt.get("maxw", 3000)), opt.get("dev", "mps")
GATE = json.load(open(opt["gate"]))
HAND, DOUBT = GATE["hand_ids"], GATE["doubt_ids"]
TAIL = "\n\nEnd with a line: Answer: <your answer>"
CLAIM = re.compile(r"\\boxed\{([^{}]{1,40})\}|(?:answer is|answer =|answer:|answer)\s*:?\s*\$?([^\n$]{1,40}?)\$?(?=[.\n]|$)", re.I)


def first_right_claim(r):
    for mm in CLAIM.finditer(r["thinking"] or ""):
        x = next(g for g in mm.groups() if g)
        if right({**r, "reply": x}, x):
            return mm.end()
    return None


runs = {}
for s in srcs:
    for l in open(s):
        r = json.loads(l)
        if len((r["thinking"] or "").split()) > 50:
            runs.setdefault(r["id"], []).append(r)
qs = sorted(runs)
ease = {q: -float(np.mean([np.log(len(r["thinking"].split())) for r in runs[q]])) for q in qs}
print(f"{len(qs)} questions, {sum(len(v) for v in runs.values())} runs", flush=True)

RAW = out + "_raw.pt"
D = torch.load(RAW, weights_only=False) if os.path.exists(RAW) else None
if D is None:
    from patients import _lfm3
    p = _lfm3(DEV, HF)
    tok, m = p.tok, p.model
    print("hand-in openers:", [tok.decode([i]) for i in HAND], " doubt openers:", [tok.decode([i]) for i in DOUBT], flush=True)
    H = {L: [] for L in LAYERS}
    meta = {"q": [], "pos": [], "belief": [], "drive": []}
    with torch.no_grad():
        for n, q in enumerate(qs):
            for r in runs[q]:
                pr = tok(tok.apply_chat_template([{"role": "user", "content": r["q"] + TAIL}], tokenize=False,
                                                 add_generation_prompt=True, enable_thinking=True), add_special_tokens=False).input_ids
                txt = "\n" + r["thinking"]
                enc = tok(txt, add_special_tokens=False, return_offsets_mapping=True)
                th = enc.input_ids[:MAXW]
                dec = [t for t in range(8, len(th)) if txt[max(0, enc.offset_mapping[t][0] - 2):enc.offset_mapping[t][0]].rstrip(" ").endswith(("\n", ".", "?", "!"))]
                if not dec:
                    continue
                dec = [dec[i] for i in np.linspace(0, len(dec) - 1, min(PER, len(dec))).round().astype(int)]
                o = m(input_ids=torch.tensor([pr + th], device=DEV), output_hidden_states=True)
                at = torch.tensor([len(pr) + t - 1 for t in dec], device=DEV)   # the word that picks the next opener
                lg = o.logits[0, at].float()
                drive = torch.logsumexp(lg[:, HAND], -1) - torch.logsumexp(lg[:, DOUBT], -1)
                c = first_right_claim(r)
                for L in LAYERS:
                    H[L].append(o.hidden_states[L][0, at].float().cpu().half())
                meta["q"] += [q] * len(dec); meta["pos"] += dec
                meta["belief"] += [int(c is not None and c + 1 <= enc.offset_mapping[t][0]) for t in dec]
                meta["drive"] += drive.cpu().tolist()
                del o
            print(f"  {n + 1}/{len(qs)} {q}", flush=True)
    D = {"H": {L: torch.cat(v) for L, v in H.items()}, **meta}
    torch.save(D, RAW)

g = np.array(D["q"]); pos = np.array(D["pos"], float); drive = np.array(D["drive"])
targets = {"belief": np.array(D["belief"], float), "clock": np.log(pos), "ease": np.array([ease[q] for q in g])}
fold = np.array([{q: i % 5 for i, q in enumerate(np.random.RandomState(0).permutation(qs))}[q] for q in g])


def ridge_read(X, y, al=1e3):
    """Held-out read: fit on 4/5 of the questions, read the other 1/5 (fixed strength; plenty of rows)."""
    s = np.zeros(len(y))
    for f in range(5):
        tr, te = fold != f, fold == f
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Z = (X[tr] - mu) / sd
        w = np.linalg.solve(Z.T @ Z + al * np.eye(Z.shape[1]), Z.T @ (y[tr] - y[tr].mean()))
        s[te] = ((X[te] - mu) / sd) @ w
    return s


def auroc(s, y):
    """Chance a moment with the answer stated reads higher than one without (Mann-Whitney; ranks start at 0)."""
    rk = rank(s) + 1
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    return float((rk[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def within(a, b):
    """Rank agreement inside each question (questions with >= 10 moments and some spread), averaged."""
    v = [rho(a[g == q], b[g == q]) for q in np.unique(g) if (g == q).sum() >= 10 and a[g == q].std() > 0 and b[g == q].std() > 0]
    return float(np.mean(v)), len(v)


lines = [f"Which signals does he have but barely use? {len(qs)} questions, {len(g)} sentence starts (up to {PER} per run, "
         f"thinking words < {MAXW}). Wrap-up drive = hand-in openers minus doubt openers (higher = closer to handing in).",
         f"drive overall: median {np.median(drive):+.2f}; with the right answer already stated {np.median(drive[targets['belief'] == 1]):+.2f}, "
         f"not yet {np.median(drive[targets['belief'] == 0]):+.2f}", "",
         "ligand   layer  clarity (held out)          use: drive follows his read (within question)   drive follows the truth"]
best = {}
for nm, y in targets.items():
    for L in LAYERS:
        s = ridge_read(D["H"][L].float().numpy(), y)
        cl = (f"AUROC {auroc(s, y):.2f}" if nm == "belief" else f"rho {rho(s, y):+.2f}")
        if nm == "ease":                                    # one value per question: clarity across questions
            qm = np.array([s[g == q].mean() for q in qs]); cl = f"rho {rho(qm, np.array([ease[q] for q in qs])):+.2f} (per question)"
        u, nq = within(s, drive) if nm != "ease" else (rho(np.array([s[g == q].mean() for q in qs]), np.array([drive[g == q].mean() for q in qs])), len(qs))
        tr = within(y, drive)[0] if nm != "ease" else rho(np.array([ease[q] for q in qs]), np.array([drive[g == q].mean() for q in qs]))
        lines.append(f"{nm:8} {L:5d}  {cl:26}  {u:+.2f} ({nq} questions)                          {tr:+.2f}")
    lines.append("")
lines.append("ease 'use' is across questions (mean drive per question), the others inside each question.")
open(out + ".txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
print("LIGAND_DONE")
