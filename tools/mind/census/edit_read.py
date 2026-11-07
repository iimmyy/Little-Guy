#!/usr/bin/env python3
"""A sharper "does not equal the expected value" (Immy 2026-11-03). alarm_quality.py: the formula's line in c5s falls on 93% of
planted edits, yet a fixed threshold catches only 20% at 3.4 false alarms per 100 lines: the fall is real, but lines differ
from one another by more than an edit moves them. So: fit a new line at c5's key input on the same kind of material it has
to work on, long bare thinking, same line clean and edited, so what differs between lines stops counting. Two forms, side by side:
- abs: a line fit to |log(said / true)|, the form c5s uses. A straight line cannot draw an absolute value.
- spring: a line fit to the signed log(said / true), alarm on its square, as Immy's scene energy E = sum(springs^2) is written. Like
  a null test: said minus expected cancels on a right line whatever the line's level, and what is left is the mistake.
Both 0 on clean lines and ordinary line ends. Nothing here changes the model.
Edits: any stated whole number ("= 12", "is 12") within 60 characters of its line end, up to `per` per run, across depths.
Built on 2/3 of the problems, reported on the other 1/3 (train split only, never the night test).
usage: edit_read.py <model folder> <out.npz> <out.txt> <runs.jsonl> [...] [per=5 ord=12 layer=17 dev=cuda]
"""
import hashlib, json, os, random, re, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from patients import _lfm3
from natural_error import auroc
from planted_edit import nearby_wrong

args = [a for a in sys.argv[1:] if "=" not in a]
opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
mdir, outz, outt, runs = args[0], args[1], args[2], args[3:]
PER, ORD, KL, DEV = int(opt.get("per", 5)), int(opt.get("ord", 12)), int(opt.get("layer", 17)), opt.get("dev", "cuda")
STATED = re.compile(r"(?:=|\bis\b|\bare\b|\bgives\b|\bequals\b|\bget\b)\s*\$?\\?\(?\s*(\d{1,6})(?![\d.,/])")
p = _lfm3(DEV, mdir)
tok, m = p.tok, p.model
got = {}
m.model.layers[KL].feed_forward.register_forward_pre_hook(lambda mod, a: got.__setitem__("x", a[0][0].float()))


def read(ids, pos):
    with torch.no_grad():
        m.model(input_ids=torch.tensor([ids], device=DEV))
    return got["x"][pos].cpu().numpy().astype(np.float32)


rng = random.Random(0)
Xc, Xe, Xo, ERR, DEPTH, SHOWN, PE, PO, IDS = [], [], [], [], [], [], [], [], []
nrun = 0
for f in runs:
    for t in map(json.loads, open(f)):
        th = t["thinking"]
        if not th or len(th.split()) < 40:
            continue
        cand = []
        for mm in STATED.finditer(th):
            a, b = mm.span(1)
            v = int(mm.group(1))
            nl = th.find("\n", b)
            nl = nl if nl > 0 else len(th)
            after = th[b:b + 4].lstrip(" $\\")
            if v < 2 or nl - b > 60 or (after and re.match(r"[a-zA-Z^*+\-(\[{_]", after)):
                continue
            cand.append((a, b, nl, v))
        if not cand:
            continue
        rng.shuffle(cand)
        chosen, used = [], set()
        for c_ in cand:
            if c_[2] not in used:
                used.add(c_[2]); chosen.append(c_)
            if len(chosen) == PER:
                break
        head = tok.apply_chat_template([{"role": "user", "content": t["q"]}], tokenize=False, add_generation_prompt=True, enable_thinking=True) + "\n"
        # every read goes the same way: the text cut at the line end, own tokenizer pass, own forward, last word. (v1 read clean and
        # ordinary lines out of one full forward and edits out of a cut one, so a line could tell the two paths apart: AUROC 0.999.)
        line_ends = [mm.start() for mm in re.finditer(r"\n", th) if mm.start() > 0]
        for nl in rng.sample(line_ends, min(ORD, len(line_ends))):
            Xo.append(read(tok(head + th[:nl], add_special_tokens=False).input_ids, -1)); PO.append(t["id"])
        for a, b, nl, v in chosen:
            w = nearby_wrong(v, random.Random(f"{t['id']}|{a}"))
            Xc.append(read(tok(head + th[:nl], add_special_tokens=False).input_ids, -1))
            Xe.append(read(tok(head + th[:a] + str(w) + th[b:nl], add_special_tokens=False).input_ids, -1))
            ERR.append(float(np.log(max(w, 1) / v))); DEPTH.append(len(th[:a].split())); PE.append(t["id"])
            SHOWN.append(bool(re.search(r"[\d)]\s*[-+*/\u00d7\\]", th[:a].rsplit("\n", 1)[-1])))
        nrun += 1
        if nrun % 20 == 0:
            print(f"{nrun} runs, {len(Xe)} edits", flush=True)
Xc, Xe, Xo = np.stack(Xc), np.stack(Xe), np.stack(Xo)
ERR, DEPTH, SHOWN, PE, PO = np.array(ERR), np.array(DEPTH), np.array(SHOWN), np.array(PE), np.array(PO)
np.savez(outz, Xc=Xc, Xe=Xe, Xo=Xo, err=ERR, depth=DEPTH, shown=SHOWN, pe=PE, po=PO)
part = lambda ids_: np.array([int(hashlib.md5(i.encode()).hexdigest(), 16) % 3 != 2 for i in ids_])
be, bo = part(PE), part(PO)
L = [f"edit read at layer {KL} input on {os.path.basename(mdir.rstrip('/'))}: {nrun} runs, {len(Xe)} edits, {len(Xo)} ordinary lines; build {be.sum()} edits, held out {(~be).sum()} edits, {(~bo).sum()} ordinary lines"]
Xa = np.concatenate([Xe[be], Xc[be], Xo[bo]]).astype(np.float64)
wt = np.concatenate([np.ones(be.sum()), np.ones(be.sum()), np.full(bo.sum(), be.sum() / max(bo.sum(), 1))])
mu, sd = Xa.mean(0), Xa.std(0) + 1e-6
Z = (Xa - mu) / sd
ps = np.concatenate([PE[be], PE[be], PO[bo]])
val = np.array([int(hashlib.md5(i.encode()).hexdigest(), 16) % 4 == 0 for i in ps])
nv = be.sum()
isedit = np.concatenate([np.ones(nv, bool), np.zeros(len(ps) - nv, bool)])


def fit(Z_, y_, w_, al):
    sw = np.sqrt(w_)[:, None]
    A_ = Z_ * sw
    return np.linalg.solve(A_.T @ A_ + al * np.eye(Z_.shape[1]), A_.T @ (y_ * sw[:, 0]))


saved = {}
for form, target, alarm in (("abs", np.abs, lambda r: r), ("spring", lambda e: e, lambda r: r ** 2)):
    y = np.concatenate([target(ERR[be]), np.zeros(be.sum()), np.zeros(bo.sum())])
    best = None
    for al in (1e2, 1e3, 1e4, 1e5, 1e6):
        c = fit(Z[~val], y[~val], wt[~val], al)
        sc = alarm(Z[val] @ c)
        a_ = auroc(sc[isedit[val]], sc[~isedit[val]])
        L.append(f"  {form}: alpha {al:.0e}: validation AUROC {a_:.3f}")
        if best is None or a_ > best[0]:
            best = (a_, al)
    c = fit(Z, y, wt, best[1])
    sc_ = lambda X: alarm(((X.astype(np.float64) - mu) / sd) @ c)
    se, sc0, so = sc_(Xe[~be]), sc_(Xc[~be]), sc_(Xo[~bo])
    d, sh = DEPTH[~be], SHOWN[~be]
    L.append(f"{form}, held out (alpha {best[1]:.0e}): AUROC edited vs clean {auroc(se, sc0):.3f}; edited vs ordinary lines {auroc(se, so):.3f}; edited higher in {np.mean(se > sc0):.0%} of pairs")
    for lab, s_ in (("deep 300+", d >= 300), ("shallow <300", d < 300), ("shown on the line", sh), ("not shown", ~sh)):
        if s_.any():
            L.append(f"  {lab:18s} n {int(s_.sum()):4d}  AUROC vs clean {auroc(se[s_], sc0[s_]):.3f}, higher in {np.mean(se[s_] > sc0[s_]):.0%} of pairs")
    for fa in (0.01, 0.02, 0.05, 0.1):
        th_ = np.quantile(so, 1 - fa)
        L.append(f"  alarm over the {fa:.0%} line of ordinary lines: catches {np.mean(se > th_):.0%} of edits (deep {np.mean(se[d >= 300] > th_):.0%}, not shown {np.mean(se[~sh] > th_):.0%}), fires on {np.mean(sc0 > th_):.0%} of the clean copies")
    saved[form] = c
np.savez(outz.replace(".npz", "-lines.npz"), abs=saved["abs"], spring=saved["spring"], mu=mu, sd=sd)
open(outt, "w").write("\n".join(L) + "\n")
print("\n".join(L))
