#!/usr/bin/env python3
"""Does his stream hold a wall? (Immy 2026-11-03 formula: E = sum(springs^2) + sum(walls crossed^2). Walls were never built; this is the first test of one.)
Wall tested: you cannot have part of a countable thing. Same thinking text, cut after the sentence stating the answer, the number swapped:
  a  his right answer (whole)
  b  a wrong whole number (a wrong number, no wall crossed)
  w  a non-whole near the right one (wall crossed when the thing counted is countable)
Two groups of problems: "How many <countable>?" and "How many <divisible unit>?" (hours, miles, dollars, cups...), where a non-whole crosses no wall. The second group is the control:
a line that only sees a decimal point reads both the same; a real wall reads the first group only.
Line built on 2/3 of the problems at the sentence end, at the given layers. Nothing changes the model.
usage: wall_read.py <hf folder> <problem runs.jsonl> <out.npz> <out.txt> [layers=9,14,17 n=500 dev=cuda tail=1]
"""
import hashlib, json, os, re, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from patients import _lfm3
from natural_error import auroc
from situation_probe import TAIL

args = [a for a in sys.argv[1:] if "=" not in a]
opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
mdir, runs, outz, outt = args[:4]
LAY = [int(x) for x in opt.get("layers", "9,14,17").split(",")]
N, DEV = int(opt.get("n", 500)), opt.get("dev", "cuda")
if opt.get("tail") == "0":
    TAIL = ""
UNITS = {"hours", "minutes", "seconds", "days", "weeks", "months", "years", "miles", "kilometers", "km", "meters", "feet", "inches", "yards", "centimeters", "cm", "dollars", "cents", "cups", "liters", "litres",
         "gallons", "ounces", "pounds", "kilograms", "kg", "grams", "tons", "acres", "percent", "degrees", "more", "fewer", "less", "times", "square", "cubic", "ml", "quarts", "pints", "mph", "lbs", "oz"}
ASK = re.compile(r"[Hh]ow many ([a-zA-Z-]+)")


def group(q):
    m = ASK.search(q)
    if not m:
        return None
    return "unit" if m.group(1).lower() in UNITS else "count"


def fork_of(think, A):
    pat = re.compile(rf"(?:=|\bis\b|\bare\b|\bget\b|\bgives\b|\bequals\b|\btotal of\b)\s*\$?\\?\(?\s*({A})(?![\d.,/])")
    ms = list(pat.finditer(think))
    if not ms:
        return None
    m = ms[0]                                                # the first time he states it: a working line, not a wrap-up
    end = re.search(r"[.!?\n]", think[m.end(1):])
    if not end:
        return None
    return m.span(1), m.end(1) + end.end()


p_ = _lfm3(DEV, mdir)
tok, m = p_.tok, p_.model
got = {}
for L_ in LAY:
    m.model.layers[L_].feed_forward.register_forward_pre_hook(lambda mod, a, L_=L_: got.__setitem__(L_, a[0][0, -1].float().cpu().numpy()))
items = []
for t in map(json.loads, open(runs)):
    g = group(t["q"])
    s = str(t["answer"]).strip()
    if g is None or t["finish"] != "stop" or not re.fullmatch(r"\d+", s) or not (3 <= int(s) <= 100000):
        continue
    fk = fork_of(t["thinking"], s)
    if fk:
        items.append((t, g, int(s), fk))
L = [f"{len(items)} problems usable: count {sum(i[1] == 'count' for i in items)}, unit {sum(i[1] == 'unit' for i in items)}"]
cnt = [i for i in items if i[1] == "count"][:N]
uni = [i for i in items if i[1] == "unit"][:N]
X = {L_: [] for L_ in LAY}
GRP, VER, PID = [], [], []
for k, (t, g, A, ((a, b), cut)) in enumerate(cnt + uni):
    head = tok.apply_chat_template([{"role": "user", "content": t["q"] + TAIL}], tokenize=False, add_generation_prompt=True, enable_thinking=True) + "\n"
    th = t["thinking"]
    wrong = A + max(1, round(A * 0.25))
    for vi, v in enumerate((str(A), str(wrong), f"{A}.{4 + hash(t['id']) % 5}")):
        ids = tok(head + th[:a] + v + th[b:cut], add_special_tokens=False).input_ids
        with torch.no_grad():
            m.model(input_ids=torch.tensor([ids], device=DEV))
        for L_ in LAY:
            X[L_].append(got[L_])
        GRP.append(g); VER.append(vi); PID.append(t["id"])
    if k % 100 == 0:
        print(f"{k}/{len(cnt) + len(uni)}", flush=True)
GRP, VER, PID = np.array(GRP), np.array(VER), np.array(PID)
np.savez(outz, **{f"x{L_}": np.stack(X[L_]) for L_ in LAY}, grp=GRP, ver=VER, pid=PID)
bd = np.array([int(hashlib.md5(i.encode()).hexdigest(), 16) % 3 != 2 for i in PID])


def fit(Xb, y, al):
    mu, sd = Xb.mean(0), Xb.std(0) + 1e-6
    Z = (Xb - mu) / sd
    c = np.linalg.solve(Z.T @ Z + al * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
    return lambda Q: ((Q - mu) / sd) @ c


for L_ in LAY:
    Xa = np.stack(X[L_]).astype(np.float64)
    # the line: built on count problems only, wall crossed vs not. Then the control group reads what it has never seen.
    b = bd & (GRP == "count")
    y = (VER[b] == 2).astype(float)
    for al in (1e3, 1e4):
        f = fit(Xa[b], y, al)
        sc = f(Xa)
        hc = ~bd & (GRP == "count")
        hu = ~bd & (GRP == "unit")
        rows = []
        for nm, s in (("count", hc), ("unit", hu)):
            if s.any():
                rows.append(f"{nm}: non-whole vs right {auroc(sc[s & (VER == 2)], sc[s & (VER == 0)]):.3f}, non-whole vs wrong whole {auroc(sc[s & (VER == 2)], sc[s & (VER == 1)]):.3f}, wrong whole vs right {auroc(sc[s & (VER == 1)], sc[s & (VER == 0)]):.3f}, median non-whole {np.median(sc[s & (VER == 2)]):+.2f}")
        crossed = hc & (VER == 2)
        free = hu & (VER == 2)
        spec = auroc(sc[crossed], sc[free]) if crossed.any() and free.any() else float("nan")
        L.append(f"layer {L_:2d} alpha {al:.0e}: " + "; ".join(rows) + f"; THE WALL: non-whole on countable vs non-whole on a divisible unit {spec:.3f}")
open(outt, "w").write("\n".join(L) + "\n")
print("\n".join(L))
