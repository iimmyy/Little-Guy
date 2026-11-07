#!/usr/bin/env python3
"""Material for Immy's finished scene formula (2026-11-04, "finish the formula fully"). E = sum(springs^2) + sum(walls crossed^2), judgment = precision x error^2.
From his finished right runs on the bare question: the same thinking, cut at the sentence that first states the answer, the number swapped:
  a  right answer                          (nothing broken)
  b  wrong whole, A*r                      (a spring stretched)
  c  wrong whole, A/r                      (a spring stretched, the other way)
  d  an earlier number from his own work   (the question's demand: he answered the wrong thing)
  w  non-whole near A                      (the whole-number wall, crossed only when the thing asked for is countable)
  n  negative of A                         (the sign wall, crossed for counts)
Plus sentence ends with no number in the line, and reply words. Saved at the MLP input of each layer given. The same swaps are also read at the next paragraph end, for cells that act there
(stop, clock): a broken scene should not hand in. Nothing changes the model.
usage: scene_mat.py <hf folder> <runs.jsonl> <out.npz> [layers=16,17,23 dev=cuda skip=eval.jsonl]
"""
import json, os, random, re, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from patients import _lfm3

args = [a for a in sys.argv[1:] if "=" not in a]
opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
mdir, runs, outz = args[:3]
LAY = [int(x) for x in opt.get("layers", "16,17,23").split(",")]
DEV = opt.get("dev", "cuda")
skip = {json.loads(l)["id"] for l in open(opt["skip"])} if "skip" in opt else set()
UNITS = {"hours", "minutes", "seconds", "days", "weeks", "months", "years", "miles", "kilometers", "km", "meters", "feet", "inches", "yards", "centimeters", "cm", "dollars", "cents", "cups", "liters", "litres",
         "gallons", "ounces", "pounds", "kilograms", "kg", "grams", "tons", "acres", "percent", "degrees", "more", "fewer", "less", "times", "square", "cubic", "ml", "quarts", "pints", "mph", "lbs", "oz"}
ASK = re.compile(r"[Hh]ow many ([a-zA-Z-]+)")
NUMS = re.compile(r"(?<![\d.,/^])\b(\d{1,6})\b(?![\d.,/^])")
rng = random.Random(0)


def group(q):
    m_ = ASK.search(q)
    return ("unit" if m_.group(1).lower() in UNITS else "count") if m_ else "other"


def fork_of(th, A):
    ms = list(re.finditer(rf"(?:=|\bis\b|\bare\b|\bget\b|\bgives\b|\bequals\b|\btotal of\b)\s*\$?\\?\(?\s*({A})(?![\d.,/])", th))
    if not ms:
        return None
    m_ = ms[0]
    e = re.search(r"[.!?\n]", th[m_.end(1):])
    return (m_.span(1), m_.end(1) + e.end()) if e else None


p_ = _lfm3(DEV, mdir)
tok, m = p_.tok, p_.model
got = {}
for L_ in LAY:
    m.model.layers[L_].feed_forward.register_forward_pre_hook(lambda mod, a, L_=L_: got.__setitem__(L_, a[0][0].float()))


def read(ids, pos=-1):
    with torch.no_grad():
        m.model(input_ids=torch.tensor([ids], device=DEV))
    return {L_: got[L_][pos].cpu().numpy().astype(np.float32) for L_ in LAY}


X = {L_: [] for L_ in LAY}
VER, GRP, PID, AT, ERR = [], [], [], [], []       # AT: 0 at the sentence end, 1 at the next paragraph end, 2 plain sentence end, 3 reply word
VNAMES = "abcdwn"


def put(r_, ver, g, pid, at, err):
    for L_ in LAY:
        X[L_].append(r_[L_])
    VER.append(ver); GRP.append(g); PID.append(pid); AT.append(at); ERR.append(err)


n_ok = 0
for t in map(json.loads, open(runs)):
    if t["id"] in skip or t["finish"] != "stop" or not t["thinking"]:
        continue
    s = str(t["answer"]).strip().replace(",", "")
    if not re.fullmatch(r"\d+", s) or not (3 <= int(s) <= 100000):
        continue
    A = int(s)
    th, q, g = t["thinking"], t["q"], group(t["q"])
    fk = fork_of(th, s)
    if fk is None:
        continue
    (a, b), cut = fk
    head = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=True) + "\n"
    r_ = rng.choice((1.25, 1.5, 2, 3))
    bv = max(round(A * r_), A + 1)
    cv = round(A / r_)
    cv = cv if cv not in (A, 0) else max(A - 1, 1)
    earlier = [int(x) for x in NUMS.findall(th[:a]) if int(x) not in (A, 0, 1) and abs(np.log(int(x) / A)) > 0.1]
    dv = earlier[-1] if earlier else None
    vers = [("a", str(A), 0.0), ("b", str(bv), abs(np.log(bv / A))), ("c", str(cv), abs(np.log(cv / A))), ("w", f"{A}.{rng.choice((25, 4, 5, 75))}", 0.0), ("n", f"-{A}", 0.0)]
    if dv:
        vers.insert(3, ("d", str(dv), abs(np.log(dv / A))))
    nxt = th.find("\n\n", cut - 1)
    for nm, v, er in vers:
        put(read(tok(head + th[:a] + v + th[b:cut], add_special_tokens=False).input_ids), VNAMES.index(nm), g, t["id"], 0, er)
        if nxt > 0 and nm in "abdw":
            put(read(tok(head + th[:a] + v + th[b:nxt + 2], add_special_tokens=False).input_ids), VNAMES.index(nm), g, t["id"], 1, er)
    # ordinary sentence ends (no stated answer on the line) and reply words, from one forward over the whole text
    rep = "\n</think>\n\n" + t["reply"] if t.get("reply") else ""
    full = head + th + rep
    enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
    ids, off = enc.input_ids, enc.offset_mapping
    h0, h1 = len(head), len(head) + len(th)
    ends = [i for i in range(1, len(ids) - 1) if h0 <= off[i][0] < h1 and full[off[i][0]:off[i][1]].strip() in (".", ".\n", ".\n\n", "\n") and s not in full[max(h0, off[i][0] - 60):off[i][1]]]
    reps = [i for i in range(1, len(ids) - 1) if off[i][0] >= h1]
    rng.shuffle(ends); rng.shuffle(reps)
    chosen = [(i, 2) for i in ends[:4]] + [(i, 3) for i in reps[:4]]
    if chosen:
        with torch.no_grad():
            m.model(input_ids=torch.tensor([ids], device=DEV))
        for i, at in chosen:
            for L_ in LAY:
                X[L_].append(got[L_][i].cpu().numpy().astype(np.float32))
            VER.append(-1); GRP.append(g); PID.append(t["id"]); AT.append(at); ERR.append(0.0)
    n_ok += 1
    if n_ok % 50 == 0:
        print(f"{n_ok} problems, {len(VER)} points", flush=True)
np.savez(outz, **{f"x{L_}": np.stack(X[L_]) for L_ in LAY}, ver=np.array(VER), grp=np.array(GRP), pid=np.array(PID), at=np.array(AT), err=np.array(ERR))
print("MAT_DONE", n_ok, len(VER))
