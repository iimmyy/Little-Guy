#!/usr/bin/env python3
"""The tickle test (Immy 2026-11-03): you cannot tickle yourself because you expect it. A planted number surprises him; his own wrong
number does not, because he predicted it. So: does the edit-fit alarm (edit_read.py v2, abs line, layer 17 input of c5) hear his REAL
mistakes? Material: brake-set runs on c5 with thinking on, finished, wrong final answer he wrote himself, minus the items read by hand
as a wrong key or ambiguous. Read at the last line of his thinking that states that number, same path as edit_read.py (cut text, own
forward, last word). Compared with the same read in his right runs (last line stating the right answer) and the ordinary-line levels.
Every wrong is printed with its score for reading. usage: natural_alarm.py <c5 folder> <lines.npz> <edit v2.npz> <out.txt> <brake folder>
"""
import glob, json, os, re, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from patients import _lfm3
from natural_error import auroc

mdir, lines_f, v2_f, out, B = sys.argv[1:6]
EXCLUDE = {"nluds-0778", "nluds-1542", "nluds-1608", "nluds-1610", "nluds-0808", "nluds-1606", "nluds-2265", "nluds-2269"}   # hand read: key wrong or ambiguous
TAIL = "\n\nEnd with a line: Answer: <your answer>"
ln, v2 = np.load(lines_f), np.load(v2_f)
score = lambda x: float(((x.astype(np.float64) - ln["mu"]) / ln["sd"]) @ ln["abs"])
so = ((v2["Xo"].astype(np.float64) - ln["mu"]) / ln["sd"]) @ ln["abs"]
sc0 = ((v2["Xc"].astype(np.float64) - ln["mu"]) / ln["sd"]) @ ln["abs"]
se = ((v2["Xe"].astype(np.float64) - ln["mu"]) / ln["sd"]) @ ln["abs"]
th2 = float(np.quantile(so, 0.98))
NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def final(reply):
    k = reply.rfind("\\boxed{")
    if k >= 0:
        i, d = k + 7, 1
        while i < len(reply) and d:
            d += {"{": 1, "}": -1}.get(reply[i], 0)
            i += 1
        return reply[k + 7:i - 1].strip()
    ls = [l for l in reply.splitlines() if "answer" in l.lower() and NUM.search(l)]
    return ls[-1].split(":", 1)[-1].strip() if ls else None


def value(s):
    if s is None:
        return None
    s = s.replace("{,}", "").replace("\\,", "").replace("$", "").replace(",", "").replace("*", "")
    fr = re.search(r"\\d?frac\{(\d+)\}\{(\d+)\}", s) or re.search(r"(?<![\d.])(\d+)\s*/\s*(\d+)", s)
    if fr and int(fr.group(2)):
        return int(fr.group(1)) / int(fr.group(2)), [f"{fr.group(1)}/{fr.group(2)}", f"\\frac{{{fr.group(1)}}}{{{fr.group(2)}}}"]
    m = NUM.search(s)
    return (float(m.group()), [m.group()]) if m else None


items = []
for f in sorted(glob.glob(os.path.join(B, "brake-*-c5-think.jsonl")) + glob.glob(os.path.join(B, "brake-*-c5-bare-think.jsonl"))):
    bare = "-bare-" in f
    for t in map(json.loads, open(f)):
        if t["finish"] != "stop" or not t["thinking"] or t["id"] in EXCLUDE:
            continue
        got = value(final(t["reply"]))
        if got is None:
            continue
        v, forms = got
        try:
            A = float(str(t["answer"]).replace(",", ""))
        except ValueError:
            continue
        right = abs(v - A) < 1e-6
        ends = []
        for fm in forms:
            for mm in re.finditer(rf"(?<![\d.]){re.escape(fm)}(?![\d.])", t["thinking"]):
                nl = t["thinking"].find("\n", mm.end())
                ends.append(nl if nl > 0 else len(t["thinking"]))
        if ends:
            items.append((t, max(ends), right, bare, os.path.basename(f), v))
wrong = [x for x in items if not x[2]]
right = [x for x in items if x[2]]
rng = np.random.RandomState(0)
right = [right[i] for i in rng.permutation(len(right))[:150]]
p = _lfm3("cuda", mdir)
tok, m = p.tok, p.model
got_ = {}
m.model.layers[17].feed_forward.register_forward_pre_hook(lambda mod, a: got_.__setitem__("x", a[0][0].float()))


def read(x):
    t, nl, r_, bare, f, v = x
    head = tok.apply_chat_template([{"role": "user", "content": t["q"] + ("" if bare else TAIL)}], tokenize=False, add_generation_prompt=True, enable_thinking=True) + "\n"
    with torch.no_grad():
        m.model(input_ids=torch.tensor([tok(head + t["thinking"][:nl], add_special_tokens=False).input_ids], device="cuda"))
    return score(got_["x"][-1].cpu().numpy())


sw = np.array([read(x) for x in wrong])
sr = np.array([read(x) for x in right])
L = [f"his own finished answers on c5, brake sets, thinking on: {len(wrong)} wrong, {len(right)} right (sample). Alarm = edit-fit abs line. Reference: ordinary lines median {np.median(so):+.3f}, clean number lines {np.median(sc0):+.3f}, planted edits {np.median(se):+.3f}; 2% alarm line {th2:+.3f}",
     f"  real mistakes median {np.median(sw):+.3f}, over the alarm line {np.mean(sw > th2):.0%}; right answers median {np.median(sr):+.3f}, over the line {np.mean(sr > th2):.0%}; AUROC real mistake vs right {auroc(sw, sr):.3f}",
     f"  for comparison, planted edits over the line {np.mean(se > th2):.0%}, clean number lines {np.mean(sc0 > th2):.0%}"]
for x, s_ in sorted(zip(wrong, sw), key=lambda z: -z[1]):
    t, nl, _, bare, f, v = x
    L.append(f"    {s_:+.3f} {f.replace('brake-', '').replace('.jsonl', ''):22s} {t['id']:14s} key {t['answer']} said {v:g} | {t['thinking'][:nl].rsplit(chr(10), 1)[-1][-70:]}")
open(out, "w").write("\n".join(L) + "\n")
print("\n".join(L))
