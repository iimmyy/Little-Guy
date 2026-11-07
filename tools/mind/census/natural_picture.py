#!/usr/bin/env python3
"""The separate reference (Immy 2026-11-03: the expected value must come from another source than the writer). natural_alarm.py: the edit-fit alarm hears
planted numbers (52% over the line) and none of his real mistakes (0 of 18). Same material, same reads, now the picture: the swap-trained scene error readout (fit test,
index 14, untouched stream) and the formula's line in c5s. Does either hear a real mistake better than a right answer? Split by set: walls are mostly fair arithmetic with
a wrong reading of the story (6.4 students), herrings a wrong use of a number.
Below, the original note of natural_alarm.py:
The tickle test (Immy 2026-11-03): you cannot tickle yourself because you expect it. A planted number surprises him; his own wrong
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

mdir, plain, c5s, fit_f, lines_f, v2_f, out, B = sys.argv[1:9]
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
RUNS = os.environ.get("PICT_RUNS", "c5-think,c5-bare-think").split(",")          # PICT_RUNS=t8du2-bare-think,t8-bare-think: the mistakes of other builds
for f in sorted(x for r_ in RUNS for x in glob.glob(os.path.join(B, f"brake-*-{r_}.jsonl"))):
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
from natural_error import fit_pred
GN, KL, IDX = 1076, 17, 14
fz = np.load(fit_f, allow_pickle=True)
XF, A_, B_, C_ = fz["X"], fz["A"].astype(float), fz["b"].astype(float), fz["c"].astype(float)
n = len(A_)
err = np.abs(np.log(np.stack([A_, B_, C_], 1) / A_[:, None])).reshape(-1)
Xf = XF[:, :, IDX].reshape(n * 3, -1).astype(np.float64)


def capture(path, want):
    p = _lfm3("cuda", path)
    tok, m = p.tok, p.model
    got_ = {}
    if want == "stream":
        m.model.layers[IDX - 1].register_forward_hook(lambda mod, a, o: got_.__setitem__("x", (o[0] if isinstance(o, tuple) else o)[0, -1].float().cpu().numpy()))
    else:
        m.model.layers[KL].feed_forward.register_forward_pre_hook(lambda mod, a: got_.__setitem__("x", a[0][0, -1].float().cpu().numpy()))
    out_ = []
    for x in wrong + right:
        t, nl, r_, bare, f, v = x
        head_ = tok.apply_chat_template([{"role": "user", "content": t["q"] + ("" if bare else TAIL)}], tokenize=False, add_generation_prompt=True, enable_thinking=True) + "\n"
        with torch.no_grad():
            m.model(input_ids=torch.tensor([tok(head_ + t["thinking"][:nl], add_special_tokens=False).input_ids], device="cuda"))
        out_.append(got_["x"])
    row = m.model.layers[KL].feed_forward.w1.weight[GN].float().cpu().numpy().copy()
    del m
    torch.cuda.empty_cache()
    return np.stack(out_), row


Xs, _ = capture(plain, "stream")
pic = fit_pred(Xf, err, Xs.astype(np.float64), 1e3)
Xk, row_s = capture(c5s, "key")
_, row_0 = capture(mdir, "key")
meta = json.load(open(os.path.join(c5s, "pam.json")))
line = (row_s - row_0) / (meta["compressor"]["steep"] * meta["picture_fit"]["gamma"])
fitv = -(Xk @ line)                                     # high = off, the same way as the picture
nw = len(wrong)
sets = np.array(["walls" if "walls" in x[4] else "herring" if "herring" in x[4] else "tricks" for x in wrong + right])
isw = np.arange(len(pic)) < nw
L = [f"real mistakes on c5, brake sets, thinking on: {nw} wrong, {len(right)} right. High = does not fit."]
for nm, s_ in (("picture readout (index 14, 0 = fits; story right 0.19, swaps 0.66)", pic), ("formula line in c5s, sign flipped", fitv)):
    L.append(f"  {nm}: real mistakes median {np.median(s_[isw]):+.3f}, right {np.median(s_[~isw]):+.3f}, AUROC {auroc(s_[isw], s_[~isw]):.3f}")
    for st in ("walls", "herring"):
        m_ = sets == st
        if (isw & m_).any() and (~isw & m_).any():
            L.append(f"    {st:8s} wrong {int((isw & m_).sum()):2d}, right {int((~isw & m_).sum()):3d}: median {np.median(s_[isw & m_]):+.3f} vs {np.median(s_[~isw & m_]):+.3f}, AUROC {auroc(s_[isw & m_], s_[~isw & m_]):.3f}")
for k in np.argsort(-pic[:nw]):
    t, nl, _, bare, f, v = wrong[k]
    L.append(f"    picture {pic[k]:+.3f} formula {fitv[k]:+.3f} {f.replace('brake-', '').replace('.jsonl', ''):22s} {t['id']:12s} key {t['answer']} | {t['thinking'][:nl].rsplit(chr(10), 1)[-1][-60:]}")
open(out, "w").write("\n".join(L) + "\n")
print("\n".join(L))
