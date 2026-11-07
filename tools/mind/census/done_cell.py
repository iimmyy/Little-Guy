#!/usr/bin/env python3
"""The hand-in cell (Immy 2026-11-03: "I'm done"; more gain, nothing broken). Bolt-on, no training, no new neurons, same file shape.
In his finished runs a paragraph after the answer opens the commit sentence ("Thus final ...") about 26% of the time; in the runs that
loop until they are cut off about 2-7%, and before the answer about 0.2%. The switch at the end of that sentence always works. So two quiet
neurons in the last layer get rewired, each reading the same kind of signal (the answer has been on the page for a while) at one word
of the sentence: one at paragraph ends leans toward "Thus", one at the opener word leans toward " final"/" answer". The neuron's gate grows with
the reading (silu of a line), like urgency against a bound (Cisek): a lean only, so a word he has no use for gets nothing from it.
Lines are fit on done_cap.py's points (2/3 of the build problems), lean strength set on those, and everything reported on the other 1/3.
mat=<scene_mat.npz>: Immy's verdict: "breaks" must not hand in. The same swaps read at the next paragraph end (a wrong number, an earlier number, a crossed wall) go in at -1, so the cell leans only on a scene that holds.
usage: done_cell.py <base folder> <cap.npz> <out folder> <runs.jsonl> [...] [layer=23 T=80 fa=0.05 liftp=1.0 lifto=1.4 n=120 dev=cuda skip=eval.jsonl mat=]
"""
import hashlib, json, os, random, re, shutil, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from patients import _lfm3

args = [a for a in sys.argv[1:] if "=" not in a]
opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
base, capf, out, runs = args[0], args[1], args[2], args[3:]
LB, T, FA = int(opt.get("layer", 23)), int(opt.get("T", 80)), float(opt.get("fa", 0.05))
GAIN = float(opt.get("gain", 4))
LIFTP, LIFTO, N, DEV = float(opt.get("liftp", 1.0)), float(opt.get("lifto", 1.4)), int(opt.get("n", 120)), opt.get("dev", "cuda")
skip = {json.loads(l)["id"] for l in open(opt["skip"])} if "skip" in opt else set()
OPENERS = ("Thus", "So", "Therefore", "Hence")
isb = lambda g: int(hashlib.md5(str(g).encode()).hexdigest(), 16) % 3 != 2


def ridge0(X, y, w, groups=None, alphas=(1e3, 1e4, 1e5, 1e6)):
    """no offset: the line has to live on his own carriers. Strength picked on whole problems left out, not on words"""
    sc = X.std(0) + 1e-6
    Z = X / sc * np.sqrt(w)[:, None]
    Y = y * np.sqrt(w)
    f = np.arange(len(y)) % 4 if groups is None else np.array([int(hashlib.md5(str(g).encode()).hexdigest(), 16) % 4 for g in groups])
    best = None
    for al in alphas:
        err = sum(((Z[f == i] @ np.linalg.solve(Z[f != i].T @ Z[f != i] + al * np.eye(Z.shape[1]), Z[f != i].T @ Y[f != i]) - Y[f == i]) ** 2).sum() for i in range(4))
        if best is None or err < best[0]:
            best = (err, al)
    return np.linalg.solve(Z.T @ Z + best[1] * np.eye(Z.shape[1]), Z.T @ Y) / sc


z = np.load(capf, allow_pickle=True)
kind, since, gid = z["kind"], z["since"], z["gid"]
Xa = z[f"x{LB}"].astype(np.float64)
bd = np.array([isb(g) for g in gid])
if "mat" in opt:                                        # broken scenes at the next paragraph end join as paragraph-end points that must stay silent
    zm = np.load(opt["mat"], allow_pickle=True)
    brk = (zm["at"] == 1) & np.isin(zm["ver"], [1, 3, 4])
    Xa = np.concatenate([Xa, zm[f"x{LB}"][brk].astype(np.float64)])
    kind = np.concatenate([kind, np.full(brk.sum(), 0)])
    since = np.concatenate([since, np.full(brk.sum(), -1)])
    gid = np.concatenate([gid, zm["pid"][brk]])
    bd = np.array([isb(g) for g in gid])
L = [f"hand-in cell on {os.path.basename(base.rstrip('/'))}: layer {LB}, answer on the page {T}+ words, false alarm line {FA:.0%}"]
lines = {}
for k, nm in ((0, "paragraph end"), (1, "opener word")):
    s = bd
    # +1 at this kind of point once the answer has been up T words, -1 before it, and -1 at every other kind of word
    y = np.where((kind == k) & (since >= T), 1.0, -1.0)
    keep = s & ~((kind == k) & (since >= 0) & (since < T))
    pos_ = keep & (y > 0)
    w = np.where(y > 0, 0.5 / pos_.sum(), 0.5 / (keep & (y < 0)).sum()) * keep.sum()
    w = w * np.where(kind == 3, 3.0, 1.0)                             # the reply has to stay silent: it is where a stray word does the most harm
    line = ridge0(Xa[keep], y[keep], w[keep], gid[keep])
    r = Xa @ line
    before = bd & (kind == k) & (since < 0)
    th = float(np.quantile(r[before], 1 - FA))
    after = bd & (kind == k) & (since >= T)
    med = float(np.median(r[after]))
    # silu dips below zero under a gate of 0 (down to -0.28): a gate that only just goes negative still writes, the wrong way, at every
    # word. So the gate is steep: GAIN at his median point after the answer, far enough below zero everywhere else to be really silent.
    gain = GAIN / max(med - th, 1e-3)
    gate = ridge0(Xa[keep], (gain * (r - th))[keep], w[keep], gid[keep])   # offset carried by the stream too
    level = ridge0(Xa, np.ones(len(Xa)), np.ones(len(Xa)), gid)
    lines[k] = (gate, level)
    gi = Xa @ gate
    for part, sel in (("build", bd), ("held out", ~bd)):
        a, b, o, rp_ = sel & (kind == k) & (since >= T), sel & (kind == k) & (since < 0), sel & (kind == 2), sel & (kind == 3)
        L.append(f"  {nm}, {part}: gate input median after {np.median(gi[a]):+.2f}, before {np.median(gi[b]):+.2f}, ordinary {np.median(gi[o]):+.2f}, reply {np.median(gi[rp_]):+.2f}; over 0: after {np.mean(gi[a] > 0):.0%}, before {np.mean(gi[b] > 0):.0%}, ordinary {np.mean(gi[o] > 0):.1%}, reply {np.mean(gi[rp_] > 0):.1%}; level median {np.median(Xa[sel] @ level):.2f}")

p_ = _lfm3(DEV, base)
tok, m = p_.tok, p_.model
lay = m.model.layers[LB]
ff = lay.feed_forward
ids_of = lambda w: tok(w, add_special_tokens=False).input_ids[0]
THUS, FIN, ANS, END = ids_of("Thus"), ids_of(" final"), ids_of(" answer"), ids_of("</think>")
TARGET = {0: [THUS], 1: [FIN, ANS]}
# the quietest two neurons over the captured words
with torch.no_grad():
    Xt = torch.tensor(Xa, dtype=torch.float32, device=DEV)
    act = F.silu(Xt @ ff.w1.weight.float().T) * (Xt @ ff.w3.weight.float().T)
    ws = act.abs() * ff.w2.weight.float().norm(dim=0)[None]
    loud = ws.kthvalue(max(1, int(0.999 * ws.shape[0])), dim=0).values
    order = loud.argsort()
    NEU = {0: int(order[0]), 1: int(order[1])}
    del Xt, act, ws
L.append(f"  neurons: layer {LB} #{NEU[0]} (old 99.9% write {float(loud[NEU[0]]):.2e}), #{NEU[1]} ({float(loud[NEU[1]]):.2e}); median neuron {float(loud.median()):.2e}")


# ---- points from his runs (same reading as done_cap.py), for direction, strength and the report
def norm(s):
    for a, b in (("\\dfrac", "\\frac"), ("\\tfrac", "\\frac"), ("\\left", ""), ("\\right", ""), ("\\!", ""), ("\\,", ""), ("$", ""), (" ", ""), ("{,}", "")):
        s = s.replace(a, b)
    return s


def spot_char(th, ans):
    a = norm(str(ans))
    if not a or (len(a) < 2 and not a.isdigit()):
        return None
    forms = {a}
    m_ = re.fullmatch(r"(-?)\\frac\{(\d+)\}\{(\d+)\}", a)
    if m_:
        forms.add(f"{m_.group(1)}{m_.group(2)}/{m_.group(3)}")
    keep = [i for i, ch in enumerate(th) if norm(ch)]
    S = "".join(th[i] for i in keep)
    best = None
    for f in forms:
        for mm in re.finditer(re.escape(f), S):
            nxt, prv = S[mm.end():mm.end() + 1], S[mm.start() - 1:mm.start()] if mm.start() else ""
            if (f[-1].isdigit() and nxt.isdigit()) or (f[0].isdigit() and (prv.isdigit() or prv == ".")) or nxt in ("^", "+", "x", "y") or (nxt == "/" and "/" not in f):
                continue
            if any(c in S[max(0, mm.start() - 6):mm.start()] for c in ("=", "is", "get", "are", "gives", "equals", "≈", "boxed{", "answer")):
                e = keep[mm.end() - 1] + 1
                best = e if best is None else min(best, e)
                break
    return best


rng = random.Random(1)
PTS = []                # (ids, pos, kind, group, is_build, finished)
for f in runs:
    for t in map(json.loads, open(f)):
        th = t["thinking"]
        if t["id"] in skip or not th or len(th.split()) < 20:
            continue
        head = tok.apply_chat_template([{"role": "user", "content": t["q"]}], tokenize=False, add_generation_prompt=True, enable_thinking=True) + "\n"
        enc = tok(head + th, add_special_tokens=False, return_offsets_mapping=True)
        ids, off = enc.input_ids, enc.offset_mapping
        full, h0 = head + th, len(head)
        sp = spot_char(th, t["answer"])
        for i in rng.sample(range(h0 and 1, len(ids) - 1), min(800, len(ids) - 2)):
            a, b = off[i]
            if a < h0:
                continue
            txt = full[a:b]
            k = 0 if "\n\n" in txt else 1 if txt.strip() in OPENERS and "\n" in full[off[i - 1][0]:off[i - 1][1]] else 2
            e = b - h0
            since_ = -1 if sp is None or e < sp else len(th[sp:e].split())
            if k < 2 or rng.random() < 0.02:
                PTS.append((ids, i, k, since_, isb(t["id"]), t["finish"] == "stop"))
rng.shuffle(PTS)


def pick(k, grp, build, n):
    out_ = []
    for pt in PTS:
        if pt[2] != k or pt[4] != build:
            continue
        s = pt[3]
        if (grp == "after" and s >= T) or (grp == "before" and s < 0) or (grp == "cutafter" and s >= T and not pt[5]) or (grp == "finafter" and s >= T and pt[5]) or grp == "any":
            out_.append(pt)
        if len(out_) == n:
            break
    return out_


def logp(pt):
    ids, i = pt[0], pt[1]
    with torch.no_grad():
        hs = m.model(input_ids=torch.tensor([ids[:i + 1]], device=DEV)).last_hidden_state[0, -1]
        return torch.log_softmax(m.lm_head(hs).float(), -1)


def chance(pts, toks):
    return float(np.mean([float(logp(pt)[toks].exp().sum()) for pt in pts]))


# direction: what raises his target word's log-odds, measured at the MLP's output at this layer
def direction(k, n=40):
    tg = TARGET[k]
    tot = 0
    delta = {}

    def hook(mod, a, o):
        # the write goes in before the norm; the MLP returns (x, ...) or x
        x = o[0] if isinstance(o, tuple) else o
        d = torch.zeros_like(x[0, -1], requires_grad=True)
        delta["d"] = d
        x = x.clone()
        x[0, -1] = x[0, -1] + d
        return (x,) + tuple(o[1:]) if isinstance(o, tuple) else x

    h = ff.register_forward_hook(hook)
    for pt in pick(k, "after", True, n):
        hs = m.model(input_ids=torch.tensor([pt[0][: pt[1] + 1]], device=DEV)).last_hidden_state[0, -1]
        lp = torch.log_softmax(m.lm_head(hs).float(), -1)
        pt_ = torch.logsumexp(lp[tg], 0)
        odds = pt_ - torch.log1p(-pt_.exp().clamp(max=0.9999))
        g, = torch.autograd.grad(odds, delta["d"])
        tot = tot + g.detach() / (g.norm() + 1e-9)
    h.remove()
    return (tot / tot.norm()).float()


dirs = {k: direction(k) for k in (0, 1)}
sims = float(dirs[0] @ dirs[1])
L.append(f"  write directions found; between the two {sims:+.2f}")
orig = {k: (ff.w1.weight[NEU[k]].clone(), ff.w3.weight[NEU[k]].clone(), ff.w2.weight[:, NEU[k]].clone()) for k in (0, 1)}


def setn(k, scale):
    with torch.no_grad():
        ff.w1.weight[NEU[k]] = torch.tensor(lines[k][0], dtype=ff.w1.weight.dtype, device=DEV)
        ff.w3.weight[NEU[k]] = torch.tensor(lines[k][1], dtype=ff.w1.weight.dtype, device=DEV)
        ff.w2.weight[:, NEU[k]] = (scale * dirs[k]).to(ff.w2.weight.dtype)


def off_(k):
    with torch.no_grad():
        ff.w1.weight[NEU[k]], ff.w3.weight[NEU[k]], ff.w2.weight[:, NEU[k]] = orig[k]


def odds(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return float(np.log(p / (1 - p)))


# strength: set against the runs that loop, on the build half
scales = {}
for k, lift in ((0, LIFTP), (1, LIFTO)):
    pts = pick(k, "cutafter", True, 60)
    off_(k)
    p0 = chance(pts, TARGET[k])
    s = 3.0 / GAIN
    for it in range(10):
        setn(k, s)
        p1 = chance(pts, TARGET[k])
        got = odds(p1) - odds(p0)
        L.append(f"  word {k}: scale {s:.3g}: chance {p0:.3f} -> {p1:.3f} (lift {got:+.2f}, asked {lift})")
        print(L[-1], flush=True)
        if abs(got / lift - 1) < 0.12:
            break
        s *= float(np.clip(lift / max(got, 0.05), 0.3, 3.0))
    scales[k] = s
    off_(k)

# report on the held-out half, before and after, one word at a time and the sentence as a whole
GR = ((" before the answer", "before"), (" after, finished runs", "finafter"), (" after, cut-off runs", "cutafter"))
for build_part, tag in ((False, "held out"),):
    res = {}
    for name, on in (("before edit", False), ("after edit", True)):
        for k in (0, 1):
            setn(k, scales[k]) if on else off_(k)
        for lab, grp in GR:
            p_w = chance(pick(0, grp, build_part, 80), TARGET[0])
            p_f = chance(pick(1, grp, build_part, 80), TARGET[1])
            res[(name, lab)] = (p_w, p_f)
    for lab, _ in GR:
        a, b = res[("before edit", lab)], res[("after edit", lab)]
        L.append(f"  {tag}{lab:24s} P(Thus) {a[0]:.3f} -> {b[0]:.3f}; P(final or answer after an opener) {a[1]:.3f} -> {b[1]:.3f}; sentence {a[0] * a[1]:.4f} -> {b[0] * b[1]:.4f}")
    # the ordinary words: how far the next-word chances move
    ords = pick(2, "any", build_part, 150)
    for k in (0, 1):
        off_(k)
    lp0 = [logp(pt) for pt in ords]
    for k in (0, 1):
        setn(k, scales[k])
    kl = [float((a.exp() * (a - logp(pt))).sum()) for a, pt in zip(lp0, ords)]
    L.append(f"  ordinary words {len(ords)}: drift (KL) mean {np.mean(kl):.4f}, 99% {np.quantile(kl, 0.99):.3f}, max {max(kl):.3f}")
for k in (0, 1):
    setn(k, scales[k])
os.makedirs(out, exist_ok=True)
m.save_pretrained(out)
tok.save_pretrained(out)
for f_ in os.listdir(base):
    if f_.endswith((".py", ".jinja", ".json", ".txt")) and not f_.endswith(".index.json") and not os.path.exists(os.path.join(out, f_)):
        shutil.copy(os.path.join(base, f_), out)
meta = json.load(open(os.path.join(base, "pam.json"))) if os.path.exists(os.path.join(base, "pam.json")) else {}
meta["handin_cell"] = {"layer": LB, "neurons": NEU, "T": T, "fa": FA, "scales": scales, "lift": [LIFTP, LIFTO]}
json.dump(meta, open(os.path.join(out, "pam.json"), "w"), indent=1)
open(os.path.join(out, "done_report.txt"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
print("SAVED", out)
