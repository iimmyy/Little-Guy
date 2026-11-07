#!/usr/bin/env python3
"""Immy's scene formula, wired in (2026-11-04: "finish the formula fully, add it in, wire it all together"). Bolt-on: no training, no new neurons, same file shape.
E = sum(springs^2) + sum(walls crossed^2): one quiet neuron per term in the compressor's key layer (17). Its gate and its level both read the term's line, so its output silu(k(t - t0)) * t is the one-sided square of that term:
nothing under t0, growing with t^2 over it. All of them write the compressor's wire downward, so the wire reads duck(belief) - E: a broken scene first takes back the duck and, past that, pushes the doubt crowd up
(Immy: "surpass the suspicion of the untouched model").
Judgment = precision x error^2: the precision line goes into every gate as an added term, so on a sharply pinned scene a term crosses sooner and on a floating one later. Adding in the gate is the log of multiplying. It is a stand-in for the product: one neuron cannot do a third-order term.
Why not one layer earlier: a first version computed E at layer 16 on a quiet lane and a judgment neuron at 17 multiplied it by precision. The lane does not carry (crossed minus right +0.01 at any write size, scene_cell v1, s1b): same finding as lane_carry.py. A note has to be written where its readers are.
a* is not solved for: no neuron can run an argmin. Check 3 found he does not hold a sharp a* but holds the comparison, so the springs read error = a - a* directly.
Lines from scene_lines.py (fit with reply and ordinary words at zero). Strengths set on the build half; numbers reported on the other 1/3.
usage: scene_cell.py <base compressor folder> <lines17.npz> <mat.npz> <out folder> <runs.jsonl> [skip= kp=2 k=8 push=1.0 n=60 dev=cuda precision=1 terms=size,demand,whole,sign]
"""
import hashlib, json, os, random, re, shutil, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from patients import _lfm3

args = [a for a in sys.argv[1:] if "=" not in a]
opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
base, l17f, matf, out, runs = args[0], args[1], args[2], args[3], args[4:]
KG, KP, PUSH, NPTS, DEV = float(opt.get("k", 8)), float(opt.get("kp", 2)), float(opt.get("push", 1.0)), int(opt.get("n", 60)), opt.get("dev", "cuda")
USE_P = opt.get("precision", "1") == "1"
skip = {json.loads(l)["id"] for l in open(opt["skip"])} if "skip" in opt else set()
TERMS = tuple(opt.get("terms", "size,demand,whole,sign").split(","))
LJ = 17
DOUBT = [1938, 2375, 3911, 7731, 8952]
H = lambda i: int(hashlib.md5(str(i).encode()).hexdigest(), 16)
isb = lambda i: H(i) % 3 != 2
ln17 = np.load(l17f)
z = np.load(matf, allow_pickle=True)
X17 = z["x17"].astype(np.float64)
ver, grp, pid, at = z["ver"], z["grp"], z["pid"], z["at"]
bd = np.array([isb(i) for i in pid])
V = {n: i for i, n in enumerate("abcdwn")}
L = [f"scene cell on {os.path.basename(base.rstrip('/'))}: terms {TERMS} in layer {LJ}, precision {'on' if USE_P else 'off'}"]


def ridge0(X, y, w, groups, alphas=(1e3, 1e4, 1e5)):
    sc = X.std(0) + 1e-6
    Z = X / sc * np.sqrt(w)[:, None]
    Y = y * np.sqrt(w)
    f = np.array([H(g) % 4 for g in groups])
    best = None
    for al in alphas:
        e = sum(((Z[f == i] @ np.linalg.solve(Z[f != i].T @ Z[f != i] + al * np.eye(Z.shape[1]), Z[f != i].T @ Y[f != i]) - Y[f == i]) ** 2).sum() for i in range(4))
        if best is None or e < best[0]:
            best = (e, al)
    return np.linalg.solve(Z.T @ Z + best[1] * np.eye(Z.shape[1]), Z.T @ Y) / sc


p_ = _lfm3(DEV, base)
tok, m = p_.tok, p_.model
meta = json.load(open(os.path.join(base, "pam.json")))
comp = meta["compressor"]
GK, GN, WIRE = comp["key_layer"], comp["key_neuron"], comp["wire"]
assert GK == LJ, GK
fJ = m.model.layers[LJ].feed_forward
postJ = m.model.layers[LJ].post_ffn_norm.weight.float()


def quiet(ff, X, n, keep=()):
    with torch.no_grad():
        Xt = torch.tensor(X[::2], dtype=torch.float32, device=DEV)
        act = F.silu(Xt @ ff.w1.weight.float().T) * (Xt @ ff.w3.weight.float().T)
        ws = act.abs() * ff.w2.weight.float().norm(dim=0)[None]
        loud = ws.kthvalue(max(1, int(0.999 * ws.shape[0])), dim=0).values
    return [int(i) for i in loud.argsort() if int(i) not in keep][:n], loud


NE, loud = quiet(fJ, X17, len(TERMS), keep={GN})
L.append(f"  neurons: layer {LJ} {dict(zip(TERMS, NE))}; key #{GN}, wire {WIRE}")
pv = X17 @ ln17["precision"]
p0 = float(np.median(pv[bd & (at == 0)]))
gates = {}
for tn in TERMS:
    t = X17 @ ln17[tn]
    ordinary = (at == 2) | (at == 3)
    t0 = float(np.quantile(t[bd & ordinary], 0.98))
    tg = KG * (t - t0) + (KP * (pv - p0) if USE_P else 0.0)
    gl = ridge0(X17, tg, np.ones(len(t)), pid)
    gates[tn] = (gl, ln17[tn], t0)
    gi = X17 @ gl
    L.append(f"  term {tn}: t0 {t0:+.3f}; gate over 0: ordinary {np.mean(gi[~bd & (at == 2)] > 0):.1%}, reply {np.mean(gi[~bd & (at == 3)] > 0):.1%}, right statement {np.mean(gi[~bd & (at == 0) & (ver == 0)] > 0):.1%}")
# ---- fork points from his runs, every version, for strengths and the held-out report
NUMS = re.compile(r"(?<![\d.,/^])\b(\d{1,6})\b(?![\d.,/^])")
rng = random.Random(3)
FORKS = []
ASK = re.compile(r"[Hh]ow many ([a-zA-Z-]+)")
UNITS = {"hours", "minutes", "seconds", "days", "weeks", "months", "years", "miles", "kilometers", "km", "meters", "feet", "inches", "yards", "centimeters", "cm", "dollars", "cents", "cups", "liters", "litres",
         "gallons", "ounces", "pounds", "kilograms", "kg", "grams", "tons", "acres", "percent", "degrees", "more", "fewer", "less", "times", "square", "cubic", "ml", "quarts", "pints", "mph", "lbs", "oz"}
for f_ in runs:
    for t in map(json.loads, open(f_)):
        if t["id"] in skip or t["finish"] != "stop" or not t["thinking"]:
            continue
        s = str(t["answer"]).strip().replace(",", "")
        if not re.fullmatch(r"\d+", s) or not (3 <= int(s) <= 100000):
            continue
        A, th = int(s), t["thinking"]
        ms = list(re.finditer(rf"(?:=|\bis\b|\bare\b|\bget\b|\bgives\b|\bequals\b|\btotal of\b)\s*\$?\\?\(?\s*({s})(?![\d.,/])", th))
        if not ms:
            continue
        e = re.search(r"[.!?\n]", th[ms[0].end(1):])
        if not e:
            continue
        a, b, cut = ms[0].start(1), ms[0].end(1), ms[0].end(1) + e.end()
        ask = ASK.search(t["q"])
        g = ("unit" if ask.group(1).lower() in UNITS else "count") if ask else "other"
        earlier = [int(x) for x in NUMS.findall(th[:a]) if int(x) not in (A, 0, 1) and abs(np.log(int(x) / A)) > 0.1]
        head = tok.apply_chat_template([{"role": "user", "content": t["q"]}], tokenize=False, add_generation_prompt=True, enable_thinking=True) + "\n"
        vs = {"a": str(A), "b": str(max(round(A * 2), A + 1)), "w": f"{A}.5", "n": f"-{A}"}
        if earlier:
            vs["d"] = str(earlier[-1])
        FORKS.append((t["id"], g, {k: tok(head + th[:a] + v + th[b:cut], add_special_tokens=False).input_ids for k, v in vs.items()}))
rng.shuffle(FORKS)
pick = lambda build, ver_, n=NPTS, group=None: [fk[2][ver_] for fk in FORKS if isb(fk[0]) == build and ver_ in fk[2] and (group is None or fk[1] == group)][:n]


def read_at(ids_list, layer):
    """stream at this layer's MLP input, last word; next-word doubt"""
    got = {}
    h = m.model.layers[layer].feed_forward.register_forward_pre_hook(lambda mod, a: got.__setitem__("x", a[0][0, -1].float()))
    xs, ds = [], []
    for ids in ids_list:
        with torch.no_grad():
            hs = m.model(input_ids=torch.tensor([ids], device=DEV)).last_hidden_state[0, -1]
            ds.append(float(torch.softmax(m.lm_head(hs).float(), -1)[DOUBT].sum()))
        xs.append(got["x"].clone())
    h.remove()
    return torch.stack(xs), np.array(ds)


orig = {nn: (fJ.w1.weight[nn].clone(), fJ.w3.weight[nn].clone(), fJ.w2.weight[:, nn].clone()) for nn in NE}
T_ = lambda a: torch.tensor(a, dtype=fJ.w1.weight.dtype, device=DEV)


def set_terms(s_):
    with torch.no_grad():
        for tn, nn in zip(TERMS, NE):
            gl, lv, _ = gates[tn]
            fJ.w1.weight[nn] = T_(gl)
            fJ.w3.weight[nn] = T_(lv)
            d = torch.zeros_like(fJ.w2.weight[:, nn]); d[WIRE] = -s_ / float(postJ[WIRE])
            fJ.w2.weight[:, nn] = d


def off():
    with torch.no_grad():
        for nn, (a_, b_, c_) in orig.items():
            fJ.w1.weight[nn], fJ.w3.weight[nn], fJ.w2.weight[:, nn] = a_, b_, c_


got_w = {}
hk = m.model.layers[LJ].register_forward_hook(lambda mod, a, o: got_w.__setitem__("w", (o[0] if isinstance(o, tuple) else o)[0, -1, WIRE].float().item()))


def wire_doubt(lst):
    w_, d_ = [], []
    for ids in lst:
        with torch.no_grad():
            hs = m.model(input_ids=torch.tensor([ids], device=DEV)).last_hidden_state[0, -1]
            d_.append(float(torch.softmax(m.lm_head(hs).float(), -1)[DOUBT].sum()))
        w_.append(got_w["w"])
    return np.array(w_), np.array(d_)


rights = pick(True, "a")
targets = pick(True, "n") + pick(True, "w", group="count") + pick(True, "d")
w_r0, _ = wire_doubt(rights)
w_t0, _ = wire_doubt(targets)
key_rise = float(np.quantile(w_r0, 0.95) - np.median(w_r0))     # how far the compressor's key lifts the wire when it fires
ASK_DROP = PUSH * max(key_rise, 1.0)
s_ = 0.2
for it in range(12):
    set_terms(s_)
    w_r1, _ = wire_doubt(rights)
    w_t1, _ = wire_doubt(targets)
    drop = float(np.median(w_t0 - w_t1))
    drop_r = float(np.median(w_r0 - w_r1))
    L.append(f"  write scale {s_:.3g}: wire drop at broken scenes {drop:+.3f} (asked {ASK_DROP:.3f}; key's rise {key_rise:.3f}), at right statements {drop_r:+.3f}")
    print(L[-1], flush=True)
    if drop > 0 and abs(drop / ASK_DROP - 1) < 0.15:
        break
    s_ *= float(np.clip(ASK_DROP / max(drop, 1e-3), 0.3, 4.0))
    if s_ > 1e3:
        L.append("  write does not reach the wire")
        break
SE = s_
hold = [("right a", pick(False, "a")), ("wrong b", pick(False, "b")), ("earlier number d", pick(False, "d")), ("non-whole, countable", pick(False, "w", group="count")), ("non-whole, unit", pick(False, "w", group="unit")), ("negative", pick(False, "n"))]
after = {nm: wire_doubt(lst) for nm, lst in hold}
off()
before = {nm: wire_doubt(lst) for nm, lst in hold}
set_terms(SE)
for nm, _ in hold:
    L.append(f"  held out, {nm:22s} wire {np.median(before[nm][0]):+.2f} -> {np.median(after[nm][0]):+.2f}; chance of doubt next {before[nm][1].mean():.3f} -> {after[nm][1].mean():.3f}")
hk.remove()
os.makedirs(out, exist_ok=True)
m.save_pretrained(out)
tok.save_pretrained(out)
for f_ in os.listdir(base):
    if f_.endswith((".py", ".jinja", ".json", ".txt")) and not f_.endswith(".index.json") and not os.path.exists(os.path.join(out, f_)):
        shutil.copy(os.path.join(base, f_), out)
meta["scene_cell"] = {"layer": LJ, "neurons": dict(zip(TERMS, NE)), "scale": SE, "precision": USE_P, "kp": KP, "k": KG, "push": PUSH}
json.dump(meta, open(os.path.join(out, "pam.json"), "w"), indent=1)
open(os.path.join(out, "scene_report.txt"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
print("SAVED", out)
