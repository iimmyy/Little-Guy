#!/usr/bin/env python3
"""Second wire into c5's doubt crowd (Immy 2026-11-03 morning, "try both": option b).
c5m2b put the misfit detector in layer 17 (c5's key layer) because LFM3 resets quiet lanes at the next layer
(lane_carry.py): selective, safe, but quiet (doubt at misfits 0.062 -> 0.076). The misfit reads clearest at index 9
(0.74). This keeps the detector there and gives it its own wire: the lane that layers 10-17 do not write and layers
10-23 read least ("private 10-23", lane_carry.py v5: carry 0.87 at d 0.5, a lane that survives because nobody resets it).
  detector   one quiet existing layer-L MLP neuron: gate = steep * (misfit(x) - th), up = misfit(x) (as pam_misfit.py),
             down = a write along the lane, calibrated to push the stream by d along it at misfit forks
  crowd      the SAME 242 neurons c5 ducks (pam.json receptors). Each crowd layer learns a second sidechain read that
             listens ONLY to the lane (plus its own near-constant line for the offset): 1 when the detector is at its
             misfit-fork height, 0 when silent. Each receptor's gate gets +gain * (gate_at_hot + shift) * read2: the
             mirror of c5's duck (c5: -(gate_at_hot + shift) * read). One alarm system, two inputs: belief ducks the
             crowd through 397, a misfit lifts the same crowd through the lane.
c5's key, wire 397 and its crowd reads are untouched. Built on 2/3 of the forks by problem, the held-out 1/3 reports.
The push cannot be raised: the lane is unwritten, so its post-norm weights are small and the write tops out at ~0.515
(d 1.5 / 3: scale x1791 for the same 0.514; a run-away scale also swamps the layer's other writes). cancel=1 cleans
the READ instead: the crowd hears the lane minus what the rest of its input predicts the lane holds (common-mode
rejection; fit on ordinary words only, where the detector is silent).
usage: pam_wire2.py <c5 folder> <problems.jsonl> <thinking.jsonl> <out folder> [layer=9 steep=30 ord=20 d=0.5 gain=1 cancel=0 dev=cuda]
"""
import json, os, random, shutil, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from patients import _lfm3
from situation_probe import TAIL
from fit_test import forks


def ridge0(Xa, y, alphas=(1, 10, 100, 1e3, 1e4), w=None):
    """no intercept, scale-normalised, strength by 5 folds, optional row weights (as pam_misfit.py)"""
    sc = np.sqrt((Xa ** 2).mean(0)) + 1e-6
    Z = Xa / sc
    if w is not None:
        sw = np.sqrt(w / w.mean())
        Z, y = Z * sw[:, None], y * sw
    f = np.arange(len(y)) % 5
    best = None
    for al in alphas:
        err = sum(((Z[f == i] @ np.linalg.solve(Z[f != i].T @ Z[f != i] + al * np.eye(Z.shape[1]), Z[f != i].T @ y[f != i]) - y[f == i]) ** 2).sum()
                  for i in range(5))
        if best is None or err < best[0]:
            best = (err, al)
    return np.linalg.solve(Z.T @ Z + best[1] * np.eye(Z.shape[1]), Z.T @ y) / sc


DOUBT_IDS = [1938, 2375, 3911, 7731, 8952]
c5, src, think_f, out = sys.argv[1:5]
opt = dict(a.split("=", 1) for a in sys.argv[5:])
DEV, LB, STEEP, NORD = opt.get("dev", "cuda"), int(opt.get("layer", 9)), float(opt.get("steep", 30)), int(opt.get("ord", 20))
D_, GAIN, CANCEL = float(opt.get("d", 0.5)), float(opt.get("gain", 1)), opt.get("cancel") == "1"
META = json.load(open(os.path.join(c5, "pam.json")))
CROWD, SHIFT = META["receptors"], float(META["shift"])
CL = sorted({c["layer"] for c in CROWD})
p = _lfm3(DEV, c5)
tok, m = p.tok, p.model
lay = m.model.layers
H = m.config.hidden_size


def gram(lo, hi, write):
    """how strongly layers lo..hi read (write=False) or write (write=True) each stream direction (lane_carry.py)"""
    Mt = torch.zeros(H, H)
    with torch.no_grad():
        for i in range(lo, hi + 1):
            l2 = lay[i]
            if write:
                ws = [(l2.post_ffn_norm.weight, l2.feed_forward.w2.weight),
                      (l2.post_operator_norm.weight, l2.self_attn.o_proj.weight if l2.is_attention_layer else l2.conv.out_proj.weight)]
                for nw, W in ws:
                    Rm = (W.float() * nw.float()[:, None]).cpu(); Mt += Rm @ Rm.T / Rm.pow(2).sum()
            else:
                ws = [(l2.ffn_norm.weight, l2.feed_forward.w1.weight), (l2.ffn_norm.weight, l2.feed_forward.w3.weight)]
                ws += ([(l2.operator_norm.weight, w.weight) for w in (l2.self_attn.q_proj, l2.self_attn.k_proj, l2.self_attn.v_proj, l2.self_attn.gate_proj)]
                       if l2.is_attention_layer else [(l2.operator_norm.weight, l2.conv.in_proj.weight)])
                for nw, W in ws:
                    Rm = (W.float() * nw.float()[None, :]).cpu(); Mt += Rm.T @ Rm / Rm.pow(2).sum()
    return Mt / Mt.trace()


LANE = torch.linalg.eigh(gram(LB + 1, len(lay) - 1, False) + gram(LB + 1, 17, True))[1][:, 0]
lane = LANE.to(DEV)

# ---- capture: the detector's own input at the forks (a right / b fits / c misfit) and at ordinary thinking words
ff = lay[LB].feed_forward
got = {}
ff.w1.register_forward_hook(lambda mod, a, o: got.__setitem__("x", a[0][0].float()))
lay[LB].register_forward_hook(lambda mod, a, o: got.__setitem__("hL", (o[0] if isinstance(o, tuple) else o)[0].float()))
for L in CL:
    lay[L].feed_forward.register_forward_pre_hook(lambda mod, a, L=L: got.__setitem__(("c", L), a[0][0].float()))
items = forks(src, think_f)
rng = random.Random(0)
seqs, X, V, G = [], [], [], []
for k, (iid, text, think, A, g, (s0, s1, cut), (b, c, r)) in enumerate(items):
    head = tok.apply_chat_template([{"role": "user", "content": text + TAIL}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=True) + "\n"
    n_head = len(tok(head, add_special_tokens=False).input_ids)
    for vi, v in enumerate((A, b, c)):
        ids = tok(head + think[:s0] + str(v) + think[s1:cut], add_special_tokens=False).input_ids
        with torch.no_grad():
            m.model(input_ids=torch.tensor([ids], device=DEV))
        poss = [(len(ids) - 1, vi)]
        if vi == 0:
            poss += [(j, 3) for j in rng.sample(list(range(n_head, len(ids) - 2)), min(NORD, len(range(n_head, len(ids) - 2))))]
        for pos, vv in poss:
            X.append(got["x"][pos].cpu()); V.append(vv); G.append(k)
            seqs.append((ids, pos, vv, k))
    if k % 100 == 0:
        print(f"capture {k}/{len(items)}", flush=True)
X, V, G = torch.stack(X).numpy().astype(np.float64), np.array(V), np.array(G)
build = G % 3 != 2
names = ["a right", "b wrong, fits", "c wrong, misfit", "ordinary words"]
L_ = [f"second wire into c5's crowd: detector layer {LB}, lane 'private {LB + 1}-{len(lay) - 1}', push {D_}, gain {GAIN}; "
      f"{len(items)} forks (build {len(set(G[build]))} problems, held out {len(set(G[~build]))}); crowd {len(CROWD)} neurons in layers {CL}"]

# ---- the detector (the pam_misfit.py recipe)
w = np.array([1.0 / (V[build] == v).sum() for v in V[build]])
line = ridge0(X[build], (V[build] == 2).astype(float), w=w)
mis = X @ line
th = float(np.quantile(mis[build & ((V == 0) | (V == 1))], 0.90))
gate_line = ridge0(X[build], mis[build] - th, w=w)
with torch.no_grad():
    Xt = torch.tensor(X, dtype=torch.float32, device=DEV)
    act = F.silu(Xt @ ff.w1.weight.float().T) * (Xt @ ff.w3.weight.float().T)
    loud = (act.abs() * ff.w2.weight.float().norm(dim=0)[None]).kthvalue(max(1, int(0.999 * act.shape[0])), dim=0).values
    BN = int(loud.argmin())
    del Xt, act
post = lay[LB].post_ffn_norm.weight.float()
psafe = torch.where(post.abs() < 1e-3, torch.full_like(post, 1e-3) * torch.sign(post + 1e-12), post)
col = lane / psafe                                     # after the post-norm the write points exactly along the lane
gt = torch.tensor(STEEP * gate_line, dtype=ff.w1.weight.dtype, device=DEV)
lt = torch.tensor(line, dtype=ff.w1.weight.dtype, device=DEV)
orig_det = (ff.w1.weight[BN].clone(), ff.w3.weight[BN].clone(), ff.w2.weight[:, BN].clone())
gin = X @ (STEEP * gate_line)
L_.append(f"  detector: layer {LB} #{BN}, threshold {th:+.3f}; fires " +
          ", ".join(f"{names[v][:1]} {(gin[~build & (V == v)] > 0).mean():.0%}" for v in range(4)) + " (held out)")


def set_det(scale):
    with torch.no_grad():
        ff.w1.weight[BN], ff.w3.weight[BN] = gt, lt
        ff.w2.weight[:, BN] = (scale * col).to(ff.w2.weight.dtype)


def lane_push(sel, scale):
    """median push along the lane right after layer LB at the chosen words (detector at scale vs off)"""
    vals = []
    for (ids, pos, vv, k) in seqs:
        if not sel(vv, k):
            continue
        with torch.no_grad():
            ff.w1.weight[BN], ff.w3.weight[BN], ff.w2.weight[:, BN] = orig_det
            m.model(input_ids=torch.tensor([ids[:pos + 1]], device=DEV)); h0 = got["hL"][pos].clone()
            set_det(scale)
            m.model(input_ids=torch.tensor([ids[:pos + 1]], device=DEV)); h1 = got["hL"][pos].clone()
        vals.append(float((h1 - h0) @ lane))
    return float(np.median(vals))


mis_b = lambda vv, k: vv == 2 and k % 3 != 2
s_ = 1.0
for it in range(8):
    pu = lane_push(mis_b, s_)
    L_.append(f"  write step {it}: scale {s_:.4g}, lane push at misfit forks {pu:+.3f} (asked {D_})"); print(L_[-1], flush=True)
    if abs(pu / D_ - 1) < 0.05:
        break
    if pu <= 0 or s_ > 1e4:
        sys.exit("the lane push does not come out of the layer; stop")
    s_ *= min(D_ / pu, 10.0)
set_det(s_)

# ---- the crowd's second read: only the lane (and the layer's own near-constant line), fit to the detector's firing
DET = F.silu(torch.tensor(gin)) * torch.tensor(mis)                        # the detector's activation (its input is unchanged)
DET = DET.numpy()
dhot = float(np.median(DET[build & (V == 2)]))
kt = np.clip(DET / dhot, 0, None)
cap = {L: [] for L in CL}
for (ids, pos, vv, k) in seqs:
    with torch.no_grad():
        m.model(input_ids=torch.tensor([ids[:pos + 1]], device=DEV))
    for L in CL:
        cap[L].append(got[("c", L)][pos].cpu())
RD2 = {}
for L in CL:
    Xc = torch.stack(cap[L]).double()
    nw = lay[L].ffn_norm.weight.double().cpu()
    u = LANE.double() / torch.where(nw.abs() < 1e-3, torch.full_like(nw, 1e-3), nw)     # x_norm . u = (x / rms) . lane
    s = Xc[torch.tensor(V == 3)].mean(0); s = s / s.norm()
    if CANCEL:
        # what the lane holds anyway, predicted from the rest of the crowd's input on ordinary words (the detector is
        # silent there, so the prediction never sees a misfit); the read listens to the lane minus that prediction
        Xo = Xc[torch.tensor(build & (V == 3))].numpy()
        beta = torch.tensor(ridge0(Xo, Xo @ u.numpy()))
        u = u - beta
    Fm = torch.stack([Xc @ u, Xc @ s], 1).numpy()
    wb = np.where(V == 3, 1.0, 5.0)[build]                                      # forks weigh more than the many ordinary words
    A_ = Fm[build] * np.sqrt(wb)[:, None]
    coef = np.linalg.lstsq(A_, kt[build] * np.sqrt(wb), rcond=None)[0]
    RD2[L] = (coef[0] * u + coef[1] * s).float()
    rd = Fm @ coef
    L_.append(f"  layer {L}: second read (held out) median " + ", ".join(f"{names[v][:1]} {np.median(rd[~build & (V == v)]):+.2f}" for v in range(4))
              + f"; ordinary 99% |read| {np.quantile(np.abs(rd[~build & (V == 3)]), .99):.2f}")
print("\n".join(L_[-len(CL):]), flush=True)


def doubt_at(part):
    """chance the next word opens doubt at each group's words (build or held-out half)"""
    res = {v: [] for v in range(4)}
    for (ids, pos, vv, k) in seqs:
        if (k % 3 != 2) != part:
            continue
        with torch.no_grad():
            lg = m(input_ids=torch.tensor([ids[:pos + 1]], device=DEV)).logits[0, pos].float()
        res[vv].append(lg)
    return res


orig_crowd = {}
for c in CROWD:
    W1 = lay[c["layer"]].feed_forward.w1.weight
    orig_crowd[(c["layer"], c["neuron"])] = W1[c["neuron"]].clone()


def set_crowd(on):
    with torch.no_grad():
        for c in CROWD:
            L, j = c["layer"], c["neuron"]
            W1 = lay[L].feed_forward.w1.weight
            row = orig_crowd[(L, j)].float()
            if on:
                row = row + GAIN * (c["gate_at_hot"] + SHIFT) * RD2[L].to(DEV)
            W1[j] = row.to(W1.dtype)


for part in (True, False):
    tag = "build" if part else "held out"
    set_crowd(True)
    after = doubt_at(part)
    with torch.no_grad():
        ff.w1.weight[BN], ff.w3.weight[BN], ff.w2.weight[:, BN] = orig_det
    set_crowd(False)
    befo = doubt_at(part)
    set_det(s_); set_crowd(True)
    pd = lambda lgs: float(np.mean([float(torch.softmax(l, -1)[DOUBT_IDS].sum()) for l in lgs]))
    kl = [float((torch.log_softmax(a, -1).exp() * (torch.log_softmax(a, -1) - torch.log_softmax(b, -1))).sum())
          for a, b in zip(befo[3], after[3])]
    L_.append(f"  {tag}: chance the next word opens doubt, c5 -> c5 + second wire: " +
              ", ".join(f"{names[v]} {pd(befo[v]):.3f} -> {pd(after[v]):.3f}" for v in range(4)) +
              f"; ordinary words drift KL mean {np.mean(kl):.4f}, 99% {np.quantile(kl, .99):.3f}")
    print(L_[-1], flush=True)
os.makedirs(out, exist_ok=True)
m.save_pretrained(out)
tok.save_pretrained(out)
for f in os.listdir(c5):
    if f.endswith((".py", ".jinja", ".json", ".txt")) and not f.endswith(".index.json") and not os.path.exists(os.path.join(out, f)):
        shutil.copy(os.path.join(c5, f), out)
META["second_wire"] = {"cancel": CANCEL, "layer": LB, "neuron": BN, "threshold": th, "steep": STEEP, "push": D_, "gain": GAIN, "scale": s_,
                       "lane": "private %d-%d" % (LB + 1, len(lay) - 1)}
json.dump(META, open(os.path.join(out, "pam.json"), "w"), indent=1)
open(os.path.join(out, "pam_wire2_report.txt"), "w").write("\n".join(L_) + "\n")
print("\n".join(L_))
print("SAVED", out)
