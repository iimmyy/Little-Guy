#!/usr/bin/env python3
"""Install the carrier head: the patient's own expected time, held while it thinks (step A of the installed time sense,
NIGHT-PLAN-2026-10-29). No training: one existing head of a full-attention layer is rewired; the file keeps its shape.

Why: the patient knows on `<think>` how long the question should take (difficulty_probe.py, rho 0.82) but drops it
once thinking starts (expected_persist.py, ~0.3). Training never made it keep the estimate. The carrier keeps it:
  query   the head's query is replaced by a constant pointing at a few unrotated head dimensions J (no position
          rotation there, so the match does not fade with distance), held by the attention input's giant dimension
  key     the head's key group gets a switch on J that fires only on `<think>` (a straight-line detector, clean by
          ~70 spreads, carrier_scan.py); the group's other heads stop reading J, so only the carrier feels it
  value   left alone: `<think>`'s own value already holds the expected time (rho 0.68-0.84, as much as the whole
          stream); a shared-value write (build 1) cost 0.058 KL
  output  the carrier reads the expected time off that value with one straight line (centered) and writes it into a
          dark stream direction (the one later layers read least), divided by the sandwich norm's weight; its
          sigmoid gate is held open
Checks on answers not used to build it: the carrier's attention share on `<think>` from thinking words, the expected
time read back mid-thinking along the dark direction (rank agreement with the real time), and drift (KL of the
next-word odds vs the untouched model, thinking and reply words).

usage: install_carrier.py <hf folder> <answers jsonl[,more]> <out folder> [layer=10 head=10 dims=37,106 target=solve
                          flag=0.3 dev=mps]
  target=solve  expected time = words until the answer first appears in the thinking; target=total = all thinking
  flag=K        the carried value's typical swing as K times the layer's summed attention output (louder = cleaner
                read-back, more pull on the sandwich norm)
"""
import json, os, shutil, sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../expert"))
ARGS, sys.argv[1:] = sys.argv[1:], []                     # grade.py grades its own arguments on import
from grade import answer_line
from patients import _lfm3
from probe_math import rho, ridge

src_dir, ans, out = ARGS[:3]
opt = dict(a.split("=", 1) for a in ARGS[3:])
L, HC, DEV = int(opt.get("layer", 10)), int(opt.get("head", 10)), opt.get("dev", "mps")
J = [int(x) for x in opt.get("dims", "37,106").split(",")]
TARGET, FLAG = opt.get("target", "solve"), float(opt.get("flag", 0.3))
TAIL = "\n\nEnd with a line: Answer: <your answer>"
p = _lfm3(DEV, src_dir)
p.model.config._attn_implementation = "eager"
tok, m = p.tok, p.model
lay = m.model.layers
ly, at = lay[L], lay[L].self_attn
H, KV, HD = at.config.num_attention_heads, at.config.num_key_value_heads, at.head_dim
G = HC // (H // KV)
mates = [h for h in range(G * (H // KV), (G + 1) * (H // KV)) if h != HC]
rows = [r for path in ans.split(",") for r in map(json.loads, open(path)) if r["thinking"]]
build = [r for i, r in enumerate(rows) if i % 2 == 0]
check = [r for i, r in enumerate(rows) if i % 2 == 1]


def target(r, th):
    T = r["think_tokens"] if r["finish"] == "stop" else 6000
    if TARGET == "total":
        return np.log(T)
    a = (answer_line(r["reply"], True) or "").strip().strip("$ ")
    k = r["thinking"].find(a) if a else -1
    s = len(th) if k < 0 else len(tok("\n" + r["thinking"][:k + len(a)], add_special_tokens=False).input_ids)
    return np.log(max(8, min(s, T)))


def seq(r):
    pr = tok(tok.apply_chat_template([{"role": "user", "content": r["q"] + TAIL}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=True), add_special_tokens=False).input_ids
    th = tok("\n" + r["thinking"], add_special_tokens=False).input_ids
    rp = tok("\n</think>\n\n" + r["reply"], add_special_tokens=False).input_ids[:200]
    return pr, th, rp


def run(R, hooks, cut=700):
    """Each answer once (question + up to `cut` thinking words + the start of the reply)."""
    res = []
    with torch.no_grad():
        for r in R:
            pr, th, rp = seq(r)
            ids = pr + th[:cut] + (rp if len(th) <= cut else [])
            cap = {}
            hs = hooks(cap)
            m(input_ids=torch.tensor([ids], device=DEV))
            for h in hs:
                h.remove()
            res.append((cap, len(pr) - 1, len(pr) + min(len(th), cut), len(ids), target(r, th)))
    return res


# ---- 1. what the layer sees (untouched model): attention input, the key group's raw keys, the MLP input
def h1(cap):
    return [ly.operator_norm.register_forward_hook(lambda mod, a, o: cap.__setitem__("x", o[0].float().cpu())),
            at.k_proj.register_forward_hook(lambda mod, a, o: cap.__setitem__("k", o[0, :, G * HD:(G + 1) * HD].float().cpu())),
            at.v_proj.register_forward_hook(lambda mod, a, o: cap.__setitem__("v", o[0, :, G * HD:(G + 1) * HD].float().cpu())),
            ly.post_operator_norm.register_forward_pre_hook(lambda mod, a: cap.__setitem__("ao", a[0][0].float().cpu())),
            ly.ffn_norm.register_forward_pre_hook(lambda mod, a: cap.__setitem__("res", a[0][0].float().cpu()))]


B = run(build, h1)
Xt = torch.stack([c["x"][P] for c, P, _, _, _ in B])
Xo = torch.cat([c["x"][P + 1:E] for c, P, E, _, _ in B])
yb = np.array([y for *_, y in B])
mu, sd = Xo.mean(0), Xo.std(0)
gd = int((mu.abs() / (sd + 1e-6)).argmax()); Gd = float(mu[gd])
Gt, Gt_sd = float(Xt[:, gd].mean()), float(Xt[:, gd].std())    # the same entry on `<think>` alone (the value's constant)
kn_think = float(torch.stack([c["k"][P] for c, P, _, _, _ in B]).norm(dim=-1).median())
ao = torch.cat([c["ao"][P + 1:E] for c, P, E, _, _ in B]).norm(dim=-1)
print(f"build {len(build)}, check {len(check)}; giant dim {gd} ({Gd:.3f}; on <think> {Gt:.3f} +- {Gt_sd / abs(Gt):.1%}); key group {G} raw key size on <think> {kn_think:.2f}; "
      f"summed attention output size {float(ao.median()):.3f}", flush=True)

# the `<think>` switch: Fisher line, other words centered to 0, `<think>` to 1
Cx = torch.cov(Xo.T); Cx = Cx + 0.05 * Cx.diagonal().mean() * torch.eye(Cx.shape[0])
dk = torch.linalg.solve(Cx, Xt.mean(0) - Xo.mean(0))
o_, t_ = float(Xo.mean(0) @ dk), float(Xt.mean(0) @ dk)
sw = dk / (t_ - o_); sw0 = -o_ / (t_ - o_)                   # sw.x + sw0: 0 on other words, 1 on `<think>`
# the expected time is already in `<think>`'s own value (carrier build 1 wrote it into a value channel the whole key
# group shares; cancelling it in the other heads cost 0.058 KL). So the value is left alone and the carrier only
# routes it: its output reads the expected time off the value with a straight line w, kept at right angles to the
# value's average on `<think>` so it reads 0 on an average question (centered: "longer / shorter than usual", so the
# sandwich norm's size noise scales the deviation, not the whole number)
Vt = torch.stack([c["v"][P] for c, P, _, _, _ in B])
mu_v = Vt.mean(0); mh = mu_v / mu_v.norm()
Pm = torch.eye(HD) - torch.outer(mh, mh)
Vc = (Vt - mu_v) @ Pm
yc = torch.tensor(yb - yb.mean(), dtype=torch.float32)
folds = torch.arange(len(yc)) % 5
best = None
for lam in (1e-3, 1e-2, 1e-1, 1, 10, 100):
    pr = torch.zeros(len(yc))
    for k in range(5):
        a_, b_ = folds != k, folds == k
        w_ = torch.linalg.solve(Vc[a_].T @ Vc[a_] + lam * torch.eye(HD), Vc[a_].T @ yc[a_])
        pr[b_] = Vc[b_] @ w_
    sc = rho(pr.numpy(), yb)
    if best is None or sc > best[0]:
        best = (sc, lam)
w = Pm @ torch.linalg.solve(Vc.T @ Vc + best[1] * torch.eye(HD), Vc.T @ yc)
ec = Vt @ w
e_mean, e_sd = float(ec.mean()), float(ec.std())
print(f"expected-time readout on <think>'s value: ridge {best[1]:g}, build-set cv rho {best[0]:+.2f}, "
      f"mean {e_mean:+.3f} (0 = centered), spread {e_sd:.2f}", flush=True)

# the dark stream direction: what layer L's own MLP and every later layer read least, where the stream barely moves
D = Xo.shape[1]
Mt = torch.zeros(D, D)
with torch.no_grad():
    for i in range(L, len(lay)):
        l2 = lay[i]
        ins = [(l2.ffn_norm.weight, l2.feed_forward.w1.weight), (l2.ffn_norm.weight, l2.feed_forward.w3.weight)]
        if i > L:
            if l2.is_attention_layer:
                a2 = l2.self_attn
                ins += [(l2.operator_norm.weight, w.weight) for w in (a2.q_proj, a2.k_proj, a2.v_proj, a2.gate_proj)]
            else:
                ins += [(l2.operator_norm.weight, l2.conv.in_proj.weight)]
        for nw, W in ins:
            Rm = (W.float() * nw.float()[None, :]).cpu()
            Mt += Rm.T @ Rm / Rm.pow(2).sum()
RES = torch.cat([c["res"][P + 1:E] for c, P, E, _, _ in B])   # the stream itself, where the flag must stand out
C = torch.cov(RES.T); C = C / C.trace()
evd, Vd = torch.linalg.eigh(Mt / Mt.trace() + C)
gdir = Vd[:, 0]
wp = ly.post_operator_norm.weight.float().cpu()
floor = 0.1 * float(wp.abs().median())
wp_safe = torch.where(wp >= 0, 1.0, -1.0) * wp.abs().clamp(min=floor)
col = gdir / wp_safe
print(f"dark direction read at {float(evd[0]):.2e} of the average", flush=True)

# ---- 2. install
ALPHA = 3.0 * kn_think                                      # the switch's size on J: 3x the raw key on `<think>`
amp = FLAG * float(ao.median()) / (e_sd * float(col.norm()) + 1e-9)
with torch.no_grad():
    Wq, Wk, Wvv, Wo, Wg = at.q_proj.weight, at.k_proj.weight, at.v_proj.weight, at.o_proj.weight, at.gate_proj.weight
    dt = Wq.dtype
    # query: a constant on J (through the giant dimension), nothing else
    q = torch.zeros(HD, D); q[J, gd] = 100.0 / Gd
    Wq[HC * HD:(HC + 1) * HD] = q.to(dt)
    for h in mates:                                         # the group's other heads stop reading J
        Wq[[h * HD + j for j in J]] = 0
    # key: the `<think>` switch on J
    Kg = Wk[G * HD:(G + 1) * HD].float().cpu()
    for j in J:
        Kg[j] += ALPHA * sw
        Kg[j, gd] += ALPHA * sw0 / Gd
    Wk[G * HD:(G + 1) * HD] = Kg.to(dt)
    # output: the carrier writes only the expected time read off the value, into the dark direction
    Wo[:, HC * HD:(HC + 1) * HD] = (amp * torch.outer(col, w)).to(dt)
    # gate: held open
    g_ = torch.zeros(D); g_[gd] = 15.0 / Gd
    Wg[HC] = g_.to(dt)


# ---- 3. checks on answers the build never saw, untouched model alongside for drift
p0 = _lfm3(DEV, src_dir)
m0 = p0.model
wf = ly.ffn_norm.weight.float().cpu()
wf_safe = torch.where(wf >= 0, 1.0, -1.0) * wf.abs().clamp(min=0.1 * float(wf.abs().median()))
readback = gdir / wf_safe                                   # the MLP input hands the stream over times ffn_norm's weight
share, rd, kl_t, kl_r = [], {t: ([], []) for t in (16, 64, 128, 256, 399)}, [], []
rd0 = {t: [] for t in rd}                                   # the untouched model along the same direction
with torch.no_grad():
    for r in check:
        pr, th, rp = seq(r)
        ids = pr + th[:400] + (rp if len(th) <= 400 else [])
        P, E = len(pr) - 1, len(pr) + min(len(th), 400)
        cap = {}
        hs = [at.register_forward_hook(lambda mod, a, o: cap.__setitem__("w", o[1][0, HC].float().cpu())),
              ly.feed_forward.register_forward_pre_hook(lambda mod, a: cap.__setitem__("mx", a[0][0].float().cpu()))]
        lp = torch.log_softmax(m(input_ids=torch.tensor([ids], device=DEV)).logits[0].float(), -1)
        for h in hs:
            h.remove()
        h0 = m0.model.layers[L].feed_forward.register_forward_pre_hook(lambda mod, a: cap.__setitem__("mx0", a[0][0].float().cpu()))
        lp0 = torch.log_softmax(m0(input_ids=torch.tensor([ids], device=DEV)).logits[0].float(), -1)
        h0.remove()
        k = (lp0.exp() * (lp0 - lp)).sum(-1).cpu()
        kl_t.append(k[P:E]); kl_r.append(k[E:])
        share.append(cap["w"][P + 1:E, P])
        s, s0 = cap["mx"] @ readback, cap["mx0"] @ readback
        y = target(r, th)
        for t in rd:
            if P + t < E:
                rd[t][0].append(float(s[P + t])); rd[t][1].append(y); rd0[t].append(float(s0[P + t]))
share = torch.cat(share)
kt, kr = torch.cat(kl_t), torch.cat([k for k in kl_r if len(k)])
lines = [f"carrier installed: layer {L} head {HC} (key group {G}, mates {mates}), dims {J}, target {TARGET}, flag {FLAG}",
         f"carrier attention on <think> from thinking words: median {float(share.median()):.3f}, 10% {float(share.quantile(.1)):.3f}",
         "expected time read back mid-thinking along the dark direction (check answers):"]
for t, (s, y) in rd.items():
    if len(s) >= 10:
        lines.append(f"  word {t:3d}: {len(s)} answers, rho {rho(s, y):+.2f} (untouched model, same direction: {rho(rd0[t], y):+.2f})")
lines.append(f"drift vs untouched (KL of next-word odds): thinking words mean {float(kt.mean()):.4f} (99% {float(kt.quantile(.99)):.3f}), "
             f"reply words mean {float(kr.mean()):.4f}")
os.makedirs(out, exist_ok=True)
m.save_pretrained(out)
tok.save_pretrained(out)
for fn in os.listdir(src_dir):
    if fn.endswith((".py", ".jinja")) and not os.path.exists(os.path.join(out, fn)):
        shutil.copy(os.path.join(src_dir, fn), out)
print("\n".join(lines))
open(os.path.join(out, "carrier_report.txt"), "w").write("\n".join(lines) + "\n")
torch.save({"layer": L, "head": HC, "dims": J, "gdir": gdir, "giant": gd, "e_mean": e_mean, "e_sd": e_sd, "target": TARGET,
            "amp": amp, "w": w}, os.path.join(out, "carrier.pt"))
print("CARRIER_DONE")
