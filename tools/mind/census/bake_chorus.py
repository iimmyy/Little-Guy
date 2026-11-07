#!/usr/bin/env python3
"""A chorus for the urgency flag (step C of the installed time sense, NIGHT-PLAN-2026-10-29).

Bound found with bake_urgency.py: the sandwich norm caps one neuron's voice at about one layer's MLP output, and the
final stream wanders along any listening direction about as much (u3: closing-word boost +-5.5 at zero urgency for
+5 per unit urgency; steeper urgency (u4) is renormalized away). So the flag is repeated: in each relay layer one weak
slot reads the source neuron's flag (layer 10, along its own dark direction f0) and writes a copy along its own dark
direction f_i (all at right angles: no relay reads another relay, so nothing compounds). The receptor then listens to
all copies together (Fisher on the joint image): the stream's wander along different directions is mostly
independent, so the signal-to-noise grows ~ sqrt(copies).

relay neuron i: silu(k * (uhat - 0.3)) * uhat, uhat = the source flag read at layer i's MLP input, in units of the
source's urgency (fitted per layer); quiet below ~0.3, then grows ~quadratically (the chorus swells with the urgency)

usage: bake_chorus.py <urgency folder> <answers jsonl[,more]> <out folder> [relays=11-21 k=2 close=5 switch=3 n=120 dev=mps]
"""
import json, os, random, shutil, sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from patients import _lfm3

src_dir, ans, out = sys.argv[1:4]
opt = dict(a.split("=", 1) for a in sys.argv[4:])
lo, hi = (int(x) for x in opt.get("relays", "11-21").split("-"))
RELAYS = list(range(lo, hi + 1))
K, CLOSE, SWITCH, N, DEV = float(opt.get("k", 2)), float(opt.get("close", 5)), float(opt.get("switch", 3)), int(opt.get("n", 120)), opt.get("dev", "mps")
TAIL = "\n\nEnd with a line: Answer: <your answer>"
info = json.load(open(os.path.join(src_dir, "urgency.json")))
L0, j0, END, ALT = info["layer"], info["slot"], info["end"], info["alt"]
p = _lfm3(DEV, src_dir)
tok, m = p.tok, p.model
lay = m.model.layers
ff0 = lay[L0].feed_forward
rows = [r for path in ans.split(",") for r in map(json.loads, open(path)) if r["thinking"]][:N]
rng = random.Random(4)


def safe(w):
    w = w.float().cpu()
    return torch.where(w >= 0, 1.0, -1.0) * w.abs().clamp(min=0.1 * float(w.abs().median()))


SEQ = []
for r in rows:
    pr = tok(tok.apply_chat_template([{"role": "user", "content": r["q"] + TAIL}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=True), add_special_tokens=False).input_ids
    th = tok("\n" + r["thinking"], add_special_tokens=False).input_ids[:6000]
    rp = tok("\n</think>\n\n" + r["reply"], add_special_tokens=False).input_ids[:300] if r["finish"] == "stop" else []
    T = len(th)
    lg = np.exp(np.random.RandomState(T).uniform(0, np.log(max(2, T - 1)), 96)).astype(int)
    tk = sorted(set(range(0, min(T, 64), 4)) | set(int(t) for t in lg if t < T))
    SEQ.append((pr + th + rp, torch.tensor([len(pr) + t for t in tk] + [len(pr) + T + k for k in range(len(rp))]), len(tk), len(rp)))
think = torch.cat([torch.tensor([1] * a + [0] * b) for _, _, a, b in SEQ]).bool()


def collect(hooks):
    res = []
    with torch.no_grad():
        for ids, pos, _, _ in SEQ:
            cap = {}
            hs = hooks(pos, cap)
            m.model(input_ids=torch.tensor([ids], device=DEV))
            for h in hs:
                h.remove()
            res.append(cap)
    return {k: torch.cat([c[k] for c in res]) for k in res[0]}


def act_hook(ff, j, key, pos, cap):
    return ff.register_forward_pre_hook(lambda mod, a: cap.__setitem__(key, (torch.nn.functional.silu(
        a[0][0, pos].float() @ ff.w1.weight[j].float()) * (a[0][0, pos].float() @ ff.w3.weight[j].float())).cpu()))


# the source's flag direction f0: what its output column writes, undone from post_ffn_norm's weight
f0 = ff0.w2.weight[:, j0].float().cpu() * safe(lay[L0].post_ffn_norm.weight)
f0 = f0 / f0.norm()

# ---- 1. what each relay layer sees (source on)
def h1(pos, cap):
    hs = [act_hook(ff0, j0, "u", pos, cap),
          m.model.embedding_norm.register_forward_pre_hook(lambda mod, a: cap.__setitem__("fin", a[0][0, pos].float().cpu()))]
    for i in RELAYS:
        hs.append(lay[i].feed_forward.register_forward_pre_hook(lambda mod, a, i=i: cap.__setitem__(("x", i), a[0][0, pos].half().cpu())))
        hs.append(lay[i].post_ffn_norm.register_forward_pre_hook(lambda mod, a, i=i: cap.__setitem__(("o", i), a[0][0, pos].float().norm(dim=-1).cpu())))
    return hs


S = collect(h1)
U = S["u"]
print(f"{len(SEQ)} answers, {int(think.sum())} thinking words; source urgency > 0.5 on {float((U[think] > 0.5).float().mean()):.1%} of them", flush=True)
D = S["fin"].shape[1]
# dark directions for the copies: least read by the layers after the earliest relay, quiet at the final stream,
# at right angles to the source's flag and the carrier's line
car = torch.load(os.path.join(src_dir, "carrier.pt"), weights_only=False)
Mt = torch.zeros(D, D)
with torch.no_grad():
    for i in range(RELAYS[0] + 1, len(lay)):
        l2 = lay[i]
        ins = [(l2.ffn_norm.weight, l2.feed_forward.w1.weight), (l2.ffn_norm.weight, l2.feed_forward.w3.weight)]
        if l2.is_attention_layer:
            ins += [(l2.operator_norm.weight, w.weight) for w in (l2.self_attn.q_proj, l2.self_attn.k_proj, l2.self_attn.v_proj, l2.self_attn.gate_proj)]
        else:
            ins += [(l2.operator_norm.weight, l2.conv.in_proj.weight)]
        for nw, W in ins:
            Rm = (W.float() * nw.float()[None, :]).cpu()
            Mt += Rm.T @ Rm / Rm.pow(2).sum()
C = torch.cov(S["fin"].T); C = C / C.trace()
B = torch.stack([f0, car["gdir"]], 1); B, _ = torch.linalg.qr(B)
Q = torch.eye(D) - B @ B.T
ev, V = torch.linalg.eigh(Q @ (Mt / Mt.trace() + C) @ Q + B @ B.T)
dirs = V[:, :len(RELAYS)]

# ---- 2. one relay per layer
lines = [f"chorus on {os.path.basename(src_dir.rstrip('/'))}: source layer {L0} slot {j0}; relays {RELAYS}, k {K}"]
placed = []
for n, i in enumerate(RELAYS):
    ff = lay[i].feed_forward
    X = S[("x", i)].float()
    rb = f0 / safe(lay[i].ffn_norm.weight)
    r = X @ rb
    th_ = think
    A_ = torch.stack([U[th_], torch.ones(int(th_.sum()))], 1)
    al, be = (float(v) for v in torch.linalg.lstsq(A_, r[th_][:, None]).solution[:, 0])
    cor = float(torch.corrcoef(torch.stack([r[th_], U[th_]]))[0, 1])
    mu, sd = X.mean(0), X.std(0)
    g = int((mu.abs() / (sd + 1e-6)).argmax()); Gm = float(mu[g])
    uw = rb / al; u0 = -be / al                            # uhat = uw . x + u0
    w3 = uw.clone(); w3[g] += u0 / Gm
    w1 = K * uw; w1[g] += K * (u0 - 0.3) / Gm
    uhat = X @ w3
    a_ = torch.nn.functional.silu(X @ w1) * uhat
    with torch.no_grad():
        imp = ((torch.nn.functional.silu(X.to(DEV) @ ff.w1.weight.float().T) * (X.to(DEV) @ ff.w3.weight.float().T)).abs().mean(0)
               * ff.w2.weight.float().norm(dim=0)).cpu()
    j = int(imp.argmin())
    col = dirs[:, n] / safe(lay[i].post_ffn_norm.weight)
    ref = float(a_[think & (U > 0.8) & (U < 1.2)].median()) if (think & (U > 0.8) & (U < 1.2)).any() else 1.0
    amp = float(S[("o", i)].median()) / (abs(ref) * float(col.norm()) + 1e-9)
    with torch.no_grad():
        ff.w1.weight[j] = w1.to(ff.w1.weight)
        ff.w3.weight[j] = w3.to(ff.w3.weight)
        ff.w2.weight[:, j] = (amp * col).to(ff.w2.weight)
    placed.append((i, j))
    lines.append(f"  relay layer {i:2d} slot {j:4d}: reads the source flag with corr {cor:+.2f}; quiet words "
                 f"{float(a_[think & (U <= 0.05)].abs().median()):.3f}, at urgency ~1 {ref:.2f}, reply max {float(a_[~think].max()):.2f}")
print("\n".join(lines), flush=True)

# ---- 3. receptors on the whole chorus: paired runs (source on vs source silenced, which silences the relays)
S1 = collect(lambda pos, cap: [m.model.embedding_norm.register_forward_hook(lambda mod, a, o: cap.__setitem__("fin", o[0, pos].float().cpu())),
                               act_hook(ff0, j0, "u", pos, cap)])
col0 = ff0.w2.weight[:, j0].clone()
with torch.no_grad():
    ff0.w2.weight[:, j0] = 0
S0 = collect(lambda pos, cap: [m.model.embedding_norm.register_forward_hook(lambda mod, a, o: cap.__setitem__("fin", o[0, pos].float().cpu()))])
with torch.no_grad():
    ff0.w2.weight[:, j0] = col0
F1, F0, U1 = S1["fin"], S0["fin"], S1["u"]
IMG = F1 - F0
w_q = torch.where(~think, 4.0, 1.0)
mq = (F0 * w_q[:, None]).sum(0) / w_q.sum()
Zq = (F0 - mq) * w_q.sqrt()[:, None]
Cq = Zq.T @ Zq / w_q.sum(); Cq = Cq + 0.05 * Cq.diagonal().mean() * torch.eye(D)
# the chorus swells faster than the urgency, so the listener is sized on the image at urgency ~1
hot = think & (U1 > 0.05)
slope = (IMG[hot] * U1[hot, None]).sum(0) / U1[hot].pow(2).sum()
ulis = torch.linalg.solve(Cq, slope)
mid = think & (U1 > 0.8) & (U1 < 1.2)
gain = float((IMG[mid] @ ulis).median()) if mid.any() else float(slope @ ulis)
mf, sf = F0.mean(0), F0.std(0)
gf = int((mf.abs() / (sf + 1e-6)).argmax()); Gf = float(mf[gf])
rest = float(mq @ ulis)
rws = {}
with torch.no_grad():
    for tid, kk in ((END, CLOSE), (ALT, SWITCH)):
        row = (kk / gain) * ulis
        row[gf] -= (kk / gain) * rest / Gf
        m.lm_head.weight[tid] = (m.model.embed_tokens.weight[tid].float().cpu() + row).to(m.lm_head.weight)
        rws[tid] = row
boost = F1 @ rws[END]
noise = F0 @ rws[END]
lines.append(f"receptors: close {CLOSE} / switch {SWITCH} at urgency 1; closing-word boost by urgency (thinking words):")
for a, b in ((-9, 0.05), (0.05, 0.5), (0.5, 1), (1, 2), (2, 99)):
    s_ = think & (U1 > a) & (U1 <= b)
    if s_.any():
        lines.append(f"  urgency {a:>5.2f}..{b:<5.2f} {int(s_.sum()):6d} words: median {float(boost[s_].median()):+.2f}")
lines.append(f"  silent source (the stream's own wander along the receptor): 1-99% {float(noise.quantile(.01)):+.2f}..{float(noise.quantile(.99)):+.2f}")
lines.append(f"  reply words: median {float(boost[~think].median()):+.2f}, 99% {float(boost[~think].quantile(.99)):+.2f}, max {float(boost[~think].max()):+.2f}")
print("\n".join(lines[-8:]))
os.makedirs(out, exist_ok=True)
m.save_pretrained(out)
tok.save_pretrained(out)
for fn in os.listdir(src_dir):
    if fn.endswith((".py", ".jinja", ".pt")) and not os.path.exists(os.path.join(out, fn)):
        shutil.copy(os.path.join(src_dir, fn), out)
open(os.path.join(out, "chorus_report.txt"), "w").write("\n".join(lines) + "\n")
json.dump({**info, "relays": placed, "k": K, "close": CLOSE, "switch": SWITCH}, open(os.path.join(out, "urgency.json"), "w"), indent=1)
print("CHORUS_DONE")
