#!/usr/bin/env python3
"""Steps 3-4 of the combined build (Immy 2026-10-29: "one big build with every trick that worked, connected").
  stuck switch   well past his own expected time (the installed overtime feeling, read late in the stream) and still
                 thinking -> leave thinking and write the solution out in the reply (what v4 showed works on the
                 grinds), only on the stuck answers: the trigger level is picked on the train answers so it almost
                 never fires before an answer that would have come on its own
  reply guard    in the reply, the close word goes down, so a forced switch cannot leak a second close word
Both are single late neurons (layer 22: one layer of wander after them) plus the untied close-word row, sized on
paired runs (neuron on vs silenced). No training, no run-time code.

usage: bake_stuck.py <folder with urgency.json + belief.json> <answers jsonl> <out folder> [layer=22 early=0.05
       k=6 kg=12 push=20 guard=20 n=110 dev=mps]
"""
import json, os, re, shutil, sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../expert"))
ARGS, sys.argv[1:] = sys.argv[1:], []
from grade import right
from patients import _lfm3
from probe_math import ridge

src, ans, out = ARGS[:3]
opt = dict(a.split("=", 1) for a in ARGS[3:])
LL, EARLY, K, PUSH, GUARD = int(opt.get("layer", 22)), float(opt.get("early", 0.05)), float(opt.get("k", 6)), float(opt.get("push", 20)), float(opt.get("guard", 20))
N, DEV = int(opt.get("n", 110)), opt.get("dev", "mps")
KG = float(opt.get("kg", 12))                             # steep: the guard must be ~0 while thinking, or it nudges every close
TAIL = "\n\nEnd with a line: Answer: <your answer>"
CLAIM = re.compile(r"\\boxed\{([^{}]{1,40})\}|(?:answer is|answer =|answer:)\s*\$?([^\n$]{1,40}?)\$?(?=[.\n]|$)", re.I)
info = json.load(open(os.path.join(src, "urgency.json")))
bel = json.load(open(os.path.join(src, "belief.json")))
TAKEN = {(info["layer"], info["slot"])} | {tuple(x) for x in info.get("relays", [])} | \
        {tuple(bel["source"])} | {tuple(x) for x in bel["relays"]}
L0, j0, END = info["layer"], info["slot"], info["end"]
p = _lfm3(DEV, src)
tok, m = p.tok, p.model
lay = m.model.layers
rows = [r for r in map(json.loads, open(ans)) if r["thinking"]][:N]


def safe(w):
    w = w.float().cpu()
    return torch.where(w >= 0, 1.0, -1.0) * w.abs().clamp(min=0.1 * float(w.abs().median()))


SEQ, meta = [], []                     # per position: (row, thinking word index or -1 for reply)
first_ok = []                          # per row: thinking word where a right answer first appears (None if never)
for n, r in enumerate(rows):
    pr = tok(tok.apply_chat_template([{"role": "user", "content": r["q"] + TAIL}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=True), add_special_tokens=False).input_ids
    think = "\n" + r["thinking"]
    enc = tok(think, add_special_tokens=False, return_offsets_mapping=True)
    th, ends = enc.input_ids[:6000], np.array([e for _, e in enc.offset_mapping[:6000]])
    fo = None
    for mt in CLAIM.finditer(think):
        x = next(g for g in mt.groups() if g is not None).strip().rstrip(".").strip()
        if x and right({**r, "reply": x}, x):
            fo = int(np.searchsorted(ends, mt.end())); break
    first_ok.append(fo)
    T = len(th)
    ts = sorted(set(range(0, T, 24)))                  # every 24th thinking word: the whole run of the overtime feeling
    rp = tok("\n</think>\n\n" + r["reply"], add_special_tokens=False).input_ids[:240] if r["finish"] == "stop" else []
    rs = list(range(2, len(rp), 3))
    SEQ.append((pr + th + rp, torch.tensor([len(pr) + t for t in ts] + [len(pr) + T + k for k in rs])))
    meta += [(n, t) for t in ts] + [(n, -1) for _ in rs]
ROW = np.array([a for a, _ in meta]); TW = np.array([b for _, b in meta]); REP = TW < 0
FIN = np.array([r["finish"] == "stop" for r in rows])


def collect(hooks):
    res = []
    with torch.no_grad():
        for ids, pos in SEQ:
            cap = {}
            hs = hooks(pos, cap)
            m.model(input_ids=torch.tensor([ids], device=DEV))
            for h in hs:
                h.remove()
            res.append(cap)
    return {k: torch.cat([c[k] for c in res]) for k in res[0]}


ff0, ffl = lay[L0].feed_forward, lay[LL].feed_forward


def act_hook(ff, j, key, pos, cap):
    return ff.register_forward_pre_hook(lambda mod, a: cap.__setitem__(key, (torch.nn.functional.silu(
        a[0][0, pos].float() @ ff.w1.weight[j].float()) * (a[0][0, pos].float() @ ff.w3.weight[j].float())).cpu()))


S = collect(lambda pos, cap: [act_hook(ff0, j0, "u", pos, cap),
                              ffl.register_forward_pre_hook(lambda mod, a: cap.__setitem__("x", a[0][0, pos].half().cpu())),
                              lay[LL].post_ffn_norm.register_forward_pre_hook(lambda mod, a: cap.__setitem__("o", a[0][0, pos].float().norm(dim=-1).cpu())),
                              m.model.embedding_norm.register_forward_pre_hook(lambda mod, a: cap.__setitem__("fin", a[0][0, pos].float().cpu()))])
U, X = S["u"], S["x"].float()
Xn = X.numpy()
th_ = ~REP
# the overtime feeling read at the late layer (the chorus carries it there); held-out by answer
ev_, od_ = ROW % 2 == 0, ROW % 2 == 1
pu, _, _, _ = ridge(Xn[ev_ & th_], U.numpy()[ev_ & th_], Xn[od_ & th_])
_, _, _, (wu, bu) = ridge(Xn[th_], U.numpy()[th_], Xn[:1])
wu = torch.tensor(wu).float()
uhat = X @ wu + bu
# thinking vs reply, 1 / 0
pi, _, _, _ = ridge(Xn[ev_], th_[ev_].astype(float), Xn[od_])
_, _, _, (wi, bi) = ridge(Xn, th_.astype(float), Xn[:1])
wi = torch.tensor(wi).float()
inside = X @ wi + bi
lines = [f"stuck switch + reply guard on {os.path.basename(src.rstrip('/'))}: layer {LL}",
         f"held-out: overtime read at layer {LL} vs the source, corr {float(np.corrcoef(pu, U.numpy()[od_ & th_])[0, 1]):+.2f}; "
         f"thinking vs reply: thinking 5% {float(np.quantile(pi[th_[od_]], .05)):+.2f}, reply 95% {float(np.quantile(pi[~th_[od_]], .95)):+.2f}"]
# the trigger level: the lowest level that fires before a right answer would have come in at most EARLY of the
# finished right answers
fine = [n for n in range(len(rows)) if FIN[n] and first_ok[n] is not None]
stuck = [n for n in range(len(rows)) if not FIN[n]]


def first_fire(n, th):
    s = (ROW == n) & th_ & (uhat.numpy() >= th)
    return int(TW[s].min()) if s.any() else None


cands = np.quantile(uhat.numpy()[th_], np.linspace(0.90, 0.9995, 60))
curve = []
for c in cands[::6]:                                      # the trade-off, for the record
    e_ = sum(1 for n in fine if (f := first_fire(n, c)) is not None and f < first_ok[n])
    c_ = [first_fire(n, c) for n in stuck if first_fire(n, c) is not None]
    curve.append(f"  level {c:+.2f}: early in {e_}/{len(fine)} finished right, catches {len(c_)}/{len(stuck)} cut-offs"
                 f" (median word {int(np.median(c_)) if c_ else -1})")
print("trade-off:\n" + "\n".join(curve), flush=True)
if opt.get("dry") == "1":
    sys.exit(0)
TH = float(cands[-1])
for c in cands:
    early = sum(1 for n in fine if (f := first_fire(n, c)) is not None and f < first_ok[n])
    if early <= EARLY * len(fine):
        TH = float(c); break
early = [n for n in fine if (f := first_fire(n, TH)) is not None and f < first_ok[n]]
caught = [(n, first_fire(n, TH)) for n in stuck if first_fire(n, TH) is not None]
lines.append(f"trigger at overtime-read {TH:+.2f}: fires before the answer in {len(early)}/{len(fine)} finished right answers; "
             f"catches {len(caught)}/{len(stuck)} cut-off answers, median at thinking word "
             f"{int(np.median([f for _, f in caught])) if caught else -1}")
print("\n".join(lines), flush=True)
mu, sd = X.mean(0), X.std(0)
g = int((mu.abs() / (sd + 1e-6)).argmax()); Gm = float(mu[g])


def place(w1, w3, ref_mask, dirn, name):
    with torch.no_grad():
        imp = ((torch.nn.functional.silu(X.to(DEV) @ ffl.w1.weight.float().T) * (X.to(DEV) @ ffl.w3.weight.float().T)).abs().mean(0)
               * ffl.w2.weight.float().norm(dim=0)).cpu()
        for l_, j_ in TAKEN:
            if l_ == LL:
                imp[j_] = float("inf")
        j = int(imp.argmin())
        a_ = torch.nn.functional.silu(X @ w1) * (X @ w3)
        ref = float(a_[ref_mask].median())
        col = dirn / safe(lay[LL].post_ffn_norm.weight)
        amp = float(S["o"].median()) / (abs(ref) * float(col.norm()) + 1e-9)
        ffl.w1.weight[j] = w1.to(ffl.w1.weight); ffl.w3.weight[j] = (w3 / ref).to(ffl.w3.weight)
        ffl.w2.weight[:, j] = (amp * col).to(ffl.w2.weight)
    TAKEN.add((LL, j))
    v = a_ / ref
    lines.append(f"  {name}: layer {LL} slot {j}; value in thinking median {float(v[th_].median()):+.2f} 99% "
                 f"{float(v[th_].quantile(.99)):+.2f}, in reply median {float(v[~th_].median()):+.2f} 99% {float(v[~th_].quantile(.99)):+.2f}")
    return j


# directions: least read after LL (only layer 23 reads them), quiet at the final stream, off every other feeling's
D = X.shape[1]
Mt = torch.zeros(D, D)
with torch.no_grad():
    for i in range(LL + 1, len(lay)):
        l2 = lay[i]
        ins = [(l2.ffn_norm.weight, l2.feed_forward.w1.weight), (l2.ffn_norm.weight, l2.feed_forward.w3.weight)]
        if l2.is_attention_layer:
            ins += [(l2.operator_norm.weight, w.weight) for w in (l2.self_attn.q_proj, l2.self_attn.k_proj, l2.self_attn.v_proj, l2.self_attn.gate_proj)]
        else:
            ins += [(l2.operator_norm.weight, l2.conv.in_proj.weight)]
        for nw, W in ins:
            Rm = (W.float() * nw.float()[None, :]).cpu()
            Mt += Rm.T @ Rm / Rm.pow(2).sum()
    AV = [lay[l_].feed_forward.w2.weight[:, j_].float().cpu() * lay[l_].post_ffn_norm.weight.float().cpu() for l_, j_ in TAKEN]
C = torch.cov(S["fin"].T); C = C / C.trace()
Bv, _ = torch.linalg.qr(torch.stack([a / a.norm() for a in AV], 1))
Qv = torch.eye(D) - Bv @ Bv.T
ev, V = torch.linalg.eigh(Qv @ (Mt / Mt.trace() + C) @ Qv + Bv @ Bv.T)
# stuck neuron: silu(k (overtime-read - trigger)) * inside; reply guard: silu(k (0.5 - inside)) * 1
w1s = K * wu; w1s[g] += K * (bu - TH) / Gm
w3s = wi.clone(); w3s[g] += bi / Gm
hot = torch.tensor(th_) & (uhat >= TH)
js = place(w1s, w3s, hot if hot.any() else torch.tensor(th_), V[:, 0], "stuck switch")
w1g = -KG * wi; w1g[g] += KG * (0.5 - bi) / Gm
w3g = torch.zeros(D); w3g[g] = 1.0 / Gm
jg = place(w1g, w3g, torch.tensor(~th_), V[:, 1], "reply guard")
print("\n".join(lines[-2:]), flush=True)


# voice: paired runs for each neuron, close-word row up (stuck) and down (guard)
def fin_with(silence):
    keep = ffl.w2.weight[:, silence].clone() if silence is not None else None
    with torch.no_grad():
        if silence is not None:
            ffl.w2.weight[:, silence] = 0
    F = collect(lambda pos, cap: [m.model.embedding_norm.register_forward_hook(lambda mod, a, o: cap.__setitem__("fin", o[0, pos].float().cpu())),
                                  act_hook(ffl, js, "s", pos, cap), act_hook(ffl, jg, "g", pos, cap)])
    with torch.no_grad():
        if silence is not None:
            ffl.w2.weight[:, silence] = keep
    return F


F1 = fin_with(None)
D_s = F1["fin"] - fin_with(js)["fin"]
D_g = F1["fin"] - fin_with(jg)["fin"]
F0 = F1["fin"] - D_s - D_g
mq = F0.mean(0); Zq = F0 - mq
Cq = Zq.T @ Zq / len(F0); Cq = Cq + 0.05 * Cq.diagonal().mean() * torch.eye(D)
mf, sf = F0.mean(0), F0.std(0)
gf = int((mf.abs() / (sf + 1e-6)).argmax()); Gf = float(mf[gf])
rowsum = torch.zeros(D)
for img, val, kk, nm in ((D_s, F1["s"], PUSH, "stuck"), (D_g, F1["g"], -GUARD, "guard")):
    hv = val > 0.05
    slope = (img[hv] * val[hv, None]).sum(0) / val[hv].pow(2).sum()
    lis = torch.linalg.solve(Cq, slope)
    mid = (val > 0.8) & (val < 1.2)
    gain = float((img[mid] @ lis).median()) if mid.any() else float(slope @ lis)
    r_ = kk * lis / gain; r_[gf] -= kk * float(mq @ lis) / gain / Gf
    rowsum += r_
with torch.no_grad():
    m.lm_head.weight[END] = (m.lm_head.weight[END].float().cpu() + rowsum).to(m.lm_head.weight)
push = F1["fin"] @ rowsum
noise = F0 @ rowsum
lines.append("close-word push (score points):")
for nm, s_ in (("thinking, overtime below trigger", th_ & (uhat.numpy() < TH)), ("thinking, past trigger", th_ & (uhat.numpy() >= TH)),
               ("reply", ~th_)):
    v = push[torch.tensor(s_)]
    if len(v):
        lines.append(f"  {nm:34s} {len(v):6d} words: median {float(v.median()):+6.2f}, 1% {float(v.quantile(.01)):+6.2f}, 99% {float(v.quantile(.99)):+6.2f}")
lines.append(f"  silenced (the stream's own wander along the push): 1-99% {float(noise.quantile(.01)):+.2f}..{float(noise.quantile(.99)):+.2f}")
print("\n".join(lines[-5:]))
os.makedirs(out, exist_ok=True)
m.save_pretrained(out)
tok.save_pretrained(out)
for fn in os.listdir(src):
    if fn.endswith((".py", ".jinja", ".pt", ".json")) and not os.path.exists(os.path.join(out, fn)):
        shutil.copy(os.path.join(src, fn), out)
open(os.path.join(out, "stuck_report.txt"), "w").write("\n".join(lines) + "\n")
json.dump({"layer": LL, "stuck": js, "guard": jg, "trigger": TH, "push": PUSH, "guard_push": GUARD},
          open(os.path.join(out, "stuck.json"), "w"), indent=1)
print("STUCK_DONE")
