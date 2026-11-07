#!/usr/bin/env python3
"""Bake the overtime feeling on top of the carrier (step B-C of the installed time sense, NIGHT-PLAN-2026-10-29).
No training; one weak MLP slot is rewritten and the output matrix is untied (as in bake_budget.py).

Immy (2026-10-29): "all he should know is his own time"; past what a question should take, he commits if he has the
answer, tries another way if not. Brain match: the urgency signal of decision making (urgency gating / collapsing
bounds, Cisek et al. 2009; Churchland, Kiani & Shadlen 2008): as time passes the bar to decide drops, so a choice
gets made on less evidence. Here:
  overtime z = own clock (log words thinking so far, read off the stream) - own expected time (the carrier's reading,
               calibrated to log words until the answer appears) - margin
  urgency  u = silu(z) inside the thinking (SwiGLU AND: w1 = z, w3 = "inside the thinking"), a ramp: 0 on time,
               ~1 per log-word past (2.7x the expected time = 1)
  receptors  the closing word gets +close*u, the "try another way" opener (token `Alternatively`) gets +switch*u.
             Graded, not a cliff: the bar drops slowly, so he stops where he is ready (after stating an answer) and
             switches where he is stuck; nothing is forced mid-sentence until the urgency is very high.

usage: bake_urgency.py <carrier folder> <answers jsonl[,more]> <out folder> [layer=10 close=8 switch=3 margin=0
                       steep=1 n=80 dev=mps]
"""
import json, os, random, shutil, sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../expert"))
ARGS, sys.argv[1:] = sys.argv[1:], []                     # grade.py grades its own arguments on import
from grade import answer_line
from patients import _lfm3
from probe_math import rho

src_dir, ans, out = ARGS[:3]
opt = dict(a.split("=", 1) for a in ARGS[3:])
L, CLOSE, SWITCH = int(opt.get("layer", 10)), float(opt.get("close", 8)), float(opt.get("switch", 3))
MARGIN, STEEP, N, DEV = float(opt.get("margin", 0)), float(opt.get("steep", 2)), int(opt.get("n", 80)), opt.get("dev", "mps")
TAIL = "\n\nEnd with a line: Answer: <your answer>"
car = torch.load(os.path.join(src_dir, "carrier.pt"), weights_only=False)
p = _lfm3(DEV, src_dir)
tok, m = p.tok, p.model
lay = m.model.layers
ff = lay[L].feed_forward
END, ALT = p.end, tok.convert_tokens_to_ids("Alternatively")
# finished answers and the ones that ran out of words (u1 fitted only finished ones and its clock never saw past ~2000
# words; the stuck grinds are exactly where overtime must register)
rows = [r for path in ans.split(",") for r in map(json.loads, open(path)) if r["thinking"]][:N]
rng = random.Random(2)
wf = lay[L].ffn_norm.weight.float().cpu()
readback = car["gdir"] / (torch.where(wf >= 0, 1.0, -1.0) * wf.abs().clamp(min=0.1 * float(wf.abs().median())))


def solve_words(r, th):
    if r["finish"] != "stop":
        return None                                         # never answered: overtime is measured from his own estimate
    a = (answer_line(r["reply"], True) or "").strip().strip("$ ")
    k = r["thinking"].find(a) if a else -1
    s = len(th) if k < 0 else len(tok("\n" + r["thinking"][:k + len(a)], add_special_tokens=False).input_ids)
    return max(8, min(s, len(th)))


def sequences():
    for r in rows:
        pr = tok(tok.apply_chat_template([{"role": "user", "content": r["q"] + TAIL}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=True), add_special_tokens=False).input_ids
        th = tok("\n" + r["thinking"], add_special_tokens=False).input_ids[:6000]
        rp = tok("\n</think>\n\n" + r["reply"], add_special_tokens=False).input_ids[:300] if r["finish"] == "stop" else []
        T = len(th)
        lg = np.exp(np.random.RandomState(len(th)).uniform(0, np.log(max(2, T - 1)), 96)).astype(int)
        tk = sorted(set(range(0, min(T, 64), 4)) | set(int(t) for t in lg if t < T))
        yield r, pr + th + rp, len(pr), T, tk, list(range(len(rp))), solve_words(r, th)


SEQ = list(sequences())                                    # sampled once, so every pass looks at the same words


def collect(hooks):
    res = []
    with torch.no_grad():
        for r, ids, P, T, tk, rk, S in SEQ:
            pos = torch.tensor([P + t for t in tk] + [P + T + k for k in rk])
            cap = {}
            hs = hooks(pos, cap)
            m.model(input_ids=torch.tensor([ids], device=DEV))     # hooks only: no output matrix needed
            for h in hs:
                h.remove()
            res.append((cap, tk, len(rk), S, T))
    return res


# ---- 1. the stream at layer L's MLP input (carrier model), and at the post-MLP norm and the final norm
S1 = collect(lambda pos, cap: [
    ff.register_forward_pre_hook(lambda mod, a: cap.__setitem__("x", a[0][0, pos].float().cpu())),
    m.model.embedding_norm.register_forward_pre_hook(lambda mod, a: cap.__setitem__("fin", a[0][0, pos].float().cpu())),
    lay[L].post_ffn_norm.register_forward_pre_hook(lambda mod, a: cap.__setitem__("ffn", a[0][0, pos].float().cpu()))])
X = torch.cat([c["x"] for c, *_ in S1]); FFN = torch.cat([c["ffn"] for c, *_ in S1]); FINs = torch.cat([c["fin"] for c, *_ in S1])
think = torch.cat([torch.tensor([1] * len(tk) + [0] * nr) for _, tk, nr, _, _ in S1]).bool()
tw = torch.cat([torch.tensor(tk + [-1] * nr) for _, tk, nr, _, _ in S1]).float()
ans_id = torch.cat([torch.full((len(tk) + nr,), i) for i, (_, tk, nr, _, _) in enumerate(S1)])
known = torch.tensor([S is not None for *_, S, _ in S1])
Ssolve = torch.tensor([np.log(S) if S is not None else 0.0 for *_, S, _ in S1]).float()
build = ans_id % 2 == 0
print(f"{len(rows)} answers, {int(think.sum())} thinking words, {int((~think).sum())} reply words", flush=True)
mu, sd = X.mean(0), X.std(0)
g = int((mu.abs() / (sd + 1e-6)).argmax()); Gm = float(mu[g])
e_g = torch.zeros_like(mu); e_g[g] = 1.0
print(f"MLP input giant dimension {g}: {Gm:.3f} +- {float(sd[g] / abs(Gm)):.1%}", flush=True)


def fit(Xa, y, lam, wt=None):
    """Ridge in the raw stream with an intercept (returns w, b); wt = per-row weights."""
    wt = torch.ones(len(y)) if wt is None else wt
    wt = wt / wt.mean()
    xm, ym = (Xa * wt[:, None]).sum(0) / wt.sum(), float((y * wt).sum() / wt.sum())
    Z = (Xa - xm) * wt.sqrt()[:, None]
    w = torch.linalg.solve(Z.T @ Z + lam * Z.pow(2).sum(0).mean() * torch.eye(Z.shape[1]), Z.T @ ((y - ym) * wt.sqrt()))
    return w, float(ym - xm @ w)


# the clock: log words thinking so far (words 1+), on thinking words
th_ = think & (tw >= 1)
lt = torch.log(tw.clamp(min=1))
best = None
for lam in (1e-3, 1e-2, 1e-1, 1.0):
    w_, b_ = fit(X[th_ & build], lt[th_ & build], lam)
    err = (X[th_ & ~build] @ w_ + b_ - lt[th_ & ~build]).abs().median()
    if best is None or err < best[0]:
        best = (float(err), lam)
cw, cb = fit(X[th_], lt[th_], best[1])
clock_err = best[0]
# the expected time: the carrier's reading along the read-back direction, calibrated to log words until the answer
r_ = X @ readback
per = torch.stack([r_[(ans_id == i) & th_].mean() for i in range(len(S1))])
A_ = torch.stack([per, torch.ones_like(per)], 1)
ab = torch.linalg.lstsq(A_[known], Ssolve[known][:, None]).solution[:, 0]
a_r, b_r = float(ab[0]), float(ab[1])
Ssolve = torch.where(known, Ssolve, a_r * per + b_r)      # never answered: his own estimate is the reference
exp_ = a_r * r_ + b_r
print(f"clock: log words from the stream, held-out median miss {clock_err:.2f} log units (x{np.exp(clock_err):.2f}); "
      f"expected time from the carrier: rank agreement with the real time-to-answer {rho(per[known].numpy(), Ssolve[known].numpy()):+.2f} "
      f"({int(known.sum())} answered, {int((~known).sum())} ran out of words)", flush=True)
# overtime z = clock - expected - margin, one straight line: w_z . x + z0
true_z = lt - Ssolve[ans_id] - MARGIN
# u2: the overtime line fitted directly on "how far past the expected time" (his clock compresses long stretches,
# Vierordt's law: at word 5000 it reads ~2000, so clock minus estimate under-reads exactly the grinds); built on half
# u3: every stretch of thinking length counts the same (most words sit in short answers, so u2's line regressed the
# long grinds toward the middle: +0.27 where the truth was +1.49)
bins = torch.bucketize(torch.log(tw.clamp(min=1)), torch.linspace(0, float(np.log(6000)), 12))
cnt = torch.bincount(bins[th_], minlength=13).float()
wt_all = 1.0 / cnt[bins].clamp(min=1)
best = None
for lam in (1e-3, 1e-2, 1e-1, 1.0):
    w_, b_ = fit(X[th_ & build], true_z[th_ & build], lam, wt_all[th_ & build])
    sc = rho((X[th_ & ~build] @ w_ + b_).numpy(), true_z[th_ & ~build].numpy())
    if best is None or sc > best[0]:
        best = (sc, lam)
w_z, z0 = fit(X[th_], true_z[th_], best[1], wt_all[th_])
z = X @ w_z + z0
print(f"direct overtime line: held-out rank agreement {best[0]:+.2f}; by thinking length (true overtime / line):", flush=True)
for lo, hi in ((1, 256), (256, 1024), (1024, 2048), (2048, 4096), (4096, 6001)):
    s_ = th_ & ~build & (tw >= lo) & (tw < hi)
    if s_.any():
        print(f"  words {lo}-{hi}: {float(true_z[s_].median()):+.2f} / {float(z[s_].median()):+.2f}", flush=True)
print(f"overtime line vs true overtime on thinking words: rank agreement {rho(z[th_].numpy(), true_z[th_].numpy()):+.2f}", flush=True)
w1 = STEEP * w_z + (STEEP * z0 / Gm) * e_g
# inside the thinking (Fisher think vs reply, shifted by a margin as in bake_budget.py build 3)
Cx = torch.cov(X.T); Cx = Cx + 0.05 * Cx.diagonal().mean() * torch.eye(Cx.shape[0])
d = torch.linalg.solve(Cx, X[think].mean(0) - X[~think].mean(0))
pt, pr_ = float(X[think].mean(0) @ d), float(X[~think].mean(0) @ d)
w3 = d / (pt - pr_) + ((-pr_ / (pt - pr_) - 0.25) / Gm) * e_g
act = torch.nn.functional.silu(X @ w1) * (X @ w3)
# the Weber check: per answer, the first thinking word where urgency passes 0.5, vs that answer's time-to-answer
fire, sol = [], []
for i in range(len(S1)):
    s_ = (ans_id == i) & think
    tws, a_ = tw[s_], act[s_]
    o = tws.argsort()
    hit = (a_[o] > 0.5).nonzero()
    if not known[i]:
        continue
    fire.append(float(tws[o][hit[0, 0]]) if len(hit) else float(S1[i][4]))   # never passes: its whole length
    sol.append(float(np.exp(Ssolve[i])))
fire, sol = np.array(fire), np.array(sol)
print(f"urgency first passes 0.5 at median word {np.median(fire):.0f} (answer appears at median {np.median(sol):.0f}); "
      f"rank agreement of fire word with time-to-answer {rho(fire, sol):+.2f}; reply words with urgency > 0.5: "
      f"{float((act[~think] > 0.5).float().mean()):.2%}", flush=True)

# ---- 2. install the urgency neuron in the weakest slot; flag = a dark direction, at right angles to the carrier's
with torch.no_grad():
    a_all = (torch.nn.functional.silu(X.to(DEV) @ ff.w1.weight.float().T) * (X.to(DEV) @ ff.w3.weight.float().T)).abs().mean(0)
    imp = (a_all * ff.w2.weight.float().norm(dim=0)).cpu()
j = int(imp.argmin())
D = X.shape[1]
Mt = torch.zeros(D, D)
FIN = None
with torch.no_grad():
    for i in range(L + 1, len(lay)):
        l2 = lay[i]
        ins = [(l2.ffn_norm.weight, l2.feed_forward.w1.weight), (l2.ffn_norm.weight, l2.feed_forward.w3.weight)]
        if l2.is_attention_layer:
            a2 = l2.self_attn
            ins += [(l2.operator_norm.weight, w.weight) for w in (a2.q_proj, a2.k_proj, a2.v_proj, a2.gate_proj)]
        else:
            ins += [(l2.operator_norm.weight, l2.conv.in_proj.weight)]
        for nw, W in ins:
            Rm = (W.float() * nw.float()[None, :]).cpu()
            Mt += Rm.T @ Rm / Rm.pow(2).sum()
# quiet where the receptor listens: the final stream's own spread (u2 used the spread at layer L, and the receptor
# heard the final stream wander +-7 log-odds); bake_budget.py chose its flag this way
C = torch.cov(FINs.T); C = C / C.trace()
Q = torch.eye(D) - torch.outer(car["gdir"], car["gdir"])      # keep off the carrier's line
ev, V = torch.linalg.eigh(Q @ (Mt / Mt.trace() + C) @ Q + torch.outer(car["gdir"], car["gdir"]))
f = V[:, 0]
gp = lay[L].post_ffn_norm.weight.float().cpu()
gp_safe = torch.where(gp >= 0, 1.0, -1.0) * gp.abs().clamp(min=0.1 * float(gp.abs().median()))
col = f / gp_safe
on = act[think] > 0.5
amp = float(FFN.norm(dim=-1).median()) / (float(act[think][on].median()) * float(col.norm()) + 1e-9) if on.any() else 1.0
with torch.no_grad():
    ff.w1.weight[j] = w1.to(ff.w1.weight)
    ff.w3.weight[j] = w3.to(ff.w3.weight)
    ff.w2.weight[:, j] = (amp * col).to(ff.w2.weight)
    m.lm_head.weight = torch.nn.Parameter(m.model.embed_tokens.weight.detach().clone())
    m.config.tie_word_embeddings = False
print(f"urgency neuron: layer {L} slot {j} (importance {float(imp[j]):.1e}, median {float(imp.median()):.1e})", flush=True)

# ---- 3. receptors: at the final norm, the best straight-line listener for the flag (Fisher, as bake_receptor.py),
# sized so the boost is close*u and switch*u (u = the neuron's urgency), resting level taken back off
S2 = collect(lambda pos, cap: [
    m.model.embedding_norm.register_forward_hook(lambda mod, a, o: cap.__setitem__("fin", o[0, pos].float().cpu())),
    ff.register_forward_pre_hook(lambda mod, a: cap.__setitem__("act", (torch.nn.functional.silu(
        a[0][0, pos].float() @ ff.w1.weight[j].float()) * (a[0][0, pos].float() @ ff.w3.weight[j].float())).cpu()))])
F1 = torch.cat([c["fin"] for c, *_ in S2]); U = torch.cat([c["act"] for c, *_ in S2])
# the same words with the neuron silenced: the flag's exact image is the difference (u1 regressed the stream on u and
# picked up the stream's own drift with time, so the boost wandered +-4 at zero urgency)
col_saved = ff.w2.weight[:, j].clone()
with torch.no_grad():
    ff.w2.weight[:, j] = 0
S0 = collect(lambda pos, cap: [m.model.embedding_norm.register_forward_hook(lambda mod, a, o: cap.__setitem__("fin", o[0, pos].float().cpu()))])
with torch.no_grad():
    ff.w2.weight[:, j] = col_saved
F0 = torch.cat([c["fin"] for c, *_ in S0])
IMG = F1 - F0
# noise = the stream with the neuron silent (every word, reply words 4x: a stray close there breaks the answer)
w_q = torch.where(~think, 4.0, 1.0)
mq = (F0 * w_q[:, None]).sum(0) / w_q.sum()
Zq = (F0 - mq) * w_q.sqrt()[:, None]
Cq = Zq.T @ Zq / w_q.sum(); Cq = Cq + 0.05 * Cq.diagonal().mean() * torch.eye(D)
# signal per unit urgency: the image regressed on u (through the origin: no urgency, no image)
hot = think & (U > 0.05)
slope = (IMG[hot] * U[hot, None]).sum(0) / U[hot].pow(2).sum()
ulis = torch.linalg.solve(Cq, slope)
gain = float(slope @ ulis)                                   # listener output per unit urgency
mf, sf = F0.mean(0), F0.std(0)
gf = int((mf.abs() / (sf + 1e-6)).argmax()); Gf = float(mf[gf])
rest = float(mq @ ulis)
rows_ = {}
with torch.no_grad():
    for tid, k in ((END, CLOSE), (ALT, SWITCH)):
        row = (k / gain) * ulis
        row[gf] -= (k / gain) * rest / Gf
        m.lm_head.weight[tid] = (m.model.embed_tokens.weight[tid].float().cpu() + row).to(m.lm_head.weight)
        rows_[tid] = row
boost = F1 @ rows_[END]
lines = [f"urgency baked on {os.path.basename(src_dir.rstrip('/'))}: layer {L} slot {j}; close {CLOSE}/unit, switch {SWITCH}/unit, margin {MARGIN}",
         f"clock held-out miss x{np.exp(clock_err):.2f}; expected-time rank agreement {rho(per[known].numpy(), Ssolve[known].numpy()):+.2f}; "
         f"overtime line vs truth {rho(z[th_].numpy(), true_z[th_].numpy()):+.2f}",
         f"urgency first > 0.5 at median word {np.median(fire):.0f} vs answer appears at {np.median(sol):.0f}; fire vs answer time rank {rho(fire, sol):+.2f}",
         "closing-word boost (log-odds) by urgency, thinking words:"]
for lo, hi in ((-9, 0.05), (0.05, 0.5), (0.5, 1), (1, 2), (2, 99)):
    s_ = think & (U > lo) & (U <= hi)
    if s_.any():
        lines.append(f"  urgency {lo:>5.2f}..{hi:<5.2f} {int(s_.sum()):6d} words: median {float(boost[s_].median()):+.2f}")
noise_b = F0 @ rows_[END]
lines.append(f"  silent neuron (the stream's own wander along the receptor): median {float(noise_b.median()):+.2f}, 1-99% "
             f"{float(noise_b.quantile(.01)):+.2f}..{float(noise_b.quantile(.99)):+.2f}")
lines.append(f"  reply words: median {float(boost[~think].median()):+.2f}, 99% {float(boost[~think].quantile(.99)):+.2f}, "
             f"max {float(boost[~think].max()):+.2f}")
print("\n".join(lines))
os.makedirs(out, exist_ok=True)
m.save_pretrained(out)
tok.save_pretrained(out)
for fn in os.listdir(src_dir):
    if fn.endswith((".py", ".jinja", ".pt")) and not os.path.exists(os.path.join(out, fn)):
        shutil.copy(os.path.join(src_dir, fn), out)
open(os.path.join(out, "urgency_report.txt"), "w").write("\n".join(lines) + "\n")
json.dump({"layer": L, "slot": j, "close": CLOSE, "switch": SWITCH, "margin": MARGIN, "end": END, "alt": ALT,
           "carrier": car["layer"], "carrier_head": car["head"]}, open(os.path.join(out, "urgency.json"), "w"), indent=1)
print("URGENCY_DONE")
