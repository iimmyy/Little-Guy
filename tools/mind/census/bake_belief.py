#!/usr/bin/env python3
"""The belief gate (Immy 2026-10-29). He knows when he is right (confidence probe: 0.97) and he hands in on his own
when the doubt crowd in layers 18-21 goes quiet (fork_votes.py: doubt is the default, ~9 score points). Immy's rule:
one check, and the check is one logic run; "make it so loud he can't ignore it".

So the gate fires at the fork after he states the answer AGAIN (his one check is done) and his belief is high:
  source neuron (layer L, weakest slot): silu(k * (checked - 0.5)) * belief
      checked = a straight line for "this is the word before the next sentence, after an answer he has stated
                before" (0 at the first statement, at other thinking words and in the reply)
      belief  = a straight line for "that answer is right", scaled to 1 at checked right answers
  chorus: one weak slot per relay layer copies it along its own dark direction (one neuron alone is heard at
      signal-to-noise ~1: the sandwich norm caps its voice; see bake_chorus.py)
  voice: the untied output rows of the hand-in openers he uses ("I'll ...", "We need to end ...", "Now ...") go up
      and the doubt openers ("But", "However", "Let's", "Wait", ...) go down, sized on paired runs (gate on vs
      gate silenced), so at gate 1 the gap moves by hand + doubt points

No training, no run-time code: rewritten weights in the same file shape, runs in any stock engine.

usage: bake_belief.py <hf folder> <answers jsonl[,more]> <out folder> [layer=12 relays=13-21 k=8 theta=0.5
       hand=6 doubt=6 rk=2 n=110 dev=mps]
"""
import json, os, random, re, shutil, sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../expert"))
ARGS, sys.argv[1:] = sys.argv[1:], []                     # grade.py grades its own arguments on import
from grade import right
from patients import _lfm3
from probe_math import ridge

src, ans, out = ARGS[:3]
opt = dict(a.split("=", 1) for a in ARGS[3:])
LS = int(opt.get("layer", 12))
lo, hi = (int(x) for x in opt.get("relays", "13-21").split("-"))
RELAYS = list(range(lo, hi + 1))
K, TH, HAND, DOUBTK, RK = (float(opt.get(k, v)) for k, v in (("k", 8), ("theta", 0.5), ("hand", 6), ("doubt", 6), ("rk", 2)))
N, DEV = int(opt.get("n", 110)), opt.get("dev", "mps")
TAIL = "\n\nEnd with a line: Answer: <your answer>"
CLAIM = re.compile(r"\\boxed\{([^{}]{1,40})\}|(?:answer is|answer =|answer:)\s*\$?([^\n$]{1,40}?)\$?(?=[.\n]|$)", re.I)
DOUBT = re.compile(r"(But|Wait|However|Hmm|Let me|Let's|Actually|Alternatively|Double|Check|Hold on)\b", re.I)
HANDOFF = re.compile(r"(I'll (produce|write|output|give|present|answer)|We need to (end|output|produce|write|present)"
                     r"|Now (we need to )?(output|produce|write)|So (the )?final (output|line)|Thus,? final\.?\s*$"
                     r"|Thus,? I'll|End with|So final answer|Thus final answer)", re.I)

p = _lfm3(DEV, src)
tok, m = p.tok, p.model
lay = m.model.layers
# building on another baked feeling (avoid=<its urgency.json>): keep off its slots and its stream directions
TAKEN, AV = set(), []
if "avoid" in opt:                                         # an urgency.json, or a carrier.pt alone (no neurons placed)
    info0 = json.load(open(opt["avoid"])) if opt["avoid"].endswith(".json") else {}
    TAKEN = ({(info0["layer"], info0["slot"])} if "slot" in info0 else set()) | {tuple(x) for x in info0.get("relays", [])}
    for l_, j_ in TAKEN:
        AV.append(lay[l_].feed_forward.w2.weight[:, j_].float().cpu() * lay[l_].post_ffn_norm.weight.float().cpu())
    cp = os.path.join(os.path.dirname(opt["avoid"]), "carrier.pt")
    if os.path.exists(cp):
        AV.append(torch.load(cp, weights_only=False)["gdir"].float())


def weakest(imp, layer):
    for l_, j_ in TAKEN:
        if l_ == layer:
            imp[j_] = float("inf")
    return int(imp.argmin())
rows = [r for path in ans.split(",") for r in map(json.loads, open(path)) if r["thinking"]][:N]


def safe(w):
    w = w.float().cpu()
    return torch.where(w >= 0, 1.0, -1.0) * w.abs().clamp(min=0.1 * float(w.abs().median()))


def auroc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, bool)
    if y.all() or not y.any():
        return float("nan")
    r = np.argsort(np.argsort(s)) + 1
    return (r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())


# ---- 0. the words: every claim's fork (the word before the next sentence), plus other thinking words and the reply
SEQ, lab = [], []                      # lab per position: (row, role 0 thinking / 1 fork / 2 reply, ok, prior, next id, kind)
for n, r in enumerate(rows):
    pr = tok(tok.apply_chat_template([{"role": "user", "content": r["q"] + TAIL}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=True), add_special_tokens=False).input_ids
    think = "\n" + r["thinking"]
    enc = tok(think, add_special_tokens=False, return_offsets_mapping=True)
    th, starts = enc.input_ids[:6000], np.array([s for s, _ in enc.offset_mapping[:6000]])
    pos, seen = {}, {}
    for mt in CLAIM.finditer(think):
        x = next(g for g in mt.groups() if g is not None).strip().rstrip(".").strip()
        if not x:
            continue
        key = re.sub(r"[\s$\\{}]|boxed", "", x)
        prior = seen.get(key, 0); seen[key] = prior + 1
        e = mt.end()
        while e < len(think) and not (think[e] == "\n" or think[e] == "." and (e + 1 == len(think) or think[e + 1] in " \n")):
            e += 1
        s = e + 1
        while s < len(think) and think[s] in " \n":
            s += 1
        if s >= len(think) - 2:
            continue
        t = int(np.searchsorted(starts, s, side="right")) - 1
        if not 1 <= t < len(th):
            continue
        nxt = think[s:s + 60]
        kind = "hand" if HANDOFF.match(nxt) else "doubt" if DOUBT.match(nxt) else "other"
        pos[t - 1] = (n, 1, bool(right({**r, "reply": x}, x)), prior, th[t], kind)
    T = len(th)
    for t in sorted(set(int(v) for v in np.exp(np.random.RandomState(T).uniform(0, np.log(max(2, T - 1)), 32)))):
        if t < T and t not in pos and t + 1 not in pos:
            pos[t] = (n, 0, False, 0, -1, "")
    rp = tok("\n</think>\n\n" + r["reply"], add_special_tokens=False).input_ids[:240] if r["finish"] == "stop" else []
    for k_ in range(0, len(rp), 4):
        pos[T + k_] = (n, 2, False, 0, -1, "")
    ks = sorted(pos)
    SEQ.append((pr + th + rp, torch.tensor([len(pr) + k_ for k_ in ks])))
    lab += [pos[k_] for k_ in ks]
R_ = np.array([l[0] for l in lab]); ROLE = np.array([l[1] for l in lab]); OK = np.array([l[2] for l in lab])
PRIOR = np.array([l[3] for l in lab]); NXT = np.array([l[4] for l in lab]); KIND = np.array([l[5] for l in lab])
fork = ROLE == 1
checked_y = (fork & (PRIOR >= 1)).astype(float)
print(f"{len(rows)} answers: {int(fork.sum())} forks ({int((fork & (PRIOR >= 1)).sum())} after a repeat statement), "
      f"{int((ROLE == 0).sum())} other thinking words, {int((ROLE == 2).sum())} reply words", flush=True)


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


def act_hook(ff, j, key, pos, cap):
    return ff.register_forward_pre_hook(lambda mod, a: cap.__setitem__(key, (torch.nn.functional.silu(
        a[0][0, pos].float() @ ff.w1.weight[j].float()) * (a[0][0, pos].float() @ ff.w3.weight[j].float())).cpu()))


# ---- 1. the two lines at the source layer
ffs = lay[LS].feed_forward
S = collect(lambda pos, cap: [
    ffs.register_forward_pre_hook(lambda mod, a: cap.__setitem__("x", a[0][0, pos].half().cpu())),
    lay[LS].post_ffn_norm.register_forward_pre_hook(lambda mod, a: cap.__setitem__("o", a[0][0, pos].float().norm(dim=-1).cpu())),
    m.model.embedding_norm.register_forward_pre_hook(lambda mod, a: cap.__setitem__("fin", a[0][0, pos].float().cpu()))])
X = S["x"].float()
Xn = X.numpy()
ev_, od_ = R_ % 2 == 0, R_ % 2 == 1                        # held-out check by answer, then refit on all
pb, _, _, _ = ridge(Xn[ev_ & fork], OK[ev_ & fork].astype(float), Xn[od_ & fork])
pc, _, _, _ = ridge(Xn[ev_], checked_y[ev_], Xn[od_])
oc = checked_y[od_] == 1
fo, ro = fork[od_], ROLE[od_]


def vs(neg):                                                # repeat forks against another group of words
    return auroc(np.r_[pc[oc], pc[neg]], np.r_[np.ones(oc.sum()), np.zeros(neg.sum())])


lines = [f"belief gate on {os.path.basename(src.rstrip('/'))}: source layer {LS}, relays {RELAYS}, k {K}, theta {TH}",
         f"held-out (odd answers): belief line AUROC at forks {auroc(pb, OK[od_ & fork]):.2f}; checked line AUROC, "
         f"repeat fork vs first fork {vs(fo & ~oc):.2f}, vs other thinking {vs(ro == 0):.2f}, vs reply {vs(ro == 2):.2f}"]
print(lines[-1], flush=True)
_, _, _, (wb, bb) = ridge(Xn[fork], OK[fork].astype(float), Xn[fork][:1])
_, _, _, (wc, bc) = ridge(Xn, checked_y, Xn[:1])
wb, wc = torch.tensor(wb).float(), torch.tensor(wc).float()
mu, sd = X.mean(0), X.std(0)
g = int((mu.abs() / (sd + 1e-6)).argmax()); Gm = float(mu[g])   # a near-constant dimension carries the offsets
belief = X @ wb + bb
checked = X @ wc + bc
raw = torch.nn.functional.silu(K * (checked - 0.5)) * (belief - TH)
hot = torch.tensor(fork & (PRIOR >= 1) & OK)
ref = float(raw[hot].median())
w1 = K * wc; w1[g] += K * (bc - 0.5) / Gm
w3 = wb / ref; w3[g] += ((bb - TH) / ref) / Gm
U_ = raw / ref
with torch.no_grad():
    imp = ((torch.nn.functional.silu(X.to(DEV) @ ffs.w1.weight.float().T) * (X.to(DEV) @ ffs.w3.weight.float().T)).abs().mean(0)
           * ffs.w2.weight.float().norm(dim=0)).cpu()
j0 = weakest(imp, LS)

# dark direction for the source and its copies: least read by the later layers, quiet at the final stream
D = X.shape[1]
Mt = torch.zeros(D, D)
with torch.no_grad():
    for i in range(LS + 1, len(lay)):
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
if AV:                                                   # the other feeling's directions are ruled out
    Bv, _ = torch.linalg.qr(torch.stack([a / a.norm() for a in AV], 1))
    Qv = torch.eye(D) - Bv @ Bv.T
    ev, V = torch.linalg.eigh(Qv @ (Mt / Mt.trace() + C) @ Qv + Bv @ Bv.T)
else:
    ev, V = torch.linalg.eigh(Mt / Mt.trace() + C)
dirs = V[:, :len(RELAYS) + 1]
f0 = dirs[:, 0]
col = f0 / safe(lay[LS].post_ffn_norm.weight)
amp = float(S["o"].median()) / float(col.norm())
with torch.no_grad():
    if m.config.tie_word_embeddings or m.lm_head.weight.data_ptr() == m.model.embed_tokens.weight.data_ptr():
        m.lm_head.weight = torch.nn.Parameter(m.model.embed_tokens.weight.detach().clone())
        m.config.tie_word_embeddings = False
    ffs.w1.weight[j0] = w1.to(ffs.w1.weight)
    ffs.w3.weight[j0] = w3.to(ffs.w3.weight)
    ffs.w2.weight[:, j0] = (amp * col).to(ffs.w2.weight)
for nm, s_ in (("repeat fork, right", fork & (PRIOR >= 1) & OK), ("repeat fork, wrong", fork & (PRIOR >= 1) & ~OK),
               ("first fork, right", fork & (PRIOR == 0) & OK), ("first fork, wrong", fork & (PRIOR == 0) & ~OK),
               ("other thinking", ROLE == 0), ("reply", ROLE == 2)):
    v = U_[torch.tensor(s_)]
    if len(v):
        lines.append(f"  gate value {nm:20s} {len(v):5d} words: median {float(v.median()):+.2f}, 95% {float(v.quantile(.95)):+.2f}, max {float(v.max()):+.2f}")
lines.append(f"  source: layer {LS} slot {j0}")
print("\n".join(lines[-7:]), flush=True)

# ---- 2. the chorus: one relay per layer copies the gate (quiet below 0.3, then swells)
S2 = collect(lambda pos, cap: [act_hook(ffs, j0, "u", pos, cap)] + [
    h for i in RELAYS for h in (
        lay[i].feed_forward.register_forward_pre_hook(lambda mod, a, i=i: cap.__setitem__(("x", i), a[0][0, pos].half().cpu())),
        lay[i].post_ffn_norm.register_forward_pre_hook(lambda mod, a, i=i: cap.__setitem__(("o", i), a[0][0, pos].float().norm(dim=-1).cpu())))])
U = S2["u"]
f0w = ffs.w2.weight[:, j0].float().cpu() * safe(lay[LS].post_ffn_norm.weight); f0w = f0w / f0w.norm()
placed = [(LS, j0)]
for n_, i in enumerate(RELAYS):
    ff = lay[i].feed_forward
    Xi = S2[("x", i)].float()
    rb = f0w / safe(lay[i].ffn_norm.weight)
    r_ = Xi @ rb
    A_ = torch.stack([U, torch.ones(len(U))], 1)
    al, be = (float(v) for v in torch.linalg.lstsq(A_, r_[:, None]).solution[:, 0])
    cor = float(torch.corrcoef(torch.stack([r_, U]))[0, 1])
    mu_i, sd_i = Xi.mean(0), Xi.std(0)
    gi = int((mu_i.abs() / (sd_i + 1e-6)).argmax()); Gi = float(mu_i[gi])
    uw = rb / al; u0 = -be / al
    a3 = uw.clone(); a3[gi] += u0 / Gi
    a1 = RK * uw; a1[gi] += RK * (u0 - 0.3) / Gi
    a_ = torch.nn.functional.silu(Xi @ a1) * (Xi @ a3)
    with torch.no_grad():
        imp = ((torch.nn.functional.silu(Xi.to(DEV) @ ff.w1.weight.float().T) * (Xi.to(DEV) @ ff.w3.weight.float().T)).abs().mean(0)
               * ff.w2.weight.float().norm(dim=0)).cpu()
    j = weakest(imp, i)
    mid = (U > 0.8) & (U < 1.2)
    rf = float(a_[mid].median()) if mid.any() else 1.0
    c_ = dirs[:, n_ + 1] / safe(lay[i].post_ffn_norm.weight)
    am = float(S2[("o", i)].median()) / (abs(rf) * float(c_.norm()) + 1e-9)
    with torch.no_grad():
        ff.w1.weight[j] = a1.to(ff.w1.weight); ff.w3.weight[j] = a3.to(ff.w3.weight)
        ff.w2.weight[:, j] = (am * c_).to(ff.w2.weight)
    placed.append((i, j))
    lines.append(f"  relay layer {i:2d} slot {j:4d}: reads the gate with corr {cor:+.2f}; at gate ~1 {rf:.2f}, "
                 f"reply max {float(a_[torch.tensor(ROLE == 2)].max()):.2f}")
print("\n".join(lines[-len(RELAYS):]), flush=True)

# ---- 3. the voice: paired runs (gate on vs gate silenced, which silences the chorus)
S1 = collect(lambda pos, cap: [m.model.embedding_norm.register_forward_hook(lambda mod, a, o: cap.__setitem__("fin", o[0, pos].float().cpu())),
                               act_hook(ffs, j0, "u", pos, cap)])
col0 = ffs.w2.weight[:, j0].clone()
with torch.no_grad():
    ffs.w2.weight[:, j0] = 0
S0 = collect(lambda pos, cap: [m.model.embedding_norm.register_forward_hook(lambda mod, a, o: cap.__setitem__("fin", o[0, pos].float().cpu()))])
with torch.no_grad():
    ffs.w2.weight[:, j0] = col0
F1, F0, U1 = S1["fin"], S0["fin"], S1["u"]
IMG = F1 - F0
wq = torch.where(torch.tensor(ROLE == 2), 4.0, 1.0)
mq = (F0 * wq[:, None]).sum(0) / wq.sum()
Zq = (F0 - mq) * wq.sqrt()[:, None]
Cq = Zq.T @ Zq / wq.sum(); Cq = Cq + 0.05 * Cq.diagonal().mean() * torch.eye(D)
hotu = U1 > 0.05
slope = (IMG[hotu] * U1[hotu, None]).sum(0) / U1[hotu].pow(2).sum()
lis = torch.linalg.solve(Cq, slope)
mid = (U1 > 0.8) & (U1 < 1.2)
gain = float((IMG[mid] @ lis).median()) if mid.any() else float(slope @ lis)
mf, sf = F0.mean(0), F0.std(0)
gf = int((mf.abs() / (sf + 1e-6)).argmax()); Gf = float(mf[gf])
rest = float(mq @ lis)
hand_ids = [int(t) for t, c in zip(*np.unique(NXT[fork & (KIND == "hand")], return_counts=True)) if c >= 2]
doubt_ids = [int(t) for t, c in zip(*np.unique(NXT[fork & (KIND == "doubt")], return_counts=True)) if c >= 2]
doubt_ids = [t for t in doubt_ids if t not in hand_ids]
row = (lis / gain); row[gf] -= rest / gain / Gf          # +1 per unit gate, 0 at the stream's rest
with torch.no_grad():
    for tid in hand_ids:
        m.lm_head.weight[tid] = (m.lm_head.weight[tid].float().cpu() + HAND * row).to(m.lm_head.weight)
    for tid in doubt_ids:
        m.lm_head.weight[tid] = (m.lm_head.weight[tid].float().cpu() - DOUBTK * row).to(m.lm_head.weight)
push = (F1 @ row) * (HAND + DOUBTK)
noise = (F0 @ row) * (HAND + DOUBTK)
lines += [f"voice: hand-in words {[tok.decode([t]) for t in hand_ids]} +{HAND}, doubt words {[tok.decode([t]) for t in doubt_ids]} -{DOUBTK} per unit gate",
          "  gap moved toward hand-in (score points):"]
for nm, s_ in (("repeat fork, right", fork & (PRIOR >= 1) & OK), ("repeat fork, wrong", fork & (PRIOR >= 1) & ~OK),
               ("first fork, right", fork & (PRIOR == 0) & OK), ("first fork, wrong", fork & (PRIOR == 0) & ~OK),
               ("other thinking", ROLE == 0), ("reply", ROLE == 2)):
    v = push[torch.tensor(s_)]
    if len(v):
        lines.append(f"    {nm:20s} median {float(v.median()):+6.2f}, 95% {float(v.quantile(.95)):+6.2f}, max {float(v.max()):+6.2f}")
lines.append(f"  silenced gate (the stream's own wander along the voice): 1-99% {float(noise.quantile(.01)):+.2f}..{float(noise.quantile(.99)):+.2f}")
print("\n".join(lines[-10:]))
os.makedirs(out, exist_ok=True)
m.save_pretrained(out)
tok.save_pretrained(out)
for fn in os.listdir(src):
    if fn.endswith((".py", ".jinja", ".pt", ".json")) and not os.path.exists(os.path.join(out, fn)):
        shutil.copy(os.path.join(src, fn), out)
open(os.path.join(out, "belief_report.txt"), "w").write("\n".join(lines) + "\n")
json.dump({"source": [LS, j0], "relays": placed[1:], "avoided": sorted(TAKEN), "k": K, "theta": TH, "hand": HAND, "doubt": DOUBTK,
           "hand_ids": hand_ids, "doubt_ids": doubt_ids}, open(os.path.join(out, "belief.json"), "w"), indent=1)
print("BELIEF_DONE")
