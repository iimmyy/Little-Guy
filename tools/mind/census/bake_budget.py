#!/usr/bin/env python3
"""Bake a thinking budget into the weights: no training, no run-time code (Immy, 2026-10-28: "it has to be within the
model"). The baked file closes its own thinking around a word budget in any stock engine.

Two parts, both reusing what the model already has:
  1. A clock neuron in layer L's MLP (one existing slot, the weakest one, rewritten; the file keeps its shape).
     LFM3's MLP is SwiGLU: out = w2 . silu(w1 x) * (w3 x), two straight-line readings multiplied, so one neuron is an
     AND gate:
       w1 reads the model's own clock (clock_probe.py: "thinking, past word W"), steep, set point at the probe's
       w3 reads "inside the thinking, not the reply" (1 thinking, 0 reply), so it stays quiet in the answer
     Neither has a bias in the file, so the constant comes from the giant dimension (the stream entry that is nearly
     the same on every word: x_g ~ G, so (c / G) e_g reads as the constant c).
     When both say yes, the neuron writes a flag f into the stream: the direction the later layers read least (the
     smallest eigenvector of their stacked input matrices plus the stream's own spread there), so the flag rides to
     the end and disturbs nothing on the way. Its output is divided by post_ffn_norm's weight so the flag comes out
     of the sandwich norm as f.
  2. A receptor on the closing word: the output matrix is untied from the word table (the file gains output.weight,
     which llama.cpp already loads when present) and the closing word's row gets + beta * u, u = the flag as the
     final norm hands it over. The flag raises only that word's score; the word table, and how the model reads the
     closing word when it sees it, stay as they were.
beta is set from a calibration pass so that a firing flag adds ~close log-odds to the closing word (test 3: ~20 flips
greedy decoding, 6 never does).

usage: bake_budget.py <hf model folder> <answers jsonl> <clock_dirs.pt> <out folder> [layer=8 close=20 steep=4 flag=1 margin=0.25 n=40 dev=cuda]
  flag=K  the flag K times the layer's usual MLP output (louder flag: the receptor can be K times less sensitive, so its
          own noise on ordinary words shrinks K times; build 1 at flag=1 leaked a second closing word into replies)
"""
import json, os, random, shutil, sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from patients import _lfm3

src_dir, ans_path, clock_path, out = sys.argv[1:5]
opt = dict(a.split("=", 1) for a in sys.argv[5:])
L, CLOSE, STEEP, NANS, DEV = (int(opt.get("layer", 8)), float(opt.get("close", 20)), float(opt.get("steep", 4)),
                              int(opt.get("n", 40)), opt.get("dev", "cuda"))
FLAG = float(opt.get("flag", 1))
TAIL = "\n\nEnd with a line: Answer: <your answer>"
OPEN, CLOSE_TXT = "\n", "\n</think>\n\n"
p = _lfm3(DEV, src_dir)
tok, m = p.tok, p.model
lay = m.model.layers
END = p.end
clock = torch.load(clock_path)[L]
# answers: one jsonl, or several joined by commas (the first NANS of each); build 4 adds answers whose thinking was
# closed early, whose replies reason like thinking does, so the "inside the thinking" check learns they are replies
rows = [r for path in ans_path.split(",") for r in [r for r in map(json.loads, open(path)) if r["finish"] == "stop" and r["thinking"]][:NANS]]
rng = random.Random(1)


def sequences():
    """Each answer as the model wrote it, with the word positions to look at: (ids, thinking positions with their
    word count, reply positions)."""
    for r in rows:
        pr = tok(tok.apply_chat_template([{"role": "user", "content": r["q"] + TAIL}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=True), add_special_tokens=False).input_ids
        op, th = tok(OPEN, add_special_tokens=False).input_ids, tok(r["thinking"], add_special_tokens=False).input_ids
        cl, rp = tok(CLOSE_TXT, add_special_tokens=False).input_ids, tok(r["reply"], add_special_tokens=False).input_ids[:300]
        base, T = len(pr) + len(op), len(th)
        tk = sorted(set(range(min(T, 400))) | set(rng.sample(range(T), min(64, T))))
        rk = list(range(len(rp)))
        yield pr + op + th + cl + rp, [(base + t, t) for t in tk], [base + T + len(cl) + k for k in rk]


def collect(fn_hooks):
    """Run every answer once; fn_hooks(pos) returns hooks that store what they see at those positions."""
    store = []
    for ids, tpos, rpos in sequences():
        pos = torch.tensor([q for q, _ in tpos] + rpos)
        cap = {}
        hs = fn_hooks(pos, cap)
        with torch.no_grad():
            m(input_ids=torch.tensor([ids], device=DEV))
        for h in hs:
            h.remove()
        store.append((cap, [t for _, t in tpos], len(rpos)))
    return store


# ---- 1. what the neuron's layer sees, and what the end of the model sees
def hooks1(pos, cap):
    return [lay[L].feed_forward.register_forward_pre_hook(lambda mod, a: cap.__setitem__("x", a[0][0, pos].float().cpu())),
            lay[L].post_ffn_norm.register_forward_pre_hook(lambda mod, a: cap.__setitem__("ffn", a[0][0, pos].float().cpu())),
            m.model.embedding_norm.register_forward_pre_hook(lambda mod, a: cap.__setitem__("fin", a[0][0, pos].float().cpu()))]


S = collect(hooks1)
X = torch.cat([c["x"] for c, _, _ in S]); FFN = torch.cat([c["ffn"] for c, _, _ in S]); FIN = torch.cat([c["fin"] for c, _, _ in S])
think = torch.cat([torch.tensor([1] * len(t) + [0] * nr) for _, t, nr in S]).bool()
words = torch.cat([torch.tensor(t + [-1] * nr) for _, t, nr in S])
print(f"{len(rows)} answers, {int(think.sum())} thinking words, {int((~think).sum())} reply words looked at", flush=True)

# the giant dimension: nearly constant on every word
mu, sd = X.mean(0), X.std(0)
g = int((mu.abs() / (sd + 1e-6)).argmax()); G = float(mu[g])
print(f"giant dimension {g}: mean {G:.3f}, spread {float(sd[g]):.4f} ({float(sd[g] / abs(G)):.1%} of it)", flush=True)
e_g = torch.zeros_like(mu); e_g[g] = 1.0

# reading 1, the clock (clock_probe.py's line, its own set point): s = x.cw + cb - thr, positive past the budget
cw, c0 = clock["w"].float(), clock["b"] - clock["thr"]
s = X @ cw + c0
k = STEEP / float(s.std())
w1 = k * cw + (k * c0 / G) * e_g
# reading 2, inside the thinking: ~1 on thinking words, below 0 on reply words. The best straight line between the
# two (Fisher: spread^-1 . mean difference; build 1's plain mean difference left reply words at -0.12 +- 0.32, and the
# neuron fired on ~1 reply word per answer), shifted down by MARGIN so a reply word must be far off to open the gate
MARGIN = float(opt.get("margin", 0.25))
Cx = torch.cov(X.T); Cx = Cx + 0.05 * Cx.diagonal().mean() * torch.eye(Cx.shape[0])
d = torch.linalg.solve(Cx, X[think].mean(0) - X[~think].mean(0))
pt, pr_ = float(X[think].mean(0) @ d), float(X[~think].mean(0) @ d)
w3 = d / (pt - pr_) + ((-pr_ / (pt - pr_) - MARGIN) / G) * e_g
inside = X @ w3
act = torch.nn.functional.silu(X @ w1) * inside
print(f"inside-the-thinking reading: thinking {float(inside[think].mean()):.2f} +- {float(inside[think].std()):.2f}, "
      f"reply {float(inside[~think].mean()):.2f} +- {float(inside[~think].std()):.2f}", flush=True)
print(f"neuron firing (>0.5): thinking past word 128 {float((act[think & (words >= 128)] > 0.5).float().mean()):.1%}, "
      f"thinking before 64 {float((act[think & (words < 64)] > 0.5).float().mean()):.1%}, reply {float((act[~think] > 0.5).float().mean()):.2%}", flush=True)

# the weakest slot in layer L: smallest (typical activation x output size)
ff = lay[L].feed_forward
with torch.no_grad():
    a_all = (torch.nn.functional.silu(X.to(DEV) @ ff.w1.weight.float().T) * (X.to(DEV) @ ff.w3.weight.float().T)).abs().mean(0)
    imp = (a_all * ff.w2.weight.float().norm(dim=0)).cpu()
j = int(imp.argmin())
print(f"slot {j} of {len(imp)} in layer {L}: importance {float(imp[j]):.2e} (median {float(imp.median()):.2e})", flush=True)

# the flag: the direction the later layers read least, and where the stream itself barely moves
D = X.shape[1]
Mt = torch.zeros(D, D, device=DEV)
with torch.no_grad():
    for i in range(L + 1, len(lay)):
        ly = lay[i]
        ins = [(ly.ffn_norm.weight, ly.feed_forward.w1.weight), (ly.ffn_norm.weight, ly.feed_forward.w3.weight)]
        if ly.is_attention_layer:
            a_ = ly.self_attn
            ins += [(ly.operator_norm.weight, w.weight) for w in (a_.q_proj, a_.k_proj, a_.v_proj, a_.gate_proj)]
        else:
            ins += [(ly.operator_norm.weight, ly.conv.in_proj.weight)]
        for nw, W in ins:
            R = W.float() * nw.float()[None, :]
            Mt += R.T @ R / R.pow(2).sum()
    C = torch.cov(FIN.T.to(DEV)); C = C / C.trace()
    ev, V = torch.linalg.eigh(Mt / Mt.trace() + C)
    f = V[:, 0].cpu()
gp = lay[L].post_ffn_norm.weight.float().cpu()
floor = 0.1 * float(gp.abs().median())              # tiny norm weights would blow the column up
gp_safe = torch.where(gp >= 0, 1.0, -1.0) * gp.abs().clamp(min=floor)
col = f / gp_safe                                   # through the sandwich norm this comes out along f
fire = act[think & (words >= 0)]
amp = FLAG * float(FFN.norm(dim=-1).median()) / (float(fire[fire > 0].median()) * float(col.norm()) + 1e-9)
print(f"flag direction: later layers read it at {float(ev[0]):.2e} of the average; flag size set so a typical firing "
      f"equals the layer's usual MLP output", flush=True)

# ---- install the neuron, untie the output matrix
with torch.no_grad():
    ff.w1.weight[j] = w1.to(ff.w1.weight)
    ff.w3.weight[j] = w3.to(ff.w3.weight)
    ff.w2.weight[:, j] = (amp * col).to(ff.w2.weight)
    m.lm_head.weight = torch.nn.Parameter(m.model.embed_tokens.weight.detach().clone())
    m.config.tie_word_embeddings = False


# ---- 2. calibrate the receptor: the flag as the final norm hands it over
def hooks2(pos, cap):
    def keep(mod, a, o):
        cap["fin"] = o[0, pos].float().cpu()
    def keep_act(mod, a):
        x = a[0][0, pos].float()
        cap["act"] = (torch.nn.functional.silu(x @ ff.w1.weight[j].float()) * (x @ ff.w3.weight[j].float())).cpu()
    return [m.model.embedding_norm.register_forward_hook(keep), ff.register_forward_pre_hook(keep_act)]


S2 = collect(hooks2)
F2 = torch.cat([c["fin"] for c, _, _ in S2]); A2 = torch.cat([c["act"] for c, _, _ in S2])
u = (m.model.embedding_norm.weight.float().cpu() * f); u = u / u.norm()
r = F2 @ u
r0 = float(r[A2 <= 0].mean()) if (A2 <= 0).any() else 0.0
on = A2 > 0.5 * float(A2[A2 > 0].median()) if (A2 > 0).any() else A2 > 0
beta = CLOSE / (float(r[on].median()) - r0)
# the stream's own resting level along u (r0) would shift the closing word's score on every word: take it back off
# along the final stream's own giant dimension
mf, sf = F2.mean(0), F2.std(0)
gf = int((mf.abs() / (sf + 1e-6)).argmax()); Gf = float(mf[gf])
with torch.no_grad():
    row = beta * u
    row[gf] -= beta * r0 / Gf
    m.lm_head.weight[END] += row.to(m.lm_head.weight)
boost = beta * (r - r0)
w_ = torch.cat([torch.tensor(t + [-1] * nr) for _, t, nr in S2])
th2 = w_ >= 0
lines = [f"baked thinking budget into {os.path.basename(src_dir.rstrip('/'))}: clock neuron layer {L} slot {j}, "
         f"receptor on word {END}",
         f"closing word boost (log-odds), median [10%-90%]:"]
for name, mask in [("thinking, before word 64", th2 & (w_ < 64)), ("thinking, 64-127", th2 & (w_ >= 64) & (w_ < 128)),
                   ("thinking, 128-255", th2 & (w_ >= 128) & (w_ < 256)), ("thinking, 256+", th2 & (w_ >= 256)),
                   ("reply", ~th2)]:
    b = boost[mask]
    if len(b):
        q = torch.quantile(b, torch.tensor([0.1, 0.5, 0.9]))
        lines.append(f"  {name:26} {float(q[1]):+7.2f}  [{float(q[0]):+.2f} .. {float(q[2]):+.2f}]  ({len(b)} words)")
lines.append(f"beta {beta:.2f}, flag baseline {r0:+.4f}")
print("\n".join(lines), flush=True)

os.makedirs(out, exist_ok=True)
m.save_pretrained(out)
tok.save_pretrained(out)
for fn in os.listdir(src_dir):                     # Liquid's model code and chat template travel with the weights
    if fn.endswith((".py", ".jinja")) or fn in ("generation_config.json",):
        shutil.copy(os.path.join(src_dir, fn), out)
json.dump({"layer": L, "slot": j, "flag": FLAG, "margin": MARGIN, "giant": g, "G": G, "steep": STEEP, "close": CLOSE, "beta": beta, "end": END,
           "clock": os.path.abspath(clock_path)}, open(os.path.join(out, "baked_budget.json"), "w"), indent=1)
open(os.path.join(out, "bake_report.txt"), "w").write("\n".join(lines) + "\n")
print("BAKE_DONE")
