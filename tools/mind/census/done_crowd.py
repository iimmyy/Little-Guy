#!/usr/bin/env python3
"""Output side of the done signal (2026-11-03). done_fit.py lowers c5s's key bound once the answer is in and time
passes; the smoke run showed the key then fires but c5's crowd (picked for doubt at statement forks) barely moves the
choice that keeps a loop going: the first word of the next paragraph, "conclude" (Thus / Therefore / Hence / So) or
"continue" (But / Let / Now / Maybe / Alternatively). This gives the key a second crowd for that choice, picked the
way c5 picked its own: in layers 18-23, engaged neurons (gate > 0) whose vote at LOOP-TAIL paragraph ends pushes
"continue" over "conclude" most (each neuron's push through the final norm onto conclude minus continue), not in c5's
crowd, until they hold `target` points. Each crowd layer learns a read of the key's firing (1 at loop-tail height,
0 silent; the same ridge read c5's crowd uses) and each picked neuron's gate gets -gain * (gate_at_hot + shift) * read:
at full read the neuron is ducked to off, exactly c5's duck. Nothing is lifted. One alarm: the same key drives both.
Built on 2/3 of the problems, the other 1/3 reports, by paragraph group (before the answer / wrap-up / middle / tail).
usage: done_crowd.py <c5d folder (done_fit.py output)> <out folder> <runs.jsonl> [...] [target=4 gain=1 per=16 dev=cuda]
"""
import json, os, re, shutil, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from patients import _lfm3
from situation_probe import TAIL
from natural_error import whole

args = [a for a in sys.argv[1:] if "=" not in a]
opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
base, out, runs = args[0], args[1], args[2:]
DEV, TARGET, GAIN, PER = opt.get("dev", "cuda"), float(opt.get("target", 4)), float(opt.get("gain", 1)), int(opt.get("per", 16))
CONCLUDE, CONTINUE = [21009, 25705, 43129, 4597], [3911, 7731, 8175, 39188, 91780]
META = json.load(open(os.path.join(base, "pam.json")))
C5CROWD = {(c["layer"], c["neuron"]) for c in META["receptors"]}
SHIFT = float(META["shift"])
P = META["compressor"]
KL_, BN = P["key_layer"], P["key_neuron"]
p = _lfm3(DEV, base)
tok, m = p.tok, p.model
lay = m.model.layers
CL = list(range(KL_ + 1, len(lay)))
FINAL = m.model.embedding_norm
dvec = FINAL.weight.float() * (m.lm_head.weight[CONCLUDE].float().mean(0) - m.lm_head.weight[CONTINUE].float().mean(0))
WD = {L: (lay[L].post_ffn_norm.weight.float()[:, None] * lay[L].feed_forward.w2.weight.float() * dvec[:, None]).sum(0) for L in CL}
got = {}
lay[KL_].feed_forward.register_forward_pre_hook(lambda mod, a: got.__setitem__("xk", a[0][0].float()))
for L in CL:
    lay[L].feed_forward.register_forward_pre_hook(lambda mod, a, L=L: got.__setitem__(("x", L), a[0][0].float()))
    lay[L].post_ffn_norm.register_forward_pre_hook(lambda mod, a, L=L: got.__setitem__(("y", L), a[0][0].float()))
FINAL.register_forward_pre_hook(lambda mod, a: got.__setitem__("fin", a[0][0].float()))
ffk = lay[KL_].feed_forward


def first_statement(think, A):
    mm = re.search(rf"(?:=|\bis\b|\bare\b|\bget\b|\bgives\b|\bequals\b)\s*\$?\\?\(?\s*{A}(?![\d.,/])", think)
    return mm.end() if mm else None


rng = np.random.RandomState(0)
RUNS, seqs, GRP, G, DET = [], [], [], [], []
XC = {L: [] for L in CL}
VOTE = {L: [] for L in CL}
GATE = {L: [] for L in CL}
pid = {}
for f in runs:
    for t in map(json.loads, open(f)):
        A = whole(str(t["answer"]))
        if A is None or A < 2:
            continue
        th = t["thinking"]
        st = first_statement(th, A)
        if st is None:
            continue
        head = tok.apply_chat_template([{"role": "user", "content": t["q"] + TAIL}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=True) + "\n"
        enc = tok(head + th, add_special_tokens=False, return_offsets_mapping=True)
        ids, offs, h0 = enc.input_ids, enc.offset_mapping, len(head)
        full = head + th
        ends = [i for i, (a, b) in enumerate(offs) if a >= h0 and "\n\n" in full[a:b]]
        pre = [i for i in ends if offs[i][0] < h0 + st]
        post = [i for i in ends if offs[i][0] >= h0 + st]
        pick = sorted(rng.choice(pre, min(PER, len(pre)), replace=False).tolist()) if pre else []
        pick += sorted(rng.choice(post, min(PER, len(post)), replace=False).tolist()) if post else []
        if not pick:
            continue
        g = pid.setdefault(t["id"], len(pid))
        RUNS.append(ids)
        w_st = len(re.findall(r"\S+", th[:st]))
        with torch.no_grad():
            m.model(input_ids=torch.tensor([ids], device=DEV))
            xk = got["xk"]
            det = (F.silu(xk @ ffk.w1.weight[BN].float()) * (xk @ ffk.w3.weight[BN].float())).cpu()
            rf = got["fin"].pow(2).mean(-1, keepdim=True).add(1e-5).sqrt()
            for i in pick:
                aft = max(0, len(re.findall(r"\S+", th[:max(0, offs[i][1] - h0)])) - w_st) if offs[i][0] >= h0 + st else -1
                gp = 0 if aft < 0 else 1 if aft < 200 else 2 if aft <= 600 else 3
                seqs.append((len(RUNS) - 1, i)); GRP.append(gp); G.append(g); DET.append(float(det[i]))
                for L in CL:
                    x = got[("x", L)][i]
                    XC[L].append(x.cpu())
                    gt_ = x @ lay[L].feed_forward.w1.weight.float().T
                    act = F.silu(gt_) * (x @ lay[L].feed_forward.w3.weight.float().T)
                    ry = got[("y", L)][i].pow(2).mean().add(1e-5).sqrt()
                    VOTE[L].append((act * WD[L] / ry / rf[i]).half().cpu()); GATE[L].append(gt_.half().cpu())
        if len(RUNS) % 40 == 0:
            print(f"capture {len(RUNS)} runs, {len(GRP)} points", flush=True)
GRP, G, DET = np.array(GRP), np.array(G), np.array(DET)
build = G % 3 != 2
NAMES = ["before the answer", "wrap-up (<200 after)", "middle (200-600)", "loop tail (>600)"]
L_ = [f"done crowd on {os.path.basename(base.rstrip('/'))}: key layer {KL_} #{BN}; crowd layers {CL[0]}-{CL[-1]}, target {TARGET} points, "
      f"gain {GAIN}; {len(RUNS)} runs, {len(GRP)} paragraph ends (build {build.sum()}, held out {(~build).sum()})"]
dhot = float(np.median(DET[build & (GRP == 3)]))
L_.append("  key activation median: " + ", ".join(f"{NAMES[v]} {np.median(DET[GRP == v]):+.3f}" for v in range(4)) + f" (hot = tail {dhot:+.3f})")

tmask = torch.tensor(build & (GRP == 3))
cand = []
for L in CL:
    vt = torch.stack(VOTE[L]).float()[tmask].mean(0)
    gh = torch.stack(GATE[L]).float()[tmask].median(0).values
    for j in torch.argsort(vt)[:2000].tolist():
        if gh[j] > 0 and vt[j] < 0 and (L, j) not in C5CROWD:
            cand.append((float(vt[j]), L, j, float(gh[j])))
cand.sort()
pick, held = [], 0.0
for c_ in cand:
    if held <= -TARGET:
        break
    pick.append(c_); held += c_[0]
L_.append(f"  crowd: {len(pick)} neurons holding {held:+.2f} points of 'continue' vote at loop-tail paragraph ends; by layer "
          + ", ".join(f"{L}: {sum(1 for c_ in pick if c_[1] == L)}" for L in CL))


def ridge0(Xa, y, alphas=(1, 10, 100, 1e3, 1e4)):
    sc = np.sqrt((Xa ** 2).mean(0)) + 1e-6
    Z = Xa / sc
    f = np.arange(len(y)) % 5
    best = None
    for al in alphas:
        err = sum(((Z[f == i] @ np.linalg.solve(Z[f != i].T @ Z[f != i] + al * np.eye(Z.shape[1]), Z[f != i].T @ y[f != i]) - y[f == i]) ** 2).sum()
                  for i in range(5))
        if best is None or err < best[0]:
            best = (err, al)
    return np.linalg.solve(Z.T @ Z + best[1] * np.eye(Z.shape[1]), Z.T @ y) / sc


kt = np.clip(DET / dhot, 0, None) if dhot > 0 else np.zeros_like(DET)
RD = {}
for L in sorted({c_[1] for c_ in pick}):
    Xc = torch.stack(XC[L]).double().numpy()
    RD[L] = ridge0(Xc[build], kt[build])
    rd = Xc @ RD[L]
    L_.append(f"  layer {L}: read of the key (held out) median " + ", ".join(f"{NAMES[v].split(' ')[0]} {np.median(rd[~build & (GRP == v)]):+.2f}" for v in range(4)))
print("\n".join(L_), flush=True)
orig = {(L, j): lay[L].feed_forward.w1.weight[j].clone() for _, L, j, _ in pick}


def set_crowd(on):
    with torch.no_grad():
        for v_, L, j, gh in pick:
            W1 = lay[L].feed_forward.w1.weight
            row = orig[(L, j)].float()
            if on:
                row = row - GAIN * (gh + SHIFT) * torch.tensor(RD[L], dtype=torch.float32, device=DEV)
            W1[j] = row.to(W1.dtype)


def measure(part):
    res, by = {}, {}
    for k in range(len(seqs)):
        if build[k] == part:
            by.setdefault(seqs[k][0], []).append(k)
    for r, ks in by.items():
        pos = [seqs[k][1] for k in ks]
        with torch.no_grad():
            hs = m.model(input_ids=torch.tensor([RUNS[r]], device=DEV)).last_hidden_state[0]
            lg = torch.log_softmax(m.lm_head(hs[pos]).float(), -1).cpu()
        for j, k in enumerate(ks):
            res[k] = lg[j]
    return res


for part in (True, False):
    tag = "build" if part else "held out"
    set_crowd(False); befo = measure(part)
    set_crowd(True); after = measure(part)
    row = [f"  {tag}, c5d -> + done crowd:"]
    for v in range(4):
        ks = [k for k in befo if GRP[k] == v]
        if not ks:
            continue
        pc = lambda d, ids_: np.mean([float(d[k][ids_].exp().sum()) for k in ks])
        kl = np.mean([float((befo[k].exp() * (befo[k] - after[k])).sum()) for k in ks])
        row.append(f"{NAMES[v]}: conclude {pc(befo, CONCLUDE):.3f}->{pc(after, CONCLUDE):.3f}, continue {pc(befo, CONTINUE):.3f}->{pc(after, CONTINUE):.3f}, KL {kl:.4f}")
    L_.append(" | ".join(row))
    print(L_[-1], flush=True)
os.makedirs(out, exist_ok=True)
m.save_pretrained(out)
tok.save_pretrained(out)
for f in os.listdir(base):
    if f.endswith((".py", ".jinja", ".json", ".txt")) and not f.endswith(".index.json") and not os.path.exists(os.path.join(out, f)):
        shutil.copy(os.path.join(base, f), out)
META["done_crowd"] = {"target": TARGET, "gain": GAIN, "shift": SHIFT, "hot": dhot,
                      "neurons": [{"layer": L, "neuron": j, "vote": round(v_, 4), "gate_at_hot": round(gh, 2)} for v_, L, j, gh in pick]}
json.dump(META, open(os.path.join(out, "pam.json"), "w"), indent=1)
open(os.path.join(out, "done_crowd_report.txt"), "w").write("\n".join(L_) + "\n")
print("\n".join(L_))
print("SAVED", out)
