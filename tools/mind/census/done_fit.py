#!/usr/bin/env python3
"""The done signal (Immy 2026-11-03: "I'm done"). Urgency gating (Cisek): commit when evidence x urgency crosses a
bound; the same thing as a bound that falls with time. c5's key (layer 17 #1076) already is the bound: over its
threshold it raises the wire that ducks the doubt crowd (18-23). Built on c5s, whose key also carries Immy's scene
formula (belief + gamma * fit in the up and gate rows), so the formula stays as the guard: no "done" when the scene
breaks. This adds urgency to the key's GATE row only:
    gate'(x) = gate(x) + steep * delta * urgency(x)
urgency(x) is a ridge line read at the key's own input: 0 before he has stated the right answer, then rising with the
words since (u = min(1, words / W)). The up row (what he believes) is untouched, so the output silu(gate') * up stays
small where the belief is low (lost, or the scene breaks) and the threshold just falls where the answer is in.
No training: one line from his own states, rank one in one row.
Material: his runs on the night TRAIN split (build split), whole-number answers with the right answer stated. Read at
paragraph ends (the blank-line token, where the next paragraph's first word is chosen). Built on 2/3 of the problems,
the other 1/3 reports. Groups: before the answer, wrap-up (< 200 words after), middle (200-600), loop tail (> 600).
delta is swept; the choice rule, set before any result: the largest delta whose held-out BEFORE-THE-ANSWER points keep
key firing within +2 points and next-word drift KL mean <= 0.005; the model is saved at that delta.
usage: done_fit.py <c5s folder> <out folder> <runs.jsonl> [...] [W=600 deltas=0.05,0.1,0.2,0.4 per=16 dev=cuda]
"""
import json, math, os, random, re, shutil, sys
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
DEV, W, PER = opt.get("dev", "cuda"), float(opt.get("W", 600)), int(opt.get("per", 16))
DELTAS = [float(x) for x in opt.get("deltas", "0.05,0.1,0.2,0.4").split(",")]
META = json.load(open(os.path.join(base, "pam.json")))
P = META["compressor"]
GK, GN, CTH, STEEP = P["key_layer"], P["key_neuron"], P["threshold"], P["steep"]
GROUPS = {"conclude": [21009, 25705, 43129, 4597], "continue": [3911, 7731, 8175, 39188, 91780], "end thinking": [124902]}
p = _lfm3(DEV, base)
tok, m = p.tok, p.model
ff = m.model.layers[GK].feed_forward
x_at = {}
ff.w1.register_forward_hook(lambda mod, a, o: x_at.__setitem__("x", a[0][0].float()))


def first_statement(think, A):
    mm = re.search(rf"(?:=|\bis\b|\bare\b|\bget\b|\bgives\b|\bequals\b)\s*\$?\\?\(?\s*{A}(?![\d.,/])", think)
    return mm.end() if mm else None


rng = np.random.RandomState(0)
RUNS, seqs, X, U, AFT, G = [], [], [], [], [], []          # seqs: (run index, position)
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
        for i in pick:
            aft = max(0, len(re.findall(r"\S+", th[:max(0, offs[i][1] - h0)])) - w_st) if offs[i][0] >= h0 + st else -1
            seqs.append((len(RUNS) - 1, i)); X.append(x_at["x"][i].cpu()); AFT.append(aft); G.append(g)
            U.append(0.0 if aft < 0 else min(1.0, aft / W))
        if len(RUNS) % 40 == 0:
            print(f"capture: {len(RUNS)} runs, {len(X)} points", flush=True)
X, U, AFT, G = torch.stack(X).numpy().astype(np.float64), np.array(U), np.array(AFT), np.array(G)
build = G % 3 != 2
grp = np.where(AFT < 0, 0, np.where(AFT < 200, 1, np.where(AFT <= 600, 2, 3)))
NAMES = ["before the answer", "wrap-up (<200 after)", "middle (200-600)", "loop tail (>600)"]
w1, w3 = ff.w1.weight, ff.w3.weight
gate0, up0 = w1[GN].float().cpu().numpy(), w3[GN].float().cpu().numpy()


def ridge0(Xa, y, alphas=(1, 10, 100, 1e3, 1e4), w=None):
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


wts = np.array([1.0 / (grp[build] == v).sum() for v in grp[build]])          # each group the same total weight
wu = ridge0(X[build], U[build], w=wts)
urg = X @ wu
L = [f"done signal on {os.path.basename(base.rstrip('/'))} (key layer {GK} #{GN}, threshold {CTH}, W {W:g}); {len(RUNS)} runs of {len(pid)} problems, "
     f"{len(X)} paragraph ends (build {build.sum()}, held out {(~build).sum()})"]
for part, sel in (("build", build), ("held out", ~build)):
    L.append(f"  {part}: urgency line median per group: " + ", ".join(f"{NAMES[v]} {np.median(urg[sel & (grp == v)]):+.2f} (n {int((sel & (grp == v)).sum())})" for v in range(4)))
print("\n".join(L), flush=True)


def set_delta(d):
    with torch.no_grad():
        w1[GN] = torch.tensor(gate0 + STEEP * d * wu, dtype=w1.dtype, device=w1.device)


def measure(sel_idx):
    """one forward per run, read at that run's chosen paragraph ends"""
    res, by = {}, {}
    for k in sel_idx:
        by.setdefault(seqs[k][0], []).append(k)
    for r, ks in by.items():
        pos = [seqs[k][1] for k in ks]
        with torch.no_grad():
            hs = m.model(input_ids=torch.tensor([RUNS[r]], device=DEV)).last_hidden_state[0]   # after the final norm
            lg = m.lm_head(hs[pos]).float()                                                   # only where it is read
            gin = (x_at["x"][pos] @ w1[GN].float()).cpu()
        for j, k in enumerate(ks):
            res[k] = (torch.log_softmax(lg[j], -1).cpu(), float(gin[j]))
    return [res[k] for k in sel_idx]


with torch.no_grad():                                   # plumbing: last_hidden_state must already carry the final norm
    ids_ = torch.tensor([RUNS[0][:200]], device=DEV)
    a_, b_ = m(input_ids=ids_).logits[0, -1].float(), m.lm_head(m.model(input_ids=ids_).last_hidden_state[0, -1]).float()
    gap = float((a_ - b_).abs().max())
assert gap < 0.25 and int(a_.argmax()) == int(b_.argmax()), f"lm_head(last_hidden_state) differs from the logits by {gap}"   # bf16 rounding only
held = [k for k in range(len(seqs)) if not build[k]]
set_delta(0.0)
ref = measure(held)
rows, chosen = [], 0.0
for d in [0.0] + DELTAS:
    set_delta(d)
    cur = ref if d == 0.0 else measure(held)
    line = [f"  delta {d:g}:"]
    ok = True
    for v in range(4):
        idx = [j for j, k in enumerate(held) if grp[k] == v]
        if not idx:
            continue
        fire = np.mean([cur[j][1] > 0 for j in idx]); fire0 = np.mean([ref[j][1] > 0 for j in idx])
        pr = {nm: np.mean([float(cur[j][0][ids_].exp().sum()) for j in idx]) for nm, ids_ in GROUPS.items()}
        kl = np.mean([float((ref[j][0].exp() * (ref[j][0] - cur[j][0])).sum()) for j in idx])
        line.append(f"{NAMES[v]}: key fires {fire0:.0%}->{fire:.0%}, " + ", ".join(f"{nm} {pr[nm]:.3f}" for nm in GROUPS) + f", KL {kl:.4f}")
        if v == 0 and d > 0 and (fire - fire0 > 0.02 or kl > 0.005):
            ok = False
    if d > 0 and ok:
        chosen = d
    L.append(" | ".join(line) + ("" if d == 0 else ("  [passes the before-answer rule]" if ok else "  [fails the before-answer rule]")))
    print(L[-1], flush=True)
L.append(f"chosen delta {chosen:g} (largest that passes; 0 = none passes, nothing saved)")
print(L[-1], flush=True)
if chosen > 0:
    set_delta(chosen)
    os.makedirs(out, exist_ok=True)
    m.save_pretrained(out)
    tok.save_pretrained(out)
    for f in os.listdir(base):
        if f.endswith((".py", ".jinja", ".json", ".txt")) and not f.endswith(".index.json") and not os.path.exists(os.path.join(out, f)):
            shutil.copy(os.path.join(base, f), out)
    META["done_signal"] = {"W": W, "delta": chosen, "rows": f"layer {GK} w1[{GN}] (gate only)", "runs": runs, "points": len(X)}
    json.dump(META, open(os.path.join(out, "pam.json"), "w"), indent=1)
    open(os.path.join(out, "done_fit_report.txt"), "w").write("\n".join(L) + "\n")
    print("SAVED", out)
print("\n".join(L))
