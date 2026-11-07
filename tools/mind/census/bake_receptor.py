#!/usr/bin/env python3
"""Tune the closing word's receptor to the flag (second step of bake_budget.py; bake_check.py showed the flag rides
to the end but the receptor listened along a direction where the stream's own swings were ~3x the flag).

At the final norm, on the same answers in both models: the flag's image (baked minus original, on words where the
clock neuron fires) and the original stream's own spread. The receptor becomes the best straight-line listener for
that image in that noise (Fisher's discriminant: spread^-1 . image, with a little ridge), sized so a typical firing
adds `close` log-odds to the closing word, with its resting level taken back off along the final stream's steadiest
entry. Rewrites only the closing word's row of the (already untied) output matrix.

usage: bake_receptor.py <original folder> <baked folder> <answers jsonl> [close=20 ridge=0.05 n=40 dev=mps]
"""
import json, os, sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from patients import _lfm3

orig, baked, src = sys.argv[1:4]
opt = dict(a.split("=", 1) for a in sys.argv[4:])
CLOSE, RIDGE, N, DEV = float(opt.get("close", 20)), float(opt.get("ridge", 0.05)), int(opt.get("n", 40)), opt.get("dev", "mps")
info = json.load(open(os.path.join(baked, "baked_budget.json")))
L, j, END = info["layer"], info["slot"], info["end"]
TAIL = "\n\nEnd with a line: Answer: <your answer>"
rows = [r for path in src.split(",") for r in [r for r in map(json.loads, open(path)) if r["finish"] == "stop" and r["thinking"]][:N]]   # the bake's own answers


def run(p, with_act):
    tok, m = p.tok, p.model
    ff = m.model.layers[L].feed_forward
    fins, acts, part = [], [], []
    for r in rows:
        pr = tok(tok.apply_chat_template([{"role": "user", "content": r["q"] + TAIL}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=True), add_special_tokens=False).input_ids
        th = tok("\n" + r["thinking"], add_special_tokens=False).input_ids[:1200]
        rp = tok("\n</think>\n\n" + r["reply"], add_special_tokens=False).input_ids[:300]
        ids = pr + th + rp
        part.append(torch.tensor([0] * len(pr) + [1] * len(th) + [2] * len(rp)))
        cap = {}
        hs = [m.model.embedding_norm.register_forward_hook(lambda mod, a, o: cap.__setitem__("fin", o[0].float().cpu()))]
        if with_act:
            hs.append(ff.register_forward_pre_hook(lambda mod, a: cap.__setitem__("act", (
                torch.nn.functional.silu(a[0][0].float() @ ff.w1.weight[j].float()) * (a[0][0].float() @ ff.w3.weight[j].float())).cpu())))
        with torch.no_grad():
            m(input_ids=torch.tensor([ids], device=DEV))
        for h in hs:
            h.remove()
        fins.append(cap["fin"]); acts.append(cap.get("act"))
    return torch.cat(fins), (torch.cat(acts) if with_act else None), torch.cat(part)


po = _lfm3(DEV, orig)
F0, _, K = run(po, False)
del po
pb = _lfm3(DEV, baked)
F1, A, _ = run(pb, True)
fire = A > 2.0
# the words that must stay quiet: every word where the clock neuron is off (prompt, early thinking) and every reply
# word, in the baked model itself (build 2: tuned on the original stream alone, reply words still reached +15 at 1%)
quiet = (A <= 0.5) | (K == 2)
w_q = torch.where(K == 2, 4.0, 1.0)[quiet]                   # reply words count 4x: a stray close there breaks the answer
Fq = F1[quiet]
mq = (Fq * w_q[:, None]).sum(0) / w_q.sum()
Z = (Fq - mq) * w_q.sqrt()[:, None]
C = Z.T @ Z / w_q.sum()
C = C + RIDGE * C.diagonal().mean() * torch.eye(C.shape[0])
delta = F1[fire].mean(0) - mq
u = torch.linalg.solve(C, delta)
gain = float(delta @ u)
beta = CLOSE / gain
rest = float(mq @ u) * beta
mf, sf = F0.mean(0), F0.std(0)
gf = int((mf.abs() / (sf + 1e-6)).argmax()); Gf = float(mf[gf])
row = beta * u
row[gf] -= rest / Gf
m = pb.model
with torch.no_grad():
    m.lm_head.weight[END] = (m.model.embed_tokens.weight[END].float().cpu() + row).to(m.lm_head.weight)
# what it does now, on these answers
score = lambda F: F @ row
nat, now = score(F0), score(F1)
lines = [f"receptor retuned (Fisher, ridge {RIDGE}): gain per firing {gain:.4f}, beta {beta:.2f}; steadiest final entry {gf} "
         f"(spread {float(sf[gf] / abs(Gf)):.1%} of its level)",
         f"closing word boost, original stream (noise): median {float(nat.median()):+.2f}, 10-90% {float(nat.quantile(.1)):+.2f}..{float(nat.quantile(.9)):+.2f}",
         f"closing word boost, baked, clock neuron firing (>2): median {float(now[fire].median()):+.2f}, 10-90% "
         f"{float(now[fire].quantile(.1)):+.2f}..{float(now[fire].quantile(.9)):+.2f}",
         f"closing word boost, baked, clock neuron quiet (<=0): median {float(now[A <= 0].median()):+.2f}, 10-90% "
         f"{float(now[A <= 0].quantile(.1)):+.2f}..{float(now[A <= 0].quantile(.9)):+.2f}",
         f"closing word boost, baked, reply words (must stay low): median {float(now[K == 2].median()):+.2f}, 90% "
         f"{float(now[K == 2].quantile(.9)):+.2f}, 99% {float(now[K == 2].quantile(.99)):+.2f}, max {float(now[K == 2].max()):+.2f}"]
print("\n".join(lines))
m.save_pretrained(baked)
info.update({"receptor": "fisher-vs-quiet-and-reply", "ridge": RIDGE, "beta": beta, "gain": gain})
json.dump(info, open(os.path.join(baked, "baked_budget.json"), "w"), indent=1)
open(os.path.join(baked, "bake_receptor.txt"), "w").write("\n".join(lines) + "\n")
print("RECEPTOR_DONE")
