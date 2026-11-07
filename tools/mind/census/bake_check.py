#!/usr/bin/env python3
"""Is the baked flag heard at the end, or drowned? Reads a baked model (bake_budget.py) and the original on the same
answers: the clock neuron's output, and the receptor's reading at the final norm (along u), in both models.
  natural   the original model's own level along u (no flag exists there): its spread is the noise floor
  baked     the same reading in the baked model, binned by how hard the clock neuron fires
usage: bake_check.py <original folder> <baked folder> <answers jsonl> [n=12 dev=mps]
"""
import json, os, sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from patients import _lfm3

orig, baked, src = sys.argv[1:4]
opt = dict(a.split("=", 1) for a in sys.argv[4:])
N, DEV = int(opt.get("n", 12)), opt.get("dev", "mps")
info = json.load(open(os.path.join(baked, "baked_budget.json")))
L, j, END = info["layer"], info["slot"], info["end"]
TAIL = "\n\nEnd with a line: Answer: <your answer>"
rows = [r for r in map(json.loads, open(src)) if r["finish"] == "stop" and r["thinking"]][-N:]   # not the bake's first 40


def run(path):
    p = _lfm3(DEV, path)
    tok, m = p.tok, p.model
    ff = m.model.layers[L].feed_forward
    W = m.lm_head.weight[END].float() - m.model.embed_tokens.weight[END].float()      # the receptor (zero in the original)
    out = []
    for r in rows:
        pr = tok(tok.apply_chat_template([{"role": "user", "content": r["q"] + TAIL}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=True), add_special_tokens=False).input_ids
        th = tok("\n" + r["thinking"], add_special_tokens=False).input_ids
        rp = tok("\n</think>\n\n" + r["reply"], add_special_tokens=False).input_ids[:300]
        cap = {}
        h1 = ff.register_forward_pre_hook(lambda mod, a: cap.__setitem__("x", a[0][0].float()))
        h2 = m.model.embedding_norm.register_forward_hook(lambda mod, a, o: cap.__setitem__("fin", o[0].float()))
        with torch.no_grad():
            m(input_ids=torch.tensor([pr + th + rp], device=DEV))
        h1.remove(); h2.remove()
        x = cap["x"]
        act = torch.nn.functional.silu(x @ ff.w1.weight[j].float()) * (x @ ff.w3.weight[j].float())
        out.append((act.cpu(), cap["fin"].cpu(), len(pr), len(th)))
    return out, W.cpu()


base, _ = run(orig)
bk, W = run(baked)
u = W / W.norm()
lines = [f"baked flag check: {N} answers the bake never saw; receptor size {float(W.norm()):.2f}"]
nat = torch.cat([(f @ u) for _, f, _, _ in base])
lines.append(f"natural level along u (original model): mean {float(nat.mean()):+.4f}, spread {float(nat.std()):.4f}")
A = torch.cat([a for a, _, _, _ in bk]); R = torch.cat([(f @ u) for _, f, _, _ in bk])
R0 = torch.cat([(f @ u) for _, f, _, _ in base])
part = []
for a, f, P, T in bk:
    k = torch.zeros(len(a), dtype=torch.long)
    k[P:P + T] = 1; k[P + T:] = 2
    part.append(k)
K = torch.cat(part)
lines.append(f"baked minus original along u, by clock-neuron output (the flag's own effect):")
for lo, hi in [(-1e9, 0), (0, 0.5), (0.5, 2), (2, 8), (8, 1e9)]:
    s = (A > lo) & (A <= hi)
    if s.any():
        d = (R - R0)[s]
        lines.append(f"  output {lo:>6.1f}..{hi:<6.1f} {int(s.sum()):6d} words: flag {float(d.mean()):+.4f} +- {float(d.std()):.4f}"
                     f"   x receptor = {float(d.mean() * W.norm()):+.2f} log-odds")
for name, kk in [("prompt", 0), ("thinking", 1), ("reply", 2)]:
    s = K == kk
    lines.append(f"  {name:9} clock neuron output: median {float(A[s].median()):+.2f}, firing (>0.5) on {float((A[s] > 0.5).float().mean()):.1%}")
lines.append(f"closing-word score change from the receptor alone (W . fin), baked model, thinking vs reply: "
             f"{float((R[K == 1] * W.norm()).median()):+.2f} vs {float((R[K == 2] * W.norm()).median()):+.2f}")
print("\n".join(lines))
open(os.path.join(baked, "bake_check.txt"), "w").write("\n".join(lines) + "\n")
