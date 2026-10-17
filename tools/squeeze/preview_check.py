"""Rung 1 check: how often does the dictionary preview's shortlist miss what the full dictionary would pick?

For every recorded position (the final hidden state that scores words, from llama-imatrix --process-output with
LG_ACT_DUMP), score all words with the rough copy and with the full dictionary, keep the rough top-n as the engine
does (src/models/gemma4.cpp), and count against the full dictionary's own scores:
  top1_miss   the full top word is not on the shortlist (greedy output would change)
  top64_miss  some of the full top 64 words are not on it (what a top-k 64 sampler can pick from)
  mass_out    probability the full dictionary puts outside the shortlist (the engine gives those words zero)
  kl          KL(shortlist-renormalized || full) = -log(1 - mass_out), the drift the preview itself adds

usage: python preview_check.py <brain.gguf> <rough.gguf> <hidden.f16> [<hidden.f16> ...]
"""
import os, sys
import numpy as np
import torch
from gguf import GGUFReader
from gguf.quants import dequantize


def table(path, name="token_embd.weight"):
    r = GGUFReader(path)
    t = next(t for t in r.tensors if t.name == name)
    w = dequantize(t.data, t.tensor_type).reshape(-1, int(t.shape[0]))
    cap = next((f.contents() for k, f in r.fields.items() if k.endswith("final_logit_softcapping")), None)
    return torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32)), cap


brain, rough_p, hidden = sys.argv[1], sys.argv[2], sys.argv[3:]
dev = "cuda"
E, cap = table(brain)
P, _ = table(rough_p)
E, P = E.to(dev), P.to(dev)
print(f"dictionary {tuple(E.shape)}, rough copy {tuple(P.shape)}, softcap {cap}")
ns = [int(x) for x in os.environ.get("LG_NS", "256,512,1024,2048,4096").split(",")]
for hp in hidden:
    H = torch.from_numpy(np.fromfile(hp, dtype=np.float16).reshape(-1, E.shape[1]).astype(np.float32))
    acc = {n: {"top1": 0, "top64": 0, "mass": [], "kl": []} for n in ns}
    for i in range(0, H.shape[0], 256):
        h = H[i:i + 256].to(dev)
        z = h @ E.T
        if cap:
            z = torch.tanh(z / cap) * cap
        p = torch.softmax(z, dim=-1)
        top1 = z.argmax(-1, keepdim=True)
        top64 = z.topk(64, dim=-1).indices
        rough = h @ P.T
        order = rough.topk(max(ns), dim=-1).indices
        for n in ns:
            s = order[:, :n]
            insl = torch.zeros_like(z, dtype=torch.bool).scatter_(1, s, True)
            acc[n]["top1"] += int((~insl.gather(1, top1)).sum())
            acc[n]["top64"] += int((~insl.gather(1, top64)).any(-1).sum())
            m = (p * ~insl).sum(-1).clamp(0, 1)
            acc[n]["mass"].append(m.cpu()); acc[n]["kl"].append((-torch.log1p(-m.clamp(max=1 - 1e-12))).cpu())
    N = H.shape[0]
    print(f"\n{hp}: {N} positions")
    print(f"{'shortlist':>9} | {'top-1 missed':>14} | {'top-64 incomplete':>17} | {'mass outside: mean':>18} {'max':>9} | {'KL mean':>9}")
    for n in ns:
        m = torch.cat(acc[n]["mass"]); kl = torch.cat(acc[n]["kl"])
        print(f"{n:>9} | {acc[n]['top1']:>6} ({100 * acc[n]['top1'] / N:5.3f}%) | {acc[n]['top64']:>6} ({100 * acc[n]['top64'] / N:5.2f}%) "
              f"| {m.mean().item():18.2e} {m.max().item():9.2e} | {kl.mean().item():9.2e}")
