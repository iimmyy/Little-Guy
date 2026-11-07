#!/usr/bin/env python3
"""What does "Be brief." do inside any patient (patients.py)? brief_probe.py for E4B, made model-agnostic (Immy,
2026-10-28: "find where the hot spots are, then just turn those up ... baked in, no fine tuning").

The same answer text is read twice: after the plain question, and after the question + " Be brief.". The words that
follow are identical, so every difference inside comes from the instruction alone. Per layer, over the answer's words
(thinking + reply), averaged over problems:
  loud      how much the instruction changes the stream's loudness (RMS), in %
  stream    size of the stream change relative to the stream
  top1%     share of the MLP's output change carried by the top 1% of neurons (0.01 = evenly spread)
  same      how much the 100 most-changed neurons agree between problems (overlap 0..1; chance ~0.01)
  dir       how much the stream change points the same way on every problem (mean cosine between problems)
Also: how much more likely the instruction makes it close its thinking, per thinking word.
Saved for the bake (brief_dirs.pt), per layer: the mean stream change (its direction and size, per word), and each
MLP neuron's mean signed change (which neurons the instruction turns up or down).

usage: brief_patient.py <patient> <out folder> <answers jsonl (thinking + reply per problem)> [n=40 dev=cuda]
"""
import json, os, sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from patients import load

name, out, src = sys.argv[1:4]
opt = dict(a.split("=", 1) for a in sys.argv[4:])
NPROB, DEV = int(opt.get("n", 40)), opt.get("dev", "cuda")
TAIL = "\n\nEnd with a line: Answer: <your answer>"
os.makedirs(out, exist_ok=True)
p = load(name, DEV)
tok, layers = p.tok, p.model.model.layers
L = len(layers)
MLP_OUT = {"e4b": lambda l: l.mlp.down_proj, "lfm3": lambda l: l.feed_forward.w2}[name]
# the answer exactly as the model writes it after the generation prompt (which ends E4B's turn start / LFM3's <think>)
ANSWER = {"e4b": lambda r: "<|channel>thought\n" + r["thinking"] + "<channel|>" + r["reply"],
          "lfm3": lambda r: "\n" + r["thinking"] + "\n</think>\n\n" + r["reply"]}[name]
rows = [json.loads(l) for l in open(src)][:NPROB]

cap, span = {}, [0, 0]                                     # keep only the answer's words, in normal memory
for i, l in enumerate(layers):
    l.register_forward_pre_hook(lambda m, a, k, i=i: cap.__setitem__(
        ("h", i), (a[0] if a else k["hidden_states"])[0, span[0]:span[1]].float().cpu()), with_kwargs=True)
    MLP_OUT(l).register_forward_pre_hook(lambda m, a, i=i: cap.__setitem__(
        ("p", i), a[0][0, span[0]:span[1]].float().cpu()))
dn = [MLP_OUT(l).weight.float().norm(dim=0).cpu() for l in layers]
N = dn[0].shape[0]
MAXC = 1500


def read(prompt, cont):
    pr = tok(prompt, add_special_tokens=False).input_ids
    c = tok(cont, add_special_tokens=False).input_ids[:MAXC]
    s = len(pr)
    cap.clear()
    span[0], span[1] = s, s + len(c)
    with torch.no_grad():
        h = p.body(torch.tensor([pr + c], device=DEV))[0, s - 1:s - 1 + len(c)]
        pe = []
        for c0 in range(0, h.shape[0], 256):                          # chance of closing the thinking, per word
            lg = p.logits(h[c0:c0 + 256])
            pe.append((lg[:, p.end] - lg.logsumexp(-1)).cpu())
    return dict(cap), torch.cat(pe), c


acc = {i: {"loud": [], "stream": [], "top1": [], "tops": [], "dirs": [], "mean": [], "neur": []} for i in range(L)}
close = []
for n, r in enumerate(rows):
    q = r["q"] + TAIL
    prompts = [tok.apply_chat_template([{"role": "user", "content": x}], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=True) for x in (q, q + " Be brief.")]
    (a, pa, c), (b, pb, _) = read(prompts[0], ANSWER(r)), read(prompts[1], ANSWER(r))
    nt = c.index(p.end) if p.end in c else len(c)
    close.append((pb[:nt] - pa[:nt]).mean().item())
    for i in range(L):
        ha, hb = a[("h", i)], b[("h", i)]
        ra, rb = ha.pow(2).mean(-1).sqrt(), hb.pow(2).mean(-1).sqrt()
        acc[i]["loud"].append(((rb - ra) / ra).mean().item() * 100)
        d = hb - ha
        acc[i]["stream"].append((d.norm(dim=-1) / ha.norm(dim=-1)).mean().item())
        acc[i]["dirs"].append(torch.nn.functional.normalize(d.mean(0), dim=0))
        acc[i]["mean"].append(d.mean(0))
        dp = b[("p", i)] - a[("p", i)]
        acc[i]["neur"].append(dp.mean(0))                                 # signed: turned up (+) or down (-)
        ch = (dp.abs() * dn[i]).mean(0)                                   # each neuron's output change
        acc[i]["top1"].append((ch.topk(N // 100).values.sum() / ch.sum()).item())
        acc[i]["tops"].append(set(ch.topk(100).indices.tolist()))
    print(f"  {n + 1}/{len(rows)}", flush=True)


def overlap(sets):
    return float(np.mean([len(x & y) / 100 for j, x in enumerate(sets) for y in sets[j + 1:]]))


def cosines(dirs):
    M = torch.stack(dirs)
    C = M @ M.T
    return float((C.sum() - C.diag().sum()) / (len(dirs) ** 2 - len(dirs)))


lines = [f'"Be brief." inside {name}: the same answer read with and without it ({len(rows)} problems, answer words only)',
         "", f"{'layer':>5} {'loud %':>7} {'stream':>7} {'top1%':>6} {'same':>5} {'dir':>5}"]
save = {}
for i in range(L):
    A = acc[i]
    lines.append(f"{i:>5} {np.mean(A['loud']):>+7.2f} {np.mean(A['stream']):>7.3f} {np.mean(A['top1']):>6.2f} "
                 f"{overlap(A['tops']):>5.2f} {cosines(A['dirs']):>5.2f}")
    save[i] = {"mean": torch.stack(A["mean"]).mean(0), "dir": torch.stack(A["dirs"]).mean(0),
               "neur": torch.stack(A["neur"]).mean(0), "neur_sd": torch.stack(A["neur"]).std(0)}
lines += ["", f"chance of closing the thinking (log-odds of the end-of-thinking word per thinking word), brief minus "
          f"plain: mean {np.mean(close):+.3f}, raised on {sum(x > 0 for x in close)}/{len(close)} problems",
          "top1% = 0.01 means evenly spread; same/dir near 0 = no shared switch, near 1 = the same switch every problem"]
torch.save(save, os.path.join(out, "brief_dirs.pt"))
open(os.path.join(out, "brief_probe.txt"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
print("BRIEF_DONE")
