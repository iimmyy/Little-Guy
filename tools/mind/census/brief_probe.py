#!/usr/bin/env python3
"""What does "Be brief." do inside E4B? (Immy, 2026-10-28: "look at the neuron activation when you tell them be brief")

The same answer text is read twice: after the plain question, and after the question + " Be brief.". The words that
follow are identical, so every difference inside him comes from the instruction alone. Per layer, over the answer's
words (thinking + reply), averaged over problems:
  giant     how far the instruction moves the giant (in units of the giant's own wobble on plain answers)
  loud      how much the instruction changes the bloodstream's loudness (RMS), in %
  stream    size of the bloodstream change relative to the bloodstream (all directions)
  top1%     share of the MLP's output change carried by the top 1% of neurons (1% = evenly spread)
  same      how much the 100 most-changed neurons agree between problems (overlap, 0..1; chance ~0.01)
  dir       how much the bloodstream change points the same way on every problem (mean cosine between problems)
Also: how much more likely the instruction makes him close his thinking at each thinking word.

usage: brief_probe.py <out folder> <answers jsonl (thinking + reply per problem)> [n=40 dev=mps]
writes brief_probe.txt and brief_dirs.pt (per layer: mean bloodstream change, top neurons)
"""
import json, os, sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from e4b_mac import load

out, src = sys.argv[1], sys.argv[2]
opt = dict(a.split("=", 1) for a in sys.argv[3:])
NPROB, DEV = int(opt.get("n", 40)), opt.get("dev", "mps")
TAIL = "\n\nEnd with a line: Answer: <your answer>"
model, tok = load(DEV)
layers = model.model.layers
L, D, N = len(layers), model.config.hidden_size, layers[0].mlp.intermediate_size
rows = [json.loads(l) for l in open(src)][:NPROB]
END = tok.convert_tokens_to_ids("<channel|>")

cap, span = {}, [0, 0]                                     # keep only the answer's words, compact, in normal memory
for i, l in enumerate(layers):
    l.register_forward_pre_hook(lambda m, a, k, i=i: cap.__setitem__(
        ("h", i), (a[0] if a else k["hidden_states"])[0, span[0]:span[1]].to(torch.bfloat16).cpu()), with_kwargs=True)
    l.mlp.down_proj.register_forward_pre_hook(lambda m, a, i=i: cap.__setitem__(
        ("p", i), a[0][0, span[0]:span[1]].to(torch.bfloat16).cpu()))
dn = [l.mlp.down_proj.weight.float().norm(dim=0).cpu() for l in layers]


MAXC = 1500


def read(prompt, cont):
    p = tok(prompt, add_special_tokens=False).input_ids
    c = tok(cont, add_special_tokens=False).input_ids[:MAXC]
    s = len(p)
    cap.clear()
    span[0], span[1] = s, s + len(c)
    with torch.no_grad():
        h = model.model(input_ids=torch.tensor([p + c], device=DEV)).last_hidden_state[0, s - 1:s - 1 + len(c)]
        pe = []
        for c0 in range(0, h.shape[0], 128):                          # chance of closing the thinking, per word
            lg = model.lm_head(h[c0:c0 + 128]).float()
            cap_ = model.config.final_logit_softcapping
            if cap_:
                lg = cap_ * torch.tanh(lg / cap_)
            pe.append((lg[:, END] - lg.logsumexp(-1)).cpu())
    return {k: v.float() for k, v in cap.items()}, torch.cat(pe), c


acc = {i: {"giant": [], "loud": [], "stream": [], "top1": [], "tops": [], "dirs": []} for i in range(L)}
wob = {}
close = []
for n, r in enumerate(rows):
    q = r["q"] + TAIL
    cont = "<|channel>thought\n" + r["thinking"] + "<channel|>" + r["reply"]
    prompts = [tok.apply_chat_template([{"role": "user", "content": x}], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=True) for x in (q, q + " Be brief.")]
    (a, pa, c), (b, pb, _) = read(prompts[0], cont), read(prompts[1], cont)
    nt = c.index(END) if END in c else len(c)
    close.append((pb[:nt] - pa[:nt]).mean().item())
    for i in range(L):
        ha, hb = a[("h", i)], b[("h", i)]
        k = int(ha.pow(2).mean(0).argmax())                               # the giant (loudest number) at this layer
        wob.setdefault(i, []).append(ha[:, k].std().item())
        acc[i]["giant"].append(((hb[:, k] - ha[:, k]).mean() / ha[:, k].std().clamp_min(1e-6)).item())
        ra, rb = ha.pow(2).mean(-1).sqrt(), hb.pow(2).mean(-1).sqrt()
        acc[i]["loud"].append(((rb - ra) / ra).mean().item() * 100)
        d = hb - ha
        acc[i]["stream"].append((d.norm(dim=-1) / ha.norm(dim=-1)).mean().item())
        acc[i]["dirs"].append(torch.nn.functional.normalize(d.mean(0), dim=0).cpu())
        ch = ((b[("p", i)] - a[("p", i)]).abs() * dn[i]).mean(0)          # each neuron's output change
        top = ch.topk(N // 100).values.sum() / ch.sum()
        acc[i]["top1"].append(top.item())
        acc[i]["tops"].append(set(ch.topk(100).indices.tolist()))
    if DEV == "mps":
        torch.mps.empty_cache()
    print(f"  {n + 1}/{len(rows)}", flush=True)


def overlap(sets):
    v = [len(x & y) / 100 for j, x in enumerate(sets) for y in sets[j + 1:]]
    return float(np.mean(v))


def cosines(dirs):
    M = torch.stack(dirs)
    C = M @ M.T
    n = len(dirs)
    return float((C.sum() - C.diag().sum()) / (n * n - n))


lines = [f'"Be brief." inside E4B: the same answer read with and without it ({len(rows)} problems, answer words only)', "",
         f"{'layer':>5} {'giant':>8} {'loud %':>7} {'stream':>7} {'top1%':>6} {'same':>5} {'dir':>5}"]
save = {}
for i in range(L):
    A = acc[i]
    lines.append(f"{i:>5} {np.mean(A['giant']):>+8.2f} {np.mean(A['loud']):>+7.2f} {np.mean(A['stream']):>7.3f} "
                 f"{np.mean(A['top1']):>6.2f} {overlap(A['tops']):>5.2f} {cosines(A['dirs']):>5.2f}")
    save[i] = {"dir": torch.stack(A["dirs"]).mean(0), "tops": A["tops"]}
lines += ["", f"chance of closing the thinking (log-odds of <channel|> per thinking word), brief minus plain: "
          f"mean {np.mean(close):+.3f}, raised on {sum(x > 0 for x in close)}/{len(close)} problems",
          "giant = shift in units of the giant's own wobble; top1% = 0.01 means evenly spread; same/dir near 0 = no shared "
          "switch, near 1 = the same switch on every problem"]
torch.save(save, os.path.join(out, "brief_dirs.pt"))
open(os.path.join(out, "brief_probe.txt"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
print("BRIEF_DONE")
