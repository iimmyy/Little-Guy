#!/usr/bin/env python3
"""Does any attention head look back at the question while the patient thinks? Step 3 of Immy's time sense: the
expected time is sharp on the question's last word (difficulty_probe.py) but fades in the thinking stream
(expected_persist.py). If some head already reads that word from the thinking, the "working-memory slot" is half
built: strengthen that head instead of installing a new one.

Full-attention layers only (sliding ones see 256 words back and lose the question in long thinking). Per head, from
thinking words t >= 32, the attention share on:
  last      the question's last word (where the expected time lives)
  question  all the question's other words (first word left out: most heads park unused attention there)
  first     the first word (the parking spot, reported so its size is visible)
'x even' = last share / the share an evenly spread head would give one word (1 / words visible).

usage: lookback.py <patient> <answers jsonl with thinking> <out folder> [n=40 dev=mps]
"""
import json, os, sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from patients import load

name, src, out = sys.argv[1:4]
opt = dict(a.split("=", 1) for a in sys.argv[4:])
N, DEV = int(opt.get("n", 40)), opt.get("dev", "mps")
TAIL = "\n\nEnd with a line: Answer: <your answer>"
OPEN = {"lfm3": "\n"}[name]
p = load(name, DEV)
p.model.config._attn_implementation = "eager"            # the eager path hands back the attention weights
tok, layers = p.tok, p.model.model.layers
FULL = [i for i, t in enumerate(p.model.config.layer_types) if t == "full_attention"]
rows = [r for r in map(json.loads, open(src)) if r["thinking"] and r["think_tokens"] >= 160][:N]
cap = {}
for i in FULL:
    layers[i].self_attn.register_forward_hook(lambda m, a, o, i=i: cap.__setitem__(i, o[1][0].float()))

H = layers[FULL[0]].self_attn.config.num_attention_heads
S = {k: {i: [] for i in FULL} for k in ("last", "question", "first", "xeven")}
with torch.no_grad():
    for n, r in enumerate(rows):
        pr = tok(tok.apply_chat_template([{"role": "user", "content": r["q"] + TAIL}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=True), add_special_tokens=False).input_ids
        th = tok(OPEN + r["thinking"], add_special_tokens=False).input_ids[:600]
        L = len(pr)
        cap.clear()
        p.model.model(input_ids=torch.tensor([pr + th], device=DEV))
        q = torch.arange(L - 1 + 32, L - 1 + len(th))        # thinking words t >= 32
        for i in FULL:
            A = cap[i][:, q, :]                              # heads x queries x keys
            last = A[:, :, L - 1]
            S["last"][i].append(last.mean(1).cpu().numpy())
            S["question"][i].append(A[:, :, 1:L - 1].sum(-1).mean(1).cpu().numpy())
            S["first"][i].append(A[:, :, 0].mean(1).cpu().numpy())
            S["xeven"][i].append((last * (q + 1).to(last)[None]).mean(1).cpu().numpy())
        if n == 0:
            print("question's last word:", repr(tok.decode(pr[-1:])), flush=True)
        if (n + 1) % 10 == 0:
            print(f"  {n + 1}/{len(rows)}", flush=True)

lines = [f"Look-back while thinking ({name}, {len(rows)} answers, thinking words 32-600, full-attention layers {FULL})",
         f"question's last word = {tok.decode(pr[-1:])!r}",
         "top heads by share on the question's last word:",
         "layer head   last   x even   question   first"]
allh = [(float(np.mean(S["last"][i], 0)[h]), i, h) for i in FULL for h in range(H)]
for s, i, h in sorted(allh, reverse=True)[:12]:
    lines.append(f"{i:5d} {h:4d}  {s:6.3f}  {float(np.mean(S['xeven'][i], 0)[h]):6.1f}    {float(np.mean(S['question'][i], 0)[h]):6.3f}"
                 f"    {float(np.mean(S['first'][i], 0)[h]):6.3f}")
lines.append("per layer, mean over heads:  last / question / first")
for i in FULL:
    lines.append(f"{i:5d}   {float(np.mean(S['last'][i])):.3f} / {float(np.mean(S['question'][i])):.3f} / {float(np.mean(S['first'][i])):.3f}")
print("\n".join(lines))
open(os.path.join(out, "lookback.txt"), "w").write("\n".join(lines) + "\n")
torch.save({k: {i: np.mean(v[i], 0) for i in FULL} for k, v in S.items()}, os.path.join(out, "lookback.pt"))
print("LOOKBACK_DONE")
