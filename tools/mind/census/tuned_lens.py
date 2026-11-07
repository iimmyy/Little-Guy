#!/usr/bin/env python3
"""Tuned lens on E4B: can ONE straight-line map stand in for all the layers after k? (Immy's "simplify": swap a
stack of complex layers for one simple equation; the brain contrast: a reflex instead of deliberation.)

For each layer k, fit  h_final ~ W_k h_k + b_k  by least squares on the census words (training), then on different
words (testing) compare the guess from W_k h_k with the full 42-layer guess, and gate it on its confidence.
The plain logit lens (exit_lens.py) found nothing decided before layer 28; this checks whether that was the lens.

usage: tuned_lens.py <out folder> [math traces jsonl]        writes tuned_lens.txt and tuned_lens_maps.pt
"""
import json, os, sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from e4b_mac import load

out = sys.argv[1]
traces = sys.argv[2] if len(sys.argv) > 2 else None
BENCH = os.path.join(os.path.dirname(__file__), "../../../.run/bench")
model, tok = load()
lm = model.model
cap = model.config.final_logit_softcapping
KS = [12, 16, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40]
D = model.config.hidden_size


def windows(wiki_rng, chat_rng, math_slice):
    W, S = [], []
    wiki = tok(open(f"{BENCH}/wiki.test.raw").read()[:200000], add_special_tokens=False).input_ids
    chat = tok(open(f"{BENCH}/chat_heldout.txt").read(), add_special_tokens=False).input_ids
    for i in wiki_rng:
        W.append([tok.bos_token_id] + wiki[i * 511:(i + 1) * 511]); S.append("wiki")
    for i in chat_rng:
        W.append([tok.bos_token_id] + chat[i * 511:(i + 1) * 511]); S.append("chat")
    if traces:
        for r in [json.loads(l) for l in open(traces)][math_slice]:
            p = tok.apply_chat_template([{"role": "user", "content": r["q"]}], tokenize=False,
                                        add_generation_prompt=True, enable_thinking=True)
            W.append(tok(p + "<|channel>thought\n" + r["thinking"] + "<channel|>" + r["reply"],
                         add_special_tokens=False).input_ids[:768]); S.append("math")
    return W, S


def guess(h):
    ts, cs = [], []
    for c0 in range(0, h.shape[0], 128):
        z = model.lm_head(lm.norm(h[c0:c0 + 128].to(torch.bfloat16))).float()
        z = torch.tanh(z / cap) * cap
        m, t = z.max(-1)
        ts.append(t); cs.append((m - torch.logsumexp(z, -1)).exp())
    return torch.cat(ts), torch.cat(cs)


# fit: normal equations on CPU in float64 (with a bias column) ------------------------------------------------------
train, _ = windows(range(16), range(32), slice(None, None, 7))
XtX = {k: torch.zeros(D + 1, D + 1, dtype=torch.float64) for k in KS}
XtY = {k: torch.zeros(D + 1, D, dtype=torch.float64) for k in KS}
with torch.no_grad():
    for j, w in enumerate(train):
        o = lm(input_ids=torch.tensor([w], device="mps"), output_hidden_states=True)
        y = o.last_hidden_state[0, 1:].float().cpu().double()
        for k in KS:
            x = o.hidden_states[k + 1][0, 1:].float().cpu().double()
            x = torch.cat([x, torch.ones(x.shape[0], 1, dtype=torch.float64)], 1)
            XtX[k] += x.T @ x; XtY[k] += x.T @ y
        del o
        torch.mps.empty_cache()
        if j % 20 == 19:
            print(f"  fit {j + 1}/{len(train)}", flush=True)
maps = {}
for k in KS:
    lam = 1e-3 * XtX[k].diagonal().mean()
    Wb = torch.linalg.solve(XtX[k] + lam * torch.eye(D + 1, dtype=torch.float64), XtY[k])
    maps[k] = Wb.float()
torch.save(maps, os.path.join(out, "tuned_lens_maps.pt"))

# test on different words -------------------------------------------------------------------------------------------
test, src = windows(range(16, 24), range(32, 48), slice(3, None, 7))
rec = {}
with torch.no_grad():
    for w, s in zip(test, src):
        o = lm(input_ids=torch.tensor([w], device="mps"), output_hidden_states=True)
        tf, _ = guess(o.last_hidden_state[0, 1:])
        A, C = [], []
        for k in KS:
            x = o.hidden_states[k + 1][0, 1:].float()
            Wb = maps[k].to("mps")
            t, c = guess(x @ Wb[:D] + Wb[D])
            A.append((t == tf).cpu().numpy()); C.append(c.cpu().numpy())
        r = rec.setdefault(s, {"A": [], "C": []})
        r["A"].append(np.stack(A)); r["C"].append(np.stack(C))
        del o
        torch.mps.empty_cache()

lines = ["E4B tuned lens: one straight-line map from layer k to the end, tested on words it was not fitted on", ""]
for s, r in rec.items():
    A, C = np.concatenate(r["A"], 1), np.concatenate(r["C"], 1)
    lines += [f"{s}: {A.shape[1]} words", "  layer:  " + " ".join(f"{k:>4}" for k in KS),
              "  agree%: " + " ".join(f"{a * 100:>4.0f}" for a in A.mean(1))]
    for k in (20, 24, 28, 32):
        j = KS.index(k)
        for th in (0.8, 0.95):
            m = C[j] > th
            lines.append(f"  exit after layer {k} when >{th:.0%} sure: {m.mean() * 100:5.1f}% of words exit, "
                         f"{A[j][m].mean() * 100 if m.any() else float('nan'):5.1f}% of those match")
    lines.append("")
open(os.path.join(out, "tuned_lens.txt"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
print("TUNED_DONE")
