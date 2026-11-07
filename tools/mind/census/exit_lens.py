#!/usr/bin/env python3
"""When does E4B already know its next word? (2026-10-26, the brain contrast: easy things on autopilot.)

E4B's layers 24-41 keep no memory of their own (KV sharing: they reuse the keys/values written by layers 22/23), so
for any one word those 18 layers are a private workshop that later words never read. If the answer is already
settled before them, they could be skipped for that word without breaking anything downstream.

For every word: read the model's guess after layer k (final norm + output head + soft-cap on the stream at layer k,
the "logit lens"), and compare with the full 42-layer guess. Also the guess's confidence at layer k, which is the
signal a real early exit would gate on.

usage: exit_lens.py <out folder> [math traces jsonl]      writes exit_lens.json and exit_lens.txt
"""
import json, os, sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from e4b_mac import load

out = sys.argv[1]
traces = sys.argv[2] if len(sys.argv) > 2 else None
os.makedirs(out, exist_ok=True)
BENCH = os.path.join(os.path.dirname(__file__), "../../../.run/bench")
model, tok = load()
lm = model.model
cap = model.config.final_logit_softcapping
KS = list(range(10, 42))

wins, src = [], []
wiki = tok(open(f"{BENCH}/wiki.test.raw").read()[:200000], add_special_tokens=False).input_ids
for i in range(16, 24):                                   # different windows from the census
    wins.append([tok.bos_token_id] + wiki[i * 511:(i + 1) * 511]); src.append("wiki")
chat = tok(open(f"{BENCH}/chat_heldout.txt").read(), add_special_tokens=False).input_ids
for i in range(32, 48):
    wins.append([tok.bos_token_id] + chat[i * 511:(i + 1) * 511]); src.append("chat")
if traces:
    for r in [json.loads(l) for l in open(traces)][3::7]:
        p = tok.apply_chat_template([{"role": "user", "content": r["q"]}], tokenize=False, add_generation_prompt=True,
                                    enable_thinking=True)
        wins.append(tok(p + "<|channel>thought\n" + r["thinking"] + "<channel|>" + r["reply"],
                        add_special_tokens=False).input_ids[:768]); src.append("math")


def guess(h):
    """top guess and its probability, 128 words at a time (a full 262k-word table per layer swaps the Mac)"""
    ts, cs = [], []
    for c0 in range(0, h.shape[0], 128):
        z = model.lm_head(lm.norm(h[c0:c0 + 128])).float()
        z = torch.tanh(z / cap) * cap
        m, t = z.max(-1)
        ts.append(t); cs.append((m - torch.logsumexp(z, -1)).exp())
    return torch.cat(ts), torch.cat(cs), None


rec = {s: {"agree": [], "conf": []} for s in dict.fromkeys(src)}
with torch.no_grad():
    for w, s in zip(wins, src):
        o = lm(input_ids=torch.tensor([w], device="mps"), output_hidden_states=True)
        hs = o.hidden_states                                  # embeddings + one per layer
        tf, cf, pf = guess(o.last_hidden_state[0, 1:])
        ag, cf_k = [], []
        for k in KS:
            t, c, _ = guess(hs[k + 1][0, 1:])
            ag.append((t == tf).cpu().numpy()); cf_k.append(c.cpu().numpy())
        rec[s]["agree"].append(np.stack(ag)); rec[s]["conf"].append(np.stack(cf_k))
        rec[s].setdefault("final_conf", []).append(cf.cpu().numpy())
        del o, hs
        torch.mps.empty_cache()
        print(f"  {sum(len(v['agree']) for v in rec.values())}/{len(wins)} windows", flush=True)

lines = ["E4B early-exit lens: does the guess after layer k match the full 42-layer guess?", ""]
summary = {}
for s, r in rec.items():
    A, C = np.concatenate(r["agree"], 1), np.concatenate(r["conf"], 1)
    lines.append(f"{s}: {A.shape[1]} words")
    lines.append("  layer:  " + " ".join(f"{k:>4}" for k in KS[::2]))
    lines.append("  agree%: " + " ".join(f"{A[j].mean() * 100:>4.0f}" for j in range(0, len(KS), 2)))
    for k in (24, 28, 32, 36):
        j = KS.index(k)
        for th in (0.5, 0.8, 0.95):
            m = C[j] > th
            lines.append(f"  exit after layer {k} when its guess is >{th:.0%} sure: {m.mean() * 100:5.1f}% of words exit, "
                         f"{A[j][m].mean() * 100 if m.any() else float('nan'):5.1f}% of those match the full model")
    summary[s] = {"agree": A.mean(1).tolist(), "layers": KS}
    lines.append("")
json.dump(summary, open(os.path.join(out, "exit_lens.json"), "w"), indent=1)
open(os.path.join(out, "exit_lens.txt"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
print("LENS_DONE")
