#!/usr/bin/env python3
"""Reads his own stream at the positions statement_jobs.py picked. Reading only, nothing changes the model. One forward per run, every position
taken from it, the same way for all of them. Text after the last position is not fed. Writes the inputs of the chosen layers' feed-forward
blocks, one row per position, with a jsonl of what each row is.
usage: statement_read.py <hf folder> <jobs.jsonl> <out prefix> [layers=12,16,17,20 dev=cuda]
"""
import json, sys, time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

args = [a for a in sys.argv[1:] if "=" not in a]
opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
hf, jobs, prefix = args
DEV = opt.get("dev", "cuda")
LAY = [int(x) for x in opt.get("layers", "12,16,17,20").split(",")]
tok = AutoTokenizer.from_pretrained(hf, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(hf, dtype=torch.bfloat16, trust_remote_code=True).to(DEV)
model.eval().requires_grad_(False)
got = {}
for L in LAY:
    model.model.layers[L].feed_forward.register_forward_pre_hook(lambda mod, a, L=L: got.__setitem__(L, a[0][0]))
X = {L: [] for L in LAY}
t0 = time.time()
J = [json.loads(l) for l in open(jobs)]
skipped = 0
with open(prefix + "-rows.jsonl", "w") as w:
    for k, j in enumerate(J):
        head = tok.apply_chat_template([{"role": "user", "content": j["q"]}], tokenize=False, add_generation_prompt=True, enable_thinking=True) + "\n"
        th = j["thinking"][:max(p["c"] for p in j["pos"])]
        enc = tok(head + th, add_special_tokens=False, return_offsets_mapping=True)
        ends = np.array([b for a, b in enc.offset_mapping])
        n0 = len(head)
        try:
            with torch.inference_mode():
                model.model(input_ids=torch.tensor([enc.input_ids], device=DEV))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            skipped += 1
            print(f"out of memory on {j['id']} ({len(enc.input_ids)} tokens)", flush=True)
            continue
        idx = [int(np.searchsorted(ends, n0 + p["c"], side="right")) - 1 for p in j["pos"]]
        for L in LAY:
            X[L].append(got[L][idx].float().cpu().numpy().astype(np.float32))
        for p, i in zip(j["pos"], idx):
            w.write(json.dumps({**{x: j[x] for x in ("run", "id", "src", "subject", "group")}, **p, "tokens_before": i}) + "\n")
        if k % 50 == 0:
            print(f"{k}/{len(J)} runs, {time.time() - t0:.0f} s", flush=True)
np.savez(prefix + "-x.npz", **{f"L{L}": np.concatenate(X[L]) for L in LAY})
print(f"READ_DONE {len(J) - skipped} runs, {skipped} skipped, {time.time() - t0:.0f} s", flush=True)
