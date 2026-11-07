#!/usr/bin/env python3
"""Perplexity in transformers scored exactly the way llama-perplexity scores it, so a GGUF conversion can be checked
against the original number for number: the whole text tokenized once with the start word in front, cut into chunks of
n_ctx, each chunk's first word replaced by the start word, and only the second half of each chunk scored.

usage: ppl_like_llama.py <hf model folder> <text file> [c=512 chunks=40 dev=mps]
"""
import math, sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

path, text = sys.argv[1:3]
opt = dict(a.split("=", 1) for a in sys.argv[3:])
C, N, DEV = int(opt.get("c", 512)), int(opt.get("chunks", 40)), opt.get("dev", "mps")
tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, trust_remote_code=True).to(DEV).eval()
ids = [tok.bos_token_id] + tok(open(text).read(), add_special_tokens=False).input_ids
first, nll, n = C // 2, 0.0, 0
with torch.no_grad():
    for i in range(N):
        chunk = ids[i * C:(i + 1) * C]
        chunk[0] = tok.bos_token_id
        x = torch.tensor([chunk], device=DEV)
        lg = model(input_ids=x).logits[0, first:C - 1].float()
        nll += torch.nn.functional.cross_entropy(lg, x[0, first + 1:], reduction="sum").item()
        n += C - 1 - first
print(f"{len(ids)} words in the text; c={C}, {N} chunks, {n} scored: PPL = {math.exp(nll / n):.4f}")
