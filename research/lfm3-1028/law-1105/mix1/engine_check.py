#!/usr/bin/env python3
"""Does the engine read the same as Python? usage: engine_check.py <tok> <steps> <python rows> <key> <port> <n>"""
import json, re, sys, urllib.request
import numpy as np
from transformers import AutoTokenizer
tkp, steps, rowsf, key, port, n = sys.argv[1:7]; n = int(n)
tok = AutoTokenizer.from_pretrained(tkp, trust_remote_code=True)
EXPR = re.compile(r"(?<![\d.,])\d[\d.,]*\s*(?:\*|×|\\times|\\cdot|/|\+|-)\s*\d[\d.,]*\s*=\s*$")
J = {json.loads(l)["step"]: json.loads(l) for l in open(steps)}
def head(q):
    return tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=True) + "\n"
def margin(ids, a, b):
    body = dict(prompt=ids, n_predict=1, n_probs=400, temperature=0, cache_prompt=False, stream=False)
    r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/completion", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=300))
    t = {x["id"]: x["logprob"] for x in r["completion_probabilities"][0]["top_logprobs"]}
    return (t[a] - t[b]) if a in t and b in t else None
d = []
for l in list(open(rowsf))[:n * 3]:
    if len(d) >= n * 2:
        break
    row = json.loads(l); j = J[row["step"]]
    h = head(j["q"]); expr = EXPR.search(j["pre"]).group()
    W = tok(h + j["pre"] + j["broken"], add_special_tokens=False).input_ids
    R = tok(h + j["pre"] + j["clean"], add_special_tokens=False).input_ids
    q = [i for i, (x, y) in enumerate(zip(W, R)) if x != y][0]; a, b = R[q], W[q]
    L0 = len(tok(h + j["pre"], add_special_tokens=False).input_ids)
    shared = tok.decode(R[L0:q]) if q > L0 else ""
    for k, num in enumerate((j["clean"], j["broken"])):
        ids = tok(h + j["pre"] + num + ". " + expr + shared, add_special_tokens=False).input_ids
        m = margin(ids, a, b)
        if m is not None:
            d.append(abs(m - row["reads"][key][k]))
print("reads", len(d), "median difference", round(float(np.median(d)), 3), "largest", round(float(np.max(d)), 3), flush=True)
sys.exit(0 if np.median(d) <= 0.3 and len(d) >= 10 else 1)
