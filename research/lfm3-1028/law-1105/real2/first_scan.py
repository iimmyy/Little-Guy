#!/usr/bin/env python3
"""First look at every arithmetic spot. usage: first_scan.py <hf> <steps.jsonl> <out.jsonl> <shard> <nshards>"""
import json, re, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
hf, steps, outf, sh, ns = sys.argv[1:6]
sh, ns = int(sh), int(ns)
DEV, MAXTOK, NEW = "cuda", 9000, 14
tok = AutoTokenizer.from_pretrained(hf, trust_remote_code=True)
m = AutoModelForCausalLM.from_pretrained(hf, dtype=torch.bfloat16, trust_remote_code=True).to(DEV).eval().requires_grad_(False)
NUM = re.compile(r"\s*(-?\d[\d,]*(?:\.\d+)?)")
def head(q):
    return tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=True) + "\n"
@torch.inference_mode()
def first(text):
    ids = tok(text, add_special_tokens=False).input_ids
    if len(ids) > MAXTOK:
        return None
    o = m(input_ids=torch.tensor([ids], device=DEV), use_cache=True, logits_to_keep=1)
    pkv, out, ps = o.past_key_values, [], []
    for _ in range(NEW):
        lp = o.logits[0, -1].float().softmax(0)
        t = int(lp.argmax())
        out.append(t)
        s = tok.decode(out)
        n = NUM.match(s)
        if n and n.end() < len(s.rstrip()) and not s.rstrip()[n.end():].startswith((",", ".")):
            break
        if not n and len(out) > 2:
            break
        ps.append(float(lp[t]))
        o = m(input_ids=torch.tensor([[t]], device=DEV), past_key_values=pkv, use_cache=True, logits_to_keep=1)
        pkv = o.past_key_values
    s = tok.decode(out)
    n = NUM.match(s)
    return dict(num=n.group(1).replace(",", "") if n else None, text=s, minp=min(ps) if ps else None, firstp=ps[0] if ps else None)
t0, n = time.time(), 0
with open(outf, "w") as w:
    for i, l in enumerate(open(steps)):
        if i % ns != sh:
            continue
        j = json.loads(l)
        r = first(head(j["q"]) + j["pre"])
        if r is None:
            continue
        w.write(json.dumps(dict(step=j["step"], first=r)) + "\n")
        w.flush()
        n += 1
        if n % 100 == 0:
            print(n, round(time.time() - t0), flush=True)
print("SCAN_DONE", n, round(time.time() - t0), flush=True)
