#!/usr/bin/env python3
"""Hidden second look at his own first-look number. Reading only. usage: look2_read.py <hf> <steps.jsonl> <out.jsonl> <shard> <nshards>"""
import json, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
hf, steps, outf, sh, ns = sys.argv[1:6]
sh, ns = int(sh), int(ns)
DEV, MAXTOK, NEW = "cuda", 9000, 60
tok = AutoTokenizer.from_pretrained(hf, trust_remote_code=True)
m = AutoModelForCausalLM.from_pretrained(hf, dtype=torch.bfloat16, trust_remote_code=True).to(DEV).eval().requires_grad_(False)
EOS = {tok.eos_token_id} | set(m.generation_config.eos_token_id if isinstance(m.generation_config.eos_token_id, list) else [m.generation_config.eos_token_id])
SEPS = {"plain": ". ", "now": ". Now, ", "wait": ". Wait, ", "hmm": ". Hmm, "}
def head(q):
    return tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=True) + "\n"
@torch.inference_mode()
def go(text):
    ids = tok(text, add_special_tokens=False).input_ids
    if len(ids) > MAXTOK:
        return None
    o = m(input_ids=torch.tensor([ids], device=DEV), use_cache=True, logits_to_keep=1)
    pkv, out = o.past_key_values, []
    t = int(o.logits[0, -1].argmax())
    for _ in range(NEW):
        if t in EOS:
            break
        out.append(t)
        o = m(input_ids=torch.tensor([[t]], device=DEV), past_key_values=pkv, use_cache=True, logits_to_keep=1)
        pkv = o.past_key_values
        t = int(o.logits[0, -1].argmax())
    return tok.decode(out)
t0, n = time.time(), 0
with open(outf, "w") as w:
    for i, l in enumerate(open(steps)):
        if i % ns != sh:
            continue
        j = json.loads(l)
        h = head(j["q"])
        reads = {name: go(h + j["pre"] + j["page"] + sep + j["expr"]) for name, sep in SEPS.items()}
        w.write(json.dumps(dict(step=j["step"], id=j["id"], set=j["set"], expr=j["expr"], page=j["page"], true=j["true"], minp=j["minp"], reads=reads)) + "\n")
        w.flush()
        n += 1
        if n % 20 == 0:
            print(n, round(time.time() - t0), flush=True)
print("LOOK_DONE", n, round(time.time() - t0), flush=True)
