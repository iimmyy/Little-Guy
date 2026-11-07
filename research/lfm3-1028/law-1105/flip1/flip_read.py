#!/usr/bin/env python3
"""Does an empty word change a finished answer? Reading only. usage: flip_read.py <hf> <steps.jsonl> <out.jsonl> <shard> <nshards>"""
import json, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
hf, steps, outf, sh, ns = sys.argv[1:6]
sh, ns = int(sh), int(ns)
DEV, MAXTOK, NEW = "cuda", 12000, 1500
tok = AutoTokenizer.from_pretrained(hf)
m = AutoModelForCausalLM.from_pretrained(hf, dtype=torch.bfloat16)
for tower in ("vision_tower", "audio_tower", "embed_vision", "embed_audio"):
    if hasattr(m.model, tower):
        delattr(m.model, tower)
m = m.to(DEV).eval().requires_grad_(False)
e = m.generation_config.eos_token_id
EOS = {tok.eos_token_id} | set(e if isinstance(e, list) else [e])
print("EOS", [tok.decode([x]) for x in EOS], flush=True)
OPEN, CLOSE = "<|channel>thought\n", "<channel|>"
def head(q):
    return tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
@torch.inference_mode()
def go(text, cap):
    ids = tok(text, add_special_tokens=False).input_ids
    if len(ids) > MAXTOK:
        return None, 0, False
    o = m(input_ids=torch.tensor([ids], device=DEV), use_cache=True, logits_to_keep=1)
    pkv, out = o.past_key_values, []
    t = int(o.logits[0, -1].argmax())
    done = False
    for _ in range(cap):
        if t in EOS:
            done = True
            break
        out.append(t)
        o = m(input_ids=torch.tensor([[t]], device=DEV), past_key_values=pkv, use_cache=True, logits_to_keep=1)
        pkv = o.past_key_values
        t = int(o.logits[0, -1].argmax())
    return tok.decode(out), len(out), done
t0, n = time.time(), 0
with open(outf, "w") as w:
    for i, l in enumerate(open(steps)):
        if i % ns != sh:
            continue
        j = json.loads(l)
        h = head(j["q"]) + OPEN + j["thought"].rstrip()
        row = dict(id=j["id"], set=j["set"], label=j["label"], answer=j["answer"], scale=j.get("scale"), arms={})
        for name, tail, cap in (("base", CLOSE, 900), ("now", "\n\nNow, ", NEW), ("wait", "\n\nWait, ", NEW)):
            text, k, done = go(h + tail, cap)
            row["arms"][name] = dict(text=text, tokens=k, done=done)
        w.write(json.dumps(row) + "\n")
        w.flush()
        n += 1
        if n % 5 == 0:
            print(n, round(time.time() - t0), flush=True)
print("FLIP_DONE", n, round(time.time() - t0), flush=True)
