#!/usr/bin/env python3
"""Behavior check for the trick stack: E4B 16-bit on the Mac solves a sample of the 300-problem set (2 per group, 30
problems, greedy, thinking on), untouched vs with tricks on. Right answers and thinking length, graded by grade.py
(then read by hand). Batched 6 at a time to keep the Mac GPU busy.

usage: gen_check.py <problems.jsonl> <out folder> <config name> [route=0-6 keep=0.2 zero=0.15 swap=22,38 maps=file]
"""
import json, os, sys, time

import torch

sys.path.insert(0, os.path.dirname(__file__))
from e4b_mac import load
from tricks import Tricks

src, out, name = sys.argv[1:4]
opt = dict(a.split("=", 1) for a in sys.argv[4:])
os.makedirs(out, exist_ok=True)
model, tok = load()
T = Tricks(model)
route = set()
if "route" in opt:
    a, b = map(int, opt["route"].split("-"))
    route = set(range(a, b + 1))
swap = {}
if "swap" in opt:
    maps = torch.load(opt["maps"])
    swap = {int(i): (maps[int(i)][0].to("mps"), maps[int(i)][1].to("mps")) for i in opt["swap"].split(",")}
T.set(route=route, keep=float(opt.get("keep", 1.0)), zero=float(opt.get("zero", 0.0)), swap=swap)

items = [json.loads(l) for l in open(src)]
pick, seen = [], {}
for it in items:
    key = (it["cat"], it["kind"])
    if seen.get(key, 0) < 2:
        pick.append(it); seen[key] = seen.get(key, 0) + 1
tok.padding_side = "left"
TAIL = "\n\nEnd with a line: Answer: <your answer>"
path = os.path.join(out, f"{name}.jsonl")
with open(path, "w") as f, torch.no_grad():
    for c0 in range(0, len(pick), 6):
        batch = pick[c0:c0 + 6]
        prompts = [tok.apply_chat_template([{"role": "user", "content": it["q"] + TAIL}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=True) for it in batch]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("mps")
        t = time.time()
        o = model.generate(**enc, max_new_tokens=2500, do_sample=False, pad_token_id=tok.pad_token_id)
        for it, seq in zip(batch, o[:, enc.input_ids.shape[1]:]):
            text = tok.decode(seq, skip_special_tokens=False).replace("<pad>", "")
            done = "<turn|>" in text or tok.eos_token in text
            text = text.split("<turn|>")[0].replace(tok.eos_token, "")
            think, _, reply = text.partition("<channel|>") if "<channel|>" in text else ("", "", text)
            think = think.replace("<|channel>thought", "", 1).strip()
            f.write(json.dumps({**it, "thinking": think, "reply": reply.strip(),
                                "think_tokens": len(tok(think, add_special_tokens=False).input_ids),
                                "reply_tokens": len(tok(reply, add_special_tokens=False).input_ids),
                                "finish": "stop" if done else "length"}) + "\n")
        f.flush()
        print(f"  {name}: {c0 + len(batch)}/{len(pick)} ({time.time() - t:.0f} s)", flush=True)
print("GEN_DONE", path)
