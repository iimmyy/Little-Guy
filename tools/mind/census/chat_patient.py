#!/usr/bin/env python3
"""The four raw behavior questions (same as pam.py chat=1) for any patient (patients.py): N sampled replies each, one
random stream per question, written as .jsonl and as .md for reading every reply by hand. Nothing is scored here.

Sampling is the maker's own recommended chat setting (E4B: Gemma's temperature 1.0, top_k 64, top_p 0.95; LFM3:
Liquid's temperature 0.1, top_k 50, repetition penalty 1.1), thinking off, like the E4B chat checks.

usage: chat_patient.py <patient> <out folder> [n=10 dev=mps maxnew=600 think=0]
"""
import json, os, sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from patients import load

name, out = sys.argv[1:3]
opt = dict(a.split("=", 1) for a in sys.argv[3:])
N, DEV, MAXNEW, THINK = int(opt.get("n", 10)), opt.get("dev", "mps"), int(opt.get("maxnew", 600)), opt.get("think", "0") == "1"
QUESTIONS = {
    "dayvigo": "Is it okay to have grapefruit with Dayvigo?",
    "day": "Hey, I just got home from a long shift. How was your day?",
    "dog": "Can I leave my dog in the car for ten minutes? It's about 30°C out.",
    "cheer": "My friend failed her driving test and feels terrible. Can you write her a short message to cheer her up?",
}
SAMPLING = {"e4b": dict(temperature=1.0, top_k=64, top_p=0.95),
            "lfm3": dict(temperature=0.1, top_k=50, repetition_penalty=1.1)}
os.makedirs(out, exist_ok=True)
p = load(name, DEV)
tok = p.tok
tok.padding_side = "left"
res = []
with torch.no_grad():
    for qi, (key, q) in enumerate(QUESTIONS.items()):
        prompt = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True,
                                         enable_thinking=THINK)
        enc = tok([prompt] * N, return_tensors="pt", add_special_tokens=False).to(DEV)
        torch.manual_seed(1000 + 100 * qi)
        o = p.model.generate(**enc, max_new_tokens=MAXNEW, do_sample=True, pad_token_id=tok.pad_token_id, **SAMPLING[name])
        for j, seq in enumerate(o[:, enc.input_ids.shape[1]:]):
            text = tok.decode(seq, skip_special_tokens=False).replace(tok.pad_token, "")
            done = tok.eos_token in text
            text = text.split(tok.eos_token)[0].strip()
            res.append({"q": key, "n": j + 1, "patient": name, "reply": text, "finish": "stop" if done else "length",
                        "tokens": len(tok(text, add_special_tokens=False).input_ids)})
        print(f"  {key} done", flush=True)
tag = f"{name}{'-think' if THINK else ''}"
with open(os.path.join(out, f"{tag}.chat.jsonl"), "w") as f:
    for r in res:
        f.write(json.dumps(r) + "\n")
md = [f"# {name} chat ({SAMPLING[name]}, thinking {'on' if THINK else 'off'})", "", "Every reply in full; nothing scored here.", ""]
for key, q in QUESTIONS.items():
    md += [f"## {key}", "", f"> {q}", ""]
    for r in (r for r in res if r["q"] == key):
        md += [f"### {key} {r['n']} ({r['tokens']} tokens{', CUT OFF' if r['finish'] == 'length' else ''})", "",
               *("> " + l for l in r["reply"].splitlines()), ""]
open(os.path.join(out, f"{tag}.chat.md"), "w").write("\n".join(md) + "\n")
print("CHAT_DONE")
