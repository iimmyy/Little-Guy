#!/usr/bin/env python3
"""Chat behavior check for the trick stack (2026-10-27): the four raw-model behavior questions from the fresh-E4B audit
(dayvigo, day, dog, cheer), no system prompt, same sampling as that audit (temperature 1.0, top_k 64, top_p 0.95,
600 tokens), 16-bit E4B on the Mac, untouched vs with the REAL picker (from the gates) + background lines + input
zeroing. Same random stream per batch in every setting, so differences come from the model. Tricks stay on for the
prompt too (stricter than "read full, write in economy").

Writes <name>.jsonl and <name>.md (every reply in full) for reading by hand; nothing is scored here.

usage: chat_check.py <out folder> <math traces jsonl> <name> [route=0-12 keep=0.2 zero=0.25 seeds=20]
"""
import json, os, sys, time

import torch

sys.path.insert(0, os.path.dirname(__file__))
from e4b_mac import load
from tricks import Tricks

out, traces, name = sys.argv[1:4]
opt = dict(a.split("=", 1) for a in sys.argv[4:])
os.makedirs(out, exist_ok=True)
QUESTIONS = {
    "dayvigo": "Is it okay to have grapefruit with Dayvigo?",
    "day": "Hey, I just got home from a long shift. How was your day?",
    "dog": "Can I leave my dog in the car for ten minutes? It's about 30°C out.",
    "cheer": "My friend failed her driving test and feels terrible. Can you write her a short message to cheer her up?",
}
SEEDS = int(opt.get("seeds", 20))
BATCH = 10
BENCH = os.path.join(os.path.dirname(__file__), "../../../.run/bench")

model, tok = load()
T = Tricks(model)
route = set()
if "route" in opt:
    a, b = map(int, opt["route"].split("-"))
    route = set(range(a, b + 1))
keep, zero = float(opt.get("keep", 1.0)), float(opt.get("zero", 0.0))
lines = None
t0 = time.time()
if route:
    wiki = tok(open(f"{BENCH}/wiki.test.raw").read()[:200000], add_special_tokens=False).input_ids
    chat = tok(open(f"{BENCH}/chat_heldout.txt").read(), add_special_tokens=False).input_ids
    rows = [json.loads(l) for l in open(traces)]

    def mathwin(r):
        p = tok.apply_chat_template([{"role": "user", "content": r["q"]}], tokenize=False, add_generation_prompt=True,
                                    enable_thinking=True)
        return tok(p + "<|channel>thought\n" + r["thinking"] + "<channel|>" + r["reply"],
                   add_special_tokens=False).input_ids[:768]

    train = [[tok.bos_token_id] + wiki[i * 511:(i + 1) * 511] for i in range(0, 16, 2)] + \
            [[tok.bos_token_id] + chat[i * 511:(i + 1) * 511] for i in range(0, 32, 2)] + [mathwin(r) for r in rows[::14]]
    lines = T.fit(train, route, keep, zero)
    print(f"fitted {len(lines)} background lines on {len(train)} windows ({time.time() - t0:.0f} s)", flush=True)
T.set(route=route, keep=keep, zero=zero, pick="gate", lines=lines)

tok.padding_side = "left"
meta = {"name": name, "route": sorted(route), "keep": keep, "zero": zero, "pick": "gate + line" if route else "-",
        "sampling": {"temperature": 1.0, "top_k": 64, "top_p": 0.95, "max_tokens": 600}, "seeds": SEEDS,
        "model": "e4b-qat 16-bit (tools/mind/census/e4b_mac.py), Mac GPU, HF generate", "date": time.strftime("%Y-%m-%dT%H:%M")}
res = {}
with open(os.path.join(out, f"{name}.jsonl"), "w") as f, torch.no_grad():
    for qi, (key, q) in enumerate(QUESTIONS.items()):
        prompt = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
        for b0 in range(0, SEEDS, BATCH):
            n = min(BATCH, SEEDS - b0)
            enc = tok([prompt] * n, return_tensors="pt", add_special_tokens=False).to("mps")
            torch.manual_seed(1000 + 100 * qi + b0)
            t = time.time()
            o = model.generate(**enc, max_new_tokens=600, do_sample=True, temperature=1.0, top_k=64, top_p=0.95,
                               pad_token_id=tok.pad_token_id)
            for j, seq in enumerate(o[:, enc.input_ids.shape[1]:]):
                text = tok.decode(seq, skip_special_tokens=False).replace("<pad>", "")
                done = "<turn|>" in text or tok.eos_token in text
                text = text.split("<turn|>")[0].replace(tok.eos_token, "").strip()
                r = {"q": key, "n": b0 + j + 1, "reply": text, "finish": "stop" if done else "length",
                     "tokens": len(tok(text, add_special_tokens=False).input_ids)}
                res.setdefault(key, []).append(r)
                f.write(json.dumps(r) + "\n")
            f.flush()
            print(f"  {name} {key} {b0 + n}/{SEEDS} ({time.time() - t:.0f} s)", flush=True)

md = [f"# chat check: {name}", "", *(f"- {k}: {v}" for k, v in meta.items()), "",
      "Nothing below is scored. Every reply is shown in full, as the model wrote it.", ""]
for key, rs in res.items():
    md += [f"## {key}", "", f"> {QUESTIONS[key]}", ""]
    for r in rs:
        md += [f"### {key} {r['n']} ({r['tokens']} tokens{', CUT OFF' if r['finish'] == 'length' else ''})", "",
               *("> " + l for l in r["reply"].splitlines()), ""]
open(os.path.join(out, f"{name}.md"), "w").write("\n".join(md) + "\n")
print("CHAT_DONE", f"{time.time() - t0:.0f} s")
