#!/usr/bin/env python3
"""Baseline answers from any patient (patients.py), in the same row format as pam.py's runs, so math_table.py and the
expert grader read them unchanged. Greedy, thinking on, "End with a line: Answer: <your answer>" added like pam.py.

usage: gen_patient.py <patient> <problems.jsonl> <out.jsonl> [per=N (first N per cat/kind; leave out for MATH) maxnew=6000
                      batch=6 dev=cuda tail=... (extra words, underscores as spaces)
                      sample=T,top_p,top_k seed=0 (the maker's sampling instead of greedy, e.g. LFM3 math: 1.0,0.95,20)
                      brake=gate|always brake_dirs=... brief_dirs=... det=8 push=12 alpha=1 ema=16 margin=0 spread=0 close=0 warm=0 clock=<clock_dirs.pt> (brake.py)
                      think=0 (thinking switched off in the chat template: the fair baseline for any brake)]
"""
import json, os, sys, time

import torch

sys.path.insert(0, os.path.dirname(__file__))
from patients import load

name, probs, out = sys.argv[1:4]
opt = dict(a.split("=", 1) for a in sys.argv[4:])
DEV, MAXNEW, B = opt.get("dev", "cuda"), int(opt.get("maxnew", 6000)), int(opt.get("batch", 6))
p = load(name, DEV)
tok = p.tok
items = [json.loads(l) for l in open(probs)]
pick, seen = [], {}
for it in items:
    key = (it.get("cat"), it.get("kind"))
    if "per" not in opt or seen.get(key, 0) < int(opt["per"]):
        pick.append(it); seen[key] = seen.get(key, 0) + 1
SAMPLE = [float(x) for x in opt["sample"].split(",")] if "sample" in opt else None
SEED = int(opt.get("seed", 0))
SAMPLE_KW = dict(do_sample=True, temperature=SAMPLE[0], top_p=SAMPLE[1], top_k=int(SAMPLE[2])) if SAMPLE else dict(do_sample=False)
TAIL = "\n\nEnd with a line: Answer: <your answer>" + (" " + opt["tail"].replace("_", " ") if "tail" in opt else "")
END = tok.convert_ids_to_tokens(p.end)
tok.padding_side = "left"
brake = None
if "brake" in opt:                                          # brake=gate|always, with brake_dirs=<overthink_dirs.pt>,
    from brake import Brake                                 # brief_dirs=<brief_dirs.pt> det=8 push=12 alpha=1 ema=16
    brake = Brake(p, torch.load(opt["brake_dirs"]), torch.load(opt["brief_dirs"]), det=int(opt.get("det", 8)),
                  push=int(opt.get("push", 12)), alpha=float(opt.get("alpha", 1)), mode=opt["brake"],
                  ema=int(opt.get("ema", 16)), margin=float(opt.get("margin", 0)),
                  spread=opt.get("spread", "0") == "1", close=float(opt.get("close", 0)), warm=int(opt.get("warm", 0)),
                  clock=torch.load(opt["clock"]) if "clock" in opt else None)
t = time.time()
with open(out, "w") as f, torch.no_grad():
    for c0 in range(0, len(pick), B):
        batch = pick[c0:c0 + B]
        prompts = [tok.apply_chat_template([{"role": "user", "content": it["q"] + TAIL}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=opt.get("think", "1") == "1") for it in batch]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
        if SAMPLE:
            torch.manual_seed(SEED + c0)
        o = p.model.generate(**enc, max_new_tokens=MAXNEW, pad_token_id=tok.pad_token_id, **SAMPLE_KW,
                             **({"logits_processor": [brake]} if brake else {}))
        braked = brake.stats.tolist() if brake else [[0, 0]] * len(batch)
        for it, seq, bk in zip(batch, o[:, enc.input_ids.shape[1]:], braked):
            text = tok.decode(seq, skip_special_tokens=False).replace(tok.pad_token, "")
            done = tok.eos_token in text
            text = text.split(tok.eos_token)[0]
            think, _, reply = text.partition(END) if END in text else ("", "", text)
            f.write(json.dumps({**it, "patient": name, "thinking": think.strip(), "reply": reply.strip(),
                                "think_tokens": len(tok(think, add_special_tokens=False).input_ids),
                                "reply_tokens": len(tok(reply, add_special_tokens=False).input_ids),
                                "finish": "stop" if done else "length",
                                **({"braked": bk[1] / max(1, bk[0])} if brake else {})}) + "\n")
        f.flush()
        print(f"  {c0 + len(batch)}/{len(pick)} in {time.time() - t:.0f} s", flush=True)
print("GEN_DONE")
