#!/usr/bin/env python3
"""GOAL-flip-v1, Wait and Now only, through llama-server. Reading only. usage: flip_srv2.py <hf tokenizer> <steps.jsonl> <out.jsonl> <port> <workers>"""
import json, sys, time, threading, urllib.request
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer
hf, steps, outf, port, workers = sys.argv[1:6]
tok = AutoTokenizer.from_pretrained(hf)
OPEN, MAXTOK, NEW = "<|channel>thought\n", 4600, 1500
def head(q):
    return tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
J = [json.loads(l) for l in open(steps)]
J = sorted(J, key=lambda j: j["set"] != "wrong")
JOBS = [(j, a, t) for j in J for a, t in (("wait", "\n\nWait, "), ("now", "\n\nNow, "))]
lock, n, t0, words = threading.Lock(), [0], time.time(), [0]
w = open(outf, "w")
def one(x):
    j, arm, tail = x
    ids = tok(head(j["q"]) + OPEN + j["thought"].rstrip() + tail, add_special_tokens=False).input_ids
    row = dict(id=j["id"], set=j["set"], label=j["label"], answer=j["answer"], scale=j.get("scale"), arm=arm, prompt=len(ids))
    if len(ids) > MAXTOK:
        row.update(text=None, tokens=0, done=False, stop="too long")
    else:
        body = dict(prompt=ids, n_predict=NEW, temperature=0, top_k=1, repeat_penalty=1.0, cache_prompt=False, return_tokens=True, stream=False)
        req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", json.dumps(body).encode(), {"Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=3600))
        t = r.get("tokens") or []
        row.update(text=tok.decode(t, skip_special_tokens=False), tokens=len(t), done=r.get("stop_type") == "eos", stop=r.get("stop_type"))
    with lock:
        w.write(json.dumps(row) + "\n"); w.flush(); n[0] += 1; words[0] += row["tokens"]
        if n[0] % 10 == 0:
            print(n[0], len(JOBS), "tokens", words[0], round(time.time() - t0), flush=True)
with ThreadPoolExecutor(int(workers)) as ex:
    list(ex.map(one, JOBS))
print("FLIP_DONE", n[0], round(time.time() - t0), flush=True)
