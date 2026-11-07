#!/usr/bin/env python3
"""GOAL-flip-v1 arms through llama-server, same text and greedy, faster than the Python loop. Reading only.
usage: flip_srv.py <hf tokenizer> <steps.jsonl> <out.jsonl> <port> <workers> [limit]"""
import json, sys, time, threading, urllib.request
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer
hf, steps, outf, port, workers = sys.argv[1:6]
limit = int(sys.argv[6]) if len(sys.argv) > 6 else 10 ** 9
tok = AutoTokenizer.from_pretrained(hf)
OPEN, CLOSE, MAXTOK, NEW = "<|channel>thought\n", "<channel|>", 12000, 1500
def head(q):
    return tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
def go(text):
    ids = tok(text, add_special_tokens=False).input_ids
    if len(ids) > MAXTOK:
        return dict(text=None, tokens=0, done=False, prompt=len(ids))
    body = dict(prompt=ids, n_predict=NEW, temperature=0, top_k=1, repeat_penalty=1.0, cache_prompt=False, return_tokens=True, stream=False, seed=0)
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=3600))
    t = r.get("tokens") or []
    return dict(text=tok.decode(t, skip_special_tokens=False), tokens=len(t), done=r.get("stop_type") == "eos", prompt=len(ids), stop=r.get("stop_type"))
lock, n, t0 = threading.Lock(), [0], time.time()
w = open(outf, "w")
def one(j):
    h = head(j["q"]) + OPEN + j["thought"].rstrip()
    row = dict(id=j["id"], set=j["set"], label=j["label"], answer=j["answer"], scale=j.get("scale"), arms={})
    for name, tail in (("base", CLOSE), ("now", "\n\nNow, "), ("wait", "\n\nWait, ")):
        row["arms"][name] = go(h + tail)
    with lock:
        w.write(json.dumps(row) + "\n"); w.flush(); n[0] += 1
        if n[0] % 5 == 0:
            print(n[0], round(time.time() - t0), flush=True)
J = [json.loads(l) for l in open(steps)][:limit]
with ThreadPoolExecutor(int(workers)) as ex:
    list(ex.map(one, J))
print("FLIP_DONE", n[0], round(time.time() - t0), flush=True)
