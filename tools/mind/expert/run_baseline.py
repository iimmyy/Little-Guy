#!/usr/bin/env python3
"""Plain E4B (no neuron changes) on the expert-route problem set, thinking on, greedy, through a running llama-server
(e4b-grid4 on the Mac: same top word as the 16-bit 99.7% of the time). Saves the thinking and the reply separately,
with their token counts, so routes and waste can be read and counted.

The only thing added to each question is a neutral "End with a line: Answer: ..." so replies can be graded by code;
it names no route.

usage: run_baseline.py <problems.jsonl> <out.jsonl> [port] [parallel]      resumes: skips ids already in out.jsonl
"""
import json, os, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

src, out = sys.argv[1], sys.argv[2]
port = sys.argv[3] if len(sys.argv) > 3 else "8790"
par = int(sys.argv[4]) if len(sys.argv) > 4 else 4
URL = f"http://127.0.0.1:{port}"
TAIL = "\n\nEnd with a line: Answer: <your answer>"
MAXNEW = int(os.environ.get("MAXNEW", 6000))            # 8192 for Qwen3-4B-Thinking (Steer2Edit's eval budget)
# SAMPLE="T,top_p,top_k" SEED=n: sampled repeat runs past the greedy noise floor (left out: greedy)
SAMP = ({"temperature": float(os.environ["SAMPLE"].split(",")[0]), "top_p": float(os.environ["SAMPLE"].split(",")[1]),
         "top_k": int(os.environ["SAMPLE"].split(",")[2]), "seed": int(os.environ.get("SEED", 0))}
        if os.environ.get("SAMPLE") else {"temperature": 0})


def post(path, body):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=900))


def ntok(text):
    return len(post("/tokenize", {"content": text})["tokens"]) if text else 0


def solve(it):
    t = time.time()
    d = post("/v1/chat/completions", {"messages": [{"role": "user", "content": it["q"] + TAIL}],
                                      "chat_template_kwargs": {"enable_thinking": True},
                                      **SAMP, "max_tokens": MAXNEW})
    msg = d["choices"][0]["message"]
    text = msg.get("content") or ""
    if msg.get("reasoning_content") is not None:          # servers that split <think> themselves (Qwen3, LFM3)
        think, reply = msg["reasoning_content"].strip(), text
    else:                                                 # Gemma 4 with --reasoning-format none: split by hand
        think, _, reply = text.partition("<channel|>") if "<channel|>" in text else ("", "", text)
        think = think.replace("<|channel>thought", "", 1).strip()
    return {**it, "thinking": think, "reply": reply.strip(), "think_tokens": ntok(think), "reply_tokens": ntok(reply),
            "total_tokens": d["usage"]["completion_tokens"], "finish": d["choices"][0]["finish_reason"],
            "seconds": round(time.time() - t, 1)}


items = [json.loads(l) for l in open(src)]
done = {json.loads(l)["id"] for l in open(out)} if os.path.exists(out) else set()
todo = [it for it in items if it["id"] not in done]
print(f"{len(todo)} to do ({len(done)} already done)", flush=True)
with open(out, "a") as f, ThreadPoolExecutor(par) as pool:
    for k, r in enumerate(pool.map(solve, todo), 1):
        f.write(json.dumps(r) + "\n"); f.flush()
        print(f"{k}/{len(todo)} {r['id']}: think {r['think_tokens']} + reply {r['reply_tokens']} tokens, "
              f"{r['finish']}, {r['seconds']} s", flush=True)
print("BASELINE_DONE")
