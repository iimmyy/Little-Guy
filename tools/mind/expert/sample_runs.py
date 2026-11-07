#!/usr/bin/env python3
"""Several sampled answers per problem (thinking on) through a running llama-server, to see how short E4B's own
SHORTEST correct answers are next to his usual length: the room a "learn from your own best" training run has
(Self-Training Elicits Concise Reasoning, KAIST 2025). Same neutral answer line as run_baseline.py.

Sampling = Gemma's defaults (temperature 1.0, top_k 64, top_p 0.95); seed = 1000 + sample number, so runs repeat.

usage: sample_runs.py <problems.jsonl> <out.jsonl> [n=8] [port=8790] [parallel=8]      resumes on (id, sample)
"""
import json, os, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

src, out = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 8
port = sys.argv[4] if len(sys.argv) > 4 else "8790"
par = int(sys.argv[5]) if len(sys.argv) > 5 else 8
URL = f"http://127.0.0.1:{port}"
TAIL = "\n\nEnd with a line: Answer: <your answer>"


def post(path, body):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=1800))


def ntok(text):
    return len(post("/tokenize", {"content": text})["tokens"]) if text else 0


def solve(job):
    it, s = job
    t = time.time()
    d = post("/v1/chat/completions", {"messages": [{"role": "user", "content": it["q"] + TAIL}],
                                      "chat_template_kwargs": {"enable_thinking": True},
                                      "temperature": 1.0, "top_k": 64, "top_p": 0.95, "seed": 1000 + s,
                                      "max_tokens": 4000})
    msg = d["choices"][0]["message"]
    text = msg["content"]
    think, _, reply = text.partition("<channel|>") if "<channel|>" in text else ("", "", text)
    think = (msg.get("reasoning_content") or think.replace("<|channel>thought", "", 1)).strip()   # newer servers split it out
    return {**it, "sample": s, "thinking": think, "reply": reply.strip(), "think_tokens": ntok(think),
            "reply_tokens": ntok(reply), "total_tokens": d["usage"]["completion_tokens"],
            "finish": d["choices"][0]["finish_reason"], "seconds": round(time.time() - t, 1)}


items = [json.loads(l) for l in open(src)]
done = {(r["id"], r["sample"]) for r in map(json.loads, open(out))} if os.path.exists(out) else set()
todo = [(it, s) for s in range(n) for it in items if (it["id"], s) not in done]
print(f"{len(todo)} to do ({len(done)} already done)", flush=True)
t0 = time.time()
with open(out, "a") as f, ThreadPoolExecutor(par) as pool:
    for k, r in enumerate(pool.map(solve, todo), 1):
        f.write(json.dumps(r) + "\n"); f.flush()
        if k % 50 == 0:
            print(f"{k}/{len(todo)} ({time.time() - t0:.0f} s)", flush=True)
print("SAMPLES_DONE")
