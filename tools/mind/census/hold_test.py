#!/usr/bin/env python3
"""Does waiting fix anything? (Immy's formula: breaks -> wait; unsettled -> keep working.) No build needed: takes the thinking a touched build closed on, and
continues it on a stock llama-server through raw /completion, with the same sampler and the same 6000-token limit as the original run.
arms: reply = close at once and write the reply again (control: how much do answers move from reply sampling alone)
      hold=K = the close word is banned for K more tokens, then he is free
      free = no ban at all: whatever model is on the server decides when to close
usage: hold_test.py <server url> <jobs.jsonl> <out.jsonl> <hf folder> arm=reply|hold=K|free [par=24 seed=11 sample=1.0,0.95,20 maxnew=6000]"""
import json, os, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer

url, jobs, out, hf = sys.argv[1:5]
opt = dict(a.split("=", 1) for a in sys.argv[5:])
ARM = opt["arm"] if opt["arm"] in ("reply", "free") else "hold"
K = int(opt["arm"].split("=")[1]) if "=" in opt["arm"] else int(opt.get("hold", 0))
PAR, SEED, MAXNEW = int(opt.get("par", 24)), int(opt.get("seed", 11)), int(opt.get("maxnew", 6000))
t_, tp, tk = opt.get("sample", "1.0,0.95,20").split(",")
tok = AutoTokenizer.from_pretrained(hf, trust_remote_code=True)
END = "</think>"
ENDID = tok(END, add_special_tokens=False).input_ids[0]
samp = {"temperature": float(t_), "top_p": float(tp), "top_k": int(tk), "seed": SEED, "cache_prompt": False}


def comp(prompt, n, ban=False):
    body = {"prompt": prompt, "n_predict": n, **samp}
    if ban:
        body["logit_bias"] = [[ENDID, False], [tok.eos_token_id, False]]    # with only the close word banned he ends the turn instead, thinking still open
    r = json.load(urllib.request.urlopen(urllib.request.Request(url + "/completion", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=3600))
    return r["content"], int(r.get("tokens_predicted", 0)), r.get("stop_type", "")


def run(j):
    t0 = time.time()
    head = tok.apply_chat_template([{"role": "user", "content": j["q"]}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
    th = j["thinking"]
    used = len(tok(th, add_special_tokens=False).input_ids)
    left = MAXNEW - used
    base = head + "\n" + th.rstrip()
    text, n = "", 0
    stop = ""
    if ARM == "hold":
        a, m, stop = comp(base, min(K, left), ban=True)
        text, n = a, m
    if ARM == "reply":
        text, n, stop = END, 1, ""     # he writes the close word straight after his last word, with no separator
    if left - n > 0 and stop != "eos":
        a, m, stop = comp(base + text, left - n)
        text += a
        n += m
    new_think, sep, reply = text.partition(END)
    closed = bool(sep)
    finished = closed and stop == "eos"
    row = {k: j[k] for k in j if k not in ("thinking", "reply")}
    row.update(arm=opt["arm"], orig_thinking_tokens=used, added_thinking=new_think, thinking=th.rstrip() + new_think, reply=reply if closed else "", finish="stop" if finished else "length",
               new_tokens=n, gen_tokens=used + n, added_think_tokens=len(tok(new_think, add_special_tokens=False).input_ids), reply_tokens=len(tok(reply, add_special_tokens=False).input_ids), seconds=time.time() - t0)
    return row


done = {(json.loads(l)["run"], json.loads(l)["id"]) for l in open(out)} if os.path.exists(out) else set()
todo = [j for j in map(json.loads, open(jobs)) if (j["run"], j["id"]) not in done]
print(len(todo), "to do", flush=True)
with open(out, "a") as f, ThreadPoolExecutor(PAR) as ex:
    for i, r in enumerate(ex.map(run, todo)):
        f.write(json.dumps(r) + "\n"); f.flush()
        if i % 20 == 0:
            print(i, r["id"][-6:], r["finish"], r["new_tokens"], flush=True)
print("HOLD_DONE", flush=True)
