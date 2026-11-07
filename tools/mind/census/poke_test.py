#!/usr/bin/env python3
"""The poke test: is he done here? Key-free, no repeat runs, works on any set. Take a paragraph break anywhere in a run, close the thinking there by force, and let him write on.
Done: he answers. Not done: he carries on thinking, and the close word comes up again in what he writes. Run on a build whose clock is live, so a carried-on thought shows as a repeated close word.
usage: poke_test.py <server url> <hf folder> <out.jsonl> <label>=<runs.jsonl>,<n breaks> [...] [skip=eval.jsonl n=500 par=16 seed=11 sample=1.0,0.95,20 perrun=3]"""
import json, os, random, re, sys
from concurrent.futures import ThreadPoolExecutor
import urllib.request
from transformers import AutoTokenizer
url, hf, out = sys.argv[1:4]
opt = dict(a.split("=", 1) for a in sys.argv[4:])
skip = {json.loads(l)["id"] for l in open(opt.pop("skip"))} if "skip" in opt else set()
N, PAR, SEED, PER = int(opt.pop("n", 500)), int(opt.pop("par", 16)), int(opt.pop("seed", 11)), int(opt.pop("perrun", 3))
t_, tp, tk = opt.pop("sample", "1.0,0.95,20").split(",")
tok = AutoTokenizer.from_pretrained(hf, trust_remote_code=True)
END = "</think>"
rng = random.Random(7)
jobs = []
for label, spec in opt.items():
    path, n = spec.rsplit(",", 1)
    J = [json.loads(l) for l in open(path)]
    J = [j for j in J if j["id"] not in skip and j.get("thinking") and len(j["thinking"].split()) >= 40]
    rng.shuffle(J)
    got = 0
    for j in J:
        if got >= int(n):
            break
        th = j["thinking"].rstrip()
        marks = [m.start() for m in re.finditer(r"\n\n", th) if len(th[:m.start()].split()) >= 20]
        marks = marks + [len(th)]      # the point he reached on his own counts too
        rng.shuffle(marks)
        for c in sorted(marks[:PER]):
            jobs.append(dict(set=label, id=j["id"], run=j.get("run") or os.path.basename(path), src=j.get("src"), q=j["q"], c=c, words=len(th[:c].split()), at_end=c == len(th), of=len(th.split()), own_finish=j.get("finish"), thinking=th[:c]))
            got += 1
done = set()
if os.path.exists(out):
    done = {(r["set"], r["run"], r["id"], r["c"]) for r in map(json.loads, open(out))}
todo = [j for j in jobs if (j["set"], j["run"], j["id"], j["c"]) not in done]
print(len(jobs), "pokes,", len(todo), "to do", flush=True)
def run(j):
    head = tok.apply_chat_template([{"role": "user", "content": j["q"]}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
    body = {"prompt": head + "\n" + j["thinking"] + "\n\n" + END, "n_predict": N, "temperature": float(t_), "top_p": float(tp), "top_k": int(tk), "seed": SEED, "cache_prompt": False}
    r = json.load(urllib.request.urlopen(urllib.request.Request(url + "/completion", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=3600))
    t = r["content"]
    o = {k: v for k, v in j.items() if k not in ("q",)}
    o.pop("thinking")
    o.update(tail=j["thinking"][-200:], after=t[:700], tags=t.count(END), first_tag=t.find(END), finished=r.get("stop_type") == "eos", new=int(r.get("tokens_predicted", 0)))
    return o
with open(out, "a") as f, ThreadPoolExecutor(PAR) as ex:
    for i, o in enumerate(ex.map(run, todo)):
        f.write(json.dumps(o) + "\n"); f.flush()
        if i % 50 == 0:
            print(i, o["set"], o["words"], "tags", o["tags"], flush=True)
print("POKE_DONE", flush=True)
