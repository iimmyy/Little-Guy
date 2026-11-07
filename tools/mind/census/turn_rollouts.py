#!/usr/bin/env python3
"""Where did a failed run go wrong? (Immy 2026-11-02: a "smells wrong" trigger, built two ways, needs real wrong-turn
moments to learn from.) Practice problems only: the night test problems are never used to build anything.

For every practice problem the untouched model solves only sometimes (split-hunt: 4 sampled runs each), take each
failed run, stop its thinking at a few sentence ends along the way, and let the same model finish from there several
times with a fresh word budget. The share of finishes that reach the right answer is how alive that moment still is.
The wrong turn is where it falls from "usually" to "never". The solved runs of the same problems get the same
treatment as the reference (they should stay alive to the end).

usage: turn_rollouts.py <server url> <out.jsonl> <hf folder> <runs.jsonl ...>
                        [cuts=6 n=4 budget=5000 par=16 sample=0.7,0.95,40 model=lfm3|e4b fail=any|wrong]
                        resumes: skips (id, run, cut, seed) done
fail=wrong (E4B, which hands in confident wrong answers): the failed runs are the finished-wrong ones, on every problem
that has one; the problem's right runs are the reference, and cut-off runs are left out.
"""
import json, os, re, sys, urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

from transformers import AutoTokenizer

ARGV, sys.argv[1:] = sys.argv[1:], []                     # grade.py reads the command line when imported
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "expert"))
from grade import right_reply

url, out, hf = ARGV[:3]
runs = [a for a in ARGV[3:] if "=" not in a]
opt = dict(a.split("=", 1) for a in ARGV[3:] if "=" in a)
CUTS, N, BUDGET, PAR = int(opt.get("cuts", 6)), int(opt.get("n", 4)), int(opt.get("budget", 5000)), int(opt.get("par", 16))
T, TP, TK = opt.get("sample", "0.7,0.95,40").split(",")
tok = AutoTokenizer.from_pretrained(hf, trust_remote_code=True)
TAIL = "\n\nEnd with a line: Answer: <your answer>"
OPEN, END = {"lfm3": ("\n", "</think>"), "e4b": ("<|channel>thought\n", "<channel|>")}[opt.get("model", "lfm3")]
WRONG = opt.get("fail", "any") == "wrong"

rows = defaultdict(list)                                   # id -> [(run name, row)]
for f in runs:
    for l in open(f):
        r = json.loads(l)
        rows[r["id"]].append((os.path.basename(f).replace(".jsonl", ""), r))
ok = lambda r: r["finish"] == "stop" and bool(right_reply(r, r["reply"]))
if WRONG:
    some = {i: [(n, r) for n, r in v if r["finish"] == "stop"] for i, v in rows.items()
            if any(r["finish"] == "stop" and not ok(r) for _, r in v)}
    print(f"{len(some)} problems with a wrong hand-in, {sum(not ok(r) for v in some.values() for _, r in v)} wrong runs, "
          f"{sum(ok(r) for v in some.values() for _, r in v)} right runs as reference", flush=True)
else:
    some = {i: v for i, v in rows.items() if 0 < sum(ok(r) for _, r in v) < len(v)}
    print(f"{len(some)} sometimes-solved problems, {sum(not ok(r) for v in some.values() for _, r in v)} failed runs", flush=True)


def cut_points(think):
    """CUTS sentence ends spread evenly through the thinking (the cut keeps the sentence's own end)"""
    ends = [m.end() for m in re.finditer(r"(?:[.!?](?= )|\n)", think)]
    picks = []
    for k in range(1, CUTS + 1):
        target = len(think) * k / (CUTS + 1)
        e = min(ends, key=lambda x: abs(x - target)) if ends else int(target)
        if e not in picks:
            picks.append(e)
    return picks


jobs = []
for i, v in some.items():
    for name, r in v:
        for c, e in enumerate(cut_points(r["thinking"])):
            for s in range(1, (N if not ok(r) else max(1, N // 2)) + 1):     # solved runs are only the reference
                jobs.append((i, name, r, c, e, s))
done = set()
if os.path.exists(out):
    done = {(d["id"], d["run"], d["cut"], d["seed"]) for d in map(json.loads, open(out))}
jobs = [j for j in jobs if (j[0], j[1], j[3], j[5]) not in done]
print(f"{len(jobs)} finishes to run ({len(done)} done)", flush=True)


def finish(job):
    i, name, r, c, e, s = job
    prompt = tok.apply_chat_template([{"role": "user", "content": r["q"] + TAIL}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=True)
    prefix = r["thinking"][:e]
    body = json.dumps({"prompt": prompt + OPEN + prefix, "n_predict": BUDGET, "cache_prompt": True,
                       "temperature": float(T), "top_p": float(TP), "top_k": int(TK), "seed": 1000 * c + s}).encode()
    res = json.load(urllib.request.urlopen(urllib.request.Request(url + "/completion", body, {"Content-Type": "application/json"}),
                                           timeout=3600))
    text = res["content"]
    more, _, reply = text.partition(END) if END in text else (text, "", "")
    stop = res.get("stop_type") in ("eos", "word") or bool(res.get("stopped_eos"))
    row = {**r, "reply": reply.strip(), "finish": "stop" if stop and END in text else "length"}
    if END not in text and stop and opt.get("model") == "e4b":   # no thought channel: run_baseline.py reads it all as the reply
        row = {**r, "reply": text.strip(), "finish": "stop"}
    return row, {"id": i, "run": name, "cut": c, "cut_frac": round(e / max(1, len(r["thinking"])), 3),
                 "prefix_tokens": len(tok(prefix, add_special_tokens=False).input_ids), "seed": s,
                 "finish": row["finish"], "added_tokens": res.get("timings", {}).get("predicted_n"),
                 "continuation": more, "reply": row["reply"]}, r


with open(out, "a") as f, ThreadPoolExecutor(PAR) as pool:
    for k, (row, d, r) in enumerate(pool.map(finish, jobs), 1):
        d["run_solved"], d["right"] = ok(r), ok(row)          # graded here: math_verify's timeout only works on the main thread
        f.write(json.dumps(d) + "\n"); f.flush()
        if k % 50 == 0:
            print(f"  {k}/{len(jobs)}", flush=True)

# summary: how alive each moment is, failed runs vs solved runs
R = [json.loads(l) for l in open(out)]
alive = defaultdict(list)
for d in R:
    alive[(d["run_solved"], d["cut"])].append(d["right"])
fin = Counter((d["run_solved"], d["finish"], d["right"]) for d in R)
print("finishes (run solved, finish, right):", dict(fin))
for solved in (False, True):
    print(("solved runs" if solved else "failed runs") + ": share of finishes right, by cut " +
          "  ".join(f"{c}:{sum(alive[(solved, c)]) / max(1, len(alive[(solved, c)])):.2f}" for c in range(CUTS)), flush=True)
print("ROLLOUTS_DONE")
