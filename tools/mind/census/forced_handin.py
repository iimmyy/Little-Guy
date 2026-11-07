#!/usr/bin/env python3
"""How much of his thinking does he need? (Immy 2026-11-03: "up to 50%, 60%". 32% of his thinking words sit in runs that are
cut off without ever writing the answer, so a stop after the answer cannot reach that.) Take his finished and cut-off runs,
and make him hand in at a word budget, at the last paragraph end before it, in his own commit wording. Runs already shorter
than the budget stay as they are. The score at each budget is what a perfect "time to hand in" could get with no other change:
right, wrong, unfinished, and thinking words. Nothing here changes the model. It is the baseline any build has to beat too,
since a server's budget flag can do the same thing from outside.
join=thus adds "Thus final" and lets him go on; join=end closes the thinking straight away.
usage: forced_handin.py <url> <hf> <out.jsonl> <runs.jsonl> [...] [budgets=500,1000,1500,2000,3000,4000 joins=thus,end par=16 maxnew=1200]
report: forced_handin.py report <out.txt> <rows.jsonl> <runs.jsonl> [...]
"""
import json, os, re, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.expanduser("~/work/ligands/tools/mind/expert"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "expert"))
ARGV_, sys.argv[1:] = sys.argv[1:], []                     # grade.py grades its own arguments on import
from grade import right_reply
sys.argv[1:] = ARGV_

END = "</think>"
JOIN = {"thus": "\n\nThus final", "end": "\n\n"}


def score(r):
    if r["finish"] != "stop":
        return "cut"
    return "right" if right_reply(r, r["reply"]) else "wrong"


if sys.argv[1] == "report":
    out, rowf, runs = sys.argv[2], sys.argv[3], sys.argv[4:]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.path.expanduser("~/.littleguy/hf/lfm3-1.7b"), trust_remote_code=True)
    rlen = lambda r: len(tok(r["reply"], add_special_tokens=False).input_ids)
    base = [dict(json.loads(l), src=f) for f in runs for l in open(f)]
    rows = {}
    for r in map(json.loads, open(rowf)):
        rows[(r["src"], r["id"], r["join"], r["budget"])] = r
    n = len(base)
    tb = sum(r["think_tokens"] + r["reply_tokens"] for r in base)
    c = {k: sum(score(r) == k for r in base) for k in ("right", "wrong", "cut")}
    L = [f"his own runs: {n}, right {c['right']}, wrong {c['wrong']}, cut {c['cut']}, thinking + reply words {tb} (mean {tb / n:.0f})"]
    for j, b in sorted({(k[2], k[3]) for k in rows}):
        C = {"right": 0, "wrong": 0, "cut": 0}
        tot = 0
        gained = lost = 0
        for r in base:
            f = rows.get((r["src"], r["id"], j, b))
            if f is None:
                C[score(r)] += 1
                tot += r["think_tokens"] + r["reply_tokens"]
                continue
            s = score(f)
            C[s] += 1
            tot += f["think_tokens"] + rlen(f)
            gained += s == "right" and score(r) != "right"
            lost += s != "right" and score(r) == "right"
        L.append(f"  {j:5s} at {b:5d}: right {C['right']:3d}, wrong {C['wrong']:3d}, unfinished {C['cut']:3d}; thinking + reply words {tot} ({tot / tb - 1:+.0%}); right vs his own: gained {gained}, lost {lost}")
    open(out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    sys.exit()

from transformers import AutoTokenizer

args = [a for a in sys.argv[1:] if "=" not in a]
opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
url, hf, out, runs = args[0], args[1], args[2], args[3:]
BUD = [int(x) for x in opt.get("budgets", "500,1000,1500,2000,3000,4000").split(",")]
JOINS = opt.get("joins", "thus,end").split(",")
PAR, MAXNEW = int(opt.get("par", 16)), int(opt.get("maxnew", 1200))
tok = AutoTokenizer.from_pretrained(hf, trust_remote_code=True)
done = set()
if os.path.exists(out):
    for r in map(json.loads, open(out)):
        done.add((r["src"], r["id"], r["join"], r["budget"]))


def cut_at(th, n):
    """the thinking up to the last paragraph end inside n tokens (else the last line end, else the last sentence end)"""
    enc = tok(th, add_special_tokens=False, return_offsets_mapping=True)
    if len(enc.input_ids) <= n:
        return None, len(enc.input_ids)
    c = enc.offset_mapping[n - 1][1]
    part = th[:c]
    for sep in ("\n\n", "\n", ". "):
        k = part.rfind(sep)
        if k > c * 0.5:
            part = part[:k].rstrip()
            break
    return part, len(tok(part, add_special_tokens=False).input_ids)


jobs = []
for f in runs:
    for t in map(json.loads, open(f)):
        for b in BUD:
            part, used = cut_at(t["thinking"], b)
            if part is None:
                continue
            for j in JOINS:
                if (f, t["id"], j, b) not in done:
                    head = tok.apply_chat_template([{"role": "user", "content": t["q"]}], tokenize=False, add_generation_prompt=True, enable_thinking=True) + "\n"
                    jobs.append({"src": f, "id": t["id"], "q": t["q"], "answer": t["answer"], "cat": t.get("cat"), "join": j, "budget": b, "kept": used,
                                 "prompt": head + part + JOIN[j] + (END if j == "end" else "")})
print(f"{len(jobs)} hand-ins to run", flush=True)


def ask(j):
    body = json.dumps({"prompt": j["prompt"], "n_predict": MAXNEW, "cache_prompt": False, "temperature": 0.0, "top_k": 1}).encode()
    res = json.load(urllib.request.urlopen(urllib.request.Request(url + "/completion", body, {"Content-Type": "application/json"}), timeout=3600))
    return j, res["content"], res.get("stop_type")


with open(out, "a") as fo, ThreadPoolExecutor(PAR) as pool:
    for k, (j, text, stop) in enumerate(pool.map(ask, jobs), 1):
        if j["join"] == "end":
            think, sep, reply = "", True, text
        else:
            think, sep, reply = text.partition(END)
        row = {x: j[x] for x in ("src", "id", "q", "answer", "cat", "join", "budget", "kept")}
        row.update(reply=reply.strip(), more=think, finish="stop" if sep and stop in ("eos", "word") else "length",
                   think_tokens=j["kept"] + len(tok(think, add_special_tokens=False).input_ids))
        fo.write(json.dumps(row) + "\n")
        fo.flush()
        if k % 50 == 0:
            print(f"{k}/{len(jobs)}", flush=True)
print("HANDIN_DONE")
