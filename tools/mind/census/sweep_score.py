#!/usr/bin/env python3
"""Scoreboard for gain builds on one problem set. Every measure at once, because they tell different stories:
right answers within a word budget, total thinking words, thinking on problems every model gets right, wrong answers, cut-offs, and replies carrying stray closing tags (junk: read them by hand).
Runs of one model are paired with the base model's runs by position (greedy with greedy, s1 with s1).
usage: sweep_score.py <out.txt> base=g.jsonl,s1.jsonl name=g.jsonl,s1.jsonl ...
"""
import json, os, sys

sys.path.insert(0, os.path.expanduser("~/work/ligands/tools/mind/expert"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "expert"))
ARGV, sys.argv[1:] = sys.argv[1:], []
from grade import right_reply

out, sets = ARGV[0], [a.split("=", 1) for a in ARGV[1:]]
B = (500, 1000, 1500, 2000, 3000, 4000, 6000)
ok = lambda r: r["finish"] == "stop" and bool(right_reply(r, r["reply"]))
wrong = lambda r: r["finish"] == "stop" and not ok(r)
R = {n: [{r["id"]: r for r in map(json.loads, open(f))} for f in fs.split(",")] for n, fs in sets}
base = sets[0][0]
ids = set(R[base][0])
for n in R:
    for x in R[n]:
        ids &= set(x)
ids = sorted(ids)
toks = lambda r: r.get("think_tokens", 0) + r.get("reply_tokens", 0)
L = [f"{len(ids)} problems common to all, {len(R[base])} runs each; right within a word budget (thinking + reply), by budget {B}"]
always = [i for i in ids if all(ok(x[i]) for n in R for x in R[n])]
b_tot = sum(x[i].get("think_tokens", 0) for x in R[base] for i in ids)
b_all = sum(toks(x[i]) for x in R[base] for i in ids)
b_alw = sum(x[i].get("think_tokens", 0) for x in R[base] for i in always) or 1
for n, X in R.items():
    rows = [x[i] for x in X for i in ids]
    curve = [sum(ok(r) and toks(r) <= b for r in rows) for b in B]
    t = sum(r.get("think_tokens", 0) for r in rows)
    ta = sum(x[i].get("think_tokens", 0) for x in X for i in always)
    bet = sum(ok(x[i]) and not ok(y[i]) for x, y in zip(X, R[base]) for i in ids)
    wor = sum(not ok(x[i]) and ok(y[i]) for x, y in zip(X, R[base]) for i in ids)
    all_ = sum(toks(r) for r in rows)
    junk = sum(("</think>" in r["reply"] or "</tool_call>" in r["reply"] or "</parameter>" in r["reply"]) for r in rows)
    L.append(f"  {n:8s} right {' '.join(f'{c:4d}' for c in curve)} | wrong {sum(wrong(r) for r in rows):3d} cut {sum(r['finish'] != 'stop' for r in rows):3d} junk {junk:3d} | words {all_ / b_all - 1:+.1%} (thinking + reply), thinking {t / b_tot - 1:+.1%}, {ta / b_alw - 1:+.1%} on {len(always)} always-right | vs {base}: better {bet}, worse {wor}")
open(out, "w").write("\n".join(L) + "\n")
print("\n".join(L))
