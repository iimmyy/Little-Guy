#!/usr/bin/env python3
"""GOAL-refade-v1, screen. usage: screen_score.py <dir> [arm ...] (base arm is 'base')"""
import json, os, sys
from collections import Counter
ROOT = os.environ["LITTLEGUY_ROOT"] + ""
BASE = f"{ROOT}/.run/results/mind/lfm3-1028/science"
sys.path.insert(0, BASE + "/grades-v4"); sys.path.insert(0, f"{ROOT}/tools/mind/expert")
ARGV, sys.argv[1:] = sys.argv[1:], []
from keyed_grade import keyed_grade
from grade import right_reply
d, arms = ARGV[0], ARGV[1:] or ["base", "conv05", "conv15", "attn05", "attn15"]
def Gs(r):
    x = keyed_grade(r.get("finish"), r.get("reply"), r["id"], r["answer"], r.get("scale")); return "wrong" if x == "sign" else x
def Gm(r):
    return "cut" if r["finish"] != "stop" else ("right" if right_reply(r, r["reply"]) else "wrong")
load = lambda p: {r["id"]: r for r in map(json.loads, open(p))}
T = lambda r: (r.get("think_tokens") or 0) + (r.get("reply_tokens") or 0)
R = {}
for a in arms:
    pS, pM = os.path.join(d, f"screen-{a}-sci.jsonl"), os.path.join(d, f"screen-{a}-math.jsonl")
    if os.path.exists(pS) and os.path.exists(pM):
        R[a] = (load(pS), load(pM))
b = R["base"]; ids = (sorted(b[0]), sorted(b[1]))
tot = lambda a: Counter(Gs(R[a][0][i]) for i in ids[0]) + Counter(Gm(R[a][1][i]) for i in ids[1])
w = lambda a: sum(T(R[a][0][i]) for i in ids[0]) + sum(T(R[a][1][i]) for i in ids[1])
c0, w0 = tot("base"), w("base")
print(f"{len(ids[0])} science + {len(ids[1])} math")
for a in R:
    miss = [i for k in (0, 1) for i in ids[k] if i not in R[a][k]]
    if miss:
        print(a, "INCOMPLETE, missing", len(miss)); continue
    c, wd = tot(a), w(a)
    s = Counter(Gs(R[a][0][i]) for i in ids[0]); m = Counter(Gm(R[a][1][i]) for i in ids[1])
    dr, dw, dt = c["right"] - c0["right"], c["wrong"] - c0["wrong"], wd / w0 - 1
    ok = (dr >= 3 and dw <= 1 and dt <= .05) or (dr >= -1 and dw <= 1 and dt <= -.10)
    print(f"{a:<8} right {c['right']:>3} wrong {c['wrong']:>2} cut {c['cut']:>2} words {wd:>8} {100*dt:+6.1f}% | science {s['right']}/{s['wrong']}/{s['cut']} math {m['right']}/{m['wrong']}/{m['cut']} | vs base right {dr:+d} wrong {dw:+d} | {'GOES ON' if ok and a != 'base' else ''}")
