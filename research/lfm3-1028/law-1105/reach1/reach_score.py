#!/usr/bin/env python3
"""GOAL-reach-v1. usage: reach_score.py <dir with reach-g.jsonl, reach-s1.jsonl> [corrections.json]. Corrections: {"run:id": "right" or "wrong"}"""
import json, os, sys
from collections import Counter
BASE = os.environ["LITTLEGUY_ROOT"] + "/.run/results/mind/lfm3-1028/science"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE + "/grades-v4")
from keyed_grade import keyed_grade
d = sys.argv[1]
fix = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else {}
RUNS = ("g", "s1")
ids = [json.loads(l)["id"] for l in open(os.path.join(HERE, "..", "mix1", "sci-held-81.jsonl"))]
hard = set(json.load(open(os.path.join(HERE, "f5-wrong-twice-ids-v1.json"))))
load = lambda p: {r["id"]: r for r in map(json.loads, open(p))}
R = {u: load(os.path.join(d, f"reach-{u}.jsonl")) for u in RUNS}
F = {u: load(f"{BASE}/final-f5-1104/f5-{u}.jsonl") for u in ("g", "s1", "s2", "s3")}
S = {u: load(f"{BASE}/final-lfm3-1104/lfm3-base-{u}.jsonl") for u in RUNS}
ok = [i for i in ids if all(i in R[u] for u in RUNS)]
print(f"problems finished in both reach runs: {len(ok)} of {len(ids)}, runs {len(ok)*2}; wrong-twice problems among them {len(hard & set(ok))} of {len(hard)}")
T = lambda r: (r.get("think_tokens") or 0) + (r.get("reply_tokens") or 0)
def G(r, run=None):
    if run and f"{run}:{r['id']}" in fix and r.get("finish") == "stop":
        return fix[f"{run}:{r['id']}"]
    x = keyed_grade(r.get("finish"), r.get("reply"), r["id"], r["answer"], r.get("scale"))
    return "wrong" if x == "sign" else x
def tally(rows, cap=None, run_fix=False):
    c = Counter()
    for u, r in rows:
        g = G(r, u if run_fix else None)
        if cap is not None and T(r) > cap:
            g = "cut"
        c[g] += 1
    return c
def line(label, c):
    N = sum(c.values()) or 1
    extra = "".join(f" {k} {v}" for k, v in c.items() if k not in ("right", "wrong", "cut"))
    fin = c["right"] + c["wrong"]
    print(f"{label:<26} right {c['right']:>3} ({100*c['right']/N:.1f}%) wrong {c['wrong']:>3} ({100*c['wrong']/N:.1f}%) cut {c['cut']:>3} ({100*c['cut']/N:.1f}%) right when finished {100*c['right']/(fin or 1):.1f}%{extra}")
A = [(u, R[u][i]) for u in RUNS for i in ok]
H = [(u, R[u][i]) for u in RUNS for i in ok if i in hard]
print("\nALL 81"); c = tally(A); line("reach machine", c)
if fix:
    line("reach hand", tally(A, run_fix=True))
line("f5 same two runs", tally([(u, F[u][i]) for u in RUNS for i in ok]))
line("f5 all four runs", tally([(u, F[u][i]) for u in F for i in ok]))
print("\nWRONG-TWICE PROBLEMS"); h = tally(H); line("reach machine", h)
if fix:
    line("reach hand", tally(H, run_fix=True))
line("f5 same two runs", tally([(u, F[u][i]) for u in RUNS for i in ok if i in hard]))
print("\nCURVE, a run past the cap counts as cut")
for cap in (6000, 8000, 12000, 16000, 20000, 24000):
    line(f"all 81 at {cap}", tally(A, cap))
for cap in (6000, 8000, 12000, 16000, 20000, 24000):
    line(f"wrong-twice at {cap}", tally(H, cap))
print("\nPRICE")
tr = sum(T(r) for _, r in A); tf = sum(T(F[u][i]) for u in RUNS for i in ok); ts = sum(T(S[u][i]) for u in RUNS for i in ok if i in S[u])
print(f"tokens: reach {tr}, f5 same two runs {tf}, ratio {tr/tf:.2f}x; saved untouched capped at 6,000: {ts}")
print(f"median tokens per run: reach {sorted(T(r) for _, r in A)[len(A)//2]}, f5 {sorted(T(F[u][i]) for u in RUNS for i in ok)[len(A)//2]}")
print("\nMOVES, f5 same run to reach")
m = Counter((G(F[u][i]), G(R[u][i])) for u in RUNS for i in ok)
for k in sorted(m):
    print(f"  {k[0]:>6} to {k[1]:<6} {m[k]}")
print("  on wrong-twice only:")
m = Counter((G(F[u][i]), G(R[u][i])) for u in RUNS for i in ok if i in hard)
for k in sorted(m):
    print(f"  {k[0]:>6} to {k[1]:<6} {m[k]}")
print("\nSAVED 6,000-CAP RUNS AGAINST THESE, up to 6,000")
same_status = same_grade = same_text = n = 0
for u in RUNS:
    for i in ok:
        if i not in S[u]:
            continue
        a, b = S[u][i], R[u][i]; n += 1
        ga, gb = G(a), ("cut" if T(b) > 6000 else G(b))
        same_grade += ga == gb
        same_status += (ga == "cut") == (gb == "cut")
        same_text += ((a.get("thinking") or "")[:2000]) == ((b.get("thinking") or "")[:2000])
print(f"of {n} pairs: same finished-or-cut {same_status}, same grade {same_grade}, first 2,000 thinking characters identical {same_text}")
if len(ok) != len(ids) or len(hard & set(ok)) != len(hard):
    sys.exit("INCOMPLETE: no verdict on partial rows")
N = len(A); cu = c["cut"] / N; r = c["right"] / N
v = "STILL CUTTING" if cu > .25 else "THE TARGET IS IN REACH" if r >= .8 else "MORE IS IN REACH, NOT ALL OF IT" if r >= .69 else "SAME CEILING AS F5"
NH = len(H); p = h["right"] / (NH or 1)
w = "ROOM FIXES F5'S WRONGS (story A)" if p >= .5 else "THEY'RE OUT OF REACH (story B)" if p <= .2 else "PARTLY"
print(f"\nVERDICT ALL 81: {v} (R {100*r:.1f}%, cut {100*cu:.1f}%)")
print(f"VERDICT WRONG-TWICE: {w} (P {100*p:.1f}%)")
out = os.path.join(HERE, "wrongs-to-read-reach-v1.txt")
with open(out, "w") as f:
    for u in RUNS:
        for i in ok:
            r = R[u][i]
            if G(r) == "wrong":
                f.write(f"=== {u}:{i} hard={i in hard} key {r['answer']} scale {r.get('scale')} unit {r.get('unit')} tokens {T(r)} f5 same run {G(F[u][i])}\nQ: {r['q'][:400]!r}\nEND: {(r.get('reply') or '')[-600:]!r}\n\n")
