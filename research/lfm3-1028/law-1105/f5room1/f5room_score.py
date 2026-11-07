#!/usr/bin/env python3
"""GOAL-f5room-v1. usage: f5room_score.py <dir with rows> [corrections.json]. Corrections: {"arm:run:id": "right" or "wrong"}"""
import json, os, sys
from collections import Counter
BASE = os.environ["LITTLEGUY_ROOT"] + "/.run/results/mind/lfm3-1028/science"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE + "/grades-v4")
from keyed_grade import keyed_grade
d = sys.argv[1]
fix = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else {}
ids = [json.loads(l)["id"] for l in open(os.path.join(HERE, "..", "mix1", "sci-held-81.jsonl"))]
hard = set(json.load(open(os.path.join(HERE, "..", "reach1", "f5-wrong-twice-ids-v1.json"))))
load = lambda p: {r["id"]: r for r in map(json.loads, open(p))}
R4 = ("g", "s1", "s2", "s3"); R2 = ("g", "s1")
F6 = {u: load(f"{BASE}/final-f5-1104/f5-{u}.jsonl") for u in R4}
FR = {u: load(os.path.join(d, f"f5room-{u}.jsonl")) for u in R4}
have_c5 = all(os.path.exists(os.path.join(d, f"c5room-{u}.jsonl")) for u in R2)
C = {u: load(os.path.join(d, f"c5room-{u}.jsonl")) for u in R2} if have_c5 else None
RE = {u: load(os.path.join(HERE, "..", "reach1", f"reach-{u}.jsonl")) for u in R2}
T = lambda r: (r.get("think_tokens") or 0) + (r.get("reply_tokens") or 0)
def G(r, key=None):
    if key and key in fix and r.get("finish") == "stop":
        return fix[key]
    x = keyed_grade(r.get("finish"), r.get("reply"), r["id"], r["answer"], r.get("scale"))
    return "wrong" if x == "sign" else x
def show(label, c, tok=None):
    N = sum(c.values()) or 1; fin = c["right"] + c["wrong"]
    t = f" tokens {tok}" if tok is not None else ""
    extra = "".join(f" {k} {v}" for k, v in c.items() if k not in ("right", "wrong", "cut"))
    print(f"{label:<24} right {c['right']:>3} ({100*c['right']/N:.1f}%) wrong {c['wrong']:>3} ({100*c['wrong']/N:.1f}%) cut {c['cut']:>3} ({100*c['cut']/N:.1f}%) right when finished {100*c['right']/(fin or 1):.1f}%{t}{extra}")
def tab(S, runs, ok=ids, cap=None, arm=None):
    c = Counter()
    for u in runs:
        for i in ok:
            r = S[u][i]; g = G(r, f"{arm}:{u}:{i}" if arm else None)
            if cap and T(r) > cap:
                g = "cut"
            c[g] += 1
    return c, sum(T(S[u][i]) for u in runs for i in ok)
ok4 = [i for i in ids if all(i in FR[u] for u in R4)]
print(f"f5 room rows: {len(ok4)} of {len(ids)} problems in all four runs")
print("\nF5"); cr, tr = tab(FR, R4, ok4); c6, t6 = tab(F6, R4, ok4)
show("f5 at 24,000, 4 runs", cr, tr); show("f5 at 6,000, 4 runs", c6, t6)
if fix:
    show("f5 at 24,000, hand", tab(FR, R4, ok4, arm="f5room")[0])
print("right by run:", [sum(G(FR[u][i]) == "right" for i in ok4) for u in R4], "saved:", [sum(G(F6[u][i]) == "right" for i in ok4) for u in R4])
print(f"tokens against f5 at 6,000: {tr/t6:.2f}x")
print("\nCURVE, f5 at each cap")
for cap in (6000, 8000, 12000, 16000, 20000, 24000):
    show(f"at {cap}", tab(FR, R4, ok4, cap)[0])
print("\nWRONG-TWICE, 13 problems"); show("f5 at 24,000", tab(FR, R4, [i for i in ok4 if i in hard])[0]); show("f5 at 6,000", tab(F6, R4, [i for i in ok4 if i in hard])[0])
late = [(u, i) for u in R4 for i in ok4 if T(FR[u][i]) > 6000]
lc = Counter(G(FR[u][i]) for u, i in late); fin = lc["right"] + lc["wrong"]
print(f"\nRuns past 6,000 tokens: {len(late)}, right {lc['right']} wrong {lc['wrong']} cut {lc['cut']}, right among finished {100*lc['right']/(fin or 1):.1f}%")
print("\nWHERE THE WORK IS, f5 at 24,000")
for g in ("right", "wrong", "cut"):
    rows = [FR[u][i] for u in R4 for i in ok4 if G(FR[u][i]) == g]
    q = lambda v: sorted(v)[len(v) // 2] if v else 0
    print(f"  {g:<6} n {len(rows):>3} thinking median {q([r.get('think_tokens',0) for r in rows]):>6} reply median {q([r.get('reply_tokens',0) for r in rows]):>6}, cut with no reply {sum(not (r.get('reply') or '').strip() for r in rows)}")
for a, b in ((0, 500), (500, 1000), (1000, 1500), (1500, 2500), (2500, 4000), (4000, 8000), (8000, 99999)):
    c = Counter(G(FR[u][i]) for u in R4 for i in ok4 if a <= FR[u][i].get("reply_tokens", 0) < b); n = c["right"] + c["wrong"]
    print(f"  reply {a:>5}-{b:<6} right {c['right']:>3} wrong {c['wrong']:>3} cut {c['cut']:>3} right when finished {100*c['right']/(n or 1):.0f}%")
print("\nMOVES, f5 at 6,000 to f5 at 24,000, same run label")
m = Counter((G(F6[u][i]), G(FR[u][i])) for u in R4 for i in ok4)
for k in sorted(m):
    print(f"  {k[0]:>6} to {k[1]:<6} {m[k]}")
if len(ok4) != len(ids):
    sys.exit("INCOMPLETE: no verdict on partial rows")
N = len(ok4) * 4; r = cr["right"] / N; w = cr["wrong"] / N
print("\nVERDICT RIGHT:", "ROOM ALONE REACHES THE TARGET" if r >= .8 else "ROOM GETS HIM MOST OF THE WAY" if r >= .72 else "ROOM IS NOT ENOUGH", f"(R {100*r:.1f}%)")
print("VERDICT WRONG:", "THE WRONGS HOLD" if w <= .18 else "THE WRONGS GROW" if w > .22 else "SOME NEW WRONGS", f"(W {100*w:.1f}%)")
s = lc["right"] / (fin or 1)
print("VERDICT CUTS:", "THE CUTS WERE MOSTLY HEADED RIGHT" if s >= .6 else "MOSTLY HEADED WRONG" if s <= .4 else "MIXED", f"({100*s:.1f}%)")
if have_c5:
    ok2 = [i for i in ids if all(i in C[u] for u in R2)]
    if len(ok2) != len(ids):
        sys.exit("INCOMPLETE: no c5 verdict on partial rows")
    print(f"\nC5 at 24,000, {len(ok2)} problems")
    cc, tc = tab(C, R2, ok2); cu, tu = tab(RE, R2, ok2); show("c5 at 24,000", cc, tc); show("untouched at 24,000", cu, tu)
    fr2, tf2 = tab(FR, R2, ok2); show("f5 at 24,000, same runs", fr2, tf2)
    dr = cc["right"] - cu["right"]; dt = tc / tu - 1
    print(f"c5 against untouched: right {dr:+d}, wrong {cc['wrong']-cu['wrong']:+d}, cut {cc['cut']-cu['cut']:+d}, words {100*dt:+.1f}%; f5 against untouched: words {100*(tf2/tu-1):+.1f}%")
    print("CURVE c5:")
    for cap in (6000, 8000, 12000, 16000, 20000, 24000):
        show(f"at {cap}", tab(C, R2, ok2, cap)[0])
    if dr <= -8: v = "THE COMPRESSOR COSTS REACH"
    elif abs(dr) <= 7 and dt <= -.10: v = "THE COMPRESSOR KEEPS HIS REACH AND SAVES WORDS"
    elif abs(dr) <= 7 and abs(dt) < .10: v = "THE COMPRESSOR DOES NOTHING HERE"
    elif dr >= 8 and dt <= -.10: v = "BOTH"
    else: v = "OUTSIDE THE BARS, read by hand"
    print("VERDICT C5:", v)
with open(os.path.join(HERE, "wrongs-to-read-f5room-v1.txt"), "w") as f:
    for name, S, runs in (("f5room", FR, R4),) + ((("c5room", C, R2),) if have_c5 else ()):
        for u in runs:
            for i in ids:
                if i in S[u] and G(S[u][i]) == "wrong":
                    r = S[u][i]
                    f.write(f"=== {name}:{u}:{i} key {r['answer']} scale {r.get('scale')} unit {r.get('unit')} tokens {T(r)} thinking {r.get('think_tokens')} reply {r.get('reply_tokens')}\nQ: {r['q'][:300]!r}\nEND: {(r.get('reply') or '')[-500:]!r}\n\n")
