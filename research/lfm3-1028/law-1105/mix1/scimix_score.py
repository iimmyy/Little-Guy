#!/usr/bin/env python3
"""GOAL-sci-v1, mix against c5 on the 81 held science problems, keyed grade. Also c5 alone against f5 and untouched. usage: scimix_score.py <dir> [corrections.json]. Corrections: {"arm:run:id": "right" or "wrong"}"""
import json, os, sys
from collections import Counter
BASE = os.environ["LITTLEGUY_ROOT"] + "/.run/results/mind/lfm3-1028/science"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE + "/grades-v4")
from keyed_grade import keyed_grade
d = sys.argv[1]
fix = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else {}
ids = [json.loads(l)["id"] for l in open(os.path.join(HERE, "sci-held-81.jsonl"))]
load = lambda p: {r["id"]: r for r in map(json.loads, open(p))}
FILE = {"g": "eval-{}.jsonl", "s1": "eval-{}-s1.jsonl"}
A = {n: {u: load(os.path.join(d, FILE[u].format(n))) for u in FILE} for n in ("scimix", "scic5")}
F = {u: load(f"{BASE}/final-f5-1104/f5-{u}.jsonl") for u in FILE}
U = {u: load(f"{BASE}/final-lfm3-1104/lfm3-base-{u}.jsonl") for u in FILE}
ok = [i for i in ids if all(i in A[n][u] for n in A for u in FILE)]
print(f"problems in all arms {len(ok)} of {len(ids)}, runs {len(ok)*2}")
T = lambda r: (r.get("think_tokens") or 0) + (r.get("reply_tokens") or 0)
def G(r, key=None):
    if key and key in fix and r.get("finish") == "stop":
        return fix[key]
    x = keyed_grade(r.get("finish"), r.get("reply"), r["id"], r["answer"], r.get("scale"))
    return "wrong" if x == "sign" else x
def table(label, S, arm=None):
    c = Counter(G(S[u][i], f"{arm}:{u}:{i}" if arm else None) for u in FILE for i in ok)
    N = len(ok) * 2; tok = sum(T(S[u][i]) for u in FILE for i in ok)
    extra = "".join(f" {k} {v}" for k, v in c.items() if k not in ("right", "wrong", "cut"))
    by = [sum(G(S[u][i], f"{arm}:{u}:{i}" if arm else None) == "right" for i in ok) for u in FILE]
    print(f"{label:<12} right {c['right']:>3} ({100*c['right']/N:.1f}%) wrong {c['wrong']:>3} ({100*c['wrong']/N:.1f}%) cut {c['cut']:>3} ({100*c['cut']/N:.1f}%) tokens {tok} right by run {by}{extra}")
    return c, tok
cm, tm = table("mix", A["scimix"]); cc, tc = table("c5", A["scic5"])
if fix:
    cm, _ = table("mix hand", A["scimix"], "scimix"); cc, _ = table("c5 hand", A["scic5"], "scic5")
cf, tf = table("f5 g+s1", F); cu, tu = table("untouched", U)
dr, dw, dt = cm["right"] - cc["right"], cm["wrong"] - cc["wrong"], tm / tc - 1
print(f"\nmix against c5: right {dr:+d}, wrong {dw:+d}, cut {cm['cut']-cc['cut']:+d}, words {100*dt:+.1f}%")
print(f"c5 against untouched: right {cc['right']-cu['right']:+d}, wrong {cc['wrong']-cu['wrong']:+d}, cut {cc['cut']-cu['cut']:+d}, words {100*(tc/tu-1):+.1f}%")
print(f"f5 against c5: right {cf['right']-cc['right']:+d}, wrong {cf['wrong']-cc['wrong']:+d}, cut {cf['cut']-cc['cut']:+d}, words {100*(tf/tc-1):+.1f}%")
m = Counter((G(A["scic5"][u][i]), G(A["scimix"][u][i])) for u in FILE for i in ok)
print("moves c5 to mix:", sorted(m.items()))
m = Counter((G(A["scic5"][u][i]), G(F[u][i])) for u in FILE for i in ok)
print("moves c5 to f5:", sorted(m.items()))
if dr <= -8 or dw >= 4: v = "IT HURTS"
elif dt >= .10: v = "IT COSTS WORDS"
elif dr >= 8 and dw <= 2 and dt <= -.10: v = "THE MIX WORKS ON SCIENCE"
elif dr >= 8 and dw <= 2: v = "MORE RIGHT, SAME WORDS"
elif abs(dr) <= 7 and dw <= 2 and dt <= -.10: v = "SAME RIGHT, FEWER WORDS"
elif abs(dr) <= 7 and abs(dt) < .10: v = "NOTHING"
else: v = "OUTSIDE THE BARS, read by hand"
print("VERDICT:", v)
with open(os.path.join(HERE, "wrongs-to-read-scimix-v1.txt"), "w") as f:
    for n in A:
        for u in FILE:
            for i in ok:
                r = A[n][u][i]
                if G(r) == "wrong":
                    f.write(f"=== {n}:{u}:{i} key {r['answer']} scale {r.get('scale')} unit {r.get('unit')} tokens {T(r)}\nQ: {r['q'][:300]!r}\nEND: {(r.get('reply') or '')[-500:]!r}\n\n")
