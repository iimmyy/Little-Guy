"""Fresh-test grader (fresh-1102/PROTOCOL.md). usage: fgrade.py base=g.jsonl,s1.jsonl build=g.jsonl,s1.jsonl ...
Runs are paired by position (greedy with greedy, s1 with s1). Cut = not right. Prints every finished-wrong reply end
for hand reading."""
import os
import json, sys
from math import comb
sys.argv, ARGV = sys.argv[:1], sys.argv[1:]
sys.path.insert(0, os.path.expanduser("~") + "/work/ligands/tools/mind/expert")
from grade import right_reply
HAND = set()            # hand=build:run:id,... finished replies read by hand and equal to the key (PROTOCOL rule)
for a in [a for a in ARGV if a.startswith("hand=")]:
    ARGV.remove(a)
    HAND |= {tuple(x.split(":", 2)) for x in a[5:].split(",") if x}
ok = lambda r: r["finish"] == "stop" and (r.get("_hand") or bool(right_reply(r, r["reply"])))
wrong = lambda r: r["finish"] == "stop" and not ok(r)
def load(f):            # fresh MATH rows are tagged "math-fresh": same competition answers, so the MATH-500 checker
    rows = [{**r, "cat": "math500"} if r.get("cat") == "math-fresh" else r for r in map(json.loads, open(f))]
    return {r["id"]: r for r in rows}
think = lambda r: len(r.get("thinking") or "") or r.get("think_tokens", 0)
sets = {n: [load(f) for f in fs.split(",")] for n, fs in (a.split("=", 1) for a in ARGV)}
for n, k, i in HAND:
    sets[n][int(k)][i]["_hand"] = True
base = list(sets)[0]
B = sets[base]
ids = sorted(B[0])
def sign_p(a, b):
    n, k = a + b, min(a, b)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
for n, R in sets.items():
    assert all(set(x) == set(ids) for x in R), f"{n}: problem ids differ"
    right = [sum(ok(x[i]) for i in ids) for x in R]
    cut = [sum(x[i]["finish"] != "stop" for i in ids) for x in R]
    wr = [sum(wrong(x[i]) for i in ids) for x in R]
    line = f"{n:10s} right {right} = {sum(right)}  cut {cut}  wrong {wr}"
    if n != base:
        created = sum(wrong(x[i]) and not wrong(b[i]) for x, b in zip(R, B) for i in ids)
        fixed = sum(wrong(b[i]) and not wrong(x[i]) for x, b in zip(R, B) for i in ids)
        better = sum(sum(ok(x[i]) for x in R) > sum(ok(b[i]) for b in B) for i in ids)
        worse = sum(sum(ok(x[i]) for x in R) < sum(ok(b[i]) for b in B) for i in ids)
        both = [i for i in ids if all(ok(x[i]) for x in R + B)]
        th = sum(think(x[i]) for x in R for i in both) / max(1, sum(think(b[i]) for b in B for i in both))
        line += (f"  | wrong created {created} fixed {fixed}  problems better {better} worse {worse}"
                 f" (sign p={sign_p(better, worse):.3f})  thinking on {len(both)} always-right: {th - 1:+.1%}")
    print(line)
    for k, x in enumerate(R):
        for i in ids:
            if wrong(x[i]):
                print(f"    WRONG run{k} {i}: key {x[i].get('answer')!r} end {x[i]['reply'][-110:]!r}")
