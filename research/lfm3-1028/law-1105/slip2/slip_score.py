#!/usr/bin/env python3
"""Scores GOAL-slip-v1. The read is fixed there; nothing is chosen or fitted here.
usage: slip_score.py <slip prefix> <broken prefix> <broken steps jsonl>"""
import json, sys
from math import comb
import numpy as np

sp, bp, bsteps = sys.argv[1:4]
A, B = "op9", "ffn11"
HELD = "ffn23"
WIN = (1, 2, 3)


def load(prefix):
    D = np.load(prefix + "-x.npz")
    order = list(D["order"])
    iu = np.triu_indices(len(order))
    pos = {(int(p), int(q)): t for t, (p, q) in enumerate(zip(*iu))}
    rows = [json.loads(l) for l in open(prefix + "-rows.jsonl")]
    key = {(r["step"], r["arm"], r["k"]): i for i, r in enumerate(rows)}
    return dict(G=D["G"], EC=D["EC"], FO=D["FO"], order=order, pos=pos, rows=rows, key=key)


def g(d, i, a, b):
    p, q = sorted((d["order"].index(a), d["order"].index(b)))
    return float(d["G"][i, d["pos"][(p, q)]])


def pair(d, i):
    return g(d, i, A, B) / np.sqrt(g(d, i, A, A) * g(d, i, B, B))


def held(d, i):
    """+cos(state just before the last neuron write, last neuron write)."""
    order = d["order"]
    n = order.index(HELD)
    dot = sum(g(d, i, order[j], HELD) for j in range(n))
    sq = sum(g(d, i, order[j], order[m]) for j in range(n) for m in range(n))
    return dot / np.sqrt(sq * g(d, i, HELD, HELD))


def window(d, step, arm, f):
    v = [f(d, d["key"][(step, arm, k)]) for k in WIN if (step, arm, k) in d["key"]]
    return float(np.mean(v)) if v else np.nan


S = load(sp)
steps = sorted({r["step"] for r in S["rows"]})
print(f"slips read {len(steps)} of 10 kept by hand")
if len(steps) < 8:
    print("fewer than 8 survive: the test cannot answer, bar not run")
    sys.exit()
surp = {a: np.array([S["rows"][S["key"][(s, a, 1)]]["num_surprise"] for s in steps]) for a in ("slip", "fixed")}
prem = float((surp["slip"] < surp["fixed"]).mean())
print(f"\npremise, number surprise lower on the slip: {prem*100:.0f}% (needs 75%) -> {'holds' if prem >= 0.75 else 'VOID'}")
print(f"  median bits: slip {np.median(surp['slip']):.1f}, fixed {np.median(surp['fixed']):.1f}")
read = {a: np.array([window(S, s, a, pair) for s in steps]) for a in ("slip", "fixed")}
wins = float((read["slip"] > read["fixed"]).sum() + 0.5 * (read["slip"] == read["fixed"]).sum())
n = len(steps)
score = wins / n
k = int(np.ceil(wins))
p = sum(comb(n, x) for x in range(k, n + 1)) / 2 ** n
b1 = score >= 0.75 and p < 0.05

# the line: 95th percentile of this same read on his held-out right steps from the broken test
Br = load(bp)
bh = sorted({r["step"] for r in Br["rows"] if r["split"] == "held"})
clean = np.array([window(Br, s, "clean", pair) for s in bh])
line = float(np.nanpercentile(clean, 95))
catch = float((read["slip"] > line).mean())
b2 = catch >= 0.5
print(f"\nthe read: cos({A}, {B}), mean of words {WIN} after the number; higher = wrong (locked)")
print(f"B1 slip above fixed: {int(wins)} of {n}, score {score:.3f}, exact one-sided p {p:.4f} -> {'met' if b1 else 'not met'}")
print(f"B2 line {line:.4f} from {len(clean)} held-out right steps; catches {int((read['slip'] > line).sum())} of {n} ({catch*100:.0f}%) -> {'met' if b2 else 'not met'}")
print(f"   fixed versions over the same line: {int((read['fixed'] > line).sum())} of {n}")
verdict = "VOID" if prem < 0.75 else "PASS" if b1 and b2 else "REACTS, NO ALARM" if b1 else "FAIL"
print(f"\nGOAL: {verdict}")

print("\nstep by step:")
print("  step  read slip  read fixed  slip higher  over line  bits slip  bits fixed")
for i, s in enumerate(steps):
    print(f"  {s:>4}  {read['slip'][i]:+.4f}    {read['fixed'][i]:+.4f}     {str(bool(read['slip'][i] > read['fixed'][i])):<11}  {str(bool(read['slip'][i] > line)):<9}  {surp['slip'][i]:>7.1f}  {surp['fixed'][i]:>9.1f}")


def report(name, a, b):
    w = (a > b).sum() + 0.5 * (a == b).sum()
    print(f"  {name}: slip above fixed {w:g} of {n}, score {w/n:.3f}")


print("\npredicted to follow surprise, at or below 0.5:")
report("number surprise", surp["slip"], surp["fixed"])
next_s = {a: np.array([window(S, s, a, lambda d, i: float(d["FO"][i, 3])) for s in steps]) for a in ("slip", "fixed")}
report("next-word surprise", next_s["slip"], next_s["fixed"])
ec = {a: np.array([window(S, s, a, lambda d, i: -float(d["EC"][i, 13])) for s in steps]) for a in ("slip", "fixed")}
report("closed energy at L13, falling", ec["slip"], ec["fixed"])
hd = {a: np.array([window(S, s, a, held) for s in steps]) for a in ("slip", "fixed")}
report("last neuron write vs held", hd["slip"], hd["fixed"])

print("\nsame read, word by word, slip above fixed:")
for w in (0, 1, 2, 3, 5, 10, 20, 50):
    ok = [s for s in steps if (s, "slip", w) in S["key"] and (s, "fixed", w) in S["key"]]
    v = [pair(S, S["key"][(s, "slip", w)]) > pair(S, S["key"][(s, "fixed", w)]) for s in ok]
    print(f"  word {w}: {sum(v)} of {len(v)}")
pre = [abs(pair(S, S["key"][(s, "slip", -1)]) - pair(S, S["key"][(s, "fixed", -1)])) for s in steps]
print(f"\ncheck, word before the number, biggest difference between versions (should be about 0): {max(pre):.6f}")

where = {}
for l in open(bsteps):
    j = json.loads(l)
    where[j["step"]] = len(j["pre"])
x = np.array([where[s] for s in bh], float)
ok = ~np.isnan(clean)
print(f"check, how far into the thinking vs the read on right steps: correlation {np.corrcoef(x[ok], clean[ok])[0, 1]:+.3f} (right steps come from the first 12,000 characters; some slips sit later)")
