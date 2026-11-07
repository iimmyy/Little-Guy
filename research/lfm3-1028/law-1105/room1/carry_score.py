#!/usr/bin/env python3
"""Unblinds and scores GOAL-carry-v1. usage: carry_score.py <codes> <key> <rows>"""
import collections, json, sys
import numpy as np
codes_f, key_f, rows_f = sys.argv[1:4]
key = json.load(open(key_f))
prob = {}
for l in open(rows_f):
    r = json.loads(l)
    prob[r["step"]] = r["id"]
c = {"A": {}, "B": {}, "C": {}}
for l in open(codes_f):
    s, x, y, z = l.split()
    for lab, v in zip("XYZ", (x, y, z)):
        c[key[s][lab]][int(s)] = v
steps = sorted(c["A"])
assert len(steps) == 60
P = np.array([prob[s] for s in steps])
u = np.unique(P)
gp = {p: np.where(P == p)[0] for p in u}
rng = np.random.default_rng(1106)
print(f"steps {len(steps)}, problems {len(u)}")
ALL = ["R", "W", "O", "F", "DR", "DW", "DO", "DF", "D", "C"]
print("code   " + "".join(f"{a:>7}" for a in "ABC"))
for k in ALL:
    print(f"{k:<7}" + "".join(f"{sum(c[a][s]==k for s in steps)/60*100:>6.1f}%" for a in "ABC"))
def col(a, ks):
    return np.array([float(c[a][s] in ks) for s in steps])
def ci(d):
    v = [d[np.concatenate([gp[p] for p in rng.choice(u, len(u))])].mean() for _ in range(5000)]
    return np.percentile(v, 2.5) * 100, np.percentile(v, 97.5) * 100
def show(name, d):
    lo, hi = ci(d)
    print(f"{name}: {d.mean()*100:+.1f} points (95% {lo:+.1f} to {hi:+.1f})")
    return d.mean() * 100, lo
print()
ba, l1 = show("code R, B - A", col("B", {"R"}) - col("A", {"R"}))
bc, l2 = show("code R, B - C", col("B", {"R"}) - col("C", {"R"}))
ca, l3 = show("other shows (O, DO), C - A", col("C", {"O", "DO"}) - col("A", {"O", "DO"}))
read_right = ba >= 15 and l1 > 0 and bc >= 15 and l2 > 0
read_other = ca >= 5 and l3 > 0
v = "HE READS IT" if read_right and read_other else "HE READS ONLY A RIGHT VALUE" if read_right else "HE CANNOT READ IT"
mis, _ = show("bar 3 as written, calm wrong: B (F + W) - A (F)", col("B", {"F", "W"}) - col("A", {"F"}))
if read_right and mis > 5:
    v = "READS BUT MISLEADS (" + v + ")"
print(f"\nGOAL: {v}")
print("\nno bar, same-footing views:")
show("calm wrong, B (F + W) - A (F + W)", col("B", {"F", "W"}) - col("A", {"F", "W"}))
show("calm F only, B - A", col("B", {"F"}) - col("A", {"F"}))
show("calm F only, C - A", col("C", {"F"}) - col("A", {"F"}))
show("any doubt code, B - A", col("B", {"DR", "DW", "DO", "DF", "D"}) - col("A", {"DR", "DW", "DO", "DF", "D"}))
show("any doubt code, B - C", col("B", {"DR", "DW", "DO", "DF", "D"}) - col("C", {"DR", "DW", "DO", "DF", "D"}))
show("code O, C - B", col("C", {"O"}) - col("B", {"O"}))
trip = [s for s in steps if c["A"][s] == "W" and c["B"][s] == "R" and c["C"][s] == "O"]
print(f"clean triples, A carries written, B right, C other: {len(trip)} of 60: {trip}")
print(f"A carries written calmly and B carries right calmly: {sum(c['A'][s]=='W' and c['B'][s]=='R' for s in steps)}; A calm written, B calm written anyway: {sum(c['A'][s]=='W' and c['B'][s]=='W' for s in steps)}; A right by itself after doubt, B carries wrong calmly: {sum(c['A'][s] in ('DR','R') and c['B'][s] in ('W','F') for s in steps)}")
print("\nstep codes A B C:")
for s in steps:
    print(s, c["A"][s], c["B"][s], c["C"][s])
