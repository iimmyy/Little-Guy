#!/usr/bin/env python3
"""Scores GOAL-side-v1. usage: side_score.py <rows.jsonl> <phi or all> [held]"""
import collections, json, re, sys
import numpy as np

rows = [json.loads(l) for l in open(sys.argv[1])]
want = sys.argv[2]
DOUBT = re.compile(r"\b(wait|actually|hold on|hmm|oops|mistake|wrong|recalc\w*|recompute|re-compute|not accurate|too high|too low|seems off|that's off|is off|seems wrong|no,)", re.I)
clean = lambda t: t.replace(",", "")


def found(r):
    return clean(r["right"]) in clean(r["text"])


def doubt(r):
    return bool(DOUBT.search(r["text"][:120]))


def loops(r):
    w = r["text"].split()
    c = collections.Counter(tuple(w[i:i + 5]) for i in range(len(w) - 4))
    return bool(c) and max(c.values()) >= 3


by = collections.defaultdict(dict)
for r in rows:
    by[r["step"]][(r["arm"], r["phi"])] = r
steps = sorted(by)
prob = np.array([by[s][("A", 0.0)]["id"] for s in steps])
phis = sorted({r["phi"] for r in rows if r["arm"] == "B"})
print(f"steps {len(steps)}, problems {len(set(prob))}, phis {phis}")


def col(arm, p, f):
    return np.array([float(f(by[s][(arm, p)])) for s in steps])


A = col("A", 0.0, found)
print(f"baseline A: right number appears {A.mean()*100:.1f}%, doubt word {col('A',0.0,doubt).mean()*100:.1f}%, loops {col('A',0.0,loops).mean()*100:.1f}%")
rng = np.random.default_rng(1105)
u = np.unique(prob)
gp = {p: np.where(prob == p)[0] for p in u}


def ci(d):
    v = [d[np.concatenate([gp[p] for p in rng.choice(u, len(u))])].mean() for _ in range(2000)]
    return np.percentile(v, 2.5) * 100, np.percentile(v, 97.5) * 100


for p in phis:
    B, C = col("B", p, found), col("C", p, found)
    lo1, hi1 = ci(B - A)
    lo2, hi2 = ci(B - C)
    print(f"\nphi {p}: right number appears B {B.mean()*100:.1f}%, C {C.mean()*100:.1f}%, A {A.mean()*100:.1f}%")
    print(f"  B - A {(B-A).mean()*100:+.1f} points (95% {lo1:+.1f} to {hi1:+.1f}); B - C {(B-C).mean()*100:+.1f} points (95% {lo2:+.1f} to {hi2:+.1f})")
    print(f"  doubt word: B {col('B',p,doubt).mean()*100:.1f}%, C {col('C',p,doubt).mean()*100:.1f}%; loops: B {col('B',p,loops).mean()*100:.1f}%, C {col('C',p,loops).mean()*100:.1f}%")
    if want != "all" and float(want) == p:
        b1 = (B - A).mean() * 100 >= 10 and lo1 > 0
        b2 = (B - C).mean() * 100 >= 10 and lo2 > 0
        broke = (col("B", p, loops) - col("A", 0.0, loops)).mean() * 100 > 5
        v = "READS BUT BREAKS" if b1 and b2 and broke else "HE CAN READ A SIDE WORD" if b1 and b2 else "ANY SIDE SETS HIM OFF" if b1 else "HE CANNOT READ IT, at this phi"
        print(f"\nGOAL: {v}")


def other_number(r):
    """The number he would have written had the side been the other token."""
    return r["written"].replace(r["b"], r["o"], 1)


print("\nhow often each number appears, no bar (right / written / other):")
for arm, p in [("A", 0.0)] + [(n, x) for x in phis for n in ("B", "C")]:
    v = [by[s][(arm, p)] for s in steps]
    t = [clean(r["text"]) for r in v]
    print(f"  {arm} phi {p}: right {np.mean([clean(r['right']) in x for r, x in zip(v, t)])*100:.1f}%, written {np.mean([clean(r['written']) in x for r, x in zip(v, t)])*100:.1f}%, other {np.mean([clean(other_number(r)) in x for r, x in zip(v, t)])*100:.1f}%")
