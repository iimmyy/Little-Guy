#!/usr/bin/env python3
"""Scores GOAL-silent-v1. usage: silent_score.py <rows>"""
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
A = [r for r in rows if r["kind"] == "wrong" and not r["qmark"] and not r["doubt_next"]]
B = [r for r in rows if r["kind"] == "wrong" and (r["qmark"] or r["doubt_next"])]
C = [r for r in rows if r["kind"] == "right"]
def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None
def same(x, v, written):
    d = written.replace(",", "")
    ulp = 10 ** (-len(d.split(".")[1])) if "." in d else 1
    n = num(x)
    return n is not None and (abs(n - v) <= 2 * ulp or abs(n - v) <= 0.0005 * abs(v))
def counts(S, arm):
    no = sum(num(r["reads"][arm][0]) is None for r in S)
    agree = sum(same(r["reads"][arm][0], float(r["written"].replace(",", "")), r["written"]) for r in S)
    right = sum(same(r["reads"][arm][0], r["true"], r["written"]) for r in S)
    return dict(n=len(S), no_number=no, agrees_with_written=agree, lands_on_true=right, disagrees=len(S) - agree - no)
for name, S in (("A believed wrong", A), ("B doubted wrong", B), ("C right", C)):
    print(name)
    for arm in ("first", "plain", "now", "wait"):
        print(f"  {arm:<6}", counts(S, arm))
a = counts(A, "wait"); p = counts(A, "plain")
fw, fp = a["lands_on_true"] / len(A), p["lands_on_true"] / len(A)
v = "HE HAD IT, A DOUBT WORD REACHES IT" if fw >= .5 and fw - fp >= .2 else "A SECOND LOOK IS ENOUGH, THE WORD ISN'T NEEDED" if fp >= .5 else "THE SENTENCE HOLDS HERE, HE NEEDS OUTSIDE INPUT" if fw <= .2 and fp <= .2 else "IN BETWEEN"
print(f"\nBAR 1: wait lands on true {a['lands_on_true']} of {len(A)}, plain {p['lands_on_true']}: {v}")
AB = A + B
w = counts(AB, "wait"); c = counts(C, "wait")
catch = w["disagrees"] / len(AB); fa = c["disagrees"] / len(C)
print(f"BAR 2: wait disagrees with what he wrote on {w['disagrees']} of {len(AB)} wrong ({catch*100:.1f}%), {c['disagrees']} of {len(C)} right ({fa*100:.1f}%): {'ONE NUMBER DOES BOTH' if catch >= .5 and fa <= .05 else 'NOT YET'}")
for arm in ("first", "plain", "now"):
    w, c = counts(AB, arm), counts(C, arm)
    print(f"  no bar, {arm}: wrong {w['disagrees']} of {len(AB)} ({w['disagrees']/len(AB)*100:.1f}%), right {c['disagrees']} of {len(C)} ({c['disagrees']/len(C)*100:.1f}%)")
d = [r for r in AB if not same(r["reads"]["wait"][0], float(r["written"].replace(",", "")), r["written"]) and num(r["reads"]["wait"][0]) is not None]
print(f"of wait reads that disagree on wrong steps, how many are the true value: {sum(same(r['reads']['wait'][0], r['true'], r['written']) for r in d)} of {len(d)}")
print("\nset A, step by step (written | true | first | plain | now | wait):")
for r in A:
    print(f"  {r['step']} {r['id'][:28]} {r['expr'].strip()} {r['written']} | {r['true']:.6g} | " + " | ".join(str(r['reads'][k][0]) for k in ('first', 'plain', 'now', 'wait')))
print("\nset B, step by step:")
for r in B:
    print(f"  {r['step']} {r['id'][:20]} {r['expr'].strip()} {r['written']} | {r['true']:.6g} | " + " | ".join(str(r['reads'][k][0]) for k in ('first', 'plain', 'now', 'wait')))
print("\nright steps where wait disagrees:")
for r in C:
    if not same(r["reads"]["wait"][0], float(r["written"].replace(",", "")), r["written"]) and num(r["reads"]["wait"][0]) is not None:
        print(f"  {r['step']} {r['expr'].strip()} {r['written']} | {r['true']:.6g} | first {r['reads']['first'][0]} | wait {r['reads']['wait'][0]}")
