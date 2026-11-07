#!/usr/bin/env python3
"""The "never" pile (cut-offs where the right answer never shows up in his thinking, see cutoff_sort.py), per problem:
in how many of all the given runs is that problem solved? Solved somewhere = within his reach (slow or lost on a wrong
road this time); solved nowhere = beyond him in the word budget. Prints both lists with how many runs solved each and
the solving runs' thinking length.

usage: never_sort.py <pile runs, comma-separated> <all runs, comma-separated> (from a folder with tools/mind/expert/grade.py)
"""
import json, re, sys

sys.path.insert(0, "tools/mind/expert")
PILE, ALL = sys.argv[1].split(","), sys.argv[2].split(",")
sys.argv[1:] = []
from grade import right, right_reply

CLAIM = re.compile(r"\\boxed\{([^{}]{1,40})\}|(?:answer is|answer =|answer:)\s*\$?([^\n$]{1,40}?)\$?(?=\.(?:\s|$)|\n|$)", re.I)


def had_it(r):
    for mt in CLAIM.finditer(r["thinking"] or ""):
        x = next(g for g in mt.groups() if g is not None).strip().rstrip(".").strip()
        if x and right({**r, "reply": x}, x):
            return True
    return False


runs = {p: {r["id"]: r for r in map(json.loads, open(p))} for p in ALL}
solved = {}
for p, R in runs.items():
    for i, r in R.items():
        if r["finish"] == "stop" and right_reply(r, r["reply"]):
            solved.setdefault(i, []).append(r["think_tokens"])
never = {}
for p in PILE:
    for i, r in runs[p].items() if p in runs else ((r["id"], r) for r in map(json.loads, open(p))):
        if r["finish"] != "stop" and not had_it(r):
            never.setdefault(i, []).append(p)
reach = sorted((i for i in never if i in solved), key=lambda i: -len(solved[i]))
beyond = sorted(i for i in never if i not in solved)
print(f"never pile: {sum(map(len, never.values()))} cut-offs on {len(never)} problems; solved in some run: {len(reach)} problems, "
      f"solved nowhere ({len(ALL)} runs): {len(beyond)}")
for i in reach:
    print(f"  REACH {i} key={runs[ALL[0]][i]['answer']!r}: never in {len(never[i])} pile runs, solved in {len(solved[i])}/{len(ALL)} runs, "
          f"solving thinking {sorted(solved[i])}")
for i in beyond:
    print(f"  BEYOND {i} L{runs[ALL[0]][i]['level']} key={runs[ALL[0]][i]['answer']!r}: never in {len(never[i])} pile runs")
