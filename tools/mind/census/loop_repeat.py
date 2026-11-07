#!/usr/bin/env python3
"""How much of his thinking AFTER he first states the right answer is repetition? (2026-11-03: hand read of
number_theory/1055 showed the same check over and over once the answer was in.) Text only, no model.
Thinking is split into paragraphs (blank lines). A paragraph is a repeat if most of its word 4-grams already appeared
earlier in the same thinking (share >= `share`). Reported per run kind (handed in right, cut off): words before vs after
the first statement of the right answer (fit_test.fork_of style, first match), and the share of after-words that sit
in repeat paragraphs. Only whole-number answers (the statement can be found by its digits).
Scene level: a paragraph with numbers brings nothing new if every number in it already appeared earlier in the thinking.
usage: loop_repeat.py <out.txt> <runs.jsonl> [...] [share=0.6]
"""
import json, os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from natural_error import whole

out = sys.argv[1]
runs = [a for a in sys.argv[2:] if "=" not in a]
opt = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)
SHARE = float(opt.get("share", 0.6))


def grams(words):
    return {tuple(words[i:i + 4]) for i in range(len(words) - 3)}


def first_statement(think, A):
    m = re.search(rf"(?:=|\bis\b|\bare\b|\bget\b|\bgives\b|\bequals\b)\s*\$?\\?\(?\s*{A}(?![\d.,/])", think)
    return m.start() if m else None


rows = {"handed in": [], "cut off": []}
for f in runs:
    for t in map(json.loads, open(f)):
        A = whole(str(t["answer"]))
        if A is None or A < 2:
            continue
        th = t["thinking"]
        st = first_statement(th, A)
        if st is None:
            continue
        if t["finish"] == "stop":
            got = re.findall(r"-?\d[\d,]*", t["reply"].split("Answer:")[-1])[:1] if "Answer:" in t["reply"] else []
            if not got or whole(got[0]) != A:
                continue
            kind = "handed in"
        else:
            kind = "cut off"
        seen, before, after, rep = set(), 0, 0, 0
        nums, nn_after, nn_old = set(), 0, 0                # scene level: does a paragraph bring any number not seen before?
        pos = 0
        for para in th.split("\n\n"):
            w = re.findall(r"\S+", para)
            g = grams(w)
            is_rep = len(g) >= 3 and len(g & seen) / len(g) >= SHARE
            ns = set(re.findall(r"\d+(?:\.\d+)?", para))
            if pos >= st:
                after += len(w); rep += len(w) if is_rep else 0
                if ns:
                    nn_after += len(w); nn_old += len(w) if ns <= nums else 0
            else:
                before += len(w)
            seen |= g; nums |= ns
            pos += len(para) + 2
        rows[kind].append((t["id"], before, after, rep, nn_after, nn_old))
L = [f"repetition after the first right statement (paragraph = repeat if >= {SHARE:.0%} of its 4-grams came earlier)"]
for kind, r in rows.items():
    if not r:
        continue
    b, a, p, na, no = (np.array([x[i] for x in r]) for i in (1, 2, 3, 4, 5))
    L.append(f"  {kind}: {len(r)} runs ({len({x[0] for x in r})} problems); words before the first right statement median {np.median(b):.0f}, "
             f"after {np.median(a):.0f}; share of after-words in repeat paragraphs: median {np.median(p / np.maximum(a, 1)):.0%}, "
             f"pooled {p.sum() / max(a.sum(), 1):.0%}; share of after-words (in paragraphs with numbers) bringing NO new number: "
             f"median {np.median(no / np.maximum(na, 1)):.0%}, pooled {no.sum() / max(na.sum(), 1):.0%}")
    for x in sorted(r, key=lambda x: -x[2])[:8] if kind == "cut off" else []:
        L.append(f"      {x[0]}: before {x[1]}, after {x[2]}, repeated {x[3] / max(x[2], 1):.0%}, no new number {x[5] / max(x[4], 1):.0%}")
open(out, "w").write("\n".join(L) + "\n")
print("\n".join(L))
