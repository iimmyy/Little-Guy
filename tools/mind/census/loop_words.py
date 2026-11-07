#!/usr/bin/env python3
"""What keeps a loop going? After his first statement of the right answer, the first word of every new paragraph,
counted separately for runs he handed in and runs that ran out of words. Text only, no model. Whole-number answers.
usage: loop_words.py <runs.jsonl> [...]
"""
import collections, json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from natural_error import whole


def first_statement(think, A):                          # same as loop_repeat.py (that file runs on import)
    m = re.search(rf"(?:=|\bis\b|\bare\b|\bget\b|\bgives\b|\bequals\b)\s*\$?\\?\(?\s*{A}(?![\d.,/])", think)
    return m.start() if m else None

C = {"handed in": collections.Counter(), "cut off": collections.Counter()}
n = collections.Counter()
for f in sys.argv[1:]:
    for t in map(json.loads, open(f)):
        A = whole(str(t["answer"]))
        if A is None or A < 2:
            continue
        th = t["thinking"]
        st = first_statement(th, A)
        if st is None:
            continue
        if t["finish"] == "stop":
            g = re.findall(r"-?\d[\d,]*", t["reply"].split("Answer:")[-1])[:1] if "Answer:" in t["reply"] else []
            if not g or whole(g[0]) != A:
                continue
            k = "handed in"
        else:
            k = "cut off"
        n[k] += 1
        for para in th[st:].split("\n\n")[1:]:
            w = re.findall(r"[A-Za-z]+", para[:40])
            if w:
                C[k][w[0]] += 1
for k, c in C.items():
    tot = sum(c.values())
    print(f"{k}: {n[k]} runs, {tot} paragraphs after the answer (per run {tot / max(n[k], 1):.1f}): "
          + ", ".join(f"{w} {v / tot:.0%}" for w, v in c.most_common(16)))
