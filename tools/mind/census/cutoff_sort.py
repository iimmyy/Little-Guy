#!/usr/bin/env python3
"""Sort a night run's cut-offs (ran out of thinking words) into two piles (2026-11-01, after compressor c2):
  had it    he wrote the right answer ("answer is X" / boxed) somewhere in his thinking and kept going
  never     the right answer never shows up: a problem he could not solve in the word budget
For "had it": how many times he stated it, where the first statement sits (share of his thinking), and what the next
sentence did after each statement (doubt / hand-in / other). Prints the thinking end of every "had it" cut-off for reading.

usage: cutoff_sort.py <name=run.jsonl> ... (run from a folder with tools/mind/expert/grade.py)
"""
import json, re, sys

sys.path.insert(0, "tools/mind/expert")
ARGS, sys.argv[1:] = sys.argv[1:], []
from grade import right

CLAIM = re.compile(r"\\boxed\{([^{}]{1,40})\}|(?:answer is|answer =|answer:)\s*\$?([^\n$]{1,40}?)\$?(?=\.(?:\s|$)|\n|$)", re.I)
DOUBT = re.compile(r"\s*(But|Wait|However|Hmm|Let me|Let's|Actually|Alternatively|Double|Check|Hold on)\b", re.I)
HANDOFF = re.compile(r"\s*(I'll (produce|write|output|give|present|answer)|We need to (end|output|produce|write|present)"
                     r"|Now (we need to )?(output|produce|write)|So (the )?final (output|line)|Thus,? final"
                     r"|Thus,? I'll|End with|So final answer|Thus final answer)", re.I)

for a in ARGS:
    name, path = a.split("=", 1)
    rows = [r for r in map(json.loads, open(path)) if r["finish"] != "stop"]
    had, never = [], []
    for r in rows:
        t = r["thinking"] or ""
        hits = []
        for mt in CLAIM.finditer(t):
            x = next(g for g in mt.groups() if g is not None).strip().rstrip(".").strip()
            if x and right({**r, "reply": x}, x):
                nxt = t[mt.end():mt.end() + 80].lstrip(" .$}\n")
                hits.append((mt.start() / max(1, len(t)), "hand" if HANDOFF.match(nxt) else "doubt" if DOUBT.match(nxt) else "other"))
        (had if hits else never).append((r, hits))
    print(f"== {name}: {len(rows)} cut-offs; had the right answer {len(had)}, never wrote it {len(never)}")
    for r, hits in had:
        kinds = {k: sum(h[1] == k for h in hits) for k in ("doubt", "hand", "other")}
        print(f"  HAD {r['id']} L{r['level']} key={r['answer']!r}: stated {len(hits)}x, first at {hits[0][0]:.0%} of thinking; "
              f"next sentence doubt {kinds['doubt']} / hand-in {kinds['hand']} / other {kinds['other']}")
        print(f"     end: {(r['thinking'] or '')[-300:]!r}")
    for r, _ in never:
        print(f"  NEVER {r['id']} L{r['level']} key={r['answer']!r}")
