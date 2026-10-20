#!/usr/bin/env python3
"""Practice days -> a batch file for Immy's yes or no (README.md): every reply he gave, and a few of his own thoughts
per day, in the order they happened, each with what he heard (or the moment) and exactly what he said.

usage: export.py BATCH OUT.jsonl DAY_DIR [DAY_DIR ...]      (DAY_DIR: a practice.py day, with notebook/)
env: THOUGHTS (his own thoughts kept per day, default 3, picked at random with a fixed seed)
"""
import json, os, random, sys, time

def jl(p):
    try:
        return [json.loads(l) for l in open(p) if l.strip()]
    except OSError:
        return []

when = lambda t: time.strftime("%A %-I:%M %p", time.localtime(t))

if __name__ == "__main__":
    batch, out, dirs = int(sys.argv[1]), sys.argv[2], sys.argv[3:]
    keep, rng, rows = int(os.environ.get("THOUGHTS", "3")), random.Random(batch), []
    for d in dirs:
        nb, day = os.path.join(d, "notebook"), "/".join(os.path.normpath(d).split(os.sep)[-2:])
        for c in jl(os.path.join(nb, "chat.jsonl")):
            if c.get("reply"):
                rows.append({"t": c["t"], "kind": "reply", "when": when(c["t"]), "heard": c["heard"],
                             "reply": c["reply"], "day": day})
        thoughts = [x for x in jl(os.path.join(nb, "thoughts.jsonl")) if x.get("text") and not x.get("alarm")]
        for x in rng.sample(thoughts, min(keep, len(thoughts))):
            about = "a quiet moment, thinking about something coming up" if x.get("task") else "a quiet moment, his mind wandering"
            rows.append({"t": x["t"], "kind": "thought", "when": x.get("when") or when(x["t"]), "heard": about,
                         "reply": x["text"], "day": day})
    rows.sort(key=lambda r: (r["day"], r["t"]))
    with open(out, "w") as f:
        for n, r in enumerate(rows, 1):
            f.write(json.dumps({"id": f"b{batch}-{n:04d}", "batch": batch, "n": n, **r}, ensure_ascii=False) + "\n")
    print(json.dumps({"items": len(rows), "replies": sum(r["kind"] == "reply" for r in rows),
                      "thoughts": sum(r["kind"] == "thought" for r in rows)}))
