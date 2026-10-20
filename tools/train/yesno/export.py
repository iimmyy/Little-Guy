#!/usr/bin/env python3
"""Practice days -> a batch file for Immy's yes or no (README.md): every reply he gave, and a few of his own thoughts
per day, in the order they happened, each with what he heard (or the moment) and exactly what he said. "cut": he
ran into his word limit there (the brain call in steps.jsonl used every word it was given), so the page can say so and
training never adds an end to it. "step": that brain call's line in the day's steps.jsonl (its exact prompt, feelings
and raw words, which training uses, since his mind takes reminder lines out of what Immy sees).

usage: export.py BATCH OUT.jsonl DAY_DIR [DAY_DIR ...]      (DAY_DIR: a practice.py day, with notebook/)
env: THOUGHTS (his own thoughts kept per day, default 3, picked at random with a fixed seed)
     KEEP_CUT=1 keeps items that ran into the word limit (by default they're left out: Immy can't judge a reply he
     never sees the end of, so practice days for judging raise the limits, REPLY_WORDS and THOUGHT_WORDS)
"""
import json, os, random, re, sys, time, unicodedata

def jl(p):
    try:
        return [json.loads(l) for l in open(p) if l.strip()]
    except OSError:
        return []

core = lambda x: " ".join(re.sub(r"\[[^\]\n]*\]|^TASK:.*$|\.\.\.", " ", "".join(
    c for c in x if c == "\n" or not unicodedata.category(c).startswith("C")), flags=re.M).split())   # as his mind shows it
when = lambda t: time.strftime("%A %-I:%M %p", time.localtime(t))

if __name__ == "__main__":
    batch, out, dirs = int(sys.argv[1]), sys.argv[2], sys.argv[3:]
    keep, rng, rows = int(os.environ.get("THOUGHTS", "3")), random.Random(batch), []
    for d in dirs:
        nb, day = os.path.join(d, "notebook"), "/".join(os.path.normpath(d).split(os.sep)[-2:])
        calls = [(i, x) for i, x in enumerate(jl(os.path.join(d, "steps.jsonl")))
                 if x.get("op") == "think" and x["out"].get("text")]

        def source(text, t):
            """the brain call that wrote this (his mind takes reminder lines and some faces out of what you see):
            its line in steps.jsonl, and whether it used every word it was given"""
            want = core(text)[:60]
            hits = [(abs(float(x.get("t", 0)) - float(t)), i, x) for i, x in calls if want and want in core(x["out"]["text"])]
            if not hits:
                return None, None
            _, i, x = min(hits, key=lambda h: h[0])
            return i, x["out"].get("n_tok", 0) >= x["req"].get("max_tokens", 1e9)
        for c in jl(os.path.join(nb, "chat.jsonl")):
            if c.get("reply") and core(c["reply"]):             # "..." = he only set a reminder: nothing to judge
                step, cut = source(c["reply"], c["t"])
                rows.append({"t": c["t"], "kind": "reply", "when": when(c["t"]), "heard": c["heard"],
                             "reply": c["reply"], "cut": cut, "day": day, "step": step})
        thoughts = [x for x in jl(os.path.join(nb, "thoughts.jsonl")) if x.get("text") and not x.get("alarm")]
        for x in rng.sample(thoughts, min(keep, len(thoughts))):
            about = "a quiet moment, thinking about something coming up" if x.get("task") else "a quiet moment, his mind wandering"
            step, cut = source(x["text"], x["t"])
            rows.append({"t": x["t"], "kind": "thought", "when": x.get("when") or when(x["t"]), "heard": about,
                         "reply": x["text"], "cut": cut, "day": day, "step": step})
    dropped = 0 if os.environ.get("KEEP_CUT") else sum(r["cut"] is True for r in rows)
    rows = rows if os.environ.get("KEEP_CUT") else [r for r in rows if r["cut"] is not True]
    rows.sort(key=lambda r: (r["day"], r["t"]))
    with open(out, "w") as f:
        for n, r in enumerate(rows, 1):
            f.write(json.dumps({"id": f"b{batch}-{n:04d}", "batch": batch, "n": n, **r}, ensure_ascii=False) + "\n")
    print(json.dumps({"items": len(rows), "replies": sum(r["kind"] == "reply" for r in rows),
                      "thoughts": sum(r["kind"] == "thought" for r in rows), "cut": sum(r["cut"] is True for r in rows),
                      "unmatched": sum(r["step"] is None for r in rows), "left_out_cut": dropped}))
