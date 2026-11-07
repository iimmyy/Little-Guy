#!/usr/bin/env python3
"""Practice days -> a batch file for Immy's yes or no (README.md): every reply he gave, and his own thoughts
per day, in the order they happened, each with what he heard (or the moment) and exactly what he said. "cut": he
ran into his word limit there (the brain call in steps.jsonl used every word it was given), so the page can say so and
training never adds an end to it. "step": that brain call's line in the day's steps.jsonl (its exact prompt, feelings
and raw words, which training uses, since his mind takes reminder lines out of what Immy sees).

usage: export.py BATCH OUT.jsonl DAY_DIR [DAY_DIR ...]      (DAY_DIR: a practice.py day, with notebook/)
env: THOUGHTS (optional explicit limit per day; default keeps every thought)
     PARTS (default replies,thoughts,thinking): which of his words to collect; "thinking" is what he thought privately
     before answering, shown as a thought ("part": "thinking"), right before the reply it led to (Immy: "so I can say
     no to the checklist habit")
     KEEP_CUT=0 explicitly excludes cut replies; default keeps and marks incomplete items.
Existing output files are never overwritten because their IDs may already have judgments.
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
    if os.path.exists(out):
        raise SystemExit("Export exists. Preserve its IDs; use a new batch and output for new data.")
    keep, rng, rows = int(os.environ.get("THOUGHTS", "-1")), random.Random(batch), []
    parts = set(os.environ.get("PARTS", "replies,thoughts,thinking").split(","))
    for d in dirs:
        nb, day = os.path.join(d, "notebook"), "/".join(os.path.normpath(d).split(os.sep)[-2:])
        calls = [(i, x) for i, x in enumerate(jl(os.path.join(d, "steps.jsonl")))
                 if x.get("op") == "think" and x["out"].get("text")]
        calls_by = dict(calls)

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
                if "replies" in parts:
                    rows.append({"t": c["t"], "kind": "reply", "when": when(c["t"]), "heard": c["heard"],
                                 "reply": c["reply"], "cut": cut, "day": day, "step": step})
                s = calls_by.get(step)
                if "thinking" in parts and s and s["req"].get("thinking") and s["out"].get("thought"):
                    rows.append({"t": float(c["t"]) - 0.5, "kind": "thought", "part": "thinking", "when": when(c["t"]),
                                 "heard": f'Before answering "{c["heard"]}" (what he thought privately first)',
                                 "reply": s["out"]["thought"], "day": day, "step": step,
                                 "cut": s["out"].get("n_thought", 0) >= s["req"].get("think_tokens", 1e9)})
        thoughts = [x for x in jl(os.path.join(nb, "thoughts.jsonl")) if x.get("text") and not x.get("alarm")]
        for x in ((thoughts if keep < 0 else rng.sample(thoughts, min(keep, len(thoughts)))) if "thoughts" in parts else []):
            about = "a quiet moment, thinking about something coming up" if x.get("task") else "a quiet moment, his mind wandering"
            step, cut = source(x["text"], x["t"])
            rows.append({"t": x["t"], "kind": "thought", "when": x.get("when") or when(x["t"]), "heard": about,
                         "reply": x["text"], "cut": cut, "day": day, "step": step})
    # Incomplete items remain judgeable and visibly marked; filtering must be explicit.
    gone = lambda r: r["cut"] is True and r.get("part") != "thinking" and os.environ.get("KEEP_CUT", "1") == "0"
    dropped = sum(map(gone, rows))
    rows = [r for r in rows if not gone(r)]
    rows.sort(key=lambda r: (r["day"], r["t"]))
    with open(out, "x") as f:
        for n, r in enumerate(rows, 1):
            f.write(json.dumps({"id": f"b{batch}-{n:04d}", "batch": batch, "n": n, **r}, ensure_ascii=False) + "\n")
    print(json.dumps({"items": len(rows), "replies": sum(r["kind"] == "reply" for r in rows),
                      "thoughts": sum(r["kind"] == "thought" and r.get("part") != "thinking" for r in rows),
                      "thinking": sum(r.get("part") == "thinking" for r in rows), "cut": sum(r["cut"] is True for r in rows),
                      "unmatched": sum(r["step"] is None for r in rows), "left_out_cut": dropped}))
