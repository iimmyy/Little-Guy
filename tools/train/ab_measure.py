#!/usr/bin/env python3
"""Two brains on the same made-up days (ab_days.sh), measured without any AI judging how human they sound:
  - Immy's own rules, broken how often (the ones written from his no's: exclamation marks, emojis, lists or bold,
    robot and banned words, "[thought]" scripts, a quiet thought that repeats one he already had that day)
  - his mannerisms from the mirror (mirror.py: phrases he says far more often than people), per 1000 words
  - how long his replies and thoughts run
usage: ab_measure.py MIRROR.json OUT.md A_DAY_DIR... -- B_DAY_DIR...
"""
import json, os, re, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from distill import why_dropped, FACE
from mirror import words, spoken


def profile(day_dirs, phrases):
    out = {"reply": {"n": 0, "broken": {}, "words": []}, "thought": {"n": 0, "broken": {}, "words": []}}
    hits, total_words = {p: 0 for p in phrases}, 0
    for d in day_dirs:
        earlier = []
        for l in open(os.path.join(d, "steps.jsonl")):
            x = json.loads(l)
            if x["op"] != "think" or not x["out"].get("text"):
                continue
            text = x["out"]["text"]
            kind = "reply" if "is talking with you" in x["req"]["messages"][0]["content"] else "thought"
            o = out[kind]
            o["n"] += 1
            why = why_dropped(text, earlier if kind == "thought" else [])
            earlier.append(set(re.findall(r"[a-z']+", FACE.sub(" ", text).lower())))
            if why:
                why = why.split(":")[0]
                o["broken"][why] = o["broken"].get(why, 0) + 1
            ws = words(spoken(text))
            o["words"].append(len(ws))
            total_words += len(ws)
            joined = " " + " ".join(ws) + " "
            for p in phrases:
                hits[p] += joined.count(" " + p + " ")
    for o in out.values():
        o["median words"] = statistics.median(o["words"]) if o["words"] else 0
        o["broken total"] = round(sum(o["broken"].values()) / max(1, o["n"]), 3)
        del o["words"]
    per_k = {p: round(1000 * h / max(1, total_words), 2) for p, h in hits.items()}
    return out, per_k, total_words


if __name__ == "__main__":
    mirror, out_md = sys.argv[1], sys.argv[2]
    rest = sys.argv[3:]
    a_dirs, b_dirs = rest[:rest.index("--")], rest[rest.index("--") + 1:]
    m = json.load(open(mirror))["phrases"]
    phrases = [r["phrase"] for n in ("2", "3") for r in m[n] if r["echo"] < 0.5][:30] + \
              [r["phrase"] for r in m["1"] if r["echo"] < 0.5][:30]
    A, B = profile(a_dirs, phrases), profile(b_dirs, phrases)
    res = {"A": {"days": len(a_dirs), "words": A[2], **A[0]}, "B": {"days": len(b_dirs), "words": B[2], **B[0]},
           "mannerisms per 1000 words (A, B)": {p: [A[1][p], B[1][p]] for p in phrases},
           "all mannerisms per 1000 words": [round(sum(A[1].values()), 2), round(sum(B[1].values()), 2)]}
    json.dump(res, open(out_md.replace(".md", ".json"), "w"), indent=1)
    with open(out_md, "w") as f:
        f.write("# Same days, two brains\n\n")
        for k in ("reply", "thought"):
            f.write(f"## {k}s\n\n| | A | B |\n|---|---|---|\n| how many | {res['A'][k]['n']} | {res['B'][k]['n']} |\n"
                    f"| median words | {res['A'][k]['median words']} | {res['B'][k]['median words']} |\n"
                    f"| share breaking one of Immy's rules | {res['A'][k]['broken total']} | {res['B'][k]['broken total']} |\n")
            for why in sorted(set(res["A"][k]["broken"]) | set(res["B"][k]["broken"])):
                f.write(f"| {why} | {res['A'][k]['broken'].get(why, 0)} | {res['B'][k]['broken'].get(why, 0)} |\n")
            f.write("\n")
        f.write(f"## Mannerisms (the mirror's top phrases), per 1000 words\n\nall: A {res['all mannerisms per 1000 words'][0]}, "
                f"B {res['all mannerisms per 1000 words'][1]}\n\n| phrase | A | B |\n|---|---|---|\n")
        for p, (x, y) in res["mannerisms per 1000 words (A, B)"].items():
            f.write(f"| {p} | {x} | {y} |\n")
    print(json.dumps({k: res[k] for k in ("A", "B", "all mannerisms per 1000 words")}, indent=1))
