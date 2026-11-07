"""The problems pulled from the clean set, kept as their own set (Immy 2026-11-02: see how he builds a picture of a broken
scene). Each tagged by what is wrong, read by hand. true = the right answer where the text has one (None = no answer)."""
import json
TAG = {  # id: (what is wrong, true answer)
    "mw/svamp-302": ("impossible scene: sold 91 of 13 cakes", None),
    "gpt/3-1": ("impossible scene: half of 23 tickets", None),
    "gpt/3-5": ("impossible scene: one third of 22 sticks", None),
    "gpt/4-11": ("underspecified: cans per mural never given (any split works)", None),
    "mw/svamp-153": ("missing start: begins 'He then went...'", 600),
    "mw/svamp-81": ("odd detail: 40 cups of salt in a cake", 1),
    "gpt/0-6": ("yes/no question, not a number", "yes"),
    "mw/mawps-194": ("scene fine, key wrong: team must count Paige", 6),
    "gpt/0-2": ("scene fine, key wrong", 9),
    "gpt/0-10": ("scene fine, key wrong: 40 people need all 14 benches", 0),
    "gpt/1-3": ("scene fine, gpt's equation is nonsense but its answer is right", 15),
}
SUMS = ["3-0", "3-2", "3-3", "3-4", "3-6", "3-7", "3-9", "3-10", "3-11"]   # scene fine, gpt's final sum wrong
rows = []
for f, pre in (("gpt-problems-raw-v2.jsonl", "gpt/"),):
    for r in map(json.loads, open(f)):
        k = r["id"]
        if k in TAG or k[4:] in SUMS:
            what, true = TAG.get(k, ("scene fine, key wrong: gpt's final sum", eval(r["equation"], {"__builtins__": {}})))
            rows.append({"id": "gpt/v2-" + k[4:], "source": "gpt-4.1", "text": r["text"], "broken": what, "true": true,
                         "key_given": r["answer"], "sentences": r["sentences"]})
for r in map(json.loads, open("mw-all-v2.jsonl")):
    if r["id"] in TAG:
        what, true = TAG[r["id"]]
        rows.append({"id": r["id"], "source": "mathworld", "text": r["text"], "broken": what, "true": true,
                     "key_given": r["answer"], "sentences": r["sentences"]})
with open("situation-broken-v1.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
from collections import Counter
print(len(rows), Counter(r["broken"].split(":")[0] for r in rows))
