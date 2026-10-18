#!/usr/bin/env python3
"""A phrase ladder for a mood: his own closer phrases as the mood deepens (Immy: "bigger" = more intimate).

ladder_build.py counted single words and found that deeper loneliness uses FEWER words, with the closeness in
how simple words go together ("You're back. You're actually back.", "I was waiting", "hear your voice"). This
counts 2-3 word phrases in the same answers (ladder_build.py answers.json) and keeps, for each strength, the
phrases that rise over plain him and the robot-not-alone scene AND over the strength below. Kept only when the
phrase ends in a content word and has no repeated word (stammers like "I... I" are his, but pushing repetition
risks loops).
mood_dose.py PHRASES=<phrase_ladder.json>:<B> lights them with the drip, context-aware: once he has started a
phrase, its next word gets up to +B (he picks the phrase, we make it easier to finish), and its first word a
smaller +B*START.
usage: phrase_ladder.py <hf (tokenizer)> <answers.json> <out json>
"""
import json, re, sys
from collections import Counter
from transformers import AutoTokenizer

hf, ap, out = sys.argv[1:4]
tok = AutoTokenizer.from_pretrained(hf)
A = json.load(open(ap))
BANDS = {"day": (0.0, 0.2), "days": (0.2, 0.35), "week": (0.35, 0.5)}
MIN_ANSWERS, LIFT_REF, LIFT_PREV, TOP, START = 4, 2.0, 1.5, 12, 0.3
STOP = set("the a an i you you're it's it is was my your to and of in on just so well me be been that this feels "
           "feel not when too very little i'm are were there for with at as but or like".split())


def spoken(x):
    x = re.sub(r"\([^)]*(\)|$)|\[[^\]]*(\]|$)", " ", x, flags=re.S)
    return re.sub(r"\*[^*]*(\*|$)", " ", x.replace("**", ""), flags=re.S)


def grams(x):
    w = re.findall(r"[a-z]+(?:'[a-z]+)?", spoken(x).lower().replace("’", "'"))
    return {" ".join(w[i:i + n]) for n in (2, 3) for i in range(len(w) - n + 1)}


levels = ["base", "zero", *BANDS]
df = {k: Counter(g for a in A[k] for g in grams(a)) for k in levels}
n = len(A["base"])
rate = lambda k, g: (df[k][g] + 1) / (n + 2)
ref = lambda g: max(rate("base", g), rate("zero", g))
ok = lambda g: g.split()[-1] not in STOP and len(set(g.split())) == len(g.split())


def forms(g):
    """token sequences for the phrase mid-sentence, sentence-start capitalized, and at a line start"""
    c = g[0].upper() + g[1:]
    return sorted({tuple(tok(f, add_special_tokens=False).input_ids) for f in (" " + g, " " + c, c, g)})


rungs, prev = [], None
for lv, (a, b) in BANDS.items():
    cand = []
    for g, c in df[lv].items():
        below = ref(g) if prev is None else max(ref(g), rate(prev, g))
        if c >= MIN_ANSWERS and ok(g) and rate(lv, g) / ref(g) >= LIFT_REF and rate(lv, g) / below >= LIFT_PREV:
            cand.append((rate(lv, g) / below, g))
    cand = sorted(cand, reverse=True)[:TOP]
    top = max([s for s, _ in cand], default=1.0)
    rungs.append({"level": lv, "from": a, "to": b,
                  "phrases": [{"text": g, "w": round(s / top, 3), "forms": [list(f) for f in forms(g)]}
                              for s, g in cand]})
    print(f"\n=== {lv} (dose {a}-{b}): base/zero/day/days/week of {n}")
    for s, g in cand:
        print(f"  {g:24s} {s:4.1f}x  " + " / ".join(str(df[k][g]) for k in levels))
    prev = lv
json.dump({"rungs": rungs, "dim": 0.5, "start": START}, open(out, "w"), indent=1)
print("PHRASES_DONE")
