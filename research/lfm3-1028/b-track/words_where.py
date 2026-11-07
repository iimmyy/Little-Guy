"""Where do the words go? Sentence starts in LFM3 thinking, per 1000 words: cut-off runs that never wrote the key vs
finished-right runs (night dev set). usage: words_where.py name=a.jsonl,b.jsonl ..."""
import json, re, sys
from collections import Counter
KINDS = {
    "wait/hmm": r"(wait|hmm|hold on|oops)\b",
    "but/however": r"(but|however|though)\b",
    "actually/no": r"(actually|no[,.]|not quite|that's wrong|mistake)",
    "re-check": r"(let'?s|let me|we should|we need to|need to)\s+(double[- ]check|check|verify|re-?check|re-?examine|confirm|recompute|re-?evaluate)",
    "maybe/alt": r"(alternatively|maybe|perhaps|or maybe|another (way|approach)|instead)",
    "case/try": r"(case|try|suppose|if we|consider|test)\b",
}
def sents(t):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", t) if s.strip()]
def key_in(r, t):
    k = re.sub(r"\\!|\\,|\\text\{[^}]*\}|\$|\\left|\\right|\s", "", r["answer"])
    return len(k) > 2 and k in re.sub(r"\s", "", t)
for a in sys.argv[1:]:
    n, fs = a.split("=", 1)
    rows = [json.loads(l) for f in fs.split(",") for l in open(f)]
    groups = {"never (cut)": [r for r in rows if r["finish"] != "stop" and not key_in(r, r["thinking"])],
              "had it (cut)": [r for r in rows if r["finish"] != "stop" and key_in(r, r["thinking"])],
              "finished": [r for r in rows if r["finish"] == "stop"]}
    print(f"== {n}")
    for g, R in groups.items():
        c, words, ns = Counter(), 0, 0
        for r in R:
            ss = sents(r["thinking"]); ns += len(ss); words += len(r["thinking"].split())
            for s in ss:
                for k, p in KINDS.items():
                    if re.match(r"\W*" + p, s, re.I):
                        c[k] += 1; break
        if not R: continue
        per = lambda x: 1000 * x / max(1, words)
        print(f"  {g:13s} runs {len(R):3d}  words/run {words // len(R):5d}  sentences/1k {per(ns):5.1f}  " +
              "  ".join(f"{k} {per(c[k]):4.1f}" for k in KINDS) + f"  | all doubt-type {100 * sum(c.values()) / max(1, ns):4.1f}% of sentences")
