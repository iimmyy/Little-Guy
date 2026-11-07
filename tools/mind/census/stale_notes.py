#!/usr/bin/env python3
"""Do stale notes hurt him? (Immy 2026-11-02: a mental image holds while you solve and then fades; his full-attention
layers keep every note forever.) Text-level first look, untouched model's own thinking, before any build.

A result he RETRACTS: a sentence states a value (after "=", "is", "gives", "get", "equals") and the very next sentence
opens with a doubt word (wait / no / actually / hmm / oops / that's wrong / mistake / not right). Given numbers (in the
question) and 0, 1, 2 are ignored.
v2 (after reading v1's hits by hand, most were "wait... actually 60" re-confirming the same right value): a retraction
counts only if the next two sentences state a different value and not the same one.
A STALE REUSE: later, at least one sentence after he stated some other value, the retracted value is stated as a
result again and the next sentence does not doubt it.
Per run: retractions, stale reuses, and whether the final answer is a value he himself retracted earlier.
Split by outcome (right / finished wrong / cut off). Every hit is printed for reading by hand.

usage: stale_notes.py <out.txt> <answers.jsonl[,more]>
"""
import json, re, sys
from collections import defaultdict
sys.path.insert(0, __file__.rsplit("/", 1)[0] + "/../expert")
out, files = sys.argv[1], sys.argv[2].split(",")
sys.argv = sys.argv[:1]
from grade import right_reply

DOUBT = re.compile(r"\W*(wait|no[,.!]|actually|hmm|oops|hold on|that'?s (wrong|not right|incorrect)|mistake|not right|"
                   r"that can'?t be|that doesn'?t|this is wrong|i made)", re.I)
NUM = r"-?\d+(?:\.\d+)?(?:/\d+)?"
RES = re.compile(rf"(?:=|\bis\b|\bgives\b|\bget\b|\bequals\b|\bbecomes\b)\s*\$?\\?\(?\s*({NUM})(?![\d/.]*\d)")


def norm(v):
    try:
        if "/" in v:
            a, b = v.split("/")
            return round(float(a) / float(b), 6)
        return round(float(v), 6)
    except (ValueError, ZeroDivisionError):
        return None


def sentences(t):
    return [s for s in re.split(r"(?<=[.!?])\s+|\n+", t) if s.strip()]


given_of = lambda q: {norm(x) for x in re.findall(NUM, q)}
rows, lines = [], []
for f in files:
    for k, r in enumerate(map(json.loads, open(f))):
        if r.get("cat") == "math-fresh":
            r["cat"] = "math500"
        out_k = "cut" if r["finish"] != "stop" else ("right" if right_reply(r, r["reply"]) else "wrong")
        S = sentences(r["thinking"])
        given = given_of(r["q"]) | {0.0, 1.0, 2.0}
        vals = [[norm(m[1]) for m in RES.finditer(s)] for s in S]
        retracted, events, last_other = {}, [], {}
        for i, vs in enumerate(vals):
            # v2: a real retraction = doubted AND the next two sentences state a DIFFERENT result (he replaced it);
            # "wait... yes, 60" re-confirms the same value and is counted apart as a re-check
            nxt = {w for vv in vals[i + 1:i + 3] for w in vv if w is not None and w not in given}
            doubted = i + 1 < len(S) and bool(DOUBT.match(S[i + 1])) and any(w not in vs for w in nxt) and not set(vs) & nxt
            for v in vs:
                if v is None or v in given:
                    continue
                if v in retracted and not doubted and any(
                        w not in (None, v) and w not in given for vv in vals[retracted[v] + 1:i] for w in vv):
                    events.append((retracted[v], i, v))
                if doubted:
                    retracted[v] = i
        final = norm(m[1]) if (m := re.search(r"boxed\{\s*(" + NUM + r")\s*\}", r["reply"])) else None
        stale_final = final is not None and final in retracted and out_k != "right"
        rows.append((f.rsplit("/", 1)[-1], r["id"], out_k, len(retracted), len(events), stale_final))
        for a, b, v in events[:2]:
            lines.append(f"  [{out_k}] {r['id']} ({f.rsplit('/', 1)[-1]}): {v:g} retracted at sentence {a}, back at {b}:"
                         f"\n      RETRACTED: {S[a][-140:]!r} -> {S[a + 1][:80]!r}\n      BACK:      {S[b][-160:]!r}")
by = defaultdict(list)
for r in rows:
    by[r[2]].append(r)
L = [f"stale notes, text-level, untouched runs: {', '.join(f.rsplit('/', 1)[-1] for f in files)} ({len(rows)} runs)"]
for k in ("right", "wrong", "cut"):
    R = by[k]
    if R:
        L.append(f"  {k:5s} runs {len(R):3d}: with a retraction {sum(r[3] > 0 for r in R):3d}, with a stale reuse {sum(r[4] > 0 for r in R):3d} "
                 f"({100 * sum(r[4] > 0 for r in R) / len(R):.0f}%), reuses per run {sum(r[4] for r in R) / len(R):.2f}"
                 + (f", final answer = a value he retracted {sum(r[5] for r in R)}" if k != "right" else ""))
L.append("hits (first two per run), for reading:")
open(out, "w").write("\n".join(L + lines) + "\n")
print("\n".join(L[:5]))
