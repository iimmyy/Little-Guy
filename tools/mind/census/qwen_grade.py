#!/usr/bin/env python3
"""Grade Qwen3-Thinking night answers. Qwen reads "Answer: <your answer>" literally and writes "Answer: <4>",
"<Answer> (1, -16, -4, 43) </Answer>" or wraps the value in <span style="color:blue">...</span>; the shared grader reads
those as wrong (2026-11-02: all 9 "finished wrong" on the untouched night run were right). The wrappers are stripped
before grading, the same for every Qwen file so comparisons stay fair. 2026-11-02 (before the fresh test was graded):
also "Answer: boxed{9}", "answer16/answer" (tags lost) and a copied "Answer: your answer" line (its last \\boxed{} is used);
when the Answer line still fails, the last boxed value is read as a second chance (same for every file).

usage: qwen_grade.py name=answers.jsonl ...
"""
import json, os, re, sys

ARGS, sys.argv[1:] = sys.argv[1:], []
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "expert"))
from grade import right_reply


def clean(reply):
    reply = re.sub(r"</?span[^>]*>", "", reply)
    reply = re.sub(r"<Answer>\s*(.*?)\s*</Answer>", r"Answer: \1", reply, flags=re.S)
    reply = re.sub(r"<\s*([^<>\n]{1,80}?)\s*>", r"\1", reply)         # "<4>" -> "4", never touches "a < b > c" spans over lines
    reply = re.sub(r"(?im)^(\W*answer\W*[:：]?\s*)boxed\{([^{}\n]*)\}\s*$", r"\1\2", reply)    # "Answer: boxed{9}"
    reply = re.sub(r"(?im)^\s*answer\s*([^\n/]{1,80}?)\s*/answer\s*$", r"Answer: \1", reply)   # "answer16/answer" (tags lost)
    if re.search(r"(?im)^\W*answer\W*[:：]\s*your answer\W*$", reply):   # copied the prompt line: its last \boxed{} is the answer
        b = re.findall(r"\\boxed\{([^{}]*)\}", reply)
        if b:
            reply += f"\nAnswer: {b[-1]}"
    return reply


def boxed_last(reply):
    """the last boxed value, also in the broken forms Qwen writes ("boxed2>", "boxed\\dfrac{a}{b}", "\\boxed{-256}")"""
    m = re.findall(r"\\?boxed\s*\{((?:[^{}]|\{[^{}]*\})*)\}", reply) or re.findall(r"\\?boxed\s*(\\?[^\s<>$]+)", reply)
    return m[-1] if m else None


def ok(r):
    if r["finish"] != "stop":
        return False
    c = clean(r["reply"])
    if right_reply(r, c):
        return True
    b = boxed_last(c)                                                    # second reading: the boxed value, if any
    return bool(b) and bool(right_reply(r, f"Answer: {b}"))


for a in ARGS:
    n, f = a.split("=", 1)
    R = [json.loads(l) for l in open(f)]
    right = sum(ok(r) for r in R)
    cut = sum(r["finish"] != "stop" for r in R)
    print(f"{n:12} n={len(R)} right {right} cut {cut} finished-wrong {len(R) - right - cut} thinking {sum(r['think_tokens'] for r in R):,}")
    for r in R:
        if r["finish"] == "stop" and not ok(r):
            print(f"    WRONG {r['id']} key {r['answer']!r} reply end {clean(r['reply'])[-100:]!r}")
