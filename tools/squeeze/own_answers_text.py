#!/usr/bin/env python3
"""A drift ruler made of the 16-bit original's own answers (from gen_bench.sh samples).

Each question in Gemma's chat format followed by the exact answer the original wrote (greedy), so drift measured on
this text is drift at the original's own decision points: wherever a 4-bit brain would pick a different next word
than the original did, its answer would start to change. GSM8K keeps only the last question (not the 8 worked
examples) so the text is mostly the original's own words. The three tests are interleaved (math, instructions, code,
math, ...) because the ruler reads only the first N windows. Same format as chat_heldout.txt (no <bos>).
usage: own_answers_text.py <original's -short/<name> sample dir> <out.txt>
"""
import json, sys
from pathlib import Path

src, out = Path(sys.argv[1]), Path(sys.argv[2])
per, counts = {}, {}
for task in ("gsm8k_cot", "ifeval", "humaneval_instruct"):
    f = sorted(src.glob(f"samples_{task}_*.jsonl"))[-1]
    seen = set()
    for line in open(f):
        d = json.loads(line)
        if d["doc_id"] in seen:            # GSM8K logs each question once per answer filter; the reply is the same
            continue
        seen.add(d["doc_id"])
        prompt, reply = d["arguments"]["gen_args_0"]["arg_0"], d["resps"][0][0]
        prompt = prompt.removeprefix("<bos>")
        if task == "gsm8k_cot":
            prompt = prompt[prompt.rindex("<|turn>user"):]
        per.setdefault(task, []).append(prompt + reply.rstrip() + "<turn|>\n")
        counts[task] = counts.get(task, 0) + 1
parts = [q for row in __import__("itertools").zip_longest(*per.values()) for q in row if q]
out.write_text("".join(parts))
print(f"{out}: {counts}, {sum(map(len, parts)):,} characters")
