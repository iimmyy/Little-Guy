#!/usr/bin/env python3
"""When has a patient already solved it? Labels for the overthinking detector (Immy, 2026-10-28: "track what happens
when the model's overthinking internally ... when enough of those flags hit, route the be-brief neurons up").

Each answer's thinking is cut at 10 points (10%, 20% ... 100% of its thinking words). At each cut the thinking is
closed by force and the patient has to answer right away ("Answer:", greedy, a few words). Once its forced answer is
right and stays right at every later cut, it already had the answer: every thinking word after that point is
overthinking (re-checking a solved problem). Words before it are working. (Forced early answers to read a reasoning
model's progress: DEER 2025, "dynamic early exit"; here only as labels, nothing is cut at run time.)

Two steps, because the MATH grader (math_verify) lives on the Mac:
usage: overthink_label.py gen <patient> <answers jsonl> <out jsonl> [cuts=10 dev=cuda batch=10]
       overthink_label.py grade <out jsonl>      adds right-per-cut and the solved point; prints the summary
"""
import json, os, sys

mode = sys.argv[1]
if mode == "grade":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../expert"))
    path, argv = sys.argv[2], sys.argv
    sys.argv = argv[:1]                                        # grade.py grades its own command line on import
    from grade import right
    sys.argv = argv
    rows = [json.loads(l) for l in open(path)]
    for r in rows:
        r["forced_right"] = [bool(right(r, line)) for line in r["forced"]]
        k = len(r["forced_right"])
        while k > 0 and r["forced_right"][k - 1]:             # earliest cut from which it stays right
            k -= 1
        r["solved_cut"] = k if k < len(r["forced_right"]) else None      # index into cuts; None = never settled
    with open(path.replace(".jsonl", ".graded.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    rows = [r for r in rows if r["think_tokens"] > 0]          # cut-offs never closed their thinking: no thinking text
    settled = [r for r in rows if r["solved_cut"] is not None]
    over = [1 - r["cuts"][r["solved_cut"]] / r["think_tokens"] for r in settled]
    print(f"{len(rows)} answers; settled on the right answer before the end: {len(settled)}")
    print(f"  share of the thinking after it had the answer (overthinking): mean {sum(over) / max(1, len(over)):.0%}")
    print(f"  forced answer right at each cut: " + " ".join(
        f"{sum(r['forced_right'][j] for r in rows)}" for j in range(len(rows[0]["cuts"]))))
    sys.exit()

import torch

sys.path.insert(0, os.path.dirname(__file__))
from patients import load

name, src, out = sys.argv[2:5]
opt = dict(a.split("=", 1) for a in sys.argv[5:])
CUTS, DEV, BATCH = int(opt.get("cuts", 10)), opt.get("dev", "cuda"), int(opt.get("batch", 10))
TAIL = "\n\nEnd with a line: Answer: <your answer>"
FORCE = {"e4b": lambda th: "<|channel>thought\n" + th + "<channel|>Answer:",
         "lfm3": lambda th: "\n" + th + "\n</think>\n\nAnswer:"}[name]
p = load(name, DEV)
tok = p.tok
tok.padding_side = "left"
done = {json.loads(l)["id"] for l in open(out)} if os.path.exists(out) else set()
with open(out, "a") as f, torch.no_grad():
    for r in (json.loads(l) for l in open(src)):
        if r["id"] in done:
            continue
        prompt = tok.apply_chat_template([{"role": "user", "content": r["q"] + TAIL}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=True)
        th = tok(r["thinking"], add_special_tokens=False).input_ids
        T = len(th)
        cuts = [max(1, round(T * (j + 1) / CUTS)) for j in range(CUTS)]
        texts = [prompt + FORCE(tok.decode(th[:c])) for c in cuts]
        lines = []
        for b0 in range(0, len(texts), BATCH):
            enc = tok(texts[b0:b0 + BATCH], return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
            o = p.model.generate(**enc, max_new_tokens=40, do_sample=False, pad_token_id=tok.pad_token_id)
            for s in o[:, enc.input_ids.shape[1]:]:
                lines.append(tok.decode(s, skip_special_tokens=True).strip().split("\n")[0])
        f.write(json.dumps({"id": r["id"], "cat": r["cat"], "answer": r["answer"], "reply": r["reply"],
                            "think_tokens": T, "cuts": cuts, "forced": lines, "finish": r["finish"]}) + "\n")
        f.flush()
        print(f"  {r['id']}: {T} thinking words, forced: {' | '.join(x[:20] for x in lines)}", flush=True)
print("LABEL_DONE")
