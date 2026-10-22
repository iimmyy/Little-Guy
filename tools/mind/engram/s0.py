"""Engram S0: can his real engine (llama.cpp) keep a moment's KV cache and slide it into another chat?

Writes the 64 made-up moments of rungs 0 and 1 as cases, runs engine/kv-check on a brain (exactness checks first,
then every moment three ways: read fresh, saved cache moved in, nothing) and grades the greedy answers.

  python s0.py --brain BRAIN.gguf --out DIR [--layout before|after|recall] [--lora ADAPTER.gguf] [--no-swa-full]

Layouts (where the memory sits in chat B):
  before  right before the question; he reads the question once with the memory already in mind
  after   right after the question, still in the question's turn (nothing re-read)
  recall  after the question, at the start of his own turn, tagged with where it came from:
          You told me: "<memory>" (a memory comes up carrying its source, like human recollection; nothing re-read)
"""
import argparse, json, re, subprocess
from pathlib import Path

from moments import moments, prefix

HERE = Path(__file__).resolve().parent
KV_CHECK = HERE.parents[2] / "engine" / "kv-check" / "build" / "kv-check"


TURN = "<turn|>\n<|turn>model\n"


def layout(m, how):
    """(chat B up to the memory, chat B after it)"""
    b, q = prefix(m["ctx_b"]), m["question"]
    if how == "before":
        return b, "\n\n" + q + TURN
    if how == "after":
        return b + q + "\n\n", TURN
    return b + q + TURN + 'You told me: "', '"\n\n'


def has(text, value):
    return re.search(r"\b" + re.escape(value.lower()) + r"\b", text.lower()) is not None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layout", choices=["before", "after", "recall"], default="before")
    ap.add_argument("--lora", help="his talk adapter, worn for everything")
    ap.add_argument("--no-swa-full", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ms = moments()
    with open(out / "cases.jsonl", "w") as f:
        for m in ms:
            b, q = layout(m, a.layout)
            f.write(json.dumps(dict(a=prefix(m["ctx_a"]), moment=m["text"], b=b, q=q)) + "\n")
    cmd = [str(KV_CHECK), a.brain, str(out / "cases.jsonl"), str(out / "answers.jsonl")]
    if a.no_swa_full:
        cmd.append("--no-swa-full")
    if a.lora:
        cmd += ["--lora", a.lora]
    r = subprocess.run(cmd, capture_output=True, text=True)
    (out / "kv-check.log").write_text(r.stderr)
    lines = [json.loads(l) for l in r.stdout.splitlines() if l.startswith("{")]
    json.dump(lines, open(out / "checks.json", "w"), indent=1)
    for l in lines:
        print(json.dumps(l))
    if r.returncode != 0:
        print(f"kv-check stopped with code {r.returncode} (see kv-check.log)")
        return
    rows = [json.loads(l) for l in open(out / "answers.jsonl")]
    for row in rows:
        m = ms[row["i"]]
        row.update(kind=m["kind"], want=m["answer"], right=has(row["said"], m["answer"]))
    with open(out / "graded.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    summary = {w: sum(r["right"] for r in rows if r["way"] == w) for w in ("text", "kv", "none")}
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    print("said right of 64:", summary)


if __name__ == "__main__":
    main()
