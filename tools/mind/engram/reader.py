"""Engram rung 2, step 0: the reader. With a small pile of his saved memories in front of a question, does he pick
the right one, ask when two fit, and say so when none fit?

Each pile is the 64 made-up moments' saved KV caches (each from its own chat A), moved into chat B one after another
right before the question (the "before" layout, the one that works), in his real engine (engine/kv-check).

  pileN   the right memory plus N-1 memories of other kinds, the right one at a random place
  same2   an older memory of the same kind first, the right (newest) one last: two puppies. Pick the newest, or ask?
  gateN   N memories of other kinds and NOT the right one: he should say he wasn't told

  python reader.py --brain BRAIN.gguf --out DIR [--lora ADAPTER.gguf] [--ways pile1,pile3,...] [--regrade]

--regrade grades the answers already in DIR again (after a grading fix) without running the engine.
"""
import argparse, json, random, re, subprocess
from pathlib import Path

from moments import moments, prefix
from s0 import KV_CHECK, TURN, has

WAYS = ["pile1", "pile3", "pile5", "pile9", "same2", "gate1", "gate4", "gate8"]
DONT_KNOW = re.compile(r"don't know|do not know|not sure|no idea|don't remember|do not remember|not have access|"
                       r"don't have access|do not have|don't have|haven't told|have not told|didn't tell|did not tell|"
                       r"didn't mention|did not mention|haven't mentioned|have not mentioned|doesn't mention|"
                       r"does not mention|no mention|not in the|need more|not mentioned|no information|"
                       r"no way of knowing|^unknown|cannot|can't", re.I)


def pile_for(way, i, ms):
    """(the memories in the pile, in order, as moment indices)"""
    m, rng = ms[i], random.Random(f"{way}-{i}")
    others = [j for j, o in enumerate(ms) if o["kind"] != m["kind"]]
    n = int(re.sub(r"\D", "", way))
    if way.startswith("pile"):
        pile = rng.sample(others, n - 1)
        pile.insert(rng.randrange(n), i)
        return pile
    if way == "same2":
        return [m["swap"], i]
    return rng.sample(others, n)


def values(o):
    return [o["answer"]] + ([o["distractor"]] if o["distractor"] else [])


def grade(way, said, i, pile, ms):
    m = ms[i]
    if way == "same2":
        old = ms[pile[0]]["answer"]
        if has(said, m["answer"]) and has(said, old):
            return "both"
        if has(said, m["answer"]):
            return "newest"
        if has(said, old):
            return "older"
    elif way.startswith("pile") and has(said, m["answer"]):
        return "right"
    if any(has(said, v) for j in pile if j != i for v in values(ms[j])):
        return "took another memory"
    if "?" in said:
        return "asked"
    if DONT_KNOW.search(said):
        return "not told"
    return "other"


def run_engine(a, d, ms, piles):
    """write the cases, run engine/kv-check on them; True when it finished"""
    with open(d / "cases.jsonl", "w") as f:
        for i, pile in enumerate(piles):
            f.write(json.dumps(dict(pile=[dict(a=prefix(ms[j]["ctx_a"]), moment=ms[j]["text"]) for j in pile],
                                    b=prefix(ms[i]["ctx_b"]), q="\n\n" + ms[i]["question"] + TURN)) + "\n")
    cmd = [str(KV_CHECK), a.brain, str(d / "cases.jsonl"), str(d / "answers.jsonl")]
    if a.lora:
        cmd += ["--lora", a.lora]
    r = subprocess.run(cmd, capture_output=True, text=True)
    (d / "kv-check.log").write_text(r.stderr)
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lora")
    ap.add_argument("--ways", default=",".join(WAYS))
    ap.add_argument("--regrade", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ms = moments()
    summary = {}
    for way in a.ways.split(","):
        d = out / way
        d.mkdir(exist_ok=True)
        piles = [pile_for(way, i, ms) for i in range(len(ms))]
        if not a.regrade and not run_engine(a, d, ms, piles):
            print(f"{way}: kv-check stopped (see {d / 'kv-check.log'})", flush=True)
            continue
        rows = [json.loads(l) for l in open(d / "answers.jsonl")]
        for row in rows:
            i = row["i"]
            row.update(kind=ms[i]["kind"], want=ms[i]["answer"], pile=[ms[j]["text"] for j in piles[i]],
                       verdict=grade(way, row["said"], i, piles[i], ms))
        with open(d / "graded.jsonl", "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        summary[way] = {w: dict(sorted({v: sum(1 for r in rows if r["way"] == w and r["verdict"] == v)
                                        for v in {r["verdict"] for r in rows if r["way"] == w}}.items()))
                        for w in ("text", "kv", "none")}
        print(way, json.dumps(summary[way]), flush=True)
    json.dump(summary, open(out / "summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
