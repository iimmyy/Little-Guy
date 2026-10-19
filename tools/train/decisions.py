#!/usr/bin/env python3
"""His one-word decisions from practice days, each with what every option would have earned: the training set for
choices.py (counterfactual credit: like grading a multiple-choice answer with the key for every option, so training
never waits for a lucky draw; research 2026-10-19: group-relative training learns nothing when all draws agree).

  gate  a quiet moment with something coming up: think about a task (its word), wander, or no. A task word earns the
        difference one more check at that minute makes to that task's score (tasks.score with and without it: an
        early check to get his bearings +0.2, one while getting ready +0.5, each check -0.03, far from it -0.05 more);
        wander and no earn 0. Moments with nothing coming up teach nothing here and are left out.
  act   after a thought about a task: now or later. now earns what doing it then scores (on time +1, late +0.3, too
        early -1.3: the -0.3 and the +1 it gave up), later 0.
  text  what he said in his own words (thoughts, replies), for keeping him himself while training (each word's odds
        kept close to his untrained self)

Every row keeps the exact prompt he saw (rendered with his brain's own chat template), the face he made before
choosing, and his feelings at the time (the drip), so training sees what the live brainstem saw.
usage: decisions.py OUT.jsonl DIR [DIR ...]     (each DIR a practice.py day, notebook/ + steps.jsonl, or a drills
       folder, drills.jsonl + steps.jsonl: moments near and far from due, many of them on time)
env: TEMPLATE (his chat template, default ~/.littleguy/brain/brain.gguf.jinja)
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "room"))
sys.path.insert(0, os.path.join(HERE, "..", "mind"))
import tasks as tk
from brain import chat_template

render = chat_template(os.path.expanduser(os.environ.get("TEMPLATE", "~/.littleguy/brain/brain.gguf")))


def jl(p):
    try:
        return [json.loads(l) for l in open(p) if l.strip()]
    except OSError:
        return []


def structural(task, checks):
    """the part of a task's score that only the timing of his checks decides"""
    if not checks:
        return 0.0
    p = tk.score({**task, "status": "missed"}, [{"t": t} for t in checks])["parts"]
    return p["prep"] + p["calibrate"] + p["cost"] + p["far"]


def day(d):
    nb = f"{d}/notebook"
    tasks = {t["id"]: t for t in json.load(open(f"{nb}/tasks.json"))} if os.path.exists(f"{nb}/tasks.json") else {}
    own = {}
    for x in jl(f"{nb}/thoughts.jsonl"):
        if x.get("task") and not x.get("alarm"):
            own.setdefault(x["task"], []).append(x["t"])
    acts = {x["t"]: x for x in jl(f"{nb}/thoughts.jsonl") if x.get("act")}
    steps = jl(f"{d}/steps.jsonl")
    chooses = {}
    for s in steps:
        if s["op"] == "choose":
            chooses.setdefault((s["t"], tuple(s["req"]["choices"])), s)
    rows = []
    for m in jl(f"{nb}/moments.jsonl"):
        if m["kind"] != "silence" or not m.get("tasks"):
            continue
        s = chooses.get((m["t"], tuple(m["options"])))
        if not s:
            continue
        r = {}
        for word, tid in zip(m["options"], m["tasks"]):
            t, others = tasks[tid], [c for c in own.get(tid, []) if c != m["t"]]
            r[word] = round(structural(t, others + [m["t"]]) - structural(t, others), 4)
        r.update(wander=0.0, no=0.0)
        rows.append({"kind": "gate", "t": m["t"], "when": m["when"], "rewards": r, "picked": m["pick"],
                     "probs_then": m["probs"], **req(s)})
    for t, x in acts.items():
        s = chooses.get((t, ("now", "later")))
        if not s:
            continue
        task = tasks[x["task"]]
        now = tk.thought_reward(task, t, "", did=True)["act"]
        rows.append({"kind": "act", "t": t, "when": x["when"], "rewards": {"now": now, "later": 0.0},
                     "picked": x["act"]["pick"], "probs_then": x["act"]["probs"], "alarm": x.get("alarm"),
                     "left_min": x["true_left"], **req(s)})
    for s in steps:
        if s["op"] == "think" and s["out"].get("text") and not s["req"].get("thinking"):
            rows.append({"kind": "text", "t": s["t"], "prompt": render(s["req"]["messages"]), "text": s["out"]["text"],
                         "v": s["req"]["v"], "a": s["req"]["a"]})
    return rows


def drills(d):
    """practice.py drills: one moment, several thoughts, each followed by his now-or-later choice (no alarm)"""
    rows, by_cue = [], {r["messages"][1]["content"]: r for r in jl(f"{d}/drills.jsonl")}
    part = lambda row: f"{os.path.basename(d.rstrip('/'))}-{row['i'] % 5}"      # fifths, so some can be held out
    for s in jl(f"{d}/steps.jsonl"):
        msgs = s["req"].get("messages", [])
        row = by_cue.get(msgs[1]["content"]) if len(msgs) > 1 else None
        if row is None:
            continue
        if s["op"] == "choose" and s["req"]["choices"] == ["now", "later"]:
            left = row["left_min"]
            now = tk.thought_reward({"due": left * 60}, 0, "", did=True)["act"]
            rows.append({"kind": "act", "t": s["t"], "when": row["when"], "rewards": {"now": now, "later": 0.0},
                         "picked": None, "probs_then": None, "alarm": False, "left_min": left, "day": part(row),
                         **req(s)})
        elif s["op"] == "think" and len(msgs) == 2 and s["out"].get("text"):
            rows.append({"kind": "text", "t": s["t"], "prompt": render(msgs), "text": s["out"]["text"],
                         "v": s["req"]["v"], "a": s["req"]["a"], "day": part(row)})
    return rows


def req(s):
    return {"prompt": render(s["req"]["messages"]), "face": s["out"].get("face", ""), "choices": s["req"]["choices"],
            "v": s["req"]["v"], "a": s["req"]["a"], "logprobs_then": s["out"]["logprobs"]}


if __name__ == "__main__":
    out, dirs = sys.argv[1], sys.argv[2:]
    n = {}
    with open(out, "w") as f:
        for d in dirs:
            for r in (drills(d) if os.path.exists(f"{d}/drills.jsonl") else day(d)):
                # the stream's folder too: every stream numbers its days from day-000 (days-g1/day-018)
                r.setdefault("day", "/".join(os.path.normpath(d).split(os.sep)[-2:]))
                n[r["kind"]] = n.get(r["kind"], 0) + 1
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(n))
