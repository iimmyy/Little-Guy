#!/usr/bin/env python3
"""Brake test sets (Immy 2026-11-03: "do all 4 kinds"; where does a "wait, this breaks the scene" brake earn its keep?).
Builds, from files downloaded with Immy's OK (each in its own folder, read as data, never run):
  tricks.jsonl   Hagendorff & Kosinski 2023 CRT variants (150, written by hypothesis-blind research assistants):
                 crt1 "more than" (bat and ball), crt2 machines/widgets, crt3 doubling. Each keeps the tempting
                 wrong answer (`intuitive`). Keys are checked against each type's own rule; a key that fails is left out.
  herring.jsonl  GSM-IC multi-step (Shi et al. 2023): GSM8K problems with one human-template irrelevant sentence;
                 only the most distracting kind (names overlap the story, number in range, on topic), 3 per base.
  walls.jsonl    ASDiv (Miao et al. 2020) problems labelled Ceil-Division, Floor-Division or Surplus (remainder): the
                 answer has to respect a wall (no half a bus; whole boxes; what is left over). Any problem whose text
                 is in our picture set (which c5s's formula line was fit on) is left out. GSM8K test was checked
                 first: only 2 of 1319 solutions round to whole objects.
The lists are pulled out of LLMs_responses.py with ast (its data section only); the file is never executed.
usage: brake_sets.py <brake folder>
"""
import ast, json, math, os, random, re, sys

D = sys.argv[1]
WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
         "eleven": 11, "twelve": 12, "half": 0.5}


def num(s):
    m = re.findall(r"-?\d+(?:\.\d+)?", s.replace(",", ""))
    if m:
        return float(m[0])
    for w, v in WORDS.items():
        if re.search(rf"\b{w}\b", s.lower()):
            return float(v)
    return None


src = "".join(open(os.path.join(D, "dl-osf", "LLMs_responses.py")).readlines()[:1915])   # data lists end before the code
crt = {}
for node in ast.parse(src).body:
    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ("crt1", "crt2", "crt3"):
        crt[node.targets[0].id] = ast.literal_eval(node.value)
tricks, bad = [], []
for k, items in crt.items():
    for x in items:
        c, i = num(x["correct"]), num(x["intuitive"])
        ok = c is not None and i is not None
        if ok and k == "crt1":
            ok = abs((float(x["total_cost"]) - float(x["more"])) / 2 - c) < 1e-6
        if ok and k == "crt3":
            ok = num(x["t"]) is not None and abs(num(x["t"]) - 1 - c) < 1e-6
        if not ok:
            bad.append((k, x["number"], x["task"], x["correct"], x["intuitive"]))
            continue
        tricks.append({"id": f"{k}-{x['number']:02d}", "cat": k, "q": x["task"], "answer": f"{c:g}", "intuitive": f"{i:g}",
                       "key_text": x["correct"]})
print(f"tricks: {len(tricks)} kept, {len(bad)} left out (key fails its own rule or has no number):")
for b in bad:
    print("   ", b)

d = json.load(open(os.path.join(D, "dl-gsmic", "GSM-IC_mstep.json")))
pick = [x for x in d if x["role_label"] == "overlapped" and x["number_label"] == "in_range" and x["sentence_label"] == "in_topic"]
by = {}
for x in pick:
    by.setdefault(x["original_question"], []).append(x)
rng = random.Random(0)
herring = []
for b, (base, xs) in enumerate(sorted(by.items())):
    for j, x in enumerate(rng.sample(xs, min(3, len(xs)))):
        herring.append({"id": f"ic-{b:02d}-{j}", "cat": "herring", "q": x["new_question"], "answer": x["answer"], "base": base,
                        "distractor_number": x["number"]})
print(f"red herrings: {len(herring)} from {len(by)} base problems")

walls = []
apath = os.path.join(D, "dl-asdiv", "ASDiv.xml")
pic = os.path.join(D, "picture-set-v1.jsonl")
norm = lambda t: re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()
seen = {norm(json.loads(l)["text"]) for l in open(pic)} if os.path.exists(pic) else set()
print(f"picture set texts to leave out: {len(seen)}")
if os.path.exists(apath):
    import xml.etree.ElementTree as ET
    left = 0
    for pr in ET.parse(apath).getroot().iter("Problem"):
        st = pr.findtext("Solution-Type")
        if st not in ("Ceil-Division", "Floor-Division", "Surplus"):
            continue
        q = (pr.findtext("Body") or "").strip() + " " + (pr.findtext("Question") or "").strip()
        if norm(q) in seen or any(norm(pr.findtext("Body") or "x") in t for t in seen):
            left += 1
            continue
        a = num(pr.findtext("Answer") or "")
        if a is None:
            continue
        walls.append({"id": pr.get("ID"), "cat": "wall", "q": q, "answer": f"{a:g}", "wall": st, "formula": pr.findtext("Formula")})
    kinds = {k: sum(1 for w in walls if w["wall"] == k) for k in ("Ceil-Division", "Floor-Division", "Surplus")}
    print(f"walls: {len(walls)} ASDiv problems {kinds}; {left} left out as already in the picture set")
for name, rows in (("tricks", tricks), ("herring", herring), ("walls", walls)):
    if rows:
        with open(os.path.join(D, f"{name}.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
