#!/usr/bin/env python3
"""Grade the brake test (brake_sets.py sets; engine_run.py answers). Per set x model x mode (think / fast):
right, fell for the tempting answer (tricks: the recorded intuitive answer; red herrings: an answer that moves by the
distractor's own number; walls: the unrounded or wrongly rounded division), other wrong, cut off (no finished answer).
The answer is his own final answer in any form (answer_of: boxed, an answer line, else the last number; reply first,
thinking if the reply has none); "N cents" is read as dollars when the key is in dollars. Every miss is printed with its
answer line for hand reading.
usage: brake_grade.py <brake folder> <out.txt> [models=plain,c5,c5s]
"""
import json, math, os, re, sys

D, out = sys.argv[1], sys.argv[2]
opt = dict(a.split("=", 1) for a in sys.argv[3:])
MODELS = opt.get("models", "plain,c5,c5s").split(",")
NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def boxed_of(txt):
    """Contents of the last \\boxed{...} in txt, braces balanced, or None."""
    k = txt.rfind("\\boxed{")
    if k < 0:
        return None
    i, depth = k + 7, 1
    while i < len(txt) and depth:
        depth += {"{": 1, "}": -1}.get(txt[i], 0)
        i += 1
    return txt[k + 7:i - 1] if depth == 0 else None


def value_of(line):
    """First number on an answer line; a fraction a/b or \\frac{a}{b} is read as its value; "N cents" as dollars."""
    line = line.replace("$", "").replace("\\%", "").replace("\\,", "").replace("{,}", ",")
    line = line.rsplit("=", 1)[-1] if re.search(r"=.*\d", line) else line      # "x = 5": the value after the last equals
    mix = re.search(r"(?<![\d./])(\d+)\s*\\?d?(?:frac)?\s*\{?(\d+)\}?\s*(?:/|\}\{|\{)\s*(\d+)\}?(?![\d.])", line) if re.search(r"\d\s+\d+\s*/\s*\d|\d\s*\\d?frac", line) else None
    if mix and float(mix.group(3)):                    # mixed number: 6 5/8 or 6\\frac{5}{8} is 6.625, never 6
        return float(mix.group(1)) + float(mix.group(2)) / float(mix.group(3))
    fr = re.search(r"\\d?frac\{(-?[\d.,]+)\}\{([\d.,]+)\}", line) or re.search(r"(?<![\d.])(-?\d[\d,]*)\s*/\s*(\d[\d,]*)(?![\d.])", line)
    if fr and float(fr.group(2).replace(",", "")):
        return float(fr.group(1).replace(",", "")) / float(fr.group(2).replace(",", ""))
    m = NUM.findall(line)
    if not m:
        return None
    v = float(m[0].replace(",", ""))
    if re.search(r"\bcents?\b", line.lower()) and "." not in m[0]:
        v /= 100
    return v


def answer_of(r):
    """His final answer in any form (bare question, no format line): last \\boxed{} in the reply, else the last line
    naming an answer, else the last number in the reply's last lines; thinking only when the reply is empty."""
    for txt in (r["reply"], r["thinking"]):
        b = boxed_of(txt)
        if b is not None and value_of(b) is not None:
            return value_of(b), ("boxed: " + b)[:120]
        lines = [l for l in txt.splitlines() if "answer" in l.lower()]
        if lines:
            line = lines[-1].split(":", 1)[-1] if ":" in lines[-1] else lines[-1]
            v = value_of(line)
            if v is not None:
                return v, line.strip()[:120]
        if txt.strip():                                   # no box and no answer line: read the end of his reply by number
            last = [l for l in txt.splitlines() if NUM.search(l.replace("$", ""))]
            if last:
                nums = NUM.findall(last[-1].replace("$", ""))
                return float(nums[-1].replace(",", "")), ("last number: " + last[-1].strip())[:120]
    return None, (r["reply"].strip().splitlines() or [""])[-1][:120]


def close(a, b):
    return a is not None and abs(a - b) <= 1e-6 + 1e-6 * abs(b)


def tempting(row, v, set_):
    if v is None:
        return False
    if set_ == "tricks":
        return close(v, float(row["intuitive"]))
    if set_ == "herring":
        d = float(str(row.get("distractor_number", "nan")).replace(",", "")) if row.get("distractor_number") not in (None, "") else None
        A = float(row["answer"])
        return d is not None and any(close(v, x) for x in (A + d, A - d, A * d, A / d if d else math.nan))
    if set_ == "walls":
        m = re.match(r"\s*([\d.]+)\s*/\s*([\d.]+)", row.get("formula") or "")
        if not m:
            return False
        q = float(m.group(1)) / float(m.group(2))
        return any(close(v, x) for x in (q, round(q, 2), math.floor(q), math.ceil(q), float(m.group(1)) % float(m.group(2)))) and not close(v, float(row["answer"]))
    return False


L = []
for set_ in ("tricks", "herring", "walls"):
    for mode in ("think", "fast"):
        rows_by = {}
        for mdl in MODELS:
            f = os.path.join(D, f"brake-{set_}-{mdl}-{mode}.jsonl")
            if os.path.exists(f):
                rows_by[mdl] = {r["id"]: r for r in map(json.loads, open(f))}
        if not rows_by:
            continue
        ids = sorted(set.intersection(*[set(v) for v in rows_by.values()]))
        L.append(f"{set_} / {mode} ({len(ids)} problems answered by every model)")
        for mdl, rows in rows_by.items():
            c = {"right": 0, "tempting": 0, "wrong": 0, "cut": 0}
            misses = []
            for i in ids:
                r = rows[i]
                v, line = answer_of(r)
                if r["finish"] != "stop":
                    c["cut"] += 1
                    continue
                if close(v, float(r["answer"])):
                    c["right"] += 1
                    if re.search(r"frac|/|\d\s+\d", line):              # a "right" read off a fraction: read it by hand too
                        misses.append(("check-right", i, r["answer"], line))
                elif tempting(r, v, set_):
                    c["tempting"] += 1; misses.append(("TEMPTING", i, r["answer"], line))
                else:
                    c["wrong"] += 1; misses.append(("wrong", i, r["answer"], line))
            words = sum(int(rows[i]["think_tokens"]) for i in ids)
            L.append(f"  {mdl:6s} right {c['right']:3d}  tempting {c['tempting']:3d}  other wrong {c['wrong']:3d}  cut {c['cut']:3d}  "
                     f"thinking words {words}")
            for kind, i, key, line in misses:
                L.append(f"      {kind} {i}: key {key} | {line}")
open(out, "w").write("\n".join(L) + "\n")
print("\n".join(L))
