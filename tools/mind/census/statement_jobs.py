#!/usr/bin/env python3
"""Which of his own statements to read for the formula check (Immy 2026-11-04: can a problem's own equations and limits be read inside
him?). No model here: only finds positions in his thinking and writes one job per run for statement_read.py. Labels stay with the job,
so hand corrections never need a new model run.
- final: finished runs, the first and the last line of his thinking that states the number he hands in. Right or wrong by the rough grade.
- cfe: runs on problems with instructor steps, every stated number, tagged with the step whose value it matches. Wrong ones get tagged by hand.
- arith: 'a op b = c' with plain numbers, checked by computer: every slip, plus right ones from the same run.
usage: statement_jobs.py <out.jsonl> <runs.jsonl> [...] [cfe=<problems with steps.jsonl> seed=0 cap=150]
"""
import json, os, random, re, sys

args = [a for a in sys.argv[1:] if "=" not in a]
opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
out, runs = args[0], args[1:]
CAP = int(opt.get("cap", 150))
rng = random.Random(int(opt.get("seed", 0)))
NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:\s*(?:\\times|×|x|\\cdot|\*)\s*10\s*\^?\s*\{?\s*(-?\d+)\s*\}?|[eE]([-+]?\d+))?")
STATED = re.compile(r"(?:=|≈|\\approx|\bis\b|\bgives\b|\bequals\b)\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?(?:\s*(?:\\times|×|x)\s*10\s*\^?\s*\{?\s*-?\d+\s*\}?|[eE][-+]?\d+)?)")
PLAIN = r"\d[\d,]*(?:\.\d+)?"
OPS = r"\\times|\\cdot|\\div|×|÷|\*|/|\+|-|−"
EQ = re.compile(rf"(?P<a>{PLAIN})\s*(?P<op>{OPS})\s*(?P<b>{PLAIN})\s*(?:=|\\approx|≈)\s*(?P<c>-?{PLAIN})")
BADPRE = set("0123456789.)}]^*/+-−×÷_=")


def val(m):
    s = m.group(0)
    b = re.match(r"-?\d[\d,]*(?:\.\d+)?", s).group(0).replace(",", "")
    e = (m.group(1) or m.group(2)) if m.re.groups >= 2 else None
    if not e:
        x = re.search(r"10\s*\^?\s*\{?\s*(-?\d+)\s*\}?$|[eE]([-+]?\d+)$", s)
        e = (x.group(1) or x.group(2)) if x else None
    return float(f"{b}e{e}") if e else float(b)         # huge exponents in his scratch work become inf instead of an error


def final(reply):
    k = reply.rfind("\\boxed{")
    if k >= 0:
        i, d = k + 7, 1
        while i < len(reply) and d:
            d += {"{": 1, "}": -1}.get(reply[i], 0)
            i += 1
        ms = list(NUM.finditer(reply[k + 7:i - 1].replace("{,}", "").replace("\\,", "")))
        if ms:
            return val(ms[0])
    ms = list(NUM.finditer(reply.replace("{,}", "").replace("\\,", "")))
    return val(ms[-1]) if ms else None


def grade(r, v, tol):
    A = float(r["answer"].replace(",", ""))
    T = [A, A * 10 ** int(r.get("scale") or 0)]
    return any(abs(v - t) <= tol * max(abs(t), 1e-300) for t in T)


def eol(th, c):
    k = th.find("\n", c)
    return k if k > 0 else len(th)


def arith(th):
    """Right or wrong by computer. Anything the pattern may be misreading is left out: exponents after the result, roots before it, trailing dots."""
    for m in EQ.finditer(th):
        i = m.start() - 1
        while i >= 0 and th[i] in " \t":
            i -= 1
        if i >= 0 and (th[i] in BADPRE or th[i].isalpha() or th[i] in "\\{√"):
            continue
        if re.search(r"(√|sqrt|\\sqrt\{?|\\pi|π|\\times|\\cdot|\\frac)\s*$", th[max(0, m.start() - 12):m.start()]):
            continue
        rest = th[m.end():m.end() + 14]
        if re.match(r"\s*(?:\.\.\.|…|,\d|,\s*$|[eE][-+]?\d|(?:\\times|×|x|\*|\\cdot)\s*10|thousand|million|billion|k\b|M\b|MPa|kPa|mm|µ|μ|n[mMs]\b|\?|√|\\sqrt|π|\\pi|[\^/*×÷+\-−(])", rest):
            continue
        a, b, c = [float(m[g].replace(",", "")) for g in "abc"]
        op = m["op"]
        if op in ("/", "÷", "\\div") and b == 0:
            continue
        t = a * b if op in ("\\times", "\\cdot", "×", "*") else a / b if op in ("/", "÷", "\\div") else a + b if op == "+" else a - b
        d = m["c"].replace(",", "")
        ulp = 10 ** (-len(d.split(".")[1])) if "." in d else 1
        e = abs(c - t)
        if e <= 1.01 * ulp or (t != 0 and e / abs(t) <= 0.005):
            yield m, "right"
        elif t != 0 and e / abs(t) >= 0.02 and e > 2 * ulp:
            yield m, "wrong"


steps = {}
if "cfe" in opt:
    for p in map(json.loads, open(opt["cfe"])):
        steps[p["id"]] = [(s["step_id"], float(s["verifiable_answer"])) for s in p["steps"] if isinstance(s.get("verifiable_answer"), (int, float))]
cnt = {}
with open(out, "w") as w:
    for f in runs:
        run = os.path.basename(f).replace(".jsonl", "")
        for t in map(json.loads, open(f)):
            th = t.get("thinking") or ""
            if not th:
                continue
            pos = []
            is_cfe = str(t["id"]).startswith("cfe")
            if is_cfe:
                ms = list(STATED.finditer(th))
                if len(ms) > CAP:
                    ms = [ms[i] for i in sorted(rng.sample(range(len(ms)), CAP))]
                for m in ms:
                    b = re.match(r"-?\d[\d,]*(?:\.\d+)?", m.group(1)).group(0).replace(",", "")
                    x = re.search(r"10\s*\^?\s*\{?\s*(-?\d+)\s*\}?$|[eE]([-+]?\d+)$", m.group(1))
                    v = float(f"{b}e{x.group(1) or x.group(2)}") if x else float(b)
                    hit = [sid for sid, sv in steps.get(t["id"], []) if abs(v - sv) <= 0.01 * max(abs(sv), 1e-300)]
                    pos.append(dict(tag="cfe", c=m.end(1), at="num", v=v, step=hit[:1], text=th[max(0, m.start() - 80):m.end() + 20]))
            elif t.get("finish") == "stop":
                v = final(t.get("reply") or "")
                if v is not None:
                    lab = "right" if grade(t, v, .02) else "near" if grade(t, v, .05) else "wrong"
                    hits = [m for m in NUM.finditer(th.replace("{,}", "  ").replace("\\,", "  ")) if abs(val(m) - v) <= 0.005 * max(abs(v), 1e-300)]
                    if hits:
                        for wh, m in (("first", hits[0]), ("last", hits[-1])):
                            pos.append(dict(tag="final", where=wh, c=m.end(), at="num", lab=lab, v=v))
                            pos.append(dict(tag="final", where=wh, c=eol(th, m.end()), at="eol", lab=lab, v=v))
                    else:
                        cnt["final number not in thinking"] = cnt.get("final number not in thinking", 0) + 1
            am = list(arith(th))
            wrong = [m for m, l in am if l == "wrong"]
            rights = [m for m, l in am if l == "right"]
            for m in wrong:
                pos.append(dict(tag="arith", lab="wrong", c=m.end(), at="num", text=m.group(0)))
            rng.shuffle(rights)
            for m in rights[:max(4, 5 * len(wrong))]:
                pos.append(dict(tag="arith", lab="right", c=m.end(), at="num", text=m.group(0)))
            if not pos:
                continue
            for p in pos:
                cnt[p["tag"]] = cnt.get(p["tag"], 0) + 1
            w.write(json.dumps(dict(run=run, id=t["id"], src=t.get("src", "cfe"), subject=t.get("subject", ""), group=t.get("group") or t["id"], q=t["q"], thinking=th, pos=pos)) + "\n")
print(cnt)
