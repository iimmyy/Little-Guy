#!/usr/bin/env python3
"""His real arithmetic in other saved untouched LFM3 runs. No sealed sets. No model."""
import collections, json, random, re
PLAIN = r"\d[\d,]*(?:\.\d+)?"
OPS = r"\\times|\\cdot|\\div|×|÷|\*|/|\+|-|−"
EQ = re.compile(rf"(?P<a>{PLAIN})\s*(?P<op>{OPS})\s*(?P<b>{PLAIN})\s*(?P<eq>=|\\approx|≈)\s*(?P<c>{PLAIN})")
BADPRE = set("0123456789.)}]^*/+-−×÷_=")
DOUBT = re.compile(r"\b(wait|actually|hold on|hmm|oops|double.?check|re-?check|verify|recalc\w*|recompute|re-compute|let me check|let's check|let me verify|let's verify|not right|seems off|that's off|seems wrong|no,|too high|too low)", re.I)
R = '../..'
FILES = ['sweep/eval-plain.jsonl', 'sweep/eval-plain-s1.jsonl', 'scene/scene-think-plain.jsonl', 'math100-plain.jsonl', 'science/peek/base-g.jsonl', 'science/peek/base-s1.jsonl', 'science/peek/base-s2.jsonl', 'science/peek/base-s3.jsonl', 'science/hold-1104/plain-free-v1.jsonl', 'situation/sit-think-plain.jsonl', 'situation/picture-think-plain-box.jsonl']
old = set()
for l in open('../real1/real-steps-v1.jsonl'):
    j = json.loads(l)
    old.add((j['id'], j['expr'], j['written']))
T = collections.Counter()
out, seen = [], set()
for f in FILES:
    for l in open(f'{R}/{f}'):
        r = json.loads(l)
        if r.get('src') == 'cfe' or not r.get('thinking'):
            continue
        th = r['thinking']
        for m in EQ.finditer(th):
            i = m.start() - 1
            while i >= 0 and th[i] in " \t":
                i -= 1
            if i >= 0 and (th[i] in BADPRE or th[i].isalpha() or th[i] in "\\{√"):
                continue
            if re.search(r"(√|sqrt|\\sqrt\{?|\\pi|π|\\times|\\cdot|\\frac)\s*$", th[max(0, m.start() - 12):m.start()]):
                continue
            rest = th[m.end():m.end() + 14]
            if re.match(r"\s*(?:\.\.\.|…|,\d|,\s*$|[eE][-+]?\d|(?:\\times|×|x|\*|\\cdot)\s*10|thousand|million|billion|k\b|M\b|MPa|kPa|mm|µ|μ|n[mMs]\b|√|\\sqrt|π|\\pi|[\^/*×÷+\-−(])", rest):
                continue
            if m["c"].endswith(","):
                continue
            a, b, c = [float(m[g].replace(",", "")) for g in "abc"]
            op = m["op"]
            if op in ("/", "÷", "\\div") and b == 0:
                continue
            t = a * b if op in ("\\times", "\\cdot", "×", "*") else a / b if op in ("/", "÷", "\\div") else a + b if op == "+" else a - b
            d = m["c"].replace(",", "")
            ulp = 10 ** (-(len(d.split(".")[1]) if "." in d else 0))
            e = abs(c - t)
            if t <= 0:
                continue
            if e <= 1.01 * ulp or e / t <= 0.0005:
                kind = "right"
            elif e > 2 * ulp and e / t >= 0.005:
                kind = "wrong"
            else:
                continue
            exprs = th[m.start():m.start("c")]
            key = (r['id'], exprs, m["c"])
            if key in old or key in seen:
                continue
            seen.add(key)
            after = th[m.end():m.end() + 120]
            out.append(dict(id=r['id'], run=f, q=r['q'], pre=th[:m.start("c")], written=m["c"], true=t, expr=exprs, kind=kind, eq=m["eq"], qmark=rest.lstrip().startswith("?"), doubt_next=bool(DOUBT.search(after)), rel=e / t, after=after))
            T[kind] += 1
            T[f"{kind}:{f.split('/')[-1]}"] += 1
for i, s in enumerate(out):
    s["step"] = 100000 + i
print({k: T[k] for k in sorted(T)})
with open('real2-steps-v1.jsonl', 'w') as w:
    for s in out:
        w.write(json.dumps(s) + "\n")
v = [s for s in out if s["kind"] == "wrong"]
with open('real2-wrong-list-v1.txt', 'w') as w:
    for j in v:
        w.write(f"{j['step']} {j['run'].split('/')[-1][:14]} {j['id'][:24]} true {j['true']:.6g} rel {j['rel']:.3f}\n   ...{j['pre'][-60:]!r}[[{j['written']}]]{j['after'][:60]!r}\n")
print('wrong problems', len({s['id'] for s in v}), 'wrong share', len(v) / len(out))
