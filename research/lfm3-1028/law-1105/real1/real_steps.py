#!/usr/bin/env python3
"""His own arithmetic in saved runs, right or wrong by computer, including the steps the old parser skipped because a "?" follows. No model."""
import collections, json, re, zlib
PLAIN = r"\d[\d,]*(?:\.\d+)?"
OPS = r"\\times|\\cdot|\\div|×|÷|\*|/|\+|-|−"
EQ = re.compile(rf"(?P<a>{PLAIN})\s*(?P<op>{OPS})\s*(?P<b>{PLAIN})\s*(?P<eq>=|\\approx|≈)\s*(?P<c>{PLAIN})")
BADPRE = set("0123456789.)}]^*/+-−×÷_=")
DOUBT = re.compile(r"\b(wait|actually|hold on|hmm|oops|double.?check|re-?check|verify|recalc\w*|recompute|re-compute|let me check|let's check|let me verify|let's verify|not right|seems off|that's off|seems wrong|no,)", re.I)
B = '../../science/final-lfm3-1104/lfm3-base-%s.jsonl'
T = collections.Counter()
out = []
for run in ('g', 's1', 's2', 's3'):
    for l in open(B % run):
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
            a, b, c = [float(m[g].replace(",", "")) for g in "abc"]
            op = m["op"]
            if op in ("/", "÷", "\\div") and b == 0:
                continue
            t = a * b if op in ("\\times", "\\cdot", "×", "*") else a / b if op in ("/", "÷", "\\div") else a + b if op == "+" else a - b
            d = m["c"].replace(",", "")
            dec = len(d.split(".")[1]) if "." in d else 0
            ulp = 10 ** (-dec)
            e = abs(c - t)
            if t <= 0:
                continue
            if e <= 1.01 * ulp or e / t <= 0.0005:
                kind = "right"
            elif e > 2 * ulp and e / t >= 0.005:
                kind = "wrong"
            else:
                T["between"] += 1
                continue
            after = th[m.end():m.end() + 120]
            q = rest.lstrip().startswith("?")
            exprs = th[m.start():m.start("c")]
            step = dict(id=r['id'], run=run, q=r['q'], pre=th[:m.start("c")], written=m["c"], true=t, expr=exprs, kind=kind, eq=m["eq"], qmark=q, doubt_next=bool(DOUBT.search(after)),
                        rel=e / t, split="build" if zlib.crc32(r['id'].encode()) % 3 else "held", after=after)
            out.append(step)
            T[kind] += 1
            T[f"{kind}:{run}"] += 1
            T[f"{kind}:eq{m['eq'][:1]}"] += 1
            T[f"{kind}:qmark"] += q
            T[f"{kind}:doubt"] += step["doubt_next"]
            T[f"{kind}:{step['split']}"] += 1
for i, s in enumerate(out):
    s["step"] = i
print({k: T[k] for k in sorted(T)})
print(f"wrong share {T['wrong']/(T['wrong']+T['right'])*100:.2f}% of checked steps")
print(f"doubt word in the next 120 chars: wrong {T['wrong:doubt']/T['wrong']*100:.1f}%, right {T['right:doubt']/T['right']*100:.1f}%")
print(f"'?' right after: wrong {T['wrong:qmark']/T['wrong']*100:.1f}%, right {T['right:qmark']/T['right']*100:.1f}%")
for f in ("wrong", "right"):
    v = [s for s in out if s["kind"] == f]
    print(f, "problems", len({s['id'] for s in v}), "greedy steps", sum(s['run']=='g' for s in v))
with open('real-steps-v1.jsonl', 'w') as w:
    for s in out:
        w.write(json.dumps(s) + "\n")
import random
rng = random.Random(1108)
v = [s for s in out if s["kind"] == "wrong"]
rng.shuffle(v)
with open('real-wrong-sample-v1.txt', 'w') as w:
    for s in v[:40]:
        w.write(f"### {s['step']} {s['run']} {s['id']} rel {s['rel']:.3f} true {s['true']:.6g} eq {s['eq']}\n...{s['pre'][-90:]}[[{s['written']}]]{s['after']}\n\n")
