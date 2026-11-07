#!/usr/bin/env python3
"""E4B's own arithmetic from its saved thinking, right or wrong by computer. No model. Plants one wrong digit on right steps for the K + x C read."""
import collections, json, random, re, zlib
PLAIN = r"\d[\d,]*(?:\.\d+)?"
OPS = r"\\times|\\cdot|\\div|×|÷|\*|/|\+|-|−"
EQ = re.compile(rf"(?P<a>{PLAIN})\s*(?P<op>{OPS})\s*(?P<b>{PLAIN})\s*(?P<eq>=|\\approx|≈)\s*(?P<c>{PLAIN})")
BADPRE = set("0123456789.)}]^*/+-−×÷_=")
OPEN = "<|channel>thought\n"
T = collections.Counter()
rng = random.Random(1109)
out = []
for l in open('../../science/e4b-mac/e4b-g.jsonl'):
    r = json.loads(l)
    if r.get('src') == 'cfe':
        continue
    rep = r.get('reply') or ''
    if not rep.startswith(OPEN) or '<channel|>' not in rep:
        T['no_thought'] += 1
        continue
    th = rep[len(OPEN):rep.index('<channel|>')]
    T['runs'] += 1
    T['words'] += len(th.split())
    for m in EQ.finditer(th):
        i = m.start() - 1
        while i >= 0 and th[i] in " \t":
            i -= 1
        if i >= 0 and (th[i] in BADPRE or th[i].isalpha() or th[i] in "\\{√"):
            continue
        if re.search(r"(√|sqrt|\\sqrt\{?|\\pi|π|\\times|\\cdot|\\frac)\s*$", th[max(0, m.start() - 12):m.start()]):
            continue
        rest = th[m.end():m.end() + 14]
        if re.match(r"\s*(?:\.\.\.|…|,\d|,\s*$|[eE][-+]?\d|(?:\\times|×|x|\*|\\cdot)\s*10|thousand|million|billion|k\b|M\b|MPa|kPa|mm|µ|μ|n[mMs]\b|√|\\sqrt|π|\\pi|[\^/*×÷+\-−(])", rest) or m["c"].endswith(","):
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
        kind = "right" if e <= 1.01 * ulp or e / t <= 0.0005 else "wrong" if e > 2 * ulp and e / t >= 0.005 else None
        if kind is None:
            continue
        T[kind] += 1
        j = dict(id=r['id'], q=r['q'], pre=OPEN + th[:m.start("c")], written=m["c"], true=t, expr=th[m.start():m.start("c")], kind=kind, rel=e / t, qmark=rest.lstrip().startswith("?"), split="build" if zlib.crc32(r['id'].encode()) % 3 else "held", after=th[m.end():m.end() + 120])
        out.append(j)
print(dict(T), 'problems with a step', len({j['id'] for j in out}), 'problems wrong', len({j['id'] for j in out if j['kind']=='wrong'}))
for i, j in enumerate(out):
    j['step'] = 200000 + i
planted = []
EXPR = re.compile(r"(?<![\d.,])\d[\d.,]*\s*(?:\*|×|\\times|\\cdot|/|\+|-)\s*\d[\d.,]*\s*=\s*$")
for j in out:
    if j['kind'] != 'right' or not EXPR.search(j['pre']):
        continue
    w = j['written']
    idx = [i for i, ch in enumerate(w) if ch.isdigit()]
    sig = [i for i in idx if w[i] != '0' or any(w[k] not in '0.,' for k in idx if k < i)]
    if len(sig) < 2:
        continue
    pos = sig[0] if len(sig) == 2 else rng.choice(sig[:2])
    alt = rng.choice([x for x in '123456789' if x != w[pos]])
    broken = w[:pos] + alt + w[pos + 1:]
    v = float(broken.replace(',', ''))
    if abs(v - j['true']) / j['true'] < 0.02:
        continue
    planted.append(dict(step=j['step'], id=j['id'], split=j['split'], q=j['q'], pre=j['pre'], clean=w, broken=broken))
print('planted', len(planted), collections.Counter(p['split'] for p in planted), 'problems', len({p['id'] for p in planted}))
with open('e4b-steps-v1.jsonl', 'w') as w:
    for j in out:
        w.write(json.dumps(j) + '\n')
with open('e4b-planted-v1.jsonl', 'w') as w:
    for j in planted:
        w.write(json.dumps(j) + '\n')
with open('e4b-wrong-list-v1.txt', 'w') as w:
    for j in out:
        if j['kind'] == 'wrong':
            w.write(f"{j['step']} {j['id'][:24]} true {j['true']:.6g} rel {j['rel']:.3f}\n   ...{j['pre'][-60:]!r}[[{j['written']}]]{j['after'][:60]!r}\n")
