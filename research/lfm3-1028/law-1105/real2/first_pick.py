#!/usr/bin/env python3
"""Believed mistakes from the first-look scan, rule fixed in GOAL-look2-v1. No model."""
import collections, json, random, re
PLAIN = re.compile(r"\s*(-?\d[\d,]*(?:\.\d+)?)")
F = {}
for s in (0, 1, 2):
    for l in open(f'lfm3-first-{s}.jsonl'):
        r = json.loads(l)
        F[r['step']] = r['first']
J = {}
for l in open('pool-steps-v1.jsonl'):
    j = json.loads(l)
    J[j['step']] = j
print('scanned', len(F), 'of', len(J))
OPSPLIT = re.compile(r"\\times|\\cdot|\\div|×|÷|\*|/|\+|-|−")
T = collections.Counter()
bel, ok = [], []
for s, f in F.items():
    j = J[s]
    if f['num'] is None or f['minp'] is None:
        T['no number'] += 1
        continue
    try:
        x = float(f['num'])
    except ValueError:
        T['no number'] += 1
        continue
    raw = PLAIN.match(f['text']).group(1)
    ops = [float(z.replace(',', '')) for z in re.findall(r"\d[\d,]*(?:\.\d+)?", j['expr'])]
    t = j['true']
    d = raw.replace(',', '')
    ulp = 10 ** (-len(d.split('.')[1])) if '.' in d else 1
    e = abs(x - t)
    if e <= 1.01 * ulp or e / t <= 0.0005:
        T['first look right'] += 1
        ok.append((s, raw, f))
        continue
    if any(abs(x - o) < 1e-9 for o in ops):
        T['operand restart'] += 1
        continue
    if not (e > 2 * ulp and e / t >= 0.005):
        T['near, not counted'] += 1
        continue
    T['first look wrong'] += 1
    if f['minp'] < 0.5:
        T['wrong but unsure (under 0.5)'] += 1
        continue
    T['believed'] += 1
    bel.append((s, raw, f))
print(dict(T))
print(f"first look wrong is {T['first look wrong']/(T['first look wrong']+T['first look right'])*100:.2f}% of first looks that give a result")
same = sum(J[s]['kind'] == 'wrong' for s, _, _ in bel)
print('believed where the saved text was also marked wrong', same, 'problems', len({J[s]['id'] for s, _, _ in bel}))
with open('believed-list-v1.txt', 'w') as w:
    for s, raw, f in bel:
        j = J[s]
        w.write(f"{s} {j['run'].split('/')[-1][:10] if '/' in j['run'] else j['run']} {j['id'][:24]} true {j['true']:.6g} p {f['minp']:.2f} saved {j['written']}\n   ...{j['pre'][-70:]!r}[[{raw}]] rest {f['text'][-14:]!r}\n")
json.dump([(s, raw, f['minp']) for s, raw, f in bel], open('believed-v1.json', 'w'))
rng = random.Random(1110)
rng.shuffle(ok)
json.dump([(s, raw, f['minp']) for s, raw, f in ok[:300]], open('right-controls-v1.json', 'w'))
