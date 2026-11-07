#!/usr/bin/env python3
"""How much of his thinking is working numbers out? Fixed rule, saved text, no model."""
import json, random, re
import numpy as np
B = '../../science/final-lfm3-1104/lfm3-base-%s.jsonl'
NUM = re.compile(r'(?<![A-Za-z_])\d[\d,]*\.?\d*')
OP = re.compile(r'[=≈×÷*/+^]|\s-\s|\\times|\\cdot|\\frac')
SPLIT = re.compile(r'(?<=[.?!;:])\s+|\n+')
rng = random.Random(1105)
T = dict(w=0, c=0, nw=0, nc=0)
rows, sam, other = [], [], []
for run in ('g', 's1', 's2', 's3'):
    for l in open(B % run):
        r = json.loads(l)
        if r.get('src') == 'cfe' or not r.get('thinking'):
            continue
        w = c = nw = nc = 0
        for s in SPLIT.split(r['thinking']):
            if not s or not s.strip():
                continue
            a, b = len(s.split()), len(s)
            w += a
            c += b
            if len(NUM.findall(s)) >= 2 and OP.search(s):
                nw += a
                nc += b
                if rng.random() < 0.002:
                    sam.append(s.strip()[:170])
            elif rng.random() < 0.0008:
                other.append(s.strip()[:170])
        T['w'] += w; T['c'] += c; T['nw'] += nw; T['nc'] += nc
        rows.append((nw / max(w, 1), nc / max(c, 1), r.get('finish'), r.get('think_tokens', 0)))
print(f"number work: {T['nw']/T['w']*100:.1f}% of words, {T['nc']/T['c']*100:.1f}% of characters; runs {len(rows)}")
for f in ('stop', 'length'):
    v = [x for x in rows if x[2] == f]
    print(f"finish {f}: runs {len(v)}, median number-work share {np.median([x[0] for x in v])*100:.1f}% of words, {np.median([x[1] for x in v])*100:.1f}% of characters")
with open('numwork-sample-v1.txt', 'w') as h:
    rng.shuffle(sam); rng.shuffle(other)
    h.write('NUMBER WORK\n' + '\n'.join(sam[:20]) + '\n\nNOT NUMBER WORK\n' + '\n'.join(other[:20]) + '\n')
