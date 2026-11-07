#!/usr/bin/env python3
"""Sizes both sides of E from saved thinking text. No model."""
import os
import collections, json, random, re
import numpy as np
s = open(os.environ["LITTLEGUY_ROOT"] + '/tools/mind/census/statement_jobs.py').read()
head = s[s.index('PLAIN ='):s.index('BADPRE')] + s[s.index('BADPRE'):].split('\n')[0] + '\n'
a = s[s.index('def arith'):]
m = re.search(r'\n(?=\S)', a)
exec(head + a[:m.start()] + '\n')
B = '../../science/final-lfm3-1104/lfm3-base-%s.jsonl'
OPS = {'\\times': '*', '\\cdot': '*', '×': '*', '*': '*', '/': '/', '÷': '/', '\\div': '/', '+': '+', '-': '-'}
num = lambda t: float(t.replace(',', ''))
def key(x):
    p, q, o = num(x['a']), num(x['b']), OPS.get(x['op'], x['op'])
    if o in '*+':
        p, q = sorted((p, q))
    return (round(p, 8), o, round(q, 8))
DOUBT = re.compile(r"\b(wait|actually|hold on|hmm|double.?check|re-?check|verify|recalc\w*|recompute|re-compute|let me check|let's check|again)\b", re.I)
tot = collections.Counter()
per_run = []
sample = []
random.seed(1105)
for run in ('g', 's1', 's2', 's3'):
    for l in open(B % run):
        r = json.loads(l)
        if r.get('src') == 'cfe' or not r.get('thinking'):
            continue
        th = r['thinking']
        words = len(th.split())
        seen = {}
        marks = []
        for x, y in arith(th):
            k = key(x)
            c = num(x['c'].rstrip(','))
            rep = k in seen
            tot['steps'] += 1
            tot['wrong'] += y != 'right'
            if rep:
                tot['repeats'] += 1
                old = seen[k]
                same = abs(c - old) <= 0.02 * max(abs(old), 1e-9)
                tot['repeat_agree'] += same
                tot['repeat_conflict'] += not same
            else:
                seen[k] = c
            marks.append((x.start(), rep))
        pos = 0
        redo = doubt = withstep = 0
        for seg in re.split(r'(?<=\n)', th):
            st, en = pos, pos + len(seg)
            pos = en
            w = len(seg.split())
            inside = [rp for p, rp in marks if st <= p < en]
            if inside:
                withstep += w
                if all(inside):
                    redo += w
                    if len(seg) > 40:
                        sample.append((run, r['id'], seg.strip()[:300]))
            if DOUBT.search(seg):
                doubt += w
        tot['words'] += words
        tot['redo_words'] += redo
        tot['doubt_words'] += doubt
        tot['step_words'] += withstep
        per_run.append((words, redo / max(words, 1), doubt / max(words, 1), r.get('finish')))
W = tot['words']
print(f"runs {len(per_run)}, words {W:,}, steps {tot['steps']:,}, machine-flagged wrong {tot['wrong']}")
print(f"E != 0 side: flagged wrong steps are {tot['wrong']/tot['steps']*100:.2f}% of steps (10 real by hand)")
print(f"repeat of a step already on the graph: {tot['repeats']:,} of steps ({tot['repeats']/tot['steps']*100:.1f}%); agrees with the earlier value {tot['repeat_agree']:,}, conflicts {tot['repeat_conflict']:,}")
print(f"E = 0 side: words on lines that only re-do held steps {tot['redo_words']/W*100:.1f}% of all thinking; lines with any step {tot['step_words']/W*100:.1f}%")
print(f"pointer, not the bar: words on lines with a re-check word {tot['doubt_words']/W*100:.1f}%")
for f in ('stop', 'length'):
    v = [x for x in per_run if x[3] == f]
    if v:
        print(f"finish {f}: runs {len(v)}, median words {int(np.median([x[0] for x in v]))}, median re-do share {np.median([x[1] for x in v])*100:.1f}%, median re-check-word share {np.median([x[2] for x in v])*100:.1f}%")
print('lines that only re-do held steps:', len(sample))
random.shuffle(sample)
with open('redo-sample-v1.txt', 'w') as w:
    for run, i, t in sample[:40]:
        w.write(f"[{run} {i}] {t}\n\n")
