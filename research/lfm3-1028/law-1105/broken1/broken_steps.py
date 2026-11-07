#!/usr/bin/env python3
"""Picks machine-checked right steps from his own thinking and writes three arms for each."""
import os
import collections, hashlib, json, random, re

s = open(os.environ["LITTLEGUY_ROOT"] + '/tools/mind/census/statement_jobs.py').read()
head = s[s.index('PLAIN ='):s.index('BADPRE')] + s[s.index('BADPRE'):].split('\n')[0] + '\n'
a = s[s.index('def arith'):]
m = re.search(r'\n(?=\S)', a)
exec(head + a[:m.start()] + '\n')
B = '../../science/final-lfm3-1104/lfm3-base-%s.jsonl'
MAXCHARS = 12000
rng = random.Random(1105)


def true_value(x):
    p, q = float(x['a'].replace(',', '')), float(x['b'].replace(',', ''))
    op = x['op']
    return p * q if op in ('\\times', '\\cdot', '×', '*') else p / q if op in ('/', '÷', '\\div') else p + q if op == '+' else p - q


def break_digit(d, t):
    """All one-digit changes that are clearly wrong by the machine check; returns one at random."""
    clean = d.replace(',', '')
    ulp = 10 ** (-len(clean.split('.')[1])) if '.' in clean else 1
    opts = []
    for i, ch in enumerate(d):
        if not ch.isdigit():
            continue
        first = d.lstrip('-')[0] is ch and i == (1 if d.startswith('-') else 0)
        for r in '0123456789':
            if r == ch or (first and r == '0' and len(clean.lstrip('-')) > 1 and '.' not in clean[:2]):
                continue
            v = float((d[:i] + r + d[i + 1:]).replace(',', ''))
            e = abs(v - t)
            if t != 0 and e / abs(t) >= 0.02 and e > 2 * ulp:
                opts.append(d[:i] + r + d[i + 1:])
    return rng.choice(opts) if opts else None


def reformat(d):
    if ',' in d:
        return d.replace(',', '')
    return d + '0' if '.' in d else d + '.0'


byp = collections.defaultdict(list)
total = collections.Counter()
for run in ('g', 's1', 's2', 's3'):
    for l in open(B % run):
        r = json.loads(l)
        if r.get('src') == 'cfe' or not r.get('thinking'):
            continue
        th = r['thinking']
        for x, y in arith(th):
            total[y] += 1
            if y == 'right' and x.start('c') <= MAXCHARS:
                byp[r['id']].append((run, r, x))
out, n = open('broken-steps-v2.jsonl', 'w'), 0
for pid in sorted(byp):
    cands = byp[pid]
    rng.shuffle(cands)
    took = 0
    for run, r, x in cands:
        if took == 3:
            break
        d = x['c'].rstrip(',')
        t = true_value(x)
        bk = break_digit(d, t)
        if not bk:
            continue
        s_, e_ = x.start('c'), x.start('c') + len(d)
        th = r['thinking']
        out.write(json.dumps(dict(
            step=n, id=pid, run=run, q=r['q'], split='held' if int(hashlib.sha1(pid.encode()).hexdigest(), 16) % 3 == 0 else 'build',
            pre=th[:s_], post=th[e_:e_ + 500], clean=d, broken=bk, reformat=reformat(d), text=x.group(0), true=t,
            seen_before=d in th[:x.start()])) + '\n')
        n += 1
        took += 1
print('all', total, 'problems', len(byp), 'steps written', n)
