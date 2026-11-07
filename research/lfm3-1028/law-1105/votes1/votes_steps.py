#!/usr/bin/env python3
"""Every machine-checked right step in the untouched base runs. No model."""
import os
import collections, hashlib, json, re
s = open(os.environ["LITTLEGUY_ROOT"] + '/tools/mind/census/statement_jobs.py').read()
head = s[s.index('PLAIN ='):s.index('BADPRE')] + s[s.index('BADPRE'):].split('\n')[0] + '\n'
a = s[s.index('def arith'):]
m = re.search(r'\n(?=\S)', a)
exec(head + a[:m.start()] + '\n')
B = '../../science/final-lfm3-1104/lfm3-base-%s.jsonl'
def true_value(x):
    p, q = float(x['a'].replace(',', '')), float(x['b'].replace(',', ''))
    op = x['op']
    return p * q if op in ('\\times', '\\cdot', '×', '*') else p / q if op in ('/', '÷', '\\div') else p + q if op == '+' else p - q
n = 0
c = collections.Counter()
with open('votes-steps-v1.jsonl', 'w') as w:
    for run in ('g', 's1', 's2', 's3'):
        for l in open(B % run):
            r = json.loads(l)
            if r.get('src') == 'cfe' or not r.get('thinking'):
                continue
            th = r['thinking']
            for x, y in arith(th):
                if y != 'right':
                    continue
                d = x['c'].rstrip(',')
                w.write(json.dumps(dict(step=n, id=r['id'], run=run, q=r['q'], split='held' if int(hashlib.sha1(r['id'].encode()).hexdigest(), 16) % 3 == 0 else 'build',
                                        pre=th[:x.start('c')], number=d, true=true_value(x), seen_before=d in th[:x.start()])) + '\n')
                n += 1
                c[run] += 1
print('steps', n, dict(c))
