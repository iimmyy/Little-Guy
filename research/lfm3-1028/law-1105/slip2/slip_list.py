#!/usr/bin/env python3
"""Lists every machine-found wrong step in the untouched base runs, in context, for reading by hand."""
import os
import json, re
s = open(os.environ["LITTLEGUY_ROOT"] + '/tools/mind/census/statement_jobs.py').read()
head = s[s.index('PLAIN ='):s.index('BADPRE')] + s[s.index('BADPRE'):].split('\n')[0] + '\n'
a = s[s.index('def arith'):]
m = re.search(r'\n(?=\S)', a)
exec(head + a[:m.start()] + '\n')
B = '../../science/final-lfm3-1104/lfm3-base-%s.jsonl'
def tv(x):
    p, q = float(x['a'].replace(',', '')), float(x['b'].replace(',', ''))
    op = x['op']
    return p * q if op in ('\\times', '\\cdot', '×', '*') else p / q if op in ('/', '÷', '\\div') else p + q if op == '+' else p - q
n = 0
out = open('slip-candidates-v1.jsonl', 'w')
for run in ('g', 's1', 's2', 's3'):
    for l in open(B % run):
        r = json.loads(l)
        if r.get('src') == 'cfe' or not r.get('thinking'):
            continue
        th = r['thinking']
        for x, y in arith(th):
            if y != 'wrong':
                continue
            d = x['c'].rstrip(',')
            s_, e_ = x.start('c'), x.start('c') + len(d)
            print(f"\n=== #{n} {run} {r['id'][:30]} chars {s_} of {len(th)} | {x.group(0)!r} true {tv(x):.6g}")
            print('BEFORE:', th[max(0, x.start() - 260):x.start()].replace('\n', ' / '))
            print('STEP  :', th[x.start():e_])
            print('AFTER :', th[e_:e_ + 230].replace('\n', ' / '))
            out.write(json.dumps(dict(cand=n, run=run, id=r['id'], q=r['q'], pre=th[:s_], post=th[e_:e_ + 500], written=d, true=tv(x), text=x.group(0))) + '\n')
            n += 1
print('\ntotal', n)
