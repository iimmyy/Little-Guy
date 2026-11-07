#!/usr/bin/env python3
"""How often does he voice doubt right after a step? Text only, no model. Word list fixed before counting."""
import os
import json, re, random, statistics as st
s = open(os.environ["LITTLEGUY_ROOT"] + '/tools/mind/census/statement_jobs.py').read()
head = s[s.index('PLAIN ='):s.index('BADPRE')] + s[s.index('BADPRE'):].split('\n')[0] + '\n'
a = s[s.index('def arith'):]
m = re.search(r'\n(?=\S)', a)
exec(head + a[:m.start()] + '\n')
B = '../../science/final-lfm3-1104/lfm3-base-%s.jsonl'
DOUBT = re.compile(r"\b(wait|actually|hold on|hmm|oops|mistake|wrong|recalc\w*|recompute|re-compute|not accurate|too high|too low|seems off|that's off|is off|seems wrong|no,)", re.I)
KEEP = {2, 3, 4, 5, 6, 7, 10, 11, 12, 13}
rows = []
n = 0
for run in ('g', 's1', 's2', 's3'):
    for l in open(B % run):
        r = json.loads(l)
        if r.get('src') == 'cfe' or not r.get('thinking'):
            continue
        th = r['thinking']
        steps = []
        for x, y in arith(th):
            e = x.start('c') + len(x['c'].rstrip(','))
            lab = y
            if y == 'wrong':
                lab = 'slip' if n in KEEP else 'dropped'
                n += 1
            steps.append((lab, e, th[e:e + 120], th[e:e + 400], r['id'], run))
        rows += [dict(lab=q[0], e=q[1], w120=bool(DOUBT.search(q[2])), w400=bool(DOUBT.search(q[3])), id=q[4], run=q[5]) for q in steps]
sl = [r for r in rows if r['lab'] == 'slip']
rt = [r for r in rows if r['lab'] == 'right']
rate = lambda R, k: sum(x[k] for x in R) / max(1, len(R))
print('slips', len(sl), 'right steps', len(rt))
for k, name in (('w120', 'within 120 characters'), ('w400', 'within 400 characters')):
    print(f"{name}: doubt after slip {rate(sl,k)*100:.0f}% ({sum(x[k] for x in sl)} of {len(sl)}); after right step {rate(rt,k)*100:.1f}%")
    near = []
    for q in sl:
        others = [r for r in rt if r['id'] == q['id'] and r['run'] == q['run'] and abs(r['e'] - q['e']) <= 1500]
        if others:
            near.append((q[k], rate(others, k), len(others)))
    print(f"  same run, right steps within 1500 characters of a slip: {len(near)} slips have neighbours; doubt after slip {sum(a for a,_,_ in near)/max(1,len(near))*100:.0f}%, after neighbours {st.mean(b for _,b,_ in near)*100:.0f}%")
    random.seed(1); ps = []
    for _ in range(20000):
        ps.append(sum(random.random() < rate(rt, k) for _ in sl))
    obs = sum(x[k] for x in sl)
    print(f"  chance of {obs}+ of {len(sl)} at the right-step rate: {sum(v >= obs for v in ps)/len(ps):.4f}")
