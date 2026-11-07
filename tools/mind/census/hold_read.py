#!/usr/bin/env python3
"""Reads the hold test: for each arm, what happened to the runs that were wrong and the runs that were right. Machine grade with repaired keys, same as touched_read.py. Run from science/.
usage: hold_read.py <jobs.jsonl> <name>=<arm.jsonl> [...]"""
import json, sys, collections, statistics as st
b = {}; exec(open('grades-v4/sci_grade2.py').read(), b)
K = json.load(open('grades-v4/keys-v2.json'))
rep = lambda i: next((v for k, v in K['repaired'].items() if i.endswith(k)), None)
def gr(r):
    if r.get('finish') != 'stop': return 'cut'
    x = rep(r['id'])
    if x:
        for a in x['answer']:
            q = dict(r); q['answer'] = str(a); q['scale'] = 0
            if b['grade'](q) == 'right': return 'right'
        return 'wrong'
    g = b['grade'](r); return 'wrong' if g == 'sign' else g
J = {(j['run'], j['id']): j for j in map(json.loads, open(sys.argv[1]))}
for spec in sys.argv[2:]:
    name, path = spec.split('=', 1)
    R = [json.loads(l) for l in open(path)]
    print(f"{name}: {len(R)} of {len(J)} runs")
    for start in ('wrong', 'right'):
        S = [r for r in R if J[(r['run'], r['id'])]['f5_grade'] == start]
        if not S: continue
        c = collections.Counter(gr(r) for r in S)
        n = len(S)
        print(f"  was {start:5s} ({n}): right {c['right']/n:.0%}, wrong {c['wrong']/n:.0%}, cut {c['cut']/n:.0%}; added thinking median {st.median(r['added_think_tokens'] for r in S):.0f}, new tokens median {st.median(r['new_tokens'] for r in S):.0f}, total median {st.median(r['gen_tokens'] for r in S):.0f}")
    c = collections.Counter(); 
    for r in R:
        j = J[(r['run'], r['id'])]; c[(j['f5_grade'], gr(r))] += 1
    print('  moves:', dict(c))
