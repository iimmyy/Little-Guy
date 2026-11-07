#!/usr/bin/env python3
"""Who are the best wrong tokens? Saved rows and step text only, no model."""
import collections, json
import numpy as np
rows=[json.loads(l) for l in open('lfm3-votes-v1-rows.jsonl')]
steps={}
want={r['step'] for r in rows}
for l in open('votes-steps-v1.jsonl'):
    j=json.loads(l)
    if j['step'] in want: steps[j['step']]=j
R=[r for r in rows if abs(r['m'])<=3]
ina=sum(r['a'] in steps[r['step']]['pre'] for r in R); inb=sum(r['b'] in steps[r['step']]['pre'] for r in R)
print('spots |m|<=3',len(R),'right token anywhere earlier in his text',ina,'best wrong token anywhere earlier',inb)
print('token length',collections.Counter(len(r['a']) for r in R))
print('position in number: first token',sum(r['pos']==0 for r in R),'later',sum(r['pos']>0 for r in R))
print('most common best-wrong tokens',collections.Counter(r['b'] for r in R).most_common(14))
print('most common right tokens',collections.Counter(r['a'] for r in R).most_common(14))
print('runs',collections.Counter(r['run'] for r in R),'problems',len({r['id'] for r in R}))
pc=collections.Counter(r['id'] for r in R)
print('top problems by spots',pc.most_common(6))
print('p(right) median',round(float(np.median([r['pa'] for r in R])),3),'p(best wrong) median',round(float(np.median([r['pb'] for r in R])),3),'others',round(float(np.median([1-r['pa']-r['pb'] for r in R])),3))
