#!/usr/bin/env python3
"""Attention vs the rest on his 10 real slips, shares fitted on the 175 right spots only."""
import json
from math import comb
import numpy as np
D=np.load('lfm3-votes-v1-x.npz'); V=D['V'].astype(float); order=list(D['order'])
rows=[json.loads(l) for l in open('lfm3-votes-v1-rows.jsonl')]
m=np.array([r['m'] for r in rows]); band=np.abs(m)<=3
X=np.c_[np.ones(band.sum()),m[band],m[band]**3]
coef=np.linalg.lstsq(X,V[band],rcond=None)[0]
attn=[i for i,n in enumerate(order) if n.startswith('op') and int(n[2:])>=2 and int(n[2:])%2==0]
conv=[i for i,n in enumerate(order) if n.startswith('op') and i not in attn]
ffn=[i for i,n in enumerate(order) if n.startswith('ffn')]
print('sum of lean over all 175 spots: attention %+.2f, conv %+.2f, neuron %+.2f, word vector %+.2f'%(coef[0][attn].sum(),coef[0][conv].sum(),coef[0][ffn].sum(),coef[0][0]))
print('share of the decision: attention %+.2f, conv %+.2f, neuron %+.2f'%(coef[1][attn].sum(),coef[1][conv].sum(),coef[1][ffn].sum()))
S=np.load('lfm3-votes-slips-v1-x.npz'); Vs=S['V'].astype(float)
sr=[json.loads(l) for l in open('lfm3-votes-slips-v1-rows.jsonl')]
print('step right wrote gap  attention conv neuron (each: votes minus its share of the gap, + toward right)')
neg=0
for i,r in enumerate(sr):
    pred=coef[1]*r['m']+coef[2]*r['m']**3
    res=Vs[i]-pred
    a,c,f=res[attn].sum(),res[conv].sum(),res[ffn].sum()
    neg+=a<0
    print(f"{r['step']:>4} {r['a']:>4} {r['b']:>4} {r['m']:+6.2f}  {a:+6.2f}  {c:+6.2f}  {f:+6.2f}")
p=sum(comb(10,x) for x in range(neg,11))/2**10
print(f'attention below 0 on {neg} of 10, exact one-sided p {p:.3f}; predicted direction')
