#!/usr/bin/env python3
"""Why the premise failed: how sure was he when he wrote each slip, against his right steps? Existing reads only, no model."""
import json
import numpy as np
S=[json.loads(l) for l in open('lfm3-slip-v1-rows.jsonl')]
B=[json.loads(l) for l in open('../broken1/lfm3-broken-v1-rows.jsonl')]
sl={r['step']:r['num_surprise'] for r in S if r['arm']=='slip' and r['k']==1}
fx={r['step']:r['num_surprise'] for r in S if r['arm']=='fixed' and r['k']==1}
right=np.array(sorted({r['step']:r['num_surprise'] for r in B if r['arm']=='clean' and r['k']==1 and r['split']=='held'}.values()))
print('right steps',len(right),'median bits',round(float(np.median(right)),2),'share at or above 1 bit',round(float((right>=1).mean()),3),'at or above 2 bits',round(float((right>=2).mean()),3))
print('step  slip bits  chance of the whole number  fixed bits  share of right steps this unsure or more')
for s in sorted(sl):
    print(f"{s:>4}  {sl[s]:>8.1f}  {2**-sl[s]*100:>24.1f}%  {fx[s]:>9.1f}  {(right>=sl[s]).mean()*100:>6.1f}%")
v=np.array(list(sl.values()))
a=np.mean([(x>right).mean()+0.5*(x==right).mean() for x in v])
print('slip vs right step, how often the slip was less expected (pooled):',round(float(a),3))
line=np.percentile(right,95)
print('line at 5% of right steps:',round(float(line),2),'bits; slips over it:',int((v>line).sum()),'of',len(v))
