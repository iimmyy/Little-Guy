#!/usr/bin/env python3
"""E4B greedy vs LFM3 on the same problems (rough grade). E4B's thinking sits in reply: grade only the text after <channel|>."""
import json,re,sys,collections,statistics as st,importlib.util,types
src=open(sys.argv[1]).read().split("rows=[json.loads")[0]
g={}; exec(src,g); grade=g['grade']
def load(f): return [json.loads(l) for l in open(f)]
def clean(r):
    r=dict(r); x=r.get('reply') or ''
    if '<|channel>thought' in x:
        if '<channel|>' in x: r['reply']=x.split('<channel|>')[-1]
        else: r['finish']='length' if r.get('finish')!='stop' else 'nothought'; r['reply']=''
    return r
e4={r['id']:clean(r) for r in load(sys.argv[2])}
lf=collections.defaultdict(list)
for f in sys.argv[3:]:
    for r in load(f): lf[r['id']].append((f,r))
ids=[i for i in e4 if e4[i].get('src')!='cfe' and i in lf]
def tab(lab,keep):
    I=[i for i in ids if keep(e4[i])]
    if not I: return
    ce=collections.Counter(grade(e4[i],.02) for i in I)
    lg=[r for i in I for f,r in lf[i] if 'base-g' in f]
    cl=collections.Counter(grade(r,.02) for r in lg)
    la=[r for i in I for f,r in lf[i]]
    ca=collections.Counter(grade(r,.02) for r in la)
    n=len(I)
    onces=sum(any(grade(r,.02)=='right' for f,r in lf[i]) for i in I)
    print(f"{lab:16s} n {n:3d} | E4B right {ce['right']/n:.0%} wrong {ce['wrong']/n:.0%} cut {ce['cut']/n:.0%} | LFM3 first-try right {cl['right']/max(len(lg),1):.0%} wrong {cl['wrong']/max(len(lg),1):.0%} cut {cl['cut']/max(len(lg),1):.0%} | LFM3 all runs right {ca['right']/len(la):.0%} cut {ca['cut']/len(la):.0%} | LFM3 right once in 4 {onces/n:.0%}")
tab('all',lambda r:True)
for k in sorted({(e4[i]['src'],e4[i]['subject']) for i in ids}):
    tab(f"{k[0]} {k[1]}",lambda r,k=k:(r['src'],r['subject'])==k)
both=[i for i in ids if grade(e4[i],.02)=='right']
print('E4B right:',len(both),'of',len(ids),'; of those LFM3 also right at least once:',sum(any(grade(r,.02)=='right' for f,r in lf[i]) for i in both))
l_once=[i for i in ids if any(grade(r,.02)=='right' for f,r in lf[i])]
print('LFM3 right at least once:',len(l_once),'; of those E4B right first try:',sum(grade(e4[i],.02)=='right' for i in l_once))
cutall=[i for i in ids if all(grade(r,.02)=='cut' for f,r in lf[i])]
print('LFM3 cut all 4:',len(cutall),'; E4B on them:',dict(collections.Counter(grade(e4[i],.02) for i in cutall)))
print('words used, median tokens: E4B',st.median(e4[i]['gen_tokens'] for i in ids),'LFM3',st.median(r['gen_tokens'] for i in ids for f,r in lf[i]))
fin=[r['gen_tokens'] for i in ids for f,r in lf[i] if r.get('finish')=='stop']
print('LFM3 finished runs median tokens',st.median(fin),'share within 3000',sum(x<=3000 for x in fin)/len(fin))
