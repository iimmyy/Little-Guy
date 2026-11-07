exec(open('../law-1105/bits1/bits.py').read().split('print({k:len(v)')[0])
M={'untouched':load('final-lfm3-1104/lfm3-base-%s.jsonl'),'f5 (Codex-corrected)':load('final-f5-1104/f5-%s.jsonl',True),'f5 (machine)':load('final-f5-1104/f5-%s.jsonl'),'thinking-off (machine)':load('../law-1105/thinkoff-partial/plain-off-%s.jsonl')}
k0=list(M); ids=[i for i in M['thinking-off (machine)'] if len(M['untouched'].get(i,[]))>=2 and len(M['f5 (machine)'].get(i,[]))>=2 and len(M['thinking-off (machine)'][i])>=2]
print("same problems",len(ids))
for k in k0:
    g=[r['g']=='right' for i in ids for r in M[k][i]];print(f"{k}: right among finished {sum(g)}/{len(g)}={sum(g)/len(g):.3f}, mean words {st.mean(r['w'] for i in ids for r in M[k][i]):.0f}")
# all attempts, thinking-off vs f5 vs untouched, right per attempt on problems thinking-off reached
import collections as C
def perattempt(pat,co=False):
    o=C.defaultdict(lambda:[0,0])
    for n in('g','s1','s2','s3'):
        for l in open(pat%n):
            r=json.loads(l);i=r['id']
            if r.get('src')=='cfe':continue
            g='cut' if r.get('finish')!='stop' else (cor.get((n,i),gr(r)) if co else gr(r))
            o[(n,i)]=g
    return o
off=perattempt('../law-1105/thinkoff-partial/plain-off-%s.jsonl');u=perattempt('final-lfm3-1104/lfm3-base-%s.jsonl');f=perattempt('final-f5-1104/f5-%s.jsonl',True)
keys=[k for k in off if k in u and k in f];print("same attempts",len(keys))
for nm,d in (('thinking-off machine',off),('untouched machine',u),('f5 Codex-corrected',f)):
    c=C.Counter(d[k] for k in keys);print(nm,{x:round(c[x]/len(keys),3) for x in ('right','wrong','cut')},dict(c))
# poke: spread by depth
R=[json.loads(l) for l in open('poke-1104/poke-v1.jsonl')]
for s in ('base','f5','math'):
    B=C.defaultdict(lambda:C.defaultdict(list))
    for r in R:
        if r['set']!=s or not r.get('finished'):continue
        a=ans(dict(reply=r['after']))
        if a is None:continue
        w=r['words'];bk=0 if w<200 else 1 if w<500 else 2 if w<1000 else 3
        B[bk][r['id']].append((r['run'],a))
    out={}
    for bk in sorted(B):
        ag=n=0
        for i,x in B[bk].items():
            for (ra,a),(rb,c) in I.combinations(x,2):
                if ra!=rb: n+=1;ag+=same(a,c)
        out[['<200','200-500','500-1000','1000+'][bk]]=(f"{ag}/{n}",round(-math.log2(ag/n),2) if ag else None)
    print("poke",s,"pairs agree, U bits:",out)
