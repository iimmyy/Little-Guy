import json,collections as C,itertools as I
b={};exec(open('grades-v4/sci_grade2.py').read(),b)
K=json.load(open('grades-v4/keys-v2.json')); SC=b['SCALES']
rep=lambda i:next((v for k,v in K['repaired'].items() if i.endswith(k)),None)
unc=lambda i:any(i.endswith(k) for k in K.get('unclear',K.get('uncertain',[])))
def gr(r):
    x=rep(r['id'])
    if x:
        for a in x['answer']:
            q=dict(r);q['answer']=str(a);q['scale']=0
            if b['grade'](q)=='right':return'right'
        return'wrong'
    g=b['grade'](r);return'wrong' if g=='sign' else g
def ans(r):
    p=b['tail'](r.get('reply') or '');f=b['forms'](p[0]) if len(p)>1 else []
    if not f:f=b['forms'](p[-1])[-1:]
    return abs(f[0]) if f else None
same=lambda x,y:bool(x) and bool(y) and any(abs(x*s-y)/max(abs(x*s),abs(y))<.02 for s in SC+[1.0])
cor={}
try:
    for x in json.load(open('manual-audit-codex-f5-1105/grades-confirmed-corrections-only.json')):
        if x['model'].lower().startswith('f5'): cor[(x['run'],x['id'])]=x['confirmed_corrections_grade']
except Exception as e: print("no corrected grades:",e)
for name,pat,usecor in (('f5 machine','final-f5-1104/f5-%s.jsonl',False),('f5 Codex corrected','final-f5-1104/f5-%s.jsonl',True),('untouched machine','final-lfm3-1104/lfm3-base-%s.jsonl',False)):
    A=C.defaultdict(list)
    for n in('g','s1','s2','s3'):
        for l in open(pat%n):
            r=json.loads(l);i=r['id']
            if r.get('src')=='cfe' or r.get('finish')!='stop' or unc(i):continue
            a=ans(r)
            if a is None:continue
            g=cor.get((n,i),gr(r)) if usecor else gr(r)
            A[i].append((a,g))
    P=C.Counter();est=[];tru=[];bysize=C.defaultdict(lambda:[0,0])
    for i,xs in A.items():
        if len(xs)<2:continue
        for (a,g),(c,h) in I.combinations(xs,2):
            rr,ww=g=='right' and h=='right',g!='right' and h!='right'; s=same(a,c)
            P['pairs']+=1;P['agree']+=s;P['agree_right']+=s and g=='right' and h=='right';P['agree_mixed']+=s and (g=='right')!=(h=='right')
            P['rr']+=rr;P['ww']+=ww;P['rr_agree']+=rr and s;P['ww_agree']+=ww and s
        sz=[sum(same(a,c) for c,_ in xs) for a,_ in xs];top=max(sz)
        est.append(top/len(xs) if top>=2 else 0);tru.append(sum(g=='right' for _,g in xs)/len(xs))
        for (a,g),s in zip(xs,sz):
            bysize[s/len(xs)>=.75 if len(xs)>=4 else None][0]+=g=='right';bysize[s/len(xs)>=.75 if len(xs)>=4 else None][1]+=1
    print(f"\n{name}: problems {len(est)}, finished-run pairs {P['pairs']}")
    print(f"  agreement C {P['agree']/P['pairs']:.3f}; right when two runs agree {P['agree_right']}/{P['agree']} = {P['agree_right']/max(P['agree'],1):.3f}; agreed but one graded right one wrong {P['agree_mixed']}")
    print(f"  k (two wrong runs land together) {P['ww_agree']}/{P['ww']} = {P['ww_agree']/max(P['ww'],1):.3f}; two right runs land together {P['rr_agree']}/{P['rr']} = {P['rr_agree']/max(P['rr'],1):.3f}")
    print(f"  key-free estimate {sum(est)/len(est):.3f} vs true right among finished {sum(tru)/len(tru):.3f}")
