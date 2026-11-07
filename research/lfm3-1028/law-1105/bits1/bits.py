import json,math,collections as C,itertools as I,glob,statistics as st
b={};exec(open('grades-v4/sci_grade2.py').read(),b)
K=json.load(open('grades-v4/keys-v2.json'));SC=b['SCALES']
rep=lambda i:next((v for k,v in K['repaired'].items() if i.endswith(k)),None)
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
for x in json.load(open('manual-audit-codex-f5-1105/grades-confirmed-corrections-only.json')):
    if x['model']=='f5':cor[(x['run'],x['id'])]=x['confirmed_corrections_grade']
words=lambda r:len((r.get('thinking') or '').split())+len((r.get('reply') or '').split())
def load(pat,co=False):
    A=C.defaultdict(list)
    for n in('g','s1','s2','s3'):
        for l in open(pat%n):
            r=json.loads(l);i=r['id']
            if r.get('src')=='cfe' or r.get('finish')!='stop':continue
            a=ans(r)
            if a is None:continue
            A[i].append(dict(a=a,g=cor.get((n,i),gr(r)) if co else gr(r),w=words(r)))
    return A
M={'untouched':load('final-lfm3-1104/lfm3-base-%s.jsonl'),'f5':load('final-f5-1104/f5-%s.jsonl',True),'off':load('../law-1105/thinkoff-partial/plain-off-%s.jsonl')}
print({k:len(v) for k,v in M.items()})
print("\nRead 1: surprise of a landing vs right rate, problems with 4 finished runs")
for k in('untouched','f5'):
    t=C.defaultdict(lambda:[0,0])
    for i,x in M[k].items():
        if len(x)!=4:continue
        for r in x:
            s=max(1,sum(same(r['a'],q['a']) for q in x))
            t[s][0]+=r['g']=='right';t[s][1]+=1
    print(k,{f"share {s}/4 J={-math.log2(s/4):.2f} bits":f"{t[s][0]}/{t[s][1]}={t[s][0]/t[s][1]:.2f}" for s in sorted(t,reverse=True)})
def stat(x):
    prs=list(I.combinations(x,2))
    return sum(same(a['a'],c['a']) for a,c in prs),len(prs),st.mean(r['w'] for r in x)
print("\nRead 2: same problems, 2 or more finished runs in all three")
ids=[i for i in M['off'] if all(len(M[k].get(i,[]))>=2 for k in M)]
print("problems",len(ids))
R={}
for k in M:
    a=c=0;w=[]
    for i in ids:
        s,n,m=stat(M[k][i]);a+=s;c+=n;w.append(m)
    Cc=a/c if c else 0;U=-math.log2(Cc) if Cc>0 else float('inf');R[k]=(Cc,U,st.mean(w));print(f"{k}: C {Cc:.3f}, U {U:.3f} bits, mean words {st.mean(w):.0f}")
for k in('untouched','f5'):
    dU=R['off'][1]-R[k][1];dW=R[k][2]-R['off'][2];print(f"r {k}: {dU:.3f} bits / {dW:.0f} words = {1000*dU/dW:.3f} bits per 1000 words")
print("\nAll problems, untouched vs f5 only, 2 or more finished in both")
ids=[i for i in M['f5'] if len(M['f5'][i])>=2 and len(M['untouched'].get(i,[]))>=2];print("problems",len(ids))
for k in('untouched','f5'):
    a=c=0;w=[]
    for i in ids:
        s,n,m=stat(M[k][i]);a+=s;c+=n;w.append(m)
    print(f"{k}: C {a/c:.3f}, U {-math.log2(a/c):.3f} bits, mean words {st.mean(w):.0f}")
