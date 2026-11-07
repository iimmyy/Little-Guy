import json,random,collections
b={};exec(open('grades-v4/sci_grade2.py').read(),b)
K=json.load(open('grades-v4/keys-v2.json')); V=json.load(open('grades-v4/grades-v4.json'))['lfm3']
RUNS=('g','s1','s2','s3')
rep=lambda i:next((v for k,v in K['repaired'].items() if i.endswith(k)),None); unc=lambda i:any(i.endswith(k) for k in K['unclear'])
def gr(r):
    x=rep(r['id'])
    if x:
        for a in x['answer']:
            q=dict(r); q['answer']=str(a); q['scale']=0
            if b['grade'](q)=='right': return 'right'
        return 'wrong'
    g=b['grade'](r); return 'wrong' if g=='sign' else g
W,R=[],[]
for n in RUNS:
    for l in open(f'final-f5-1104/f5-{n}.jsonl'):
        r=json.loads(l);i=r['id']
        if r.get('src')=='cfe' or unc(i) or i not in V[n] or r.get('finish')!='stop' or not r.get('thinking'): continue
        g=gr(r); r['run']=n; r['f5_grade']=g; r['base_grade']=V[n][i]
        (W if g=='wrong' else R).append(r)
rng=random.Random(5); rng.shuffle(W); rng.shuffle(R)
def pick(L,N):
    c=collections.Counter();o=[]
    for r in L:
        if c[r['id']]<2 and len(o)<N: o.append(r);c[r['id']]+=1
    return o
S=pick(W,120)+pick(R,80)
print(len(W),len(R),len(S),'problems',len({r['id'] for r in S}))
with open('hold-1104/jobs-v1.jsonl','w') as f:
    for r in S: f.write(json.dumps(r)+'\n')
