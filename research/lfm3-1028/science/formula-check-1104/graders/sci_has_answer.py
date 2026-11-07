import json,re,sys,collections,statistics as st,random
S=sys.argv[1]
b={};exec(open(S+'/sci_grade2.py').read(),b)
G=json.load(open('formula-check-1104/grades-v3.json'))['lfm3']; ge=json.load(open('formula-check-1104/grades-v3.json'))['e4b']
RUNS=('g','s1','s2','s3')
NUM=b['NUM'];val=b['val']
def sig(a):
    d=a.replace(',','').replace('-','').replace('.','').lstrip('0'); return len(d)
def nums(th):
    th=th.replace('{,}','  ').replace('\\,','  ')
    for m in NUM.finditer(th):
        try: v=val(m)
        except Exception: continue
        yield m.start(),v
def targets(r):
    A=float(r['answer'].replace(',','')); sc=int(r.get('scale') or 0)
    return [abs(A*s) for s in b['SCALES']]+([abs(A*10**sc*s) for s in b['SCALES']] if sc else [])
def first(th,T,tol=.005):
    for p,v in nums(th):
        if any(abs(abs(v)-t)<=tol*t for t in T if t>0): return p
    return None
R=[r for n in RUNS for r in map(lambda l:dict(json.loads(l),_r=n),open(f'final-lfm3-1104/lfm3-base-{n}.jsonl')) if r.get('src')!='cfe' and G[n].get(r['id'])!='drop']
# only keys with 3+ significant digits: small round numbers appear by chance
good=[r for r in R if sig(r['answer'])>=3 and float(r['answer'].replace(',',''))!=0]
cut=[r for r in good if r['finish']!='stop']; ok=[r for r in good if G[r['_r']].get(r['id'])=='right']
rng=random.Random(0)
def stat(lab,rows):
    pos=[]; n=0
    for r in rows:
        th=r.get('thinking') or ''
        p=first(th,targets(r))
        if p is not None: pos.append(p/max(len(th),1)); n+=1
    print(f"{lab}: {len(rows)} runs, key stated in {n} ({n/max(len(rows),1):.0%}); first stated at median {st.median(pos) if pos else 0:.0%} of the run")
stat('cut runs, 3+ digit keys',cut); stat('right finished runs, 3+ digit keys',ok)
# chance control: search each cut run for another problem's key
others=[r for r in good]
hit=0
for r in cut:
    o=rng.choice([x for x in others if x['id']!=r['id']])
    hit+=first(r.get('thinking') or '',targets(o)) is not None
print(f"chance control, another problem's key in a cut run: {hit/len(cut):.0%}")
# problems cut all 4 with 3+ digit key
byp=collections.defaultdict(list)
for r in good: byp[r['id']].append(r)
ca=[i for i,v in byp.items() if all(x['finish']!='stop' for x in v)]
wh=sum(any(first(x.get('thinking') or '',targets(x)) is not None for x in byp[i]) for i in ca)
print(f"problems with 3+ digit keys cut all 4: {len(ca)}; key stated in at least one of the 4: {wh} ({wh/max(len(ca),1):.0%}); E4B right on these: {sum(ge.get(i)=='right' for i in ca)}")
# how long the rest of the run after first key
tail=[];ptail=[]
for r in cut:
    th=r.get('thinking') or ''
    p=first(th,targets(r))
    if p is not None: tail.append(1-p/max(len(th),1))
print(f"cut runs with the key: share of the run spent after it, median {st.median(tail):.0%}")
# repeats and length: how far would finished runs land if more room
fin=[r['gen_tokens'] for r in R if r['finish']=='stop']
print('finished runs tokens: median',st.median(fin),'p90',sorted(fin)[int(.9*len(fin))],'share at 4000-6000',sum(x>=4000 for x in fin)/len(fin))
print('E4B tokens median', st.median(json.loads(l)['gen_tokens'] for l in open('e4b-mac/e4b-g.jsonl')))
print('--- keys not in the question, direct unit only')
def direct(r):
    A=abs(float(r['answer'].replace(',',''))); sc=int(r.get('scale') or 0)
    return [A,A*10**sc] if sc else [A]
def inq(r):
    return first(r['q'],targets(r)) is not None
cut2=[r for r in cut if not inq(r)]
n=sum(first(r.get('thinking') or '',direct(r)) is not None for r in cut2)
ctl=0
for r in cut2:
    o=rng.choice([x for x in good if x['id']!=r['id']])
    ctl+=first(r.get('thinking') or '',direct(o)) is not None
print(f"cut runs, key absent from the question: {len(cut2)}; key stated with no unit change {n} ({n/len(cut2):.0%}); chance control {ctl/len(cut2):.0%}")
rows=[r for r in cut2 if first(r.get('thinking') or '',direct(r)) is not None]
rng.shuffle(rows)
for r in rows[:14]:
    th=r['thinking']; p=first(th,direct(r))
    print(f"[{r['id'][-6:]} {r['_r']} key {r['answer']} at {p/len(th):.0%}] ...{th[max(0,p-110):p+60]!r} ==END== {th[-110:]!r}")
