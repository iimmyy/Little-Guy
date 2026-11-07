# The formula's a* from his own runs, no answer key: per problem, cluster the final answers of every finished run (f5 and untouched). In the biggest group = agrees; off it = off. Tied biggest groups are left out of the agree label.
import json,collections,statistics as st
b={};exec(open('grades-v4/sci_grade2.py').read(),b)
K=json.load(open('grades-v4/keys-v2.json'))
RUNS=('g','s1','s2','s3')
SC=b['SCALES']
rep=lambda i:next((v for k,v in K['repaired'].items() if i.endswith(k)),None); unc=lambda i:any(i.endswith(k) for k in K['unclear'])
def gr(r):
    x=rep(r['id'])
    if x:
        for a in x['answer']:
            q=dict(r); q['answer']=str(a); q['scale']=0
            if b['grade'](q)=='right': return 'right'
        return 'wrong'
    g=b['grade'](r); return 'wrong' if g=='sign' else g
def ans(r):
    p=b['tail'](r.get('reply') or '')
    f=b['forms'](p[0]) if len(p)>1 else []
    if not f: f=b['forms'](p[-1])[-1:]
    return abs(f[0]) if f else None
def same(x,y):
    if not x or not y: return False
    return any(abs(x*s-y)/max(abs(x*s),abs(y))<.02 for s in SC+[1.0])
A=collections.defaultdict(list)
for src,pat in (('f5','final-f5-1104/f5-%s.jsonl'),('base','final-lfm3-1104/lfm3-base-%s.jsonl')):
    for n in RUNS:
        for l in open(pat%n):
            r=json.loads(l);i=r['id']
            if r.get('src')=='cfe' or r.get('finish')!='stop': continue
            a=ans(r)
            if a is None: continue
            A[i].append(dict(src=src,run=n,a=a,r=r))
rows=[];lab={};tight={}
cnt=collections.Counter()
for i,xs in A.items():
    if len(xs)<3: continue
    size=[sum(same(x['a'],y['a']) for y in xs) for x in xs]
    top=max(size); nt=sum(s==top for s in size)
    # is the top unique, i.e. do all runs at the top size agree with each other?
    tops=[x for x,s in zip(xs,size) if s==top]
    uniq=all(same(tops[0]['a'],y['a']) for y in tops)
    tight[i]=top/len(xs)
    for x,s in zip(xs,size):
        if x['src']!='f5': continue
        if top<2 or not uniq: state='tie'
        elif same(x['a'],tops[0]['a']): state='agree'
        else: state='off'
        cnt[state]+=1
        r=x['r']
        if unc(i) or not r.get('thinking'): continue
        lab[f"{x['run']}|{i}"]=dict(state=state,grade=gr(r),tight=tight[i],n=len(xs))
        th=r['thinking'].rstrip()+'\n\n'    # the server strips the paragraph break he writes before closing; the close decision is made there
        pos=[dict(tag='end',c=len(th),at='eol',lab=state),dict(tag='word',c=len(th)-2,at='eol',lab=state)]
        # the paragraph end nearest halfway: can it be heard earlier?
        k=th.find('\n\n',len(th)//2)
        if 0<k<len(th)-22: pos.append(dict(tag='mid',c=k,at='eol',lab=state))
        rows.append(dict(run=x['run'],id=i,src=r['src'],subject=r.get('subject'),group=r.get('group'),q=r['q'],thinking=th,answer=r['answer'],pos=pos))
print(cnt,len(rows),'problems',len(tight))
gc=collections.Counter((v['state'],v['grade']) for v in lab.values());print(gc)
json.dump(lab,open('agree-1104/labels-v1.json','w'))
with open('agree-1104/jobs-v1.jsonl','w') as w:
    for r in rows: w.write(json.dumps(r)+'\n')
