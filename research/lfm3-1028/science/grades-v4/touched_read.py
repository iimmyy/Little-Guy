#!/usr/bin/env python3
"""A touched build against the untouched base on the same science problems, on repaired keys. Machine grade only: its wrongs still get read by hand.
usage: touched_read.py <name> <folder with name-{g,s1,s2,s3}.jsonl> [wrongs=1]"""
import json,sys,collections,re,statistics as st
b={};exec(open('grades-v4/sci_grade2.py').read(),b)
name,fold=sys.argv[1],sys.argv[2]; SHOW=any(a=='wrongs=1' for a in sys.argv)
K=json.load(open('grades-v4/keys-v2.json')); V=json.load(open('grades-v4/grades-v4.json'))['lfm3']
RUNS=('g','s1','s2','s3')
rep=lambda i:next((v for k,v in K['repaired'].items() if i.endswith(k)),None); unc=lambda i:any(i.endswith(k) for k in K['unclear'])
base={n:{r['id']:r for r in map(json.loads,open(f'final-lfm3-1104/lfm3-base-{n}.jsonl')) if r.get('src')!='cfe'} for n in RUNS}
def grade(r):
    i=r['id']
    if r.get('finish')!='stop': return 'cut'
    if unc(i): return 'unclear'
    x=rep(i)
    if x:
        best='wrong'
        for a in x['answer']:
            q=dict(r); q['answer']=str(a); q['scale']=0
            if b['grade'](q)=='right': best='right'
        return best
    g=b['grade'](r)
    return 'wrong' if g=='sign' else g
T={}
for n in RUNS:
    try: T[n]={r['id']:r for r in map(json.loads,open(f'{fold}/{name}-{n}.jsonl')) if r.get('src')!='cfe'}
    except FileNotFoundError: T[n]={}
pairs=[(n,i) for n in RUNS for i in T[n] if i in base[n] and not unc(i)]
gt={p:grade(T[p[0]][p[1]]) for p in pairs}; gb={p:V[p[0]][p[1]] for p in pairs}
c,cb=collections.Counter(gt.values()),collections.Counter(gb.values()); N=len(pairs)
of=lambda d:d['right']/max(d['right']+d['wrong'],1)
print(f"{N} matched runs of {4*264}")
print(f"untouched: right {cb['right']/N:.1%} wrong {cb['wrong']/N:.1%} cut {cb['cut']/N:.1%} right when finished {of(cb):.1%}")
print(f"{name:9s}: right {c['right']/N:.1%} wrong {c['wrong']/N:.1%} cut {c['cut']/N:.1%} right when finished {of(c):.1%} (machine grade, wrongs not yet read)")
w0=sum(base[n][i]['gen_tokens'] for n,i in pairs); w1=sum(T[n][i]['gen_tokens'] for n,i in pairs)
print(f"total tokens {w1/w0-1:+.1%}; median per run {st.median(base[n][i]['gen_tokens'] for n,i in pairs):.0f} -> {st.median(T[n][i]['gen_tokens'] for n,i in pairs):.0f}")
print('moves (untouched -> touched):',dict(collections.Counter((gb[p],gt[p]) for p in pairs)))
if all(len(T[n])>200 for n in RUNS):
    ids=[i for i in T['g'] if not unc(i) and all(i in T[n] for n in RUNS)]
    o0=sum(any(V[n][i]=='right' for n in RUNS) for i in ids); o1=sum(any(grade(T[n][i])=='right' for n in RUNS) for i in ids)
    print(f"right at least once in 4: {o0/len(ids):.1%} -> {o1/len(ids):.1%} on {len(ids)} problems")
if SHOW:
    for p in sorted([p for p in pairs if gt[p]=='wrong'],key=lambda p:p[1]):
        r=T[p[0]][p[1]]; t=re.sub(r'\s+',' ',r.get('reply') or '')
        print(f"{p[0]:2s} {p[1][-6:]} base {gb[p]:5s} K {r['answer']} {re.sub(r'[$~{}]|mathrm|\\\\','',r.get('unit',''))[:9]} tok {r['gen_tokens']}| {t[-140:]}")
