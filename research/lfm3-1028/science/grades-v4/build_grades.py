# grades-v4: grades-v3 (formula-check-1104) + repaired keys + Codex's confirmed corrections + unclear problems marked. Partially audited: unsampled 'right' labels are inherited.
import json,collections,csv,re
V3=json.load(open('formula-check-1104/grades-v3.json'))
K=json.load(open('grades-v4/keys-v2.json')); C=json.load(open('manual-audit-codex-1105/confirmed-grade-corrections.json'))
RUNS=('g','s1','s2','s3')
rep=lambda i:any(i.endswith(k) for k in K['repaired']); unc=lambda i:any(i.endswith(k) for k in K['unclear'])
fin={}
for n in RUNS:
    fin[('lfm3',n)]={r['id']:r['finish'] for r in map(json.loads,open(f'final-lfm3-1104/lfm3-base-{n}.jsonl')) if r.get('src')!='cfe'}
fin[('e4b','g')]={}
for l in open('e4b-mac/e4b-g.jsonl'):
    r=json.loads(l)
    if r.get('src')=='cfe': continue
    fin[('e4b','g')][r['id']]='stop' if (r['finish']=='stop' and '<channel|>' in (r.get('reply') or '')) else 'length'
G={'lfm3':{n:{} for n in RUNS},'e4b':{'g':{}}}
for (m,n),F in fin.items():
    old=V3['lfm3'][n] if m=='lfm3' else V3['e4b']
    for i,f in F.items():
        if f!='stop': g='cut'
        elif unc(i): g='unclear'
        elif rep(i): g='right'           # every finished answer on these six was read: all give the repaired key
        else: g=old[i]
        G[m][n][i]=g
for c in C:
    if not unc(c['id']): G[c['model']][c['run']][c['id']]=c['recommended_answer_grade']
G['e4b']['g'][[i for i in G['e4b']['g'] if i.endswith('a896f8')][0]]='right'   # Claude read: 2.99 K/bar = 29.9 K/MPa, the key; a missed unit step
assert not [g for m in G.values() for d in m.values() for g in d.values() if g not in('right','wrong','cut','unclear')]
json.dump({'_note':'partially audited; see keys-v2.json and manual-audit-codex-1105','lfm3':G['lfm3'],'e4b':G['e4b']['g']},open('grades-v4/grades-v4.json','w'))
json.dump({f'lfm3-base-{n}|{i}':g for n,d in G['lfm3'].items() for i,g in d.items()},open('grades-v4/labels-v4.json','w'))
ids=list(G['e4b']['g']); meta={json.loads(l)['id']:json.loads(l) for l in open('science-build-all-v1.jsonl')}
main=[i for i in ids if not unc(i)]
def tab(lab,I):
    n=len(I); e=collections.Counter(G['e4b']['g'][i] for i in I); gg=collections.Counter(G['lfm3']['g'][i] for i in I); a=collections.Counter(G['lfm3'][k][i] for k in RUNS for i in I)
    on=sum(any(G['lfm3'][k][i]=='right' for k in RUNS) for i in I)
    of=lambda c:c['right']/max(c['right']+c['wrong'],1)
    print(f"{lab:14s} n{n:4d} | E4B R{e['right']/n:4.0%} W{e['wrong']/n:4.0%} C{e['cut']/n:4.0%} fin {of(e):.1%} | LFM3 first R{gg['right']/n:4.0%} W{gg['wrong']/n:4.0%} C{gg['cut']/n:4.0%} fin {of(gg):.1%} | LFM3 four R{a['right']/(4*n):4.1%} W{a['wrong']/(4*n):4.1%} C{a['cut']/(4*n):4.1%} fin {of(a):.1%} ({a['right']}/{a['right']+a['wrong']}) | once {on/n:.0%}")
tab('all',main)
for k in sorted({(meta[i]['src'],meta[i]['subject']) for i in main}): tab(' '.join(k),[i for i in main if (meta[i]['src'],meta[i]['subject'])==k])
once={i for i in main if any(G['lfm3'][k][i]=='right' for k in RUNS)}; ca={i for i in main if all(G['lfm3'][k][i]=='cut' for k in RUNS)}; nv=set(main)-once-ca
print('problems: once',len(once),'cut all 4',len(ca),'finished never right',len(nv),'| total',len(main))
print('E4B on once',collections.Counter(G['e4b']['g'][i] for i in once),'on cut-all-4',collections.Counter(G['e4b']['g'][i] for i in ca),'on never',collections.Counter(G['e4b']['g'][i] for i in nv))
print('LFM3 wrong runs now:',[(n,i[-6:]) for n in RUNS for i in main if G['lfm3'][n][i]=='wrong'])
print('E4B right, LFM3 never',sum(G['e4b']['g'][i]=='right' and i not in once for i in main),'| LFM3 once, E4B not',sum(G['e4b']['g'][i]!='right' and i in once for i in main))
bad=0;tot=0
for r in csv.DictReader(open('manual-audit-codex-1105/manual-grades.csv')):
    if r['src']=='cfe' or r['id'].startswith('cfe'): continue
    g=(G['lfm3'][r['run']] if r['model']=='lfm3' else G['e4b']['g']).get(r['id'])
    want={'correct':'right','incorrect':'wrong'}.get(r['verdict'],'unclear'); tot+=1
    if g!=want: bad+=1; print('  blind differs',r['model'],r['run'],r['id'][-6:],'v4',g,'blind',r['verdict'],'|',r['reason'][:120])
print('blind sample agreement',tot-bad,'of',tot)
