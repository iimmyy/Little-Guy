#!/usr/bin/env python3
"""Rough science grader: right = his final number within 2% of the key (also shows 5%). Final number: last boxed, else the last number in the reply. a x 10^b handled. Rough pointer only: read by hand before trusting."""
import json,glob,re,sys,collections,statistics as st
NUM=re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:\s*(?:\\times|×|x|\\cdot|\*)\s*10\s*\^?\s*\{?\s*(-?\d+)\s*\}?|[eE]([-+]?\d+))?")
def val(m):
    s=m.group(0); b=re.match(r"-?\d[\d,]*(?:\.\d+)?",s).group(0).replace(',','')
    e=m.group(1) or m.group(2); return float(b)*(10**int(e) if e else 1)
def final(reply):
    k=reply.rfind('\\boxed{')
    if k>=0:
        i,d=k+7,1
        while i<len(reply) and d: d+={'{':1,'}':-1}.get(reply[i],0); i+=1
        ms=list(NUM.finditer(reply[k+7:i-1].replace('{,}','').replace('\\,','')))
        if ms: return val(ms[0])
    ms=list(NUM.finditer(reply.replace('{,}','').replace('\\,','')))
    return val(ms[-1]) if ms else None
def grade(r,tol):
    if r.get('finish')!='stop': return 'cut'
    try: A=float(r['answer'].replace(',',''))
    except: return 'skip'
    v=final(r.get('reply') or '')
    if v is None: return 'wrong'
    T=[A,A*10**int(r.get('scale') or 0)]
    return 'right' if any(abs(v-t)<=tol*max(abs(t),1e-300) for t in T) else 'wrong'
rows=[json.loads(l) for f in sorted(sys.argv[1:]) for l in open(f)]
rows=[r for r in rows if r.get('src')!='cfe']
for lab,g in [('all',rows)]+[(k,[r for r in rows if (r['src'],r['subject'])==k]) for k in sorted({(r['src'],r['subject']) for r in rows})]:
    if not g: continue
    c=collections.Counter(grade(r,.02) for r in g); c5=collections.Counter(grade(r,.05) for r in g)
    print(lab,len(g),dict(c),'| right at 5%',c5['right'],f"| right {c['right']/len(g):.0%}, cut {c['cut']/len(g):.0%}, right of finished {c['right']/max(1,len(g)-c['cut']):.0%}")
print('median total tokens',st.median(r.get('gen_tokens',0) for r in rows))
