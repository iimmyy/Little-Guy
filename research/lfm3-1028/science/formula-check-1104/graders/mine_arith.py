#!/usr/bin/env python3
"""Mine his own arithmetic from his thinking and check it by computer: 'a op b = c' with plain numbers. No model here."""
import json,re,sys,collections,random
NUM=r"\d[\d,]*(?:\.\d+)?"
OPS=r"\\times|\\cdot|\\div|×|÷|\*|/|\+|-|−"
EQ=re.compile(rf"(?P<a>{NUM})\s*(?P<op>{OPS})\s*(?P<b>{NUM})\s*(?:=|\\approx|≈)\s*(?P<c>-?{NUM})")
BADPRE=set("0123456789.)}]^*/+-−×÷_=")
def num(s): return float(s.replace(',',''))
def check(m,th):
    a,b,c=num(m['a']),num(m['b']),num(m['c']); op=m['op']
    if op in('\\times','\\cdot','×','*'): t=a*b
    elif op in('\\div','÷','/'):
        if b==0: return None
        t=a/b
    elif op=='+': t=a+b
    else: t=a-b
    d=m['c'].replace(',','')
    ulp=10**(-len(d.split('.')[1])) if '.' in d else 1
    e=abs(c-t)
    if e<=1.01*ulp or (t!=0 and e/abs(t)<=0.005): return 'right',t
    if t!=0 and e/abs(t)>=0.02 and e>2*ulp: return 'wrong',t
    return None
def stmts(th):
    for m in EQ.finditer(th):
        i=m.start()-1
        while i>=0 and th[i] in ' \t': i-=1
        if i>=0 and (th[i] in BADPRE or th[i].isalpha() or th[i]=='\\' or th[i]=='{'): continue
        if th[max(0,m.start()-6):m.start()].rstrip().endswith(('\\times','\\cdot','\\frac')): continue
        k=m.end(); rest=th[k:k+6]; r=rest.lstrip(' ')
        if r[:1] and (r[0] in '^/*×÷+-−(\\{[?' or r[:1].isalpha() and len(r)>1 and r[1:2].isalpha() is False and False): 
            if not r.startswith(('\\text','\\mathrm','\\,','\\ ','\\)','\\]','\\\\')): continue
        if r[:1] in ('.',) and r[1:2].isdigit(): continue
        v=check(m,th)
        if v: yield m,v[0],v[1]
if __name__=="__main__":
    C=collections.Counter(); wr=[]; perrun=[]
    for f in sys.argv[1:]:
        for l in open(f):
            t=json.loads(l); th=t.get('thinking') or ''; n=collections.Counter()
            for m,lab,tv in stmts(th):
                C[lab]+=1; n[lab]+=1
                if lab=='wrong': wr.append((t['id'],f.split('-')[-1][:2],m.group(0),tv,th[max(0,m.start()-50):m.end()+40].replace('\n',' ⏎ ')))
            perrun.append(n)
    print(C, 'runs',len(perrun),'runs with a wrong',sum(1 for n in perrun if n['wrong']),'problems with a wrong',len({w[0] for w in wr}))
    random.Random(0).shuffle(wr)
    for w in wr[:45]: print(w[0][-10:],'|',w[2],'| true %.5g |'%w[3],w[4][:150])
