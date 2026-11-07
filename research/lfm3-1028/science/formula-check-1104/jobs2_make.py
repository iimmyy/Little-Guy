# Second job list after hand-correcting the grades: final statements re-found with the corrected number, plus the end of thinking, which needs no number at all.
import json,re,sys,collections
S=sys.argv[1]
b={};exec(open(S+'/sci_grade2.py').read(),b)
G=json.load(open('formula-check-1104/grades-v3.json'))['lfm3']
NUM=b['NUM']; val=b['val']
def strip(t): return re.sub(r"_\{[^}]*\}|\^\{[^}]*\}(?=\s*\\mathrm)|\\text\{[^}]*\}|\\mathrm\{[^}]*\}|\\rm\s*\w+",' ',t)
def cands(reply):
    k=reply.rfind('\\boxed{'); out=[]
    if k>=0:
        i,d=k+7,1
        while i<len(reply) and d: d+={'{':1,'}':-1}.get(reply[i],0); i+=1
        out+=b['forms'](strip(reply[k+7:i-1]))
    bold=re.findall(r"\*\*([^*]+)\*\*",reply[-300:])
    if bold: out+=b['forms'](strip(bold[-1]))
    out+=b['forms'](strip(reply[-150:]))[-3:]
    return out
def eol(th,c):
    k=th.find('\n',c); return k if k>0 else len(th)
c=collections.Counter(); show=[]
with open('formula-check-1104/jobs2.jsonl','w') as w:
    for n in ('g','s1','s2','s3'):
        for t in map(json.loads,open(f'final-lfm3-1104/lfm3-base-{n}.jsonl')):
            if t.get('src')=='cfe' or t.get('finish')!='stop' or not t.get('thinking'): continue
            lab=G[n].get(t['id'])
            if lab not in('right','wrong'): continue
            th=t['thinking']; pos=[]
            lines=[m for m in re.finditer(r"[^\n]*\d[^\n]*",th)]
            if lines: pos.append(dict(tag='end',c=lines[-1].end(),at='eol',lab=lab))
            A=float(t['answer'].replace(',','')); sc=int(t.get('scale') or 0)
            C=cands(t.get('reply') or '')
            if lab=='right':
                keep=[v for v in C if any(abs(abs(v)-abs(x*s))<=.07*max(abs(x*s),1e-300) for x in (A,A*10**sc) for s in b['SCALES'])]
            else: keep=C
            v=keep[0] if keep else None
            hits=[]
            if v is not None:
                tt=th.replace('{,}','  ').replace('\\,','  ')
                for m in NUM.finditer(tt):
                    try:x=val(m)
                    except: continue
                    if abs(abs(x)-abs(v))<=.005*max(abs(v),1e-300): hits.append(m)
            if hits:
                for wh,m in (('first',hits[0]),('last',hits[-1])):
                    pos.append(dict(tag='final2',where=wh,c=eol(th,m.end()),at='eol',lab=lab,v=v))
            c[lab,'num found' if hits else 'no num']+=1
            if lab=='wrong': show.append((n,t['id'][-6:],t['answer'],v,len(hits),(th[max(0,hits[0].start()-50):hits[0].end()+20].replace('\n',' / ') if hits else '')))
            if pos: w.write(json.dumps(dict(run=f'lfm3-base-{n}',id=t['id'],src=t['src'],subject=t['subject'],group=t.get('group') or t['id'],q=t['q'],thinking=th,pos=pos))+'\n')
print(c)
for s in show: print(s)
