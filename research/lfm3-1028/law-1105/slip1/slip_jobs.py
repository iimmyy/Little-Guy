import os
import re,json,random,collections as C
s=open(os.environ["LITTLEGUY_ROOT"] + '/tools/mind/census/statement_jobs.py').read()
head=s[s.index('PLAIN ='):s.index('BADPRE')]+s[s.index('BADPRE'):].split('\n')[0]+'\n'
a=s[s.index('def arith'):];m=re.search(r'\n(?=\S)',a);exec(head+a[:m.start()]+'\n')
B='../../science/'
rng=random.Random(11);out=open('slip-jobs-v1.jsonl','w');cnt=C.Counter();show=[]
for model,pat in (('f5',B+'final-f5-1104/f5-%s.jsonl'),('plain',B+'final-lfm3-1104/lfm3-base-%s.jsonl')):
    for run in('g','s1','s2','s3'):
        for l in open(pat%run):
            r=json.loads(l)
            if r.get('src')=='cfe':continue
            th=r.get('thinking') or ''
            if not th:continue
            am=list(arith(th));W=[x for x,y in am if y=='wrong'];Rt=[x for x,y in am if y=='right']
            cnt[(model,run,'statements')]+=len(am);cnt[(model,run,'wrong')]+=len(W)
            rng.shuffle(Rt);keep=Rt[:max(4,5*len(W))] if W else (Rt[:2] if rng.random()<.25 else [])
            pos=[dict(lab='wrong',s=x.start('c'),e=x.end('c'),text=x.group(0)) for x in W]+[dict(lab='right',s=x.start('c'),e=x.end('c'),text=x.group(0)) for x in keep]
            if not pos:continue
            out.write(json.dumps(dict(model=model,run=run,id=r['id'],q=r['q'],thinking=th,pos=pos))+'\n')
            for x in W:show.append((model,run,r['id'][:26],x.group(0)))
print(sorted(cnt.items()));print('slips',len(show))
rng.shuffle(show)
for x in show[:45]:print(x)
