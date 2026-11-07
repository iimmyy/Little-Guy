# Second read: the same runs, but read in the reply at the moment he writes his final value, and at the reply's halfway paragraph. By then the value exists, so an alarm has something to hear.
import json,re
lab=json.load(open('agree-1104/labels-v1.json'))
n=c=0
with open('agree-1104/jobs2-v1.jsonl','w') as w:
    for run in ('g','s1','s2','s3'):
        for l in open(f'final-f5-1104/f5-{run}.jsonl'):
            r=json.loads(l);k=f"{run}|{r['id']}"
            if k not in lab or not (r.get('reply') or '').strip(): continue
            th=r['thinking'].rstrip(); rp=r['reply']; pre=th+'</think>'
            b=rp.rfind('\\boxed{')
            if b>=0:
                i,d=b+7,1
                while i<len(rp) and d: d+={'{':1,'}':-1}.get(rp[i],0); i+=1
                end=i
            else:
                m=list(re.finditer(r'\d[^\n]*',rp[-400:])); end=len(rp)-400+m[-1].end() if m and len(rp)>=400 else len(rp)
                c+=1
            pos=[dict(tag='ans',c=len(pre)+end,at='eol',lab=lab[k]['state'])]
            q=rp.find('\n\n',len(rp)//2)
            if 0<q<end-20: pos.append(dict(tag='rmid',c=len(pre)+q,at='eol',lab=lab[k]['state']))
            w.write(json.dumps(dict(run=run,id=r['id'],src=r['src'],subject=r.get('subject'),group=r.get('group'),q=r['q'],thinking=pre+rp,pos=pos))+'\n'); n+=1
print(n,'runs',c,'without a box')
