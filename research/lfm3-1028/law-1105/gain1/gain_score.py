#!/usr/bin/env python3
"""Scores gain_read output with no fitted line: best single candidate and direction chosen on build 2/3 of problems, scored on held-out 1/3. usage: gain_score.py <prefix> <labels-stray.json>"""
import json,sys,hashlib,numpy as np,collections as C
pre,lf=sys.argv[1:3];R=[json.loads(l) for l in open(pre+"-rows.jsonl")];Z=np.load(pre+"-x.npz",allow_pickle=True);Fe,O,names=Z["F"],Z["O"],list(Z["names"]);Lab=json.load(open(lf))
ids=np.array([r["id"] for r in R]);hold=np.array([int(hashlib.md5(i.encode()).hexdigest(),16)%3==2 for i in ids])
lab=[Lab.get(f"{r['run']}|{r['id']}") for r in R];print("rows",len(R),"with labels",sum(x is not None for x in lab),"held out",int(hold.sum()))
X={};N=Fe.shape[1]
for L in range(N):
    for k,n in enumerate(names):
        v=Fe[:,L,k];X[f"L{L}.{n}"]=np.log(np.maximum(v,1e-9)) if n!="align" else v
    X[f"L{L}.cor/amp"]=np.log(np.maximum(Fe[:,L,names.index('cor')],1e-9))-np.log(np.maximum(Fe[:,L,names.index('amp')],1e-9))
    X[f"L{L}.ffn/state"]=np.log(np.maximum(Fe[:,L,0],1e-9))-np.log(np.maximum(Fe[:,L,2],1e-9))
X["entropy"]=O[:,0];X["margin"]=O[:,1]
X["FAMILY.loud.total"]=sum(np.log(np.maximum(Fe[:,L,0],1e-9)) for L in range(N))/N
base={"words":np.log(np.array([r["words"] for r in R])+1.0)}
def auc(s,y):
    o=np.argsort(np.argsort(s))+1.0;p=y.sum();n=len(y)-p
    return (o[y==1].sum()-p*(p+1)/2)/(p*n) if p and n else float("nan")
def catch(s,y,fa=.05):
    t=np.quantile(s[y==0],1-fa);return float((s[y==1]>t).mean())
def pairs(s,y,g):
    w=t=0
    d=C.defaultdict(lambda:([],[]))
    for a,b,c in zip(s,y,g):d[c][int(b)].append(a)
    for c,(n,p) in d.items():
        for a in p:
            for b in n:t+=1;w+=(a>b)+.5*(a==b)
    return (w/t if t else float("nan")),t
rng=np.random.default_rng(0)
LABS={"not done (stray tag) vs done":lambda x:(1 if x["tags"]>0 else 0),"wrong vs right (Codex)":lambda x:{"wrong":1,"right":0}.get(x["grade"])}
if len(sys.argv)>3:
    L2=json.load(open(sys.argv[3]));lab=[L2.get(f"{r['run']}|{r['id']}") for r in R];LABS={"lands off vs with his runs (cross-run labels)":lambda x:{"off":1,"agree":0}.get(x["state"])}
for nm,fn in LABS.items():
    y=np.array([(-1 if (l is None or fn(l) is None) else fn(l)) for l in lab]);use=y>=0;b=use&~hold;h=use&hold
    print(f"\n== {nm}: build {int(b.sum())} ({int(y[b].sum())} positive), held out {int(h.sum())} ({int(y[h].sum())} positive)")
    A=sorted(((auc(v[b],y[b]),k) for k,v in X.items()),key=lambda t:-abs(t[0]-.5))
    print(" best on build:",[(k,round(a,3)) for a,k in A[:6]])
    a,k=A[0];d=1 if a>.5 else -1;s=d*X[k]
    ps,pn=pairs(s[h],y[h],ids[h]);sh=y[h].copy();rng.shuffle(sh)
    B=[]
    pid=np.unique(ids[h])
    for _ in range(300):
        c=rng.choice(pid,len(pid));m=np.concatenate([np.where(ids[h]==i)[0] for i in c]);B.append(auc(s[h][m],y[h][m]))
    print(f" chosen {k} dir {d:+d}: HELD OUT score {auc(s[h],y[h]):.3f} (95% {np.nanquantile(B,.025):.2f}-{np.nanquantile(B,.975):.2f}), catches {catch(s[h],y[h])*100:.0f}% at 5% false alarms, same problem {ps:.3f} ({pn} pairs), shuffled {auc(s[h],sh):.3f}")
    fam=[(f,round(auc(X[f][h],y[h]),3)) for f in("entropy","margin","FAMILY.loud.total")];print(" fixed candidates, held out:",fam)
    print(f" plain fact words, held out {auc(base['words'][h],y[h]):.3f}")
