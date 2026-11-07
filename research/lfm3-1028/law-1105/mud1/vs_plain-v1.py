import os
import json, sys
from collections import Counter
ROOT=os.environ["LITTLEGUY_ROOT"] + ""
BASE=f"{ROOT}/.run/results/mind/lfm3-1028/science"
sys.path.insert(0,BASE+"/grades-v4"); sys.path.insert(0,f"{ROOT}/tools/mind/expert")
sys.argv[1:]=[]
from keyed_grade import keyed_grade
from grade import right_reply
def Gs(r):
    x=keyed_grade(r.get("finish"),r.get("reply"),r["id"],r["answer"],r.get("scale")); return "wrong" if x=="sign" else x
def Gm(r):
    return "cut" if r["finish"]!="stop" else ("right" if right_reply(r,r["reply"]) else "wrong")
load=lambda p:{r["id"]:r for r in map(json.loads,open(p))}
T=lambda r:int(r.get("think_tokens") or 0)+int(r.get("reply_tokens") or 0)
A={}
for a in ["plain","base","base2","base3","base4","m-a12","m-a12b","f-a12","s-a12","f5"]:
    S,M=load(f"screen-{a}-sci.jsonl"),load(f"screen-{a}-math.jsonl")
    g={("s",i):Gs(r) for i,r in S.items()}; g.update({("m",i):Gm(r) for i,r in M.items()})
    w={("s",i):T(r) for i,r in S.items()}; w.update({("m",i):T(r) for i,r in M.items()})
    A[a]=(g,w)
keys=sorted(A["plain"][0]); print("problems",len(keys),"missing",{a:len(set(keys)-set(A[a][0])) for a in A if set(keys)-set(A[a][0])})
g0,w0=A["plain"]; W0=sum(w0[k] for k in keys)
for a,(g,w) in A.items():
    t=Counter(g[k] for k in keys); s=Counter(g[k] for k in keys if k[0]=="s"); m=Counter(g[k] for k in keys if k[0]=="m"); W=sum(w[k] for k in keys)
    up=sum(g0[k]!="right" and g[k]=="right" for k in keys); dn=sum(g0[k]=="right" and g[k]!="right" for k in keys)
    print(f"{a:7} {t['right']:>2}/{t['wrong']}/{t['cut']} unclear {t['unclear']} sci {s['right']}/{s['wrong']}/{s['cut']} math {m['right']}/{m['wrong']}/{m['cut']} words {W} {100*(W/W0-1):+.1f}% vs bare gained {up} lost {dn}")
