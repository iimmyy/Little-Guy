import os
import json, sys, statistics as st
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
T=lambda r:(r.get("think_tokens") or 0)+(r.get("reply_tokens") or 0)
A={};R={}
for a in ["base","base2","base3","base4","f-a12","s-a12","m-a12","m-a12b"]:
    S,M=load(f"screen-{a}-sci.jsonl"),load(f"screen-{a}-math.jsonl")
    g={("s",i):Gs(r) for i,r in S.items()}; g.update({("m",i):Gm(r) for i,r in M.items()})
    w={("s",i):T(r) for i,r in S.items()}; w.update({("m",i):T(r) for i,r in M.items()})
    A[a]=(g,w); R[a]={**{("s",i):r for i,r in S.items()},**{("m",i):r for i,r in M.items()}}
keys=sorted(A["base"][0]); c5=["base","base2","base3","base4"]
mean=st.mean(sum(A[a][1][k] for k in keys) for a in c5)
print("c5 mean right",st.mean(sum(A[a][0][k]=="right" for k in keys) for a in c5),"mean words",round(mean))
cnt={k:sum(A[a][0][k]=="right" for a in c5) for k in keys}
print("c5 right count per problem",sorted(Counter(cnt.values()).items()))
for x in ["f-a12","s-a12","m-a12","m-a12b"]:
    g,w=A[x]
    print(x,"words vs c5 mean %.1f%%"%(100*(sum(w[k] for k in keys)/mean-1)),{n:(sum(cnt[k]==n for k in keys),sum(g[k]=="right" for k in keys if cnt[k]==n)) for n in range(5)})
    fin=[k for k in keys if all(A[a][0][k]!="cut" for a in c5+[x])]
    b=[st.mean(A[a][1][k] for a in c5) for k in fin]; m=[w[k] for k in fin]
    print("  finished in all",len(fin),"words %.1f%%"%(100*(sum(m)/sum(b)-1)),"shorter on",sum(p<q for p,q in zip(m,b)),"median ratio %.3f"%st.median(p/q for p,q in zip(m,b)))
g=A["f-a12"][0]
for k in keys:
    if g[k]=="wrong":
        r=R["f-a12"][k]; print("WRONG",k,"key",r.get("answer"),r.get("scale"),"c5 reads",[A[a][0][k] for a in c5]); print(" reply end:",(r.get("reply") or "")[-500:].replace("\n"," "))
print("lost from always-right:",[(k,g[k]) for k in keys if cnt[k]==4 and g[k]!="right"])
print("gained on never-right:",[k for k in keys if cnt[k]==0 and g[k]=="right"])
