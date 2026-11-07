import json, os, sys, statistics as st
from collections import Counter
ROOT=os.environ["LITTLEGUY_ROOT"] + ""
BASE=f"{ROOT}/.run/results/mind/lfm3-1028/science"
sys.path.insert(0,BASE+"/grades-v4"); sys.path.insert(0,f"{ROOT}/tools/mind/expert")
ARGV,sys.argv[1:]=sys.argv[1:],[]
from keyed_grade import keyed_grade
from grade import right_reply
d=f"{ROOT}/.run/results/mind/lfm3-1028/law-1105/mud1"
def Gs(r):
    x=keyed_grade(r.get("finish"),r.get("reply"),r["id"],r["answer"],r.get("scale")); return "wrong" if x=="sign" else x
def Gm(r):
    return "cut" if r["finish"]!="stop" else ("right" if right_reply(r,r["reply"]) else "wrong")
load=lambda p:{r["id"]:r for r in map(json.loads,open(p))}
T=lambda r:(r.get("think_tokens") or 0)+(r.get("reply_tokens") or 0)
A={}
for a in ["base","base2","base3","m-a12","e-a12","l-a12","m-c12","l-c12"]:
    S,M=load(f"{d}/screen-{a}-sci.jsonl"),load(f"{d}/screen-{a}-math.jsonl")
    g={("s",i):Gs(r) for i,r in S.items()}; g.update({("m",i):Gm(r) for i,r in M.items()})
    w={("s",i):T(r) for i,r in S.items()}; w.update({("m",i):T(r) for i,r in M.items()})
    A[a]=(g,w)
keys=sorted(A["base"][0])
c5=["base","base2","base3"]
for a in A:
    g,w=A[a]
    s=Counter(g[k] for k in keys if k[0]=="s"); m=Counter(g[k] for k in keys if k[0]=="m"); t=s+m
    print(f"{a:6} all {t['right']}/{t['wrong']}/{t['cut']} sci {s['right']}/{s['wrong']}/{s['cut']} math {m['right']}/{m['wrong']}/{m['cut']} words {sum(w[k] for k in keys)}")
print('--- pairs between c5 reads and x1.25')
def diff(a,b):
    ga,gb=A[a][0],A[b][0]
    gain=[k for k in keys if ga[k]!="right" and gb[k]=="right"]; lose=[k for k in keys if ga[k]=="right" and gb[k]!="right"]
    return len(gain),len(lose)
for x,y in [("base","base2"),("base","base3"),("base2","base3"),("base","m-a12"),("base2","m-a12"),("base3","m-a12")]:
    print(x,'->',y,'gain %d lose %d'%diff(x,y))
print('--- against all three c5 reads')
cnt={k:sum(A[a][0][k]=="right" for a in c5) for k in keys}
print('c5 right count per problem',Counter(cnt.values()))
ma=A["m-a12"][0]
print('x1.25 right, by how many of 3 c5 reads were right:')
for n in range(4):
    ks=[k for k in keys if cnt[k]==n]
    print(n,len(ks),sum(ma[k]=="right" for k in ks), Counter(ma[k] for k in ks))
print('--- words, finished in all')
fin=[k for k in keys if all(A[a][0][k]!="cut" for a in c5+["m-a12"])]
wb=[st.mean(A[a][1][k] for a in c5) for k in fin]; wm=[A["m-a12"][1][k] for k in fin]
print('problems',len(fin),'c5 mean total',round(sum(wb)),'x1.25 total',sum(wm),'change %.1f%%'%(100*(sum(wm)/sum(wb)-1)))
print('median ratio %.3f'%st.median(m/b for m,b in zip(wm,wb)),'x1.25 shorter on',sum(m<b for m,b in zip(wm,wb)))
print('whole set vs mean of 3 c5: %.1f%%'%(100*(sum(A["m-a12"][1][k] for k in keys)/st.mean(sum(A[a][1][k] for k in keys) for a in c5)-1)))
print('--- finish rate and accuracy among finished')
for a in c5+["m-a12"]:
    g=A[a][0]; f=[k for k in keys if g[k]!="cut"]
    print(a,'finished',len(f),'right of finished %.3f'%(sum(g[k]=="right" for k in f)/len(f)))
