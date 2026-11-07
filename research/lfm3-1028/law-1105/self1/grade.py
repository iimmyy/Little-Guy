import json,itertools,collections as C
K=json.load(open("KEY-do-not-print.json")); O=K["order"]; P={p["i"]:p for p in K["problems"]}
L=[l.strip() for l in open("answers-v1.txt") if l.strip()]
A={}
for n,l in enumerate(L[:71]): A[n]=int(l)
for q,l in zip([72,73,74,75,76,77,78,80,81,82],L[71:81]): A[q]=int(l)
for l in open("more-answers-v1.txt"):
    q,v=l.split()
    if v.isdigit(): A[int(q)]=int(v)
missing=[n for n in range(120) if n not in A]; print("no usable answer:",missing)
R=C.defaultdict(dict)
for n,(i,t,x,y) in enumerate(O):
    if n not in A: continue
    p=P[i]; a,b=p["a"],p["b"]; v=A[n]
    if t=="diff1": v-=p["d1"]*b
    if t=="diff2": v-=p["d2"]*a
    if t=="diff3": v-=p["d3"]*b+p["d4"]*a+p["d3"]*p["d4"]
    R[i][t]=v
right=tot=0; byt=C.Counter(); nbyt=C.Counter()
pairs={"diff":[0,0,0,0],"same":[0,0,0,0]} # pairs, agree, agree&right, both wrong
ww={"diff":[0,0],"same":[0,0]}; est=[];tru=[]; rows=[]
for i,r in R.items():
    t=P[i]["a"]*P[i]["b"]
    for k,v in r.items():
        tot+=1; right+=v==t; byt[k]+=v==t; nbyt[k]+=1
    for (k1,v1),(k2,v2) in itertools.combinations(r.items(),2):
        g="same" if {k1,k2}=={"same1","same2"} else "diff"
        pairs[g][0]+=1
        if v1==v2:
            pairs[g][1]+=1; pairs[g][2]+=v1==t
        if v1!=t and v2!=t:
            ww[g][0]+=1; ww[g][1]+=v1==v2
    cnt=C.Counter(r.values()); top,n=cnt.most_common(1)[0]
    est.append(n/len(r) if n>=2 else 0); tru.append(sum(v==t for v in r.values())/len(r))
    rows.append((i,len(r),n,top==t,sum(v==t for v in r.values())))
print("true accuracy",right,"/",tot,"= %.3f"%(right/tot)); print("by route",{k:f"{byt[k]}/{nbyt[k]}" for k in sorted(nbyt)})
for g in pairs:
    a,b,c,_=pairs[g]; print(g,"pairs",a,"agree",b,"C=%.3f"%(b/a if a else 0),"right when agree %d/%d"%(c,b),"k=%d/%d"%(ww[g][1],ww[g][0]))
print("key-free estimate %.3f vs true %.3f"%(sum(est)/len(est),sum(tru)/len(tru)))
print("problem rows (id, routes, biggest group, group right?, routes right):"); [print(r) for r in rows]
