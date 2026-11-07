import json,re,sys,collections as C
for fn in sys.argv[1:]:
    R=[json.loads(l) for l in open(fn)];print("\n==",fn,len(R))
    T=C.defaultdict(C.Counter);W=C.defaultdict(list)
    for r in R:
        t=r["thinking"]+"\n"+r["reply"]
        m=re.findall(r"ANSWER:\s*\**\s*\$?\\?(?:boxed\{)?([\d,]+)",t,re.I)
        if r["finish"]!="stop": g="cut"
        elif not m: g="none"
        else: g="right" if int(m[-1].replace(",",""))==r["key"] else "wrong"
        T[r["arm"]][g]+=1;T[(r["arm"],r["a"],r["b"])][g]+=1;W[r["arm"]].append(r["tokens"] or 0)
    for arm in("bare","eq"):
        c=T[arm];n=sum(c.values()) or 1;print(arm,dict(c),"right %.0f%%"%(100*c["right"]/n),"mean tokens %.0f"%(sum(W[arm])/max(len(W[arm]),1)))
        print("  ",{f"{a}x{b}":dict(T[(arm,a,b)]) for a,b in[(3,3),(4,3),(4,4),(5,4),(5,8)]})
