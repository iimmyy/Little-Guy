import sys,collections as C
def graph(a,b):
    V=[(i,j) for i in range(a) for j in range(b)];E=[]
    for i,j in V:
        if a>2 or i==0: E.append(((i,j),((i+1)%a,j)))
        if j+1<b: E.append(((i,j),(i,j+1)))
    E=[tuple(sorted(e)) for e in E]; return V,sorted(set(E),key=lambda e:(e[0][1],e[0][0],e[1][1],e[1][0]))
def frontier(V,E):
    last={}
    for k,(u,v) in enumerate(E): last[u]=k;last[v]=k
    st={():1};tot=0;idx={v:i for i,v in enumerate(V)};n=len(V);maxs=0
    S={frozenset():1};total=0
    for k,(u,v) in enumerate(E):
        N=C.Counter()
        for s,w in S.items():
            m=dict(s);mu=m.get(u,u);mv=m.get(v,v)
            def retire(mm,c):
                for x in (u,v):
                    if last[x]==k:
                        if mm.get(x,x)!=None: return None
                        mm.pop(x,None)
                return frozenset(mm.items())
            r=retire(dict(m),None)
            if r is not None:N[r]+=w
            if mu is None or mv is None: continue
            if mu==v:
                # closes a cycle: valid only if every vertex is finished
                rest=[x for x,y in m.items() if x not in(u,v) and y is not None]
                seen=set(x for e in E[:k+1] for x in e)
                if not rest and len(seen)==n and all(m.get(x)is None for x in V if x not in(u,v)): total+=w
                continue
            mm=dict(m);mm[mu]=mv;mm[mv]=mu
            if mu!=u: mm[u]=None
            if mv!=v: mm[v]=None
            r=retire(mm,None)
            if r is not None:N[r]+=w
        S=N;maxs=max(maxs,len(S))
    return total,maxs
def brute(V,E):
    A=C.defaultdict(list)
    for u,v in E:A[u].append(v);A[v].append(u)
    n=len(V);s=V[0];c=0
    def go(x,seen,k):
        nonlocal c
        if k==n:
            c+=s in A[x];return
        for y in A[x]:
            if y not in seen:
                seen.add(y);go(y,seen,k+1);seen.remove(y)
    go(s,{s},1);return c//2
print("route check, frontier vs brute force")
for a,b in[(3,2),(3,3),(4,2),(4,3),(4,4),(5,2),(5,3),(6,3),(4,5)]:
    V,E=graph(a,b);f,_=frontier(V,E);g=brute(V,E);print(a,b,f,g,"OK" if f==g else "BREAK")
print("values")
for a,b in[(4,6),(4,7),(4,8),(4,9),(4,10),(5,6),(5,7),(5,8),(6,5),(6,6),(7,5)]:
    V,E=graph(a,b);f,m=frontier(V,E);print(a,b,len(V),"cycles",f,"max states",m)
