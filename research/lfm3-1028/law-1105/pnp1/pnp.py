import random,itertools as I,collections as C
rng=random.Random(20261105)
def sols_cnf(n,cl): return [x for x in I.product([0,1],repeat=n) if all(any((x[abs(l)-1]==1)==(l>0) for l in c) for c in cl)]
def sols_xor(n,eq): return [x for x in I.product([0,1],repeat=n) if all(sum(x[i] for i in v)%2==b for v,b in eq)]
maj=lambda a,b,c:tuple(int(x+y+z>=2) for x,y,z in zip(a,b,c)); AND=lambda a,b:tuple(x&y for x,y in zip(a,b)); OR=lambda a,b:tuple(x|y for x,y in zip(a,b)); X3=lambda a,b,c:tuple(x^y^z for x,y,z in zip(a,b,c))
def test(S):
    s=set(S); r=C.Counter(); L=list(S)
    if len(L)>40: L=rng.sample(L,40)
    for a,b in I.product(L,L):
        r["and_t"]+=1; r["and_b"]+=AND(a,b) not in s; r["or_t"]+=1; r["or_b"]+=OR(a,b) not in s
    L2=L[:18]
    for a,b,c in I.product(L2,L2,L2):
        r["maj_t"]+=1; r["maj_b"]+=maj(a,b,c) not in s; r["x3_t"]+=1; r["x3_b"]+=X3(a,b,c) not in s
    return r
def rep(name,G):
    t=C.Counter(); n=0
    for S in G:
        if len(S)<3: continue
        t+=test(S); n+=1
    print(f"{name:26s} inst {n:3d} | break rate AND {t['and_b']/max(t['and_t'],1):.3f} OR {t['or_b']/max(t['or_t'],1):.3f} MAJ {t['maj_b']/max(t['maj_t'],1):.3f} XOR3 {t['x3_b']/max(t['x3_t'],1):.3f}")
def rc(n,k,m,horn=False):
    o=[]
    for _ in range(m):
        v=rng.sample(range(1,n+1),k); c=[x*rng.choice([-1,1]) for x in v]
        if horn:
            c=[-abs(x) for x in c]
            if rng.random()<.6: c[0]=abs(c[0])
        o.append(c)
    return o
N=12
rep("2-SAT",[sols_cnf(N,rc(N,2,10)) for _ in range(30)])
rep("Horn",[sols_cnf(N,rc(N,3,14,True)) for _ in range(30)])
rep("XOR-SAT",[sols_xor(N,[(rng.sample(range(N),3),rng.randint(0,1)) for _ in range(5)]) for _ in range(30)])
rep("3-SAT",[sols_cnf(N,rc(N,3,30)) for _ in range(30)])
def edges(n,p):
    E=[(i,j) for i,j in I.combinations(range(n),2) if rng.random()<p]; return E
def hampaths(n,E):
    idx={e:i for i,e in enumerate(E)}; A=C.defaultdict(list)
    for i,j in E: A[i].append(j);A[j].append(i)
    out=set()
    def go(p,seen):
        if len(p)==n:
            v=[0]*len(E)
            for a,b in zip(p,p[1:]): v[idx[(min(a,b),max(a,b))]]=1
            out.add(tuple(v)); return
        if len(out)>400: return
        for w in A[p[-1]]:
            if w not in seen: go(p+[w],seen|{w})
    for s in range(n): go([s],{s})
    return list(out)
def matchings(n,E):
    idx={e:i for i,e in enumerate(E)}; out=[]
    def go(free,cur):
        if len(out)>400: return
        if not free:
            v=[0]*len(E)
            for e in cur: v[idx[e]]=1
            out.append(tuple(v)); return
        a=free[0]
        for b in free[1:]:
            if (a,b) in idx: go([x for x in free[1:] if x!=b],cur+[(a,b)])
    go(list(range(n)),[]); return out
rep("Hamiltonian path",[hampaths(9,edges(9,.55)) for _ in range(30)])
rep("Perfect matching (in P)",[matchings(10,edges(10,.6)) for _ in range(30)])
# no-backtrack test
def is_conn_after(adj,u,v):
    # is v still reachable from u after removing edge u-v?
    adj[u].remove(v);adj[v].remove(u); st=[u];seen={u}
    while st:
        x=st.pop()
        for y in adj[x]:
            if y not in seen: seen.add(y);st.append(y)
    adj[u].append(v);adj[v].append(u); return v in seen
def rand_euler(n):
    while True:
        adj={i:[] for i in range(n)}
        for _ in range(n):  # union of random cycles, keeps all degrees even
            k=rng.randint(3,n); c=rng.sample(range(n),k)
            for a,b in zip(c,c[1:]+c[:1]):
                adj[a].append(b);adj[b].append(a)
        st=[0];seen={0}
        while st:
            x=st.pop()
            for y in adj[x]:
                if y not in seen: seen.add(y);st.append(y)
        if all(i in seen or not adj[i] for i in adj): return adj
def fleury(adj):
    adj={k:list(v) for k,v in adj.items()}; m=sum(len(v) for v in adj.values())//2; u=0; used=0
    while True:
        if not adj[u]: return used==m
        ch=list(adj[u]); rng.shuffle(ch); pick=None
        for v in ch:
            if len(adj[u])==1 or is_conn_after(adj,u,v): pick=v;break
        if pick is None: pick=ch[0]
        adj[u].remove(pick);adj[pick].remove(u);u=pick;used+=1
def cubic(n):
    while True:
        pts=[i for i in range(n) for _ in range(3)]; rng.shuffle(pts); E=set();ok=True
        for a,b in zip(pts[::2],pts[1::2]):
            e=(min(a,b),max(a,b))
            if a==b or e in E: ok=False;break
            E.add(e)
        if ok: return E
def greedy_ham(n,E):
    A=C.defaultdict(set)
    for a,b in E: A[a].add(b);A[b].add(a)
    u=rng.randrange(n);seen={u};p=1
    while True:
        c=[w for w in A[u] if w not in seen]
        if not c: return p==n
        # Warnsdorff style: go to the neighbor with fewest free neighbors
        u=min(c,key=lambda w:(len([x for x in A[w] if x not in seen]),rng.random())); seen.add(u);p+=1
print("Fleury no-backtrack Euler success:",[ (n,sum(fleury(rand_euler(n)) for _ in range(200))/200) for n in (10,20,40)])
print("Greedy no-backtrack Hamiltonian success, 3-regular:",[(n,sum(greedy_ham(n,cubic(n)) for _ in range(300))/300) for n in (10,20,40,80,160)])
