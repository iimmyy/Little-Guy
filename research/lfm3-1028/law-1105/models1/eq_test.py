import json,sys,os,random,urllib.request,time
from concurrent.futures import ThreadPoolExecutor
url,out=sys.argv[1:3];opt=dict(a.split("=",1) for a in sys.argv[3:]);RUNS=int(opt.get("runs",8));PAR=int(opt.get("par",8));MAX=int(opt.get("maxnew",8000))
Q=[(3,3,6),(4,3,22),(4,4,82),(5,4,160),(5,8,127860)]
EQ="""THE EQUATION
Use this actively while you think, on every step. It is not a final check.

S = 1/2 x m x e^2

- S is surprise: how much this step's result should surprise you.
- m is precision: how tightly independent routes to this same result land together.
- e is error: how far this step's result sits from where those routes land.

The idea behind it: what you come up with has to make sense, and that in itself is the formula. A step makes sense only if it follows from the question and the earlier steps.

How to read it:
- S near 0, routes land together: trust the step and move on. Do not re-check it.
- S high, routes disagree: go back to the earliest step that broke. Keep what came before it, drop what depended on it, turn whatever broke it into a new fact, then continue.
- You have earned a step but have not taken it: take it.
- No way to compare: say so, and find a truly different route.

"""
def question(a,b):
    n=a*b
    return f"Let G be the graph C_{a} x P_{b}: the Cartesian product of a {a}-cycle and a path on {b} vertices. It has {n} labeled vertices, arranged as {b} rings of {a} vertices each, with each vertex joined to the matching vertex in the next ring. How many Hamiltonian cycles does G have? Count each cycle once as a set of edges (no starting point or direction), and treat all {n} vertices as distinct. Give the exact integer.\n\nEnd with a line: ANSWER: <integer>"
jobs=[(arm,a,b,k,s) for arm in("bare","eq") for a,b,k in Q for s in range(1,RUNS+1)]
random.Random(5).shuffle(jobs)
done=set()
if os.path.exists(out):
    for l in open(out):
        r=json.loads(l);done.add((r["arm"],r["a"],r["b"],r["seed"]))
jobs=[j for j in jobs if (j[0],j[1],j[2],j[4]) not in done]
def ask(j):
    arm,a,b,k,s=j;p=question(a,b);p=EQ+"QUESTION\n"+p if arm=="eq" else p;t=time.time()
    body=json.dumps({"messages":[{"role":"user","content":p}],"max_tokens":MAX,"cache_prompt":False,"temperature":1.0,"top_p":.95,"top_k":20,"seed":s,"chat_template_kwargs":{"enable_thinking":True}}).encode()
    r=json.load(urllib.request.urlopen(urllib.request.Request(url+"/v1/chat/completions",body,{"Content-Type":"application/json"}),timeout=7200))
    c=r["choices"][0];m=c["message"]
    return dict(arm=arm,a=a,b=b,key=k,seed=s,thinking=m.get("reasoning_content") or "",reply=m.get("content") or "",finish=c.get("finish_reason"),tokens=r.get("usage",{}).get("completion_tokens"),secs=round(time.time()-t,1))
with open(out,"a") as f,ThreadPoolExecutor(PAR) as P:
    for x in P.map(ask,jobs):
        f.write(json.dumps(x)+"\n");f.flush();print(x["arm"],x["a"],x["b"],x["seed"],x["finish"],x["tokens"],flush=True)
