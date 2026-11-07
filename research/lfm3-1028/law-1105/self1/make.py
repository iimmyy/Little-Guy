import json,random,secrets
r=random.Random(secrets.randbits(64)); P=[]; L=[]
for i in range(24):
    a,b=r.randint(1000,9999),r.randint(1000,9999)
    d1,d2,d3,d4=[r.choice([-1,1])*r.randint(3,97) for _ in range(4)]
    P.append(dict(i=i,a=a,b=b,d1=d1,d2=d2,d3=d3,d4=d4))
    L+=[(i,"same1",a,b),(i,"same2",b,a),(i,"diff1",a+d1,b),(i,"diff2",a,b+d2),(i,"diff3",a+d3,b+d4)]
r.shuffle(L)
json.dump(dict(problems=P,order=L),open("KEY-do-not-print.json","w"))
for n,(i,t,x,y) in enumerate(L): print(f"{n}: {x} x {y}")
