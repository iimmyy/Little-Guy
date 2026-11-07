#!/usr/bin/env python3
"""Second science grader, after reading 51 'wrong' answers by hand: most were unit changes or exact forms. Looks at the last box and the last 400 characters of the reply.
Collects every number there, plus \\frac{a}{b}, a/b, \\sqrt, pi forms. right = any of them within 2% of the key, directly, by the key's scale, or by a unit step 10^(3k) (k -4..4) or 10^(+-2) for cm/pm style; angle pi forms to degrees.
sign: a number matching in size but with the opposite sign is 'sign' (read by hand). No number matches: 'wrong'. This gives credit generously, so every 'right' that the first grader called wrong gets sampled by hand."""
import json,re,sys,math
NUM=re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:\s*(?:\\times|×|x|\\cdot|\*)\s*10\s*\^?\s*\{?\s*(-?\d+)\s*\}?|[eE]([-+]?\d+))?")
def val(m):
    s=m.group(0); b=re.match(r"-?\d[\d,]*(?:\.\d+)?",s).group(0).replace(',','')
    e=m.group(1) or m.group(2)
    return float(f"{b}e{e}") if e else float(b)
def forms(t):
    t=t.replace('{,}','').replace('\\,',' ').replace('\\dfrac','\\frac').replace('\\tfrac','\\frac').replace('−','-').replace('\u2011','-').replace('–','-').replace('\u202f',' ')
    out=[]
    for m in NUM.finditer(t):
        try: out.append(val(m))
        except: pass
    for m in re.finditer(r"(-?)\\frac\{\s*(\d+(?:\.\d+)?)\s*\}\{\s*(\d+(?:\.\d+)?)\s*\}",t):
        if float(m[3]): out.append((-1 if m[1] else 1)*float(m[2])/float(m[3]))
    for m in re.finditer(r"(?<![\d.])(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)(?![\d.])",t):
        if float(m[2]): out.append(float(m[1])/float(m[2]))
    for m in re.finditer(r"(?:\\sqrt\{|√\(?|sqrt\()\s*(\d+(?:\.\d+)?)",t): out.append(math.sqrt(float(m[1])))
    x=[]
    if re.search(r"\\pi|π",t):
        for m in re.finditer(r"(-?)(?:(\d+(?:\.\d+)?)\s*)?(?:\\pi|π)(?:\s*/\s*(\d+))?",t):
            a=float(m[2]) if m[2] else 1.0; d=float(m[3]) if m[3] else 1.0
            x.append((-1 if m[1] else 1)*a*math.pi/d)
        for m in re.finditer(r"\\frac\{(\d*)\s*(?:\\pi|π)\}\{(\d+)\}",t):
            x.append((float(m[1]) if m[1] else 1.0)*math.pi/float(m[2]))
    out+=x; out+=[v*180/math.pi for v in x]
    return out
def tail(reply):
    parts=[reply[-400:]]
    k=reply.rfind('\\boxed{')
    if k>=0:
        i,d=k+7,1
        while i<len(reply) and d: d+={'{':1,'}':-1}.get(reply[i],0); i+=1
        parts.insert(0,reply[k+7:i-1])
    return parts
SCALES=[10.0**(3*k) for k in range(-4,5)]+[1e2,1e-2,1e4,1e-4]
def grade(r,tol=.02):
    if r.get('finish')!='stop': return 'cut'
    try: A=float(r['answer'].replace(',',''))
    except: return 'skip'
    reply=r.get('reply') or ''
    box,*rest=tail(reply) if '\\boxed{' in reply else (None,reply[-400:])
    T=[A,A*10**int(r.get('scale') or 0)]
    d=r['answer'].replace(',','').lstrip('-')                    # a key printed with few digits: his number rounds to it, e.g. key 0.11 and 0.107, key 0.8 and 0.783
    ulp=10.0**(-len(d.split('.')[1])) if '.' in d else 1.0
    rnd=max(tol,0.5*ulp/max(abs(A),1e-300)) if abs(A)>0 else tol
    rnd=min(rnd,0.07)                                            # never wider than 7%, so a one-digit key does not swallow everything
    def match(v,signed):
        for t in T:
            for s in SCALES:
                x=t*s
                if abs((v if signed else abs(v))-(x if signed else abs(x)))<=rnd*max(abs(x),1e-300): return True
        return False
    vals=forms(reply[-400:])+(forms(box) if box else [])
    if not reply.strip(): return 'wrong'
    if any(match(v,True) for v in vals): return 'right'
    if any(match(v,False) for v in vals): return 'sign'
    return 'wrong'
