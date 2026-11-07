import sys,glob,re,os,json,torch,collections as C
from safetensors import safe_open
cos=lambda a,b:torch.nn.functional.cosine_similarity(a,b,dim=1)
for d in sys.argv[1:]:
    fs=glob.glob(d+'/*.safetensors');print("\n==",d,len(fs),"files")
    if not fs: continue
    idx={}
    for f in fs:
        with safe_open(f,'pt') as s:
            for k in s.keys():
                if 'vision' in k or 'audio' in k or 'embed_' in k: continue
                m=re.search(r'layers\.(\d+)\.(?:mlp|feed_forward)\.(gate_proj|up_proj|down_proj|w1|w3|w2)\.weight$',k)
                if m: idx[(int(m[1]),m[2])]=(f,k)
    L=sorted({a for a,_ in idx});print("layers",len(L));flip=None
    torch.manual_seed(0);rows=[]
    for a in L:
        def g(n):
            f,k=idx[(a,n)]
            with safe_open(f,'pt') as s:return s.get_tensor(k).float()
        lv=g('up_proj') if (a,'up_proj') in idx else g('w3');out=(g('down_proj') if (a,'down_proj') in idx else g('w2')).T
        ol=cos(out,lv);m=float(ol.median());p=torch.randperm(lv.shape[0]);rows.append((a,m,float((ol<-.1).float().mean()),float((ol>.1).float().mean()),float((cos(out,lv[p]).abs()>.1).float().mean())))
    for a,m,b,c,r in rows[:3]+rows[len(rows)//2:len(rows)//2+2]+rows[-9:]:print(f"L{a:2d} median {m:+.3f} below -0.1 {b*100:4.1f}% above +0.1 {c*100:4.1f}% shuffled {r*100:.1f}%")
    neg=[a for a,m,*_ in rows if m<0];print("layers with negative median",neg, "first at fraction", round(neg[0]/len(L),3) if neg else None)
