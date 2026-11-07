import os,sys,glob,json,torch,numpy as np
from safetensors import safe_open
hf=os.path.expanduser('~/.littleguy/hf/lfm3-1.7b');T={}
for f in glob.glob(hf+'/*.safetensors'):
    with safe_open(f,'pt') as s:
        for k in s.keys():
            if any(x in k for x in('feed_forward','ffn_norm','post_ffn_norm')): T[k]=s.get_tensor(k).float()
print(len(T),sorted(k for k in T if '.17.' in k))
torch.manual_seed(0);tot=[]
for L in range(24):
    p=f'model.layers.{L}.'
    w1,w2,w3=T[p+'feed_forward.w1.weight'],T[p+'feed_forward.w2.weight'],T[p+'feed_forward.w3.weight'];n1=T.get(p+'ffn_norm.weight');n2=T.get(p+'post_ffn_norm.weight')
    out=w2.T;lv=w3;gt=w1
    cos=lambda a,b:torch.nn.functional.cosine_similarity(a,b,dim=1)
    ol=cos(out,lv);gl=cos(gt,lv);perm=torch.randperm(lv.shape[0]);olr=cos(out,lv[perm]);glr=cos(gt,lv[perm])
    effo=out*n2[None] if n2 is not None else out;effl=lv*n1[None] if n1 is not None else lv;ole=cos(effo,effl);oler=cos(effo,effl[perm])
    q=lambda v:[round(float(torch.quantile(v,x)),3) for x in(.01,.5,.99)]
    a=torch.linalg.norm(lv,dim=1)*torch.linalg.norm(out,dim=1)
    wm=float((ol*a).sum()/a.sum());tot.append((L,float(ol.mean()),float(olr.mean()),float((ol<-.1).float().mean()),float((ol>.1).float().mean()),float((olr.abs()>.1).float().mean()),wm))
    print(f"L{L:2d} out-level q01/50/99 {q(ol)} shuffled {q(olr)} | below -0.1: {float((ol<-.1).float().mean())*100:4.1f}% above +0.1: {float((ol>.1).float().mean())*100:4.1f}% shuffled either side: {float((olr.abs()>.1).float().mean())*100:4.1f}% | strength-weighted mean {wm:+.3f} | effective q50 {float(ole.median()):+.3f} vs {float(oler.median()):+.3f} | gate-level q50/99 {q(gl)[1:]} shuffled q99 {q(glr)[2]}")
