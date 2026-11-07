import os
import sys,numpy as np,torch
sys.path.insert(0,os.path.expanduser("~") + '/.littleguy/src/llama.cpp-lfm3/gguf-py')
from gguf import GGUFReader
r=GGUFReader(os.environ["LITTLEGUY_WEIGHTS"] + '/fresh/e4b-full-bf16.gguf','r')
T={t.name:t for t in r.tensors}
def g(n):
    t=T[n];x=torch.frombuffer(bytearray(np.asarray(t.data).tobytes()),dtype=torch.int16)
    return x.view(torch.bfloat16).float().reshape(tuple(int(d) for d in t.shape[::-1]))
torch.manual_seed(0)
cos=lambda a,b:torch.nn.functional.cosine_similarity(a,b,dim=1)
N=1+max(int(k.split('.')[1]) for k in T if k.startswith('blk.') and 'ffn_gate' in k);print("layers",N)
for L in range(N):
    up,gt,dn=g(f'blk.{L}.ffn_up.weight'),g(f'blk.{L}.ffn_gate.weight'),g(f'blk.{L}.ffn_down.weight')
    if L==0: print("shapes",up.shape,gt.shape,dn.shape)
    if up.shape[0]<up.shape[1]: up=up.T;gt=gt.T
    if dn.shape[0]>dn.shape[1]: dn=dn.T
    o=dn.T;ol=cos(o,up);gl=cos(gt,up);p=torch.randperm(up.shape[0]);olr=cos(o,up[p])
    print(f"L{L:2d} out-level median {float(ol.median()):+.3f} below -0.1: {float((ol<-.1).float().mean())*100:4.1f}% above +0.1: {float((ol>.1).float().mean())*100:4.1f}% shuffled either side {float((olr.abs()>.1).float().mean())*100:4.1f}% | gate-level median {float(gl.median()):+.3f} q99 {float(torch.quantile(gl,.99)):.2f}",flush=True)
