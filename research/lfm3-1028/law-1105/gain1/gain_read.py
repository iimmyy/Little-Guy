#!/usr/bin/env python3
"""Reads what each layer computes before the normalizers throw its size away. Reading only. usage: gain_read.py <hf> <jobs.jsonl> <out prefix>"""
import json,sys,time
import numpy as np,torch,torch.nn.functional as F
from transformers import AutoModelForCausalLM,AutoTokenizer
hf,jobs,prefix=sys.argv[1:4];DEV="cuda"
tok=AutoTokenizer.from_pretrained(hf,trust_remote_code=True)
m=AutoModelForCausalLM.from_pretrained(hf,dtype=torch.bfloat16,trust_remote_code=True).to(DEV).eval().requires_grad_(False)
LS=m.model.layers;N=len(LS);got={}
sign={}
for L,b in enumerate(LS):
    ff=b.feed_forward
    c=F.cosine_similarity(ff.w2.weight.float().T,ff.w3.weight.float(),dim=1);sign[L]=(c>0);got[L]={}
    wn=ff.w2.weight.float().norm(dim=0)
    def pre(mod,a,L=L,ff=ff,wn=wn):
        x=a[0][0].float();got[L]["inx"]=x;got[L]["y"]=(F.silu(ff.w1(a[0]).float()[0])*ff.w3(a[0]).float()[0]).abs()*wn
    ff.register_forward_pre_hook(pre)
    b.post_ffn_norm.register_forward_pre_hook(lambda mod,a,L=L:got[L].__setitem__("ffn",a[0][0].float()))
    b.post_operator_norm.register_forward_pre_hook(lambda mod,a,L=L:got[L].__setitem__("op",a[0][0].float()))
    b.ffn_norm.register_forward_pre_hook(lambda mod,a,L=L:got[L].__setitem__("state",a[0][0].float()))
J=[json.loads(l) for l in open(jobs)];FE=[];OUT=[];t0=time.time();skipped=0
NAMES=["ffn_rms","op_rms","state_rms","align","amp","cor"]
rms=lambda v:v.pow(2).mean(-1).sqrt()
with open(prefix+"-rows.jsonl","w") as w:
    for k,j in enumerate(J):
        head=tok.apply_chat_template([{"role":"user","content":j["q"]}],tokenize=False,add_generation_prompt=True,enable_thinking=True)+"\n"
        th=j["thinking"][:max(p["c"] for p in j["pos"])]
        enc=tok(head+th,add_special_tokens=False,return_offsets_mapping=True);ends=np.array([b for a,b in enc.offset_mapping]);n0=len(head)
        idx=[int(np.searchsorted(ends,n0+p["c"],side="right"))-1 for p in j["pos"]]
        try:
            with torch.inference_mode():
                hs=m.model(input_ids=torch.tensor([enc.input_ids],device=DEV)).last_hidden_state[0]
                ii=torch.tensor(idx,device=DEV);lo=m.lm_head(hs[ii]).float();P=lo.log_softmax(-1);ent=-(P.exp()*P).sum(-1)/np.log(2);top=lo.topk(2,dim=-1).values;mar=top[:,0]-top[:,1]
                A=[]
                for L in range(N):
                    g=got[L];f=g["ffn"][ii];s=sign[L];y=g["y"][ii]
                    A.append(torch.stack([rms(f),rms(g["op"][ii]),rms(g["state"][ii]),F.cosine_similarity(f,g["inx"][ii],dim=-1),y[:,s].sum(-1),y[:,~s].sum(-1)],-1))
                FE.append(torch.stack(A,1).cpu().numpy().astype(np.float32));OUT.append(torch.stack([ent,mar],-1).cpu().numpy().astype(np.float32))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache();skipped+=1;print("oom",j["id"],len(enc.input_ids),flush=True);continue
        for p,i in zip(j["pos"],idx):
            w.write(json.dumps({**{x:j[x] for x in("run","id","src","subject","group")},**{a:b for a,b in p.items()},"tokens_before":i,"words":len(th.split())})+"\n")
        if k%100==0:print(k,len(J),round(time.time()-t0),flush=True)
np.savez(prefix+"-x.npz",F=np.concatenate(FE),O=np.concatenate(OUT),names=np.array(NAMES));print("READ_DONE",len(J)-skipped,skipped,round(time.time()-t0),flush=True)
