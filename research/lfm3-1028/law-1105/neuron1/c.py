import os,glob,torch
from safetensors import safe_open
hf=os.path.expanduser('~/.littleguy/hf/lfm3-1.7b')
with safe_open(glob.glob(hf+'/*.safetensors')[0],'pt') as s:
    for L in(0,8,16,17,20,23):
        w=s.get_tensor(f'model.layers.{L}.post_ffn_norm.weight').float()
        print(f"L{L} post_ffn_norm rms {float(w.pow(2).mean().sqrt()):.3f}, step if output is even across dimensions {float(w.pow(2).sum().sqrt()):.2f}, spread of weight sd/mean {float(w.std()/w.abs().mean()):.2f}")
