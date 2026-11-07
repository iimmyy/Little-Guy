import os
import sys,numpy as np
sys.path.insert(0,os.path.expanduser("~") + '/.littleguy/src/llama.cpp-lfm3/gguf-py')
from gguf import GGUFReader
r=GGUFReader(os.environ["LITTLEGUY_WEIGHTS"] + '/fresh/e4b-full-bf16.gguf','r')
T={t.name:t for t in r.tensors}
print(sorted(k for k in T if k.startswith('blk.17.'))[:20])
print(T[[k for k in T if k.startswith('blk.17.') and 'ffn' in k][0]].tensor_type, T[[k for k in T if k.startswith('blk.17.') and 'ffn' in k][0]].shape)
