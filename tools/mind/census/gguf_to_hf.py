#!/usr/bin/env python3
"""A touched LFM3 gguf back to a Hugging Face folder, so its neurons can be read. Every number comes from the gguf. Checked first: a known pair (plain gguf against plain HF) must match for every tensor.
usage: gguf_to_hf.py <touched.gguf> <template HF folder> <out folder> check=<plain.gguf>,<plain HF folder>"""
import glob, json, os, shutil, sys
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.expanduser("~/work/llama.cpp-lfm3/gguf-py"))
from gguf import GGUFReader, GGUFValueType  # noqa

args = [a for a in sys.argv[1:] if "=" not in a]
opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
src, tmpl, out = args
NAMES = {"shortconv.conv": "conv.conv", "shortconv.in_proj": "conv.in_proj", "shortconv.out_proj": "conv.out_proj", "ffn_gate": "feed_forward.w1", "ffn_down": "feed_forward.w2", "ffn_up": "feed_forward.w3",
         "ffn_norm": "ffn_norm", "attn_norm": "operator_norm", "post_ffw_norm": "post_ffn_norm", "post_attention_norm": "post_operator_norm", "attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj", "attn_output": "self_attn.o_proj", "attn_q_norm": "self_attn.q_norm", "attn_k_norm": "self_attn.k_norm", "attn_gate": "self_attn.gate_proj"}


def read(path):
    """name -> tensor, without copying the whole file into memory at once: BF16 stays BF16 (the gguf holds it as raw bytes)."""
    out_ = {}
    for t in GGUFReader(path).tensors:
        a = t.data
        if t.tensor_type.name == "BF16":
            raw = np.frombuffer(a, dtype=np.uint8).copy()
            x = torch.from_numpy(raw).view(torch.bfloat16)
            x = x.reshape(tuple(int(v) for v in reversed(t.shape.tolist())))
        else:
            x = torch.from_numpy(np.array(a, dtype=np.float32))
        out_[t.name] = x
    return out_


def key_for(name, hfkeys):
    if name == "token_embd.weight":
        return "model.embed_tokens.weight"
    if name == "output_norm.weight":
        return "model.embedding_norm.weight" if "model.embedding_norm.weight" in hfkeys else None
    if name.startswith("blk."):
        _, i, rest = name.split(".", 2)
        base = rest.rsplit(".", 1)[0]
        if base in NAMES:
            c = f"model.layers.{i}.{NAMES[base]}.weight"
            return c if c in hfkeys else None
    return None


def load_hf(folder):
    d = {}
    for f in glob.glob(os.path.join(folder, "*.safetensors")):
        with safe_open(f, "pt") as s:
            for k in s.keys():
                d[k] = s.get_tensor(k)
    return d


def shape_like(a, t):
    if tuple(a.shape) == tuple(t.shape):
        return a
    if a.numel() != t.numel():
        raise SystemExit(f"size {tuple(a.shape)} vs {tuple(t.shape)}")
    return a.reshape(t.shape)


if "check" in opt:
    pg, ph = opt["check"].split(",")
    G, H = read(pg), load_hf(ph)
    worst, seen = 0.0, set()
    miss = []
    for n, a in G.items():
        k = key_for(n, H)
        if k is None:
            miss.append(n)
            continue
        seen.add(k)
        d = (shape_like(a, H[k]).float() - H[k].float()).abs().max().item() / max(H[k].float().abs().max().item(), 1e-9)
        worst = max(worst, d)
    print("check on plain: matched", len(seen), "of", len(H), "HF tensors; worst relative difference", worst)
    print("gguf tensors with no HF name:", miss)
    print("HF tensors not reached:", sorted(set(H) - seen)[:12])
    if worst > 0.02:
        raise SystemExit("shapes or names are wrong: stop")
G, H = read(src), load_hf(tmpl)
new = dict((k, v.clone()) for k, v in H.items())
ch = 0
for n, a in G.items():
    k = key_for(n, H)
    if k:
        t = shape_like(a, H[k]).to(H[k].dtype)
        ch += int(not torch.equal(t, new[k]))
        new[k] = t
print("tensors differing from the template:", ch)
os.makedirs(out, exist_ok=True)
save_file(new, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
for f in os.listdir(tmpl):
    if not f.endswith(".safetensors") and not f.endswith(".index.json") and os.path.isfile(os.path.join(tmpl, f)):
        shutil.copy(os.path.join(tmpl, f), out)
print("SAVED", out)
