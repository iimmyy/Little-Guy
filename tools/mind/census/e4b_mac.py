"""E4B 16-bit (QAT master, text only) on the Mac's GPU in ~9 GB instead of 14: the 5.25 GB per-layer word table
(embed_tokens_per_layer, 262144 x 10752) stays in the file on disk and only the rows for the words in use are read.

    from e4b_mac import load
    model, tok = load()          # model on "mps", bf16, eval, no grad
"""
import json, os, struct, warnings

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM, Gemma4TextRotaryEmbedding

def setting(name):
    """A machine setting: the environment first, then littleguy.env at the repo root (copy littleguy.env.example)."""
    if os.environ.get(name):
        return os.environ[name]
    env = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "littleguy.env")
    if os.path.exists(env):
        for line in open(env):
            key, _, value = line.strip().partition("=")
            if key == name and value:
                return value
    return None


def weights(path):
    """A file under LITTLEGUY_WEIGHTS, the folder that holds the model files."""
    root = setting("LITTLEGUY_WEIGHTS")
    if not root:
        raise SystemExit("set LITTLEGUY_WEIGHTS (environment or littleguy.env) to the folder that holds the model files")
    return os.path.join(root, path)


MASTER = os.environ.get("E4B_MASTER") or (weights("masters/e4b-qat") if setting("LITTLEGUY_WEIGHTS") else "")
FILE = MASTER + "/model.safetensors"
PLE = "model.language_model.embed_tokens_per_layer.weight"


class DiskTable(torch.nn.Module):
    """The per-layer word table read straight from the safetensors file (memory-mapped, bf16 as raw uint16)."""

    def __init__(self, scale, dev, file=None):
        super().__init__()
        file = file or FILE
        with open(file, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            meta = json.loads(f.read(n))[PLE]
        shape, start = meta["shape"], 8 + n + meta["data_offsets"][0]
        self.rows = np.memmap(file, dtype=np.uint16, mode="r", offset=start, shape=tuple(shape))
        self.scale, self.dev = scale, dev

    def forward(self, ids):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = torch.from_numpy(np.ascontiguousarray(self.rows[ids.cpu().numpy()])).view(torch.bfloat16)
        return r.to(self.dev) * self.scale


def load(dev="mps", master=None):
    """master: another E4B folder (a changed copy) instead of E4B_MASTER."""
    master = master or MASTER or weights("masters/e4b-qat")
    file = master + "/model.safetensors"
    cfg = AutoConfig.from_pretrained(master).text_config
    with torch.device("meta"):
        model = Gemma4ForCausalLM(cfg).to(torch.bfloat16)                # built as float32 otherwise: 2x memory
    lm = model.model
    lm.embed_tokens_per_layer = DiskTable(lm.embed_tokens_per_layer.scalar_embed_scale, dev, file)
    model.to_empty(device=dev)
    with safe_open(file, "pt") as f:
        own = dict(model.named_parameters()) | dict(model.named_buffers())
        for k in f.keys():
            if not k.startswith("model.language_model.") or k == PLE:
                continue
            name = "model." + k[len("model.language_model."):]
            with torch.no_grad():
                own[name].copy_(f.get_tensor(k).to(own[name].dtype))
    model.lm_head.weight = lm.embed_tokens.weight                      # tied
    lm.embed_tokens.embed_scale = torch.tensor(lm.embed_tokens.scalar_embed_scale, device=dev)
    lm.rotary_emb = Gemma4TextRotaryEmbedding(cfg, device=dev)       # inv_freq buffers are not in the file
    model.eval().requires_grad_(False)
    return model, AutoTokenizer.from_pretrained(master)
