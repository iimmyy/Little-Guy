#!/usr/bin/env python3
"""Does LFM3 run the same on every transformers version? (2026-10-28: on transformers 5.15 Liquid's own config code
renames the conv layers "linear_attention" and warns it does not recognise the per-type rope settings; on 5.21 it
reads them fine.) Prints what the model ended up with (layer types, the rotary frequencies each attention type uses)
and its held-out loss, so two versions can be compared number for number.

usage: lfm3_versions.py <model folder> [dev=cuda nwin=8]
"""
import math, os, sys

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

path = sys.argv[1]
opt = dict(a.split("=", 1) for a in sys.argv[2:])
DEV, NWIN = opt.get("dev", "cuda"), int(opt.get("nwin", 8))
BENCH = os.path.join(os.path.dirname(__file__), "../../../.run/bench")
tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, trust_remote_code=True).to(DEV).eval()
cfg = model.config
print("transformers", transformers.__version__)
print("layer types:", " ".join(t[:4] for t in cfg.layer_types))
print("rope parameters as read:", cfg.rope_parameters)
rot = model.model.rotary_emb
for t in getattr(rot, "layer_types", []):
    f = getattr(rot, f"{t}_inv_freq").float().cpu()
    print(f"  {t}: {int((f > 0).sum())} of {f.numel()} frequencies rotate, fastest {f.max():.4g}, slowest nonzero "
          f"{f[f > 0].min():.4g}")
if not getattr(rot, "layer_types", None):
    print("  rotary module without per-type tables:", {k: tuple(v.shape) for k, v in rot.named_buffers()})
print("attention layers built:", sum(hasattr(l, "self_attn") for l in model.model.layers),
      "conv layers built:", sum(hasattr(l, "conv") for l in model.model.layers))

bos = [tok.bos_token_id]
for name, fn, step in (("wiki", "wiki.test.raw", 511), ("chat", "chat_heldout.txt", 511)):
    text = open(f"{BENCH}/{fn}").read()[:400000]
    ids = tok(text, add_special_tokens=False).input_ids
    nll, n = 0.0, 0
    with torch.no_grad():
        for i in range(NWIN):
            w = torch.tensor([bos + ids[i * step:(i + 1) * step]], device=DEV)
            lg = model(input_ids=w).logits[0, :-1].float()
            nll += torch.nn.functional.cross_entropy(lg, w[0, 1:], reduction="sum").item()
            n += w.shape[1] - 1
    print(f"{name}: loss {nll / n:.4f}  perplexity {math.exp(nll / n):.3f}  ({n} words)")
long = tok(open(f"{BENCH}/wiki.test.raw").read()[:400000], add_special_tokens=False).input_ids[:4095]
with torch.no_grad():
    w = torch.tensor([bos + long], device=DEV)
    lg = model(input_ids=w).logits[0, :-1].float()
    per = torch.nn.functional.cross_entropy(lg, w[0, 1:], reduction="none")
print("one 4096-word window, loss by stretch:", " ".join(f"{per[a:a + 512].mean():.3f}" for a in range(0, 4096, 512)))
print("VERSIONS_DONE")
