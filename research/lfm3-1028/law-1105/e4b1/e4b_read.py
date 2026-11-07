#!/usr/bin/env python3
"""K + x C on E4B, reading only. usage: e4b_read.py <hf> <planted.jsonl> <out.jsonl>"""
import json, re, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
hf, steps, outf = sys.argv[1:4]
DEV, MAXTOK = "cuda", 7000
tok = AutoTokenizer.from_pretrained(hf)
m = AutoModelForCausalLM.from_pretrained(hf, dtype=torch.bfloat16)
for tower in ("vision_tower", "audio_tower", "embed_vision", "embed_audio"):
    if hasattr(m.model, tower):
        delattr(m.model, tower)
m = m.to(DEV).eval().requires_grad_(False)
print(type(m).__name__, "params GB", round(sum(p.numel() for p in m.parameters()) * 2 / 1e9, 2), flush=True)
LM = m.model.language_model
BL = [None]


def hook(mod, inp, out):
    """The side word: blend the written token with another, in the main embedding and in the per-layer table."""
    if BL[0] is None:
        return out
    q, phi, a = BL[0]
    BL[0] = None
    v = mod(torch.tensor([[a]], device=DEV))
    BL[0] = (q, phi, a)
    out = out.clone()
    out[0, q] = (1 - phi) * out[0, q] + phi * v[0, 0]
    return out


LM.embed_tokens.register_forward_hook(hook)
LM.embed_tokens_per_layer.register_forward_hook(hook)
EXPR = re.compile(r"(?<![\d.,])\d[\d.,]*\s*(?:\*|×|\\times|\\cdot|/|\+|-)\s*\d[\d.,]*\s*=\s*$")
ARMS = {"plain": ". ", "so": ". So ", "now": ". Now, ", "wait": ". Wait, ", "hmm": ". Hmm, ", "actually": ". Actually, ", "yes": ". Yes, ", "right": ". That's right. ", "wrong": ". That's wrong. "}
def head(q):
    return tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
print(repr(head("Q")), flush=True)
@torch.inference_mode()
def read(ids, a, b, blend=None):
    BL[0] = blend
    l = m(input_ids=torch.tensor([ids], device=DEV), logits_to_keep=1).logits[0, -1].float()
    BL[0] = None
    return float(l[a] - l[b])
J = [json.loads(l) for l in open(steps)]
t0, n, skipped, checked = time.time(), 0, 0, False
with open(outf, "w") as w:
    for j in J:
        h = head(j["q"])
        written, right = j["broken"], j["clean"]
        expr = EXPR.search(j["pre"]).group()
        W = tok(h + j["pre"] + written, add_special_tokens=False).input_ids
        R = tok(h + j["pre"] + right, add_special_tokens=False).input_ids
        diff = [i for i, (x, y) in enumerate(zip(W, R)) if x != y]
        if len(W) != len(R) or len(diff) != 1 or len(W) > MAXTOK:
            skipped += 1
            continue
        q = diff[0]
        a, b = R[q], W[q]
        L0 = len(tok(h + j["pre"], add_special_tokens=False).input_ids)
        shared = tok.decode(R[L0:q]) if q > L0 else ""
        reads, ok = {}, True
        for name, sep in ARMS.items():
            for page, num, ids0 in (("R", right, R), ("W", written, W)):
                ids = tok(h + j["pre"] + num + sep + expr + shared, add_special_tokens=False).input_ids
                if ids[:len(ids0)] != ids0:
                    ok = False
                    break
                reads[f"{name}:{page}"] = read(ids, a, b)
                if not checked:
                    print("check, blend of phi 0 changes nothing:", reads[f"{name}:{page}"], read(ids, a, b, (q, 0.0, a)), "phi 1 on wrong page equals right page:", flush=True)
                    checked = True
            if not ok:
                break
        if not ok:
            skipped += 1
            continue
        ids = tok(h + j["pre"] + written + ARMS["plain"] + expr + shared, add_special_tokens=False).input_ids
        for phi in (0.25, 0.5, 0.75):
            reads["side:W" if phi == 0.5 else f"phi{phi}:W"] = read(ids, a, b, (q, phi, a))
        reads["phi1check:W"] = read(ids, a, b, (q, 1.0, a))
        reads["m0"] = read(R[:q], a, b)
        w.write(json.dumps(dict(step=j["step"], id=j["id"], split=j["split"], written=written, right=right, a=tok.decode([a]), b=tok.decode([b]), reads=reads)) + "\n")
        w.flush()
        n += 1
        if n % 10 == 0:
            print(n, len(J), "skipped", skipped, round(time.time() - t0), flush=True)
print("E4B_DONE", n, "skipped", skipped, round(time.time() - t0), flush=True)
