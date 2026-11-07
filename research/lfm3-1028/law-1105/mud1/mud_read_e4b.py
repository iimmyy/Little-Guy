#!/usr/bin/env python3
"""Mud meter, E4B. Reading only. Writes per layer: attention, neurons, per-layer input. Each layer ends with stream *= layer_scalar, so every write is carried forward by the later scalars.
usage: mud_read_e4b.py <hf> <flip-steps.jsonl> <out prefix> [pages=150 maxtok=4000 words=4]"""
import json, sys, time, random
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
args = [a for a in sys.argv[1:] if "=" not in a]; opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
hf, stepsf, prefix = args
PAGES, MAXTOK, WORDS = int(opt.get("pages", 150)), int(opt.get("maxtok", 4000)), int(opt.get("words", 4))
DEV = "cuda"; OPEN = "<|channel>thought\n"
tok = AutoTokenizer.from_pretrained(hf)
m = AutoModelForCausalLM.from_pretrained(hf, dtype=torch.bfloat16)
for tower in ("vision_tower", "audio_tower", "embed_vision", "embed_audio"):
    if hasattr(m.model, tower):
        delattr(m.model, tower)
m = m.to(DEV).eval().requires_grad_(False)
LM = m.model.language_model; LS = LM.layers; N = len(LS)
U = m.lm_head.weight.float(); nrm = LM.norm
c = getattr(nrm, "weight", None); c = c.float() if c is not None else None
eps = getattr(nrm, "eps", 1e-6)
sc = [float(getattr(b, "layer_scalar", torch.tensor(1.0))) for b in LS]
carry = np.ones(N)
for L in range(N - 2, -1, -1):
    carry[L] = carry[L + 1] * sc[L + 1]
carry = [carry[L] * sc[L] for L in range(N)]
print("layer_scalar first, last", sc[0], sc[-1], "carry first", carry[0], "norm has weight", c is not None, flush=True)
IDX, got = {}, {}
def save(name):
    def f(mod, a, o=None):
        v = a[0] if o is None else o
        got[name] = v[0][IDX["i"]].float()
    return f
LS[0].input_layernorm.register_forward_pre_hook(save("x0"))
nrm.register_forward_pre_hook(save("xN"))
for L, b in enumerate(LS):
    b.post_attention_layernorm.register_forward_hook(save(f"op{L}"))
    b.post_feedforward_layernorm.register_forward_hook(save(f"ffn{L}"))
    if getattr(b, "post_per_layer_input_norm", None) is not None:
        b.post_per_layer_input_norm.register_forward_hook(save(f"per{L}"))
has_per = hasattr(LS[0], "post_per_layer_input_norm") and LS[0].post_per_layer_input_norm is not None
order = ["x0"] + [n for L in range(N) for n in ((f"op{L}", f"ffn{L}") + ((f"per{L}",) if has_per else ()))]
scale = torch.tensor([carry[0]] + [carry[L] for L in range(N) for _ in range(3 if has_per else 2)], device=DEV, dtype=torch.float32)[:, None]
J = [json.loads(l) for l in open(stepsf)]
J.sort(key=lambda j: j["id"]); random.Random(1106).shuffle(J)
head = lambda q: tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
rows, VV, done, t0 = [], [], 0, time.time()
for j in J:
    if done >= PAGES:
        break
    h = head(j["q"]) + OPEN
    ids = tok(h + j["thought"].rstrip(), add_special_tokens=False).input_ids
    first = len(tok(h, add_special_tokens=False).input_ids)
    if len(ids) > MAXTOK or len(ids) - first < 40:
        continue
    rng = np.random.default_rng(abs(hash(j["id"])) % (2 ** 31) if False else int.from_bytes(j["id"].encode()[-6:], "little"))
    low = max(first, len(ids) - 1500)
    pos = sorted(rng.choice(np.arange(low, len(ids) - 1), size=WORDS, replace=False).tolist())
    IDX["i"] = torch.tensor(pos, device=DEV)
    try:
        with torch.inference_mode():
            hs = LM(input_ids=torch.tensor([ids], device=DEV)).last_hidden_state[0][IDX["i"]].float()
            logits = hs @ U.T
            W = torch.stack([got[n] for n in order], 1) * scale
            xN = got["xN"]
            r = torch.sqrt(xN.pow(2).mean(-1, keepdim=True) + eps)
            miss = (W.sum(1) - xN).pow(2).sum(-1).sqrt() / xN.pow(2).sum(-1).sqrt()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache(); print("oom skip", j["id"], len(ids), flush=True); continue
    top = logits.topk(2, -1)
    cc = c if c is not None else torch.ones_like(U[0])
    for n, p in enumerate(pos):
        a, b = int(top.indices[n, 0]), int(top.indices[n, 1])
        V = ((W[n] @ (cc * (U[a] - U[b]))) / r[n]).cpu().numpy().astype(np.float32)
        w = W[n, 1:]
        s = float(w.sum(0).pow(2).sum() / w.pow(2).sum(-1).sum())
        cn = torch.nn.functional.cosine_similarity(w[:-1], w[1:], dim=-1).cpu().numpy().astype(np.float32)
        rows.append(dict(step=j["id"], id=j["id"], p=p, top=tok.decode([a]), gap=float(top.values[n, 0] - top.values[n, 1]), S=s, stream_miss=float(miss[n]), sizes=w.pow(2).sum(-1).sqrt().cpu().numpy().round(4).tolist(), cos=cn.round(4).tolist()))
        VV.append(V)
    done += 1
    if done % 25 == 0:
        print(done, "pages", round(time.time() - t0), "s", flush=True)
with open(prefix + "-rows.jsonl", "w") as f:
    for x in rows:
        f.write(json.dumps(x) + "\n")
np.savez(prefix + "-votes.npz", V=np.stack(VV), order=np.array(order), types=np.array([str(x) for x in m.config.text_config.layer_types] if hasattr(m.config, "text_config") else []))
print("MUD_READ_DONE", done, "pages", len(rows), "words", round(time.time() - t0), "s", flush=True)
