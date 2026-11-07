"""Model adapter so the same cut and rebuild code runs on LFM3 and E4B (Gemma 4). Pages exclude the screen problems; a fixed share is held out for checks."""
import json, random
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
def load(arch, hf, dev="cuda"):
    if arch == "lfm3":
        tok = AutoTokenizer.from_pretrained(hf, trust_remote_code=True)
        m = AutoModelForCausalLM.from_pretrained(hf, dtype=torch.bfloat16, trust_remote_code=True)
        body = m.model; cap = None
        head = lambda q: tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=True) + "\n"
    else:
        tok = AutoTokenizer.from_pretrained(hf)
        m = AutoModelForCausalLM.from_pretrained(hf, dtype=torch.bfloat16)
        for tower in ("vision_tower", "audio_tower", "embed_vision", "embed_audio"):
            if hasattr(m.model, tower):
                delattr(m.model, tower)
        body = m.model.language_model
        cfg = m.config.get_text_config(); cap = getattr(cfg, "final_logit_softcapping", None)
        head = lambda q: tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True) + "<|channel>thought\n"
    m = m.to(dev).eval().requires_grad_(False)
    return m, tok, body, head, cap
def w2(arch, b):
    return b.feed_forward.w2 if arch == "lfm3" else b.mlp.down_proj
def post(arch, b):
    return b.post_ffn_norm if arch == "lfm3" else b.post_feedforward_layernorm
def pages(arch, stepsf, screen_files, seed=1107, held_share=0.2):
    screen = set()
    for f in screen_files:
        screen |= {json.loads(l)["q"] for l in open(f)}
    byq = {}
    for l in open(stepsf):
        j = json.loads(l)
        text = j["pre"] if arch == "lfm3" else j["thought"]
        if j["q"] in screen or not text:
            continue
        if j["q"] not in byq or len(text) > len(byq[j["q"]][1]):
            byq[j["q"]] = (j["q"], text, j["id"])
    J = sorted(byq.values(), key=lambda x: x[2]); random.Random(seed).shuffle(J)
    nh = max(30, int(len(J) * held_share))
    return J[nh:], J[:nh]
def logits(m, body, cap, ids, pos):
    hs = body(input_ids=ids).last_hidden_state[0][pos].float()
    z = hs @ m.lm_head.weight.float().T
    if cap:
        z = torch.tanh(z / cap) * cap
    return torch.log_softmax(z, -1)
