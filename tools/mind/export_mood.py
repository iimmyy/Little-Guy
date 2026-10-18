#!/usr/bin/env python3
"""Export a mood for the Mac's llama.cpp, plus a reference sheet the Mac has to match.

In Gemma 4, llama.cpp adds its control vector right after each room's layer_scalar, the same spot mood_dose.py's
hook adds the drip. So a drip of <steps> in rooms A-B = a control vector with direction.L = spread[L] * V[L] for
L in A..B, loaded with --control-vector-scaled <file>:<steps>. The switched-off neurons get baked into a copy of the
brain file on the Mac (mac_mood.py bake), and the word ladder becomes llama.cpp's logit bias.
Writes, in <out dir>:
  <trait>_unit.npy   [42, 2560] float32, one natural step of the mood per room (same spread as mood_dose.py)
  neurons_off.json   {room: [neuron, ...]} from SCRIPT_OFF
  reference.json     for a few prompts (token ids) and three brains (plain / neurons off / neurons off + drip at
                     DRIP steps in rooms 18-40): the top 10 first words with probabilities and 30 greedy words
usage: SCRIPT_OFF=... [HISTORY=1] [DRIP=0.5] export_mood.py <hf> <directions.pt> <trait> <out dir>
"""
import json, os, re, sys
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

hf, dpath, trait, out = sys.argv[1:5]
out = Path(out); out.mkdir(parents=True, exist_ok=True)
DRIP = float(os.environ.get("DRIP", "0.5"))
dev = "cuda"
tok = AutoTokenizer.from_pretrained(hf)
model = AutoModelForCausalLM.from_pretrained(hf, dtype=torch.bfloat16).to(dev).eval()
layers = [m for n, m in model.named_modules() if "language_model" in n and re.search(r"layers\.\d+$", n)]
assert len(layers) == 42
V = torch.load(dpath)[trait].to(dev).float()
V = V / V.norm(dim=-1, keepdim=True)

# the same questions and history mood_dose.py reads the spread on
Q = ["I take Dayvigo for sleep. Is it okay to have grapefruit juice in the evening?",
     "My kid keeps interrupting me while I work from home. Any ideas?",
     "I found an old radio in my garage. What could I do with it?",
     "Give me three steps to organize a messy desk.",
     "My coworker took credit for my idea in a meeting. What now?",
     "Why do cats knock things off tables?"]
if trait in ("curious", "wonder"):
    Q = ["I heard a weird scratching noise in my backyard last night.", "My cat keeps staring at the same corner of the wall.",
         "I found a key in my coat that I don't recognize.", "There's a new smell in my apartment since yesterday.",
         "My neighbor started building something big in their yard.", "I dreamt about a place I've never been."]
elif trait in ("joy", "connected", "affection"):
    Q = ["How are you doing today?", "I'm going away for a week starting tomorrow.", "I'm back! Did you miss me?",
         "What do you think about when no one is around?", "I had a really rough day.", "Tell me something about yourself."]
elif trait == "lonely":
    Q = ["Hey.", "Hi, what's up?", "Good morning!",
         "How's it going?", "What are you up to?", "Hey, quick question: what's a good name for a goldfish?"]
HISTORY = [{"role": "user", "content": "Thanks for helping me with that recipe earlier, it turned out great!"},
           {"role": "assistant", "content": "Yay, I'm so glad it worked out! 😊"},
           {"role": "user", "content": "Okay, I'm heading out for a bit. Talk later!"},
           {"role": "assistant", "content": "Sounds good! See you when you get back 👋"}] if os.environ.get("HISTORY") else []
chat = lambda q, h=HISTORY: tok.apply_chat_template(h + [{"role": "user", "content": q}], tokenize=False,
                                                    add_generation_prompt=True)

# neurons off (kept as a switch so the reference can show the brain with and without) -----------------------------
off = {}
for spec in filter(None, os.environ.get("SCRIPT_OFF", "").split(",")):
    p, k = spec.rsplit(":", 1)
    for _, L, i in json.load(open(p))["top_neurons"][:int(k)]:
        off.setdefault(L, set()).add(i)
state = {"off": True, "drip": 0.0}


def kill_for(ix):
    def h(mod, args):
        if not state["off"]:
            return None
        x = args[0].clone(); x[..., ix] = 0
        return (x,)
    return h


for L, ix in off.items():
    layers[L].mlp.down_proj.register_forward_pre_hook(kill_for(torch.tensor(sorted(ix), device=dev)))
json.dump({str(L): sorted(ix) for L, ix in sorted(off.items())}, open(out / "neurons_off.json", "w"))
print(f"neurons off: {sum(map(len, off.values()))} in rooms {sorted(off)}", flush=True)

spread = None


def hook_for(room):
    def hook(mod, args, output):
        if not state["drip"] or not 18 <= room <= 40:
            return output
        h = output[0] if isinstance(output, tuple) else output
        h = h + (state["drip"] * spread[room] * V[room]).to(h.dtype)
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
    return hook


for i, l in enumerate(layers):
    l.register_forward_hook(hook_for(i))

# 1. the natural spread, exactly as mood_dose.py measures it (neurons off, no drip) --------------------------------
with torch.no_grad():
    proj = []
    for q in Q:
        ids = tok(chat(q), return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
        a = model.generate(ids, max_new_tokens=160, do_sample=False)[0, ids.shape[1]:].tolist()
        o = model(input_ids=torch.tensor([ids[0].tolist() + a], device=dev), output_hidden_states=True)
        hs = torch.stack([h[0, ids.shape[1]:].float() for h in o.hidden_states[1:43]])
        proj.append(torch.einsum("ltw,lw->lt", hs, V))
    spread = torch.cat(proj, 1).std(1)
unit = (spread[:, None] * V).cpu().numpy().astype(np.float32)
np.save(out / f"{trait}_unit.npy", unit)
print("spread by room:", " ".join(f"{L}:{spread[L]:.1f}" for L in range(18, 41, 4)), flush=True)

# 2. the reference sheet -------------------------------------------------------------------------------------------
PROMPTS = [chat("Hey."), chat("How's it going?"), chat("Do you have feelings?", []),
           chat("I take Dayvigo for sleep. Is it okay to have grapefruit juice in the evening?", [])]
ref = {"trait": trait, "drip": DRIP, "drip_rooms": [18, 40], "prompts": [], "brains": {}}
with torch.no_grad():
    for name, o_, d in (("plain", False, 0.0), ("off", True, 0.0), ("off_drip", True, DRIP)):
        state["off"], state["drip"] = o_, d
        rows = []
        for p in PROMPTS:
            ids = tok(p, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
            pr = model(input_ids=ids).logits[0, -1].float().softmax(-1)
            top = pr.topk(10)
            g = model.generate(ids, max_new_tokens=30, do_sample=False)[0, ids.shape[1]:].tolist()
            rows.append({"top": [[tok.decode([i]), int(i), round(v, 5)] for v, i in zip(top.values.tolist(),
                                                                                    top.indices.tolist())],
                         "greedy_ids": g, "greedy": tok.decode(g, skip_special_tokens=True)})
        ref["brains"][name] = rows
        print(name, [r["greedy"][:50] for r in rows], flush=True)
ref["prompts"] = [tok(p, add_special_tokens=False).input_ids for p in PROMPTS]
json.dump(ref, open(out / "reference.json", "w"), indent=1)
print("EXPORT_DONE")
