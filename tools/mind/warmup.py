#!/usr/bin/env python3
"""Does he warm back up as you talk after a long time away? (Immy: "does he naturally return to longer responses
after you've been talking for a while, like after a week?")

A six-turn conversation after the shared history (the recipe, "heading out for a bit"), three ways:
  none   no mood at all (the normal length of each reply, since the questions differ)
  held   the lonely drip held at a week's dose the whole time: does talking to him warm him up on its own?
  drain  the dose drains turn by turn like a drive with company back (blueprint 3.3): 0.5 -> 0.35 -> 0.2 -> 0.1 -> 0
Four separate conversations per way (sampled), each reply goes back into the conversation. Reports spoken words per
turn and writes every conversation to read.
The drip is the exported one natural step per room (export_mood.py <trait>_unit.npy), in rooms 18-40, with the
script neurons off (SCRIPT_OFF), the same as mood_dose.py dose:18-40:<steps>.
usage: SCRIPT_OFF=... warmup.py <hf> <unit.npy> <out json>
"""
import json, os, re, sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

hf, unitp, out = sys.argv[1:4]
torch.manual_seed(0)
dev = "cuda"
tok = AutoTokenizer.from_pretrained(hf); tok.padding_side = "left"
tok.pad_token = tok.pad_token or tok.eos_token
model = AutoModelForCausalLM.from_pretrained(hf, dtype=torch.bfloat16).to(dev).eval()
layers = [m for n, m in model.named_modules() if "language_model" in n and re.search(r"layers\.\d+$", n)]
assert len(layers) == 42
unit = torch.tensor(np.load(unitp), device=dev)

off = {}
for spec in filter(None, os.environ.get("SCRIPT_OFF", "").split(",")):
    p, k = spec.rsplit(":", 1)
    for _, L, i in json.load(open(p))["top_neurons"][:int(k)]:
        off.setdefault(L, set()).add(i)
def kill_for(ix):
    def h(mod, args):
        x = args[0].clone(); x[..., ix] = 0
        return (x,)
    return h
for L, ix in off.items():
    layers[L].mlp.down_proj.register_forward_pre_hook(kill_for(torch.tensor(sorted(ix), device=dev)))

state = {"dose": 0.0}
def drip(room):
    def hook(mod, args, output):
        if not state["dose"] or not 18 <= room <= 40:
            return output
        h = output[0] if isinstance(output, tuple) else output
        h = h + (state["dose"] * unit[room]).to(h.dtype)
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
    return hook
for i, l in enumerate(layers):
    l.register_forward_hook(drip(i))

HISTORY = [{"role": "user", "content": "Thanks for helping me with that recipe earlier, it turned out great!"},
           {"role": "assistant", "content": "Yay, I'm so glad it worked out! 😊"},
           {"role": "user", "content": "Okay, I'm heading out for a bit. Talk later!"},
           {"role": "assistant", "content": "Sounds good! See you when you get back 👋"}]
TURNS = ["Hey.", "What did you do while I was gone?", "I'm making dinner, any ideas for something with chicken?",
         "That sounds good. How are you feeling?", "Tell me something you've been thinking about.",
         "Okay, I'm going to eat. I'll be around tonight."]
WAYS = {"none": [0, 0, 0, 0, 0, 0], "held": [0.5] * 6, "drain": [0.5, 0.35, 0.2, 0.1, 0, 0]}
N = 4


def spoken_words(x):
    x = re.sub(r"\([^)]*(\)|$)|\[[^\]]*(\]|$)|\*[^*]*(\*|$)", " ", x.replace("**", ""), flags=re.S)
    return len(x.split())


res = {}
for way, doses in WAYS.items():
    convs = [list(HISTORY) for _ in range(N)]
    for t, (q, d) in enumerate(zip(TURNS, doses)):
        for c in convs:
            c.append({"role": "user", "content": q})
        state["dose"] = d
        enc = tok([tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in convs],
                  return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        with torch.no_grad():
            o = model.generate(**enc, max_new_tokens=220, do_sample=True, temperature=1.0, top_k=64, top_p=0.95,
                               pad_token_id=tok.pad_token_id)
        for c, x in zip(convs, o):
            c.append({"role": "assistant", "content": tok.decode(x[enc.input_ids.shape[1]:],
                                                                  skip_special_tokens=True).strip()})
    state["dose"] = 0.0
    words = [[spoken_words(c[len(HISTORY) + 2 * t + 1]["content"]) for c in convs] for t in range(len(TURNS))]
    res[way] = {"doses": doses, "words": words, "convs": [c[len(HISTORY):] for c in convs]}
    print(f"{way:6s} words per turn: " + "  ".join(f"{sum(w) / N:5.1f}" for w in words), flush=True)
json.dump({"turns": TURNS, **res}, open(out, "w"), indent=1, ensure_ascii=False)
print("WARMUP_DONE")
