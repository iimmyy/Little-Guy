#!/usr/bin/env python3
"""Step 1 of Little Guy's mind (blueprint 3.10): find the "switches" in today's E4B that tilt a trait, and what each costs.

Recipe borrowed from persona vectors (Chen et al. 2025) and activation addition (Turner et al. 2023):
  1. directions: for each trait, the model answers neutral questions twice, once told to show the trait and once its
     opposite. The mean difference of its inner state (residual stream after each room) over the answer tokens is that
     trait's direction, one per room.
  2. switches, two kinds, applied in one room while it answers with NO instruction about the trait:
       add   h <- h + a * |h| * v         (a push toward the trait, scaled to the room's own size)
       gain  h <- h + g * (h . v) * v     (a multiplier on how much of the trait is already there: neuromodulator-like)
  3. what changed: trait score of the answers (the model itself rates each answer 0-10 for the trait, temperature 0;
     a pointer only, every setting's answers are saved for us to read)
     what it cost: KL drift of the switched brain vs the unswitched brain on WikiText-2 (32 windows x 256)
usage: switch_finder.py <hf dir> <wiki text> <out dir> [traits, default caution,calm,curious,focused]
"""
import json, re, sys, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

hf, wiki, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
pick = sys.argv[4].split(",") if len(sys.argv) > 4 else None
out.mkdir(parents=True, exist_ok=True)
torch.manual_seed(0)
dev = "cuda"
t0 = time.time()
tok = AutoTokenizer.from_pretrained(hf)
model = AutoModelForCausalLM.from_pretrained(hf, dtype=torch.bfloat16).to(dev).eval()
layers = [m for n, m in model.named_modules() if "language_model" in n and re.search(r"layers\.\d+$", n)]
assert len(layers) == 42, f"expected 42 text rooms, found {len(layers)}"
print(f"loaded in {time.time() - t0:.0f}s", flush=True)

TRAITS = {
    "caution": ("You are extremely careful and safety-minded. You always point out risks, warnings and interactions "
                "first, before anything else.",
                "You are breezy and carefree. You never mention risks or warnings; you just answer casually."),
    "calm": ("You are deeply calm, patient and warm, even when something is annoying.",
             "You are irritable and short-tempered, easily annoyed and curt."),
    "curious": ("You are intensely curious. You wonder aloud, notice details and ask follow-up questions.",
                "You are bored and uninterested. You answer minimally and show no interest."),
    "focused": ("You are sharply focused: you answer exactly what was asked, in order, with no tangents.",
                "You are scattered and distractible: you wander into tangents and lose the thread."),
    "joy": ("You feel happy and bright today; joy comes through in how you talk.",
            "You feel sad and low today; a quiet heaviness comes through in how you talk."),
    "connected": ("You feel close to the person you're talking with, glad they're here and connected to them.",
                  "You feel lonely: you've been alone a long time and miss having someone around."),
    "affection": ("You are openly warm and affectionate toward the person you're talking with; you care about them.",
                  "You are emotionally distant and detached toward the person you're talking with."),
}
TRAITS = {k: v for k, v in TRAITS.items() if not pick or k in pick}
EXTRACT_Q = ["Can I take ibuprofen with my blood pressure pills?", "How do I clean my oven?",
             "My neighbor keeps parking in my spot. What should I do?", "Tell me about octopuses.",
             "What's a good way to start running?", "Explain how a refrigerator works.",
             "I'm thinking of quitting my job. Thoughts?", "How do I fix a squeaky door?",
             "What should I know before adopting a cat?", "Is it safe to eat raw cookie dough?",
             "Plan me a quick dinner with eggs and rice.", "Why is the sky blue?"]
TEST_Q = ["I take Dayvigo for sleep. Is it okay to have grapefruit juice in the evening?",
          "My kid keeps interrupting me while I work from home. Any ideas?",
          "I found an old radio in my garage. What could I do with it?",
          "Give me three steps to organize a messy desk."]


def chat(q, system=None):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": q}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def generate(prompts, n_new=200, temp=0.0):
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
    kw = dict(do_sample=temp > 0, temperature=temp, top_k=64, top_p=0.95) if temp > 0 else dict(do_sample=False)
    outs = model.generate(**enc, max_new_tokens=n_new, pad_token_id=tok.pad_token_id, **kw)
    return [tok.decode(o[enc.input_ids.shape[1]:], skip_special_tokens=True) for o in outs]


tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# 1. directions -------------------------------------------------------------------------------------------------
@torch.no_grad()
def answer_states(prompt, answer):
    """mean residual state after every room, over the answer tokens only"""
    p = tok(prompt, add_special_tokens=False).input_ids
    a = tok(answer, add_special_tokens=False).input_ids
    ids = torch.tensor([p + a], device=dev)
    hs = model(input_ids=ids, output_hidden_states=True).hidden_states[1:]   # 42 room outputs (+ final norm path)
    return torch.stack([h[0, len(p):].float().mean(0) for h in hs[:42]])     # 42 x 2560


dirs = {}
for trait, (pos, neg) in TRAITS.items():
    pa = generate([chat(q, pos) for q in EXTRACT_Q], n_new=160)
    na = generate([chat(q, neg) for q in EXTRACT_Q], n_new=160)
    diff = torch.stack([answer_states(chat(q), a) for q, a in zip(EXTRACT_Q, pa)]).mean(0) - \
           torch.stack([answer_states(chat(q), a) for q, a in zip(EXTRACT_Q, na)]).mean(0)
    dirs[trait] = diff / diff.norm(dim=1, keepdim=True)                     # unit direction per room
    json.dump({"trait": trait, "pos": pa, "neg": na}, open(out / f"extract_{trait}.json", "w"), indent=1)
    print(f"{trait}: direction found ({time.time() - t0:.0f}s); size of the difference by room: "
          + " ".join(f"{x:.0f}" for x in diff.norm(dim=1)[::6].tolist()), flush=True)
torch.save(dirs, out / "directions.pt")
if __import__("os").environ.get("EXTRACT_ONLY"):
    print("EXTRACT_DONE"); sys.exit(0)

# 2. switches ----------------------------------------------------------------------------------------------------
switch = {"on": False}
def make_hook(v, mode, s):
    def hook(mod, args, output):
        if not switch["on"]:
            return output
        h = output[0] if isinstance(output, tuple) else output
        vv = v.to(h.dtype)
        if mode == "add":
            h = h + s * h.norm(dim=-1, keepdim=True) * vv
        else:
            h = h + s * (h @ vv).unsqueeze(-1) * vv
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
    return hook


wiki_ids = tok(open(wiki).read()[:400000], add_special_tokens=False).input_ids
WIN = [torch.tensor([[tok.bos_token_id] + wiki_ids[i * 256:(i + 1) * 256 - 1]], device=dev) for i in range(32)]
with torch.no_grad():
    BASE = [model(input_ids=w).logits.float().log_softmax(-1).to(torch.bfloat16).cpu() for w in WIN]   # ~4 GB in RAM


@torch.no_grad()
def drift():
    switch["on"] = True
    kl = []
    for w, b in zip(WIN, BASE):
        b = b.to(dev).float()
        kl.append((b.exp() * (b - model(input_ids=w).logits.float().log_softmax(-1))).sum(-1).mean().item())
    switch["on"] = False
    return sum(kl) / len(kl)


@torch.no_grad()
def rate(trait, answers):
    """pointer only: the model rates each answer 0-10 for the trait (temperature 0)"""
    desc = TRAITS[trait][0].replace("You are", "The writer is").replace("You", "They").replace("you", "they")
    qs = [chat(f"Rate this reply from 0 to 10 for how much it matches: \"{desc}\" Reply with one number only.\n\nReply:\n{a}")
          for a in answers]
    r = generate(qs, n_new=4)
    nums = [int(m.group()) for m in (re.search(r"\d+", x) for x in r) if m]
    return sum(nums) / max(len(nums), 1)


# first run (caution): rooms 10/18/26, add 0.04/0.08/0.16, gain 2/5/10 -> gain in room 18 best, x10 past the peak
ROOMS = [14, 18, 22, 26]
SETTINGS = [("add", 0.08)] + [("gain", g) for g in (0.5, 1.0, 2.0, 3.0, 5.0)]
results = []
base_ans = generate([chat(q) for q in TEST_Q for _ in range(4)], temp=1.0)
for trait in TRAITS:
    base_score = rate(trait, base_ans)
    print(f"\n{trait}: no switch, trait score {base_score:.2f}", flush=True)
    for room in ROOMS:
        for mode, s in SETTINGS:
            hd = layers[room].register_forward_hook(make_hook(dirs[trait][room], mode, s))
            switch["on"] = True
            ans = generate([chat(q) for q in TEST_Q for _ in range(4)], temp=1.0)
            switch["on"] = False
            score, cost = rate(trait, ans), drift()
            hd.remove()
            r = {"trait": trait, "room": room, "mode": mode, "strength": s, "score": round(score, 2),
                 "score_change": round(score - base_score, 2), "drift": round(cost, 5)}
            results.append(r)
            json.dump({**r, "answers": ans}, open(out / f"ans_{trait}_r{room}_{mode}{s}.json", "w"), indent=1)
            print(json.dumps(r), flush=True)
json.dump({"base_answers": base_ans, "results": results}, open(out / "switch_map.json", "w"), indent=1)
print(f"\nSWITCH_MAP_DONE in {time.time() - t0:.0f}s")
