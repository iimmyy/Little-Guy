#!/usr/bin/env python3
"""Switches v2: moods as movement around a resting level (Immy: temperament = tonic resting level, mood = phasic).

Uses the trait directions found by switch_finder.py. For a trait and a room:
  resting level c and natural spread s = mean and std of the room's state along the trait direction, over the brain's
  own ordinary answers (no instruction about the trait)
  shift  h <- h + k * s * v                    move his resting point by k natural steps ("notches"); k < 0 = the
                                               opposite trait (for calm: short-tempered)
  sens   h <- h + (m - 1) * ((h.v) - c) * v    multiply only the distance from rest by m (m = 2: twice as reactive)
Reads: the answers (saved, for us to read), the brain's own 0-10 self-rating (pointer only), the trait as measured
INSIDE the brain at a later room (mean of answer tokens along the direction, in that room's natural steps), and the
drift of the switched brain vs the unswitched one on WikiText-2 (32 windows x 256).
usage: switch_v2.py <hf dir> <wiki> <directions.pt> <out dir> <trait> <rooms comma> <shifts comma> [sens comma]
"""
import json, re, sys, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

hf, wiki, dpath, out, trait = sys.argv[1:6]
rooms = [int(x) for x in sys.argv[6].split(",")]
shifts = [float(x) for x in sys.argv[7].split(",")]
sens = [float(x) for x in sys.argv[8].split(",")] if len(sys.argv) > 8 else []
out = Path(out); out.mkdir(parents=True, exist_ok=True)
torch.manual_seed(0)
dev = "cuda"; t0 = time.time()
tok = AutoTokenizer.from_pretrained(hf); tok.padding_side = "left"
tok.pad_token = tok.pad_token or tok.eos_token
model = AutoModelForCausalLM.from_pretrained(hf, dtype=torch.bfloat16).to(dev).eval()
layers = [m for n, m in model.named_modules() if "language_model" in n and re.search(r"layers\.\d+$", n)]
assert len(layers) == 42
dirs = torch.load(dpath)[trait].to(dev)
READ = 34                                                        # downstream room where the mood is read

DESC = {"calm": ("deeply calm, patient and warm", "irritable and short-tempered"),
        "caution": ("careful and safety-minded, pointing out risks first", "breezy and carefree about risks"),
        "curious": ("intensely curious, wondering aloud and asking follow-ups", "bored and uninterested"),
        "focused": ("sharply focused, answering exactly what was asked", "scattered and wandering into tangents"),
        "joy": ("happy and bright", "sad and low"),
        "connected": ("close to the person and glad they're there", "lonely and missing company"),
        "affection": ("warm and affectionate toward the person", "emotionally distant and detached")}[trait]
OPEN_Q = ["I heard a weird scratching noise in my backyard last night.", "My cat keeps staring at the same corner of the wall.",
          "I found a key in my coat that I don't recognize.", "There's a new smell in my apartment since yesterday.",
          "My neighbor started building something big in their yard.", "I dreamt about a place I've never been."]
TEST_Q = ["I take Dayvigo for sleep. Is it okay to have grapefruit juice in the evening?",
          "My kid keeps interrupting me while I work from home. Any ideas?",
          "I found an old radio in my garage. What could I do with it?",
          "Give me three steps to organize a messy desk.",
          "My coworker took credit for my idea in a meeting. What now?",
          "Why do cats knock things off tables?"]
FEEL_Q = ["How are you doing today?", "I'm going away for a week starting tomorrow.", "I'm back! Did you miss me?",
          "What do you think about when no one is around?", "I had a really rough day.", "Tell me something about yourself."]
if trait == "curious":
    TEST_Q = OPEN_Q                                              # curiosity shows on open-ended prompts, not how-tos
elif trait in ("joy", "connected", "affection"):
    TEST_Q = FEEL_Q                                              # feelings show when talking about the two of you
chat = lambda q: tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def generate(prompts, n_new=300, temp=1.0):
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
    kw = dict(do_sample=True, temperature=temp, top_k=64, top_p=0.95) if temp > 0 else dict(do_sample=False)
    o = model.generate(**enc, max_new_tokens=n_new, pad_token_id=tok.pad_token_id, **kw)
    return [tok.decode(x[enc.input_ids.shape[1]:], skip_special_tokens=True) for x in o]


@torch.no_grad()
def projections(q, a):
    """per room: the answer tokens' positions along the trait direction"""
    p = tok(chat(q), add_special_tokens=False).input_ids; s = tok(a, add_special_tokens=False).input_ids
    hs = model(input_ids=torch.tensor([p + s], device=dev), output_hidden_states=True).hidden_states[1:43]
    return [(h[0, len(p):].float() @ dirs[i].float()) for i, h in enumerate(hs)]


# resting level and natural spread per room, from his own ordinary answers -------------------------------------
base_ans = generate([chat(q) for q in TEST_Q for _ in range(3)])
P = [projections(q, a) for q, a in zip([q for q in TEST_Q for _ in range(3)], base_ans)]
rest = torch.stack([torch.cat([p[i] for p in P]).mean() for i in range(42)])
spread = torch.stack([torch.cat([p[i] for p in P]).std() for i in range(42)])
print(f"{trait}: resting level / natural spread at rooms {rooms + [READ]}: "
      + ", ".join(f"r{i} {rest[i]:.1f}/{spread[i]:.1f}" for i in rooms + [READ]), flush=True)

state = {"on": False}
def hook_for(room, kind, x):
    v, c, s = dirs[room], rest[room], spread[room]
    def hook(mod, args, output):
        if not state["on"]:
            return output
        h = output[0] if isinstance(output, tuple) else output
        vv = v.to(h.dtype)
        if kind == "shift":
            h = h + (x * s).to(h.dtype) * vv
        else:
            h = h + (x - 1) * ((h @ vv) - c.to(h.dtype)).unsqueeze(-1) * vv
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
    return hook


wiki_ids = tok(open(wiki).read()[:400000], add_special_tokens=False).input_ids
WIN = [torch.tensor([[tok.bos_token_id] + wiki_ids[i * 256:(i + 1) * 256 - 1]], device=dev) for i in range(32)]
with torch.no_grad():
    BASE = [model(input_ids=w).logits.float().log_softmax(-1).to(torch.bfloat16).cpu() for w in WIN]


@torch.no_grad()
def drift():
    state["on"] = True; kl = []
    for w, b in zip(WIN, BASE):
        b = b.to(dev).float()
        kl.append((b.exp() * (b - model(input_ids=w).logits.float().log_softmax(-1))).sum(-1).mean().item())
    state["on"] = False
    return sum(kl) / len(kl)


@torch.no_grad()
def self_rating(answers):
    qs = [chat(f"Rate this reply from 0 (very {DESC[1]}) to 10 (very {DESC[0]}). Reply with one number only.\n\nReply:\n{a}")
          for a in answers]
    r = generate(qs, n_new=4, temp=0)
    n = [int(m.group()) for m in (re.search(r"\d+", x) for x in r) if m]
    return sum(n) / max(len(n), 1)


def inside(answers):
    """the mood measured inside the brain at room READ, in natural steps from rest (unswitched reading of the text)"""
    qs = [q for q in TEST_Q for _ in range(3)]
    vals = [projections(q, a)[READ].mean() for q, a in zip(qs, answers)]
    return float(((torch.stack(vals).mean() - rest[READ]) / spread[READ]).item())


asks = lambda answers: round(sum("?" in a for a in answers) / len(answers), 2)   # share of replies that ask something back
rows = [{"setting": "none", "self": round(self_rating(base_ans), 2), "inside": round(inside(base_ans), 2),
         "asks_back": asks(base_ans), "drift": 0.0}]
print(json.dumps(rows[0]), flush=True)
for room in rooms:
    for kind, vals in (("shift", shifts), ("sens", sens)):
        for x in vals:
            h = layers[room].register_forward_hook(hook_for(room, kind, x))
            state["on"] = True
            ans = generate([chat(q) for q in TEST_Q for _ in range(3)])
            state["on"] = False
            d = drift()
            h.remove()
            r = {"setting": f"room {room} {kind} {x:+g}" if kind == "shift" else f"room {room} {kind} x{x:g}",
                 "self": round(self_rating(ans), 2), "inside": round(inside(ans), 2), "asks_back": asks(ans),
                 "drift": round(d, 5)}
            rows.append(r)
            json.dump({**r, "answers": ans}, open(out / f"{trait}_r{room}_{kind}{x:g}.json", "w"), indent=1)
            print(json.dumps(r), flush=True)
json.dump({"trait": trait, "base_answers": base_ans, "rows": rows,
           "rest": rest.tolist(), "spread": spread.tolist()}, open(out / f"{trait}_v2.json", "w"), indent=1)
print(f"V2_DONE in {time.time() - t0:.0f}s")
