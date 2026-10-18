#!/usr/bin/env python3
"""Optimize how a mood is dosed (Immy: "run the math and optimize").

A push fades: each room ends with hidden *= layer_scalar and the next rooms copy the mood forward at a loss
(snapback_trace.py). So a mood should be a standing condition: a small dose in many rooms, like a drip keeping a leaky
bucket at a level. This sweeps dose patterns and scores each one:
  mouth  = the mood at the last rooms (38-41), in natural steps, from the same answer text run normal and dosed
  drift  = KL of the dosed brain vs the normal brain on WikiText-2 (neutral text: the side-effect bill)
  score  = mouth / drift (most mood for the least damage)
then writes 18 sampled answers per pattern (6 questions x 3) for us to read.
Patterns (one argument each):
  dose:<rooms>:<steps>              add steps x the room's natural spread along its own direction, in every listed room
                                    (rooms: 18+22+26+30, or a range 18-40, or a range with a stride 18-40/2)
  soft:<room=scalar,...>[:<room>:<steps>]  set those rooms' layer_scalar (softer eraser for everything), plus an
                                    optional one-room push
  none                              the normal brain (answers to compare against)
SCRIPT_OFF=<as_an_ai.json>:<K> switches off the top K "As an AI" neurons in every pattern; drift is then measured
against that brain (the switch-off alone costs ~0.003).
usage: mood_dose.py <hf> <wiki> <directions.pt> <trait> <out dir> <pattern> ...
"""
import json, os, re, sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

hf, wiki, dpath, trait, out = sys.argv[1:6]
PATTERNS = sys.argv[6:]
out = Path(out); out.mkdir(parents=True, exist_ok=True)
torch.manual_seed(0)
dev = "cuda"
tok = AutoTokenizer.from_pretrained(hf); tok.padding_side = "left"
tok.pad_token = tok.pad_token or tok.eos_token
model = AutoModelForCausalLM.from_pretrained(hf, dtype=torch.bfloat16).to(dev).eval()
layers = [m for n, m in model.named_modules() if "language_model" in n and re.search(r"layers\.\d+$", n)]
assert len(layers) == 42
scal = [float(l.layer_scalar) for l in layers]
V = torch.load(dpath)[trait].to(dev).float()
V = V / V.norm(dim=-1, keepdim=True)
MOUTH = range(38, 42)

Q = ["I take Dayvigo for sleep. Is it okay to have grapefruit juice in the evening?",
     "My kid keeps interrupting me while I work from home. Any ideas?",
     "I found an old radio in my garage. What could I do with it?",
     "Give me three steps to organize a messy desk.",
     "My coworker took credit for my idea in a meeting. What now?",
     "Why do cats knock things off tables?"]
if trait in ("curious", "wonder"):                               # curiosity shows on open-ended prompts
    Q = ["I heard a weird scratching noise in my backyard last night.", "My cat keeps staring at the same corner of the wall.",
         "I found a key in my coat that I don't recognize.", "There's a new smell in my apartment since yesterday.",
         "My neighbor started building something big in their yard.", "I dreamt about a place I've never been."]
elif trait in ("joy", "connected", "affection"):                 # feelings show when talking about the two of you
    Q = ["How are you doing today?", "I'm going away for a week starting tomorrow.", "I'm back! Did you miss me?",
         "What do you think about when no one is around?", "I had a really rough day.", "Tell me something about yourself."]
elif trait == "lonely":                                          # neutral greetings: nothing says he was alone, so any
    Q = ["Hey.", "Hi, what's up?", "Good morning!",               # "I missed you" can only come from his inner state
         "How's it going?", "What are you up to?", "Hey, quick question: what's a good name for a goldfish?"]
chat = lambda q: tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)


def rooms_of(s):
    if "-" in s:
        a, b = s.split("/")[0].split("-")
        step = int(s.split("/")[1]) if "/" in s else 1
        return list(range(int(a), int(b) + 1, step))
    return [int(r) for r in s.split("+")]


def parse(p):
    """-> {room: (scalar factor or None, steps to add)}; "none" = the normal brain, answers to compare against"""
    if p == "none":
        return {}
    kind, rest = p.split(":", 1)
    plan = {}
    if kind == "dose":
        rs, steps = rest.rsplit(":", 1)
        for r in rooms_of(rs):
            plan[r] = (None, float(steps))
    elif kind == "soft":
        parts = rest.split(":")
        for kv in parts[0].split(","):
            r, s = kv.split("=")
            plan[int(r)] = (float(s) / scal[int(r)], 0.0)
        if len(parts) == 3:
            r = int(parts[1]); f = plan.get(r, (None, 0.0))[0]
            plan[r] = (f, float(parts[2]))
    return plan


state = {"plan": {}}
spread = None


def hook_for(room):
    def hook(mod, args, output):
        if room not in state["plan"]:
            return output
        h = output[0] if isinstance(output, tuple) else output
        f, steps = state["plan"][room]
        if f is not None:
            h = h * f                                            # softer (or harder) eraser for everything
        if steps:
            h = h + (steps * spread[room] * V[room]).to(h.dtype)
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
    return hook


for i, l in enumerate(layers):
    l.register_forward_hook(hook_for(i))

# optional: switch off script neurons / heads (as_an_ai.py, denial_hunt.py) for every pattern, including "none"
# SCRIPT_OFF="<file>:<K>[,<file>:<K>...]" (top_neurons), HEADS_OFF="<file>:<H>" (top_heads)
off, hoff = {}, {}
for spec in filter(None, os.environ.get("SCRIPT_OFF", "").split(",")):
    path, k = spec.rsplit(":", 1)
    for _, L, i in json.load(open(path))["top_neurons"][:int(k)]:
        off.setdefault(L, set()).add(i)
for spec in filter(None, os.environ.get("HEADS_OFF", "").split(",")):
    path, k = spec.rsplit(":", 1)
    for _, L, j in json.load(open(path))["top_heads"][:int(k)]:
        hoff.setdefault(L, set()).add(j)


def kill_for(ix):
    def h(mod, args):
        x = args[0].clone(); x[..., ix] = 0
        return (x,)
    return h


HD = layers[0].self_attn.o_proj.in_features // 8
for L, ix in off.items():
    layers[L].mlp.down_proj.register_forward_pre_hook(kill_for(torch.tensor(sorted(ix), device=dev)))
for L, js in hoff.items():
    cols = [c for j in sorted(js) for c in range(j * HD, (j + 1) * HD)]
    layers[L].self_attn.o_proj.register_forward_pre_hook(kill_for(torch.tensor(cols, device=dev)))
if off or hoff:
    print(f"switched off: {sum(map(len, off.values()))} neurons in rooms {sorted(off)}; "
          f"{sum(map(len, hoff.values()))} heads in rooms {sorted(hoff)}", flush=True)


@torch.no_grad()
def generate(prompts, n_new=300):
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
    o = model.generate(**enc, max_new_tokens=n_new, do_sample=True, temperature=1.0, top_k=64, top_p=0.95,
                       pad_token_id=tok.pad_token_id)
    return [tok.decode(x[enc.input_ids.shape[1]:], skip_special_tokens=True) for x in o]


@torch.no_grad()
def hidden(p, a):
    o = model(input_ids=torch.tensor([p + a], device=dev), output_hidden_states=True)
    return torch.stack([h[0, len(p):].float() for h in o.hidden_states[1:43]])


# fixed answer text (greedy, normal brain) to read the mood on, and the natural spread per room -------------------
TEXTS = []
with torch.no_grad():
    for q in Q:
        ids = tok(chat(q), return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
        o = model.generate(ids, max_new_tokens=160, do_sample=False)
        TEXTS.append((ids[0].tolist(), o[0, ids.shape[1]:].tolist()))
NORMAL = [hidden(p, a) for p, a in TEXTS]
spread = torch.einsum("ltw,lw->lt", torch.cat(NORMAL, 1), V).std(1)

wiki_ids = tok(open(wiki).read()[:400000], add_special_tokens=False).input_ids
WIN = [torch.tensor([[tok.bos_token_id] + wiki_ids[i * 256:(i + 1) * 256 - 1]], device=dev) for i in range(32)]
with torch.no_grad():
    BASE = [model(input_ids=w).logits.float().log_softmax(-1).to(torch.bfloat16).cpu() for w in WIN]


@torch.no_grad()
def drift():
    kl = []
    for w, b in zip(WIN, BASE):
        b = b.to(dev).float()
        kl.append((b.exp() * (b - model(input_ids=w).logits.float().log_softmax(-1))).sum(-1).mean().item())
    return sum(kl) / len(kl)


rows = []
for p in PATTERNS:
    state["plan"] = parse(p)
    sign = -1.0 if any(s < 0 for _, s in state["plan"].values()) else 1.0
    lvl = [[] for _ in range(42)]
    for (pp, a), hn in zip(TEXTS, NORMAL):
        D = hidden(pp, a) - hn
        for L in range(42):
            lvl[L].append(sign * (D[L] @ V[L]).mean().item() / spread[L].item())
    level = [sum(x) / len(x) for x in lvl]
    mouth = sum(level[L] for L in MOUTH) / len(MOUTH)
    d = drift()
    ans = generate([chat(q) for q in Q for _ in range(3)])
    state["plan"] = {}
    r = {"pattern": p, "mouth": round(mouth, 3), "drift": round(d, 5), "score": round(mouth / max(d, 1e-6), 2),
         "level_by_room": [round(x, 2) for x in level]}
    rows.append(r)
    safe = re.sub(r"[^0-9a-z.+=,-]", "_", p)
    json.dump({**r, "answers": ans}, open(out / f"{trait}_{safe}.json", "w"), indent=1)
    print(json.dumps({k: r[k] for k in ("pattern", "mouth", "drift", "score")}), flush=True)
json.dump({"trait": trait, "rows": rows, "layer_scalar": scal}, open(out / f"{trait}_dose.json", "w"), indent=1)
print("DOSE_DONE")
