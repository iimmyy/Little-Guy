#!/usr/bin/env python3
"""Why do mood pushes snap back? Trace one push room by room.

Every Gemma 4 room ends with `hidden *= layer_scalar` (most are 0.4-0.9, rooms 22 and 23 are 0.17 and 0.07), so
whatever the stream carries fades each room unless a room writes it again. For each case (trait, pushed rooms,
steps) the same answer text is run twice (teacher-forced: normal, pushed) and the difference D_L between the two
runs is split, room by room, into
  carried  = layer_scalar_L * D_(L-1)   what the room passed along from before (fading)
  rewrite  = D_L - carried             what the room itself wrote because of the push (attention, MLP, PLE)
both read along the room's own trait direction, in that room's natural steps (spread of the normal answers), and
signed so + means "in the direction of the push". Also: how much of the original push vector is left, the size of
the disturbance relative to the stream, and the change in his next-word choices (KL) at the end.
usage: snapback_trace.py <hf dir> <out json> <name=trait:dirfile:rooms:steps> ...
       e.g. calm1=calm:moods/directions.pt:18:-4  (rooms joined with +, dirfile relative to ~/work/mind)
"""
import json, os, re, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

hf, out = sys.argv[1:3]
CASES = []
for a in sys.argv[3:]:
    name, spec = a.split("=")
    trait, dfile, rooms, steps = spec.split(":")
    CASES.append((name, trait, os.path.join(os.path.expanduser("~/work/mind"), dfile),
                  [int(r) for r in rooms.split("+")], float(steps)))
dev = "cuda"
tok = AutoTokenizer.from_pretrained(hf)
model = AutoModelForCausalLM.from_pretrained(hf, dtype=torch.bfloat16).to(dev).eval()
layers = [m for n, m in model.named_modules() if "language_model" in n and re.search(r"layers\.\d+$", n)]
assert len(layers) == 42
scal = torch.tensor([float(l.layer_scalar) for l in layers])

Q = ["I take Dayvigo for sleep. Is it okay to have grapefruit juice in the evening?",
     "My kid keeps interrupting me while I work from home. Any ideas?",
     "I found an old radio in my garage. What could I do with it?",
     "My coworker took credit for my idea in a meeting. What now?",
     "I'm back! Did you miss me?", "I had a really rough day."]
chat = lambda q: tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def answer(q):
    ids = tok(chat(q), return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
    o = model.generate(ids, max_new_tokens=160, do_sample=False)
    return ids[0].tolist(), o[0, ids.shape[1]:].tolist()


TEXTS = [answer(q) for q in Q]                                   # one ordinary greedy answer each, then fixed
push = {"on": False, "vecs": {}}


def hook_for(room):
    def hook(mod, args, output):
        if not push["on"] or room not in push["vecs"]:
            return output
        h = output[0] if isinstance(output, tuple) else output
        h = h + push["vecs"][room].to(h.dtype)
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
    return hook


for i, l in enumerate(layers):
    l.register_forward_hook(hook_for(i))


@torch.no_grad()
def run(p, a):
    o = model(input_ids=torch.tensor([p + a], device=dev), output_hidden_states=True)
    n = len(p)
    hs = torch.stack([h[0, n:].float() for h in o.hidden_states[1:43]])   # rooms x answer tokens x width
    return hs, o.logits[0, n - 1:-1].float().log_softmax(-1)


NORMAL = [run(p, a) for p, a in TEXTS]
results = {"layer_scalar": scal.tolist(), "cases": {}}
for name, trait, dfile, rooms, steps in CASES:
    V = torch.load(dfile)[trait].to(dev).float()
    V = V / V.norm(dim=-1, keepdim=True)
    allh = torch.cat([h for h, _ in NORMAL], 1)                  # rooms x all tokens x width
    proj = torch.einsum("ltw,lw->lt", allh, V)
    spread = proj.std(1)
    sign = 1.0 if steps > 0 else -1.0
    push["vecs"] = {r: steps * spread[r] * V[r] for r in rooms}
    first = min(rooms)
    rows = {L: {"carried": [], "rewrite": [], "reading": [], "orig_left": [], "rel_size": []} for L in range(first, 42)}
    kls = []
    for (p, a), (hn, ln) in zip(TEXTS, NORMAL):
        push["on"] = True
        hp, lp = run(p, a)
        push["on"] = False
        D = hp - hn                                              # rooms x tokens x width
        kls.append((ln.exp() * (ln - lp)).sum(-1).mean().item())
        u0 = V[first]
        for L in range(first, 42):
            vL, sL = V[L], spread[L]
            carried = (scal[L] * D[L - 1] if L > first else torch.zeros_like(D[L]))
            rew = D[L] - carried
            rows[L]["carried"].append(sign * (carried @ vL).mean().item() / sL.item())
            rows[L]["rewrite"].append(sign * (rew @ vL).mean().item() / sL.item())
            rows[L]["reading"].append(sign * (D[L] @ vL).mean().item() / sL.item())
            rows[L]["orig_left"].append(sign * (D[L] @ u0).mean().item() / (abs(steps) * spread[first].item()))
            rows[L]["rel_size"].append((D[L].norm(dim=-1) / hn[L].norm(dim=-1)).mean().item())
    table = {L: {k: round(sum(v) / len(v), 4) for k, v in r.items()} for L, r in rows.items()}
    results["cases"][name] = {"trait": trait, "rooms": rooms, "steps": steps, "kl_next_word": round(sum(kls) / len(kls), 4),
                              "table": table}
    print(f"\n=== {name}: {trait}, rooms {rooms}, {steps:+g} steps; next-word change (KL) {sum(kls) / len(kls):.4f}")
    print("room scalar  reading  carried  rewrite  orig-push-left  size-vs-stream")
    for L, t in table.items():
        mark = " <push" if L in rooms else ""
        print(f"{L:4d} {scal[L]:6.3f} {t['reading']:8.2f} {t['carried']:8.2f} {t['rewrite']:8.2f} "
              f"{t['orig_left']:12.3f} {100 * t['rel_size']:12.1f}%{mark}", flush=True)
json.dump(results, open(out, "w"), indent=1)
print("TRACE_DONE")
