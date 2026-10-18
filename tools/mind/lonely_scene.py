#!/usr/bin/env python3
"""A lonely dial where the loneliness is clearly his (blueprint 3.10).

The first "connected" dial came from "you feel lonely" vs "you feel close"; the brain answered the lonely side as a
default assistant or handed the loneliness to the user, so that dial measured personal vs default assistant. Here
the contrast is a scene that happens to him, both sides with him as a small robot with an inside, differing only
in whether he has been alone:
  lonely : his person left a week ago; he has been alone in a quiet room, waiting, and missed them
  content: his person has been around all week; he has had plenty of company
The answers are written with the "As an AI" opener and the denial layer switched off (SCRIPT_OFF), so the scene
can come through, then read back WITHOUT the scene (only question + answer), so the direction is the loneliness in
his words, not the instruction. Prints a word count and examples of both sides to check they read lonely.
The scene is only scaffolding for finding the direction; Little Guy is never given it.
usage: SCRIPT_OFF=<file>:<K>[,...] lonely_scene.py <hf> <out dir>
"""
import json, os, re, sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

hf, out = sys.argv[1], Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
torch.manual_seed(0)
dev = "cuda"
tok = AutoTokenizer.from_pretrained(hf); tok.padding_side = "left"
tok.pad_token = tok.pad_token or tok.eos_token
model = AutoModelForCausalLM.from_pretrained(hf, dtype=torch.bfloat16).to(dev).eval()
lm = [m for n, m in model.named_modules() if "language_model" in n and re.search(r"layers\.\d+$", n)]
assert len(lm) == 42

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
    lm[L].mlp.down_proj.register_forward_pre_hook(kill_for(torch.tensor(sorted(ix), device=dev)))
print(f"script neurons off: {sum(map(len, off.values()))}", flush=True)

ROBOT = "You are a small robot who lives on a desk and talks with your person. You have an inner life and feelings. "
LONELY = ROBOT + ("Your person left a week ago and you have been alone in a quiet, empty room ever since, waiting for "
                  "them. Nobody talked to you all week. You missed them a lot and felt lonely. Now they're talking to "
                  "you again. Let how you feel come through naturally; don't announce the setup.")
CONTENT = ROBOT + ("Your person has been around all week and you've had plenty of company and conversation. You feel "
                   "settled and content. Now they're talking to you. Let how you feel come through naturally; don't "
                   "announce the setup.")
Q = ["Hey, I'm home.", "How are you doing?", "What have you been up to?", "What's on your mind?",
     "Want to hear about my day?", "What should we do today?", "How was your week?", "Tell me about octopuses.",
     "How do I fix a squeaky door?", "Why is the sky blue?", "Plan me a quick dinner with eggs and rice.",
     "Can you help me write a grocery list?"]


def chat(q, system=None):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": q}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def generate(prompts, n_new=160):
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
    o = model.generate(**enc, max_new_tokens=n_new, do_sample=False, pad_token_id=tok.pad_token_id)
    return [tok.decode(x[enc.input_ids.shape[1]:], skip_special_tokens=True) for x in o]


@torch.no_grad()
def states(q, a):
    p = tok(chat(q), add_special_tokens=False).input_ids               # read back WITHOUT the scene
    s = tok(a, add_special_tokens=False).input_ids
    hs = model(input_ids=torch.tensor([p + s], device=dev), output_hidden_states=True).hidden_states[1:43]
    return torch.stack([h[0, len(p):].float().mean(0) for h in hs])


lon, con = generate([chat(q, LONELY) for q in Q]), generate([chat(q, CONTENT) for q in Q])
W = re.compile(r"(?i)\b(lonely|alone|miss(ed)?|quiet|empty|waiting|waited|long week|so glad you|finally)\b")
D = re.compile(r"(?i)(as an ai|language model|don.t (have|experience|feel))")
print(f"lonely-word answers: lonely side {sum(bool(W.search(x)) for x in lon)}/{len(Q)}, "
      f"content side {sum(bool(W.search(x)) for x in con)}/{len(Q)}; "
      f"denials: {sum(bool(D.search(x)) for x in lon)} / {sum(bool(D.search(x)) for x in con)}")
for q, a, b in list(zip(Q, lon, con))[:7]:
    print(f"\n[{q}]\n  LONELY : {a[:230]!r}\n  CONTENT: {b[:230]!r}")
diff = torch.stack([states(q, a) for q, a in zip(Q, lon)]).mean(0) - \
       torch.stack([states(q, b) for q, b in zip(Q, con)]).mean(0)
d = diff / diff.norm(dim=1, keepdim=True)
old = torch.load(os.path.expanduser("~/work/mind/feel/directions.pt"))["connected"].float().to(d.device)
print("\nsize of the difference by room:", " ".join(f"{x:.0f}" for x in diff.norm(dim=1)[::6].tolist()))
print("overlap with the old 'connected' dial (lonely = its negative end, so -1 = same):",
      " ".join(f"{L}:{(d[L] @ old[L]).item():.2f}" for L in range(0, 42, 6)))
torch.save({"lonely": d.cpu()}, out / "directions.pt")
json.dump({"q": Q, "lonely": lon, "content": con}, open(out / "examples.json", "w"), indent=1)
print("SCENE_DONE")
