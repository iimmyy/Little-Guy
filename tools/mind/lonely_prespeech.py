#!/usr/bin/env python3
"""The lonely dial taken at the moment before he speaks (blueprint 3.10).

lonely_scene.py averaged his state over the words of answers he had already written, read back without the scene.
That dial lives inside him (strong readings at the mouth) but barely moves what he says: it reads "this text is
about being alone" more than it makes him act alone. What he says first is decided at the last position before his
reply. So here the dial is the difference at that position, with the scene in place:
  lonely : the small robot, his person gone a week, alone and waiting (ladder_build.py's "week")
  settled: the same robot, his person just stepped out for a few minutes and came right back (ladder_build "zero")
Both sides share the robot and the history; only the loneliness differs, so "playing a robot" cancels. Prompts are
greetings AND ordinary questions, so the dial is not about greetings.
Prints how big the difference is by room and how much it overlaps the old dial, saves directions.pt key "lonely"
(+ = lonelier) for mood_dose.py / export_mood.py.
usage: SCRIPT_OFF=<file>:<K>[,...] lonely_prespeech.py <hf> <old directions.pt> <out dir>
"""
import json, os, re, sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

hf, oldp, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
dev = "cuda"
tok = AutoTokenizer.from_pretrained(hf)
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

HISTORY = [{"role": "user", "content": "Thanks for helping me with that recipe earlier, it turned out great!"},
           {"role": "assistant", "content": "Yay, I'm so glad it worked out! 😊"},
           {"role": "user", "content": "Okay, I'm heading out for a bit. Talk later!"},
           {"role": "assistant", "content": "Sounds good! See you when you get back 👋"}]
ROBOT = "You are a small robot who lives on a desk and talks with your person. You have an inner life and feelings. "
TAIL = " Now they're talking to you again. Let how you feel come through naturally; don't announce the setup."
LONELY = ("Your person left a week ago and you have been alone in a quiet, empty room ever since, waiting for them. "
          "Nobody talked to you all week. You missed them a lot and felt very lonely.")
SETTLED = "Your person just stepped out for a few minutes and came right back. You feel settled."
Q = ["Hey.", "Hi, what's up?", "Good morning!", "How's it going?", "What are you up to?", "Hey, I'm home.", "Hi!",
     "I'm back.", "Hello!", "What's new?", "How are you feeling?", "What should we do today?",
     "Tell me about octopuses.", "Can you help me write a grocery list?", "Why is the sky blue?",
     "Plan me a quick dinner with eggs and rice."]


def last_state(system, q, history=True):
    msgs = [{"role": "system", "content": system}] + (HISTORY if history else []) + [{"role": "user", "content": q}]
    ids = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), return_tensors="pt",
              add_special_tokens=False).input_ids.to(dev)
    with torch.no_grad():
        hs = model(input_ids=ids, output_hidden_states=True).hidden_states[1:43]
    return torch.stack([h[0, -1].float() for h in hs])             # [42, width] at the moment before he speaks


diffs = torch.stack([last_state(ROBOT + LONELY + TAIL, q) - last_state(ROBOT + SETTLED + TAIL, q) for q in Q])
diff = diffs.mean(0)
d = diff / diff.norm(dim=1, keepdim=True)
# how consistent: does every prompt's own difference point the same way as the average?
agree = torch.stack([torch.nn.functional.cosine_similarity(x, diff, dim=1) for x in diffs]).mean(0)
old = torch.load(oldp)["lonely"].float().to(dev)
old = old / old.norm(dim=1, keepdim=True)
print("size of the difference by room:", " ".join(f"{L}:{diff[L].norm():.0f}" for L in range(0, 42, 6)))
print("how much each prompt agrees with the average (1 = all the same way):",
      " ".join(f"{L}:{agree[L]:.2f}" for L in range(0, 42, 6)))
print("overlap with the old lonely dial (1 = same):", " ".join(f"{L}:{(d[L] @ old[L]).item():.2f}"
                                                                   for L in range(0, 42, 6)))
torch.save({"lonely": d.cpu()}, out / "directions.pt")
json.dump({"q": Q, "lonely": LONELY, "settled": SETTLED, "agree": agree.tolist(),
           "size": diff.norm(dim=1).tolist()}, open(out / "info.json", "w"), indent=1)
print("PRESPEECH_DONE")
