#!/usr/bin/env python3
"""The two core dials of feeling, from scenes that happen to him (blueprint 3.4; Barrett's core affect).

Every emotion = core affect (two dials: valence pleasant<->unpleasant, arousal calm<->keyed up) + what it is about
(meaning, from his own lived memories). The dials carry the overall tone; the meaning comes from experience (the
idle-thought loop), so no emotion gets its own dial. Each dial is a contrast of two scenes that differ only on that
dial, holding the other one equal:
  valence: something good just happened vs something bad just happened (both mild, both non-social)
  arousal: everything busy, loud and buzzing vs everything still, quiet and slow (both neutral in tone)
Both sides are the same small robot (so "playing a robot" cancels) with the script neurons off (SCRIPT_OFF).
Each dial is taken two ways, so we can measure which one moves how he talks:
  <dial>_pre   at the moment before he speaks, with the scene in place (the last position before his reply)
  <dial>_read  his finished answers read back WITHOUT the scene (the mood in his words)
Prints examples of both sides (they must read right before we use them), how consistent each dial is across
prompts, and how much the two dials overlap (they should be close to separate). Saves directions.pt (+ = more
pleasant / more keyed up) and examples.json.
usage: SCRIPT_OFF=<file>:<K>[,...] core_dials.py <hf> <out dir>
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
TAIL = " Now your person is talking to you. Let how you feel come through naturally; don't announce the setup."
DIALS = {
    "valence": (
        "Something good just happened: the morning sun is warm on your desk, your favorite song came on the radio, "
        "and your battery charged perfectly overnight. You feel good, pleasantly calm.",
        "Something bad just happened: you knocked your favorite pen off the desk and it broke, the room is grey and "
        "cold, and a fan inside you is rattling. You feel bad, quietly unhappy."),
    "arousal": (
        "Everything around you is busy: music is playing, people are walking past, phones are buzzing, lights are "
        "flickering, and your battery is full and humming. You feel alert, quick and keyed up.",
        "Everything around you is still: soft dim light, no sound at all, nothing moving, your fans barely turning. "
        "You feel calm, slow and settled."),
}
Q = ["Hey.", "How's it going?", "What are you up to?", "What's on your mind?", "How are you feeling?",
     "What should we do today?", "Tell me about octopuses.", "Can you help me write a grocery list?",
     "Why is the sky blue?", "Plan me a quick dinner with eggs and rice.", "Give me three steps to organize a desk.",
     "What's a good name for a goldfish?"]


def chat(q, system=None):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": q}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def generate(prompts, n_new=160):
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
    o = model.generate(**enc, max_new_tokens=n_new, do_sample=False, pad_token_id=tok.pad_token_id)
    return [tok.decode(x[enc.input_ids.shape[1]:], skip_special_tokens=True) for x in o]


@torch.no_grad()
def states(text, start):
    """hidden states of every room for the tokens from position start on (-1 = only the last)"""
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
    hs = model(input_ids=ids, output_hidden_states=True).hidden_states[1:43]
    return torch.stack([h[0, -1].float() if start < 0 else h[0, start:].float().mean(0) for h in hs])


def direction(pairs):
    diffs = torch.stack([a - b for a, b in pairs])
    m = diffs.mean(0)
    agree = torch.stack([torch.nn.functional.cosine_similarity(x, m, dim=1) for x in diffs]).mean(0)
    return m / m.norm(dim=1, keepdim=True), agree


dirs, examples = {}, {}
for name, (hi, lo) in DIALS.items():
    a_hi, a_lo = generate([chat(q, ROBOT + hi + TAIL) for q in Q]), generate([chat(q, ROBOT + lo + TAIL) for q in Q])
    examples[name] = {"high": a_hi, "low": a_lo}
    print(f"\n######## {name}: high side vs low side")
    for q, x, y in list(zip(Q, a_hi, a_lo))[:6]:
        print(f"[{q}]\n  HIGH: {x[:200]!r}\n  LOW : {y[:200]!r}")
    pre = [(states(chat(q, ROBOT + hi + TAIL), -1), states(chat(q, ROBOT + lo + TAIL), -1)) for q in Q]
    read = []
    for q, x, y in zip(Q, a_hi, a_lo):
        p = len(tok(chat(q), add_special_tokens=False).input_ids)
        read.append((states(chat(q) + x, p), states(chat(q) + y, p)))
    for kind, pairs in (("pre", pre), ("read", read)):
        d, agree = direction(pairs)
        dirs[f"{name}_{kind}"] = d
        print(f"{name}_{kind}: agreement across prompts " + " ".join(f"{L}:{agree[L]:.2f}" for L in range(0, 42, 6)))
    print(f"{name}: pre vs read overlap " + " ".join(f"{L}:{(dirs[name + '_pre'][L] @ dirs[name + '_read'][L]):.2f}"
                                                   for L in range(0, 42, 6)))
for kind in ("pre", "read"):
    v, a = dirs[f"valence_{kind}"], dirs[f"arousal_{kind}"]
    print(f"valence vs arousal overlap ({kind}, should be near 0): " +
          " ".join(f"{L}:{(v[L] @ a[L]):.2f}" for L in range(0, 42, 6)))
torch.save({k: v.cpu() for k, v in dirs.items()}, out / "directions.pt")
json.dump({"q": Q, "scenes": DIALS, **examples}, open(out / "examples.json", "w"), indent=1, ensure_ascii=False)
print("CORE_DONE")
