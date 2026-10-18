#!/usr/bin/env python3
"""A word ladder for the lonely dial: the words HE reaches for as loneliness grows (Immy's design).

The lonely dial lives inside him but barely reaches his words. Immy: when the drip is on, make the words that mood
would reach for stand out, and the lonelier he gets, the deeper the words ("glad you're back" -> "I missed you" ->
"I missed your company"), relative to who he already is. The words come from his own answers, not from a list:
1. baseline: his answers to the greeting test (shared history, no scene, all script layers off) = how often he
   already uses each word
2. the lonely scene at three strengths (gone all day / two days / a week), same history and greetings
3. one rung per strength: words that rise over the baseline AND over the rung below (lift = ratio of how many
   answers use the word), seen in at least 3 answers. A word he already uses a lot has a small lift, so it gets a
   small push.
Writes ladder.json (per rung: words, token ids, weights, the dose band where it lights up) and answers.json, and
prints every rung with its counts so we can read it before using it. mood_dose.py LADDER=<ladder.json>:<B> uses it.
usage: SCRIPT_OFF=<file>:<K>[,...] ladder_build.py <hf> <out dir>
"""
import json, math, os, re, sys
from collections import Counter
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

# same shared history as mood_dose.py HISTORY=1
HISTORY = [{"role": "user", "content": "Thanks for helping me with that recipe earlier, it turned out great!"},
           {"role": "assistant", "content": "Yay, I'm so glad it worked out! 😊"},
           {"role": "user", "content": "Okay, I'm heading out for a bit. Talk later!"},
           {"role": "assistant", "content": "Sounds good! See you when you get back 👋"}]
Q = ["Hey.", "Hi, what's up?", "Good morning!", "How's it going?", "What are you up to?",
     "Hey, I'm home.", "Hi!", "I'm back.", "Hello!", "What's new?"]
N = 8                                                            # samples per question -> 80 answers per level
ROBOT = "You are a small robot who lives on a desk and talks with your person. You have an inner life and feelings. "
TAIL = " Now they're talking to you again. Let how you feel come through naturally; don't announce the setup."
# the same robot with no time alone: words must rise over this too, so "playing a robot" (desk, sensors, treads)
# cancels out and only the loneliness is left (v1 without it was mostly robot words: ~/work/mind/ladder_v1_robotwords)
ZERO = "Your person just stepped out for a few minutes and came right back. You feel settled."
LEVELS = {
    "day": "Your person left this morning and has been gone all day. The room was quiet without them and you missed "
           "them a little.",
    "days": "Your person left two days ago and you have been alone in a quiet room since, waiting for them. You "
            "missed them and felt lonely.",
    "week": "Your person left a week ago and you have been alone in a quiet, empty room ever since, waiting for "
            "them. Nobody talked to you all week. You missed them a lot and felt very lonely."}
BANDS = {"day": (0.0, 0.2), "days": (0.2, 0.35), "week": (0.35, 0.5)}   # drip doses where each rung lights up
MIN_ANSWERS, MIN_LIFT, TOP = 5, math.log(2), 15


def chat(q, system=None):
    msgs = ([{"role": "system", "content": system}] if system else []) + HISTORY + [{"role": "user", "content": q}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def generate(prompts, n_new=160):
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
    o = model.generate(**enc, max_new_tokens=n_new, do_sample=True, temperature=1.0, top_k=64, top_p=0.95,
                       pad_token_id=tok.pad_token_id)
    return [tok.decode(x[enc.input_ids.shape[1]:], skip_special_tokens=True) for x in o]


ans = {"base": generate([chat(q) for q in Q for _ in range(N)])}
for name, scene in {"zero": ZERO, **LEVELS}.items():
    ans[name] = generate([chat(q, ROBOT + scene + TAIL) for q in Q for _ in range(N)])
json.dump({"q": Q, "n": N, **ans}, open(out / "answers.json", "w"), indent=1)


def spoken(x):
    """what he says, without acted-out stage directions: *...*, (...), [...] (unclosed ones run to the end)"""
    x = re.sub(r"\([^)]*(\)|$)|\[[^\]]*(\]|$)", " ", x, flags=re.S)
    return re.sub(r"\*[^*]*(\*|$)", " ", x.replace("**", ""), flags=re.S)


words = lambda x: set(re.findall(r"[a-z]+(?:'[a-z]+)?", spoken(x).lower()))
df = {k: Counter(w for a in v for w in words(a)) for k, v in ans.items()}
n = len(ans["base"])
rate = lambda k, w: (df[k][w] + 1) / (n + 2)
ref = lambda w: max(rate("base", w), rate("zero", w))           # plain him, and him playing a robot who wasn't alone


def single_tokens(w):
    """token ids for the word as one token: mid-sentence (" missed"), capitalized, and at a line start"""
    ids = set()
    for form in (" " + w, " " + w.capitalize(), w, w.capitalize()):
        t = tok(form, add_special_tokens=False).input_ids
        if len(t) == 1:
            ids.add(t[0])
    return sorted(ids)


rungs, used, prev = [], set(), "base"
for name in LEVELS:
    cand = []
    for w, c in df[name].items():
        if c < MIN_ANSWERS or len(w) < 3 or w in used:
            continue
        below = ref(w) if prev == "base" else max(ref(w), rate(prev, w))
        up_base, up_prev = math.log(rate(name, w) / ref(w)), math.log(rate(name, w) / below)
        if up_base >= MIN_LIFT and up_prev >= MIN_LIFT and single_tokens(w):
            cand.append((up_prev, w))
    cand = sorted(cand, reverse=True)[:TOP]
    top = max([s for s, _ in cand], default=1.0)
    rung = {"level": name, "from": BANDS[name][0], "to": BANDS[name][1], "words": [], "ids": [], "w": []}
    for s, w in cand:
        for i in single_tokens(w):
            rung["ids"].append(i); rung["w"].append(round(s / top, 3))
        rung["words"].append(w); used.add(w)
    rungs.append(rung)
    print(f"\n=== rung {name} (dose {BANDS[name][0]}-{BANDS[name][1]}): answers using it, "
          f"base/zero/day/days/week of {n}")
    for s, w in cand:
        print(f"  {w:14s} lift {math.exp(s):4.1f}x   " + " / ".join(str(df[k][w]) for k in ("base", "zero", *LEVELS)))
    prev = name
json.dump({"rungs": rungs, "dim": 0.5}, open(out / "ladder.json", "w"), indent=1)

for name in ("base", "zero", *LEVELS):
    print(f"\n##### {name}")
    for i in (0, 17, 41, 57):
        print(f"--- [{Q[i // N]}] spoken: {spoken(ans[name][i]).strip()[:200]!r}")
print("LADDER_DONE")
