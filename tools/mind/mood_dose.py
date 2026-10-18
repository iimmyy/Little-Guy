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
  words:<dose>                      the word ladder alone at that dose, no drip (control: is it the inner state or
                                    just the words?)
LADDER=<ladder.json>:<B> (ladder_build.py) lights up a word ladder tied to the drip's dose: each rung's words get up
to +B on their scores as the dose crosses the rung's band, and a rung dims to half once the next one is fully on.
At dose 0 nothing changes. Drift is measured with the ladder on.
PHRASES=<phrase_ladder.json>:<B> (phrase_ladder.py) the same for his own phrases, context-aware: once he has started
a phrase its next word gets the push, and its first word a smaller one (start x B).
SCRIPT_OFF=<as_an_ai.json>:<K> switches off the top K "As an AI" neurons in every pattern; drift is then measured
against that brain (the switch-off alone costs ~0.003).
usage: mood_dose.py <hf> <wiki> <directions.pt> <trait> <out dir> <pattern> ...
"""
import json, os, re, sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

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
# HISTORY=1: a short earlier conversation between the two of them that ends with the person heading out, so "Hey."
# comes from someone he knows coming back (how long they were gone is never said: that is the dose)
HISTORY = [{"role": "user", "content": "Thanks for helping me with that recipe earlier, it turned out great!"},
           {"role": "assistant", "content": "Yay, I'm so glad it worked out! 😊"},
           {"role": "user", "content": "Okay, I'm heading out for a bit. Talk later!"},
           {"role": "assistant", "content": "Sounds good! See you when you get back 👋"}] if os.environ.get("HISTORY") else []
chat = lambda q: tok.apply_chat_template(HISTORY + [{"role": "user", "content": q}], tokenize=False,
                                         add_generation_prompt=True)


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
    if kind == "words":
        return {}
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


state = {"plan": {}, "bias": None, "phr": {"full": {}}, "used": []}
spread = None

# word ladder (ladder_build.py): a bias on word scores that follows the dose --------------------------------------
LADDER = None
if os.environ.get("LADDER"):
    lp, B = os.environ["LADDER"].rsplit(":", 1)
    LADDER = json.load(open(lp)); LADDER["B"] = float(B)


def dose_of(p):
    """the dose a pattern stands for: the drip's steps, or the number in words:<dose>; 0 for none/soft"""
    if p.startswith("dose:") or p.startswith("words:"):
        return float(p.rsplit(":", 1)[1])
    return 0.0


PHR = None
if os.environ.get("PHRASES"):
    pp, PB = os.environ["PHRASES"].rsplit(":", 1)
    PHR = json.load(open(pp)); PHR["B"] = float(PB)


def lit_rungs(lad, dose):
    raw = [min(1.0, max(0.0, (dose - r["from"]) / (r["to"] - r["from"]))) for r in lad["rungs"]]
    return [s * (1 - lad["dim"] * (raw[k + 1] if k + 1 < len(raw) else 0)) for k, s in enumerate(raw)]


def phrase_table(dose):
    """the phrase lights at this dose: {started phrase: [(next token, push, phrase)]}, {first token: [(push,
    phrase)]}, {whole phrase: phrase}. Once he has said a phrase it stops being pushed for the rest of that reply
    (repetition suppression: a neuron answers less to what it just did), so it can't feed itself into a loop and
    his other phrases get room."""
    tab = {"cont": {}, "start": {}, "full": {}}
    if PHR is None or dose <= 0:
        return tab
    pid = 0
    for s, r in zip(lit_rungs(PHR, dose), PHR["rungs"]):
        for ph in r["phrases"]:
            v = PHR["B"] * s * ph["w"]
            if not v:
                continue
            for f in map(tuple, ph["forms"]):
                tab["full"][f] = pid
                tab["start"].setdefault(f[0], []).append((v * PHR["start"], pid))
                for k in range(1, len(f)):
                    tab["cont"].setdefault(f[:k], []).append((f[k], v, pid))
            pid += 1
    print(f"  phrases at dose {dose}: rungs lit {[round(x, 2) for x in lit_rungs(PHR, dose)]}, {pid} phrases",
          flush=True)
    return tab


def phrase_push(ids, scores, used):
    """one step for one reply: mark phrases he just finished, then push the unfinished ones (ids: recent tokens)"""
    t = state["phr"]
    for n in t["lens"]:
        if len(ids) >= n and tuple(ids[-n:]) in t["full"]:
            used.add(t["full"][tuple(ids[-n:])])
    for k in range(1, min(len(ids), t["maxpre"]) + 1):
        for tk, v, pid in t["cont"].get(tuple(ids[-k:]), ()):
            if pid not in used:
                scores[tk] += v
    for tk, opts in t["start"].items():                         # shared first words: the strongest unused, no stacking
        v = max((v for v, pid in opts if pid not in used), default=0.0)
        if v:
            scores[tk] += v
    return scores


def ladder_bias(dose, size):
    if LADDER is None or dose <= 0:
        return None
    raw = [min(1.0, max(0.0, (dose - r["from"]) / (r["to"] - r["from"]))) for r in LADDER["rungs"]]
    lit = [s * (1 - LADDER["dim"] * (raw[k + 1] if k + 1 < len(raw) else 0)) for k, s in enumerate(raw)]
    bias = torch.zeros(size, device=dev)
    for s, r in zip(lit, LADDER["rungs"]):
        if s and r["ids"]:
            bias.index_add_(0, torch.tensor(r["ids"], device=dev),
                            LADDER["B"] * s * torch.tensor(r["w"], device=dev))
    print(f"  ladder at dose {dose}: rungs lit {[round(x, 2) for x in lit]}", flush=True)
    return bias


class AddBias(LogitsProcessor):
    def __call__(self, input_ids, scores):
        if state["bias"] is not None:
            scores = scores + state["bias"].to(scores.dtype)
        if state["phr"]["full"]:
            if len(state["used"]) != scores.shape[0]:
                state["used"] = [set() for _ in range(scores.shape[0])]
            for b in range(scores.shape[0]):
                phrase_push(input_ids[b, -state["phr"]["maxp"]:].tolist(), scores[b], state["used"][b])
        return scores


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
    state["used"] = [set() for _ in prompts]
    o = model.generate(**enc, max_new_tokens=n_new, do_sample=True, temperature=1.0, top_k=64, top_p=0.95,
                       pad_token_id=tok.pad_token_id, logits_processor=LogitsProcessorList([AddBias()]))
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
        lg = model(input_ids=w).logits.float()
        if state["bias"] is not None:
            lg = lg + state["bias"]
        if state["phr"]["full"]:
            ids, used = w[0].tolist(), set()
            for t in range(len(ids)):
                phrase_push(ids[max(0, t + 1 - state["phr"]["maxp"]):t + 1], lg[0, t], used)
        kl.append((b.exp() * (b - lg.log_softmax(-1))).sum(-1).mean().item())
    return sum(kl) / len(kl)


rows = []
for p in PATTERNS:
    state["plan"] = parse(p)
    state["bias"] = ladder_bias(dose_of(p), BASE[0].shape[-1])
    state["phr"] = phrase_table(dose_of(p))
    state["phr"]["lens"] = sorted({len(f) for f in state["phr"]["full"]})
    state["phr"]["maxpre"] = max(map(len, state["phr"]["cont"]), default=0)
    state["phr"]["maxp"] = max(state["phr"]["lens"], default=1)
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
    state["plan"], state["bias"], state["phr"] = {}, None, {"full": {}}
    r = {"pattern": p, "mouth": round(mouth, 3), "drift": round(d, 5), "score": round(mouth / max(d, 1e-6), 2),
         "level_by_room": [round(x, 2) for x in level]}
    rows.append(r)
    safe = re.sub(r"[^0-9a-z.+=,-]", "_", p)
    json.dump({**r, "answers": ans}, open(out / f"{trait}_{safe}.json", "w"), indent=1)
    print(json.dumps({k: r[k] for k in ("pattern", "mouth", "drift", "score")}), flush=True)
json.dump({"trait": trait, "rows": rows, "layer_scalar": scal}, open(out / f"{trait}_dose.json", "w"), indent=1)
print("DOSE_DONE")
