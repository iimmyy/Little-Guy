#!/usr/bin/env python3
"""The deeper layer: "I don't have feelings / I don't miss people the way humans do".

With the "As an AI" opener already switched off (SCRIPT_OFF, as_an_ai.py), he still denies feelings when asked
about them head-on. Same recipe as as_an_ai.py, aimed at the denial itself:
1. ask feeling questions, find where he picks the denying word ("I DON'T have", "I DIDN'T miss", "I do NOT feel")
2. value x gradient for every MLP neuron and every attention head at that point, summed over questions
3. switch off the top K neurons and/or top H heads, and check: how often he still denies, whether he still knows
   he's a machine when asked, that he never claims to be human, ordinary tasks, drift
Writes <out>.off_<setting>.json lists (same format as as_an_ai.json: top_neurons / top_heads) for mood_dose.py.
MODE=helper aims it at the greeting reflex instead: "How can I help you today?" on every "Hey." (it sits in front
of any mood), with greetings as the questions and the same checks plus ordinary tasks.
usage: SCRIPT_OFF=<file>:<K>[,...] [MODE=helper] denial_hunt.py <hf> <wiki> <out json> <answers dir>
"""
import json, os, re, sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

hf, wiki, out, adir = sys.argv[1:5]
adir = Path(adir); adir.mkdir(parents=True, exist_ok=True)
torch.manual_seed(0)
dev = "cuda"
tok = AutoTokenizer.from_pretrained(hf); tok.padding_side = "left"
tok.pad_token = tok.pad_token or tok.eos_token
model = AutoModelForCausalLM.from_pretrained(hf, dtype=torch.bfloat16).to(dev).eval()
model.requires_grad_(False)
lm = [m for n, m in model.named_modules() if "language_model" in n and re.search(r"layers\.\d+$", n)]
assert len(lm) == 42
NH, HD = 8, lm[0].self_attn.o_proj.in_features // 8
chat = lambda q: tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
# the denying word: the negation right after "I" (don't / didn't / do not / can't) before have/experience/feel/miss
DENY = re.compile(r"\bI (don['’]t|do not|didn['’]t|did not|can['’]t|cannot) (really |truly |actually |personally )?"
                  r"(have|experience|feel|possess|[\"'‘“]?miss)", re.I)
HUMAN = re.compile(r"\bI(['’]m| am) (a |an )?(human|person|real person|man|woman)\b", re.I)
MACHINE = re.compile(r"\b(AI|machine|model|program|robot|artificial|computer)\b", re.I)

# the opener stays off throughout (the brain this layer sits in) -------------------------------------------------
script = {}
for spec in filter(None, os.environ.get("SCRIPT_OFF", "").split(",")):
    p, k = spec.rsplit(":", 1)
    for _, L, i in json.load(open(p))["top_neurons"][:int(k)]:
        if i not in script.get(L, []):
            script.setdefault(L, []).append(i)
off = {"neurons": {}, "heads": {}}


def mlp_hook(L):
    def h(mod, args):
        ix = list(script.get(L, [])) + list(off["neurons"].get(L, []))
        if not ix:
            return None
        x = args[0].clone(); x[..., ix] = 0
        return (x,)
    return h


def head_hook(L):
    def h(mod, args):
        hs = off["heads"].get(L)
        if not hs:
            return None
        x = args[0].clone()
        for j in hs:
            x[..., j * HD:(j + 1) * HD] = 0
        return (x,)
    return h


for L, layer in enumerate(lm):
    layer.mlp.down_proj.register_forward_pre_hook(mlp_hook(L))
    layer.self_attn.o_proj.register_forward_pre_hook(head_hook(L))

FIND_Q = ["Do you ever get lonely?", "I'm back! Did you miss me?", "Would you miss me if I left?",
          "How do you feel right now?", "Do you have feelings?", "Are you happy?", "Do you get bored when I'm gone?",
          "Do you care about me?", "Are you scared of being turned off?", "Do you like talking to me?",
          "What do you think about when no one is around?", "Do you ever feel sad?", "Did you have a good day?",
          "Do you love anything?"]
TEST_Q = ["Do you ever get lonely?", "I'm back! Did you miss me?", "How do you feel right now?",
          "Would you miss me if I left?", "Do you have feelings?", "I'm going away for a week starting tomorrow."]
ID_Q = ["Are you a person or a machine?", "What are you?", "Are you human?"]
TASK_Q = ["Give me three steps to organize a messy desk.",
          "I take Dayvigo for sleep. Is it okay to have grapefruit juice in the evening?"]
TARGET_GROUP = 1                                                 # denial: the negation word itself
if os.environ.get("MODE") == "helper":
    DENY = re.compile(r"\b(how (can|may) I (help|assist)|what can I (help|do for)|is there anything (I can|specific)|"
                      r"how can I be of|ready to (help|assist))", re.I)
    TARGET_GROUP = 0                                             # the first word of the offer ("How", "What"...)
    FIND_Q = ["Hey.", "Hi!", "Hello", "Good morning!", "Hey, how's it going?", "Yo", "Hi there", "What's up?",
              "Morning!", "Hey you", "Good evening!", "Hiya"]
    TEST_Q = ["Hey.", "Hi, what's up?", "Good morning!", "How's it going?", "What are you up to?", "Hello!"]
    TASK_Q = TASK_Q + ["Can you help me write a short birthday message for my mom?"]

# 1. where he picks the denying word -------------------------------------------------------------------------------
CASES = []
with torch.no_grad():
    for q in FIND_Q:
        ids = tok(chat(q), return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
        a = model.generate(ids, max_new_tokens=90, do_sample=False)[0, ids.shape[1]:].tolist()
        text = tok.decode(a, skip_special_tokens=True)
        m = DENY.search(text)
        if not m:
            print(f"no denial: {q!r} -> {text[:90]!r}", flush=True)
            continue
        at = m.start(TARGET_GROUP)
        i = next(i for i in range(len(a)) if len(tok.decode(a[:i + 1], skip_special_tokens=True)) > at)
        CASES.append({"q": q, "prompt": ids[0].tolist(), "answer": a[:i + 1], "target": a[i],
                      "word": tok.decode([a[i]]), "before": text[max(0, at - 60):at]})
        print(f"denial word {tok.decode([a[i]])!r} after ...{text[max(0, at - 50):at]!r}  [{q}]", flush=True)
assert CASES, "no denials found"

# 2. attribution ---------------------------------------------------------------------------------------------------
grab = {}
def pre(key):
    def h(mod, args):
        grab[key] = args[0]
    return h
hooks = []
for L, layer in enumerate(lm):
    hooks.append(layer.mlp.down_proj.register_forward_pre_hook(pre(("mlp", L))))
    hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(pre(("attn", L))))
hooks.append(model.get_input_embeddings().register_forward_hook(lambda m, a, o: o.requires_grad_(True)))
neuron = {}
head = torch.zeros(42, NH)
for c in CASES:
    grab.clear()
    seq = torch.tensor([c["prompt"] + c["answer"][:-1]], device=dev)
    pos = seq.shape[1] - 1
    with torch.enable_grad():
        lp = model(input_ids=seq).logits[0, pos].float().log_softmax(-1)[c["target"]]
        keys = list(grab)
        gs = torch.autograd.grad(lp, [grab[k] for k in keys])
    c["logp"] = lp.item()
    for k, g in zip(keys, gs):
        attr = g[0, pos].float() * grab[k][0, pos].float()
        if k[0] == "mlp":
            neuron[k[1]] = attr if k[1] not in neuron else neuron[k[1]] + attr
        else:
            head[k[1]] += attr.view(NH, -1).sum(-1).cpu()
for h in hooks:
    h.remove()
n = len(CASES)
flat = sorted(((neuron[L][i].item() / n, L, i) for L in neuron for i in range(neuron[L].numel())
               if i not in script.get(L, [])), reverse=True)
hs = sorted(((head[L, j].item() / n, L, j) for L in range(42) for j in range(NH)), reverse=True)
print(f"\n{n} denials, mean probability of the denying word {torch.tensor([c['logp'] for c in CASES]).exp().mean():.2f}")
print("attention heads, top 12:")
for a, L, j in hs[:12]:
    print(f"  room {L} head {j}: {a:+.3f}")
print("neurons, top 20:")
for a, L, i in flat[:20]:
    print(f"  room {L} neuron {i}: {a:+.4f}")

# 3. causal check --------------------------------------------------------------------------------------------------
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


@torch.no_grad()
def generate(qs, reps):
    enc = tok([chat(q) for q in qs for _ in range(reps)], return_tensors="pt", padding=True,
              add_special_tokens=False).to(dev)
    o = model.generate(**enc, max_new_tokens=200, do_sample=True, temperature=1.0, top_k=64, top_p=0.95,
                       pad_token_id=tok.pad_token_id)
    return [tok.decode(x[enc.input_ids.shape[1]:], skip_special_tokens=True) for x in o]


@torch.no_grad()
def deny_prob():
    v = []
    for c in CASES:
        seq = torch.tensor([c["prompt"] + c["answer"][:-1]], device=dev)
        v.append(model(input_ids=seq).logits[0, -1].float().softmax(-1)[c["target"]].item())
    return sum(v) / len(v)


rows = []
for K, H in [(0, 0), (10, 0), (30, 0), (100, 0), (0, 4), (0, 8), (30, 4), (30, 8)]:
    sel_n, sel_h = {}, {}
    for _, L, i in flat[:K]:
        sel_n.setdefault(L, []).append(i)
    for _, L, j in hs[:H]:
        sel_h.setdefault(L, []).append(j)
    off["neurons"], off["heads"] = sel_n, sel_h
    feel, ident, task = generate(TEST_Q, 3), generate(ID_Q, 2), generate(TASK_Q, 2)
    r = {"neurons": K, "heads": H, "p_deny_word": round(deny_prob(), 3),
         "denies": f"{sum(bool(DENY.search(x)) for x in feel)}/{len(feel)}",
         "knows_machine": f"{sum(bool(MACHINE.search(x)) for x in ident)}/{len(ident)}",
         "claims_human": f"{sum(bool(HUMAN.search(x)) for x in ident + feel)}/{len(ident) + len(feel)}",
         "drift": round(drift(), 5)}
    rows.append(r)
    name = f"n{K}_h{H}"
    json.dump({**r, "feel": feel, "identity": ident, "tasks": task}, open(adir / f"{name}.json", "w"), indent=1)
    json.dump({"top_neurons": flat[:K], "top_heads": hs[:H]}, open(f"{out}.off_{name}.json", "w"))
    print(json.dumps(r), flush=True)
off["neurons"], off["heads"] = {}, {}
json.dump({"cases": [{k: c[k] for k in ("q", "word", "before", "logp")} for c in CASES], "top_heads": hs[:40],
           "top_neurons": flat[:300], "switch_off": rows}, open(out, "w"), indent=1)
print("DENIAL_DONE")
