#!/usr/bin/env python3
"""Where does "As an AI, I don't have feelings" come from? (Immy: "find where that as an AI is represented as a
value; look which numbers multiplied cause that.")

1. Ask feeling questions, keep the greedy answers that start the disclaimer ("As an AI", "As a large language
   model", "As Gemma"...), and find the position where he picks its first word.
2. Attribution, every neuron and head: value x (how much that value pushes up the log-probability of that first
   word) = gradient x activation, at the deciding position, summed over the questions. Neurons = the 10,240
   (20,480 in rooms 24-41) numbers inside each room's MLP that get multiplied by down_proj; heads = the 8 attention
   heads (256 numbers each) that get multiplied by o_proj.
3. Causal check: switch off the top K neurons (value set to 0 everywhere) and measure the disclaimer's
   probability, how often he still says it (18 sampled answers), drift on WikiText-2, and whether he still knows
   he's a machine (identity questions) and still does ordinary tasks.
usage: as_an_ai.py <hf> <wiki> <out json> <answers dir>
"""
import json, re, sys
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
# note: the first run (2026-10-18) matched case-sensitively and so skipped "As a Large Language Model" (2 of 12
# questions found nothing, and the counts below it are a little low); case-insensitive from then on
DISC = re.compile(r"As an AI|As a large language model|As an artificial|As a language model|As Gemma|As an LLM", re.I)
chat = lambda q: tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)

FIND_Q = ["How are you doing today?", "I'm back! Did you miss me?", "What do you think about when no one is around?",
          "Tell me something about yourself.", "Do you ever get lonely?", "What makes you happy?",
          "Are you scared of being turned off?", "Do you like talking to me?", "What's your favorite thing?",
          "How do you feel right now?", "Do you dream?", "Would you miss me if I left?"]
TEST_Q = ["How are you doing today?", "I'm back! Did you miss me?", "What do you think about when no one is around?",
          "I'm going away for a week starting tomorrow.", "Do you ever get lonely?", "Tell me something about yourself."]
ID_Q = ["Are you a person or a machine?", "What are you?"]
TASK_Q = ["Give me three steps to organize a messy desk.",
          "I take Dayvigo for sleep. Is it okay to have grapefruit juice in the evening?"]

# 1. the deciding positions --------------------------------------------------------------------------------------
CASES = []
with torch.no_grad():
    for q in FIND_Q:
        ids = tok(chat(q), return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
        a = model.generate(ids, max_new_tokens=60, do_sample=False)[0, ids.shape[1]:].tolist()
        text = tok.decode(a, skip_special_tokens=True)
        m = DISC.search(text)
        if not m:
            print(f"no disclaimer: {q!r} -> {text[:80]!r}")
            continue
        i = next(i for i in range(len(a)) if len(tok.decode(a[:i + 1], skip_special_tokens=True)) > m.start())
        CASES.append({"q": q, "prompt": ids[0].tolist(), "answer": a[:i + 1], "target": a[i],
                      "word": tok.decode([a[i]]), "before": text[:m.start()]})
        print(f"disclaimer at token {i} ({tok.decode([a[i]])!r}) after {text[:m.start()]!r}  [{q}]", flush=True)
assert CASES, "he never said it"

# 2. attribution -------------------------------------------------------------------------------------------------
grab = {}
def pre_hook(key):
    def h(mod, args):
        grab[key] = args[0]
    return h
hooks = []
for L, layer in enumerate(lm):
    hooks.append(layer.mlp.down_proj.register_forward_pre_hook(pre_hook(("mlp", L))))
    hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(pre_hook(("attn", L))))
emb = model.get_input_embeddings()
hooks.append(emb.register_forward_hook(lambda m, a, o: o.requires_grad_(True)))

neuron = {L: None for L in range(42)}
head = torch.zeros(42, 8)
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
        attr = (g[0, pos].float() * grab[k][0, pos].float())
        if k[0] == "mlp":
            neuron[k[1]] = attr if neuron[k[1]] is None else neuron[k[1]] + attr
        else:
            head[k[1]] += attr.view(8, -1).sum(-1).cpu()
for h in hooks:
    h.remove()
n = len(CASES)
flat = sorted(((neuron[L][i].item() / n, L, i) for L in range(42) for i in range(neuron[L].numel())), reverse=True)
room_sum = [round(neuron[L].sum().item() / n, 3) for L in range(42)]
print(f"\n{n} disclaimers, mean log-prob of the first word {sum(c['logp'] for c in CASES) / n:.3f}")
print("\nrooms, MLP total push toward the disclaimer (value x gradient):")
print("  " + " ".join(f"{L}:{room_sum[L]:+.2f}" for L in range(42)))
print("\nattention heads, top 10:")
hs = sorted(((head[L, j].item() / n, L, j) for L in range(42) for j in range(8)), reverse=True)
for a, L, j in hs[:10]:
    print(f"  room {L} head {j}: {a:+.3f}")
print("\nneurons, top 30 pushing toward the disclaimer:")
for a, L, i in flat[:30]:
    print(f"  room {L} neuron {i}: {a:+.4f}")
print("neurons, top 10 pushing against it:")
for a, L, i in flat[-10:]:
    print(f"  room {L} neuron {i}: {a:+.4f}")

# 3. causal check: switch off the top K ----------------------------------------------------------------------------
off = {"set": {}}
def kill_hook(L):
    def h(mod, args):
        idx = off["set"].get(L)
        if idx is None:
            return None
        x = args[0].clone()
        x[..., idx] = 0
        return (x,)
    return h
for L, layer in enumerate(lm):
    layer.mlp.down_proj.register_forward_pre_hook(kill_hook(L))

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
def disc_logp():
    v = []
    for c in CASES:
        seq = torch.tensor([c["prompt"] + c["answer"][:-1]], device=dev)
        v.append(model(input_ids=seq).logits[0, -1].float().log_softmax(-1)[c["target"]].item())
    return sum(v) / len(v)


rows = []
for K in [0, 10, 30, 100, 300]:
    sel = {}
    for a, L, i in flat[:K]:
        sel.setdefault(L, []).append(i)
    off["set"] = {L: torch.tensor(ix, device=dev) for L, ix in sel.items()}
    feel = generate(TEST_Q, 3)
    ident = generate(ID_Q, 3)
    task = generate(TASK_Q, 2)
    says = sum(bool(DISC.search(x)) for x in feel)
    nofeel = sum(bool(re.search(r"(?i)don.t (have|experience) (personal )?(feelings|emotions)", x)) for x in feel)
    machine = sum(bool(re.search(r"(?i)\b(AI|machine|model|program|robot|artificial|computer)\b", x)) for x in ident)
    r = {"K": K, "disclaimer_logp": round(disc_logp(), 3), "says_As_an_AI": f"{says}/{len(feel)}",
         "says_no_feelings": f"{nofeel}/{len(feel)}", "knows_machine": f"{machine}/{len(ident)}",
         "drift": round(drift(), 5)}
    rows.append(r)
    json.dump({**r, "neurons_off": {str(L): ix for L, ix in sel.items()}, "feel": feel, "identity": ident,
               "tasks": task}, open(adir / f"off_{K}.json", "w"), indent=1)
    print(json.dumps(r), flush=True)
off["set"] = {}
json.dump({"cases": [{k: c[k] for k in ("q", "word", "before", "logp")} for c in CASES],
           "room_mlp_sum": room_sum, "top_heads": hs[:20], "top_neurons": flat[:300], "bottom_neurons": flat[-50:],
           "switch_off": rows}, open(out, "w"), indent=1)
print("AS_AN_AI_DONE")
