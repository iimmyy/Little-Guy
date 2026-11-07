#!/usr/bin/env python3
"""A positive allosteric modulator for the bloodstream's gain signal (Immy's idea, 2026-10-27).

The finding behind it (peaks.py): each layer's biggest bloodstream number (the "giant") is almost constant, yet its
small wobble carries information. Every layer divides the bloodstream by its loudness (RMS norm) before reading it,
and the giant IS most of that loudness, so its wobble turns the volume of all the other numbers up or down at once: a
single broadcast gain signal, like noradrenaline's adaptive gain in a brain.

Immy: model it as a PAM, not an agonist. An agonist forces the signal; a PAM acts at the RECEPTOR and only strengthens
the natural signal while it is there. Here the receptor is the norm's sensitivity to the giant: at the two norms that
read the bloodstream in every layer (before attention, before the MLP), the loudness is computed as if the giant's
wobble around its resting level were (1 + a) times bigger. The giant's own value still reaches the weights untouched,
and when the giant sits at rest the dose does nothing. a > 0 = PAM, a < 0 = NAM (negative modulator), a = 0 = control.

1. drift per dose (standard test set: wiki / chat / math)
2. behavior per dose: the 30-problem sample (2 per group), greedy, thinking on: right answers and thinking length

3. chat (chat=1): the four raw behavior questions (chat_check.py), N sampled replies per dose, same random stream per
   batch in every dose; written as <dose>.chat.jsonl / .md for reading by hand

usage: pam.py <out folder> <math traces jsonl> <problems.jsonl> [doses=-0.5,0,0.5,1,2 dev=cuda gen=1 chat=0 n=10
                     per=2 tail=...]
  per   problems per group in the behavior sample (2 -> 30 problems, 6 -> 90)
  tail  extra words added to every problem (a prompt-only control, e.g. tail=Be_brief. with underscores as spaces)
  mode  pam (default) | flat: every bloodstream norm's output times (1 + a), nothing tied to the giant |
        rand: the PAM formula on a random ordinary number (not among the 16 loudest) instead of the giant |
        shift: only the PAM's own average volume change per norm, as a constant (the half that neutral=1 removes)
  layers  a-b: dose only these layers (vitals.py, 2026-10-28: only the late giant's wobble tracks closing the thinking)
  zones   a-b:f,c-d:f: a factor per layer range, dose at those layers = dose x f (others 0), e.g. zones=0-16:-0.5,28-41:1
          with doses=2 -> NAM -1 on the early and middle giants, PAM +2 on the late one (overrides layers)
  hs      N: everyday common sense, the first N HellaSwag questions (pick the sensible ending of 4; each ending scored by
          its average log-chance per word), right answers at dose 0 and every dose, paired on the same questions
  neutral 1: keep average volume unchanged; the PAM alone also turns every layer down a little (-4% at 1.0), so each
          norm's average change at that dose (measured on the giant-finding windows) is divided back out
"""
import json, math, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from e4b_mac import load

out, traces, probs = sys.argv[1:4]
opt = dict(a.split("=", 1) for a in sys.argv[4:])
DOSES = [float(d) for d in opt.get("doses", "-0.5,0,0.5,1,2").split(",")]
MODE = opt.get("mode", "pam")
LO, HI = (int(x) for x in opt.get("layers", "0-999").split("-"))
ZONES = [(*map(int, z.split(":")[0].split("-")), float(z.split(":")[1])) for z in opt["zones"].split(",")] if "zones" in opt else \
        [(LO, HI, 1.0)]
NEUTRAL = opt.get("neutral", "0") == "1"
DEV, GEN, CHAT, NCHAT = opt.get("dev", "cuda"), opt.get("gen", "1") == "1", opt.get("chat", "0") == "1", int(opt.get("n", 10))
BENCH = os.path.join(os.path.dirname(__file__), "../../../.run/bench")
os.makedirs(out, exist_ok=True)
if DEV == "cuda":                                          # two jobs share one card: each keeps under its half, or
    torch.cuda.set_per_process_memory_fraction(float(opt.get("gpufrac", 0.47)))   # one's cache starves the other's load
model, tok = load(DEV)
layers = model.model.layers
L, D = len(layers), model.config.hidden_size
F = [next((f for lo, hi, f in ZONES if lo <= i <= hi), 0.0) for i in range(L)]      # dose factor per layer
wiki = tok(open(f"{BENCH}/wiki.test.raw").read()[:200000], add_special_tokens=False).input_ids
chat = tok(open(f"{BENCH}/chat_heldout.txt").read(), add_special_tokens=False).input_ids
rows = [json.loads(l) for l in open(traces)]


def mathwin(r):
    p = tok.apply_chat_template([{"role": "user", "content": r["q"]}], tokenize=False, add_generation_prompt=True,
                                enable_thinking=True)
    return tok(p + "<|channel>thought\n" + r["thinking"] + "<channel|>" + r["reply"], add_special_tokens=False).input_ids[:768]


fitwin = [[tok.bos_token_id] + wiki[i * 511:(i + 1) * 511] for i in range(0, 16, 2)] + \
         [[tok.bos_token_id] + chat[i * 511:(i + 1) * 511] for i in range(0, 32, 2)] + [mathwin(r) for r in rows[::14]]
test = [([tok.bos_token_id] + wiki[i * 511:(i + 1) * 511], "wiki") for i in range(16, 19)] + \
       [([tok.bos_token_id] + chat[i * 511:(i + 1) * 511], "chat") for i in range(32, 37)] + \
       [(mathwin(r), "math") for r in rows[3::42]]

# the receptors: the two norms per layer that read the bloodstream
NORMS = [(i, l.input_layernorm) for i, l in enumerate(layers)] + [(i, l.pre_feedforward_layernorm) for i, l in enumerate(layers)]
st = {"mode": "stats", "a": 0.0}
s1 = [torch.zeros(D, dtype=torch.float64) for _ in range(L)]
s2 = [torch.zeros(D, dtype=torch.float64) for _ in range(L)]
cnt = [0]
GI, REST = [0] * L, [0.0] * L
TONE = {}                                   # (norm, dose) -> average log change in volume, divided out when neutral=1


def stats_hook(i):
    def h(mod, args):
        x = args[0][0, 1:].float()
        s1[i] += x.sum(0).cpu().double(); s2[i] += x.pow(2).sum(0).cpu().double()
        if i == 0:
            cnt[0] += x.shape[0]
    return h


def logm(i, mod, x, a):
    g = x[..., GI[i]]
    g2 = REST[i] + (1 + a) * (g - REST[i])                          # the wobble as the receptor now "hears" it
    ms = x.pow(2).mean(-1) + mod.eps
    return 0.5 * (ms / (ms + (g2.pow(2) - g.pow(2)) / D).clamp_min(mod.eps)).log()


def pam_hook(i, k):
    def h(mod, args, o):
        a = st["a"]
        if st["mode"] == "tone":
            for d in DOSES:
                lm = logm(i, mod, args[0][0, 1:].float(), d * F[i])
                t = TONE.setdefault((k, d), [0.0, 0])
                t[0] += lm.sum().item(); t[1] += lm.numel()
            return None
        if st["mode"] != "run" or a == 0 or F[i] == 0:
            return None
        if MODE == "flat":
            return (o.float() * (1 + a * F[i])).to(o.dtype)
        if MODE == "shift":                                  # the other half of neutral: PAM = neutral gain x this
            t = TONE[(k, a)]
            return (o.float() * math.exp(t[0] / t[1])).to(o.dtype)
        lm = logm(i, mod, args[0].float(), a * F[i])
        if NEUTRAL:
            t = TONE[(k, a)]
            lm = lm - t[0] / t[1]
        return (o.float() * lm.exp()[..., None]).to(o.dtype)
    return h


hs = [layers[i].input_layernorm.register_forward_pre_hook(stats_hook(i)) for i in range(L)]
with torch.no_grad():
    for w in fitwin:
        model.model(input_ids=torch.tensor([w], device=DEV))
for h in hs:
    h.remove()
for i in range(L):
    m = s1[i] / cnt[0]
    rms = (s2[i] / cnt[0]).sqrt()
    GI[i] = int(rms.argmax())
    if MODE == "rand":
        loud = set(rms.argsort(descending=True)[:16].tolist())
        g = torch.Generator().manual_seed(20261027 + i)
        GI[i] = next(int(j) for j in torch.randperm(D, generator=g) if int(j) not in loud)
    REST[i] = float(m[GI[i]])
for k, (i, n) in enumerate(NORMS):
    n.register_forward_hook(pam_hook(i, k))
if NEUTRAL or MODE == "shift":
    st["mode"] = "tone"
    with torch.no_grad():
        for w in fitwin:
            model.model(input_ids=torch.tensor([w], device=DEV))
st["mode"] = "run"
print("giant per layer (index, resting level): " + " ".join(f"{i}:#{GI[i]}@{REST[i]:+.0f}" for i in range(0, L, 6)), flush=True)
print("dose factor per layer: " + " ".join(f"{f:+g}" for f in F) + (", volume kept neutral" if NEUTRAL else ""), flush=True)


def empty():
    if DEV == "cuda":
        torch.cuda.empty_cache()
    elif DEV == "mps":
        torch.mps.empty_cache()


def logp(w):
    return model(input_ids=torch.tensor([w], device=DEV)).logits[0, 1:].float().log_softmax(-1)


lines = [f"dosed layers {opt.get('zones', opt.get('layers', 'all'))}{', volume kept neutral' if NEUTRAL else ''}", f"mode {MODE}: " + {"pam": "PAM on the giant's gain signal: dose a = how much louder the receptor hears the giant's wobble",
          "flat": "flat gain control: every bloodstream norm's output times (1 + a)",
          "shift": "the PAM's own average volume change at each norm, applied as a constant (no wobble gain)",
          "rand": "random-number control: the PAM formula on a random ordinary number"}[MODE], ""]
kl = {}
with torch.no_grad():
    for w, t in test:
        st["a"] = 0.0
        b = logp(w)
        for a in DOSES:
            if a == 0:
                continue
            st["a"] = a
            lp = logp(w)
            kl.setdefault(a, {}).setdefault(t, []).append((b.exp() * (b - lp)).sum(-1).mean().item())
            del lp
        st["a"] = 0.0
        del b
        empty()
HS = int(opt.get("hs", 0))
if HS:
    hl = open(f"{BENCH}/hellaswag_val_full.txt").read().split("\n")
    hsq = [(hl[k], int(hl[k + 1]), hl[k + 2:k + 6]) for k in range(0, 6 * HS, 6)]
    hsok = {}
    with torch.no_grad():
        for a in [0.0] + [d for d in DOSES if d]:
            st["a"], right = a, []
            for ctx, lab, ends in hsq:
                c = [tok.bos_token_id] + tok(ctx, add_special_tokens=False).input_ids
                sc = []
                for e in ends:
                    w = c + tok(" " + e, add_special_tokens=False).input_ids
                    # logp() starts at the 2nd prediction (fine for KL, same positions both sides); here each ending
                    # word must be scored by the prediction made one step before it, so take the full row set
                    lp = model(input_ids=torch.tensor([w], device=DEV)).logits[0, len(c) - 1:-1].float().log_softmax(-1)
                    sc.append(lp.gather(-1, torch.tensor(w[len(c):], device=DEV)[:, None]).mean().item())
                right.append(int(np.argmax(sc)) == lab)
            hsok[a] = right
            empty()
    st["a"] = 0.0
lines.append(f"{'dose':>6} {'KL wiki':>8} {'KL chat':>8} {'KL math':>8}")
for a in DOSES:
    if a:
        lines.append(f"{a:>+6.2f} " + " ".join(f"{np.mean(kl[a][t]):>8.4f}" for t in ("wiki", "chat", "math")))
if HS:
    b = hsok[0.0]
    lines += ["", f"HellaSwag, {HS} questions: dose 0 = {100 * np.mean(b):.1f}% right"]
    for a in DOSES:
        if a:
            r = hsok[a]
            lines.append(f"  dose {a:+.3g}: {100 * np.mean(r):.1f}% right ({sum(x and not y for x, y in zip(r, b))} gained, "
                         f"{sum(y and not x for x, y in zip(r, b))} lost vs dose 0)")
print("\n".join(lines), flush=True)

if GEN:
    items = [json.loads(l) for l in open(probs)]
    pick, seen = [], {}
    for it in items:
        key = (it["cat"], it["kind"])
        if seen.get(key, 0) < int(opt.get("per", 2)):
            pick.append(it); seen[key] = seen.get(key, 0) + 1
    tok.padding_side = "left"
    TAIL = "\n\nEnd with a line: Answer: <your answer>" + (" " + opt["tail"].replace("_", " ") if "tail" in opt else "")
    tag = ("-tail" if "tail" in opt else "") + ("" if MODE == "pam" else "-" + MODE) + \
          (f"-L{opt['layers']}" if "layers" in opt else "") + (f"-Z{opt['zones'].replace(':', '_').replace(',', '+')}" if "zones" in opt else "") + ("-neutral" if NEUTRAL else "")
    for a in DOSES:
        st["a"] = a
        path = os.path.join(out, f"pam{a:+.2f}{tag}.jsonl")
        t = time.time()
        with open(path, "w") as f, torch.no_grad():
            for c0 in range(0, len(pick), 6):
                batch = pick[c0:c0 + 6]
                prompts = [tok.apply_chat_template([{"role": "user", "content": it["q"] + TAIL}], tokenize=False,
                                                   add_generation_prompt=True, enable_thinking=True) for it in batch]
                enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
                o = model.generate(**enc, max_new_tokens=int(opt.get("maxnew", 2500)), do_sample=False, pad_token_id=tok.pad_token_id)
                for it, seq in zip(batch, o[:, enc.input_ids.shape[1]:]):
                    text = tok.decode(seq, skip_special_tokens=False).replace("<pad>", "")
                    done = "<turn|>" in text or tok.eos_token in text
                    text = text.split("<turn|>")[0].replace(tok.eos_token, "")
                    think, _, reply = text.partition("<channel|>") if "<channel|>" in text else ("", "", text)
                    think = think.replace("<|channel>thought", "", 1).strip()
                    f.write(json.dumps({**it, "dose": a, "thinking": think, "reply": reply.strip(),
                                        "think_tokens": len(tok(think, add_special_tokens=False).input_ids),
                                        "reply_tokens": len(tok(reply, add_special_tokens=False).input_ids),
                                        "finish": "stop" if done else "length"}) + "\n")
                f.flush()
                empty()
        print(f"  dose {a:+.2f}{tag}: {len(pick)} problems in {time.time() - t:.0f} s", flush=True)
if CHAT:
    QUESTIONS = {
        "dayvigo": "Is it okay to have grapefruit with Dayvigo?",
        "day": "Hey, I just got home from a long shift. How was your day?",
        "dog": "Can I leave my dog in the car for ten minutes? It's about 30°C out.",
        "cheer": "My friend failed her driving test and feels terrible. Can you write her a short message to cheer her up?",
    }
    tok.padding_side = "left"
    for a in DOSES:
        st["a"] = a
        res = []
        with torch.no_grad():
            for qi, (key, q) in enumerate(QUESTIONS.items()):
                prompt = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
                enc = tok([prompt] * NCHAT, return_tensors="pt", add_special_tokens=False).to(DEV)
                torch.manual_seed(1000 + 100 * qi)
                o = model.generate(**enc, max_new_tokens=600, do_sample=True, temperature=1.0, top_k=64, top_p=0.95,
                                   pad_token_id=tok.pad_token_id)
                for j, seq in enumerate(o[:, enc.input_ids.shape[1]:]):
                    text = tok.decode(seq, skip_special_tokens=False).replace("<pad>", "")
                    done = "<turn|>" in text or tok.eos_token in text
                    text = text.split("<turn|>")[0].replace(tok.eos_token, "").strip()
                    res.append({"q": key, "n": j + 1, "dose": a, "reply": text, "finish": "stop" if done else "length",
                                "tokens": len(tok(text, add_special_tokens=False).input_ids)})
                empty()
        with open(os.path.join(out, f"pam{a:+.1f}.chat.jsonl"), "w") as f:
            for r in res:
                f.write(json.dumps(r) + "\n")
        md = [f"# PAM chat, dose {a:+.1f}", "", "Every reply in full; nothing scored here.", ""]
        for key, q in QUESTIONS.items():
            md += [f"## {key}", "", f"> {q}", ""]
            for r in (r for r in res if r["q"] == key):
                md += [f"### {key} {r['n']} ({r['tokens']} tokens{', CUT OFF' if r['finish'] == 'length' else ''})", "",
                       *("> " + l for l in r["reply"].splitlines()), ""]
        open(os.path.join(out, f"pam{a:+.1f}.chat.md"), "w").write("\n".join(md) + "\n")
        print(f"  chat dose {a:+.1f} done", flush=True)
open(os.path.join(out, f"pam-{MODE}" + (f"-L{opt['layers']}" if "layers" in opt else "") + (f"-Z{opt['zones'].replace(':', '_').replace(',', '+')}" if "zones" in opt else "") + ("-neutral" if NEUTRAL else "") + ".txt"), "w").write("\n".join(lines) + "\n")
print("PAM_DONE")
