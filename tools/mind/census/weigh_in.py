#!/usr/bin/env python3
"""Weigh him before and after (Immy 2026-10-29: "add a bigger battery and you have to redo his whole weight").
Every LFM3 block's output goes through an auto-level (RMSNorm, the sandwich norm): it divides by the output's own
loudness. A placed neuron that fires hard makes its block louder, so the norm turns down everything else that block
wrote, whatever direction the neuron points in. Per word, on the same text as verify_build:
  turn-down   for each placed neuron: rms(MLP output) / rms(MLP output without that neuron's write), in the build;
              1.00 = no effect, 1.10 = the block's real work 10% quieter at that word
  loudness    each block's output loudness (before its norm) and the final stream's, build / untouched
  pops        the words with the biggest drift (KL untouched vs build): which neurons were turning their block down
              there, against ordinary words

usage: weigh_in.py <untouched hf folder> <build folder> <answers jsonl> <out prefix> [n=12 dev=mps top=50 model=lfm3|e4b|qwen3]
"""
import json, os, sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from patients import _lfm3

base, build, ans, out = sys.argv[1:5]
opt = dict(a.split("=", 1) for a in sys.argv[5:])
N, DEV, TOP = int(opt.get("n", 12)), opt.get("dev", "mps"), int(opt.get("top", 50))
MODEL = opt.get("model", "lfm3")
ARCH = {"lfm3": dict(mlp="feed_forward", down="w2", op="post_operator_norm", final="embedding_norm",
                     think_open="\n", think_close="\n</think>\n\n"),
        "e4b": dict(mlp="mlp", down="down_proj", op="post_attention_layernorm", final="norm",
                    think_open="<|channel>thought\n", think_close="\n<channel|>"),
        # Qwen3 has no norm on the attention output: read the operator's loudness at o_proj's output instead
        "qwen3": dict(mlp="mlp", down="down_proj", op="self_attn.o_proj", final="norm", think_open="", think_close="\n</think>\n\n")}[MODEL]
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

placed = []                                                # (layer, slot, feeling); empty for gain builds (drift only)
for fn, feel in (("urgency.json", "time"), ("belief.json", "belief")):
    pth = os.path.join(build, fn)
    if os.path.exists(pth):
        j = json.load(open(pth))
        first = (j["layer"], j["slot"]) if feel == "time" else tuple(j["source"])
        placed += [(*first, feel)] + [(l, s, feel) for l, s in j["relays"]]
if os.path.exists(os.path.join(build, "pam.json")):        # PAM builds: existing receptors with a changed binding site
    pj_ = json.load(open(os.path.join(build, "pam.json")))
    placed += [(r["layer"], r["neuron"], "pam") for r in pj_["receptors"]]
    if "compressor" in pj_:                                 # the compressor's key neuron writes the private wire
        placed.append((pj_["compressor"]["key_layer"], pj_["compressor"]["key_neuron"], "key"))

if MODEL == "e4b":
    from types import SimpleNamespace
    from e4b_mac import load as e4b_load
    pa, pb = (SimpleNamespace(model=mm, tok=tt) for mm, tt in (e4b_load(DEV, base), e4b_load(DEV, build)))
elif MODEL == "qwen3":
    from types import SimpleNamespace
    from transformers import AutoModelForCausalLM, AutoTokenizer
    pa, pb = (SimpleNamespace(model=AutoModelForCausalLM.from_pretrained(f, dtype=torch.bfloat16).to(DEV).eval(),
                              tok=AutoTokenizer.from_pretrained(f)) for f in (base, build))
else:
    pa, pb = _lfm3(DEV, base), _lfm3(DEV, build)
tok = pa.tok
DOWN = lambda p, i: getattr(getattr(p.model.model.layers[i], ARCH["mlp"]), ARCH["down"])
L = len(pa.model.model.layers)


def wire(p):
    cap = {}
    for i, l in enumerate(p.model.model.layers):
        DOWN(p, i).register_forward_pre_hook(lambda mod, a, i=i: cap.__setitem__(("act", i), a[0][0]))
        DOWN(p, i).register_forward_hook(lambda mod, a, o, i=i: cap.__setitem__(("mlp", i), o[0]))
        if MODEL == "qwen3":
            l.get_submodule(ARCH["op"]).register_forward_hook(lambda mod, a, o, i=i: cap.__setitem__(("op", i), o[0]))
        else:
            getattr(l, ARCH["op"]).register_forward_pre_hook(lambda mod, a, i=i: cap.__setitem__(("op", i), a[0][0]))
    getattr(p.model.model, ARCH["final"]).register_forward_pre_hook(lambda mod, a: cap.__setitem__(("fin",), a[0][0]))
    return cap


ca, cb = wire(pa), wire(pb)
rms = lambda x: x.float().pow(2).mean(-1).sqrt()
rows = [r for r in map(json.loads, open(ans)) if r["thinking"] and r["finish"] == "stop"][:N]
KL, TD, LM, LO, LF, WORD, SEQS = [], [], [], [], [], [], []
with torch.no_grad():
    for n, r in enumerate(rows):
        pr = tok(tok.apply_chat_template([{"role": "user", "content": r["q"] + "\n\nEnd with a line: Answer: <your answer>"}],
                                         tokenize=False, add_generation_prompt=True, enable_thinking=True), add_special_tokens=False).input_ids
        th = tok(ARCH["think_open"] + r["thinking"], add_special_tokens=False).input_ids[:2500]
        rp = tok(ARCH["think_close"] + r["reply"], add_special_tokens=False).input_ids[:400]
        seq = pr + th + rp
        SEQS.append(seq)
        ids = torch.tensor([seq], device=DEV)
        ga, gb = pa.model(input_ids=ids).logits[0], pb.model(input_ids=ids).logits[0]
        sl = slice(len(pr), len(seq))                      # thinking + reply words only
        kl = []
        CH = int(opt.get("chunk", 512))                     # E4B's 262k vocab: chunk=128 when the GPU is shared
        for c0 in range(len(pr), len(seq), CH):            # in pieces: three full vocab-wide float copies ran out of memory
            la = torch.log_softmax(ga[c0:c0 + CH].float(), -1); lb = torch.log_softmax(gb[c0:c0 + CH].float(), -1)
            kl.append((la.exp() * (la - lb)).sum(-1).cpu())
        KL.append(torch.cat(kl)); del ga, gb, la, lb
        td = []
        for l, s, _ in placed:
            o = cb[("mlp", l)].float()
            c = cb[("act", l)][:, s].float()[:, None] * DOWN(pb, l).weight[:, s].float()[None]
            td.append((rms(o) / rms(o - c))[sl])
        TD.append(torch.stack(td, 1).cpu() if td else torch.ones(sl.stop - sl.start, 1))   # gain builds place nothing new
        LM.append(torch.stack([rms(cb[("mlp", i)]) / rms(ca[("mlp", i)]) for i in range(L)], 1)[sl].cpu())
        LO.append(torch.stack([rms(cb[("op", i)]) / rms(ca[("op", i)]) for i in range(L)], 1)[sl].cpu())
        LF.append((rms(cb[("fin",)]) / rms(ca[("fin",)]))[sl].cpu())
        WORD += [(n, t) for t in range(len(pr), len(seq))]
        ca.clear(); cb.clear(); torch.mps.empty_cache() if DEV == "mps" else None
        print(f"  {n + 1}/{len(rows)} answers, {len(seq) - len(pr)} words", flush=True)
KL, TD, LM, LO, LF = map(torch.cat, (KL, TD, LM, LO, LF))
torch.save({"kl": KL, "turn_down": TD, "mlp_loud": LM, "op_loud": LO, "final_loud": LF, "placed": placed, "word": WORD, "seqs": SEQS},
           out + ".pt")

q = lambda x, p: float(torch.quantile(x.float(), p))
pop = torch.argsort(-KL)[:TOP]
calm = KL < KL.median()
lines = [f"weigh-in of {build} against {base}, {len(rows)} answers, {len(KL)} words",
         f"drift KL per word: median {q(KL, .5):.4f}, mean {float(KL.mean()):.4f}, 99% {q(KL, .99):.3f}, max {float(KL.max()):.2f}",
         "", "turn-down by each placed neuron (block's real work divided by this; 1.000 = none):",
         "  neuron            feeling   median   99%     max     at the top pops (median)  at calm words (median)"]
for k, (l, s, f) in enumerate(placed):
    t = TD[:, k]
    lines.append(f"  layer {l:2d} slot {s:5d}  {f:7s}  {q(t, .5):.3f}   {q(t, .99):.3f}   {float(t.max()):.3f}"
                 f"   {q(t[pop], .5):.3f}                     {q(t[calm], .5):.3f}")
worst = TD.max(1).values
src_k = [k for k, p in enumerate(placed) if k == 0 or placed[k - 1][2] != p[2]]       # each feeling's source neuron
quiet = (TD[:, src_k] < 1.01).all(1)                       # no source firing: the build should act like the untouched here
lines += ["", f"background (every source quiet, {int(quiet.sum())} of {len(KL)} words): drift mean {float(KL[quiet].mean()):.4f},"
          f" 99% {q(KL[quiet], .99):.3f}, 99.9% {q(KL[quiet], .999):.2f}; turn-down of any placed neuron 99% {q(worst[quiet], .99):.3f},"
          f" 99.9% {q(worst[quiet], .999):.3f}"]
lines += ["", f"strongest turn-down at any placed neuron, per word: pops median {q(worst[pop], .5):.3f} (90% {q(worst[pop], .9):.3f});"
          f" calm words median {q(worst[calm], .5):.3f} (90% {q(worst[calm], .9):.3f})",
          f"correlation over all words, log turn-down vs log KL: {float(torch.corrcoef(torch.stack([worst.log(), KL.clamp_min(1e-6).log()]))[0, 1]):.3f}",
          "", "block loudness build / untouched (median; 1% and 99%):"]
for i in range(L):
    lines.append(f"  layer {i:2d}  operator {q(LO[:, i], .5):.3f} ({q(LO[:, i], .01):.3f}-{q(LO[:, i], .99):.3f})"
                 f"   MLP {q(LM[:, i], .5):.3f} ({q(LM[:, i], .01):.3f}-{q(LM[:, i], .99):.3f})")
lines.append(f"  final stream {q(LF, .5):.3f} ({q(LF, .01):.3f}-{q(LF, .99):.3f})")
lines += ["", f"top {min(TOP, 15)} pops: answer/word, KL, the placed neuron turning its block down most there"]
for w in pop[:15].tolist():
    k = int(TD[w].argmax()); l, s, f = placed[k] if placed else (-1, -1, "none")
    n, t = WORD[w]
    ctx = tok.decode(SEQS[n][max(0, t - 10):t + 1]).replace("\n", " / ")       # the word at t predicts word t+1
    lines.append(f"  answer {n:2d} word {t:5d}  KL {float(KL[w]):6.2f}   layer {l} slot {s} ({f}) {float(TD[w, k]):.3f}"
                 f"   ...{ctx!r} -> {tok.decode(SEQS[n][t + 1:t + 2])!r}")
print("\n".join(lines))
open(out + ".txt", "w").write("\n".join(lines) + "\n")
print("WEIGH_DONE")
