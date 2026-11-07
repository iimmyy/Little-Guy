#!/usr/bin/env python3
"""Step 1, LFM3 family. Reading only. For every write (operator, neurons, per layer) and every neighbouring pair, adds up W'W and W'W_next on real pages.
Also adds up each neuron's squared activity. Pages exclude the screen problems. usage: cancel_read.py <hf> <steps.jsonl> <out.pt> [pages=170 maxtok=4000 words=128]"""
import json, sys, time, random
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
args = [a for a in sys.argv[1:] if "=" not in a]; opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
hf, stepsf, out = args
PAGES, MAXTOK, WORDS = int(opt.get("pages", 170)), int(opt.get("maxtok", 4000)), int(opt.get("words", 128))
DEV = "cuda"
tok = AutoTokenizer.from_pretrained(hf, trust_remote_code=True)
m = AutoModelForCausalLM.from_pretrained(hf, dtype=torch.bfloat16, trust_remote_code=True).to(DEV).eval().requires_grad_(False)
LS = m.model.layers; N = len(LS); D = m.config.hidden_size
IDX, got, hact = {}, {}, {}
def save(name):
    def f(mod, a, o):
        got[name] = o[0][IDX["i"]].float()
    return f
def saveh(L):
    def f(mod, a):
        hact[L] = a[0][0][IDX["i"]].float()
    return f
for L, b in enumerate(LS):
    b.post_operator_norm.register_forward_hook(save(f"op{L}"))
    b.post_ffn_norm.register_forward_hook(save(f"ffn{L}"))
    b.feed_forward.w2.register_forward_pre_hook(saveh(L))
order = [n for L in range(N) for n in (f"op{L}", f"ffn{L}")]
screen = set()
for f in ("screen-sci-40.jsonl", "screen-math-40.jsonl"):
    screen |= {json.loads(l)["q"] for l in open(f)}
best = {}
for l in open(stepsf):
    j = json.loads(l)
    if j["q"] in screen:
        continue
    k = (j["id"], j.get("run"))
    if k not in best or len(j["pre"]) > len(best[k]["pre"]):
        best[k] = j
byq = {}
for j in best.values():
    if j["q"] not in byq or len(j["pre"]) > len(byq[j["q"]]["pre"]):
        byq[j["q"]] = j
J = sorted(byq.values(), key=lambda j: j["id"])
random.Random(1107).shuffle(J)
nheld = max(30, len(J) // 5)
heldq, fit = [j["q"] for j in J[:nheld]], J[nheld:]
json.dump(dict(held=[dict(q=j["q"], pre=j["pre"], id=j["id"]) for j in J[:nheld]]), open(out + ".held.json", "w"))
print("problems", len(J), "fit", len(fit), "held", nheld, flush=True)
head = lambda q: tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=True) + "\n"
A = torch.zeros(len(order), D, D, dtype=torch.float64, device=DEV)
C = torch.zeros(len(order) - 1, D, D, dtype=torch.float64, device=DEV)
H = torch.zeros(N, m.config.intermediate_size if hasattr(m.config, "intermediate_size") else LS[0].feed_forward.w2.in_features, dtype=torch.float64, device=DEV)
n, done, t0 = 0, 0, time.time()
for j in fit:
    if done >= PAGES:
        break
    h = head(j["q"])
    ids = tok(h + j["pre"], add_special_tokens=False).input_ids[:MAXTOK]
    first = len(tok(h, add_special_tokens=False).input_ids)
    if len(ids) - first < 40:
        continue
    rng = np.random.default_rng(len(ids) + done)
    think = rng.choice(np.arange(first, len(ids) - 1), size=min(WORDS, len(ids) - 1 - first), replace=False)
    pos = sorted(set(range(0, first)) | set(think.tolist()))
    IDX["i"] = torch.tensor(pos, device=DEV)
    try:
        with torch.inference_mode():
            m.model(input_ids=torch.tensor([ids], device=DEV))
            for k, nm in enumerate(order):
                W = got[nm]
                A[k] += (W.T @ W).double()
                if k + 1 < len(order):
                    C[k] += (W.T @ got[order[k + 1]]).double()
            for L in range(N):
                H[L] += hact[L].pow(2).sum(0).double()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache(); print("oom skip", len(ids), flush=True); continue
    n += len(pos); done += 1
    if done % 25 == 0:
        print(done, "pages", n, "words", round(time.time() - t0), "s", flush=True)
A, C, H = (A / n).float().cpu(), (C / n).float().cpu(), (H / n).float().cpu()
torch.save(dict(A=A, C=C, H=H, order=order, types=list(m.config.layer_types), n=n, pages=done), out)
print("CANCEL_READ_DONE", done, "pages", n, "words", round(time.time() - t0), "s", flush=True)
