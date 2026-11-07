#!/usr/bin/env python3
"""For every layer, E[v v'] with v = h/rms(y): what each neuron really sends, per word. With this, any neuron cut can be re-solved exactly with no more reading.
usage: gram_read.py <hf> <steps.jsonl> <read.pt> <out.pt> [pages=171 words=1000] [base weights can be replaced with state=<pt> for a fixed model]"""
import json, sys, time, random
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
args = [a for a in sys.argv[1:] if "=" not in a]; opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
hf, stepsf, readf, out = args; PAGES, WORDS = int(opt.get("pages", 171)), int(opt.get("words", 1000))
DEV = "cuda"
tok = AutoTokenizer.from_pretrained(hf, trust_remote_code=True)
m = AutoModelForCausalLM.from_pretrained(hf, dtype=torch.bfloat16, trust_remote_code=True).to(DEV).eval().requires_grad_(False)
if "state" in opt:
    print(m.load_state_dict(torch.load(opt["state"]), strict=False), flush=True)
LS = m.model.layers; NL = len(LS); IDX = {}; NEU = {}; RMS = {}
for L, b in enumerate(LS):
    b.feed_forward.w2.register_forward_pre_hook(lambda mod, a, L=L: NEU.__setitem__(L, a[0][0][IDX["i"]].float()))
    b.post_ffn_norm.register_forward_pre_hook(lambda mod, a, L=L: RMS.__setitem__(L, a[0][0][IDX["i"]].float().pow(2).mean(-1, keepdim=True).sqrt()))
held = {j["q"] for j in json.load(open(readf + ".held.json"))["held"]}
screen = set()
for f in ("screen-sci-40.jsonl", "screen-math-40.jsonl"):
    screen |= {json.loads(l)["q"] for l in open(f)}
byq = {}
for l in open(stepsf):
    j = json.loads(l)
    if j["q"] in screen or j["q"] in held:
        continue
    if j["q"] not in byq or len(j["pre"]) > len(byq[j["q"]]["pre"]):
        byq[j["q"]] = j
J = sorted(byq.values(), key=lambda j: j["id"]); random.Random(1107).shuffle(J)
head = lambda q: tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=True) + "\n"
n = LS[0].feed_forward.w2.in_features
G = torch.zeros(NL, n, n, dtype=torch.float64, device=DEV); mu = torch.zeros(NL, n, dtype=torch.float64, device=DEV)
cnt, done, t0 = 0, 0, time.time()
for j in J[:PAGES]:
    h = head(j["q"]); ids = tok(h + j["pre"], add_special_tokens=False).input_ids[:4000]; first = len(tok(h, add_special_tokens=False).input_ids)
    if len(ids) - first < 40:
        continue
    rng = np.random.default_rng(len(ids) + done)
    think = rng.choice(np.arange(first, len(ids) - 1), size=min(WORDS, len(ids) - 1 - first), replace=False)
    IDX["i"] = torch.tensor(sorted(set(range(first)) | set(think.tolist())), device=DEV)
    try:
        with torch.inference_mode():
            m.model(input_ids=torch.tensor([ids], device=DEV))
            for L in range(NL):
                v = NEU[L] / RMS[L]
                G[L] += (v.T @ v).double(); mu[L] += v.double().sum(0)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache(); print("oom skip", len(ids), flush=True); continue
    cnt += len(IDX["i"]); done += 1
    if done % 20 == 0:
        print(done, "pages", cnt, "words", round(time.time() - t0), "s", flush=True)
G /= cnt; mu /= cnt
Gc = torch.stack([G[L].float().cpu() for L in range(NL)])
torch.save(dict(G=Gc, mu=mu.float().cpu(), n=cnt, pages=done), out)
print("GRAM_READ_DONE", done, cnt, round(time.time() - t0), "s", flush=True)
