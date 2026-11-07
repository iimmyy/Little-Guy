#!/usr/bin/env python3
"""Layer-by-layer healing (2026-10-27, after heal.py's LoRA heal of the all-layer 40% cut stalled at 0.63 wiki drift):
the kept neurons of every layer are retrained in full (not small add-ons) so the shrunk MLP reproduces what the whole
MLP produced, from the untouched model's own inputs (block-wise reconstruction, "physio per muscle"). Layers train in
groups so the optimizer fits on a 24 GB GPU; each group's result is stored as a small bf16 MLP and swapped in.

Drift (the standard test set) is measured after every group: trained layers use their healed MLP, the rest use the
plain cut. Writes <name>.log.jsonl and <name>.pt (per layer: kept neuron ids + healed gate/up/down, bf16).

usage: localheal.py <out folder> <math traces jsonl> <name> [cut=0.4 group=7 steps=1500 lr=1e-4 seq=512
                    extra=file1,file2 dev=cuda]
"""
import json, os, sys, time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from e4b_mac import load

out, traces, name = sys.argv[1:4]
opt = dict(a.split("=", 1) for a in sys.argv[4:])
CUT, GROUP, STEPS, LR = float(opt.get("cut", 0.4)), int(opt.get("group", 7)), int(opt.get("steps", 1500)), float(opt.get("lr", 1e-4))
SEQ, DEV = int(opt.get("seq", 512)), opt.get("dev", "cuda")
BENCH = os.path.join(os.path.dirname(__file__), "../../../.run/bench")
os.makedirs(out, exist_ok=True)

model, tok = load(DEV)
layers = model.model.layers
L, D, N = len(layers), model.config.hidden_size, layers[0].mlp.intermediate_size
K = int((1 - CUT) * N)
wiki = tok(open(f"{BENCH}/wiki.test.raw").read()[:600000], add_special_tokens=False).input_ids
chat = tok(open(f"{BENCH}/chat_heldout.txt").read(), add_special_tokens=False).input_ids
mix = tok(open(f"{BENCH}/calib_mix.txt").read(), add_special_tokens=False).input_ids
rows = [json.loads(l) for l in open(traces)]


def mathwin(r, n=768):
    p = tok.apply_chat_template([{"role": "user", "content": r["q"]}], tokenize=False, add_generation_prompt=True,
                                enable_thinking=True)
    return tok(p + "<|channel>thought\n" + r["thinking"] + "<channel|>" + r["reply"], add_special_tokens=False).input_ids[:n]


# same test set, fit windows and training pool as heal.py
test = [([tok.bos_token_id] + wiki[i * 511:(i + 1) * 511], "wiki") for i in range(16, 19)] + \
       [([tok.bos_token_id] + chat[i * 511:(i + 1) * 511], "chat") for i in range(32, 37)] + \
       [(mathwin(r), "math") for r in rows[3::42]]
test_math = set(range(3, len(rows), 42))
fitwin = [[tok.bos_token_id] + wiki[i * 511:(i + 1) * 511] for i in range(0, 16, 2)] + \
         [[tok.bos_token_id] + chat[i * 511:(i + 1) * 511] for i in range(0, 32, 2)] + [mathwin(r) for r in rows[::14]]
pool = [[tok.bos_token_id] + wiki[i * 511:(i + 1) * 511] for i in range(60, len(wiki) // 511)] + \
       [[tok.bos_token_id] + chat[i * 511:(i + 1) * 511] for i in range(60, len(chat) // 511)] + \
       [[tok.bos_token_id] + mix[i * 511:(i + 1) * 511] for i in range(len(mix) // 511)] + \
       [mathwin(r, SEQ) for j, r in enumerate(rows) if j not in test_math and j % 14]
for path in filter(None, opt.get("extra", "").split(",")):
    ids = tok(open(path).read(), add_special_tokens=False).input_ids
    pool += [[tok.bos_token_id] + ids[i * 511:(i + 1) * 511] for i in range(len(ids) // 511)]
rng = np.random.default_rng(20261027)
print(f"training pool: {len(pool)} windows", flush=True)


class Small(torch.nn.Module):
    """a shrunk Gemma MLP: only the kept neurons (same math as the original, fewer neurons)"""

    def __init__(self, mlp, keep, dtype):
        super().__init__()
        self.gate = torch.nn.Parameter(mlp.gate_proj.weight[keep].to(dtype).clone())
        self.up = torch.nn.Parameter(mlp.up_proj.weight[keep].to(dtype).clone())
        self.down = torch.nn.Parameter(mlp.down_proj.weight[:, keep].to(dtype).clone())

    def forward(self, x):
        x = x.to(self.gate.dtype)
        return F.linear(F.gelu(F.linear(x, self.gate), approximate="tanh") * F.linear(x, self.up), self.down)


dn2 = [l.mlp.down_proj.weight.float().pow(2).sum(0) for l in layers]
energy = [torch.zeros(N, dtype=torch.float64) for _ in range(L)]
st = {"mode": "stats", "cap": {}, "train": set()}
keep = [None] * L
small = [None] * L                                          # healed bf16 MLPs, swapped in for the student
mask = [None] * L


def mlp_pre(i):
    def h(mod, args):
        if st["mode"] == "teacher" and i in st["train"]:
            st["cap"][i] = args[0][0].detach()
    return h


def down_pre(i):
    def h(mod, args):
        p = args[0]
        if st["mode"] == "stats":
            energy[i] += (p[0, 1:].float().pow(2).sum(0) * dn2[i]).cpu().double()
        elif st["mode"] == "student" and small[i] is None:
            return (p * mask[i],)                           # not healed yet: the plain cut
        return None
    return h


def mlp_post(i):
    def h(mod, args, o):
        if st["mode"] == "teacher" and i in st["train"]:
            st["cap"][(i, "y")] = o[0].detach()
        elif st["mode"] == "student" and small[i] is not None:
            return small[i](args[0]).to(o.dtype)
        return None
    return h


for i, l in enumerate(layers):
    l.mlp.register_forward_pre_hook(mlp_pre(i))
    l.mlp.down_proj.register_forward_pre_hook(down_pre(i))
    l.mlp.register_forward_hook(mlp_post(i))


def empty():
    if DEV == "cuda":
        torch.cuda.empty_cache()
    elif DEV == "mps":
        torch.mps.empty_cache()


t0 = time.time()
with torch.no_grad():
    for w in fitwin:
        model.model(input_ids=torch.tensor([w], device=DEV))
for i in range(L):
    keep[i] = energy[i].argsort(descending=True)[:K].sort().values.to(DEV)
    m = torch.zeros(N, device=DEV, dtype=torch.bfloat16); m[keep[i]] = 1
    mask[i] = m
print(f"cut {CUT:.0%} in all {L} layers: {K} of {N} neurons kept ({time.time() - t0:.0f} s)", flush=True)


def drift():
    kl = {}
    with torch.no_grad():
        for w, t in test:
            ids = torch.tensor([w], device=DEV)
            st["mode"] = "off"
            b = model(input_ids=ids).logits[0, 1:].float().log_softmax(-1)
            st["mode"] = "student"
            s = model(input_ids=ids).logits[0, 1:].float().log_softmax(-1)
            kl.setdefault(t, []).append((b.exp() * (b - s)).sum(-1).mean().item())
            del b, s
            empty()
    st["mode"] = "off"
    return {t: float(np.mean(v)) for t, v in kl.items()}


log = open(os.path.join(out, f"{name}.log.jsonl"), "w")
d = drift()
print(f"plain cut: drift wiki {d['wiki']:.4f} chat {d['chat']:.4f} math {d['math']:.4f}", flush=True)
log.write(json.dumps({"healed": 0, "drift": d, "t": time.time() - t0}) + "\n"); log.flush()

for g0 in range(0, L, GROUP):
    G = list(range(g0, min(L, g0 + GROUP)))
    st["train"] = set(G)
    stu = {i: Small(layers[i].mlp, keep[i], torch.float32) for i in G}
    opts = {i: torch.optim.AdamW(stu[i].parameters(), lr=LR, weight_decay=0.0) for i in G}
    first, last = {}, {}
    for step in range(1, STEPS + 1):
        w = pool[rng.integers(len(pool))][:SEQ]
        st["mode"], st["cap"] = "teacher", {}
        with torch.no_grad():
            model.model(input_ids=torch.tensor([w], device=DEV))
        st["mode"] = "off"
        for i in G:
            x, y = st["cap"][i][1:].float(), st["cap"][(i, "y")][1:].float()
            loss = (stu[i](x) - y).pow(2).sum(-1).mean() / y.pow(2).sum(-1).mean()   # share of the MLP output missed
            for gr in opts[i].param_groups:
                gr["lr"] = LR * min(1.0, step / 50)
            opts[i].zero_grad(set_to_none=True)
            loss.backward()
            opts[i].step()
            first.setdefault(i, loss.item())
            last[i] = 0.98 * last.get(i, loss.item()) + 0.02 * loss.item()
        st["cap"] = {}
        if step % 250 == 0:
            print(f"  layers {G[0]}-{G[-1]} step {step}: missed share " +
                  " ".join(f"{i}:{last[i]:.3f}" for i in G) + f" ({time.time() - t0:.0f} s)", flush=True)
    for i in G:
        s = Small(layers[i].mlp, keep[i], torch.bfloat16)
        with torch.no_grad():
            s.gate.copy_(stu[i].gate); s.up.copy_(stu[i].up); s.down.copy_(stu[i].down)
        small[i] = s.requires_grad_(False)
    del stu, opts
    empty()
    d = drift()
    print(f"healed layers 0-{G[-1]}: drift wiki {d['wiki']:.4f} chat {d['chat']:.4f} math {d['math']:.4f} "
          f"(missed share per layer: start " + " ".join(f"{first[i]:.3f}" for i in G) + ", end " +
          " ".join(f"{last[i]:.3f}" for i in G) + ")", flush=True)
    log.write(json.dumps({"healed": G[-1] + 1, "drift": d, "first": first, "last": last, "t": time.time() - t0}) + "\n")
    log.flush()

torch.save({"cut": CUT, "keep": [k.cpu() for k in keep],
            "mlp": [{"gate": s.gate.cpu(), "up": s.up.cpu(), "down": s.down.cpu()} for s in small], "opt": opt},
           os.path.join(out, f"{name}.pt"))
print("LOCALHEAL_DONE")
