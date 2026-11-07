#!/usr/bin/env python3
"""Cut + heal (Immy, 2026-10-27: "minimal training: do x during a little training, get x result"). E4B with the
weakest CUT share of neurons in layers 0-12 deleted for good (lean.py's ranking), healed by a short distillation:
the cut model learns to match the untouched E4B's next-word odds (forward KL on the logits, Minitron's recipe).

Only small add-ons train; everything else is frozen:
  LoRA (rank R) on gate/up/down of layers 0-12      (merged into the weights at export)
  a rank-LR straight line per cut layer, out += (x U) V + b   (can be stored as 2*LR plain Gemma neurons, lean.py)
The race: line starts EMPTY (U small random, V = 0, b = 0) vs line starts from the least-squares FIT (lean.py's
reduced-rank line). Same data, same steps, same everything else. Drift on the standard test set every EVAL steps.

The teacher is the same loaded model with the cut and the add-ons switched off, so only one copy is in memory.

usage: heal.py <out folder> <math traces jsonl> <name> [init=fit|empty cut=0.3 steps=1000 lr=2e-4 rank=32 lrank=256
               seq=512 eval=100 dev=mps layers=0-12 extra=file1,file2]
  lrank=0   no line at all (the capacity goes to LoRA instead)
  extra     more training text files (plain text or E4B's own chat-formatted answers), cut into windows
"""
import json, os, sys, time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from e4b_mac import load

out, traces, name = sys.argv[1:4]
opt = dict(a.split("=", 1) for a in sys.argv[4:])
INIT, CUT = opt.get("init", "fit"), float(opt.get("cut", 0.3))
STEPS, LR, R, LRANK = int(opt.get("steps", 1000)), float(opt.get("lr", 2e-4)), int(opt.get("rank", 32)), int(opt.get("lrank", 256))
SEQ, EVAL, DEV = int(opt.get("seq", 512)), int(opt.get("eval", 100)), opt.get("dev", "mps")
a, b = map(int, opt.get("layers", "0-12").split("-"))
LAYERS = list(range(a, b + 1))
BENCH = os.path.join(os.path.dirname(__file__), "../../../.run/bench")
os.makedirs(out, exist_ok=True)

model, tok = load(DEV)
for p in model.parameters():
    p.requires_grad_(False)
layers = model.model.layers
L, D, N = len(layers), model.config.hidden_size, layers[0].mlp.intermediate_size
wiki = tok(open(f"{BENCH}/wiki.test.raw").read()[:600000], add_special_tokens=False).input_ids
chat = tok(open(f"{BENCH}/chat_heldout.txt").read(), add_special_tokens=False).input_ids
mix = tok(open(f"{BENCH}/calib_mix.txt").read(), add_special_tokens=False).input_ids
rows = [json.loads(l) for l in open(traces)]


def mathwin(r, n=768):
    p = tok.apply_chat_template([{"role": "user", "content": r["q"]}], tokenize=False, add_generation_prompt=True,
                                enable_thinking=True)
    return tok(p + "<|channel>thought\n" + r["thinking"] + "<channel|>" + r["reply"], add_special_tokens=False).input_ids[:n]


# the standard test set (same windows as gate_line / lean / moe_check); training never touches these
test = [([tok.bos_token_id] + wiki[i * 511:(i + 1) * 511], "wiki") for i in range(16, 19)] + \
       [([tok.bos_token_id] + chat[i * 511:(i + 1) * 511], "chat") for i in range(32, 37)] + \
       [(mathwin(r), "math") for r in rows[3::42]]
test_math = set(range(3, len(rows), 42))
fitwin = [[tok.bos_token_id] + wiki[i * 511:(i + 1) * 511] for i in range(0, 16, 2)] + \
         [[tok.bos_token_id] + chat[i * 511:(i + 1) * 511] for i in range(0, 32, 2)] + [mathwin(r) for r in rows[::14]]
# training pool: wiki and chat beyond every test/probe window, the calibration mix, math traces not in the test set
pool = [[tok.bos_token_id] + wiki[i * 511:(i + 1) * 511] for i in range(60, len(wiki) // 511)] + \
       [[tok.bos_token_id] + chat[i * 511:(i + 1) * 511] for i in range(60, len(chat) // 511)] + \
       [[tok.bos_token_id] + mix[i * 511:(i + 1) * 511] for i in range(len(mix) // 511)] + \
       [mathwin(r, SEQ) for j, r in enumerate(rows) if j not in test_math and j % 14]
for path in filter(None, opt.get("extra", "").split(",")):
    ids = tok(open(path).read(), add_special_tokens=False).input_ids
    pool += [[tok.bos_token_id] + ids[i * 511:(i + 1) * 511] for i in range(len(ids) // 511)]
rng = np.random.default_rng(20261027)
print(f"training pool: {len(pool)} windows", flush=True)

dn2 = [l.mlp.down_proj.weight.float().pow(2).sum(0) for l in layers]
st = {"mode": "stats", "student": False, "x": {}}
energy = {i: torch.zeros(N, dtype=torch.float64) for i in LAYERS}
xtx = {i: torch.zeros(D + 1, D + 1, dtype=torch.float64) for i in LAYERS} if LRANK and INIT == "fit" else {}
xtr = {i: torch.zeros(D + 1, D, dtype=torch.float64) for i in LAYERS} if LRANK and INIT == "fit" else {}
mask = {}


def aug(x):
    return torch.cat([x, torch.ones(x.shape[0], 1, device=x.device)], 1).cpu().double()


class Lora(torch.nn.Module):
    def __init__(self, din, dout):
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn(din, R, device=DEV) / din ** 0.5)
        self.b = torch.nn.Parameter(torch.zeros(R, dout, device=DEV))

    def forward(self, x):
        return (x.float() @ self.a) @ self.b


class Line(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.u = torch.nn.Parameter(torch.randn(D, LRANK, device=DEV) / D ** 0.5 * 0.01)
        self.v = torch.nn.Parameter(torch.zeros(LRANK, D, device=DEV))
        self.c = torch.nn.Parameter(torch.zeros(D, device=DEV))

    def forward(self, x):
        return (x.float() @ self.u) @ self.v + self.c


adds = torch.nn.ModuleDict()
for i in LAYERS:
    m = layers[i].mlp
    adds[f"{i}g"] = Lora(D, N); adds[f"{i}u"] = Lora(D, N); adds[f"{i}d"] = Lora(N, D)
    if LRANK:
        adds[f"{i}l"] = Line()


def lora_hook(key):
    def h(mod, args, o):
        if not st["student"]:
            return None
        return o + adds[key](args[0]).to(o.dtype)
    return h


def mlp_pre(i):
    def h(mod, args):
        st["x"][i] = args[0]
    return h


def down_pre(i):
    def h(mod, args):
        p = args[0]
        if st["mode"] == "stats":
            x = st["x"][i][0, 1:].float()
            energy[i] += (p[0, 1:].float().pow(2).sum(0) * dn2[i]).cpu().double()
            if xtx:
                xa = aug(x)
                xtx[i] += xa.T @ xa
            return None
        if st["mode"] == "fit":
            r = ((p * (1 - mask[i])).float() @ mod.weight.float().T)[0, 1:].cpu().double()
            xtr[i] += aug(st["x"][i][0, 1:].float()).T @ r
            return None
        if st["student"]:
            return (p * mask[i],)
        return None
    return h


def mlp_post(i):
    def h(mod, args, o):
        if not st["student"] or not LRANK:
            return None
        return o + adds[f"{i}l"](args[0]).to(o.dtype)
    return h


for i in LAYERS:
    m = layers[i].mlp
    m.register_forward_pre_hook(mlp_pre(i))
    m.down_proj.register_forward_pre_hook(down_pre(i))
    m.register_forward_hook(mlp_post(i))
    m.gate_proj.register_forward_hook(lora_hook(f"{i}g"))
    m.up_proj.register_forward_hook(lora_hook(f"{i}u"))
    m.down_proj.register_forward_hook(lora_hook(f"{i}d"))


def sweep(mode, windows):
    st["mode"] = mode
    with torch.no_grad():
        for w in windows:
            model.model(input_ids=torch.tensor([w], device=DEV))
            if DEV == "mps":
                torch.mps.empty_cache()


t0 = time.time()
sweep("stats", fitwin)
for i in LAYERS:
    keep = energy[i].argsort(descending=True)[:int((1 - CUT) * N)]
    m = torch.zeros(N, device=DEV, dtype=torch.bfloat16); m[keep] = 1
    mask[i] = m
if INIT == "fit" and LRANK:
    sweep("fit", fitwin)
    for i in LAYERS:
        reg = 1e-4 * xtx[i].diagonal()[:D].mean() * torch.eye(D + 1, dtype=torch.float64); reg[D, D] = 0
        Wb = torch.linalg.solve(xtx[i] + reg, xtr[i])
        C = Wb.T @ xtx[i] @ Wb
        P = torch.linalg.eigh(C)[1][:, -LRANK:]                    # the LRANK output directions used most
        ln = adds[f"{i}l"]
        with torch.no_grad():
            ln.u.copy_((Wb[:D] @ P).float()); ln.v.copy_(P.T.float()); ln.c.copy_(Wb[D].float())
del xtx, xtr
st["mode"] = "train"
print(f"cut {CUT:.0%} of neurons in layers {LAYERS[0]}-{LAYERS[-1]}, LoRA rank {R}, line rank {LRANK} init {INIT}, "
      f"{sum(p.numel() for p in adds.parameters()) / 1e6:.0f}M trainable ({time.time() - t0:.0f} s)", flush=True)


def drift():
    kl = {}
    with torch.no_grad():
        for w, t in test:
            ids = torch.tensor([w], device=DEV)
            st["student"] = False
            b = model(input_ids=ids).logits[0, 1:].float().log_softmax(-1)
            st["student"] = True
            s = model(input_ids=ids).logits[0, 1:].float().log_softmax(-1)
            kl.setdefault(t, []).append((b.exp() * (b - s)).sum(-1).mean().item())
            del b, s
            if DEV == "mps":
                torch.mps.empty_cache()
            elif DEV == "cuda":
                torch.cuda.empty_cache()
    st["student"] = False
    return {t: float(np.mean(v)) for t, v in kl.items()}


opt_ = torch.optim.AdamW(adds.parameters(), lr=LR, weight_decay=0.0)
log = open(os.path.join(out, f"{name}.log.jsonl"), "w")
d = drift()
print(f"step 0: drift wiki {d['wiki']:.4f} chat {d['chat']:.4f} math {d['math']:.4f}", flush=True)
log.write(json.dumps({"step": 0, "drift": d, "t": time.time() - t0}) + "\n"); log.flush()
for step in range(1, STEPS + 1):
    w = pool[rng.integers(len(pool))][:SEQ]
    ids = torch.tensor([w], device=DEV)
    with torch.no_grad():
        st["student"] = False
        tl = model(input_ids=ids).logits[0, 1:].float().log_softmax(-1)
    st["student"] = True
    sl = model(input_ids=ids).logits[0, 1:].float().log_softmax(-1)
    loss = (tl.exp() * (tl - sl)).sum(-1).mean()
    opt_.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(adds.parameters(), 1.0)
    for g in opt_.param_groups:
        g["lr"] = LR * min(1.0, step / 50)
    opt_.step()
    st["student"] = False
    del tl, sl
    if DEV == "mps":
        torch.mps.empty_cache()
    if step % 10 == 0:
        print(f"  step {step}: loss {loss.item():.4f} ({time.time() - t0:.0f} s)", flush=True)
    if step % EVAL == 0 or step == STEPS:
        d = drift()
        print(f"step {step}: drift wiki {d['wiki']:.4f} chat {d['chat']:.4f} math {d['math']:.4f}", flush=True)
        log.write(json.dumps({"step": step, "loss": loss.item(), "drift": d, "t": time.time() - t0}) + "\n"); log.flush()
torch.save({"adds": adds.state_dict(), "mask": {i: m.cpu() for i, m in mask.items()}, "opt": opt},
           os.path.join(out, f"{name}.pt"))
print("HEAL_DONE")
