#!/usr/bin/env python3
"""Does our E4B's prediction machinery, with its word ends stripped, help predict a life?

The lives are the life model's (tools/mind/life/live.py: the chemistry engine as his body, a year of varied days
each). Here they are read in 10-minute steps, two days at a time, and every arm predicts his body 10 and 60 minutes
ahead, scored as in the life model: surprise in units of 'nothing changes', skill = 1 - surprise.

  tokyo     our E4B's 42 rooms (gemma_body.py), frozen. Only the two translators learn: one writes each 10 minutes
            of his life into the rooms' 2,560-number code (and into the per-layer slot where the word table was),
            one reads the rooms' final code out as his body ahead. (Lu et al. 2021 did this with GPT-2.)
  tokyo_random  the control: the same Tokyo with every matrix replaced by random numbers of the same spread, so
            any edge Tokyo has over it comes from what training put in the rooms, not from their shape.
  gru       the life model's kind of brain (a GRU, 256 wide), learning everything, on the same windows.
Same windows, same order, same steps for every arm. The last life is never learned from: it is the test.

usage: life_on_body.py <e4b-grid4.gguf> <lives folder> <out folder> <steps> <seeds, e.g. 0,1,2> [arms...]
An arm can carry its own learning rate: tokyo@3e-4 (default 1e-3, which suits the GRU).
A seed sets the order the windows come in and the translators' starting numbers; the held-out windows never change.
"""
import json, math, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "life"))
from gemma_body import Body
from gemma_stream import D, N_LAYER, PER, rms
from learn import load

STEP = 10                                   # minutes per step
WIN = 288                                   # steps per window: two days
H = [1, 6]                                  # steps ahead: 10 and 60 minutes
BATCH, LR = 4, 1e-3


def coarse(lives_body, lives_world):
    """minute lives -> 10-minute steps: the body as it is at the end of each step, the world as what share of the
    step each thing was happening"""
    T = lives_body.shape[1] // STEP * STEP
    b = lives_body[:, STEP - 1:T:STEP]
    w = lives_world[:, :T].reshape(lives_world.shape[0], -1, STEP, lives_world.shape[2]).mean(2)
    return b, w


class Translators(torch.nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.inp = torch.nn.Linear(n_in, D)
        self.gain = torch.nn.Parameter(torch.ones(1))             # Gemma's entry code is ~1 per number
        self.per = torch.nn.Linear(n_in, N_LAYER * PER)            # the word table's slot, starts empty
        torch.nn.init.zeros_(self.per.weight)
        torch.nn.init.zeros_(self.per.bias)
        self.out = torch.nn.Linear(D, n_out)                      # starts at zero: "nothing changes", skill 0
        torch.nn.init.zeros_(self.out.weight)
        torch.nn.init.zeros_(self.out.bias)


class OnBody(torch.nn.Module):
    def __init__(self, body, n_in, n_out):
        super().__init__()
        self.body, self.t = body, Translators(n_in, n_out)

    def forward(self, x):
        B, T, _ = x.shape
        code = rms(self.t.inp(x)) * self.t.gain
        return self.t.out(self.body(code, self.t.per(x).view(B, T, N_LAYER, PER)))


class GRU(torch.nn.Module):
    def __init__(self, n_in, n_out, hidden=256):
        super().__init__()
        self.core, self.head = torch.nn.GRU(n_in, hidden, batch_first=True), torch.nn.Linear(hidden, n_out)

    def forward(self, x):
        return self.head(self.core(x)[0])


def main():
    gguf, src, out, steps = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    seeds = [int(x) for x in sys.argv[5].split(",")]
    arms = sys.argv[6:] or ["gru", "tokyo"]
    os.makedirs(out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    meta, body, world = load(src)
    body, world = coarse(body, world)
    K = meta.get("test_lives", 1)                                           # the last K lives are strangers: never learned
    mu, sd = body[:-K].mean(axis=(0, 1)), body[:-K].std(axis=(0, 1))
    Z = np.clip((body - mu) / np.maximum(sd, 1e-8), -10, 10).astype(np.float32)
    T = Z.shape[1] - max(H)
    Dl = np.stack([Z[:, h:h + T] - Z[:, :T] for h in H], axis=2)             # [lives, T, horizons, channels]
    live = (Dl[:-K, :, -1] ** 2).mean(axis=(0, 1)) > 1e-4                  # as in the life model
    names = [n for n, ok in zip(meta["body"], live) if ok]
    Dl = Dl[:, :, :, live]
    ms = (Dl[:-K] ** 2).mean(axis=(0, 1))
    scale = torch.tensor(np.maximum(ms, 0.01 * np.median(ms, axis=1, keepdims=True)), device=dev)
    X = torch.tensor(np.concatenate([Z[:, :T, live], world[:, :T]], axis=2).astype(np.float32))
    Y = torch.tensor(Dl.astype(np.float32))
    n_in, C = X.shape[2], len(names)
    n = X.shape[0]
    test = [(l, int(s)) for l in range(n - K, n) for s in np.linspace(0, T - WIN, max(3, 24 // K)).astype(int)]
    print(f"{C} body channels, {n_in} inputs per step, {n - K} lives to learn from x {T * STEP / 1440:.0f} days, "
          f"windows of {WIN * STEP / 60:.0f} h; held out: {K} lives, {len(test)} windows", flush=True)

    def batch(pairs):
        return (torch.stack([X[l, s:s + WIN] for l, s in pairs]).to(dev),
                torch.stack([Y[l, s:s + WIN] for l, s in pairs]).to(dev))

    def surprise(pred, y):                                                  # scored on the second day of each window
        p, y = pred.view(*y.shape)[:, WIN // 2:], y[:, WIN // 2:]
        return ((p - y) ** 2 / scale).mean(dim=(0, 1, 3))

    def held_out(model):
        model.eval()
        with torch.no_grad():
            s = torch.stack([surprise(model(x), y) for x, y in (batch(test[i:i + BATCH]) for i in range(0, len(test), BATCH))])
        model.train()
        return [round(1 - float(v), 4) for v in s.mean(0)]

    results = json.load(open(os.path.join(out, "life.json"))) if os.path.exists(os.path.join(out, "life.json")) else {}
    shared_body = None
    for seed, arm in [(sd, a) for sd in seeds for a in arms]:
        key = f"{arm}|seed{seed}"
        if key in results:
            continue
        rng = np.random.default_rng(seed)
        train = [(int(l), int(s)) for l, s in zip(rng.integers(0, n - K, steps * BATCH),
                                                  rng.integers(0, T - WIN, steps * BATCH))]
        torch.manual_seed(seed)
        kind, lr = (arm.split("@")[0], float(arm.split("@")[1])) if "@" in arm else (arm, LR)
        if kind == "gru":
            model = GRU(n_in, len(H) * C).to(dev)
        else:
            if kind == "tokyo_random":
                shared_body = None
                if dev.type == "mps":
                    torch.mps.empty_cache()
                model = OnBody(Body(gguf, dev).scramble(1), n_in, len(H) * C).to(dev)
            else:
                shared_body = shared_body or Body(gguf, dev)
                model = OnBody(shared_body, n_in, len(H) * C).to(dev)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr, weight_decay=0.0)
        curve, t0 = [dict(step=0, skill=held_out(model))], time.time()
        print(f"{key:14s} step     0  held-out skill 10 / 60 min: {curve[0]['skill']}  "
              f"({sum(p.numel() for p in params):,} numbers learning)", flush=True)
        for s in range(steps):
            for g in opt.param_groups:
                g["lr"] = lr * min(1.0, (s + 1) / 50) * 0.5 * (1 + math.cos(math.pi * s / steps))
            x, y = batch(train[s * BATCH:(s + 1) * BATCH])
            loss = surprise(model(x), y).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            if (s + 1) % max(1, steps // 8) == 0 or s == steps - 1:
                curve.append(dict(step=s + 1, seconds=round(time.time() - t0), train=round(1 - float(loss), 4),
                                  skill=held_out(model)))
                c = curve[-1]
                print(f"{key:14s} step {c['step']:5d}  held-out skill 10 / 60 min: {c['skill']}  "
                      f"(train {c['train']})  {c['seconds']}s", flush=True)
        results[key] = dict(curve=curve, final=curve[-1]["skill"], channels=names)
        json.dump(dict(results, setup=dict(step_minutes=STEP, window=WIN, horizons=H, batch=BATCH, lr=LR, steps=steps,
                                           src=os.path.basename(os.path.normpath(src)))),
                  open(os.path.join(out, "life.json"), "w"), indent=1)
        results.pop("setup", None)
        torch.save({k: v for k, v in model.state_dict().items() if not k.startswith("body.")},
                   os.path.join(out, f"{arm}-seed{seed}.pt"))
        del model, opt
        if dev.type == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
