#!/usr/bin/env python3
"""Life model v1: the action wire. He lives his body as in v0 and now also does things that change it.

Every 10 minutes he picks what to do for the next 10: nothing, breathe slowly, distract himself, talk himself down
(the engine's three ways to calm down, change 30), seek company, or exercise. What he does goes into the engine as an
input and into his own senses as a copy of the command (efference copy), so he learns what he caused.

What he wants comes from his body alone: being near his normal for this time of day. Drive = how far his felt body
(heart rate, worry, amygdala, cortisol, noradrenaline) is from where it runs on a calm day at this hour, in units of how
much each one varies in a lived life. Reward = how much closer he got (drive reduction: Keramati & Gutkin 2014;
the best of the reward forms tested by Yoshida et al. 2024), minus a small effort cost for doing anything at all.
Nothing tells him which action is good for what.

He learns by acting (advantage actor-critic) while still predicting his body (v0's world model, kept as a second job).
His state is never reset. At the end he and four twins live the same 15 test days per life: one who never does
anything, one who acts at random, one who always breathes slowly, one who always talks himself down.

usage: act.py <engine folder> <lives folder from live.py> <out folder> <envs> <seconds>
"""
import json, math, os, sys, time
from multiprocessing import Pipe, Process

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from live import BODY_EXTRA, DAY, WORLD, Life
import learn

ACTIONS = ["nothing", "breathe", "distract", "reappraise", "seek_company", "exercise"]
FELT = ["heart_rate", "worry", "amygdala", "cort", "ne"]
BLOCK = 10                                                   # minutes per choice
ROLL = 18                                                    # choices per learning step (3 hours)
EFFORT = 0.02                                                # drive units per 10 minutes of doing something (theory)
GAMMA, LAM = 0.97, 0.95                                      # about a 5-hour horizon


def effect(action, hour, rng):
    """what doing this for the next 10 minutes puts into the engine, and what it shuts out"""
    if action in ("breathe", "distract", "reappraise", "exercise"):
        return {action: 1.0}, (("novelty", "company") if action == "distract" else ())   # looking away misses things
    if action == "seek_company":
        if rng.random() < (0.6 if 9 <= hour < 22 else 0.1):    # by day someone is usually around; at night rarely
            return {"company": 1.0}, ()
        if rng.random() < 0.05:                              # and now and then a try ends in a brush-off
            return {"rejection": 0.6}, ()
    return {}, ()


def calm_day(chem, idx):
    """his felt body at each minute of a calm day (awake 7-23, nothing happening): his normal for that hour"""
    e = chem.Chem(hour=7.0)
    ref = np.zeros((DAY, len(idx)))
    keys = list(e.c)
    for i in range(2 * DAY):
        e.step(1 / 60, awake=7.0 <= e.hour < 23.0)
        if i >= DAY:
            body = [e.c[x] for x in keys] + [e.S, e.heart_rate(), e.blood_pressure(), e.worry_st, e.amy_out()]
            ref[(7 * 60 + i + 1) % DAY] = [body[j] for j in idx]
    return ref


def env(conn, engine_dir, seed, felt_idx, felt_sd):
    sys.path.insert(0, os.path.abspath(engine_dir))
    import engine as chem
    life, ref = Life(chem, seed), calm_day(chem, felt_idx)
    rng = np.random.default_rng(seed + 7919)               # what his tries bring; the world's days stay the same

    def drive():
        b = life.body()
        return float(np.sqrt(np.sum(((np.array([b[j] for j in felt_idx]) - ref[life.m % DAY]) / felt_sd) ** 2)))

    while True:
        a = conn.recv()
        if a is None:
            break
        d0, frames, awake = drive(), [], 0
        add, block = effect(ACTIONS[a], (life.m % DAY) / 60, rng)
        for _ in range(BLOCK):
            awake += not life.asleep()
            body, w = life.minute(add, block)
            frames.append(np.concatenate([body, w]))
        conn.send((np.asarray(frames, np.float32), d0, drive(), awake == BLOCK))


class Brain(torch.nn.Module):
    def __init__(self, n_in, n_body, hidden=256):
        super().__init__()
        self.core = torch.nn.GRU(n_in, hidden, batch_first=True)
        self.wm = torch.nn.Linear(hidden, len(learn.H) * n_body)
        self.pi = torch.nn.Linear(hidden, len(ACTIONS))
        self.v = torch.nn.Linear(hidden, 1)
        self.n_body = n_body


class Lives:
    """a batch of lives living in their own processes, stepped together"""

    def __init__(self, engine_dir, seeds, felt_idx, felt_sd):
        self.conns, self.procs = [], []
        for s in seeds:
            a, b = Pipe()
            p = Process(target=env, args=(b, engine_dir, s, felt_idx, felt_sd), daemon=True)
            p.start()
            self.conns.append(a)
            self.procs.append(p)

    def step(self, actions):
        for c, a in zip(self.conns, actions):
            c.send(int(a))
        out = [c.recv() for c in self.conns]
        return (np.stack([o[0] for o in out]), np.array([o[1] for o in out]), np.array([o[2] for o in out]),
                np.array([o[3] for o in out]))

    def close(self):
        for c in self.conns:
            c.send(None)
        for p in self.procs:
            p.join(timeout=5)


def main():
    engine_dir, src, out, n_env, seconds = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), float(sys.argv[5])
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(0)
    torch.set_num_threads(4)
    meta, body, world = learn.load(src)                     # v0's lived lives: how much each part of him varies
    body = body[:3]                                         # three lives are plenty for that, and fit in memory
    names = meta["body"]
    NB, NW = len(names), len(WORLD)
    mu, sd = body.mean(axis=(0, 1)), np.maximum(body.std(axis=(0, 1)), 1e-6)
    felt_idx = [names.index(f) for f in FELT]
    felt_sd = sd[felt_idx]
    Z = np.clip((body - mu) / sd, -10, 10)
    ms = (learn.deltas(Z) ** 2).mean(axis=(0, 1))
    scale = torch.tensor(np.maximum(ms, 0.01 * np.median(ms, axis=1, keepdims=True)), dtype=torch.float32)
    mu_t, sd_t = torch.tensor(mu, dtype=torch.float32), torch.tensor(sd, dtype=torch.float32)

    def senses(frames, actions):
        """[lives, minutes, body+world] and the action held during them -> his input: body, world, what he did"""
        f = torch.tensor(frames)
        z = ((f[..., :NB] - mu_t) / sd_t).clamp(-10, 10)
        a = torch.nn.functional.one_hot(torch.as_tensor(actions), len(ACTIONS)).float()
        return torch.cat([z, f[..., NB:], a[:, None, :].expand(-1, f.shape[1], -1)], dim=2), z

    brain = Brain(NB + NW + len(ACTIONS), NB)
    opt = torch.optim.Adam(brain.parameters(), lr=3e-4)
    lives = Lives(engine_dir, [100 + k for k in range(n_env)], felt_idx, felt_sd)
    h = torch.zeros(1, n_env, brain.core.hidden_size)
    t0, log, step = time.time(), [], 0
    print(f"{n_env} lives acting; brain {sum(p.numel() for p in brain.parameters()):,} numbers", flush=True)
    while time.time() - t0 < seconds:
        h_start, xs, zs, acts, rew, awake, stress = h, [], [], [], [], [], []
        with torch.no_grad():
            for _ in range(ROLL):
                a = torch.distributions.Categorical(logits=brain.pi(h[-1])).sample()
                frames, d0, d1, aw = lives.step(a.numpy())
                x, z = senses(frames, a)
                _, h = brain.core(x, h)
                xs.append(x); zs.append(z); acts.append(a); awake.append(torch.tensor(aw))
                rew.append(torch.tensor(d0 - d1 - EFFORT * (a.numpy() > 0) * aw, dtype=torch.float32))
                w = frames[..., NB:]
                stress.append(torch.tensor((w[..., WORLD.index("judged")] > 0).any(1)
                                           | (w[..., WORLD.index("threat")] > 0).any(1)))
        # relive the 3 hours with gradients: the same choices, now scored
        hh, logits, values, preds = h_start, [], [], []
        for k in range(ROLL):
            logits.append(brain.pi(hh[-1]))
            values.append(brain.v(hh[-1]).squeeze(-1))
            y, hh = brain.core(xs[k], hh)
            preds.append(brain.wm(y).view(n_env, BLOCK, len(learn.H), NB))
        with torch.no_grad():
            v_end = brain.v(hh[-1]).squeeze(-1)
        V = torch.stack(values)
        R = torch.stack(rew)
        adv, gae = torch.zeros_like(R), torch.zeros(n_env)
        for k in reversed(range(ROLL)):
            nxt = v_end if k == ROLL - 1 else V[k + 1].detach()
            delta = R[k] + GAMMA * nxt - V[k].detach()
            gae = delta + GAMMA * LAM * gae
            adv[k] = gae
        ret = adv + V.detach()
        dist = torch.distributions.Categorical(logits=torch.stack(logits))
        A = torch.stack(acts)
        an = (adv - adv.mean()) / (adv.std() + 1e-6)
        loss_pi = -(dist.log_prob(A) * an).mean() - 0.01 * dist.entropy().mean()
        loss_v = 0.5 * ((V - ret) ** 2).mean()
        # the world model's job, kept: predict his body 1, 15, 60 minutes on, where the 3 hours reach that far
        Zs, P = torch.cat(zs, 1), torch.cat(preds, 1)
        wm = []
        for i, hz in enumerate(learn.H):
            wm.append(((P[:, :-hz, i] - (Zs[:, hz:] - Zs[:, :-hz])) ** 2 / scale[i]).mean())
        loss_wm = sum(wm) / len(wm)
        opt.zero_grad()
        (loss_pi + loss_v + loss_wm).backward()
        torch.nn.utils.clip_grad_norm_(brain.parameters(), 1.0)
        opt.step()
        h = h.detach()
        step += 1
        aw, st = torch.stack(awake), torch.stack(stress)
        share = lambda m: [round(float(((A == i) & m).sum() / max(1, m.sum())), 3) for i in range(len(ACTIONS))]
        log.append(dict(days=step * ROLL * BLOCK / DAY, reward=float(R.mean()), wm=float(loss_wm),
                        awake_share=share(aw), stressed_share=share(aw & st), stressed_blocks=int((aw & st).sum())))
        if step % 40 == 0:
            r = log[-40:]
            print(f"day {log[-1]['days']:6.1f}  reward {np.mean([x['reward'] for x in r]):+.4f}  "
                  f"world model {np.mean([x['wm'] for x in r]):.3f}  awake choices "
                  + " ".join(f"{n[:5]} {np.mean([x['awake_share'][i] for x in r]):.2f}" for i, n in enumerate(ACTIONS)),
                  flush=True)
    lives.close()
    torch.save(brain.state_dict(), os.path.join(out, "brain.pt"))
    json.dump(dict(actions=ACTIONS, felt=FELT, effort=EFFORT, envs=n_env, log=log), open(os.path.join(out, "acted.json"), "w"))

    # the twins: the same 15 test days per life (seeds he never lived), each living them a different way
    print("test: 15 days, he and four twins", flush=True)
    test = {}
    for who in ("him", "nothing", "random", "always_breathe", "always_reappraise"):
        lv = Lives(engine_dir, [9000 + k for k in range(n_env)], felt_idx, felt_sd)
        g = torch.Generator().manual_seed(1)
        hh, drives, stressed_d, choice, choice_st = torch.zeros(1, n_env, brain.core.hidden_size), [], [], [], []
        with torch.no_grad():
            for _ in range(15 * DAY // BLOCK):
                if who == "him":
                    a = torch.distributions.Categorical(logits=brain.pi(hh[-1])).sample()
                elif who == "random":
                    a = torch.randint(len(ACTIONS), (n_env,), generator=g)
                else:
                    a = torch.full((n_env,), {"nothing": 0, "always_breathe": 1, "always_reappraise": 3}[who])
                frames, d0, d1, aw = lv.step(a.numpy())
                x, _ = senses(frames, a)
                _, hh = brain.core(x, hh)
                w = frames[..., NB:]
                st = (w[..., WORLD.index("judged")] > 0).any(1) | (w[..., WORLD.index("threat")] > 0).any(1)
                for i in range(n_env):
                    if aw[i]:
                        drives.append(d1[i])
                        choice.append(int(a[i]))
                        if st[i]:
                            stressed_d.append(d1[i])
                            choice_st.append(int(a[i]))
        lv.close()
        cnt = lambda c: {n: round(c.count(i) / max(1, len(c)), 3) for i, n in enumerate(ACTIONS)}
        test[who] = dict(drive_awake=float(np.mean(drives)), drive_stressed=float(np.mean(stressed_d)),
                         effort_share=float(np.mean(np.array(choice) > 0)), choices=cnt(choice),
                         choices_stressed=cnt(choice_st), stressed_blocks=len(stressed_d))
        print(f"{who:18s} drive awake {test[who]['drive_awake']:.3f}  under stress {test[who]['drive_stressed']:.3f}  "
              f"doing something {test[who]['effort_share']:.2f}", flush=True)
    json.dump(test, open(os.path.join(out, "twins.json"), "w"), indent=1)
    print("him, calm:", test["him"]["choices"], "\nhim, judged or scared:", test["him"]["choices_stressed"], flush=True)


if __name__ == "__main__":
    main()
