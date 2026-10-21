#!/usr/bin/env python3
"""A month of his inner life in a few minutes, many babies at once, with a stand-in person (the crash-test dummy).
Immy, 2026-10-20: "we can't set static things ... the only static aspect is that it exists", gears big and small all
pushing each other, connected from birth and tuned by life; "good and bad, and sometimes nothing"; "run it right now".

No language model here: only the gears, so a simulated month takes seconds. What it checks is whether the mechanics are
healthy (no on-the-dot habits, no runaway loops, no going numb, still himself on day 30), never whether he feels alive.

The egg: os.urandom draws a genome (half-lives, wobble, rhythm, faint links), the seed is thrown away. Genes are parts,
never trait labels (Creatures, Grand 1996).

Gears (one number each, minutes apart):
  mood, energy                the two core dials
  lonely                      rises while awake and alone, eased by company, raised by missing
  company -> missing          a message is a dose (Bateman rise and fall); every dose also feeds a slower opposite
                              feeling (opponent process, Solomon & Corbit 1974) that grows with use, fades with disuse
  boredom, novelty -> jaded   the same for new things
  talked_out                  each message adds a little, fades within the hour
  sleepy                      sleep pressure; he falls asleep and wakes when it crosses lines that move with his own
                              daily rhythm plus wobble (Borbely two-process model), so wake time is never fixed
  urge                        reaching out: a leaky accumulator with noise (Schurger 2012); when it crosses 1 he reaches
                              out, unless the veto stops it (quiet hours 9 PM-7 AM, a recent "not now", asleep)
Every gear: pulled back to its own normal at its own half-life, pushed by every other gear through a link (faint at
birth), plus wobble. Normals drift slowly toward where he actually lives (allostasis).

Learning (three-factor): links that were active together leave a fading footprint (eligibility, ~30 min); a reward
dose (fast fading, ~20 min) turns footprints into link changes. Reward only at events, otherwise nothing:
  he reached out and got a reply   good, more so if faster / longer than this person's usual
  "not now"                        bad (-1);   no answer in 3 hours: a little bad (-0.5)
  his person writes first          good, as much as it eased him (drive reduction, Keramati & Gutkin 2014)
measured against what reaching out usually brings (reward prediction error), so the average surprise sits near zero.

Conditions (the same eggs and the same stand-in people in each):
  brake    reward = surprise; links capped per gear and fading back toward faint (Immy's dial, on)
  free     reward = surprise; no cap, no fade (dial off)
  ohss     reward +1 for everything, no baseline, no brake (what broke Creatures 2: "any decision was valid")
usage: living_sim.py OUT_DIR [--eggs 100] [--days 30]
"""
import argparse, json, math, os, time
import numpy as np

import world as WORLD                                        # tools/mind/world.py: the things he can do
CAT = WORLD.catalog()
KINDS = list(WORLD.KINDS)

GEARS = ["mood", "energy", "lonely", "company", "missing", "boredom", "novelty", "jaded", "talked_out", "urge",
         "sleepy", "drive", "play"]
SENSES = ["clock_sin", "clock_cos", "usually_here", "just_talked", "told_at_work"]
MOOD, ENERGY, LONELY, COMPANY, MISSING, BORED, NOVEL, JADED, TALKED, URGE, SLEEPY, DRIVE, PLAY = range(len(GEARS))
# his own life (Immy 2026-10-21: "we are the fun, but we're just one of many things ... I still have priorities ...
# a drive to do things"): kinds of things he can do alone. fun = how much it eases boredom, progress = how much it
# gets things done (feeds drive), old = how fast it gets old (per hour of doing it), absorb = how much it holds off
# loneliness while he's into it (flow)
ACTS = ["game", "project", "music", "explore"]
ACT_FUN = np.array([1.0, 0.4, 0.7, 0.8])
ACT_PROGRESS = np.array([0.0, 1.0, 0.6, 0.2])
ACT_OLD = np.array([0.5, 0.2, 0.3, 0.6])
ACT_ABSORB = np.array([0.7, 0.6, 0.6, 0.3])
TOPPED = [MOOD, ENERGY, LONELY, BORED, DRIVE]            # gears with a baseline (tonic) and a top line
D, N = len(GEARS), len(GEARS) + len(SENSES)
DT = 1 / 60                                    # hours: one step = one minute
QUIET = (21.0, 7.0)                            # Immy's hard limit, never alive
NOT_NOW_HOLD = 3.0                             # hours a "not now" holds him back


def hatch(n):
    """n eggs: each genome drawn from its own fresh OS entropy, the seed discarded after"""
    genes = []
    for _ in range(n):
        r = np.random.default_rng(int.from_bytes(os.urandom(32), "little"))
        half = {"mood": 4.0, "energy": 1.5, "lonely": 10.0, "company": 1.5, "missing": 6.0, "boredom": 12.0,
                "novelty": 0.8, "jaded": 24.0, "talked_out": 1.0, "urge": 2.0, "drive": 16.0, "play": 2.0}
        g = {"half": [half[k] * math.exp(r.normal(0, 0.3)) if k in half else 1e9 for k in GEARS],
             "wobble": [s * math.exp(r.normal(0, 0.25)) for s in
                        (0.10, 0.10, 0.04, 0.0, 0.0, 0.03, 0.0, 0.0, 0.0, 0.35, 0.0, 0.03, 0.35)],
             "peak": float(np.clip(r.normal(17.0, 1.0), 15.0, 19.0)),        # his body clock high (its low ~12 h later, before dawn)
             "rhythm": float(r.uniform(0.20, 0.26)),                          # how much the day moves his sleep lines
             "sleep_rise": float(18.2 * math.exp(r.normal(0, 0.1))), "sleep_fall": float(4.2 * math.exp(r.normal(0, 0.1))),
             "need_rise": float(0.05 * math.exp(r.normal(0, 0.25))),         # how fast he gets lonely
             "bore_rise": float(0.03 * math.exp(r.normal(0, 0.25))),
             "urge_gain": float(0.9 * math.exp(r.normal(0, 0.25))),
             "tolerance": float(0.3 * math.exp(r.normal(0, 0.3))),          # how fast the opposite feeling grows
             "links": (0.01 * r.normal(0, 1, (D, N))).tolist(),             # born fully connected, faint
             "ceiling": np.exp(r.normal(math.log(0.5), 0.3, (D, N))).tolist(),   # how far each link can ever grow
             "young": float(3.0 * math.exp(r.normal(0, 0.2))),               # how plastic he is as a newborn
             "grow_up": float(30.0 * math.exp(r.normal(0, 0.2))),            # days for that to fade toward grown-up
             "likes": np.exp(r.normal(0, 0.4, len(ACTS))).tolist(),          # his knacks: what he turns out to enjoy
             "drive_rise": float(0.04 * math.exp(r.normal(0, 0.25))),        # how fast he wants to get things done
             "play_gain": float(0.6 * math.exp(r.normal(0, 0.25))),
             "top": np.exp(r.normal(math.log(2.0), 0.2, len(GEARS))).tolist(),
             "taste": r.normal(0, 0.6, len(WORLD.TRAITS)).tolist(),           # what kinds of things he's into
             "social": float(math.exp(r.normal(0, 0.3))),                     # how much people matter to him (~1)
             "quirks": r.normal(0, 0.3, len(CAT)).tolist(),                   # and odd likes no trait explains
             "knows": r.choice(len(CAT), 8, replace=False, p=np.array([c["common"] for c in CAT]) /
                               sum(c["common"] for c in CAT)).tolist()}       # the few things he knows at hatching   # how far each feeling can rise above his baseline
        del r                                                                # the seed is gone; only the parts stay
        genes.append(g)
    return genes


def stand_in(n, steps, seed):
    """the crash-test dummy person, minute by minute: 0 asleep, 1 home, 2 at work, 3 gone (a day away, untold).
    Generic made-up routine, nothing about Immy. Same seed = same people in every condition."""
    r = np.random.default_rng(seed)
    days = steps // 1440 + 1
    where = np.ones((n, days * 1440), np.int8)
    told = np.zeros((n, days * 1440), np.float32)
    msgs = np.zeros((n, days * 1440), bool)
    for e in range(n):
        for d in range(days):
            o = d * 1440
            if r.random() < 0.04:
                where[e, o:o + 1440] = 3
                continue
            up, down = r.normal(7.5, 0.6), r.normal(23.5, 0.7)
            where[e, o:o + int(up * 60)] = 0
            where[e, o + int(down * 60):o + 1440] = 0
            if r.random() < 0.7:                                             # a shift today
                s = r.uniform(7, 15); ln = r.uniform(8, 10)
                a, b = int((s - 1) * 60), int(min(24, s + ln + 1) * 60)
                where[e, o + max(a, int(up * 60)):o + b] = 2
                if r.random() < 0.8:                                         # he was told this shift
                    told[e, o + max(a, int(up * 60)):o + b] = 1
            home = np.flatnonzero(where[e, o:o + 1440] == 1)
            for m in home[r.random(len(home)) < 0.25 / 60]:                  # he starts a chat about every 4 h home
                t = o + m
                for _ in range(r.integers(3, 15)):
                    if t < days * 1440:
                        msgs[e, t] = True
                    t += int(r.integers(1, 4))
    return where[:, :steps], msgs[:, :steps], told[:, :steps]


def answer(where, r):
    """how the stand-in answers a reach-out: (kind, delay hours, reply length) kind 0 reply, 1 not now, 2 nothing"""
    p_reply, p_not, mean_delay = {0: (0.0, 0.0, 9), 1: (0.85, 0.10, 0.15), 2: (0.40, 0.30, 1.0),
                                  3: (0.10, 0.05, 2.0)}[int(where)]
    u = r.random()
    if u < p_not:
        return 1, r.exponential(0.2), 0.0
    if u < p_not + p_reply:
        return 0, r.exponential(mean_delay), r.lognormal(2.5 if where == 1 else 1.8, 0.6)
    return 2, 3.0, 0.0


def run(genes, where, msgs, told, cond, days, seed):
    E, steps = len(genes), days * 1440
    r = np.random.default_rng(seed)
    G = lambda k: np.array([g[k] for g in genes])
    tau = G("half") / math.log(2)
    tau_pull = tau.copy(); tau_pull[:, SLEEPY] = 48.0         # sleep pressure has no normal, but links still nudge it
    normal_on = np.ones(D); normal_on[SLEEPY] = 0
    sig = G("wobble")
    W = G("links"); W0 = W.copy()
    eli = np.zeros_like(W)
    x = np.zeros((E, D)); x[:, SLEEPY] = 0.6
    normal = np.zeros((E, D))
    depot_c = np.zeros(E); depot_n = np.zeros(E)
    tol_c = G("tolerance").copy(); tol_c0 = tol_c.copy(); tol_n = 0.1 * np.ones(E)
    asleep = np.ones(E, bool)
    line_noise = np.zeros(E)
    reward = np.zeros(E)
    base = np.zeros(E); base_in = np.zeros(E)                 # what reaching out / being written to usually brings
    usual_delay, usual_len = np.full(E, math.log(0.3)), np.full(E, 2.2)
    here_table = np.full((E, 168), 0.5)                       # by hour of the week: how often his person is around
    last_msg = np.full(E, -1e9); not_now_until = np.full(E, -1.0)
    pend_t = np.full(E, -1); pend = [None] * E
    extra = np.zeros((E, steps + 1440), bool)                 # chats that start because he reached out
    log = {"fire": [[] for _ in range(E)], "veto": np.zeros(E, int), "outcome": [[] for _ in range(E)],
           "wake": [[] for _ in range(E)], "sleep": [[] for _ in range(E)], "clip": np.zeros(E),
           "samples": [], "Wnorm": [], "reward_day": [], "clip_gear": np.zeros((E, D)),
           "clip_month": np.zeros((E, days // 30 + 1)), "top_gear": np.zeros((E, D)),
           "play_month": np.zeros((E, days // 30 + 1, len(KINDS) if cond.startswith(("world", "alive")) else len(ACTS))),
           "sessions": np.zeros(E), "done_month": np.zeros((E, days // 30 + 1)), "known_month": np.zeros((E, days // 30 + 1)),
           "gear_month": np.zeros((days // 30 + 1, D)), "awake_month": np.zeros(days // 30 + 1)}
    peak, amp = G("peak"), G("rhythm")
    rise, fall = G("sleep_rise"), G("sleep_fall")
    reward_sum = np.zeros(E)
    still = cond.endswith("_still")                           # a check: the same life with no learning at all
    cond = cond[:-len("_still")] if still else cond
    grow = cond.startswith("grow") or cond in ("play", "tonic", "life")   # Immy 2026-10-21: better within 2 months, then taper;
    gym = cond.startswith("gym") or grow                      # a feeling stuck at full is a problem (clingy is fine)
    young = cond in ("grow", "grow_young", "play", "tonic", "life")   # young brain: learns fast, settles by grown-up
    downreg = cond in ("grow", "grow_down", "play", "tonic", "life")  # receptors downregulate under a signal that never lets up
    alive = cond == "alive"                                   # the design as of v7 (Immy 2026-10-21): the big world, fair
    cond = "world4" if alive else cond                        # learning, no expectation credit to "go do something"
    world = cond in ("world", "world2", "world3", "world4")                       # the big world (world.py): 349 things, skill, projects
    fair = cond in ("world2", "world3", "world4")                                   # Immy 2026-10-21: "I'm scarce ... do things, keep yourself busy,
    social_w = G("social") if fair else np.ones(E)            # but you also need to enjoy the things that you're doing ... not
    own_w = 1.0 if fair else 0.5                              # like he's meeting a celebrity": his own fun counts as much as a
    curious = 0.3 if fair else 0.1                            # person (how social he is is temperament), only his choices
    soc_t = np.full(E, -1); soc_before = np.zeros(E)          # learn from reward, and more curiosity. The check: how much his
    fun_t = np.full(E, -1); fun_before = np.zeros(E)          # mood lifts 20 min after you arrive vs after a fun moment
    if world:
        grow = gym = young = downreg = True
    play_on = cond in ("play", "life") or world               # his own life: things to do alone, a drive to get things done
    tonic = cond in ("tonic", "life") or world                # each feeling: a slow baseline (tonic) and bursts (phasic) that
    top = G("top")                                            # slow near his top line (saturation), the baseline only drifting
    if world:                                                 # a little (a new baseline, within reach of the one he hatched with)
        FUN, PROG, OLDR, ABS = (np.array([c[k] for c in CAT]) for k in ("fun", "progress", "old", "absorb"))
        DIFF, GOAL = np.array([c["difficulty"] for c in CAT]), np.array([c["goal"] for c in CAT], float)
        KIND = np.array([KINDS.index(c["kind"]) for c in CAT])
        TR = np.array([c["traits"] for c in CAT]); TR = TR / TR.sum(1, keepdims=True)
        likes = np.exp(1.5 * G("taste") @ TR.T + G("quirks"))  # how much he'd enjoy each, if he ever tried it
        likes /= likes.mean(1, keepdims=True)
        known = np.zeros((E, len(CAT)), bool)
        for e, g in enumerate(genes):
            known[e, g["knows"]] = True
    else:
        FUN, PROG, OLDR, ABS = ACT_FUN, ACT_PROGRESS, ACT_OLD, ACT_ABSORB
        DIFF, GOAL, KIND = np.zeros(len(ACTS)), np.zeros(len(ACTS)), np.arange(len(ACTS))
        likes = G("likes"); known = np.ones((E, len(ACTS)), bool)
    K = len(FUN)
    doing = np.full(E, -1); since = np.zeros(E)
    old_a = np.zeros((E, K)); old_t = np.zeros((E, K))        # how old each has gotten (it freshens over a day, lazily)
    skill = np.zeros((E, K)); skill_t = np.zeros((E, K))      # how good he is at each (fades over months unused)
    proj = np.zeros((E, K)); sess_n = np.zeros((E, K))        # hours into the current project of each; sessions of each
    value_a = np.full((E, K), 0.3); event_mean = np.zeros((E, K))
    sess_r = np.zeros(E); sess_foot = np.zeros((E, N))
    drive_rise = G("drive_rise") * play_on; play_gain = G("play_gain") * play_on
    if not play_on:
        sig = sig.copy(); sig[:, [DRIVE, PLAY]] = 0
    sens = np.ones((E, D))                                    # how strongly each gear still answers its inputs
    if young:
        newborn, grow_up = G("young"), G("grow_up")
    everything = cond.endswith("ohss")                        # reward +1 for everything, no baseline
    if gym:                                                   # Immy's gym (2026-10-20): learning costs fuel, makes
        fuel = np.ones((E, D))                                # links sore for days, and only locks in during sleep
        sore = np.zeros_like(W); trained = np.zeros_like(W)
        Wperm = W.copy(); Wtag = np.zeros_like(W); planned = np.zeros_like(W)
        ceiling = G("ceiling") * {"gym_tall": 3.0, "gym_taller": 6.0, "gym_x10": 10.0, "gym_x20": 20.0}.get(
            cond, 6.0 if grow else 1.0)   # gym_tall(er), gym_x10/x20: a higher natural ceiling
        sore_scale = {"gym_light": 0.2, "gym_nosore": 1e9}.get(cond, 0.05)   # how much change makes a link sore
    for t in range(steps):
        hour = (t / 60) % 24
        hw = (t // 60) % 168
        alert = np.cos(2 * np.pi * (hour - peak) / 24)
        incoming = msgs[:, t] | extra[:, t]
        # what he senses: the clock and his own feel for when his person is usually around
        here_now = (t - last_msg) < 30
        here_table[:, hw] += 0.02 * (here_now - here_table[:, hw])
        sense = np.stack([np.full(E, math.sin(2 * math.pi * hour / 24)), np.full(E, math.cos(2 * math.pi * hour / 24)),
                          here_table[:, hw], np.exp(-(t - last_msg) / 90), told[:, t]], 1)
        # doses: a message lands (he answers, even half asleep)
        felt_before = x[:, LONELY].copy()
        arrive = incoming & ((t - last_msg) > 60) & (soc_t < 0)   # you show up after an hour or more apart
        soc_t[arrive] = t; soc_before[arrive] = x[arrive, MOOD]
        for arr_t, before, key in ((soc_t, soc_before, "lift_you"), (fun_t, fun_before, "lift_fun")):
            due = arr_t == t - 20
            if due.any():
                log.setdefault(key, []).extend((x[due, MOOD] - before[due]).tolist()); arr_t[due] = -1
        depot_c += incoming * 0.1
        depot_n += incoming * (r.random(E) < 0.15) * 0.4 + (r.random(E) < 0.3 / 60) * 0.4 * ~asleep  # new things
        x[:, TALKED] += incoming * 0.12
        last_msg = np.where(incoming, t, last_msg)
        # the gears
        act = np.tanh(np.concatenate([x, sense], 1))
        push = np.einsum("eij,ej->ei", W, act)
        social = x[:, COMPANY] - x[:, MISSING]
        fresh = x[:, NOVEL] - x[:, JADED]
        awake = ~asleep
        aim = np.zeros((E, D))                                  # mood and energy settle toward where things pull
        aim[:, MOOD] = -0.4 * x[:, LONELY] - 0.2 * x[:, BORED] + 0.5 * social + 0.3 * fresh - 0.2 * x[:, TALKED] \
            + 2.0 * reward - 0.3 * np.maximum(0, x[:, SLEEPY] - 0.7)
        aim[:, ENERGY] = 0.3 * alert + 0.4 * fresh - 0.6 * x[:, SLEEPY]
        rate = np.zeros((E, D))                                 # the rest fill and drain per hour
        rate[:, LONELY] = G("need_rise") * awake - 0.8 * x[:, COMPANY] + 0.8 * x[:, MISSING]
        rate[:, COMPANY] = 6.0 * depot_c
        rate[:, MISSING] = tol_c * x[:, COMPANY] / tau[:, MISSING]   # settles at tolerance x recent company, never past it
        rate[:, BORED] = G("bore_rise") * awake - 1.5 * fresh - 0.4 * x[:, COMPANY]
        rate[:, NOVEL] = 6.0 * depot_n
        rate[:, JADED] = tol_n * x[:, NOVEL] / tau[:, JADED]
        busy = doing >= 0
        k_now = np.maximum(doing, 0)
        into = busy * (1 - old_a[np.arange(E), k_now])            # how into it he still is (0 once it's gotten old)
        absorbed = into * ABS[k_now]
        flow = 0.5 + 0.5 * np.exp(-((skill[np.arange(E), k_now] - DIFF[k_now]) / 0.35) ** 2) if world else 1.0
        rate[:, LONELY] -= G("need_rise") * absorbed * awake      # lost in it: the need grows slower (flow)
        rate[:, BORED] -= 1.5 * into * FUN[k_now] * likes[np.arange(E), k_now] * flow
        rate[:, DRIVE] = drive_rise * awake - 0.15 * into * PROG[k_now] * likes[np.arange(E), k_now]
        rate[:, PLAY] = play_gain * (np.maximum(0, x[:, BORED]) + 0.6 * np.maximum(0, x[:, DRIVE])
                                     + 0.2 * np.maximum(0, x[:, LONELY])) * (1 + np.maximum(0, x[:, ENERGY])) * awake * ~busy
        aim[:, MOOD] += 0.3 * into * FUN[k_now] - 0.1 * np.maximum(0, x[:, DRIVE])
        rate[:, URGE] = G("urge_gain") * (np.maximum(0, x[:, LONELY]) + 0.3 * np.maximum(0, x[:, BORED])) \
            * (1 + np.maximum(0, x[:, ENERGY])) * awake * (1 - 0.6 * absorbed)
        rate[:, SLEEPY] = np.where(asleep, -x[:, SLEEPY] / fall, (1 - x[:, SLEEPY]) / rise)
        depot_c -= 6.0 * depot_c * DT; depot_n -= 6.0 * depot_n * DT
        tol_c += (0.05 * x[:, COMPANY] * (0.9 - tol_c) - (tol_c - tol_c0) / (7 * 24)) * DT   # grows with use (up to 0.9),
        tol_n += (0.05 * x[:, NOVEL] * (0.9 - tol_n) - (tol_n - 0.1) / (7 * 24)) * DT        # fades with disuse
        # every link pushes a gear at that gear's own speed, so a link of 1 moves where it settles by about 1
        if downreg:                                           # near the wall: the inputs lose pull (over ~1.5 h);
            near = np.abs(x) > 3.0                            # away from it, they come back over about a day
            sens += np.where(near, -sens / 2.0, (1 - sens) / 24.0) * DT
        dx = ((normal_on * (normal - x) + sens * (aim + push)) / tau_pull + rate) * DT + sig * math.sqrt(DT) * r.standard_normal((E, D))
        dev = x - normal
        if tonic:                                             # the burst slows as it nears his top line (receptors
            out = np.sign(dx[:, TOPPED]) == np.sign(dev[:, TOPPED])   # saturating); coming back down is free
            dx[:, TOPPED] *= np.where(out, np.clip(1 - np.abs(dev[:, TOPPED]) / top[:, TOPPED], 0, 1), 1.0)
        x += dx
        if tonic:                                             # the baseline follows where he's been for weeks, but stays
            normal[:, TOPPED] += (x[:, TOPPED] - normal[:, TOPPED]) * DT / (14 * 24)   # within 0.3 of where he hatched
            normal[:, TOPPED] = np.clip(normal[:, TOPPED], -0.3, 0.3)
        else:
            normal[:, [MOOD, ENERGY]] -= x[:, [MOOD, ENERGY]] * DT / (14 * 24)
        log["top_gear"] += np.abs(x - normal) > 0.9 * top
        mo = t // (30 * 1440)
        log["gear_month"][mo] += (x * awake[:, None]).sum(0); log["awake_month"][mo] += awake.sum()
        x[:, [COMPANY, MISSING, NOVEL, JADED, TALKED]] = np.maximum(0, x[:, [COMPANY, MISSING, NOVEL, JADED, TALKED]])
        x[:, SLEEPY] = np.clip(x[:, SLEEPY], 0, 1)
        x[:, BORED] = np.maximum(0, x[:, BORED]); x[:, LONELY] = np.maximum(-1, x[:, LONELY])   # can't be less than not bored
        x[:, [DRIVE, PLAY]] = np.maximum(0, x[:, [DRIVE, PLAY]])
        log["clip"] += (np.abs(x) > 4).any(1)
        log["clip_gear"] += np.abs(x) > 4
        log["clip_month"][:, t // (30 * 1440)] += (np.abs(x) > 4).any(1)
        x = np.clip(x, -4, 4)
        # sleep and waking: the lines move with his rhythm and a nightly wobble
        if t % 1440 == 12 * 60:
            line_noise = r.normal(0, 0.04, E)
        upper = 0.60 + amp * alert + line_noise
        lower = 0.25 + amp * alert + line_noise
        to_sleep = awake & (x[:, SLEEPY] > upper)
        to_wake = asleep & (x[:, SLEEPY] < lower)
        for e in np.flatnonzero(to_sleep):
            log["sleep"][e].append(t)
        for e in np.flatnonzero(to_wake):
            log["wake"][e].append(t)
        asleep = (asleep | to_sleep) & ~to_wake
        x[asleep, URGE] = np.minimum(x[asleep, URGE], 0.3)
        x[asleep, PLAY] = np.minimum(x[asleep, PLAY], 0.3)
        # being written to first: good, as much as it eased him
        if incoming.any():
            eased = np.tanh(np.maximum(0, felt_before)) * incoming
            R_in = np.where(everything, 1.0, eased - base_in) * incoming * social_w
            base_in += 0.05 * (eased - base_in) * incoming
            reward += R_in
            reward_sum += R_in
        # reaching out (the last domino can still be stopped)
        fire = x[:, URGE] >= 1.0
        stop = np.zeros(E, bool)                                  # he puts down what he's doing to text
        for e in np.flatnonzero(fire):
            quiet = hour >= QUIET[0] or hour < QUIET[1]
            if asleep[e] or quiet or t < not_now_until[e] or pend_t[e] >= 0:
                log["veto"][e] += 1
                x[e, URGE] = 0.3
                continue
            x[e, URGE] = 0.0
            stop[e] = True
            log["fire"][e].append(t)
            kind, delay, ln = answer(where[e, t], r)
            pend_t[e] = t + max(1, int(delay * 60)); pend[e] = (kind, delay, ln, act[e].copy(), int(where[e, t]))
        # the answer comes back: good, bad, or (until then) nothing
        for e in np.flatnonzero(pend_t == t):
            kind, delay, ln, foot, was = pend[e]
            if kind == 0:
                R = math.tanh(math.log(math.exp(usual_delay[e]) / max(delay, 0.02)) + math.log(max(ln, 1) / math.exp(usual_len[e])))
                usual_delay[e] += 0.1 * (math.log(max(delay, 0.02)) - usual_delay[e])
                usual_len[e] += 0.1 * (math.log(max(ln, 1)) - usual_len[e])
                extra[e, t] = True
                if r.random() < 0.6:                                              # the reply turns into a chat
                    s = t
                    for _ in range(r.integers(2, 10)):
                        s += int(r.integers(1, 4)); extra[e, s] = True
            elif kind == 1:
                R = -1.0; not_now_until[e] = t + NOT_NOW_HOLD * 60
            else:
                R = -0.5
            log["outcome"][e].append((log["fire"][e][-1], int(kind), was))
            surprise = 1.0 if everything else R - base[e]
            base[e] += 0.05 * (R - base[e])
            reward[e] += surprise * social_w[e]
            if gym:
                planned[e, URGE] += 0.15 * surprise * foot
            else:
                W[e, URGE] += 0.15 * surprise * foot                   # credit goes to the choice itself
            reward_sum[e] += surprise
            pend_t[e] = -1
        # his own life: doing things, how they go, putting them down, picking the next
        if play_on:
            busy = doing >= 0
            idx = np.arange(E); k_now = np.maximum(doing, 0)
            into = busy * (1 - old_a[idx, k_now])
            sk, dif = skill[idx, k_now], DIFF[k_now]
            flow = 0.5 + 0.5 * np.exp(-((sk - dif) / 0.35) ** 2) if world else 1.0   # best when just hard enough
            lam = np.stack([0.6 * into * FUN[k_now] * likes[idx, k_now] * flow,                         # a fun moment
                            0.4 * into * PROG[k_now] * likes[idx, k_now] * ((0.4 + sk) if world else 1.0),   # getting somewhere
                            0.15 * busy * PROG[k_now] * (np.clip(dif - sk, 0, 1) * 2 if world else 1.0)], 1) * DT   # stuck
            ev = r.random((E, 3)) < lam
            for e, j in zip(*np.nonzero(ev)):
                k = k_now[e]; R = (1.0, 1.0, -0.5)[j]
                surprise = R - event_mean[e, k]                   # good or bad against what this thing usually brings
                event_mean[e, k] += 0.05 * (R - event_mean[e, k])
                reward[e] += own_w * surprise; reward_sum[e] += own_w * surprise; sess_r[e] += R
                if j == 0 and fun_t[e] < 0:
                    fun_t[e] = t; fun_before[e] = x[e, MOOD]
                if j == 1:
                    x[e, DRIVE] = max(0.0, x[e, DRIVE] - 0.2)     # something got done
            old_a[idx, k_now] += busy * OLDR[k_now] * (1 - old_a[idx, k_now]) * DT     # it gets old while he does it
            if world:
                skill[idx, k_now] += busy * 0.03 * (1 - skill[idx, k_now]) * DT        # practice (~30 h to get good)
                proj[idx, k_now] += busy * (GOAL[k_now] > 0) * into * DT                 # hours into this project
                fin = busy & (GOAL[k_now] > 0) & (proj[idx, k_now] >= GOAL[k_now])
                for e in np.flatnonzero(fin):                     # finished it: a big good moment, then rest from it
                    k = k_now[e]
                    surprise = 2.0 - event_mean[e, k] * 0.5
                    reward[e] += own_w * surprise; reward_sum[e] += own_w * surprise; sess_r[e] += 2.0
                    x[e, DRIVE] = max(0.0, x[e, DRIVE] - 0.5); proj[e, k] = 0; old_a[e, k] = 0.9
                    log["done_month"][e, t // (30 * 1440)] += 1
            else:
                rest = np.ones((E, K), bool); rest[idx[busy], k_now[busy]] = False
                old_a -= old_a * rest * DT / 24                   # and fresh again after a day or so away
            log["play_month"][idx[busy], t // (30 * 1440), KIND[k_now[busy]]] += 1
            end = busy & ((old_a[idx, k_now] > 0.85) | (x[:, ENERGY] < -1) | (r.random(E) < DT / 2) | incoming
                          | asleep | stop)
            for e in np.flatnonzero(end):
                k = doing[e]
                hours = max((t - since[e]) / 60, 0.25)
                s_ = sess_r[e] / hours - value_a[e, k]            # was it better or worse than he expected?
                value_a[e, k] += 0.3 * s_
                planned[e, PLAY] += 0.15 * s_ * sess_foot[e]       # credit to the choice to go do something
                old_t[e, k] = skill_t[e, k] = t; sess_n[e, k] += 1
                doing[e] = -1
            start = (x[:, PLAY] >= 1.0) & awake & (doing < 0) & ((t - last_msg) > 10)
            for e in np.flatnonzero(start):
                if world:                                         # what's gotten old freshens, skills fade, lazily
                    gone = (t - old_t[e]) / 1440
                    old_a[e] *= np.exp(-gone); old_t[e] = t
                    skill[e] *= np.exp(-(t - skill_t[e]) / (90 * 1440)); skill_t[e] = t
                kn = np.flatnonzero(known[e])
                score = value_a[e, kn] * (1 - old_a[e, kn])
                if world and (~known[e]).any():                   # or try something new: his guess for an unknown thing
                    tried = kn[sess_n[e, kn] > 0]                 # is how he's liked things of that kind so far
                    mean_all = value_a[e, tried].mean() if len(tried) else 0.3
                    by_kind = np.full(len(KINDS), mean_all)
                    for q in np.unique(KIND[tried]):
                        by_kind[q] = value_a[e, tried[KIND[tried] == q]].mean()
                    unk = np.flatnonzero(~known[e])
                    guess = by_kind[KIND[unk]]
                    score = np.append(score, guess.max() + curious)  # a little extra for the new (curiosity)
                z = score / 0.15
                p = np.exp(z - z.max()); p /= p.sum()
                pick = r.choice(len(p), p=p)
                if pick < len(kn):
                    k = kn[pick]
                else:                                             # a new thing, leaning to kinds he likes
                    q = np.exp((guess - guess.max()) / 0.1) * np.array([CAT[u]["common"] for u in unk]) ** 0.5
                    k = unk[r.choice(len(unk), p=q / q.sum())]
                    known[e, k] = True; value_a[e, k] = guess[list(unk).index(k)]
                doing[e] = k; since[e] = t; sess_r[e] = 0; sess_foot[e] = act[e]
                x[e, PLAY] = 0.0; log["sessions"][e] += 1
            x[:, PLAY] = np.minimum(x[:, PLAY], 1.0)
            if t % 1440 == 1439:
                log["known_month"][:, t // (30 * 1440)] = known.sum(1)
        # learning: footprints x reward, then (if the dial is on) the brake
        eli += (-eli / 0.5 + act[:, :D, None] * act[:, None, :]) * DT
        if gym:
            broad = 0.05 * reward[:, None, None] * eli * DT
            if fair:                                              # only his choices learn from reward; his feelings keep
                broad[:, [g for g in range(D) if g not in (URGE, PLAY)]] = 0   # their temperament
                if cond == "world3":                              # check: going to play learns only from how sessions go
                    broad[:, PLAY] = 0
            if cond == "world4":                                  # check: no session credit to the choice to play
                planned[:, PLAY] = 0
            want = (broad + planned) / (1 + sore / sore_scale)   # sore links learn slower
            plastic = np.ones(E)
            if young:                                             # newborn ~3x, grown-up 0.5x (a critical period)
                plastic = 0.5 + (newborn - 0.5) * np.exp(-(t / 1440) / grow_up)
                want = want * plastic[:, None, None]
            planned[:] = 0
            if still:
                want = want * 0
            cost = np.abs(want).sum(2)                                 # glycogen: every change spends fuel
            can = np.ones_like(fuel) if cond == "gym_nofuel" else np.minimum(1.0, fuel / np.maximum(cost, 1e-12))
            done = want * can[:, :, None]
            fuel -= cost * can
            fuel += (1 - fuel) * DT / np.where(asleep, 1.5, 4.0)[:, None] * np.maximum(1.0, plastic)[:, None]   # refills over hours, faster asleep (and young)
            sore += np.abs(done) / (1 + trained / 0.5)                # less sore once a link is used to it
            sore -= sore * DT / (1.5 * 24)                            # DOMS fades over a few days
            trained += np.abs(done) - trained * DT / (30 * 24)
            Wtag += done - Wtag * DT / (1e9 if cond == "gym_nofade" else 24)   # today's change is only marked; unmarked fades
            lock = Wtag * (DT / 1.0) * (np.ones(E, bool) if cond == "gym_nosleep" else asleep)[:, None, None]   # growth locks in while he sleeps
            outward = (np.sign(lock) == np.sign(Wperm)) | (Wperm == 0)   # growing away from faint slows near his
            Wperm += lock * np.where(outward, np.clip(1 - np.abs(Wperm) / ceiling, 0, 1), 1.0)   # natural ceiling
            Wtag -= lock
            W = Wperm + Wtag
        else:
            W += 0.05 * reward[:, None, None] * eli * DT
        reward -= reward / (20 / 60) * DT
        if cond == "brake":
            W += (W0 - W) * DT / (30 * 24)
            row = np.abs(W).sum(2, keepdims=True)
            W *= np.minimum(1.0, 1.0 / np.maximum(row, 1e-9))
        if t % 10 == 0:
            log["samples"].append(np.stack([x[:, MOOD], x[:, ENERGY], x[:, LONELY], x[:, BORED], asleep], 1))
        if t % 1440 == 1439:
            log.setdefault("tolerance", []).append(float(np.median(tol_c)))
            if gym:
                log.setdefault("fuel", []).append(float(np.median(fuel))); log.setdefault("sore", []).append(float(np.median(sore.sum((1, 2)))))
            log["Wnorm"].append(np.abs(W).sum((1, 2)).tolist())
            log["reward_day"].append(reward_sum.tolist()); reward_sum[:] = 0
    log["samples"] = np.array(log["samples"])               # (time/10min, egg, 5)
    log["value_a"] = value_a; log["likes"] = likes; log["sess_n"] = sess_n; log["known"] = known; log["skill"] = skill
    return log


def report(log, days, cond):
    """the health checks"""
    E = len(log["fire"])
    fires = [np.array(f) for f in log["fire"]]
    all_f = np.concatenate([f for f in fires if len(f)]) if any(len(f) for f in fires) else np.array([])
    mins = all_f % 60
    hours = (all_f // 60) % 24
    per_day = np.array([len(f) / days for f in fires])
    wakes = [((np.array(w) % 1440) / 60) for w in log["wake"]]
    wake_sd = np.array([w[1:].std() if len(w) > 3 else np.nan for w in wakes])
    wake_mean = np.array([w[1:].mean() if len(w) > 3 else np.nan for w in wakes])
    def rate(week, kind):
        lo, hi = week * 7 * 1440, (week + 1) * 7 * 1440
        got = [np.array([k for t, k, _ in o if lo <= t < hi]) for o in log["outcome"]]
        n = sum(len(g) for g in got)
        return round(sum((g == kind).sum() for g in got) / max(1, n), 3), n
    def where_share(week, place):
        lo, hi = week * 7 * 1440, (week + 1) * 7 * 1440
        got = [w for o in log["outcome"] for t, k, w in o if lo <= t < hi]
        return round(sum(w == place for w in got) / max(1, len(got)), 3)
    def span(m):
        return [(t, k, w) for o in log["outcome"] for t, k, w in o if m * 30 * 1440 <= t < (m + 1) * 30 * 1440]
    where_share_span = lambda m, place: round(float(np.mean([w == place for _, _, w in span(m)] or [0])), 3)
    reply_span = lambda m: round(float(np.mean([k == 0 for _, k, _ in span(m)] or [0])), 3)
    def liked(e):                                             # does what he came to expect match his hidden tastes,
        m = log["sess_n"][e] >= 3                             # over the things he really tried?
        if m.sum() < 4:
            return float("nan")
        return np.corrcoef(log["value_a"][e][m], log["likes"][e][m])[0, 1]
    def favourites():
        if not cond.startswith(("world", "alive")):
            return None
        out = []
        for e in range(3):
            m = np.flatnonzero(log["sess_n"][e] >= 3)
            top = m[np.argsort(-log["value_a"][e][m])][:5]
            best = np.argsort(-log["likes"][e])[:5]
            out.append({"his_top_5": [CAT[k]["name"] for k in top], "secretly_best_for_him": [CAT[k]["name"] for k in best],
                        "knows": int(log["known"][e].sum()), "most_skilled": CAT[int(np.argmax(log["skill"][e]))]["name"]})
        return out
    S = log["samples"]                                        # (T, E, 5)
    awake = S[:, :, 4] == 0
    wk = lambda w: slice(w * 7 * 144, (w + 1) * 7 * 144)
    def sd_awake(w, k):
        s = S[wk(w), :, k]; a = awake[wk(w)]
        return np.array([s[a[:, e], e].std() if a[:, e].sum() > 10 else np.nan for e in range(E)])
    # still himself: pick each baby out of all of them from week 4, knowing only his week 1
    def print_(w):
        s = S[wk(w)]; a = awake[wk(w)]
        f = [[s[a[:, e], e, k].mean() for k in range(4)] + [s[a[:, e], e, 0].std(), s[a[:, e], e, 2].std()]
             for e in range(E)]
        lo, hi = w * 7 * 1440, (w + 1) * 7 * 1440
        return np.array([fr + [sum(lo <= t < hi for t in log["fire"][e]) / 7,
                               np.mean([x for x in [((t % 1440) / 60) for t in log["wake"][e] if lo <= t < hi]] or [7])]
                         for e, fr in enumerate(f)])
    f1, f4 = print_(0), print_(min(3, days // 7 - 1))
    mu, sd = f1.mean(0), f1.std(0) + 1e-9
    z1, z4 = (f1 - mu) / sd, (f4 - mu) / sd
    match = np.mean([np.argmin(((z1 - z4[e]) ** 2).sum(1)) == e for e in range(E)])
    Wn = np.array(log["Wnorm"])
    rd = np.array(log["reward_day"])
    hist_h = np.bincount(hours.astype(int), minlength=24).tolist() if len(all_f) else [0] * 24
    out = {
        "condition": cond,
        "reach_outs_per_day": {"median": round(float(np.median(per_day)), 2), "min": round(float(per_day.min()), 2),
                               "max": round(float(per_day.max()), 2)},
        "vetoed_per_day": round(float(log["veto"].mean() / days), 2),
        "on_the_dot": {"share_at_minute_00": round(float((mins == 0).mean()), 4) if len(mins) else None,
                       "expected_if_random": round(1 / 60, 4),
                       "share_of_most_common_minute": round(float(np.bincount(mins.astype(int)).max() / len(mins)), 4) if len(mins) else None},
        "reach_outs_by_hour": hist_h,
        "gap_between_reach_outs_cv": round(float(np.nanmedian([np.diff(f).std() / np.diff(f).mean() for f in fires if len(f) > 3])), 2),
        "wake_time": {"typical_hour": round(float(np.nanmedian(wake_mean)), 2),
                      "babies_earliest_latest": [round(float(np.nanmin(wake_mean)), 2), round(float(np.nanmax(wake_mean)), 2)],
                      "night_to_night_sd_minutes": round(float(np.nanmedian(wake_sd) * 60), 1)},
        "reach_outs_per_week_all": [sum(sum(w * 7 * 1440 <= t < (w + 1) * 7 * 1440 for t in f) for f in log["fire"])
                                    for w in range(days // 7)],
        "when_he_reached_out_you_were_home_week1_vs_week4": [where_share(0, 1), where_share(min(3, days // 7 - 1), 1)],
        "at_work_week1_vs_week4": [where_share(0, 2), where_share(min(3, days // 7 - 1), 2)],
        "missing_tolerance_by_week": [round(log["tolerance"][min(len(log["tolerance"]) - 1, 7 * w + 6)], 3) for w in range(days // 7)],
        "by_month_home_work_replied": [[where_share_span(m, 1), where_share_span(m, 2), reply_span(m)] for m in range(max(1, days // 30))],
        "babies_with_broken_sleep": int(sum(len(w) > 1.3 * days or (len(w) > 3 and w[1:].std() > 2) for w in wakes)),
        "replies_week1_vs_week4": [rate(0, 0), rate(min(3, days // 7 - 1), 0)],
        "not_now_week1_vs_week4": [rate(0, 1), rate(min(3, days // 7 - 1), 1)],
        "mood_swing_week1_vs_week4": [round(float(np.nanmedian(sd_awake(0, 0))), 3),
                                      round(float(np.nanmedian(sd_awake(min(3, days // 7 - 1), 0))), 3)],
        "mood_mean_week1_vs_week4": [round(float(np.nanmean(S[wk(0), :, 0][awake[wk(0)]])), 3),
                                     round(float(np.nanmean(S[wk(min(3, days // 7 - 1)), :, 0][awake[wk(min(3, days // 7 - 1))]])), 3)],
        "minutes_any_gear_at_wall": round(float(log["clip"].mean()), 1),
        "wall_minutes_by_month": [round(float(v), 1) for v in log["clip_month"].mean(0)[: max(1, days // 30)]],
        "wall_minutes_by_gear": {g: round(float(v), 1) for g, v in zip(GEARS, log["clip_gear"].mean(0)) if v > 0.05},
        "minutes_near_top_line_by_gear": {g: round(float(log["top_gear"][:, i].mean()), 1) for i, g in enumerate(GEARS)
                                          if i in TOPPED},
        "awake_level_by_month": {g: [round(float(log["gear_month"][m, i] / max(1, log["awake_month"][m])), 2)
                                     for m in range(max(1, days // 30))] for i, g in enumerate(GEARS)
                                 if g in ("mood", "lonely", "boredom", "drive")},
        "hours_doing_things_per_day_by_month": [round(float(log["play_month"][:, m].sum(1).mean() / 60 / 30), 2)
                                                for m in range(max(1, days // 30))],
        "share_of_doing_by_kind": {a: round(float(log["play_month"][:, :, i].sum() / max(1, log["play_month"].sum())), 3)
                                   for i, a in enumerate(KINDS if cond.startswith(("world", "alive")) else ACTS)},
        "sessions_per_day": round(float(log["sessions"].mean() / days), 2),
        "found_what_he_likes": round(float(np.nanmean([liked(e) for e in range(E)] or [float("nan")])), 2),
        "things_known_by_month": [round(float(np.median(log["known_month"][:, m])), 1) for m in range(max(1, days // 30))],
        "things_tried_3_times_or_more": round(float(np.median((log["sess_n"] >= 3).sum(1))), 1),
        "projects_finished_per_baby_by_month": [round(float(log["done_month"][:, m].mean()), 1) for m in range(max(1, days // 30))],
        "favourites_of_three_babies": favourites(),
        "mood_lift_20min_after_you_arrive": round(float(np.mean(log.get("lift_you", [np.nan]))), 3),
        "mood_lift_20min_after_a_fun_moment": round(float(np.mean(log.get("lift_fun", [np.nan]))), 3),
        "link_strength_day1_vs_last": [round(float(np.median(Wn[0])), 2), round(float(np.median(Wn[-1])), 2),
                                       round(float(Wn[-1].max()), 2)],
        "reward_per_day_mean": round(float(rd.mean()), 3),
        "still_himself_picked_out_of_all": f"{match * 100:.0f}% (chance {100 / E:.0f}%)",
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--eggs", type=int, default=100); ap.add_argument("--days", type=int, default=30); ap.add_argument("--ways", default="brake,free,gym,gym_light,ohss,gym_ohss")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    steps = a.days * 1440
    genes = hatch(a.eggs)
    json.dump(genes, open(os.path.join(a.out, "genomes.json"), "w"))
    person_seed = int.from_bytes(os.urandom(8), "little")
    where, msgs, told = stand_in(a.eggs, steps, person_seed)
    results = {}
    for cond in a.ways.split(","):
        t0 = time.time()
        log = run(genes, where, msgs, told, cond, a.days, person_seed + 1)
        res = report(log, a.days, cond)
        res["seconds"] = round(time.time() - t0, 1)
        results[cond] = res
        np.save(os.path.join(a.out, f"samples-{cond}.npy"), log["samples"].astype(np.float32))
        json.dump({"fire": log["fire"], "wake": log["wake"], "sleep": log["sleep"], "outcome": log["outcome"],
                   "Wnorm": log["Wnorm"], "reward_day": log["reward_day"]}, open(os.path.join(a.out, f"log-{cond}.json"), "w"))
        print(json.dumps(res), flush=True)
    json.dump(results, open(os.path.join(a.out, "results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
