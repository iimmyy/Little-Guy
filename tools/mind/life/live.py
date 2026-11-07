#!/usr/bin/env python3
"""Little Guy's body lives simulated days, minute by minute, for the life model to learn from (life model v0).

Each worker lives one continuous life of the same body (the average egg): days follow each other and nothing is reset.
The days vary the way a life does: when he sleeps, company, being judged, a scare, exercise, something new, a
rejection, coffee, and now and then a pill. Every minute two things are written down: his body (what the engine's
chemistry is doing) and his world (what is happening to him, the time of day, and a pulse when he takes something).

A run can also give every life a different person (genes): each listed gene is multiplied by e^N(0, spread), so with
spread 0.25 two people typically differ by about a quarter in stress reactivity, sleep build-up, body attention, etc.
The spread is a guess, not a measured population spread. With a set number of days each life stops there instead of
at the time limit, and the last test_lives lives are marked to be held out as strangers.

usage: live.py <engine folder> <out folder> <workers> <seconds> [seed] [lives days spread test_lives]
"""
import inspect, json, math, os, sys, time
from multiprocessing import Pool

import numpy as np

# pills taken now and then, with rough elimination half-lives in hours (drug labels, rounded). The engine holds these
# drugs steady once given, so the life lets them fade itself; they are here for variety, not as pharmacokinetics
PILLS = {"propranolol": 4.0, "diphenhydramine": 6.0, "guanfacine": 17.0, "methylphenidate": 3.0,
         "benzodiazepine": 11.0, "modafinil": 13.0, "scopolamine": 5.0, "yohimbine": 1.0}
EVENTS = ["company", "touch", "judged", "threat", "exercise", "novelty", "rejection"]
WORLD = ["awake"] + EVENTS + ["took_caffeine"] + ["took_" + p for p in PILLS] + ["clock_sin", "clock_cos"]
BODY_EXTRA = ["sleep_pressure", "heart_rate", "blood_pressure", "worry", "amygdala"]
# genes that make one person differ from another (engine GENES); endozepine stays 0 (only for the flumazenil test)
PERSON = ["lc_drive", "da_tone", "ht_tone", "ach_tone", "ha_tone", "ox_cells", "gaba_tone", "npy_tone", "ecb_tone",
          "mor_tone", "ot_tone", "stress_gain", "reward_gain", "social_gain", "ne_ox_link", "gr_feedback", "net",
          "dat", "sert", "body_attention", "worry_trait", "sleep_rise", "sleep_fall"]
DAY = 1440
# the engine's numerical slice in hours, for engines that have one (5984d8e on: 3.75 s by default). One minute gives the
# same body here: awake blood pressure max 105.1 / median 98.5 over two lived days at 60 s, 15 s and the default, and
# a new person is ready in 23 s instead of minutes (tokyo-test, 2026-10-24)
SLICE = 1 / 60


def plan_day(rng, d, wake):
    """day d's bed time (absolute minutes; None for a night awake), the next morning's wake time, and the day's
    events (start, end, name, level) and doses (minute, name, dose), all inside his waking hours"""
    weekend = d % 7 in (5, 6)
    bed_h = float(np.clip(rng.normal(23.0, 0.5), 21.5, 24.5)) + 0.5 * weekend
    if rng.random() < 0.10:
        bed_h += rng.uniform(1.5, 3.0)                       # a late night
    end = int(d * DAY + bed_h * 60)
    bed = None if rng.random() < 0.04 else end               # 4% of nights he stays up all night
    wake_h = float(np.clip(rng.normal(7.0, 0.4), 5.5, 9.0)) + ((d + 1) % 7 in (5, 6))
    wake_next = int((d + 1) * DAY + wake_h * 60)
    events, doses = [], []

    def block(name, lo, hi, level=1.0, within=None):
        a, b = within or (wake, end)
        n = int(rng.integers(lo, hi + 1))
        s = int(rng.integers(a, max(a + 1, b - n)))
        events.append((s, s + n, name, level))
        return s, s + n

    if rng.random() < 0.12:
        block("judged", 10, 20)
    if rng.random() < 0.10:
        block("threat", 5, 30, rng.uniform(0.2, 0.8))
    if rng.random() < 0.25:
        block("exercise", 30, 60)
    if rng.random() < 0.05:
        block("rejection", 10, 30, rng.uniform(0.3, 1.0))
    for _ in range(rng.poisson(2.0)):
        s, t = block("company", 15, 120)
        if rng.random() < 0.3:
            block("touch", 5, 20, within=(s, t))
    for _ in range(rng.poisson(1.5)):
        block("novelty", 5, 20, rng.uniform(0.3, 1.0))
    if rng.random() < 0.6:
        doses.append((wake + int(rng.integers(0, 60)), "caffeine", 1.0))
        if rng.random() < 0.25:
            doses.append((d * DAY + int(rng.integers(13 * 60, 15 * 60)), "caffeine", 0.5))
    if rng.random() < 0.08:
        doses.append((int(rng.integers(wake, end)), str(rng.choice(list(PILLS))), float(rng.uniform(0.5, 1.0))))
    return bed, wake_next, events, doses


class Life:
    """one continuous life of the body: the engine, and the days the world brings it, a minute at a time"""

    def __init__(self, chem, seed, genes=None):
        self.rng = np.random.default_rng(seed)
        kw = dict(max_step=SLICE) if "max_step" in inspect.signature(chem.Chem).parameters else {}
        self.e = chem.Chem(genes=genes, hour=8.0, **kw)
        self.keys = list(self.e.c)
        self.m, self.d, self.wake, self.last_bed, self.pills, self.log = 8 * 60, 0, 8 * 60, None, {}, []
        self._plan()

    def _plan(self):
        self.bed, self.wake_next, self.events, self.doses = plan_day(self.rng, self.d, self.wake)
        self.log.append(dict(day=self.d, wake=self.wake, bed=self.bed, events=self.events, doses=self.doses))

    def asleep(self):
        # asleep from last night's bed (which can fall after midnight) until this morning's wake, and from tonight's
        m = self.m
        return (self.last_bed is not None and self.last_bed <= m < self.wake) or (self.bed is not None and m >= self.bed)

    def body(self):
        e = self.e
        return [e.c[x] for x in self.keys] + [e.S, e.heart_rate(), e.blood_pressure(), e.worry_st, e.amy_out()]

    def minute(self, add=None, block=()):
        """live one minute. add: inputs from what he does (they act only while he is awake); block: world inputs he
        shuts out. Returns his body and his world for this minute"""
        e, m, asleep = self.e, self.m, self.asleep()
        inp = {name: level for s, t, name, level in self.events if s <= m < t and name not in block}
        for name, level in ((add or {}).items() if not asleep else ()):
            inp[name] = max(inp.get(name, 0.0), level)
        w = np.zeros(len(WORLD), np.float32)
        for name, level in inp.items():
            if name in WORLD:
                w[WORLD.index(name)] = level
        for at, name, dose in self.doses:
            if at == m:
                w[WORLD.index("took_" + name)] = dose
                if name == "caffeine":
                    e.give(name, dose)                       # the engine clears caffeine itself
                else:
                    self.pills[name] = self.pills.get(name, 0.0) + dose
        for name in list(self.pills):
            self.pills[name] *= 2 ** (-1 / (60 * PILLS[name]))
            if self.pills[name] < 0.01:
                del self.pills[name]
                e.stop(name)
            else:
                e.drugs[name] = self.pills[name]
        w[0] = not asleep
        w[-2], w[-1] = math.sin(2 * math.pi * (m % DAY) / DAY), math.cos(2 * math.pi * (m % DAY) / DAY)
        e.step(1 / 60, awake=not asleep, **inp)
        self.m += 1
        if self.m == (self.d + 1) * DAY:
            self.last_bed, self.wake, self.d = self.bed, self.wake_next, self.d + 1
            self._plan()
        return self.body(), w


def life(args):
    engine_dir, out, k, seconds, seed, days, spread = args
    sys.path.insert(0, os.path.abspath(engine_dir))
    import engine as chem
    t0 = time.time()
    g = np.random.default_rng(seed + 7_000_000)
    genes = {n: round(chem.GENES[n] * float(np.exp(g.normal(0.0, spread))), 4) for n in PERSON} if spread else None
    lf = Life(chem, seed, genes)
    setup = time.time() - t0
    body, world = [], []
    while (lf.d < days) if days else (time.time() - t0 < seconds):   # stops only at a day's end: every day is whole
        for _ in range((lf.d + 1) * DAY - lf.m):
            b, w = lf.minute()
            body.append(b)
            world.append(w)
    np.savez_compressed(os.path.join(out, f"life_{k:02d}.npz"),
                        body=np.asarray(body, np.float32), world=np.asarray(world, np.float32))
    with open(os.path.join(out, f"life_{k:02d}.json"), "w") as f:
        json.dump(dict(seed=seed, genes=genes, days=lf.d, setup_s=round(setup, 1), seconds=round(time.time() - t0, 1),
                       body=lf.keys + BODY_EXTRA, days_log=lf.log[:-1]), f)
    return k, lf.d, lf.keys + BODY_EXTRA


if __name__ == "__main__":
    engine_dir, out, workers, seconds = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    lives, days, spread, test_lives = ((int(sys.argv[6]), int(sys.argv[7]), float(sys.argv[8]), int(sys.argv[9]))
                                       if len(sys.argv) > 9 else (workers, 0, 0.0, 1))
    os.makedirs(out, exist_ok=True)
    with Pool(workers) as pool:
        done = pool.map(life, [(engine_dir, out, k, seconds, seed * 1000 + k, days, spread) for k in range(lives)],
                        chunksize=1)
    json.dump(dict(engine=os.path.basename(os.path.abspath(engine_dir)), workers=lives, seconds=seconds,
                   spread=spread, test_lives=test_lives,
                   days={k: d for k, d, _ in done}, body=done[0][2], world=WORLD),
              open(os.path.join(out, "meta.json"), "w"), indent=1)
    print("lived", sum(d for _, d, _ in done), "days in", workers, "lives:", {k: d for k, d, _ in done})
