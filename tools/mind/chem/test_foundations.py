#!/usr/bin/env python3
"""Section F of the chemistry foundations: is each system right on its own, before anything multiplies?

Three kinds of test, all on engine.py with no language model:
  bio   biology signatures (DA prediction errors, the LC brake's timing, sleep pressure, cortisol's two receptors...)
  gain  temperament tests: a calm day must look the same, reactivity and the pattern across read-outs must not
  drug  drug challenges: each drug acts only as receptor actions, and what comes out must match people

The pass rule of every test is written here before it runs and is never loosened to fit a result. A failing test is
reported as failing, with its numbers; the engine is fixed by mechanism, never by special-casing a test.
Each test also says how sure the expected result is: checked (read in the source), known (textbook), theory.

usage: test_foundations.py <out_dir>      writes results.json and report.md there
"""
import copy
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine as E

# "wired" as Immy's paper defines it (2026-10-22; §8, §5.2.1, §3.4, §4): an NE system that is activated but poorly
# regulated: the drive stays up AND the α2A/GIRK brake is worn, with the CRF/orexin loops feeding back into the LC.
# Drive and brake are separate genes; "wired" is their interaction. The paper gives no numbers, so the levels are
# picked by WIRED_RULE from the four features it names (the wired_* tests and high_tonic_pattern), never from a drug
# result: the mildest pair (smallest |ln drive| + |ln brake|) that shows all four. Picked in
# .run/results/mind/chem/*-wired-grid/ (2026-10-22).
WIRED_RULE = "mildest (lc_drive, rx_a2a_auto) pair passing wired_background, wired_exaggerated, wired_persists, high_tonic_pattern"
WIRED_DRIVE, WIRED_BRAKE = 1.6, 1.0     # placeholder until the grid has run
EGGS = {
    "average": {},
    "high_gain": dict(stress_gain=1.6),                                   # calm the same, reacts more
    # driven: an LC pushed hard with its brake intact (Aston-Jones high-tonic mode). This was "wired" in runs 1-7;
    # it stays as a comparison subject
    "driven": dict(lc_drive=1.6, stress_gain=1.5),
    "wired": dict(lc_drive=WIRED_DRIVE, rx_a2a_auto=WIRED_BRAKE, stress_gain=1.5),
    # ~20% less net GABA (Goddard 2001: -22% cortical GABA in panic disorder), partly covered by an endozepine
    # (theory, Nutt 1990), low NPY, high orexin (Johnson 2010)
    # body_attention 1.75: the ratio of mean Anxiety Sensitivity Index scores, panic disorder (about 33.3) to the
    # non-clinical norm (19.0): Taylor, Koch & McNally 1992, Peterson & Reiss (change 25)
    "panic_prone": dict(gaba_tone=0.6, endozepine=0.3, npy_tone=0.6, ox_cells=1.3, body_attention=1.75),
    "ox_null": dict(ox_cells=0.0),                                        # orexin cells gone (narcolepsy model)
    "high_orexin": dict(ox_cells=1.6),
}
TESTS = []


def test(group, expect, evidence):
    def reg(fn):
        TESTS.append(dict(fn=fn, group=group, name=fn.__name__, expect=expect, evidence=evidence))
        return fn
    return reg


def egg(kind="average", hour=8.0, **genes):
    return E.Chem(dict(EGGS[kind], **genes), hour=hour)


def awake_sched(h):
    return 7.0 <= h < 23.0


def live(e, hours, inp=None, dt=1 / 60, sched=True, every=None):
    """run `hours`. inp: dict or function(hour) -> dict of inputs. sched: sleep 23-7. every: record every N hours.
    returns the last knobs and a trace [(hours since start, knobs, concentrations)]"""
    trace, n = [], int(round(hours / dt))
    k = e.knobs()
    for i in range(n):
        x = dict(inp(e.hour) if callable(inp) else (inp or {}))
        x.setdefault("awake", awake_sched(e.hour) if sched else True)
        k = e.step(dt, **x)
        if every and (i + 1) % max(1, int(round(every / dt))) == 0:
            trace.append(((i + 1) * dt, k, dict(e.c)))
    return k, trace


def until(e, hour, inp=None, dt=1 / 60, sched=True):
    """run until the clock reads `hour`"""
    return live(e, ((hour - e.hour) % 24) or 24, inp, dt, sched)[0]


_LIVED = {}


def _scheduled_minutes(e, start_minute, minutes, stressful_evening=False):
    """Advance a declared day without floating-point drift at sleep boundaries."""
    for minute in range(start_minute, start_minute + minutes):
        hour = (minute % (24 * 60)) / 60
        threat = 0.5 if stressful_evening and 20 <= hour < 21 else 0.0
        e.step(1 / 60, awake=awake_sched(hour), threat=threat)


def lived_egg(kind="average", hour=8.0, **genes):
    """One full normal day/night from 07:00, then the requested test-day hour.

    Cache by genes, not the mutable temperament name used by the wired grid.
    Every caller receives a deep copy and cannot change another test's history.
    """
    resolved = dict(EGGS[kind], **genes)
    key = tuple(sorted(dict(E.GENES, **resolved).items()))
    if key not in _LIVED:
        e = egg(kind, hour=7.0, **genes)
        _scheduled_minutes(e, 7 * 60, 24 * 60)
        _LIVED[key] = {0: e}
    minute = int(round(((hour - 7.0) % 24) * 60))
    cached = _LIVED[key]
    if minute not in cached:
        previous = max(t for t in cached if t <= minute)
        e = copy.deepcopy(cached[previous])
        _scheduled_minutes(e, 7 * 60 + previous, minute - previous)
        cached[minute] = e
    return copy.deepcopy(cached[minute])


def r3(x):
    if isinstance(x, dict):
        return {k: r3(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [r3(v) for v in x]
    return round(x, 3) if isinstance(x, float) else x


# a long stress stretch, made once and shared: 4 weeks of daily stress (threat 0.5, 10 AM - 4 PM), sleeping 23-7
_LONG = {}


def long_stressed(kind="average"):
    if kind not in _LONG:
        e = egg(kind, hour=7.0)
        live(e, 28 * 24, lambda h: dict(threat=0.5 if 10 <= h < 16 else 0.0), dt=1 / 12)
        until(e, 10.0, dt=1 / 12)                     # the next morning, no stress
        _LONG[kind] = e
    return copy.deepcopy(_LONG[kind])


# ======================================================================== biology signatures
@test("bio", "DA: a burst for an unpredicted reward, none once it is fully predicted (D1 3.5% at rest, ~25% in a "
      "burst; D2 75% -> >95%)", "checked (Schultz 1997; Dreyer 2010)")
def da_predicted():
    e = egg()
    first = e.reward_event("bell", 1.0)
    for _ in range(28):
        e.reward_event("bell", 1.0)
    last = e.reward_event("bell", 1.0)
    ok = (first["d1"] >= 5 * first["d1_rest"] and first["d2"] >= 0.95 and last["d1"] <= 1.3 * last["d1_rest"])
    return ok, dict(d1_rest=first["d1_rest"], d1_first=first["d1"], d2_first=first["d2"], d1_after_30=last["d1"],
                    delta_first=first["delta"], delta_after_30=last["delta"])


@test("bio", "DA: a dip when an expected reward is missing, read by D2 (falls well below its 75%)", "checked (Schultz 1997)")
def da_omission():
    e = egg()
    for _ in range(30):
        e.reward_event("bell", 1.0)
    miss = e.reward_event("bell", 0.0)
    return miss["d2"] < 0.5 * miss["d2_rest"], dict(d2_rest=miss["d2_rest"], d2_dip=miss["d2"], delta=miss["delta"])


@test("bio", "DA: a novelty burst that fades with repetition", "checked (Kakade & Dayan 2002; Ljungberg 1992)")
def da_novelty():
    e = egg()
    d = [e.reward_event("new sound", 0.0)["delta"] for _ in range(10)]
    return d[0] > d[4] > d[9] and d[9] < 0.3 * d[0], dict(deltas=d)


@test("bio", "NE: a burst then a pause (the gap after a burst is longer than the tonic gap)",
      "checked (Aston-Jones 1994: brief activation then short depression)")
def ne_burst_pause():
    t, rel, _ = E.lc_spikes(2.0, burst_at=4.0, seconds=8.0)
    pre = [b - a for a, b in zip(t[:-1], t[1:]) if 2.0 <= a and b < 3.99]
    after = [x for x in t if x > 4.09]
    gap = after[0] - 4.08
    return gap > 1.5 * (sum(pre) / len(pre)), dict(tonic_gap_s=sum(pre) / len(pre), gap_after_burst_s=gap)


@test("bio", "Brake timing (C.2 rule 1): the α2A brake cuts tonic release but a burst (3 spikes in 80 ms) gets "
      "through whole", "checked (GIRK onset 58-154 ms, peak ~0.9 s)")
def brake_timing():
    _, rel_t, _ = E.lc_spikes(3.0, seconds=4.0)
    _, rel_b, _ = E.lc_spikes(0.01, burst_at=1.0, seconds=2.0)
    return rel_t[5] / rel_t[0] < 0.8 and rel_b[2] / rel_b[0] > 0.95, dict(
        tonic_6th_vs_1st=rel_t[5] / rel_t[0], burst_3rd_vs_1st=rel_b[2] / rel_b[0])


@test("bio", "NE: a busy (high tonic) LC gives weaker bursts", "checked (Aston-Jones, Rajkowski & Cohen 1999)")
def high_tonic_weak_burst():
    _, rel_lo, _ = E.lc_spikes(1.0, burst_at=5.0, seconds=6.0)
    t_hi, rel_hi, _ = E.lc_spikes(4.0, burst_at=5.0, seconds=6.0)
    _, rel_lo_t = E.lc_spikes(1.0, burst_at=5.0, seconds=6.0)[:2]
    t_lo = E.lc_spikes(1.0, burst_at=5.0, seconds=6.0)[0]
    b_lo = sum(r for x, r in zip(t_lo, rel_lo) if 4.999 <= x <= 5.081)
    b_hi = sum(r for x, r in zip(t_hi, rel_hi) if 4.999 <= x <= 5.081)
    return b_hi < 0.85 * b_lo, dict(burst_release_1hz=b_lo, burst_release_4hz=b_hi)


@test("bio", "Arousal: the inverted U of deliberate control appears from α2A vs α1 occupancy (written nowhere); a drowsy "
      "or a revved brain turns its arousal systems (LC, DA, orexin, histamine, ACh) down or up together, and an alert "
      "calm egg sits near the top (the quiet end corrected 2026-10-22 with Immy: an under-aroused brain is not low in NE alone)",
      "checked (Arnsten 2009) / change 28")
def inverted_u():
    drives = [0.3, 0.5, 0.7, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0, 4.0]
    ne_only = []                                           # the old sweep: only the LC (report-only since change 28)
    for d in drives:
        e = egg(lc_drive=d)
        ne_only.append((e.c["ne"], e.knobs()["deliberate"]))
    out = []
    for d in drives:                                       # change 28: a drowsy or a revved brain turns ALL its arousal systems
        e = egg(lc_drive=d, da_tone=d, ox_cells=d, ha_tone=d, ach_tone=d)
        out.append((e.c["ne"], e.knobs()["deliberate"]))
    vals = [v for _, v in out]
    top = max(vals)
    i = vals.index(top)
    ok = 0 < i < len(vals) - 1 and vals[0] < 0.9 * top and vals[-1] < 0.9 * top and vals[3] >= 0.95 * top
    stress = {}
    for th in (0.0, 0.25, 0.5, 1.0):
        e = egg()
        k, _ = live(e, 0.5, dict(threat=th), sched=False)
        stress[th] = (e.c["ne"], k["deliberate"])
    return ok, dict(ne=[n for n, _ in out], deliberate=vals, peak_at_lc_drive=drives[i],
                    report_only_ne_only_sweep=dict(ne=[n for n, _ in ne_only], deliberate=[v for _, v in ne_only]),
                    report_only_threat_sweep_ne_and_deliberate=stress)


@test("bio", "NE: useful bursts are biggest at a middle background level (too quiet = drowsy, too loud = braked)",
      "checked (Aston-Jones, Rajkowski & Cohen 1999)")
def burst_middle():
    drives = [0.1, 0.2, 0.4, 0.7, 1.0, 1.6, 2.5, 4.0]
    b = [egg(lc_drive=d).knobs()["burst"] for d in drives]
    i = b.index(max(b))
    return 0 < i < len(b) - 1, dict(lc_drive=drives, burst=b)


@test("bio", "Adenosine: the process S curve, rising with τ ~18.2 h awake and falling with τ ~4.2 h asleep",
      "checked (Daan, Beersma & Borbely 1984)")
def adenosine():
    e = egg()
    s0 = e.S
    live(e, 16, sched=False)
    tau_rise = -16 / math.log((1 - e.S) / (1 - s0))
    s1 = e.S
    live(e, 8, dict(awake=False), sched=False)
    tau_fall = -8 / math.log(e.S / s1)
    return abs(tau_rise - 18.2) < 0.9 and abs(tau_fall - 4.2) < 0.2, dict(tau_rise_h=tau_rise, tau_fall_h=tau_fall)


@test("bio", "Cortisol: MR stays filled all day (>70%), GR fills only at peaks; the peak comes just after waking",
      "known (de Kloet 1998; Pruessner 1997)")
def cortisol_day():
    e = egg(hour=7.0)
    _, tr = live(e, 48, every=0.25)
    day2 = [(7 + t) % 24 for t, _, _ in tr[96:]]
    cort = [c["cort"] for _, _, c in tr[96:]]
    mr = [c / (c + 0.1) for c in cort]
    gr = [c / (c + 1.0) for c in cort]
    peak_h, low_h = day2[cort.index(max(cort))], day2[cort.index(min(cort))]
    ok = (min(mr) > 0.7 and min(gr) < 0.35 and max(gr) > 0.5 and 7.0 <= peak_h <= 8.5
          and (low_h >= 21 or low_h <= 3))
    return ok, dict(mr_min=min(mr), gr_min=min(gr), gr_max=max(gr), peak_hour=peak_h, low_hour=low_h)


@test("bio", "Orexin: B's effect fades faster than A's (when the cells stop, B halves in minutes, A in hours)",
      "checked (orexin-B is cleared much faster than orexin-A)")
def orexin_ab():
    e = egg(hour=20.0)
    until(e, 23.0)
    a0, b0 = e.c["oxa"], e.c["oxb"]
    half = {}
    for i in range(1, 300):
        e.step(1 / 60, awake=False)
        for key, v0 in (("oxa", a0), ("oxb", b0)):
            if key not in half and e.c[key] < v0 / 2 + 0.05 * 0.5:
                half[key] = i
    return half.get("oxb", 999) < 10 and half.get("oxa", 0) > 60, dict(minutes_to_half=half)


# ======================================================================== gain tests
@test("gain", "A calm day is not a test: a high-gain egg and an average egg look the same on a calm day",
      "theory (the gain rule, C.2 rule 2)")
def calm_day_same():
    a, h = egg(hour=7.0), egg("high_gain", hour=7.0)
    _, ta = live(a, 24, every=1)
    _, th = live(h, 24, every=1)
    worst = 0.0
    for (_, ka, ca), (_, kh, ch) in zip(ta, th):
        for key in ("ne", "cort", "crh", "oxa"):
            worst = max(worst, abs(ch[key] - ca[key]) / max(1e-6, abs(ca[key])))
        for key in ("anxious", "pupil", "startle"):
            worst = max(worst, abs(kh[key] - ka[key]) / max(1e-6, abs(ka[key])))
    return worst < 0.01, dict(largest_relative_difference=worst)


def _tsst(kind):
    e = egg(kind, hour=9.5)
    until(e, 10.0)
    base_c, base_n = e.c["cort"], e.c["ne"]
    _, tr1 = live(e, 0.25, dict(judged=1.0), every=1 / 60)
    _, tr2 = live(e, 1.75, every=1 / 60)
    tr = tr1 + [(t + 0.25, k, c) for t, k, c in tr2]
    cort = [c["cort"] for _, _, c in tr]
    ne = [c["ne"] for _, _, c in tr]
    t_peak = tr[cort.index(max(cort))][0]
    return max(cort) - base_c, max(ne) - base_n, base_c, t_peak


@test("gain", "Reactivity (the Trier idea): 15 minutes of being judged; a high-gain egg's cortisol and NE rise more "
      "per unit of stressor; cortisol rises a real amount (>30%) and peaks 20-60 min after the start",
      "checked (Dickerson & Kemeny 2004) / theory (gain)")
def reactivity():
    ca, na, base, tpk = _tsst("average")
    ch, nh, _, _ = _tsst("high_gain")
    ok = ch >= 1.3 * ca and nh >= 1.3 * na and ca >= 0.3 * base and 20 / 60 <= tpk <= 1.0
    return ok, dict(cort_rise_avg=ca, cort_rise_high=ch, ne_rise_avg=na, ne_rise_high=nh, cort_base=base,
                    cort_peak_minutes=tpk * 60)


@test("gain", "Running total: two weeks with a stressful hour every day leave a higher month-long cortisol total "
      "(what hair cortisol reads)", "known (Stalder 2017)")
def running_total():
    a, s = egg(hour=7.0), egg(hour=7.0)
    live(a, 14 * 24, dt=1 / 12)
    live(s, 14 * 24, lambda h: dict(judged=1.0 if 10 <= h < 11 else 0.0), dt=1 / 12)
    return s.cort_total > 1.05 * a.cort_total, dict(calm=a.cort_total, stressed=s.cort_total)


@test("gain", "Pupil: resting size (background) and the change to an event (burst) are two read-outs that move apart "
      "in a high-tonic (driven) egg", "checked (Joshi 2016: a clue, not proof)")
def pupil():
    ka, kw = egg().knobs(), egg("driven").knobs()
    ok = kw["pupil"] > 1.15 * ka["pupil"] and kw["burst"] < 0.9 * ka["burst"]
    return ok, dict(rest_avg=ka["pupil"], rest_wired=kw["pupil"], evoked_avg=ka["burst"], evoked_wired=kw["burst"])


@test("gain", "Startle: the same on a calm day, bigger under threat in a high-gain egg (fear-potentiated startle)",
      "known (Grillon 2008) / theory (gain)")
def startle():
    out = {}
    for kind in ("average", "high_gain"):
        e = egg(kind)
        calm = e.knobs()["startle"]
        k, _ = live(e, 0.5, dict(threat=0.5))
        out[kind] = (calm, k["startle"] - calm)
    ok = abs(out["high_gain"][0] - out["average"][0]) < 0.01 * out["average"][0] and \
        out["high_gain"][1] >= 1.3 * out["average"][1]
    return ok, dict(calm_avg=out["average"][0], calm_high=out["high_gain"][0], rise_avg=out["average"][1],
                    rise_high=out["high_gain"][1])


@test("gain", "The driven pattern, all three at once: loud background + weak useful bursts (smaller in absolute terms) "
      "+ worse focus. This is the original high_tonic_pattern rule, unchanged, now on the driven egg (LC pushed hard, "
      "brake intact), which is what it described in runs 1-7", "checked (Aston-Jones & Cohen 2005)")
def driven_pattern():
    ka, kd = egg().knobs(), egg("driven").knobs()
    ok = kd["pupil"] > 1.15 * ka["pupil"] and kd["burst"] < 0.9 * ka["burst"] and kd["focus"] < 0.9 * ka["focus"]
    return ok, dict(background=(ka["pupil"], kd["pupil"]), burst=(ka["burst"], kd["burst"]),
                    focus=(ka["focus"], kd["focus"]), deliberate=(ka["deliberate"], kd["deliberate"]))


@test("gain", "The wired pattern: loud background + REDUCED PHASIC-TO-TONIC CONTRAST (burst / background under 0.9x "
      "an average egg's). Corrected 2026-10-22 with Immy: §3.4 of his paper says useful signals lose contrast against "
      "the raised background even when their release is intact, so the old endpoint 'weak useful bursts' (read as "
      "smaller absolute bursts; runs 1-7, kept as driven_pattern) assessed that mechanism wrongly. A correction to how "
      "the mechanism is assessed, not independent validation. Background and absolute burst are reported separately",
      "what Immy sent (§3.4, §8) / checked (Aston-Jones & Cohen 2005)")
def high_tonic_pattern():
    ka, kw = egg().knobs(), egg("wired").knobs()
    ca, cw = ka["burst"] / ka["pupil"], kw["burst"] / kw["pupil"]
    ok = kw["pupil"] > 1.15 * ka["pupil"] and cw < 0.9 * ca
    return ok, dict(background=(ka["pupil"], kw["pupil"]), burst_absolute=(ka["burst"], kw["burst"]),
                    phasic_to_tonic_contrast=(ca, cw), deliberate=(ka["deliberate"], kw["deliberate"]))


def _ne_challenge(kind, minutes, threat=0.5):
    """an egg and its unchallenged twin from 10 AM; threat for `minutes`. Returns both (NE excess = egg - twin, so the
    time of day never counts as a response)"""
    e, twin = egg(kind, hour=9.0), egg(kind, hour=9.0)
    until(e, 10.0, sched=False); until(twin, 10.0, sched=False)
    peak = 0.0
    for _ in range(int(minutes)):
        e.step(1 / 60, awake=True, threat=threat); twin.step(1 / 60, awake=True)
        peak = max(peak, e.c["ne"] - twin.c["ne"])
    return e, twin, peak


@test("wired", "Wired, quiet: background NE modestly up at rest (+10% to +50% over an average egg; a resting floor "
      "should sit below a strong acute stressor, which raises rat prefrontal NE +50-115%)",
      "what Immy sent (§5.2.1) / cutoff: modelling choice")
def wired_background():
    a, w = egg(hour=10.0), egg("wired", hour=10.0)
    r = w.c["ne"] / a.c["ne"]
    return 1.10 <= r <= 1.50, dict(rest_ne_avg=a.c["ne"], rest_ne_wired=w.c["ne"], ratio=r)


@test("wired", "Wired, challenged: exaggerated stress response (10 minutes of threat 0.5: NE rise at least 1.3x an "
      "average egg's)", "what Immy sent (§5.2.1) / checked (Simson & Weiss 1987: a weaker α2 brake = bigger LC "
      "responses) / cutoff: the same 1.3x as the other gain tests")
def wired_exaggerated():
    ra, rw = _ne_challenge("average", 10)[2], _ne_challenge("wired", 10)[2]
    return rw >= 1.3 * ra, dict(ne_rise_avg=ra, ne_rise_wired=rw, ratio=rw / ra)


@test("wired", "Lived-day wired: after 30 minutes of threat 0.5 starting at 10:00, 15:00 and 20:00, the anxious "
      "read-out (what the receptors set off inside the cells, which outlasts NE) takes at least 1.5x average's time to "
      "fall below 10% of its own rise at ALL three times; NE and sympathetic recovery are reported only",
      "user-prespecified clock robustness test / read-out corrected 2026-10-22 with Immy: NET clears NE in seconds and "
      "the felt state outlasts it (change 21) / 1.5x cutoff: modelling choice; ENGINE-CHANGES.md")
def wired_persists():
    read = {"anxious": lambda e: e.knobs()["anxious"], "sympathetic": lambda e: e.knobs()["sympathetic"],
            "ne": lambda e: e.c["ne"]}

    def comparison(hour, fresh=False):
        out = {}
        for kind in ("average", "wired"):
            if fresh:
                e, twin, _ = _ne_challenge(kind, 30)
            else:
                e = lived_egg(kind, hour=hour)
                twin = copy.deepcopy(e)
                for _ in range(30):
                    e.step(1 / 60, awake=True, threat=0.5)
                    twin.step(1 / 60, awake=True)
            rise = {n: f(e) - f(twin) for n, f in read.items()}
            back = {n: 12 * 60 for n in read}
            trace = [dict(minute=0, **rise)]
            for i in range(1, 12 * 60):
                e.step(1 / 60, awake=True); twin.step(1 / 60, awake=True)
                excess = {n: f(e) - f(twin) for n, f in read.items()}
                trace.append(dict(minute=i, **excess))
                for n in read:
                    if back[n] == 12 * 60 and excess[n] < 0.1 * rise[n]:
                        back[n] = i
                if all(b < 12 * 60 for b in back.values()):
                    break
            out[kind] = dict(rise=rise, minutes_back=back, nonpositive_rise={n: rise[n] <= 0 for n in read},
                             censored={n: back[n] == 12 * 60 for n in read}, trace=trace)
        a, w = out["average"]["minutes_back"], out["wired"]["minutes_back"]
        return dict(passed=w["anxious"] >= 1.5 * a["anxious"], rise_avg=out["average"]["rise"]["anxious"],
                    minutes_back_avg=a["anxious"], rise_wired=out["wired"]["rise"]["anxious"],
                    minutes_back_wired=w["anxious"], report_only_minutes_back=dict(average=a, wired=w), eggs=out)

    by_hour = {hour: comparison(hour) for hour in (10, 15, 20)}
    return all(row["passed"] for row in by_hour.values()), dict(
        by_hour=by_hour, report_only_fresh_10=comparison(10, fresh=True))


@test("bio", "The body-awareness loop: pointing out his body makes a fright settle later, a more body-aware egg gains "
      "more minutes from it, and even a strong loop never runs away (settles within 12 hours)",
      "theory (Immy 2026-10-22; Clark 1986; Ehlers & Breuer 1992) / mechanism test, written before the run")
def body_loop():
    def settle(reminded=0.0, attention=1.0, loop=None):
        keep = E.BODY_LOOP
        if loop is not None:
            E.BODY_LOOP = loop
        try:
            e = lived_egg("average", hour=10, body_attention=attention)
            twin = copy.deepcopy(e)
            for _ in range(30):
                e.step(1 / 60, awake=True, threat=0.5, body_reminded=reminded)
                twin.step(1 / 60, awake=True)
            rise = e.knobs()["anxious"] - twin.knobs()["anxious"]
            for i in range(1, 12 * 60):
                e.step(1 / 60, awake=True, body_reminded=reminded)
                twin.step(1 / 60, awake=True)
                if e.knobs()["anxious"] - twin.knobs()["anxious"] < 0.1 * rise:
                    return i
            return 12 * 60
        finally:
            E.BODY_LOOP = keep

    plain, reminded = settle(0.0, 1.0), settle(1.0, 1.0)
    plain2, reminded2 = settle(0.0, 2.0), settle(1.0, 2.0)
    strong = settle(1.0, 2.0, loop=4 * E.BODY_LOOP)
    ok = reminded > plain and (reminded2 - plain2) > (reminded - plain) and strong < 12 * 60
    return ok, dict(minutes=dict(plain=plain, reminded=reminded, attention2_plain=plain2, attention2_reminded=reminded2,
                                 four_times_loop_attention2_reminded=strong), extra_minutes_attention1=reminded - plain,
                    extra_minutes_attention2=reminded2 - plain2, body_loop=E.BODY_LOOP)


@test("bio", "Disrupted clock: a body clock running 4 h late on a normal schedule shows dysregulation that follows the "
      "clock, not the schedule: cortisol at 07:00 under 0.7x an aligned egg's, at least 10% more awake at 23:00 (harder "
      "to fall asleep), and it re-entrains at travel rates (still 1 h or more off after 2 days, under 0.5 h after 7). "
      "The abrupt 12-hour flip is report-only",
      "checked (Scheer 2009: cortisol follows the clock, rhythm reversed at 12 h; Eastman & Burgess 2009; CDC) / known "
      "(delayed phase, social jet lag) / change 27")
def circadian_disruption():
    def night_and_day(shift):
        e = lived_egg("average", hour=22.0)
        e.shift = shift
        _scheduled_minutes(e, 22 * 60, 9 * 60)                     # the night, to 07:00
        cort_wake = e.c["cort"]
        _scheduled_minutes(e, 7 * 60, 8 * 60)                      # to 15:00
        cort_15 = e.c["cort"]
        _scheduled_minutes(e, 15 * 60, 8 * 60)                     # to 23:00, still awake
        k = e.knobs()
        return dict(cortisol_at_07=cort_wake, cortisol_at_15=cort_15, awake_at_23=k["awake"], melatonin_at_23=e.c["mel"],
                    shift_at_23=e.shift)

    aligned, late = night_and_day(0.0), night_and_day(4.0)
    flip = night_and_day(12.0)
    e = egg(hour=7.0)
    e.shift = 4.0
    offsets = {}
    for day in range(1, 8):
        _scheduled_minutes(e, 7 * 60, 24 * 60)
        offsets[day] = e.shift
    a = egg(hour=7.0)
    _scheduled_minutes(a, 7 * 60, 24 * 60)
    conditions = dict(cortisol_follows_clock=late["cortisol_at_07"] < 0.7 * aligned["cortisol_at_07"],
                      harder_to_fall_asleep=late["awake_at_23"] >= 1.10 * aligned["awake_at_23"],
                      slow_to_realign=offsets[2] >= 1.0 and offsets[7] < 0.5,
                      aligned_stays_aligned=a.shift == 0.0)
    return all(conditions.values()), dict(conditions=conditions, aligned=aligned, delayed_4h=late,
                                          report_only_flip_12h=flip, offset_by_day=offsets)


def _with(kind, drug, dose=1.0, minutes=30, awake=True, **genes):
    e = egg(kind, **genes)
    k0 = e.knobs()
    n0 = e.c["ne"]
    if drug:
        e.give(drug, dose)
    k, _ = live(e, minutes / 60, dict(awake=awake), sched=False)
    return e, k0, k, n0


def _matched_guanfacine():
    """the guanfacine dose that cuts an average egg's NE exactly as much as the α2A PAM at dose 1 (a fair comparison:
    dose units are arbitrary, so compare the two at the same effect where they should agree)"""
    e, _, _, n0 = _with("average", "a2a_pam")
    target = n0 - e.c["ne"]
    lo, hi = 0.0, 20.0
    for _ in range(25):
        mid = (lo + hi) / 2
        e, _, _, n0 = _with("average", "guanfacine", dose=mid)
        lo, hi = (mid, hi) if n0 - e.c["ne"] < target else (lo, mid)
    return (lo + hi) / 2


@test("gain", "A brake that scales with what it brakes (C.2 rule 4): the α2A PAM's NE drop grows with his own NE and "
      "is near zero asleep; a flat agonist (guanfacine, dose-matched at rest) cuts a low-NE egg proportionally harder",
      "theory (Immy's design: PAM self-titration) / known (agonists act without NE)")
def pam_scales():
    drops = []
    for d in (0.5, 1.0, 1.6):
        e, _, _, n0 = _with("average", "a2a_pam", lc_drive=d)
        drops.append(n0 - e.c["ne"])
    a = egg(); a.give("a2a_pam", 0.0)
    live(a, 1.0, dict(awake=False), sched=False)
    n_sleep = a.c["ne"]
    a.give("a2a_pam", 1.0)
    live(a, 0.5, dict(awake=False), sched=False)
    drop_sleep = n_sleep - a.c["ne"]
    gd = _matched_guanfacine()
    eg, _, _, n0g = _with("average", "guanfacine", dose=gd, lc_drive=0.5)
    ep, _, _, n0p = _with("average", "a2a_pam", lc_drive=0.5)
    rel_g, rel_p = (n0g - eg.c["ne"]) / n0g, (n0p - ep.c["ne"]) / n0p
    ok = drops[0] < drops[1] < drops[2] and drop_sleep < 0.03 and rel_g > 1.5 * rel_p
    return ok, dict(pam_ne_drop_by_lc_drive={"0.5": drops[0], "1.0": drops[1], "1.6": drops[2]},
                    pam_drop_asleep=drop_sleep, guanfacine_matched_dose=gd,
                    low_ne_relative_drop_guanfacine=rel_g, low_ne_relative_drop_pam=rel_p)


@test("gain", "The α2A PAM in a wired egg: background down, focus up, deliberate control not worse",
      "theory (Immy's design)")
def pam_fixes_pattern():
    _, k0, k, _ = _with("wired", "a2a_pam")
    ok = k["pupil"] < k0["pupil"] and k["focus"] > k0["focus"] and k["deliberate"] >= k0["deliberate"]
    return ok, dict(background=(k0["pupil"], k["pupil"]), burst=(k0["burst"], k["burst"]),
                    focus=(k0["focus"], k["focus"]), deliberate=(k0["deliberate"], k["deliberate"]))


@test("gain", "A flat agonist flattens both modes (C.2 rule 4): guanfacine cuts background AND bursts; dose-matched "
      "to the PAM at rest, it cuts a quiet (low-NE) egg's bursts more, while the PAM cuts a wired egg's background more",
      "known (clonidine: Aston-Jones 1994) / theory (vs PAM)")
def agonist_flattens():
    gd = _matched_guanfacine()
    _, k0, kg, _ = _with("average", "guanfacine", dose=gd)
    _, kl0, klg, _ = _with("average", "guanfacine", dose=gd, lc_drive=0.5)
    _, _, klp, _ = _with("average", "a2a_pam", lc_drive=0.5)
    _, kw0, kwg, _ = _with("wired", "guanfacine", dose=gd)
    _, _, kwp, _ = _with("wired", "a2a_pam")
    ok = (kg["pupil"] < 0.9 * k0["pupil"] and kg["burst"] < 0.9 * k0["burst"] and klg["burst"] < klp["burst"]
          and kwp["pupil"] < kwg["pupil"])
    return ok, dict(guanfacine_dose=gd, average=dict(background=(k0["pupil"], kg["pupil"]), burst=(k0["burst"], kg["burst"])),
                    quiet_egg_burst=dict(none=kl0["burst"], guanfacine=klg["burst"], pam=klp["burst"]),
                    wired_egg_background=dict(none=kw0["pupil"], guanfacine=kwg["pupil"], pam=kwp["pupil"]))


@test("gain", "Agonist tolerance: three weeks of guanfacine and its NE drop shrinks (receptors pull back); stopping "
      "it overshoots (rebound)", "known (clonidine tolerance and withdrawal rebound)")
def agonist_tolerance():
    e = egg(hour=7.0)
    live(e, 24, dt=1 / 12)
    until(e, 10.0, dt=1 / 12)
    base = e.c["ne"]
    e.give("guanfacine", 1.0)
    live(e, 0.5)
    drop1 = base - e.c["ne"]
    live(e, 21 * 24 - 0.5, dt=1 / 12)
    drop21 = base - e.c["ne"]
    e.stop("guanfacine")
    live(e, 6)
    rebound = e.c["ne"] / base
    return drop21 < 0.8 * drop1 and rebound > 1.05, dict(ne_base=base, drop_day1=drop1, drop_day21=drop21,
                                                         ne_6h_after_stopping_vs_base=rebound)


@test("gain", "Loops: the orexin envelope outlasts the CRH drive after a stressful hour", "checked (C.2 rule 3)")
def orexin_envelope():
    e = egg(hour=9.0)
    until(e, 10.0)
    c0, o0 = e.c["crh"], e.c["oxa"]
    live(e, 1.0, dict(threat=0.6))
    c1, o1 = e.c["crh"], e.c["oxa"]
    back = {}
    for i in range(1, 600):
        e.step(1 / 60, awake=True)
        if "crh" not in back and e.c["crh"] - c0 < 0.1 * (c1 - c0):
            back["crh"] = i
        if "oxa" not in back and e.c["oxa"] - o0 < 0.1 * (o1 - o0):
            back["oxa"] = i
    ok = o1 > o0 * 1.05 and back.get("oxa", 999) > 2 * back.get("crh", 999)
    return ok, dict(crh_rise=c1 - c0, oxa_rise=o1 - o0, minutes_back_crh=back.get("crh"), minutes_back_oxa=back.get("oxa"))


@test("gain", "Loops: after a short stress the amygdala comes back within hours; after a long stretch it stays easier "
      "to fire for about a week and is mostly back by two weeks, by itself; about half the DA cells fall silent "
      "meanwhile", "what Immy sent (10-14 days) / checked (Chang & Grace 2014)")
def amygdala_clock():
    s = egg(hour=9.0)
    until(s, 10.0)
    live(s, 1.0, dict(threat=0.6))
    peak = s.cea_fast + s.cea_slow
    live(s, 2.0)
    short_left = (s.cea_fast + s.cea_slow) / peak
    e = long_stressed()
    end, pop_end = e.cea_slow, e.vta_pop
    vals = {}
    for day in range(1, 22):
        live(e, 24, dt=1 / 12)
        if day in (7, 14, 21):
            vals[day] = e.cea_slow / end
    ok = (short_left < 0.1 and s.cea_slow < 0.02 and end > 0.2 and vals[7] >= 0.4 and vals[14] <= 0.25
          and vals[21] <= 0.1 and pop_end <= 0.75)
    return ok, dict(short_stress_left_after_2h=short_left, short_stress_slow_part=s.cea_slow, long_end=end,
                    long_left_day7=vals[7], long_left_day14=vals[14], long_left_day21=vals[21],
                    active_da_cells_at_end=pop_end)


# ======================================================================== drug challenges
@test("drug", "caffeine (A1/A2A blocked): more awake in the evening, sleep comes later", "checked")
def caffeine():
    out = {}
    for dose in (0.0, 1.0):
        e = egg(hour=14.0)
        until(e, 15.0)
        if dose:
            e.give("caffeine", dose)
        k = until(e, 23.0)
        cross = None
        for i in range(1, 6 * 60):
            if e.step(1 / 60, awake=True)["awake"] < 0.6:
                cross = 23 + i / 60
                break
        out[dose] = (k["awake"], cross or 29.0)
    ok = out[1.0][0] - out[0.0][0] > 0.05 and out[1.0][1] - out[0.0][1] >= 0.5
    return ok, dict(awake_at_23=(out[0.0][0], out[1.0][0]), hour_below_0_6=(out[0.0][1], out[1.0][1]))


@test("drug", "Lived-day lemborexant dose 0.5 at 08:00, 13:00 and 18:00: blocking orexin lowers the sympathetic "
      "read-out clearly more (3x) in an overactive-orexin egg than in an average one, and makes every egg sleepier, "
      "at ALL three times. The wired egg is report-only (corrected 2026-10-22: one person's drop came on top of "
      "other medicines)",
      "Li 2013; Lee 2013; Huang 2010 / user-prespecified clock robustness test; ENGINE-CHANGES.md change 26")
def lemborexant_bp():
    by_hour = {}
    for hour in (8, 13, 18):
        out = {}
        for kind in ("average", "high_orexin", "wired"):
            e = lived_egg(kind, hour=hour)
            twin = copy.deepcopy(e)
            k0, n0, oxa = e.knobs(), e.c["ne"], e.c["oxa"]
            e.give("lemborexant", 0.5)
            trace = []
            for minute in range(1, 61):
                k = e.step(1 / 60, awake=True)
                kt = twin.step(1 / 60, awake=True)
                trace.append(dict(minute=minute, ne=e.c["ne"], twin_ne=twin.c["ne"],
                                  awake=k["awake"], twin_awake=kt["awake"],
                                  sympathetic=k["sympathetic"], twin_sympathetic=kt["sympathetic"]))
            out[kind] = dict(sympathetic_drop=k0["sympathetic"] - k["sympathetic"],
                             ne_drop=n0 - e.c["ne"], awake_drop=k0["awake"] - k["awake"],
                             orexin_a_at_start=oxa, trace=trace,
                             report_only_matched_twin=dict(sympathetic_drop=kt["sympathetic"] - k["sympathetic"],
                                 ne_drop=twin.c["ne"] - e.c["ne"], awake_drop=kt["awake"] - k["awake"]))
        sa = out["average"]["sympathetic_drop"]
        ok = (out["high_orexin"]["sympathetic_drop"] > 3 * sa
              and all(v["awake_drop"] > 0.03 for v in out.values()))     # wired is report-only (change 26)
        by_hour[hour] = dict(passed=ok, eggs=out)
    return all(row["passed"] for row in by_hour.values()), dict(by_hour=by_hour)


def _sleep_dose(kind, calm=0.0, drug="daridorexant", stressful_evening=False):
    """Lived-day minimum dose at 23:00; stress comparisons are report-only."""
    base = lived_egg(kind, hour=7.0)
    base.calm_skill = calm
    _scheduled_minutes(base, 7 * 60, 16 * 60, stressful_evening=stressful_evening)
    evaluations = []

    def awake_after(dose):
        e = copy.deepcopy(base)
        if dose:
            e.give(drug, dose)
        k, _ = live(e, 0.5, dict(awake=True), sched=False)
        evaluations.append(dict(dose=dose, awake=k["awake"]))
        return k["awake"]

    awake0 = awake_after(0.0)
    details = dict(start_hour=base.hour, woke_at=base.woke_at, S=base.S,
                   sleep_loss=min(1.0, max(0.0, (base.S - 0.62) / 0.1)),
                   concentrations_at_dose=dict(base.c), calm_skill=calm,
                   stressful_evening=stressful_evening, evaluated_doses=evaluations)
    if awake0 < 0.5:
        return 0.0, awake0, details
    lo, hi = 0.0, 16.0
    if awake_after(hi) >= 0.5:
        return float("inf"), awake0, details
    for _ in range(18):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if awake_after(mid) < 0.5 else (mid, hi)
    return hi, awake0, details


@test("drug", "Lived calm-night daridorexant: a competitive blocker needs a bigger dose against more orexin: "
      "high-orexin dose > average dose and wired dose >= average dose, all finite. Ratios, the calmed wired egg and "
      "every stressful-evening dose are report-only (corrected 2026-10-22: the 1.5x and 1.25x came from one "
      "person's course)",
      "competitive antagonism (Schild) / Yu 2022 limits any stress-direction claim; ENGINE-CHANGES.md change 26")
def daridorexant_dose():
    nights = {}
    for night, stressful in (("calm", False), ("stressful_evening", True)):
        out = {}
        for name, kind, calm in (("average", "average", 0), ("wired", "wired", 0),
                                  ("wired_calmed", "wired", 0.35), ("high_orexin", "high_orexin", 0)):
            dose, awake0, details = _sleep_dose(kind, calm=calm, stressful_evening=stressful)
            out[name] = dict(dose=dose, awake_without_drug=awake0, details=details)
        nights[night] = out
    da, dw, dc = (nights["calm"][k]["dose"] for k in ("average", "wired", "wired_calmed"))
    finite = all(math.isfinite(v) for v in (da, dw, dc))
    dh = nights["calm"]["high_orexin"]["dose"]
    finite = finite and math.isfinite(dh)
    conditions = dict(high_orexin_above_average=dh > da, wired_at_least_average=dw >= da)
    return finite and all(conditions.values()), dict(
        nights=nights, calm_conditions=conditions, finite_calm_doses=finite,
        calm_ratios=dict(high_orexin=dh / da, wired=dw / da, wired_calmed=dc / da) if finite and da > 0 else None,
        report_only_stress_minus_calm={k: nights["stressful_evening"][k]["dose"] - nights["calm"][k]["dose"]
                                      for k in nights["calm"]})


@test("drug", "propranolol (β blocked): strong moments no longer written extra hard; deliberate control intact",
      "checked (Cahill 1994)")
def propranolol():
    e0, e1 = egg(), egg()
    e1.give("propranolol", 1.0)
    live(e1, 0.5, sched=False)
    b0, b1 = e0.ne_burst(), e1.ne_burst()
    w0, w1 = e0.knobs()["write"], e1.knobs()["write"]
    boost0, boost1 = b0["write"] / w0 - 1, b1["write"] / w1 - 1
    d0, d1 = e0.knobs()["deliberate"], e1.knobs()["deliberate"]
    ok = boost1 <= 0.5 * boost0 and abs(d1 - d0) < 0.05 * d0
    return ok, dict(strong_moment_boost=(boost0, boost1), deliberate=(d0, d1))


@test("drug", "guanfacine (α2A agonist): calmer, better deliberate control (under a mild stress)", "checked (Arnsten)")
def guanfacine():
    e0, e1 = egg(), egg()
    e1.give("guanfacine", 1.0)
    k0, _ = live(e0, 0.5, dict(threat=0.3), sched=False)
    k1, _ = live(e1, 0.5, dict(threat=0.3), sched=False)
    q0, q1 = egg(lc_drive=0.5), egg(lc_drive=0.5)
    q1.give("guanfacine", 1.0)
    kq0, _ = live(q0, 0.5, sched=False)
    kq1, _ = live(q1, 0.5, sched=False)
    return k1["anxious"] < k0["anxious"] and k1["deliberate"] > k0["deliberate"], dict(
        anxious=(k0["anxious"], k1["anxious"]), deliberate=(k0["deliberate"], k1["deliberate"]),
        report_only_quiet_egg_deliberate=(kq0["deliberate"], kq1["deliberate"]))


@test("drug", "naltrexone (MOR blocked): company feels a little less warm (people: small, g ≈ -0.2). Single-system "
      "part only: comfort (MOR) falls, warmth (OXTR) does not; the felt size is a group-map question (step 3)",
      "checked (Inagaki 2016)")
def naltrexone():
    out = {}
    for dose in (0.0, 1.0):
        e = egg()
        if dose:
            e.give("naltrexone", dose)
        k, _ = live(e, 1.0, dict(company=1.0), sched=False)
        out[dose] = k
    c0, c1 = out[0.0]["comfort"], out[1.0]["comfort"]
    w0, w1 = out[0.0]["warmth"], out[1.0]["warmth"]
    return c1 < 0.9 * c0 and abs(w1 - w0) < 0.02 * w0, dict(comfort=(c0, c1), warmth=(w0, w1))


@test("drug", "SSRI: little at first; more patience after the 5-HT1A brakes adapt (weeks)", "checked (Blier 1998)")
def ssri():
    e = egg(hour=7.0)
    until(e, 10.0)
    p0 = e.knobs()["patience"]
    e.give("ssri", 1.0)
    live(e, 1.0)
    d1 = e.knobs()["patience"] - p0
    live(e, 28 * 24 - 1.0, dt=1 / 12)
    d28 = e.knobs()["patience"] - p0
    return d28 > 0.03 and d1 < 0.4 * d28, dict(patience_rest=p0, change_day1=d1, change_day28=d28,
                                               autoreceptor_amount=e.amount["ht1a_auto"])


@test("drug", "methylphenidate: a low dose sharpens deliberate control, a high dose scatters it (inverted U)",
      "checked (Arnsten & Dudley 2005)")
def methylphenidate():
    doses = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
    vals = []
    for d in doses:
        _, _, k, _ = _with("average", "methylphenidate" if d else None, dose=d)
        vals.append(k["deliberate"])
    top = max(vals[1:5])
    return top > 1.03 * vals[0] and vals[-1] < 0.9 * top, dict(doses=doses, deliberate=vals)


@test("drug", "scopolamine (muscarinic blocked): new moments badly written", "checked")
def scopolamine():
    _, k0, k, _ = _with("average", "scopolamine")
    return k["write"] < 0.7 * k0["write"], dict(write=(k0["write"], k["write"]))


@test("drug", "diphenhydramine (H1 blocked): sleepy", "checked")
def diphenhydramine():
    _, k0, k, _ = _with("average", "diphenhydramine")
    return k0["awake"] - k["awake"] > 0.05, dict(awake=(k0["awake"], k["awake"]))


@test("drug", "antalarmin (CRHR1 blocked): stress no longer turns into anxious tone", "checked")
def antalarmin():
    out = {}
    for drug in (None, "antalarmin"):
        e = egg()
        if drug:
            e.give(drug, 1.0)
        live(e, 0.25, sched=False)
        a0 = e.knobs()["anxious"]
        k, _ = live(e, 0.5, dict(threat=0.6), sched=False)
        out[drug] = k["anxious"] - a0
    return out["antalarmin"] <= 0.6 * out[None], dict(anxious_rise=(out[None], out["antalarmin"]))


@test("drug", "oxytocin antagonist: can't learn who someone new is", "checked (Ferguson 2000, 2001)")
def ot_antagonist():
    out = {}
    for drug in (None, "ot_antagonist"):
        e = egg()
        if drug:
            e.give(drug, 1.0)
        k, _ = live(e, 0.5, dict(company=1.0), sched=False)
        out[drug] = k["recognize"]
    return out["ot_antagonist"] < 0.6 * out[None], dict(recognize=(out[None], out["ot_antagonist"]))


@test("drug", "KOR antagonist: stress hurts less afterwards", "checked (Land 2008)")
def kor_antagonist():
    out = {}
    for drug in (None, "kor_antagonist"):
        e = egg()
        if drug:
            e.give(drug, 1.0)
        live(e, 0.25, sched=False)
        d0 = e.knobs()["dysphoria"]
        live(e, 2.0, dict(threat=0.6), sched=False)
        k, _ = live(e, 1.0, sched=False)
        out[drug] = (k["dysphoria"] - d0, k["wanting"])
    ok = out["kor_antagonist"][0] <= 0.5 * out[None][0] and out["kor_antagonist"][1] > out[None][1]
    return ok, dict(dysphoria_rise=(out[None][0], out["kor_antagonist"][0]),
                    wanting_after=(out[None][1], out["kor_antagonist"][1]))


@test("drug", "cariprazine, low dose (mostly D3): wanting comes back after a long stress stretch; needs D3 (the same "
      "drug without D3 does not) and partial efficacy (a full D3/D2 blocker does not)",
      "checked (Duric 2017; McIntyre 2024)")
def cariprazine():
    base = egg(hour=10.0).knobs()["wanting"]
    out = {}
    for drug in (None, "cariprazine", "cariprazine_no_d3", "d3_blocker"):
        e = long_stressed()
        if drug:
            e.give(drug, 0.5)
        k, _ = live(e, 1.0, sched=False)
        out[drug or "none"] = k["wanting"]
    w = out["none"]
    ok = (w < 0.95 * base and out["cariprazine"] > 1.05 * w and out["cariprazine_no_d3"] < 1.02 * w
          and out["d3_blocker"] < 1.02 * w)
    return ok, dict(wanting_before_stress=base, **{f"wanting_{k}": v for k, v in out.items()})


@test("drug", "bupropion: a small lift in wanting only (the human anhedonia effect is small)", "known")
def bupropion():
    _, k0, k, _ = _with("average", "bupropion")
    lift = k["wanting"] / k0["wanting"] - 1
    return 0.01 <= lift <= 0.25, dict(wanting_lift=lift)


@test("drug", "modafinil: keeps him awake even with orexin cells gone (even more than normal), but awake-not-sharp "
      "without orexin", "checked (Willie 2005; Wisor 2001)")
def modafinil():
    out = {}
    for kind in ("average", "ox_null"):
        e0 = egg(kind, hour=13.0); until(e0, 14.0)
        e1 = egg(kind, hour=13.0); until(e1, 14.0)
        e1.give("modafinil", 1.0)
        k0, _ = live(e0, 1.0)
        k1, _ = live(e1, 1.0)
        out[kind] = dict(awake=(k0["awake"], k1["awake"]), deliberate_on_drug=k1["deliberate"])
    lift = {k: v["awake"][1] - v["awake"][0] for k, v in out.items()}
    ok = (lift["ox_null"] >= 0.1 and lift["ox_null"] >= lift["average"]
          and out["ox_null"]["deliberate_on_drug"] < out["average"]["deliberate_on_drug"])
    return ok, dict(out, awake_lift=lift)


@test("drug", "buspirone (5-HT1A): calmer, slowly", "checked")
def buspirone():
    e = egg(hour=7.0)
    until(e, 10.0)
    a0 = e.knobs()["anxious"]
    e.give("buspirone", 1.0)
    live(e, 1.0)
    d1 = e.knobs()["anxious"] - a0
    live(e, 21 * 24 - 1.0, dt=1 / 12)
    d21 = e.knobs()["anxious"] - a0
    return d21 < 0 and d21 < d1, dict(anxious_change_day1=d1, anxious_change_day21=d21)


@test("drug", "memantine: blocks background NMDA (open too long) and spares synaptic (≤10%); less background noise, "
      "learning intact, no glutamate surge; does more when background glutamate is high (stress)",
      "checked (Lipton 2006; Xia 2010)")
def memantine():
    e0 = egg()
    e = egg()
    e.give("memantine", 1.0)
    live(e, 0.5, sched=False)
    live(e0, 0.5, sched=False)
    nm, k, k0 = e.nmda_state(), e.knobs(), e0.knobs()
    s0, s = egg(), egg()
    s.give("memantine", 1.0)
    ks0, _ = live(s0, 1.0, dict(threat=0.6), sched=False)
    ks, _ = live(s, 1.0, dict(threat=0.6), sched=False)
    cut_rest, cut_stress = k0["noise"] - k["noise"], ks0["noise"] - ks["noise"]
    ok = (nm["syn"] <= 0.10 and nm["ex"] >= 2 * nm["syn"] and nm["int"] < 0.2 and abs(e.c["bdnf"] / e0.c["bdnf"] - 1) < 0.05
          and abs(k["write"] / k0["write"] - 1) < 0.1 and cut_rest > 0 and cut_stress > cut_rest)
    return ok, dict(block=nm, noise_cut_rest=cut_rest, noise_cut_stress=cut_stress, write=(k0["write"], k["write"]),
                    bdnf_ratio=e.c["bdnf"] / e0.c["bdnf"])


@test("drug", "ketamine, low dose: spaced out for about an hour, then a lift in wanting that lasts days (glutamate "
      "surge -> BDNF -> new synapses)", "checked (Zarate 2006; Li 2010; Autry 2011)")
def ketamine():
    e = egg(hour=9.0)
    until(e, 10.0)
    k0 = e.knobs()
    e.give("ketamine", 1.0)
    k_half, _ = live(e, 0.5)
    k_3h, _ = live(e, 2.5)
    until(e, 10.0)
    live(e, 24)
    k_d2 = e.knobs()
    live(e, 8 * 24, dt=1 / 12)
    until(e, 10.0)
    k_d10 = e.knobs()
    ok = (k_half["deliberate"] < 0.8 * k0["deliberate"] and abs(k_3h["deliberate"] / k0["deliberate"] - 1) < 0.1
          and k_d2["wanting"] > 1.05 * k0["wanting"] and k_d10["wanting"] < k_d2["wanting"])
    return ok, dict(deliberate=(k0["deliberate"], k_half["deliberate"], k_3h["deliberate"]),
                    wanting=(k0["wanting"], k_d2["wanting"], k_d10["wanting"]), note="(before, 30 min, 3 h) / (before, day 2, day 10)")


@test("drug", "benzodiazepine (synaptic GABA-A): calmer, sleepier; fewer new moments written at a high dose", "checked")
def benzodiazepine():
    _, k0, k, _ = _with("average", "benzodiazepine")
    _, _, kh, _ = _with("average", "benzodiazepine", dose=3.0)
    ok = k["anxious"] < k0["anxious"] and k["awake"] < k0["awake"] and kh["write"] < 0.9 * k0["write"]
    return ok, dict(anxious=(k0["anxious"], k["anxious"]), awake=(k0["awake"], k["awake"]),
                    write_high_dose=(k0["write"], kh["write"]))


@test("drug", "zuranolone / allopregnanolone (extrasynaptic GABA-A): calmer; lifts a low, stressed state within days",
      "checked (Deligiannidis 2021)")
def zuranolone():
    e0, e1 = long_stressed(), long_stressed()
    e1.give("zuranolone", 1.0)
    k0, _ = live(e0, 3 * 24, dt=1 / 12)
    k1, _ = live(e1, 3 * 24, dt=1 / 12)
    ok = k1["anxious"] < k0["anxious"] and k1["wanting"] > 1.03 * k0["wanting"]
    return ok, dict(anxious_day3=(k0["anxious"], k1["anxious"]), wanting_day3=(k0["wanting"], k1["wanting"]))


@test("drug", "rimonabant (CB1 blocked): more anxious, lower mood; fear fades more slowly", "checked")
def rimonabant():
    out = {}
    for drug in (None, "rimonabant"):
        e = egg()
        if drug:
            e.give(drug, 1.0)
        k, _ = live(e, 0.5, sched=False)
        e.fear_cs = 1.0
        live(e, 2.0, dict(safe_cue=True), sched=False)
        out[drug] = (k["anxious"], k["liking"], e.fear_cs)
    n, r = out[None], out["rimonabant"]
    ok = r[0] > n[0] and r[1] < n[1] and r[2] > 1.2 * n[2]
    return ok, dict(anxious=(n[0], r[0]), liking=(n[1], r[1]), fear_left_after_2h_safe=(n[2], r[2]))


@test("drug", "low-dose endotoxin (cytokines up): flat (less wanting), tired, wants company less",
      "checked (Eisenberger 2010)")
def endotoxin():
    e0 = egg(hour=10.0)
    e1 = egg(hour=10.0)
    e1.step(1 / 60, endotoxin=3.0)
    e0.step(1 / 60)
    k0, _ = live(e0, 2.0)
    k1, _ = live(e1, 2.0)
    b0, b1 = e0.reward_event("treat", 1.0), e1.reward_event("treat", 1.0)
    ok = (k1["wanting"] < 0.9 * k0["wanting"] and k1["awake"] < k0["awake"]
          and k1["social_drive"] < 0.9 * k0["social_drive"])
    return ok, dict(wanting=(k0["wanting"], k1["wanting"]), awake=(k0["awake"], k1["awake"]),
                    social_drive=(k0["social_drive"], k1["social_drive"]),
                    report_only_reward_burst_d1=(b0["d1"], b1["d1"]))


@test("drug", "yohimbine (α2 blocked, the LC brake off): NE surge, anxious; a panic-prone egg reacts much more",
      "checked (Charney 1984)")
def yohimbine():
    out = {}
    for kind in ("average", "panic_prone"):
        e, k0, k, n0 = _with(kind, "yohimbine")
        out[kind] = (e.c["ne"] / n0, k["anxious"] - k0["anxious"])
    ok = out["average"][0] >= 1.5 and out["average"][1] > 0 and out["panic_prone"][1] >= 1.5 * out["average"][1]
    return ok, dict(ne_ratio=(out["average"][0], out["panic_prone"][0]),
                    anxious_rise=(out["average"][1], out["panic_prone"][1]))


@test("drug", "flumazenil (benzodiazepine site blocked): nothing in a normal egg; anxiety in a low-GABA, panic-prone "
      "egg", "theory (Nutt 1990: the endozepine idea)")
def flumazenil():
    out = {}
    for kind in ("average", "panic_prone"):
        _, k0, k, _ = _with(kind, "flumazenil")
        out[kind] = k["anxious"] - k0["anxious"]
    return abs(out["average"]) < 0.01 and out["panic_prone"] > 0.05, dict(anxious_change=out)


# ======================================================================== run
def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    E.calibrate()
    rows = []
    for t in TESTS:
        t0 = time.time()
        try:
            ok, nums = t["fn"]()
            err = None
        except Exception as ex:                                        # a crash is a failure, shown with its error
            ok, nums, err = False, {}, f"{type(ex).__name__}: {ex}"
        rows.append(dict(group=t["group"], name=t["name"], expect=t["expect"], evidence=t["evidence"],
                         passed=bool(ok), numbers=r3(nums), error=err, seconds=round(time.time() - t0, 1)))
        print(f"{'PASS' if ok else 'FAIL'}  {t['group']:5s} {t['name']:24s} {json.dumps(r3(nums))[:220]}"
              + (f"  ERROR {err}" if err else ""), flush=True)
    json.dump(dict(rest_knobs=r3(E.CAL["knobs"]), tests=rows), open(os.path.join(out_dir, "results.json"), "w"), indent=1)
    n_pass = sum(r["passed"] for r in rows)
    L = [f"# Chemistry foundations: section F tests ({time.strftime('%Y-%m-%d %H:%M')})", "",
         f"{n_pass} of {len(rows)} pass. Engine: `tools/mind/chem/engine.py`; tests: `tools/mind/chem/test_foundations.py`.",
         "Every pass rule was written before the run. Numbers are relative units (1 = an average egg at rest).", ""]
    for g, title in (("bio", "Biology signatures"), ("gain", "Gain tests"), ("wired", "The wired phenotype (Immy's paper)"),
                     ("drug", "Drug challenges")):
        L += [f"## {title}", "", "| | test | what must happen | how sure | numbers |", "|---|---|---|---|---|"]
        for r in rows:
            if r["group"] == g:
                nums = json.dumps(r["numbers"]).replace("|", "/") if not r["error"] else r["error"]
                L.append(f"| {'✅' if r['passed'] else '❌'} | {r['name']} | {r['expect']} | {r['evidence']} | {nums} |")
        L.append("")
    open(os.path.join(out_dir, "report.md"), "w").write("\n".join(L) + "\n")
    print(f"\n{n_pass}/{len(rows)} pass -> {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1])
