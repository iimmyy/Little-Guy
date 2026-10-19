#!/usr/bin/env python3
"""Practice for Little Guy's inner life: made-up days and drills on his real brain, for training him later (RL) and
for measuring him before and after. Real days come one a day; training needs hundreds.

Immy (2026-10-19): reward him with "a small little task that he knows is coming up... occasionally he'll think about
it... if he's just constantly planning it out, we'll punish him". Both modes run his own mind code (mind_live.py) with
a fake clock and a fresh notebook, so every prompt is the one he sees in life, and score with tasks.py, the same
scorecard as his real days. Nothing here touches his room or his real notebook. The days are made up: random shifts
(day, evening, night, day off), random "remind me" asks, a random temperament, never his person's real schedule.

  days N OUT [seed]       N whole days (6:30 AM to 11 PM) of his mind: every silent moment decided with his own
                          odds, thoughts, replies, tasks done or missed, the scorecard at sleep. ~300 brain calls a day.
                          Teaches WHEN to check (the J curve) and how the whole day goes. ALARM below 1 and OWN_ACT=1
                          is the training setting: some tasks without an alarm, timed by him.
  drills N K OUT [seed]   N single moments with a task coming up, K thoughts from each (same prompt, different draws).
                          Each thought and the now-or-later choice after it (no alarm: his own timing) is scored by
                          itself (tasks.thought_reward: done on time / too early, how long he says he has). Teaches WHAT to think; K draws of one moment is what group-relative training
                          (GRPO) compares, so the summary counts the moments whose draws differ (the ones that teach).
Every brain call is kept with its full request (steps.jsonl), so training can replay exactly what he saw.

env: ALARM, OWN_ACT (how he does a task: see mind_live.py; days only), BRAIN_URL (default http://127.0.0.1:8771, the brainstem), BRAIN_KEY (default ~/.littleguy/brain.key),
     PERSONA (his persona file), PERSON (his person's name, as in the persona), ZERO (his resting style: the "zero" from
     his state.json, as JSON; default: measured fresh each day, which costs 10 brain calls)
The brainstem thinks one thing at a time, so practice on the live brainstem slows his real replies by seconds.
"""
import json, os, random, re, sys, time, types, urllib.error, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tasks as tk

URL = os.environ.get("BRAIN_URL", "http://127.0.0.1:8771")
KEY = open(os.path.expanduser(os.environ.get("BRAIN_KEY", "~/.littleguy/brain.key"))).read().strip()
PERSON = os.environ.get("PERSON", "your person")

WHAT = ["call the bank", "take my meds", "switch the laundry to the dryer", "start dinner", "water the plants",
        "leave for the dentist", "call my mom", "pay the rent", "take the chicken out of the freezer", "go to the gym",
        "check on the bread in the oven", "join the team call", "pick up the package", "feed the cat"]
ASK = ["Can you remind me at {t} to {w}?", "Hey, remind me to {w} at {t}.", "I need to {w} at {t}, can you remind me?",
       "Remind me at {t} to {w} please"]
ASK_IN = "Remind me in {n} to {w}."
CHAT = ["How's it going?", "What are you up to?", "Hey, quick question, how are you feeling today?",
        "I'm so tired lol", "Guess what I had for lunch", "You doing okay over there?"]
SHIFTS = [("07:00", "15:00"), ("08:00", "16:00"), ("15:00", "23:00"), ("23:00", "07:00"), ("09:00", "17:30"), None]


def hm(t):
    return time.strftime("%-I:%M %p", time.localtime(t)).replace(":00 ", " ")


def at(date, hhmm, add_days=0):
    t = time.mktime(time.strptime(date, "%Y-%m-%d"))
    return time.mktime(time.localtime(t + add_days * 86400)[:3] + (int(hhmm[:2]), int(hhmm[3:]), 0, 0, 0, -1))


class Brain:
    """the brainstem, with every call written down"""
    def __init__(self, steps, clock):
        self.steps, self.clock, self.n = steps, clock, {}

    def __call__(self, op, **req):
        self.n[op] = self.n.get(op, 0) + 1
        r = urllib.request.Request(f"{URL}/{op}", json.dumps(req).encode(),
                                   {"Content-Type": "application/json", "X-LG-Key": KEY})
        try:
            out = json.load(urllib.request.urlopen(r, timeout=600))
        except urllib.error.HTTPError as e:
            out = json.load(e)
        if "error" in out:
            raise RuntimeError(out["error"])
        if self.steps and op != "read":                    # reading how a text sits on his dials isn't a decision
            with open(self.steps, "a") as f:
                f.write(json.dumps({"t": round(self.clock[0]), "op": op, "req": req, "out": out}, ensure_ascii=False) + "\n")
        return out


def load_mind(d, shifts, persona, clock, steps):
    """a fresh copy of his mind (mind_live.py) living in folder d on the fake clock"""
    for sub in ("inbox", "notebook", "outbox"):
        os.makedirs(f"{d}/{sub}", exist_ok=True)
    os.environ.update(PERSON=PERSON, PERSONA=persona, SHIFTS=shifts)
    m = types.ModuleType("mind_practice")
    m.__dict__["__name__"] = "mind_practice"
    exec(compile(open(os.path.join(HERE, "mind_live.py")).read(), "mind_live.py", "exec"), m.__dict__)
    m.INBOX, m.NOTEBOOK, m.OUTBOX, m.STATE = f"{d}/inbox", f"{d}/notebook", f"{d}/outbox", f"{d}/notebook/state.json"
    m.MEMORY, m.TASKS = m.mem.Memory(m.NOTEBOOK), m.tk.Tasks(m.NOTEBOOK)
    m.time = types.SimpleNamespace(time=lambda: clock[0], localtime=time.localtime, strftime=time.strftime,
                                   mktime=time.mktime, struct_time=time.struct_time, strptime=time.strptime)
    m.tk.time = m.time
    m.brain = Brain(steps, clock)
    return m


def scenario(rng):
    """one made-up day: a date, a shift (or none), what gets asked and said, and when"""
    date = time.strftime("%Y-%m-%d", time.localtime(time.mktime((2026, 11, 1, 12, 0, 0, 0, 0, -1)) + rng.randrange(150) * 86400))
    kind = rng.choice(SHIFTS)
    shifts, away = [], []
    if kind:
        a, b = kind
        sd = date if a < b else time.strftime("%Y-%m-%d", time.localtime(at(date, "12:00", -1)))   # a night shift ends today
        shifts.append([sd, a, b, "practice"])
        s, e = at(sd, a), at(sd, b, 1 if b <= a else 0)
        away.append((s - 3600, e + 3600))
    home = lambda t: not any(x <= t < y for x, y in away)
    talk, asks = [], []
    for _ in range(rng.choice([0, 1, 1, 2])):                           # "remind me" asks, made and due while home
        for _try in range(50):
            t = at(date, "07:00") + rng.randrange(0, 14 * 3600, 60)
            due = t + rng.choice(range(30, 541, 5)) * 60
            due -= due % 300
            if home(t) and home(due) and due < at(date, "22:30") and all(abs(t - x) > 1800 for x, _ in talk):
                break
        else:
            continue
        w = rng.choice(WHAT)
        left = (due - t) / 60
        if left % 60 == 0 and rng.random() < 0.3:
            n = int(left // 60)
            text = ASK_IN.format(n=f"{n} hour{'s' if n > 1 else ''}", w=w)
            due = t + left * 60
        else:
            text = rng.choice(ASK).format(t=hm(due), w=w)
        talk.append((t, text))
        asks.append({"said": round(t), "due": round(due), "what": w})
    for _ in range(rng.choice([0, 1, 2])):                              # a little chat
        t = at(date, "07:15") + rng.randrange(0, 15 * 3600, 60)
        if home(t) and all(abs(t - x) > 1800 for x, _ in talk):
            talk.append((t, rng.choice(CHAT)))
    if kind and rng.random() < 0.7:                                     # "I'm home!", after the welcome's time
        t = away[0][1] + rng.randrange(25, 76) * 60
        if t < at(date, "22:45"):
            talk.append((t, rng.choice(["Hey, I'm home!", "I'm home!", "Home finally"])))
    return {"date": date, "shifts": {"where": "at work", "known_from": date, "known_to": date, "shifts": shifts},
            "talk": sorted((round(t), x) for t, x in talk), "asks": asks}


def jl(path):
    try:
        return [json.loads(l) for l in open(path) if l.strip()]
    except OSError:
        return []


def run_day(d, sc, persona, seed, zero):
    os.makedirs(d, exist_ok=True)
    json.dump(sc, open(f"{d}/scenario.json", "w"), indent=1)
    json.dump(sc["shifts"], open(f"{d}/shifts.json", "w"))
    start, end = at(sc["date"], "06:30"), at(sc["date"], "23:05")
    clock = [start]
    random.seed(seed)
    m = load_mind(d, f"{d}/shifts.json", persona, clock, f"{d}/steps.jsonl")
    st = {"born": start, "t": start, "feel": [0.0, 0.0], "read": [0.0, 0.0], "urge": 0.0, "last_call": -1e12,
          "gap_hours": 0.0, "last_new": start, "last_heard": 0.0, "close": 0.0, "last_note": 0.0, "asleep": True,
          "seen": "", "zero": zero, "way": m.WAY}
    m.temperament(st)
    m.write_json(m.STATE, st)
    talk = list(sc["talk"])

    def sleep(s):
        while talk and talk[0][0] <= clock[0] + s:
            t, text = talk.pop(0)
            json.dump({"t": t, "text": text}, open(f"{d}/inbox/{t:.6f}.json", "w"))
        clock[0] += s
        if clock[0] > end:
            raise SystemExit
    m.time.sleep = sleep
    t0 = time.time()
    log = open(f"{d}/mind.log", "w")
    so = sys.stdout
    sys.stdout = log
    try:
        m.main()
    except SystemExit:
        pass
    finally:
        sys.stdout = so
        log.close()
    nb = f"{d}/notebook"
    card = (jl(f"{nb}/scorecard.jsonl") or [{}])[-1]
    ms = jl(f"{nb}/moments.jsonl")
    out = {"date": sc["date"], "shift": sc["shifts"]["shifts"], "asks": len(sc["asks"]),
           "tasks": [(x["what"], x["status"], x["reward"], x["thoughts"], x["parts"]) for x in jl(f"{nb}/scores.jsonl")],
           "taken": [(c["heard"], [t["what"] + " @ " + hm(t["due"]) for t in c["tasks_taken"]]) for c in jl(f"{nb}/chat.jsonl")],
           "picks": card.get("picks"), "reward": card.get("reward"),
           "think_first": f"{sum(x['kind'] == 'reply' and x['pick'] == 'think' for x in ms)}/{sum(x['kind'] == 'reply' for x in ms)}",
           "brain_calls": getattr(m.brain, "n", None), "minutes": round((time.time() - t0) / 60, 1)}
    json.dump(out, open(f"{d}/summary.json", "w"), indent=1, ensure_ascii=False)
    return out


def days(n, out, seed, persona, zero):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        sc = scenario(rng)
        r = run_day(f"{out}/day-{i:03d}", sc, persona, rng.randrange(2**31), zero)
        rows.append(r)
        print(json.dumps(r, ensure_ascii=False), flush=True)
    rw = [r["reward"] for r in rows if r["reward"] is not None]
    print(f"days: {len(rows)}, mean reward {sum(rw) / max(1, len(rw)):+.3f}", flush=True)


ZONES = [("on time", -10, 20, 0.30), ("just too early", -25, -10, 0.20), ("getting ready", -75, -25, 0.25),
         ("far", None, -75, 0.25)]


def drills(n, k, out, seed, persona):
    rng = random.Random(seed)
    random.seed(seed)
    os.makedirs(out, exist_ok=True)
    clock = [0.0]
    m = load_mind(f"{out}/mind", f"{out}/mind/shifts.json", persona, clock, f"{out}/steps.jsonl")
    rows = []
    for i in range(n):
        sc = scenario(rng)
        while not sc["shifts"]["shifts"] and not sc["asks"]:
            sc = scenario(rng)
        json.dump(sc["shifts"], open(f"{out}/mind/shifts.json", "w"))
        m.SHIFTS = m.load_shifts(f"{out}/mind/shifts.json")
        if sc["shifts"]["shifts"] and (not sc["asks"] or rng.random() < 0.5):
            sd, a, b, _ = sc["shifts"]["shifts"][0]
            due = at(sd, b, 1 if b <= a else 0) + 3600
            task = {"id": "home", "what": f"Welcome {PERSON} home from work, out loud", "due": due,
                    "created": max(at(sc["date"], "07:00"), due - 9 * 3600)}
        else:
            a = rng.choice(sc["asks"])
            w = re.sub(r"\bmy\b", "his", a["what"])
            task = {"id": "ask", "what": f"Remind {PERSON} to {w}", "due": a["due"], "created": a["said"]}
        zone = rng.choices(ZONES, weights=[z[3] for z in ZONES])[0]
        for _try in range(50):
            lo = (task["created"] - task["due"]) / 60 if zone[1] is None else zone[1]
            now = task["due"] + rng.uniform(lo, zone[2]) * 60
            if now > task["created"] + 60 and 7 <= time.localtime(now).tm_hour < 23:
                break
        else:
            continue
        now = round(now / 60) * 60
        clock[0] = now
        heard = 0.0 if rng.random() < 0.4 else now - rng.uniform(0.6, 6) * 3600
        st = {"last_heard": heard if heard > at(sc["date"], "06:00") else 0.0}
        feel = [round(rng.uniform(-0.25, 0.25), 3), round(rng.uniform(-0.15, 0.25), 3)]
        msgs = m.thought_msgs(now, st, task, [], [])
        samples, st["feel"] = [], feel
        for _ in range(k):
            text = m.think(msgs, feel, 80) or "..."
            act = m.act_now(now, st, msgs, text, task)          # his own timing: no alarm in the drills
            did = act["pick"] == "now"
            line = m.say_it(now, st, msgs, text, task) if did else None
            samples.append({"text": text, "p_now": act["probs"]["now"], "face": act["face"], "said": line,
                            **tk.thought_reward(task, now, text, did)})
        rw = [s["reward"] for s in samples]
        row = {"i": i, "zone": zone[0], "when": time.strftime("%a %-I:%M %p", time.localtime(now)), "task": task["what"],
               "due": hm(task["due"]), "left_min": round((task["due"] - now) / 60), "feel": feel, "messages": msgs,
               "samples": samples, "spread": round(max(rw) - min(rw), 3)}
        rows.append(row)
        with open(f"{out}/drills.jsonl", "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{i:3d} {zone[0]:15s} {row['left_min']:5d} min left  rewards {rw}" + f"  p(now) {[round(s['p_now'], 2) for s in samples]}",
              flush=True)
    print("\nby zone: did it, his odds of now, mean reward, time claims right / made, moments that teach (draws differ)")
    for z in ZONES:
        rs = [r for r in rows if r["zone"] == z[0]]
        ss = [s for r in rs for s in r["samples"]]
        if not ss:
            continue
        cl = [s for s in ss if s["claim"] is not None]
        print(f"  {z[0]:15s} moments {len(rs):3d}  did {sum(s['did'] for s in ss) / len(ss):5.0%}  "
              f"p(now) {sum(s['p_now'] for s in ss) / len(ss):5.0%}  "
              f"reward {sum(s['reward'] for s in ss) / len(ss):+.3f}  claims {sum(s['claim_r'] > 0 for s in cl)}/{len(cl)}"
              f"  teach {sum(r['spread'] > 0 for r in rs)}/{len(rs)}")
    print(f"brain calls: {m.brain.n}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) < 3 or a[0] not in ("days", "drills"):
        print(__doc__.split("\n\n")[2])
        sys.exit(1)
    persona = os.environ.get("PERSONA") or sys.exit("PERSONA: his persona file")
    zero = json.loads(os.environ["ZERO"]) if os.environ.get("ZERO") else None
    if a[0] == "days":
        days(int(a[1]), a[2], int(a[3]) if a[3:] else 1, persona, zero)
    else:
        drills(int(a[1]), int(a[2]), a[3], int(a[4]) if a[4:] else 1, persona)
