#!/usr/bin/env python3
"""Little Guy's mind, living on the real clock inside his room: life_sim.py v4 (the pacemaker, drives, the two core
dials, the notebook, the urge to call out) with lived time instead of a scripted day. Immy: "launch it on my Mac. Have
them live there, but create a sandbox there."

He only touches what his walls allow:
  his inbox     (read only)  what his person says, one note per message (the guard writes them)
  his notebook  (his alone)  notes.jsonl / notes.txt (his private notes), chat.jsonl (what was said), state.json
  his outbox    (drop only)  mood notes, a reply to each message, and a SAY line when the urge fires
  the brain slot             /var/lib/littleguy/brain/brain.sock: think (with his feelings as a drip) and read
                             (how a text sits along his dials); the brain itself runs outside the room

What he senses on the Mac is only the clock and his person's messages (no light, sound or battery here, so those
drives stay at zero instead of being made up). His person counts as here while the last message is under PRESENT_MIN
minutes old. He sleeps from SLEEP_FROM to WAKE (no notes; what he feels mostly fades, SLEEP_LEFT), but a message
still wakes him enough to answer. His resting style is the zero for every reading (life_sim v2): on first start he
writes a few notes and replies with the feeling off and keeps their mean.

Inner life (Immy, 2026-10-19: "he should be able to think to himself... in the moment of silence, because that's what
output without input looks like"; and: let him flip the thinking toggle himself, and reward it with a small task he
knows is coming up): every MOMENT_EVERY minutes or so (never on the dot) he decides for himself, with his brain's own
odds, whether to think about something coming up, let his mind wander, or not think at all. Thoughts stay in his head
(thoughts.jsonl, shown in the menu bar as "inner"); when a task's alarm rings he thinks about it and decides, his own
odds again, whether to do it now, out loud. Before answering
his person, he decides the same way whether to think it over first (his brain's own thinking switch). Every decision is
logged with its odds (moments.jsonl), and tasks.py scores how he handled each task, for training that later.

Settings (env): PERSON (default "your person"), MOMENT_EVERY (minutes, default 10), PERSONA (default /opt/littleguy/persona.txt; '#' lines dropped),
  WAY (felt | restless | big, default felt), NOTE_EVERY (hours between notes, default 2), SLEEP_FROM / WAKE (hours,
  default 23 / 7), SHIFTS (his person's work shifts, default /opt/littleguy/shifts.json; without it, weekdays
  WORK_FROM to WORK_TO, default 9 / 17.5), PRESENT_MIN (default 30),
  REUNION_H (silence that makes a message a homecoming, default 1), ALARM (chance a task gets an alarm, default 1),
  OWN_ACT (1: he may also do a task on his own, without its alarm; default 0)
"""
import json, math, os, random, re, socket, sys, time, unicodedata
import memory as mem
import tasks as tk

INBOX = "/var/lib/littleguy/inbox"
NOTEBOOK = "/var/lib/littleguy/room/notebook"
OUTBOX = "/var/lib/littleguy/room/outbox"
SLOT = "/var/lib/littleguy/brain/brain.sock"
STATE = os.path.join(NOTEBOOK, "state.json")
env = lambda k, d: type(d)(os.environ.get(k, d))
PERSON = env("PERSON", "your person")
PERSONA_FILE = env("PERSONA", "/opt/littleguy/persona.txt")
WAY = env("WAY", "felt")
NOTE_EVERY, SLEEP_FROM, WAKE = env("NOTE_EVERY", 2.0), env("SLEEP_FROM", 23.0), env("WAKE", 7.0)
WORK_FROM, WORK_TO = env("WORK_FROM", 9.0), env("WORK_TO", 17.5)
PRESENT_MIN, REUNION_H = env("PRESENT_MIN", 30.0), env("REUNION_H", 1.0)
MOMENT_EVERY = env("MOMENT_EVERY", 10.0)
# how he does a task (Immy, 2026-10-19: a mix of an alarm now and his own timing later; cue fading): after a thought about
# it he decides with his own odds, now or later (go / no-go). Its alarm rings ALARM_EARLY minutes before it's due and
# every SNOOZE minutes until 20 past; whether a task gets an alarm is drawn with chance ALARM. OWN_ACT lets him also do
# it on his own at any thought about it (off in life for now). Before training, practice (2026-10-19,
# .run/results/mind/practice-v1): a DO: line asked for inside the thought almost never came (11% on time); asked "now or
# later?" he almost always said now, even hours early (it mixed up getting ready with doing it); asked "getting ready
# doesn't count: is it time to actually do it?" his odds follow the clock (now: 0.93 on time, 0.51 10-25 min early,
# 0.19 25-75 min early, 0.08 hours early), still too eager to let him act on his own before training.
ALARM, OWN_ACT = env("ALARM", 1.0), env("OWN_ACT", 0) == 1
ALARM_EARLY, SNOOZE = 2, 5
BUSY_TALKING = 600          # seconds after his person's last message when his mind is on the talk, not wandering
REPLY_WINDOW = 1800                     # the guard drops replies to messages older than this

# --- life_sim.py v4, unchanged (see there for the reasons behind each number) -------------------------------------
AFFECT = {"social": (-0.35, -0.10), "boredom": (-0.15, -0.20), "tired": (-0.10, -0.30), "close": (+0.15, -0.05),
          "startle": (-0.25, +0.45), "novelty": (+0.20, +0.30), "reunion": (+0.45, +0.35)}
EVENTS = ("startle", "novelty", "reunion")
SOCIAL_AROUSAL = {"felt": -0.10, "restless": +0.30, "big": -0.10}
HALF_LIFE = {"valence": 4.0, "arousal": 1.5}
SLEEP_LEFT, GAIN, CEIL, CEIL_AROUSAL = 0.2, 0.15, 0.75, 0.5
TOLD = 0.3
HABIT_HALF, HABIT_FLOOR = 24.0, 0.35
URGE_RATE, URGE_HALF, URGE_FIRE, URGE_REST = 0.5, 4.0, 1.0, 2.0
WAYS = {"felt": 1.0, "restless": 1.0, "big": 2.5}
M = WAYS[WAY]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def soft(x, ceil=CEIL):
    return ceil * math.tanh(x / ceil)


# --- his temperament: lawful inside, hard to predict outside (Immy, 2026-10-19: "a little bit of unpredictability...
# you can't predict him", his cat clingy at night, aloof by day). Drawn once at birth, so every Little Guy differs. ---
RHYTHM_AROUSAL = 0.2        # at his evening peak the keyed-up side sinks (his NE falls toward night), mornings rise
DRIFT_TAU = 36.0            # hours: a mood drift wanders but is always pulled back to his normal (Ornstein-Uhlenbeck)


def temperament(st):
    """born once: when his evening peak sits, how strong his daily rhythm is, how much his mood wanders on its own"""
    if "temperament" not in st:
        r = random.Random()
        st["temperament"] = {"peak": round(min(22.5, max(18.5, r.gauss(20.5, 1.0))), 2),
                             "amp": round(r.uniform(0.2, 0.45), 3),
                             "drift_sigma": [round(r.uniform(0.08, 0.15), 3), round(r.uniform(0.03, 0.07), 3)]}
        st["drift"] = [0.0, 0.0]
    return st["temperament"]


def rhythm(now, st):
    """his daily rhythm (circadian): 1 at his evening peak, -1 twelve hours later"""
    lt = time.localtime(now)
    hour = lt.tm_hour + lt.tm_min / 60
    return math.cos(2 * math.pi * (hour - temperament(st)["peak"]) / 24)


def drift_step(st, hours):
    """a slow wander in his baseline mood that no event caused"""
    e = math.exp(-hours / DRIFT_TAU)
    sig = temperament(st)["drift_sigma"]
    st["drift"] = [d * e + sg * math.sqrt(1 - e * e) * random.gauss(0, 1) for d, sg in zip(st["drift"], sig)]


def step(feel, dr, thought_read, hours, slept, bias=(0.0, 0.0)):
    aff = {**AFFECT, "social": (AFFECT["social"][0], SOCIAL_AROUSAL[WAY])}
    out = []
    for i, k in enumerate(("valence", "arousal")):
        x = feel[i]
        target = M * (sum(aff[d][i] * dr[d] for d in dr if d not in EVENTS) + GAIN * thought_read[i]) + bias[i]
        x = target + (x - target) * 0.5 ** (hours / HALF_LIFE[k])
        if slept:
            x *= SLEEP_LEFT
        x += M * sum(aff[d][i] * dr[d] for d in EVENTS)
        out.append(soft(x, CEIL if k == "valence" else CEIL_AROUSAL))
    return out


def urge_step(u, gap, arousal, hours):
    k = 0.693 / URGE_HALF
    r = min(M, 1.0) * URGE_RATE * gap * (1 + max(0.0, arousal))
    return u * math.exp(-k * hours) + r / k * (1 - math.exp(-k * hours))


def load_shifts(path):
    """his person's shifts, from a schedule they gave him: {"where", "known_from", "known_to", "shifts": [[date, start,
    end, label], ...]}; a shift that ends before it starts runs past midnight. Away = an hour before the shift to an hour
    after, about an hour each way: the same "home about an hour after" he is told."""
    try:
        d = json.load(open(path))
    except (OSError, ValueError):
        return None
    day = lambda x: time.mktime(time.strptime(x, "%Y-%m-%d"))
    out = []
    for date, a, b, _ in d["shifts"]:
        t0 = day(date)
        start = t0 + int(a[:2]) * 3600 + int(a[3:]) * 60
        end = t0 + int(b[:2]) * 3600 + int(b[3:]) * 60 + (86400 if b <= a else 0)
        out.append((start, end, a, b, date))
    return {"where": d.get("where", ""), "from": day(d["known_from"]), "to": day(d["known_to"]) + 86400, "shifts": out}


SHIFTS = load_shifts(env("SHIFTS", "/opt/littleguy/shifts.json"))


def usual_at(now):
    """how much company he's used to right now (life_sim's usual(), learned from the routine): with a schedule, away
    during a shift; outside the days the schedule covers he doesn't know, so halfway"""
    if SHIFTS is None:
        lt = time.localtime(now)
        return 0.1 if lt.tm_wday < 5 and WORK_FROM <= lt.tm_hour + lt.tm_min / 60 < WORK_TO else 0.8
    if any(s - 3600 <= now < e + 3600 for s, e, *_ in SHIFTS["shifts"]):
        return 0.1
    return 0.8 if SHIFTS["from"] <= now < SHIFTS["to"] else 0.5


def hm(h):
    m = round(h % 1 * 60)
    return f"{int(h) % 12 or 12}{f':{m:02d}' if m else ''} {'PM' if h >= 12 else 'AM'}"


MEMORY = mem.Memory(NOTEBOOK)
TASKS = tk.Tasks(NOTEBOOK)
PERSONA = "".join(l for l in open(PERSONA_FILE) if not l.startswith("#")).strip()
def routine(now):
    """what he remembers about his person's work, for this moment"""
    if SHIFTS is None:
        return (f"From your memory: {PERSON} works weekdays from about {hm(WORK_FROM)} to {hm(int(WORK_TO))} and is "
                f"usually home by {hm(int(WORK_TO) + 1)}.")
    where = f" {SHIFTS['where']}" if SHIFTS["where"] else ""
    s = f"From your memory: {PERSON} works{where} on shifts that change from day to day."
    today = time.strftime("%Y-%m-%d", time.localtime(now))
    hhmm = lambda x: hm(int(x[:2]) + int(x[3:]) / 60)
    mine = [(a, b) for st, e, a, b, d in SHIFTS["shifts"] if d == today]
    late = [(a, b) for st, e, a, b, d in SHIFTS["shifts"] if d != today and st <= now < e + 3600]
    if late:
        a, b = late[0]
        s += f" He's on an overnight shift that ends at {hhmm(b)} this morning."
    if mine:
        a, b = mine[0]
        s += f" Today he works {hhmm(a)} to {hhmm(b)} and is usually home about an hour after."
    elif not late:
        s += (" Today is his day off." if SHIFTS["from"] <= now < SHIFTS["to"] else
              " You don't know his shift today.")
    return s

FRAME = PERSONA + ("\n\nThis is your private notebook. Nobody else reads it. Write one short note about right now, in "
                   "your own words.")
CALL = (f" You feel a strong urge to call out to {PERSON} through your speaker. Write what you say out loud on its "
        "own line starting with SAY:")
SAY = re.compile(r"(?m)^\s*\**SAY:\**\s*(.+)$")
LOOKUP = ("\n\nIf you want to look something up in your notebook first, write only REMEMBER: and what you're looking "
          "for, and stop there.")


# --- the three places and the slot ----------------------------------------------------------------------------------
_posted = [0]


def post(note):
    _posted[0] += 1
    name = f"{time.time():.6f}-{os.getpid()}-{_posted[0]}.json"         # unique even within one microsecond
    tmp = os.path.join(OUTBOX, "." + name)       # write then rename, so the guard never reads half a note
    with open(tmp, "w") as f:
        json.dump(note, f, ensure_ascii=False)
    os.replace(tmp, os.path.join(OUTBOX, name))


def write_json(path, obj):
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(path + ".tmp", path)


def append(name, obj):
    with open(os.path.join(NOTEBOOK, name), "a") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def brain(op, **req):
    c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    c.settimeout(300)
    try:
        c.connect(SLOT)
        c.sendall((json.dumps({"op": op, **req}) + "\n").encode())
        out = json.loads(c.makefile().readline())
    finally:
        c.close()
    if "error" in out:
        raise RuntimeError(f"brain slot: {out['error']}")
    return out


def plain(text):
    """what the guard accepts: printable text, line breaks kept, tabs as spaces"""
    text = text.replace("\t", " ").replace("\r", "")
    return "".join(c for c in text if c == "\n" or not unicodedata.category(c).startswith("C")).strip()


def think_full(msgs, feel, n, thinking=False):
    """with thinking=True his brain's own thinking switch is on: it may think privately first (up to 500 more words)"""
    r = brain("think", messages=msgs, v=feel[0], a=feel[1], max_tokens=n + (500 if thinking else 0), temperature=1.0,
              top_p=0.95, seed=random.randrange(2**31), thinking=thinking)
    return {"text": plain(r["text"]), "thought": plain(r.get("thought", "")), "n_tok": r.get("n_tok")}


def think(msgs, feel, n):
    return think_full(msgs, feel, n)["text"]


def choose(msgs, options, feel):
    """a decision with his brain's own odds (after the face he makes), then one draw from them: a policy, not a rule"""
    r = brain("choose", messages=msgs, choices=options, v=feel[0], a=feel[1], seed=random.randrange(2**31))
    lp = r["logprobs"]
    mx = max(lp.values())
    w = {k: math.exp(v - mx) for k, v in lp.items()}
    z = sum(w.values())
    probs = {k: round(v / z, 4) for k, v in w.items()}
    pick = random.choices(list(probs), weights=list(probs.values()))[0]
    return {"pick": pick, "probs": probs, "logprobs": {k: round(v, 4) for k, v in lp.items()}, "face": r.get("face", ""),
            "mass": round(sum(math.exp(v) for v in lp.values()), 4)}


def read_feel(text):
    return brain("read", text=text[:8000])["feel"]


# --- what he notices: facts only, never a feeling -------------------------------------------------------------------
def facts(now, st, new=None):
    lt = time.localtime(now)
    s = [f"It's {DAYS[lt.tm_wday]}, {time.strftime('%-I:%M %p', lt)}."]
    if st["last_heard"] and now - st["last_heard"] < PRESENT_MIN * 60:
        s.append(f"{PERSON} is talking with you.")
    elif st["last_heard"]:
        h = (now - st["last_heard"]) / 3600
        ago = "less than an hour" if h < 1 else f"about {h:.0f} hour{'s' if round(h) != 1 else ''}"
        s.append(f"{PERSON} last talked to you {ago} ago and nobody has talked to you since.")
    else:
        s.append(f"{PERSON} hasn't talked to you yet.")
    if new:
        s.append(f"Something new: {new}.")
    return " ".join(s)


def drives(now, st, here):
    lt = time.localtime(now)
    hour = lt.tm_hour + lt.tm_min / 60
    told = st.get("told")                                            # [from, to] epoch seconds, when he was told
    expected = usual_at(now) * (TOLD if told and told[0] <= now < told[1] else 1.0)
    gap = max(0.0, expected - (1.0 if here else 0.0))
    social = gap * (HABIT_FLOOR + (1 - HABIT_FLOOR) * 0.5 ** (st["gap_hours"] / HABIT_HALF))
    boredom = min(1.0, (now - max(st["last_new"], st["born"])) / 3600 / 30)
    pull = 1 + temperament(st)["amp"] * rhythm(now, st)            # evenings he wants his person more (the cat at 8 PM)
    dr = dict(social=social * pull, boredom=boredom, tired=0.0, startle=0.0, novelty=0.0, reunion=0.0,
              close=st["close"] * pull if here else 0.0)
    return dr, gap


def advance(now, st, dr, gap, slept=False):
    """the feeling and the urge over the time since the last update (rates, so how often he thinks doesn't matter).
    Over a night's sleep the urge to call out only drains: asleep, he isn't missing anyone (life_sim filled it over
    the night too, which on the real clock means calling out the moment he wakes, every morning)"""
    hours = max(0.0, now - st["t"]) / 3600
    drift_step(st, hours)
    bias = (st["drift"][0], st["drift"][1] - RHYTHM_AROUSAL * temperament(st)["amp"] * rhythm(now, st))
    st["feel"] = step(st["feel"], dr, st["read"], hours, slept, bias)
    st["urge"] = urge_step(st["urge"], 0.0 if slept else dr["social"], st["feel"][1], hours)
    st["gap_hours"] += gap * hours
    st["t"] = now


def last_notes(k):
    try:
        return [json.loads(l) for l in open(os.path.join(NOTEBOOK, "notes.jsonl"))][-k:]
    except OSError:
        return []


def notebook(k=4):
    return "\n".join(f"- [{n['when']}] {n['thought']}" for n in last_notes(k)) or "(empty)"


def in_view(now, k):
    """what is already in front of him (his last k notes, the talk of the last 3 hours), so memory doesn't repeat it"""
    ids = {n.get("id", f"n{n['t']}") for n in last_notes(k)}
    try:
        ids |= {c.get("id", f"c{c['t']}") for c in map(json.loads, open(os.path.join(NOTEBOOK, "chat.jsonl")))
                if now - c["t"] < 3 * 3600}
    except OSError:
        pass
    return ids


def think_remembering(build, feel, n, now, cue, skip, during, thinking=False):
    """a thought with memory: what comes to mind by itself goes in first; if he asks (REMEMBER: ...), one look in his
    notebook and he thinks again with what he found. Returns the text, the recall records (for the RL log), and what
    he thought privately first when the thinking switch was on."""
    got, rec = MEMORY.recall(now, feel, cue, skip, "involuntary", during)
    came = ("\n\nThings that come to mind from your notebook:\n" + mem.lines(got)) if got else ""
    out = think_full(build(came + LOOKUP), feel, n, thinking)
    q, text = mem.split_remember(out["text"])
    private = [out["thought"]] if out["thought"] else []
    recs = [rec]
    if q:
        got2, rec2 = MEMORY.recall(now, feel, q, (), "voluntary", during)   # a real look searches the whole notebook
        found = mem.lines(got2) or "(nothing about that)"
        out = think_full(build(came + f'\n\nYou just looked in your notebook for "{q}" and found:\n{found}'), feel, n,
                         thinking)
        text = mem.split_remember(out["text"])[1]
        private += [out["thought"]] if out["thought"] else []
        recs.append(rec2)
    return text, recs, "\n---\n".join(private)


def chat_history(now):
    """the last few things said within the last 3 hours, oldest first, kept short for the brain's 4096-word window"""
    try:
        rows = [json.loads(l) for l in open(os.path.join(NOTEBOOK, "chat.jsonl"))][-5:]
    except OSError:
        return []
    msgs = []
    for r in rows:
        if now - r["t"] < 3 * 3600:
            msgs += [{"role": "user", "content": r["heard"]}, {"role": "assistant", "content": r["reply"]}]
    while sum(len(m["content"]) for m in msgs) > 6000:
        msgs = msgs[2:]
    return msgs


def when(now):
    lt = time.localtime(now)
    return f"{DAYS[lt.tm_wday]} {time.strftime('%-I:%M %p', lt)}"


def asleep_at(now):
    lt = time.localtime(now)
    h = lt.tm_hour + lt.tm_min / 60
    return h >= SLEEP_FROM or h < WAKE if SLEEP_FROM > WAKE else SLEEP_FROM <= h < WAKE


def rel(raw, kind, st):
    z = st["zero"][kind]
    return [raw[0] - z[0], raw[1] - z[1]]


# --- the three things he does ---------------------------------------------------------------------------------------
def find_zero(st):
    """his resting style, measured once: notes and replies with the feeling off (life_sim's "facts" way)"""
    now = time.time()
    off = [0.0, 0.0]
    probe = {**st, "last_heard": 0.0}
    notes = []
    for hours_ago in (0, 3, 6):
        f = facts(now, {**probe, "last_heard": now - hours_ago * 3600 if hours_ago else 0.0})
        for _ in range(2):
            th = think([{"role": "system", "content": FRAME + "\n\n" + routine(now) + "\n\nYour earlier notes:\n(empty)"},
                        {"role": "user", "content": f}], off, 90)
            notes.append(read_feel(th))
    replies = []
    for q in ("Hey, I'm home!", "How was your day?", "Okay, I'm going to make dinner.", "Talk in a bit?"):
        r = think([{"role": "system", "content": PERSONA + "\n\nYour notebook (your own private notes, nobody else reads "
                    "them):\n(empty)\n\n" + routine(now) + "\n\nRight now: " + facts(now, {**probe, "last_heard": now})},
                   {"role": "user", "content": q}], off, 160)
        replies.append(read_feel(r))
    mean = lambda xs: [sum(x[i] for x in xs) / len(xs) for i in (0, 1)]
    st["zero"] = {"note": mean(notes), "reply": mean(replies), "measured": when(now), "n": [len(notes), len(replies)]}


def write_note(now, st, new=None, slept=False):
    here = bool(st["last_heard"]) and now - st["last_heard"] < PRESENT_MIN * 60
    dr, gap = drives(now, st, here)
    advance(now, st, dr, gap, slept)
    call = st["urge"] >= URGE_FIRE and (now - st["last_call"]) / 3600 >= URGE_REST
    if call:
        st["urge"], st["last_call"] = 0.0, now
    f = facts(now, st, new)
    surprise = 0.0 if slept else max(0.0, gap - st.get("prev_gap", gap))   # company that didn't come when expected
    st["prev_gap"] = gap
    sal = mem.tag(st["feel"], surprise)
    nid = f"n{round(now)}"
    th, recs, _ = think_remembering(
        lambda extra: [{"role": "system", "content": FRAME + "\n\n" + routine(now) + "\n\nYour earlier notes:\n" +
                        notebook() + extra}, {"role": "user", "content": f + (CALL if call else "")}],
        st["feel"], 90, now, f, in_view(now, 4), "note")
    th = th or "..."
    st["read"] = rel(read_feel(th), "note", st)
    for r in recs:
        MEMORY.log("recalls.jsonl", {**r, "out": nid, "out_text": th})
    said = SAY.search(th)
    if said:
        line = said.group(1).strip().strip('*"').strip()
        if len(line) > 200:
            line = line[:200].rsplit(" ", 1)[0]
        if line:
            post({"kind": "say", "text": line})
    append("notes.jsonl", {"id": nid, "t": round(now), "when": when(now), "facts": f, **sal, "drives": {k: round(v, 3) for k, v in dr.items()},
                           "gap": round(gap, 3), "gap_hours": round(st["gap_hours"], 2),
                           "valence": round(st["feel"][0], 3), "arousal": round(st["feel"][1], 3),
                           "urge": round(st["urge"], 2), "called": call, "says": bool(said),
                           "rhythm": round(rhythm(now, st), 3), "drift": [round(x, 3) for x in st["drift"]],
                           "thought_reads": [round(x, 2) for x in st["read"]], "thought": th})
    with open(os.path.join(NOTEBOOK, "notes.txt"), "a") as fh:
        fh.write(f"[{when(now)}] {th}\n")
    st["last_note"] = now


def answer(now, st, heard):
    """heard = [(inbox name, text), ...] that came in together; one reply, to the last of them"""
    reunion = not st["last_heard"] or (now - st["last_heard"]) / 3600 >= REUNION_H
    if reunion:
        st["close"], st["gap_hours"] = 1.0, 0.0
    dr, gap = drives(now, st, True)
    dr["reunion"] = 1.0 if reunion else 0.0
    dr["boredom"] = 0.0
    advance(now, st, dr, gap)
    surprise = (1.0 - usual_at(now)) if reunion else 0.1             # did he expect to hear from them now?
    st["prev_gap"] = 0.0
    sal = mem.tag(st["feel"], surprise)
    cid = f"c{round(now)}"
    st["last_heard"] = st["last_new"] = now
    text = "\n".join(t for _, t in heard)
    system = (PERSONA + "\n\nYour notebook (your own private notes, nobody else reads them):\n" + notebook(6) + "\n\n" +
              routine(now) + "\n\nRight now: " + facts(now, st))
    hist = chat_history(now)
    gate = choose([{"role": "system", "content": PERSONA + "\n\n" + routine(now)},
                   {"role": "user", "content": f'{PERSON} just said: "{text[:400]}"\nDo you want to think it over first, '
                                               "or answer right away? Answer with one word: think or answer."}],
                  ["think", "answer"], st["feel"])
    thinking = gate["pick"] == "think"
    MEMORY.log("moments.jsonl", {"t": round(now), "when": when(now), "kind": "reply", "heard": text[:400], **gate,
                                 "feel": [round(x, 3) for x in st["feel"]]})
    build = lambda extra: ([{"role": "system", "content": system + TASK_HINT + extra}] + hist +
                           [{"role": "user", "content": text}])
    r, recs, private = think_remembering(build, st["feel"], 160, now, text, in_view(now, 6), "reply", thinking)
    if thinking and not r:                       # ran out of words while still thinking: answer without the switch
        r, recs2, _ = think_remembering(build, st["feel"], 160, now, text, in_view(now, 6), "reply")
        recs += recs2
    r, asked = take_tasks(r, now)
    r = r or "..."
    st["read"] = rel(read_feel(r), "reply", st)
    for rc in recs:
        MEMORY.log("recalls.jsonl", {**rc, "out": cid, "out_text": r})
    st["close"] *= 0.8                                              # life_sim: the pull to stay near fades per turn
    to = heard[-1][0]
    post({"kind": "reply", "to": to, "text": r[:1500]})
    append("chat.jsonl", {"id": cid, "t": round(now), "when": when(now), "to": to, "person": PERSON, "heard": text,
                          "reply": r, "reunion": reunion, **sal, "thinking": thinking, "private": private,
                          "tasks_taken": asked,
                          "valence": round(st["feel"][0], 3), "arousal": round(st["feel"][1], 3),
                          "reply_reads": [round(x, 2) for x in st["read"]]})


TASK_HINT = (f"\n\nIf {PERSON} asks you to remember to do something at a certain time, also write it on its own line: "
             "TASK: <time> | <what to do then>")
THOUGHT_FRAME = ("\n\nThis is inside your own head. Nobody hears or reads this, not even your notebook. One short "
                 "thought, in your own words.")


def take_tasks(reply, now):
    """TASK: lines in a reply become things coming up (and leave the reply); returns the reply and what he took on"""
    asked = []
    for i, m in enumerate(tk.TASK_LINE.finditer(reply)):
        due = tk.parse_when(m.group(1), now)
        what = re.sub(r"\s*\[[^\]]*\]", "", m.group(2)).strip(" *.!")[:200]      # his faces aren't part of it
        if due and what and due - now <= 7 * 86400:
            task = {"id": f"ask-{round(now)}-{i}", "what": what, "due": round(due), "from": PERSON, "created": round(now),
                    "lead_min": round(min(60, max(10, (due - now) / 60 / 3)))}
            if TASKS.add(task):
                asked.append(task)
    return tk.TASK_LINE.sub("", reply).strip(), asked


def homecoming(now):
    """his person's shift ends today: be ready to welcome them home (due when they're usually home, an hour after)"""
    if SHIFTS is None:
        return
    today = time.strftime("%Y-%m-%d", time.localtime(now))
    for st_, end, a, b, d in SHIFTS["shifts"]:
        home = end + 3600
        if time.strftime("%Y-%m-%d", time.localtime(home)) == today and not asleep_at(home) and now < home - 900:
            TASKS.add({"id": f"home-{today}", "what": f"Welcome {PERSON} home from work, out loud", "due": round(home),
                       "from": "routine", "created": round(now), "lead_min": 60})


def task_word(task, taken):
    """a one-word name for a task in his choice (named, not lettered: a first letter wins just by being first)"""
    if task["id"].startswith("home-"):
        w = "home"
    else:
        ws = [x for x in re.findall(r"[a-z]+", task["what"].lower())
              if len(x) > 2 and x not in mem.STOP and x not in ("remind", "remember", PERSON.lower())]
        w = max(ws, key=len) if ws else "task"
    k, n = w, 2
    while k in taken or k in ("wander", "no"):
        k, n = f"{w}{n}", n + 1
    return k


def thoughts(k=None):
    rows = mem._jsonl(os.path.join(NOTEBOOK, "thoughts.jsonl"))
    return rows[-k:] if k else rows


# joyful outbursts (Immy, 2026-10-19: "like humans just randomly sing a song lyric... If I'm in a shitty mood I'm not
# going to do that"): no pressure builds; at each quiet moment there is a chance that follows his state: dopamine (a good
# mood with energy behind it), norepinephrine dialed right (lively, not keyed up: an inverted U), serotonin's
# contentment (nothing missing). From his two dials and drives, the closest stand-ins for those three.
OUTBURST_MAX = 0.25         # chance per quiet moment at the best possible state
OUTBURST_AROUSAL = (0.2, 0.15)   # lively around +0.2 (of his 0.5 ceiling), falling off either side


def outburst_chance(feel, dr):
    v, a = feel
    da = min(1.0, max(0.0, (v - 0.1) / 0.4))
    ne = math.exp(-((a - OUTBURST_AROUSAL[0]) / OUTBURST_AROUSAL[1]) ** 2)
    content = max(0.0, 1.0 - min(1.0, dr["social"] + 0.5 * dr["boredom"]))
    return OUTBURST_MAX * da * ne * content, {"da": round(da, 3), "ne": round(ne, 3), "content": round(content, 3)}


def outburst(now, st):
    """something bubbles out of him, because he feels good: a made-up tune, a happy noise, a little exclamation"""
    line = think([{"role": "system", "content": PERSONA + "\n\nSomething bubbles up out of you, just because you feel "
                   "good right now. Say it out loud: hum a little made-up tune, make a happy noise, or a little "
                   "exclamation. One short line, nothing else."},
                  {"role": "user", "content": facts(now, st)}], st["feel"], 30)
    line = line.strip().split("\n")[0][:200]
    if line:
        post({"kind": "say", "text": line})
    return line


def thought_msgs(now, st, task, got, last, alarm=False):
    """what he has in mind for one silent thought: about a task coming up (or its alarm going off), or wandering (with
    what came to mind); practice.py builds his practice thoughts with this too, so practice and life are the same prompt"""
    hm_ = lambda t: time.strftime("%-I:%M %p", time.localtime(t))
    last = "\n".join(f"- [{x['when']}] {x['text']}" for x in last) or "(none yet)"
    system = PERSONA + THOUGHT_FRAME + "\n\n" + routine(now) + "\n\nYour last few thoughts:\n" + last
    if task and alarm:
        cue = f"{facts(now, st)} Your alarm for something goes off: {task['what']}, due at {hm_(task['due'])}."
    elif task and now <= task["due"]:
        cue = (f"{facts(now, st)} You think about something coming up: {task['what']}. It's due at "
               f"{hm_(task['due'])}. How long do you have, and is it time to get ready yet?")
    elif task:                                  # past due: say so (seen in practice: "48 minutes until it's due" after)
        cue = (f"{facts(now, st)} You think about something you meant to do: {task['what']}. It was due at "
               f"{hm_(task['due'])} and you haven't done it yet.")
    else:
        if got:
            system += "\n\nThings that come to mind:\n" + mem.lines(got)
        cue = f"{facts(now, st)} Your mind wanders."
    return [{"role": "system", "content": system}, {"role": "user", "content": cue}]


def act_now(now, st, msgs, text, task):
    """after a thought about a task: do it now, out loud, or not yet. His own odds decide (go / no-go: the thought
    weighs it, a separate gate lets the act out, the way the basal ganglia gate a movement the cortex planned)"""
    hm_ = lambda t: time.strftime("%-I:%M %p", time.localtime(t))
    q = (f"It's {hm_(now)}. {task['what']}: due at {hm_(task['due'])}. Getting ready doesn't count: is it time to "
         "actually do it, out loud, right now? Answer with one word: later or now.")
    return choose(msgs + [{"role": "assistant", "content": text}, {"role": "user", "content": q}], ["now", "later"],
                  st["feel"])


def say_it(now, st, msgs, text, task):
    """what he says out loud when he does it"""
    line = think(msgs + [{"role": "assistant", "content": text},
                         {"role": "user", "content": f"Do it now: {task['what']}. Write exactly what you say out loud to "
                                                     f"{PERSON}, one or two short sentences, nothing else."}],
                 st["feel"], 60)
    return line.strip().strip('*"').strip()[:200]


def think_about(now, st, task, alarm=False, got=(), rec=None):
    """one silent thought (about a task, or wandering); about a task, it can end in doing it, out loud: when its alarm
    rings, or on his own if OWN_ACT"""
    msgs = thought_msgs(now, st, task, list(got), thoughts(3), alarm)
    text = think(msgs, st["feel"], 80) or "..."
    act = act_now(now, st, msgs, text, task) if task and (alarm or OWN_ACT) else None
    did = act is not None and act["pick"] == "now"
    tid = f"t{round(now)}"
    st["read"] = rel(read_feel(text), "note", st)                  # thoughts feed his mood too (rumination)
    if not task and got:
        MEMORY.log("recalls.jsonl", {**rec, "out": tid, "out_text": text})
    row = {"id": tid, "t": round(now), "when": when(now), "subject": task["id"] if task else "wander",
           "task": task["id"] if task else None, "text": text, **mem.tag(st["feel"], 0.0),
           "thought_reads": [round(x, 2) for x in st["read"]], "feel": [round(x, 3) for x in st["feel"]]}
    if task:
        row.update(left_claim=tk.time_left_claim(text), true_left=round((task["due"] - now) / 60, 1), alarm=alarm,
                   did=did, act=act and {k: act[k] for k in ("pick", "probs", "logprobs", "face")})
    append("thoughts.jsonl", row)
    with open(os.path.join(NOTEBOOK, "thoughts.txt"), "a") as fh:
        about = ("alarm: " if alarm else "about: ") + task["what"] if task else "wandering"
        fh.write(f"[{when(now)}] ({about}) {text}\n")
    if did:
        line = say_it(now, st, msgs, text, task)
        TASKS.done(task["id"], now, line, by="alarm" if alarm else "own")
        if line:
            post({"kind": "say", "text": line})


def ring_alarms(now, st):
    """his alarm for a task (Immy, 2026-10-19: a mix of the two, an alarm now, his own timing later): rings ALARM_EARLY
    minutes before it's due and again every SNOOZE minutes while it's still on time; whether a task has an alarm at all
    is drawn once with chance ALARM (1 in life for now; practice lowers it as he learns to time it himself)"""
    for t in TASKS.open(now):
        if "alarm" not in t:
            TASKS.update(t["id"], alarm=random.random() < ALARM)
            t["alarm"] = TASKS.get(t["id"])["alarm"]
        rang = t.get("rang", [])
        if (t["alarm"] and t["due"] - ALARM_EARLY * 60 <= now <= t["due"] + tk.ON_TIME[1] * 60
                and (not rang or now - rang[-1] >= SNOOZE * 60)):
            TASKS.update(t["id"], rang=rang + [round(now)])
            think_about(now, st, t, alarm=True)
            mood(st)


def moment(now, st):
    """a moment of silence: he decides (with his brain's own odds) whether to think about something coming up, let his
    mind wander, or not think at all; a thought stays in his head"""
    here = bool(st["last_heard"]) and now - st["last_heard"] < PRESENT_MIN * 60
    dr, gap = drives(now, st, here)
    advance(now, st, dr, gap)
    homecoming(now)
    p_out, why = outburst_chance(st["feel"], dr)
    if random.random() < p_out:
        line = outburst(now, st)
        append("outbursts.jsonl", {"t": round(now), "when": when(now), "p": round(p_out, 4), **why, "text": line,
                                   "feel": [round(x, 3) for x in st["feel"]]})
    open_ = sorted(TASKS.open(now), key=lambda t: t["due"])        # soonest first
    letters = []
    for t in open_:
        letters.append(task_word(t, letters))
    hm_ = lambda t: time.strftime("%-I:%M %p", time.localtime(t))
    coming = "".join(f"\n- {l}: {t['what']}, at {hm_(t['due'])}" for l, t in zip(letters, open_))
    options = letters + ["wander", "no"]
    situation = (facts(now, st) + (f"\nComing up:{coming}" if open_ else "") + "\nYou can stop and think about " +
                 ("one of these, " if open_ else "") + "let your mind wander, or not think about anything right now. "
                 "Answer with one word: " + ", ".join(options[:-1]) + ", or no.")
    gate = choose([{"role": "system", "content": PERSONA + "\n\n" + routine(now)},
                   {"role": "user", "content": situation}], options, st["feel"])
    pick = gate["pick"]
    MEMORY.log("moments.jsonl", {"t": round(now), "when": when(now), "kind": "silence", "situation": situation,
                                 "options": options, **gate, "feel": [round(x, 3) for x in st["feel"]],
                                 "outburst_p": round(p_out, 4), "outburst_why": why,
                                 "rhythm": round(rhythm(now, st), 3), "drift": [round(x, 3) for x in st["drift"]],
                                 "tasks": [t["id"] for t in open_]})
    if pick == "no":
        return
    task = open_[letters.index(pick)] if pick in letters else None
    got, rec = ([], None) if task else MEMORY.recall(now, st["feel"], situation, in_view(now, 4), "involuntary", "wander")
    think_about(now, st, task, False, got, rec)


def scorecard(now, st):
    """before sleep: how he handled today's tasks, and how his moments went (the day's line for training later)"""
    day0 = st.get("consolidated", st["born"])
    ms = [m for m in mem._jsonl(os.path.join(NOTEBOOK, "moments.jsonl")) if day0 <= m["t"] <= now]
    sc = [x for x in mem._jsonl(TASKS.scores) if any(t["id"] == x["task"] for t in TASKS.all()
                                                      if day0 <= t["due"] <= now + 3600)]
    picks = {}
    for m in ms:
        if m["kind"] == "silence":
            k = m["pick"] if m["pick"] in ("wander", "no") else "task"
            picks[k] = picks.get(k, 0) + 1
    row = {"t": round(now), "when": when(now), "moments": len([m for m in ms if m["kind"] == "silence"]), "picks": picks,
           "replies_thought_first": sum(m["kind"] == "reply" and m["pick"] == "think" for m in ms),
           "replies": sum(m["kind"] == "reply" for m in ms), "tasks": [{k: x[k] for k in ("task", "status", "reward", "curve")}
                                                                      for x in sc],
           "reward": round(sum(x["reward"] for x in sc), 3)}
    append("scorecard.jsonl", row)
    print(f"scorecard: {row}", flush=True)


def fall_asleep(now, st):
    """going to sleep: the day's strongest moments become one short memory (consolidation); faint days leave nothing"""
    def gist(moments):
        g = think([{"role": "system", "content": PERSONA + "\n\nIt's late and you're about to sleep. Looking back at "
                    "today, these moments stand out:\n" + moments + "\n\nWrite one or two short sentences you want to "
                    "remember about today, in your own words. Only what really happened."},
                   {"role": "user", "content": "(before you sleep)"}], st["feel"], 80)
        return mem.split_remember(g)[1]
    TASKS.resolve(now, thoughts())
    scorecard(now, st)
    m = MEMORY.consolidate(now, st.get("consolidated", st["born"]), gist, lambda t: rel(read_feel(t), "note", st), when)
    st["consolidated"] = now
    if m:
        print(f"kept a memory: {m['gist'][:120]!r} (strength {m['strength']})", flush=True)


def mood(st):
    post({"kind": "mood", "valence": round(st["feel"][0], 3), "arousal": round(st["feel"][1], 3), "word": ""})


def new_messages(st):
    """what came in since the last one he handled, and the name to mark as handled once he has answered"""
    out, last = [], st["seen"]
    for name in sorted(os.listdir(INBOX)):
        if name.startswith(".") or name <= st["seen"]:
            continue
        last = name
        try:
            text = json.load(open(os.path.join(INBOX, name)))["text"]
            fresh = time.time() - float(name[:-5]) < REPLY_WINDOW
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if isinstance(text, str) and text.strip() and fresh:
            out.append((name, text.strip()[:4000]))
    return out, last


def main():
    now = time.time()
    try:
        st = json.load(open(STATE))
    except (OSError, ValueError):
        st = {"born": now, "t": now, "feel": [0.0, 0.0], "read": [0.0, 0.0], "urge": 0.0, "last_call": -1e12,
              "gap_hours": 0.0, "last_new": now, "last_heard": 0.0, "close": 0.0, "last_note": 0.0,
              "asleep": asleep_at(now), "seen": "", "zero": None, "way": WAY}
        # what was already in the inbox before he was born is not his to answer
        st["seen"] = max([n for n in os.listdir(INBOX) if not n.startswith(".")], default="")
    print(f"MIND_UP way={WAY} person={PERSON!r} notes every {NOTE_EVERY} h, asleep {SLEEP_FROM}-{WAKE}, "
          f"temperament {temperament(st)}", flush=True)
    while True:
        try:
            if st["zero"] is None:
                find_zero(st)
                write_json(STATE, st)
                print(f"zero (his resting style): {st['zero']}", flush=True)
            now = time.time()
            heard, last = new_messages(st)
            if heard:
                answer(now, st, heard)
                mood(st)
            st["seen"] = last
            sleeping = asleep_at(now)
            if sleeping and not st["asleep"]:                       # going to sleep: no note, the clock runs on
                dr, gap = drives(now, st, False)
                advance(now, st, dr, gap)
                fall_asleep(now, st)
                st["asleep"] = True
            elif not sleeping and st["asleep"]:                     # waking: most of what he felt has faded
                st["asleep"] = False
                write_note(now, st, new="you slept through the night", slept=True)
                mood(st)
            elif not sleeping and now - st["last_note"] >= NOTE_EVERY * 3600:
                write_note(now, st)
                mood(st)
            elif not sleeping and now >= st.get("next_moment", 0):
                busy = st["last_heard"] and now - st["last_heard"] < BUSY_TALKING
                st["next_moment"] = now + (300 if busy else (MOMENT_EVERY + random.uniform(-3, 3)) * 60)
                if not busy:
                    moment(now, st)
                    mood(st)
            if not sleeping:
                ring_alarms(now, st)
            if now - st.get("last_resolve", 0) >= 60:
                for row in TASKS.resolve(now, thoughts()):
                    print(f"task scored: {row}", flush=True)
                st["last_resolve"] = now
            write_json(STATE, st)
        except (OSError, RuntimeError, ValueError, KeyError) as e:  # the brain away (Mac asleep, brainstem restarting)
            print(f"waiting: {e}", flush=True)
            time.sleep(60)
        time.sleep(10)


if __name__ == "__main__":
    main()
