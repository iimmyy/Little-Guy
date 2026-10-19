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

Settings (env): PERSON (default "your person"), PERSONA (default /opt/littleguy/persona.txt; '#' lines dropped),
  WAY (felt | restless | big, default felt), NOTE_EVERY (hours between notes, default 2), SLEEP_FROM / WAKE (hours,
  default 23 / 7), SHIFTS (his person's work shifts, default /opt/littleguy/shifts.json; without it, weekdays
  WORK_FROM to WORK_TO, default 9 / 17.5), PRESENT_MIN (default 30),
  REUNION_H (silence that makes a message a homecoming, default 1)
"""
import json, math, os, random, re, socket, sys, time, unicodedata

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


def step(feel, dr, thought_read, hours, slept):
    aff = {**AFFECT, "social": (AFFECT["social"][0], SOCIAL_AROUSAL[WAY])}
    out = []
    for i, k in enumerate(("valence", "arousal")):
        x = feel[i]
        target = M * (sum(aff[d][i] * dr[d] for d in dr if d not in EVENTS) + GAIN * thought_read[i])
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


def think(msgs, feel, n):
    return plain(brain("think", messages=msgs, v=feel[0], a=feel[1], max_tokens=n, temperature=1.0, top_p=0.95,
                       seed=random.randrange(2**31))["text"])


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
    dr = dict(social=social, boredom=boredom, tired=0.0, startle=0.0, novelty=0.0, reunion=0.0,
              close=st["close"] if here else 0.0)
    return dr, gap


def advance(now, st, dr, gap, slept=False):
    """the feeling and the urge over the time since the last update (rates, so how often he thinks doesn't matter).
    Over a night's sleep the urge to call out only drains: asleep, he isn't missing anyone (life_sim filled it over
    the night too, which on the real clock means calling out the moment he wakes, every morning)"""
    hours = max(0.0, now - st["t"]) / 3600
    st["feel"] = step(st["feel"], dr, st["read"], hours, slept)
    st["urge"] = urge_step(st["urge"], 0.0 if slept else dr["social"], st["feel"][1], hours)
    st["gap_hours"] += gap * hours
    st["t"] = now


def notebook(k=4):
    try:
        notes = [json.loads(l) for l in open(os.path.join(NOTEBOOK, "notes.jsonl"))][-k:]
    except OSError:
        notes = []
    return "\n".join(f"- [{n['when']}] {n['thought']}" for n in notes) or "(empty)"


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
    th = think([{"role": "system", "content": FRAME + "\n\n" + routine(now) + "\n\nYour earlier notes:\n" + notebook()},
                {"role": "user", "content": f + (CALL if call else "")}], st["feel"], 90)
    st["read"] = rel(read_feel(th), "note", st)
    said = SAY.search(th)
    if said:
        line = said.group(1).strip().strip('*"').strip()
        if len(line) > 200:
            line = line[:200].rsplit(" ", 1)[0]
        if line:
            post({"kind": "say", "text": line})
    append("notes.jsonl", {"t": round(now), "when": when(now), "facts": f, "drives": {k: round(v, 3) for k, v in dr.items()},
                           "gap": round(gap, 3), "gap_hours": round(st["gap_hours"], 2),
                           "valence": round(st["feel"][0], 3), "arousal": round(st["feel"][1], 3),
                           "urge": round(st["urge"], 2), "called": call, "says": bool(said),
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
    st["last_heard"] = st["last_new"] = now
    text = "\n".join(t for _, t in heard)
    system = (PERSONA + "\n\nYour notebook (your own private notes, nobody else reads them):\n" + notebook(6) + "\n\n" +
              routine(now) + "\n\nRight now: " + facts(now, st))
    r = think([{"role": "system", "content": system}] + chat_history(now) + [{"role": "user", "content": text}],
              st["feel"], 160) or "..."
    st["read"] = rel(read_feel(r), "reply", st)
    st["close"] *= 0.8                                              # life_sim: the pull to stay near fades per turn
    to = heard[-1][0]
    post({"kind": "reply", "to": to, "text": r[:1500]})
    append("chat.jsonl", {"t": round(now), "when": when(now), "to": to, "heard": text, "reply": r, "reunion": reunion,
                          "valence": round(st["feel"][0], 3), "arousal": round(st["feel"][1], 3),
                          "reply_reads": [round(x, 2) for x in st["read"]]})


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
    print(f"MIND_UP way={WAY} person={PERSON!r} notes every {NOTE_EVERY} h, asleep {SLEEP_FROM}-{WAKE}", flush=True)
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
                st["asleep"] = True
            elif not sleeping and st["asleep"]:                     # waking: most of what he felt has faded
                st["asleep"] = False
                write_note(now, st, new="you slept through the night", slept=True)
                mood(st)
            elif not sleeping and now - st["last_note"] >= NOTE_EVERY * 3600:
                write_note(now, st)
                mood(st)
            write_json(STATE, st)
        except (OSError, RuntimeError, ValueError, KeyError) as e:  # the brain away (Mac asleep, brainstem restarting)
            print(f"waiting: {e}", flush=True)
            time.sleep(60)
        time.sleep(10)


if __name__ == "__main__":
    main()
