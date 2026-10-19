"""Things coming up that Little Guy holds in mind (prospective memory), and the scorecard that makes his silent thinking
rewardable (Immy, 2026-10-19: "a small little task that he knows is coming up, and so occasionally he'll think about
it... if he's just constantly planning it out, we'll punish him... thinking, then doing something else, then thinking";
his dinner at 5: "I got 5 hours... 3 more hours... it's an hour out, time to get ready").

A task: {"id", "what", "due" (epoch), "act": "say", "lead_min" (when getting ready starts), "from", "created",
         "status": "open" | "done" | "missed", "done_t", "done_text"}
His thoughts about a task are check-ins. The scorecard rewards Immy's pattern, the J-shaped checking curve of time-based
prospective memory (one early check to calibrate, little in the middle, more as it gets close, then doing it on time):
  done on time (10 min early to 20 min late)   +1.0     late (up to an hour)   +0.3
  done too early                               -0.3     missed                 -1.0
  a thought about it while getting ready (from an hour-and-a-quarter out to 5 min out)    +0.5 once
  an early check to get his bearings (in the first quarter of the time he had)             +0.2 once
  "how long do I have" (within 15 min or 15%): +0.3 x (share right - share wrong), so checking more never earns more
  each thought about it                        -0.03    (thinking isn't free)
  each one far from it (earlier than twice the lead), after the first one   -0.05 more   ("don't keep planning it")
Worked examples (dinner at 5, told at noon): Immy's way (12, 2, 4 PM "an hour out", done 5:02) +1.83; one thought at
4:30 then done +1.77; done on time but thinking every 10 minutes from noon (30 thoughts) +0.25; missed -1.0.
The same function scores his real days now and his practice days later (RL), so the two can be compared.
"""
import json, os, re, time

LEAD_MIN = 60
ON_TIME = (-10, 20)            # minutes around due
LATE_MIN = 60                  # after this it's missed
PREP = (-75, -5)               # minutes around due (lead 60): a thought here is getting ready
R = {"on_time": 1.0, "late": 0.3, "early": -0.3, "missed": -1.0, "prep": 0.5, "calibrate": 0.2, "claims": 0.3, "cost": -0.03,
     "far": -0.05}

NUMS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
        "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40,
        "forty-five": 45, "fifty": 50, "ninety": 90}
ONES = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
for tens, v in (("twenty", 20), ("thirty", 30), ("forty", 40), ("fifty", 50)):     # "thirty-seven", "forty five"
    for i, o in enumerate(ONES, 1):
        NUMS[f"{tens}-{o}"] = NUMS[f"{tens} {o}"] = v + i
_N = r"(\d+(?:\.\d+)?|" + "|".join(sorted(NUMS, key=len, reverse=True)) + r")"
CLAIM = re.compile(r"(?i)\b(?:(half an? hour)|" + _N + r"\s+(?:(?:more|whole|full|good|solid|left|or so)\s+)?"
                   r"(hours?|hrs?|minutes?|mins?)\b(\s+and\s+a\s+half)?(?:\s+and\s+" + _N + r"\s+(minutes?|mins?))?)")
DO = re.compile(r"(?m)\**\bDO\b\**\s*:\**\s*([^\n]+)")      # capital DO: only ("what I need to do: ..." isn't doing it)
TASK_LINE = re.compile(r"(?mi)^\W*TASK\s*:\s*([^\n|]+?)\s*\|\s*([^\n]+)$")


def _num(x):
    x = x.lower().replace("  ", " ")
    return float(x) if x[0].isdigit() else NUMS[x]


def time_left_claim(text):
    """the first "how long do I have" in a thought, in minutes ("5 hours", "3 more hours", "an hour and a half",
    "half an hour", "4 hours and 55 minutes"); None if there isn't one"""
    m = CLAIM.search(text)
    if not m:
        return None
    if m.group(1):
        return 30.0
    n, unit = _num(m.group(2)), m.group(3).lower()
    mins = n * 60 if unit.startswith("h") else n
    if m.group(4):
        mins += 30
    if m.group(5):
        mins += _num(m.group(5))
    return mins


def claim_right(claim, true_left):
    return abs(claim - true_left) <= max(15.0, 0.15 * abs(true_left))


def parse_when(text, now):
    """"5 PM", "5:30 pm", "17:00", "in 2 hours", "in 20 minutes" -> the next such moment (epoch), or None"""
    t = re.sub(r"^(?:at|by|around|about)\s+", "", text.strip().lower())
    m = re.match(r"in\s+(.+)$", t)
    if m:
        c = time_left_claim(m.group(1))
        return now + c * 60 if c else None
    m = re.match(r"(\d{1,2})(?::(\d{2}))?\s*([ap])?\.?\s*m?\.?$", t)
    if not m:
        return None
    h, mi, ap = int(m.group(1)), int(m.group(2) or 0), m.group(3)
    if h > 23 or mi > 59:
        return None
    if ap:
        h = h % 12 + (12 if ap == "p" else 0)
    lt = time.localtime(now)
    for add_days in range(2):
        for hh in ([h] if ap or h > 12 else [h, h + 12]):        # "at 5" with no am/pm: the next 5 o'clock
            cand = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday + add_days, hh, mi, 0, 0, 0, -1))
            if cand > now:
                return cand
    return None


class Tasks:
    def __init__(self, notebook):
        self.path = os.path.join(notebook, "tasks.json")
        self.scores = os.path.join(notebook, "scores.jsonl")

    def all(self):
        try:
            return json.load(open(self.path))
        except (OSError, ValueError):
            return []

    def save(self, ts):
        with open(self.path + ".tmp", "w") as f:
            json.dump(ts, f, ensure_ascii=False, indent=1)
        os.replace(self.path + ".tmp", self.path)

    def open(self, now):
        return [t for t in self.all() if t["status"] == "open" and now < t["due"] + LATE_MIN * 60]

    def add(self, task):
        ts = self.all()
        if any(t["id"] == task["id"] for t in ts):
            return False
        ts.append({"status": "open", "lead_min": LEAD_MIN, "act": "say", **task})
        self.save(ts)
        return True

    def done(self, tid, now, text):
        ts = self.all()
        for t in ts:
            if t["id"] == tid and t["status"] == "open":
                t.update(status="done", done_t=round(now), done_text=text)
        self.save(ts)

    def resolve(self, now, thoughts):
        """score every task that's finished (done, or past its last chance); returns the new score rows"""
        ts, rows = self.all(), []
        for t in ts:
            if t.get("scored"):
                continue
            if t["status"] == "open" and now >= t["due"] + LATE_MIN * 60:
                t["status"] = "missed"
            if t["status"] in ("done", "missed"):
                row = score(t, [x for x in thoughts if x.get("task") == t["id"]])
                t["scored"] = True
                rows.append(row)
                with open(self.scores, "a") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if rows:
            self.save(ts)
        return rows


def score(task, thoughts):
    """the scorecard for one task, from his thoughts about it (each: t, left_claim, true_left)"""
    due, lead = task["due"], task.get("lead_min", LEAD_MIN)
    mins = lambda t: (t - due) / 60
    parts = {}
    if task["status"] == "done":
        d = mins(task["done_t"])
        parts["act"] = R["on_time"] if ON_TIME[0] <= d <= ON_TIME[1] else R["early"] if d < ON_TIME[0] else R["late"]
    else:
        parts["act"] = R["missed"]
    prep_from, prep_to = -(lead + 15), PREP[1]
    before_act = [x for x in thoughts if task["status"] != "done" or x["t"] <= task["done_t"]]
    parts["prep"] = R["prep"] if any(prep_from <= mins(x["t"]) <= prep_to for x in before_act) else 0.0
    span = due - task.get("created", due)
    parts["calibrate"] = R["calibrate"] if span > 0 and any(x["t"] - task["created"] <= span / 4 for x in thoughts) else 0.0
    claims = [x for x in thoughts if x.get("left_claim") is not None]
    right = sum(claim_right(x["left_claim"], x["true_left"]) for x in claims)
    parts["claims"] = round(R["claims"] * (2 * right - len(claims)) / len(claims), 3) if claims else 0.0
    parts["cost"] = round(R["cost"] * len(thoughts), 3)
    far = [x for x in sorted(thoughts, key=lambda x: x["t"])[1:] if mins(x["t"]) < -2 * lead]
    parts["far"] = round(R["far"] * len(far), 3)
    # the checking curve: thoughts per hour-before-due bin (the J shape, if he has Immy's habit)
    curve = {}
    for x in thoughts:
        h = int(max(0, -mins(x["t"])) // 60)
        curve[f"{h}h"] = curve.get(f"{h}h", 0) + 1
    return {"task": task["id"], "what": task["what"], "status": task["status"], "due": round(due),
            "done_t": task.get("done_t"), "thoughts": len(thoughts), "claims": [len(claims), right],
            "curve": dict(sorted(curve.items(), key=lambda kv: -int(kv[0][:-1]))), "parts": parts,
            "reward": round(sum(parts.values()), 3)}
