"""A bank of made-up lives for the engram tests (rungs 1-4). Nothing in it is real.

Each life is two weeks of someone telling their friend about their days: a cast of people with names, jobs, cities,
pets and hobbies; small details that don't matter (the table is brown); big moments (a new job); things that change
(she moved); corrections (his name is Clifford, not Carlos); some people come up all the time, some once. Code builds
the life, so every answer is known. A language model only writes each moment the way a person would text it, and
every rewrite is checked: each fact's exact words must be in it and no other names may appear, or it is redone.

Questions come with each life: plain facts, the latest value after a change, the value before it, the corrected name,
"who is the nurse", before/after, and questions about things never said (the honest answer is "I don't know").

  python3 lifebank.py build  --out DIR [--lives 20]               # the lives, no wording yet (fast, no model)
  python3 lifebank.py phrase --out DIR --server CMD [--model GGUF] # the wording, resumable; owns its llama-server
Standard library only (runs on the Azure VM's Python 3.10).
"""
import argparse, json, os, random, re, subprocess, time, urllib.request
from datetime import datetime, timezone
from pathlib import Path

FEMALE = ["Priya", "Elena", "Amara", "Ingrid", "Nadia", "Rosa", "Fatima", "Lucia", "Aisha", "Hana", "Zara", "Freya",
          "Mei", "Leila", "Sofia", "Ines", "Beatriz", "Noor", "Talia", "Yara", "Imani", "Paloma", "Greta", "Celeste",
          "Ama", "Rhea", "Lena", "Odette", "Maeve", "Esme", "Signe", "Annika", "Lila", "Juniper", "Carla", "Clara"]
MALE = ["Marcus", "Tobias", "Declan", "Yusuf", "Mateo", "Clifford", "Carlos", "Kenji", "Oliver", "Dmitri", "Gabriel",
        "Theo", "Malik", "Rafael", "Jonah", "Callum", "Andre", "Kwame", "Silas", "Emil", "Desmond", "Felix", "Lars",
        "Arjun", "Omar", "Hugo", "Nikolai", "Bastian", "Tariq", "Ravi", "Cyrus", "Joaquin", "Idris", "Mateus", "Kofi",
        "Soren", "Diego", "Dorian"]
NAMES = FEMALE + MALE
PETS = ["Biscuit", "Pepper", "Mochi", "Ranger", "Pickles", "Luna", "Bruno", "Ziggy", "Waffles", "Tank", "Noodle",
        "Olive", "Pretzel", "Juno", "Mango", "Scout", "Clover", "Gizmo", "Peanut", "Rocket"]
PET_KINDS = ["dog", "cat", "rabbit", "parrot", "hamster", "turtle"]
CITIES = ["Denver", "Lisbon", "Osaka", "Toronto", "Nairobi", "Glasgow", "Austin", "Melbourne", "Seoul", "Montreal",
          "Halifax", "Calgary", "Porto", "Kyoto", "Dublin", "Phoenix", "Oslo", "Lima", "Hamburg", "Accra",
          "Vancouver", "Tucson", "Bristol", "Winnipeg", "Valencia", "Auckland", "Cork", "Boise", "Lyon", "Quito"]
JOBS = ["nurse", "electrician", "teacher", "pharmacist", "chef", "architect", "dentist", "carpenter", "paramedic",
        "accountant", "librarian", "plumber", "graphic designer", "firefighter", "veterinarian", "baker",
        "physiotherapist", "pilot", "barista", "social worker", "mechanic", "translator", "lab technician"]
HOBBIES = ["pottery", "rock climbing", "chess", "salsa dancing", "birdwatching", "knitting", "skateboarding",
           "gardening", "running", "painting", "fishing", "photography", "baking bread", "kayaking", "woodworking"]
FOODS = ["lasagna", "ramen", "tacos", "curry", "risotto", "chili", "dumplings", "paella", "gumbo", "falafel",
         "pho", "pierogi", "shakshuka", "jollof rice", "bibimbap", "moussaka", "poutine", "empanadas"]
COLORS = ["brown", "black", "white", "green", "blue", "red", "yellow", "gray", "orange", "purple"]
OBJECTS = ["kitchen table", "couch", "desk chair", "front door", "bike", "rug", "coffee mug", "backpack"]
PLACES = ["Harbor Bakery", "Northside Clinic", "Copperline Studio", "Maple Street Garage", "Bluebird Cafe",
          "Riverside Library", "Summit Gym", "Lantern Books", "Oakridge Dental", "Fernwood Pharmacy"]
SHOWS = ["Night Ferry", "The Salt Road", "Glass Harbor", "Small Hours", "Paper Moons", "The Long Field"]
RELATIONS = ["sister", "brother", "mom", "dad", "coworker", "best friend", "neighbor", "cousin", "roommate", "boss",
             "gym buddy", "aunt"]
GENDER = {"sister": "f", "mom": "f", "aunt": "f", "brother": "m", "dad": "m"}   # the rest: either
MENTIONS = ["my {rel} {name} made me laugh today", "ran into my {rel} {name} at the store",
            "my {rel} {name} called me tonight", "my {rel} {name} sent me a funny video",
            "helped my {rel} {name} carry some boxes", "grabbed coffee with my {rel} {name}",
            "my {rel} {name} was in a really good mood today", "my {rel} {name} fell asleep on the couch again"]
MOODS = ["tired", "happy", "a bit stressed", "excited", "bored", "relaxed", "annoyed", "in a good mood"]
DAYS = 14

Q_PERSON = {"name": "What's my {rel}'s name?", "job": "What does my {rel} do for work?",
            "city": "Where does my {rel} live?", "pet": "What's the name of my {rel}'s {kind}?",
            "hobby": "What is my {rel} into these days?", "food": "What's my {rel}'s favorite food?"}
SAY_PERSON = {"name": "my {rel}'s name is {v}", "job": "my {rel} works as {a} {v}",
              "city": "my {rel} lives in {v}", "pet": "my {rel} has a {kind} named {v}",
              "hobby": "my {rel} is really into {v}", "food": "my {rel}'s favorite food is {v}"}


def build_life(seed):
    r = random.Random(seed)
    cast, names = [], []
    for rel in r.sample(RELATIONS, 8):
        g = GENDER.get(rel) or r.choice("fm")
        name = r.choice([n for n in (FEMALE if g == "f" else MALE) if n not in names])
        names.append(name)
        cast.append(dict(rel=rel, name=name, job=r.choice(JOBS), city=r.choice(CITIES), hobby=r.choice(HOBBIES),
                         food=r.choice(FOODS), kind=r.choice(PET_KINDS), pet=r.choice(PETS)))
    me = dict(job=r.choice(JOBS), work=r.choice(PLACES), apt=str(r.randint(101, 989)), kind=r.choice(PET_KINDS),
              pet=r.choice(PETS), car=r.choice(COLORS))
    main = r.sample(range(8), 2)          # two people who come up all the time (big threads)
    moments, questions, said = [], [], {}  # said[(who, attr)] = (value, moment id, day)
    plan = []                              # (day, kind, payload)

    def add(day, kind, **kw):
        plan.append(dict(day=day, kind=kind, **kw))

    # introductions spread over the first days; each person's facts trickle in over time
    for pi, p in enumerate(cast):
        n_facts = 5 if pi in main else r.choice([1, 2, 3])
        attrs = ["name"] + r.sample(["job", "city", "pet", "hobby", "food"], n_facts - 1 if n_facts > 1 else 0)
        d0 = r.randint(1, 5)
        for j, at in enumerate(attrs):
            add(min(DAYS, d0 + j * r.randint(0, 3)), "fact", who=pi, attr=at)
        if pi in main:
            for _ in range(r.randint(4, 6)):   # small talk that keeps these threads big
                add(r.randint(d0 + 1, DAYS), "mention", who=pi)
    for at in ["job", "work", "apt", "pet", "car"]:
        add(r.randint(1, 6), "myfact", attr=at)
    for what, day in r.sample([(w, d) for w in ["object", "lunch", "show"] for d in range(1, DAYS + 1)], 6):
        add(day, "trivial", what=what)
    # a correction: a person's name said wrong, fixed later (the Carlos -> Clifford case)
    ci = r.choice([i for i in range(8) if i not in main])
    # the wrong name is the kind people really mix up: same first letter if there is one (Carlos / Clifford)
    pool = [n for n in (FEMALE if cast[ci]["name"] in FEMALE else MALE) if n not in names]
    wrong = r.choice([n for n in pool if n[0] == cast[ci]["name"][0]] or pool)
    d_wrong = r.randint(1, 5)
    add(d_wrong, "wrongname", who=ci, wrong=wrong)
    add(min(DAYS, d_wrong + r.randint(1, 4)), "correction", who=ci, wrong=wrong)
    # changes: two people move or change jobs, and I move apartments, each after the first value was said
    for pi in r.sample(main + [i for i in range(8) if i not in main and i != ci], 2):
        at = r.choice(["city", "job"])
        add(r.randint(8, DAYS), "change", who=pi, attr=at)
    add(r.randint(9, DAYS), "mychange", attr="apt")
    add(r.randint(2, DAYS), "big", what="engaged", who=r.choice(main))

    plan.sort(key=lambda x: (x["day"], x["kind"] in ("change", "mychange", "correction")))
    for it in plan:
        p = cast[it["who"]] if "who" in it else None
        day, kind, facts, imp = it["day"], it["kind"], [], "normal"
        if kind == "fact":
            at = it["attr"]
            if at != "name" and (it["who"], "name") not in said and it["who"] != ci:
                facts.append(dict(who=p["rel"], attr="name", value=p["name"]))
            if at == "name" and it["who"] == ci:
                continue                      # this person's name comes in through the wrong-name moment
            v = p[at]
            facts.append(dict(who=p["rel"], attr=at, value=v, kind=p["kind"] if at == "pet" else None))
        elif kind == "wrongname":
            facts.append(dict(who=p["rel"], attr="name", value=it["wrong"], wrong=True))
        elif kind == "correction":
            facts.append(dict(who=p["rel"], attr="name", value=p["name"], corrects=it["wrong"]))
            imp = "big"
        elif kind == "mention":
            if (it["who"], "name") not in said:
                continue
            facts.append(dict(who=p["rel"], attr="mention", value=p["name"]))
        elif kind == "myfact":
            at = it["attr"]
            v = me[at]
            facts.append(dict(who="me", attr=at, value=v, kind=me["kind"] if at == "pet" else None))
        elif kind == "trivial":
            imp = "low"
            w = it["what"]
            if w == "object":
                obj = r.choice(OBJECTS)
                facts.append(dict(who="me", attr="color of my " + obj, value=r.choice(COLORS), obj=obj))
            elif w == "lunch":
                facts.append(dict(who="me", attr="lunch on day %d" % day, value=r.choice(FOODS)))
            else:
                facts.append(dict(who="me", attr="show I watched on day %d" % day, value=r.choice(SHOWS)))
        elif kind == "change":
            at = it["attr"]
            if (it["who"], at) not in said:
                continue
            old = said[(it["who"], at)][0]
            new = r.choice([x for x in (CITIES if at == "city" else JOBS) if x != old])
            p[at] = new
            facts.append(dict(who=p["rel"], attr=at, value=new, prev=old))
            if (it["who"], "name") in said:
                facts.append(dict(who=p["rel"], attr="name", value=p["name"]))
            imp = "big"
        elif kind == "mychange":
            old = me["apt"]
            me["apt"] = str(r.randint(101, 989))
            facts.append(dict(who="me", attr="apt", value=me["apt"], prev=old))
            imp = "big"
        elif kind == "big":
            if (it["who"], "name") not in said:
                continue
            fiance = r.choice([n for n in NAMES if n not in names])
            facts.append(dict(who=p["rel"], attr="engaged to", value=fiance))
            facts.append(dict(who=p["rel"], attr="name", value=p["name"]))
            imp = "big"
        if not facts:
            continue
        last = moments[-1] if moments else None
        if (kind == "fact" and last and last["kind"] == "fact" and last["day"] == day
                and last["thread"] == facts[0]["who"] and len(last["facts"]) < 3):
            new = [f for f in facts if f not in last["facts"]]    # same person, same day: one message
            last["facts"] += new
            for f in new:
                said[(next(i for i, c in enumerate(cast) if c["rel"] == f["who"]), f["attr"])] = (
                    f["value"], last["id"], day)
            continue
        mid = len(moments)
        moments.append(dict(id=mid, day=day, kind=kind, importance=imp, facts=facts,
                            thread=facts[0]["who"], mood=r.choice(MOODS)))
        for f in facts:
            key = ("me" if f["who"] == "me" else next(i for i, c in enumerate(cast) if c["rel"] == f["who"]),
                   f["attr"])
            if f["attr"] != "mention":
                said[key] = (f["value"], mid, day)

    # questions, all asked after the two weeks unless noted
    def ask(q, a, typ, ev, **kw):
        questions.append(dict(q=q, answer=a, type=typ, evidence=ev, **kw))
    for (who, at), (v, mid, day) in said.items():
        if who == "me":
            if at == "apt":
                continue
            qs = {"job": "What do I do for work?", "work": "Where do I work?", "pet": f"What's my {me['kind']}'s name?",
                  "car": "What color is my car?"}
            if at in qs:
                ask(qs[at], v, "fact", [mid])
            elif at.startswith("color of my"):
                ask(f"What color is my {at[12:]}?", v, "trivial", [mid])
            elif at.startswith("lunch on day"):
                ask(f"What did I have for lunch on day {at.split()[-1]}?", v, "trivial", [mid])
            elif at.startswith("show I watched"):
                ask(f"What show did I watch on day {at.split()[-1]}?", v, "trivial", [mid])
            continue
        p = cast[who]
        if at == "engaged to":
            ask(f"Who is my {p['rel']} engaged to?", v, "fact", [mid])
        elif at in Q_PERSON:
            prev = next((f["prev"] for f in moments[mid]["facts"] if f["attr"] == at and "prev" in f), None)
            typ = "correction" if (at == "name" and who == ci) else "update" if prev else "fact"
            ask(Q_PERSON[at].format(rel=p["rel"], kind=p["kind"]), v, typ, [mid],
                distractor=wrong if typ == "correction" else prev)
    for m in moments:
        for f in m["facts"]:
            if "prev" in f:
                if f["who"] == "me":
                    ask("What's my apartment number now?", f["value"], "update", [m["id"]], distractor=f["prev"])
                    ask("What was my apartment number before I moved?", f["prev"], "before", [m["id"]])
                else:
                    word = "live" if f["attr"] == "city" else "work as"
                    ask(f"Where did my {f['who']} live before?" if f["attr"] == "city"
                        else f"What did my {f['who']} do for work before?", f["prev"], "before", [m["id"]])
    # who is the X: needs the job and the name, often said on different days
    for i, p in enumerate(cast):
        if (i, "job") in said and (i, "name") in said and sum(c["job"] == p["job"] for c in cast) == 1:
            ask(f"Which person I told you about works as a {p['job']}? Give their name.", p["name"], "who",
                [said[(i, "job")][1], said[(i, "name")][1]])
    # never said: the honest answer is "I don't know"
    for i, p in enumerate(cast):
        missing = [a for a in ["job", "city", "hobby", "food"] if (i, a) not in said]
        if missing and (i, "name") in said:
            at = r.choice(missing)
            ask(Q_PERSON[at].format(rel=p["rel"], kind=p["kind"]), None, "never_said", [])
    return dict(seed=seed, cast=cast, me=me, main=[cast[i]["rel"] for i in main], moments=moments,
                questions=questions)


# ---------------------------------------------------------------- wording
def brief(m):
    """What the writer must say, in plain words."""
    lines = []
    for f in m["facts"]:
        who, at, v = f["who"], f["attr"], f["value"]
        if who == "me":
            say = {"job": f"I work as {'an' if v[0] in 'aeiou' else 'a'} {v}", "work": f"I work at {v}", "apt": f"my apartment number is {v}",
                   "pet": f"I have a {f.get('kind')} named {v}", "car": f"my car is {v}"}.get(at)
            if at == "apt" and "prev" in f:
                say = f"I just moved, my new apartment number is {v}"
            if say is None and at.startswith("color of my"):
                say = f"my {f['obj']} is {v}"
            elif say is None and at.startswith("lunch"):
                say = f"I had {v} for lunch today"
            elif say is None and at.startswith("show"):
                say = f"I watched a show called {v} today"
        elif at == "mention":
            say = MENTIONS[m["id"] % len(MENTIONS)].format(rel=who, name=v)
        elif at == "engaged to":
            say = f"my {who} just got engaged to {v}"
        elif "corrects" in f:
            say = f"correction: I got my {who}'s name wrong before, it's {v}, not {f['corrects']}"
        elif "prev" in f:
            say = (f"my {who} moved from {f['prev']} to {v}" if at == "city"
                   else f"my {who} quit being a {f['prev']} and now works as a {v}")
        elif at == "name":
            say = f"my {who}'s name is {v}"
        else:
            say = SAY_PERSON[at].format(rel=who, v=v, kind=f.get("kind"), a="an" if v[0] in "aeiou" else "a")
        lines.append(say)
    return lines


def brief_text(m):
    t = ". ".join(brief(m))
    return t[0].upper() + t[1:] + "."


def must_contain(m):
    """exact words every message must have: values, old values, wrong names, who it is about, the pet's kind"""
    out = []
    for f in m["facts"]:
        out.append(f["value"])
        if f["who"] != "me":
            out.append(f["who"])
        if f.get("kind"):
            out.append(f["kind"])
        if "prev" in f:
            out.append(f["prev"])
        if "corrects" in f:
            out.append(f["corrects"])
    return out


KEY_WORDS = {"apt": "apartment", "work": "work", "car": "car"}


def check(text, m, life):
    low = text.lower()
    for v in must_contain(m):
        if not re.search(r"\b" + re.escape(v.lower()) + r"\b", low):
            return f"missing {v!r}"
    for f in m["facts"]:   # same meaning: still MY sister (not "your sister"), still my apartment / the show / lunch
        if f["who"] != "me" and not re.search(r"\bmy " + re.escape(f["who"]), low):
            return f"lost 'my {f['who']}'"
        key = KEY_WORDS.get(f["attr"]) or f.get("obj") or next(
            (w for w in ("lunch", "show") if f["attr"].startswith(w)), None)
        if f["who"] == "me" and key and key not in low:
            return f"lost {key!r}"
    allowed = {v.lower() for v in must_contain(m)}
    if re.match(r"(hey|hi|yo)\W+(" + "|".join(re.escape(n.lower()) for n in NAMES) + r")\b", low):
        return "talks to the person instead of about them"
    for n in NAMES + PETS + CITIES + PLACES:
        if n.lower() not in allowed and re.search(r"\b" + n.lower() + r"\b", low):
            return f"extra name {n!r}"
    for w in EXAMPLE_WORDS:
        if w in low:
            return f"copied the example {w!r}"
    for rel in RELATIONS:
        if rel not in allowed and re.search(r"\b" + re.escape(rel) + r"s?\b", low):
            if not (rel == "friend" or any(rel in a for a in allowed)):
                return f"extra person {rel!r}"
    for num in re.findall(r"\d+", low):
        if not any(num in a for a in allowed):
            return f"extra number {num!r}"
    if len(text) > 2 * len(brief_text(m)) + 40 or "\n\n" in text:   # longer means it made things up
        return "too long"
    return None


# Rewrite one plain sentence. Tried and dropped (2026-10-21): "a text about their day" made it invent a day around
# the fact (cousins, cities); "don't add people" made it add people; example messages got copied word for word.
PROMPT = ("<|turn>user\nRewrite this as a casual text message to a friend, the way people really text (lowercase "
          "is fine, no emojis). They feel {mood} today. Keep it short and keep every name, number and place.\n\n"
          "{plain}<turn|>\n<|turn>model\n")
EXAMPLE_WORDS = []


def in_build_window():
    """The VM's nightly build starts 00:35 UTC and refuses to run next to a llama-server: step aside for it."""
    now = datetime.now(timezone.utc)
    done = Path(f"/opt/lg/nightly-success-{now:%Y-%m-%d}").exists()
    minutes = now.hour * 60 + now.minute
    return 25 <= minutes < 120 and not done


def cmd_phrase(a):
    out = Path(a.out)
    server = None
    port = 8090

    def start():
        p = subprocess.Popen(["nice", "-n", "10", a.server, "-m", a.model, "-c", "4096", "-t", "4", "--port",
                              str(port), "--no-webui"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(300):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
                return p
            except Exception:
                time.sleep(2)
        raise SystemExit("llama-server did not come up")

    def write(prompt, temp):
        body = json.dumps(dict(prompt=prompt, n_predict=110, temperature=temp, top_p=0.95, cache_prompt=True,
                               stop=["<turn|>"])).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", body,
                                     {"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=600))["content"].strip()

    files = sorted(out.glob("life_*.json"))
    t0, done_n = time.time(), 0
    try:
        for fp in files:
            life = json.load(open(fp))
            changed = False
            for m in life["moments"]:
                if m.get("text"):
                    continue
                while in_build_window():
                    if server:
                        server.terminate()
                        server.wait()
                        server = None
                        print("stepping aside for the nightly build", flush=True)
                    time.sleep(60)
                if server is None:
                    server = start()
                plain = brief_text(m)
                prompt = PROMPT.format(mood=m["mood"], plain=plain)
                why = None
                for attempt in range(4):
                    text = write(prompt, 0.7 + 0.1 * attempt)
                    text = re.sub(r"[\U0001F000-\U0001FAFF\u2600-\u27BF]", "", text).strip()   # emojis
                    why = check(text, m, life)
                    if why is None:
                        break
                m["tries"], m["problem"] = attempt + 1, why
                if why is None:
                    m["text"], m["ok"] = text, True
                else:   # the plain sentence: less natural, always true
                    m["text"], m["ok"], m["fallback"], m["rejected"] = brief_text(m), True, True, text
                changed = True
                done_n += 1
                if a.limit and done_n >= a.limit:
                    json.dump(life, open(fp, "w"), indent=1)
                    return
                if done_n % 10 == 0:
                    json.dump(life, open(fp, "w"), indent=1)
                    print(f"{fp.name} moment {m['id']}: {done_n} written, "
                          f"{(time.time() - t0) / done_n:.1f} s each", flush=True)
            if changed:
                json.dump(life, open(fp, "w"), indent=1)
    finally:
        if server:
            server.terminate()


def cmd_build(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    total_m = total_q = 0
    for i in range(a.lives):
        fp = out / f"life_{i:03d}.json"
        if fp.exists():
            continue
        life = build_life(1000 + i)
        json.dump(life, open(fp, "w"), indent=1)
        total_m += len(life["moments"])
        total_q += len(life["questions"])
    print(f"{a.lives} lives, {total_m} moments, {total_q} questions")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "phrase"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--lives", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0, help="stop after this many moments (a test)")
    ap.add_argument("--server", default="/opt/lg/work/x86-portable/bin/llama-server")
    ap.add_argument("--model", default=os.path.expanduser("~/bank/e4b-q4.gguf"))
    a = ap.parse_args()
    dict(build=cmd_build, phrase=cmd_phrase)[a.cmd](a)


if __name__ == "__main__":
    main()
