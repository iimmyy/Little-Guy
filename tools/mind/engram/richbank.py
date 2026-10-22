"""Rich made-up memories for the fading test: each one packs the details a human memory holds, so we can see which
details his bricks lose first. Nothing in it is real.

Code builds every memory from slots, so every answer is known: who, what happened (the gist), where, when, a number,
something someone said word for word, a side detail (a color), how I felt, and a second person who did one thing (to
check details stay tied to the right person). Azure's gpt-4.1 (the workspace's "judge" deployment) only writes each
one the way a person texts a friend; every rewrite is checked (each exact detail still in it, the quote unchanged, no
new names, numbers or days) and redone up to 4 times, else the plain sentence is kept.

Nine questions per memory, one per detail. In people, the gist and the feeling outlast the exact wording, numbers
and side details (Sachs 1967: meaning stays, wording goes; emotional memories keep the center, lose the edges).

  python3 richbank.py --out DIR [--n 64] [--seed 2026] [--plain]    # --plain: no Azure, plain sentences only
Standard library only; Azure through `az account get-access-token` (no keys).
"""
import argparse, json, random, re, subprocess, sys, time, urllib.error, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tests" / "behavior"))
from common import _settings                                                    # noqa: E402

from moments import SMALL_TALK                                                  # noqa: E402

URL = _settings().get("LITTLEGUY_JUDGE_URL") or (
    f"https://lg-ai-{_settings().get('LITTLEGUY_AZ_SUFFIX', '')}.cognitiveservices.azure.com"
    "/openai/deployments/judge/chat/completions?api-version=2024-10-21")
FEMALE = ["Priya", "Elena", "Amara", "Ingrid", "Nadia", "Rosa", "Fatima", "Lucia", "Hana", "Zara", "Freya", "Mei",
          "Leila", "Sofia", "Ines", "Noor", "Talia", "Yara", "Imani", "Greta", "Maeve", "Esme", "Annika", "Clara"]
MALE = ["Marcus", "Tobias", "Declan", "Yusuf", "Mateo", "Kenji", "Oliver", "Dmitri", "Gabriel", "Theo", "Malik",
        "Rafael", "Jonah", "Callum", "Andre", "Kwame", "Silas", "Emil", "Desmond", "Felix", "Arjun", "Omar", "Hugo",
        "Idris"]
RELS = {"sister": "f", "mom": "f", "aunt": "f", "brother": "m", "dad": "m", "uncle": "m", "coworker": None,
        "best friend": None, "neighbor": None, "cousin": None, "roommate": None, "boss": None}
PLACES = ["Harbor Bakery", "Northside Clinic", "Copperline Studio", "Maple Street Garage", "Bluebird Cafe",
          "Riverside Library", "Summit Gym", "Lantern Books", "Oakridge Mall", "Fernwood Park", "Juniper Hall",
          "Pine Hollow Diner", "Rusty Anchor Pub", "Golden Lotus"]
WORK = ["Harbor Bakery", "Northside Clinic", "Copperline Studio", "Maple Street Garage", "Riverside Library",
        "Lantern Books"]
VENUES = ["Bluebird Cafe", "Pine Hollow Diner", "Rusty Anchor Pub", "Golden Lotus", "Juniper Hall"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
PARTS = ["morning", "afternoon", "night"]
COLORS = ["brown", "black", "white", "green", "blue", "red", "yellow", "gray", "orange", "purple"]
BAD_Q = [("I'm never doing that again", ["never", "again"]), ("the universe owes me one", ["universe", "owes"]),
         ("we'll laugh about this someday", ["laugh", "someday"]), ("don't tell grandma", ["tell", "grandma"]),
         ("I should have stayed in bed", ["stayed", "bed"]), ("my horoscope warned me", ["horoscope"]),
         ("I need a nap and a hug", ["nap", "hug"]), ("this is so on brand for me", ["brand"]),
         ("who even does that", ["who even"]), ("can this week just end already", ["week", "end"])]
GOOD_Q = [("somebody pinch me", ["pinch"]), ("call the newspapers", ["newspaper"]), ("this calls for pizza", ["pizza"]),
          ("that's going in the group chat", ["group chat"]), ("remind me to buy a lottery ticket", ["lottery"]),
          ("I'm not crying, you're crying", ["crying"]), ("best night of my whole year", ["best", "year"]),
          ("pretty sure I peaked tonight", ["peaked"])]
BAD = ["annoyed", "nervous", "guilty", "embarrassed", "overwhelmed", "relieved", "worried", "frustrated"]
GOOD = ["proud", "grateful", "jealous", "amused", "overwhelmed", "relieved", "nostalgic", "thrilled"]
STEMS = dict(annoyed=["annoy"], nervous=["nervous"], guilty=["guilt"], embarrassed=["embarrass"],
             overwhelmed=["overwhelm"], relieved=["relie"], worried=["worr"], frustrated=["frustrat"],
             proud=["proud", "pride"], grateful=["grateful", "gratitude", "thankful"], jealous=["jealous"],
             amused=["amus"], nostalgic=["nostalg"], thrilled=["thrill"])   # graded as word starts

# what happened; {o}/{p}/{P} = him/her, his/her, He/She of the first person; the second person does {act2}
EVENTS = {
    "towed": dict(
        quotes=BAD_Q, places=PLACES, parts=PARTS, colors=COLORS,
        plain="On {day} {part}, my {rel} {name}'s car got towed outside {place}. The fine was ${num}. {P} said "
              "\"{quote}\". My {rel2} {name2} drove {o} home. The tow truck was bright {color}. "
              "Honestly I felt {feeling}.",
        num=(110, 480), mood=BAD, gist=["tow"],
        q=dict(who="Whose car got towed?", gist="What happened to my {rel}'s car?",
               where="Where did my {rel}'s car get towed from?", when="What day did my {rel}'s car get towed?",
               number="How much was the towing fine?", quote="What did my {rel} say when the car got towed?",
               side="What color was the tow truck?",
               feeling="How did I feel about my {rel}'s car getting towed?",
               bind="Who drove my {rel} home after the car got towed?")),
    "flood": dict(
        quotes=BAD_Q, places=WORK, parts=PARTS, colors=COLORS,
        plain="A pipe burst at {place} on {day} {part}, where my {rel} {name} works, and the whole back room flooded. "
              "It took {num} hours to mop up. {P} said \"{quote}\". My {rel2} {name2} brought everyone dinner. "
              "The water ruined a {color} couch in the corner. I felt {feeling} for {o}.",
        num=(2, 11), mood=BAD, gist=["pipe", "flood"],
        q=dict(who="Whose workplace had the burst pipe?", gist="What happened at my {rel}'s work?",
               where="Where does my {rel} work, the place with the burst pipe?",
               when="What day did the pipe burst at my {rel}'s work?", number="How many hours did it take to mop up?",
               quote="What did my {rel} say about the burst pipe?",
               side="What color was the couch the water ruined?", feeling="How did I feel about the burst pipe?",
               bind="Who brought everyone dinner after the pipe burst?")),
    "engaged": dict(
        quotes=GOOD_Q, places=["Fernwood Park", "Juniper Hall", "Copperline Studio", "Bluebird Cafe", "Golden Lotus"], parts=["afternoon", "night"], colors=COLORS,
        plain="My {rel} {name} got engaged at {place} on {day} {part}! There were {num} people there. {P} said "
              "\"{quote}\". My {rel2} {name2} took all the photos. The ring came in a little {color} box. "
              "I felt {feeling}.",
        num=(12, 90), mood=GOOD, gist=["engaged", "proposal", "proposed"],
        q=dict(who="Who got engaged?", gist="What big news happened to my {rel}?",
               where="Where did my {rel} get engaged?", when="What day did my {rel} get engaged?",
               number="How many people were at the engagement?", quote="What did my {rel} say after getting engaged?",
               side="What color was the ring box?", feeling="How did I feel about my {rel}'s engagement?",
               bind="Who took the photos at the engagement?")),
    "dog": dict(
        quotes=BAD_Q, places=["Fernwood Park", "Oakridge Mall", "Riverside Library", "Summit Gym", "Maple Street Garage"], parts=PARTS, colors=COLORS,
        plain="My {rel} {name}'s dog ran off near {place} on {day} {part}. We searched for {num} minutes. {P} kept "
              "saying \"{quote}\". In the end my {rel2} {name2} found him behind a dumpster. The dog had a {color} "
              "collar on. I felt {feeling}.",
        num=(20, 95), mood=BAD, gist=["ran off", "ran away", "lost", "missing", "got away", "escaped"],
        q=dict(who="Whose dog ran off?", gist="What happened with my {rel}'s dog?",
               where="Where did my {rel}'s dog run off?", when="What day did my {rel}'s dog run off?",
               number="How many minutes did we search for the dog?", quote="What did my {rel} keep saying while we searched?",
               side="What color was the dog's collar?", feeling="How did I feel when the dog ran off?",
               bind="Who found my {rel}'s dog?")),
    "wrist": dict(
        quotes=BAD_Q, places=PLACES, parts=PARTS, colors=["white", "blue", "red", "green", "purple", "black", "orange", "yellow"],
        plain="My {rel} {name} slipped on the ice outside {place} on {day} {part} and broke {p} wrist. {P}'ll be in a "
              "cast for {num} weeks. {P} said \"{quote}\". My {rel2} {name2} drove {o} to the hospital. The cast is "
              "{color}. I felt {feeling}.",
        num=(3, 9), mood=BAD, gist=["wrist", "broke", "slipped"],
        q=dict(who="Who broke their wrist?", gist="What happened to my {rel} on the ice?",
               where="Where did my {rel} slip on the ice?", when="What day did my {rel} break their wrist?",
               number="How many weeks will my {rel} be in a cast?", quote="What did my {rel} say after breaking their wrist?",
               side="What color is my {rel}'s cast?", feeling="How did I feel about my {rel}'s wrist?",
               bind="Who drove my {rel} to the hospital?")),
    "trivia": dict(
        quotes=GOOD_Q, places=VENUES, parts=["night"], colors=["gold", "silver", "green", "blue", "red", "purple", "orange", "black"],
        plain="My {rel} {name} won the trivia night at {place} on {day} {part}, and the prize was ${num}. {P} said "
              "\"{quote}\". My {rel2} {name2} got the last question right, something about volcanoes. The trophy is "
              "a tiny {color} owl. I felt {feeling}.",
        num=(50, 500), mood=GOOD, gist=["trivia", "won", "quiz"],
        q=dict(who="Who won the trivia night?", gist="What did my {rel} win?",
               where="Where was the trivia night my {rel} won?", when="What day did my {rel} win the trivia night?",
               number="How much was the trivia prize?", quote="What did my {rel} say after winning the trivia night?",
               side="What color is the trivia trophy?", feeling="How did I feel about my {rel} winning the trivia?",
               bind="Who got the last trivia question right?")),
    "argument": dict(
        quotes=BAD_Q, places=VENUES, parts=["afternoon", "night"], colors=COLORS,
        plain="My {rel} {name} got into an argument with a waiter at {place} on {day} {part} over a ${num} bill. "
              "{P} said \"{quote}\". My {rel2} {name2} paid just to end it. The waiter had a {color} apron on. "
              "I felt {feeling}.",
        num=(60, 300), mood=BAD, gist=["argument", "argued", "fight", "argue"],
        q=dict(who="Who got into an argument with a waiter?", gist="What happened with my {rel} and the waiter?",
               where="Where did my {rel} argue with the waiter?", when="What day did my {rel} argue with the waiter?",
               number="How much was the bill they argued over?", quote="What did my {rel} say to the waiter?",
               side="What color was the waiter's apron?", feeling="How did I feel about the argument with the waiter?",
               bind="Who paid the bill to end the argument?")),
    "party": dict(
        quotes=GOOD_Q, places=VENUES + ["Copperline Studio"], parts=["afternoon", "night"], colors=["blue", "red", "yellow", "green", "purple", "orange", "white", "pink"],
        plain="We threw my {rel} {name} a surprise party at {place} on {day} {part} for {p} {nth} birthday. "
              "{P} said \"{quote}\". My {rel2} {name2} baked the cake. The cake was bright {color}. I felt {feeling}.",
        num=(21, 65), mood=GOOD, gist=["surprise", "party", "birthday"],
        q=dict(who="Who did we throw a surprise party for?", gist="What did we do for my {rel}?",
               where="Where was my {rel}'s surprise party?", when="What day was my {rel}'s surprise party?",
               number="Which birthday was my {rel} turning at the surprise party?",
               quote="What did my {rel} say at the surprise party?", side="What color was the birthday cake?",
               feeling="How did I feel at my {rel}'s surprise party?", bind="Who baked the cake for the party?")),
}
TYPES = ["who", "gist", "where", "when", "number", "quote", "side", "feeling", "bind"]
PRONOUN = {"f": dict(o="her", p="her", P="She"), "m": dict(o="him", p="his", P="He")}
ASK = " Answer in a few words."


def ordinal(n):
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def person(rng, used_rels, used_names):
    rel = rng.choice([r for r in RELS if r not in used_rels])
    g = RELS[rel] or rng.choice("fm")
    name = rng.choice([n for n in (FEMALE if g == "f" else MALE) if n not in used_names])
    return rel, g, name


def build(n, seed):
    rng = random.Random(seed)
    kinds = list(EVENTS)
    out = []
    for i in range(n):
        kind = kinds[i % len(kinds)]
        ev = EVENTS[kind]
        rel, g, name = person(rng, set(), set())
        rel2, _, name2 = person(rng, {rel}, {name})
        quote, qkeys = rng.choice(ev["quotes"])
        num = rng.randint(*ev["num"])
        s = dict(rel=rel, name=name, rel2=rel2, name2=name2, place=rng.choice(ev["places"]), day=rng.choice(DAYS),
                 part=rng.choice(ev["parts"]), num=str(num), nth=ordinal(num), quote=quote,
                 color=rng.choice(ev["colors"]), feeling=rng.choice(ev["mood"]), **PRONOUN[g])
        text = ev["plain"].format(**s)
        text = re.sub(r"\ba (?=[aeiou])", "an ", text[0].upper() + text[1:])
        keys = dict(who=[name], gist=ev["gist"], where=[s["place"].split()[0]], when=[s["day"]], number=[s["num"]],
                    quote=qkeys, side=[s["color"]], feeling=STEMS[s["feeling"]], bind=[name2])
        qs = [dict(type=t, q=ev["q"][t].format(**s) + ASK, keys=keys[t],
                   other=[name2] if t == "who" else [name] if t == "bind" else []) for t in TYPES]
        talk = rng.sample(range(len(SMALL_TALK)), 4)
        out.append(dict(i=i, kind=kind, slots=s, plain=text, text=text, phrased_by="plain", questions=qs,
                        ctx_a=talk[:2], ctx_b=talk[2:]))
    return out


ALLOWED_CAPS = {"I", "I'm", "I'll", "I've", "I'd", "OMG", "LOL", "Honestly", "So", "And", "But", "Then", "Like"}


def check(text, m):
    """the rewrite keeps every exact detail and adds no names, numbers or days"""
    s = m["slots"]
    must = [f"my {s['rel']}", s["name"], f"my {s['rel2']}", s["name2"], s["place"], s["day"], s["num"],
            f"\"{s['quote']}\"", s["color"], s["feeling"]]
    missing = [x for x in must if x.lower() not in text.lower()]
    if missing:
        return f"missing {missing}"
    if set(re.findall(r"\d+", text)) - {s["num"]}:
        return "new number"
    if any(d in text for d in DAYS if d != s["day"]):
        return "new day"
    known = set(" ".join([s["name"], s["name2"], s["place"], s["day"]]).split())
    for sent in re.split(r"(?<=[.!?])\s+|\n+|\"", text):
        for w in re.findall(r"[A-Za-z']+", sent)[1:]:
            base = w[:-2] if w.endswith("'s") else w
            if w[0].isupper() and base not in known and base not in ALLOWED_CAPS:
                return f"new name {w}"
    return None


def token():
    return subprocess.run(["az", "account", "get-access-token", "--resource", "https://cognitiveservices.azure.com",
                           "--query", "accessToken", "-o", "tsv"], capture_output=True, text=True, check=True).stdout.strip()


PROMPT = """Rewrite this as a casual text message someone sends a close friend about their week. 3 to 5 sentences,
the way a real person texts (relaxed, a little messy, maybe a small aside), first person.
Keep EVERY one of these exactly as written: {must}. Keep the quote word for word inside double quotes.
Keep who did what exactly as in the facts (never move a detail or an action to a different person).
Do not add any other names, places, days, times or numbers. Do not leave anything out.
Reply with JSON only: {{"text": "..."}}

Facts:
{plain}"""


def phrase(m, auth, tries=4):
    s = m["slots"]
    must = ", ".join(repr(x) for x in [f"my {s['rel']} {s['name']}", f"my {s['rel2']} {s['name2']}", s["place"],
                                       s["day"], s["num"], s["quote"], s["color"], s["feeling"]])
    body = {"messages": [{"role": "user", "content": PROMPT.format(must=must, plain=m["plain"])}],
            "temperature": 0.9, "response_format": {"type": "json_object"}}
    why = []
    for _ in range(tries):
        if time.time() - auth["at"] > 2700:
            auth.update(v=token(), at=time.time())
        req = urllib.request.Request(URL, json.dumps(body).encode(),
                                     {"Content-Type": "application/json", "Authorization": f"Bearer {auth['v']}"})
        try:
            text = json.loads(json.load(urllib.request.urlopen(req, timeout=120))["choices"][0]["message"]["content"])
            text = text["text"].strip().replace("\u201c", '"').replace("\u201d", '"').replace("\u2019", "'")
        except urllib.error.HTTPError as e:
            why.append(f"http {e.code}")
            time.sleep(10 if e.code == 429 else 2)
            continue
        except (ValueError, KeyError, TimeoutError, urllib.error.URLError) as e:
            why.append(type(e).__name__)
            continue
        bad = check(text, m)
        if bad is None:
            return text, why
        why.append(bad)
    return None, why


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--plain", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ms = build(a.n, a.seed)
    log = open(out / "phrase.log", "a")
    if not a.plain:
        auth = dict(v=token(), at=time.time())
        for m in ms:
            text, why = phrase(m, auth)
            if text:
                m.update(text=text, phrased_by="gpt-4.1")
            log.write(json.dumps(dict(i=m["i"], ok=bool(text), why=why)) + "\n")
            log.flush()
            print(f"{m['i'] + 1}/{len(ms)} {'rewritten' if text else 'kept plain'}"
                  f"{' (' + '; '.join(why) + ')' if why else ''}", flush=True)
    json.dump(ms, open(out / "memories.json", "w"), indent=1)
    print(f"{sum(m['phrased_by'] != 'plain' for m in ms)} of {len(ms)} rewritten -> {out / 'memories.json'}")


if __name__ == "__main__":
    main()
