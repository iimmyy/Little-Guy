"""Little Guy's memory v1 (Immy, 2026-10-19: "how does he know when to make a memory? ... dopamine and NE, so some
memories stick better"; and: "RL for him, but remembering memories rewards him, not RL the memories into him").

Nothing is ever deleted: every note and every chat stays in his notebook. What changes is how easily each comes back.
  tag        each moment gets a strength when it happens (encoding): ne = how keyed up / how strong the feeling is
             (his dials; norepinephrine), da = surprise (what he expected vs what happened; dopamine). Strong ink, faint ink.
  tagging    before sleep, moments within an hour of a strong one borrow some of its strength (synaptic tagging and
             capture: you remember where you were when the big news came)
  consolidate  going to sleep, he replays the day's strongest moments and writes what he wants to keep as a short
             memory (memories.jsonl; hippocampal replay into long-term). Faint days leave nothing new.
  recall     what comes to mind by itself (involuntary): strength x how recent (strong memories fade slower) x how well
             it fits his mood now x how much it shares with what's in front of him. What he looks up on purpose
             (voluntary): a REMEMBER: line, one per turn, searched by its words.
Every recall and every save decision is logged (recalls.jsonl, saves.jsonl) so his use of memory can be rewarded later
(RL on the memory policy: recall that turns out true and used earns, made-up memories don't, each recall costs a bit).
"""
import json, math, os, re, time

WORD = re.compile(r"[a-z']+")
STOP = set("""the and you your for are was were but not that this with have has had his her him she he they them their
there then than what when where who why how its it's i'm i've i'd just like into out about from been being some any all
can could would should will shall did does doing done get got gets very really still also only more most much many one
two now here say said says okay yeah yes hey well too our ours we're you're it’s don't didn't can't won't let's""".split())
# anywhere in a line, as he writes it mid-sentence too ("Hey Immy! REMEMBER: concert Thursday"); also his misspelling
# "Remeber:". Needs the colon, so "Remember, you're going..." is ordinary talk.
REMEMBER = re.compile(r"(?i)\**\bREME?M?BER\b\**\s*:\**\s*([^\n]*)")
CEIL, CEIL_AROUSAL = 0.75, 0.5          # his dials' soft ceilings (mind_live / life_sim)
TAG_HOURS, TAG_SHARE = 1.0, 0.3         # moments within an hour of a strong one borrow 30% of its strength
STRONG = 0.6                            # a strong moment, for tagging
KEEP = 0.45                             # consolidated at sleep if at least this strong (after tagging), at most KEEP_MAX
KEEP_MAX = 3
SHOW = 3                                # memories that come to mind at once


def words(text):
    return {w for w in WORD.findall(text.lower()) if len(w) > 2 and w not in STOP}


def tag(feel, surprise):
    """strength at encoding from his state right now: ne (keyed up, or a strong feeling either way) and da (surprise)"""
    ne = min(1.0, max(abs(feel[1]) / CEIL_AROUSAL, math.hypot(feel[0], feel[1]) / CEIL))
    da = min(1.0, max(0.0, surprise))
    return {"ne": round(ne, 3), "da": round(da, 3), "strength": round(min(1.0, 0.15 + 0.45 * ne + 0.4 * da), 3)}


def _jsonl(path):
    try:
        with open(path) as f:
            return [json.loads(l) for l in f if l.strip()]
    except OSError:
        return []


class Memory:
    def __init__(self, notebook):
        self.nb = notebook
        self.p = lambda name: os.path.join(notebook, name)

    def log(self, name, obj):
        with open(self.p(name), "a") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def items(self):
        """everything he could remember: his notes, what was said, and what he kept from past days"""
        out = []
        for n in _jsonl(self.p("notes.jsonl")):
            out.append({"id": n.get("id", f"n{n['t']}"), "t": n["t"], "when": n["when"], "kind": "note",
                        "text": n["thought"], "strength": n.get("strength", 0.3), "feel": n.get("thought_reads", [0, 0])})
        for c in _jsonl(self.p("chat.jsonl")):
            out.append({"id": c.get("id", f"c{c['t']}"), "t": c["t"], "when": c["when"], "kind": "talk",
                        "text": f'{c.get("person", "They")} said: "{c["heard"]}" You said: "{c["reply"]}"',
                        "strength": c.get("strength", 0.4), "feel": c.get("reply_reads", [0, 0])})
        for m in _jsonl(self.p("memories.jsonl")):
            out.append({"id": m["id"], "t": m["t"], "when": m["when"], "kind": "memory", "text": m["gist"],
                        "strength": m["strength"], "feel": m.get("feel", [0, 0])})
        return out

    def score(self, it, now, feel, query, overlap_weight):
        age_days = max(0.0, now - it["t"]) / 86400
        half_life = 2 + 28 * it["strength"]                         # faint: days; strong: weeks (a forgetting curve)
        recency = 0.5 ** (age_days / half_life)
        a, b = feel, it["feel"]
        na, nb = math.hypot(*a), math.hypot(*b)
        mood = (a[0] * b[0] + a[1] * b[1]) / (na * nb) if na > 0.05 and nb > 0.05 else 0.0   # mood-congruent recall
        q = words(query)
        overlap = len(q & words(it["text"] + " " + it["when"])) / max(1, len(q))   # "Sunday" finds Sunday's entries
        return it["strength"] * recency * (1 + 0.5 * mood) * (1 + overlap_weight * overlap), overlap

    def recall(self, now, feel, query, skip=(), kind="involuntary", during="", n=SHOW):
        """top memories for this moment; voluntary recall must share words with what he's looking for"""
        weight = 5.0 if kind == "voluntary" else 2.0
        scored = []
        for it in self.items():
            if it["id"] in skip or now - it["t"] < 60:
                continue
            s, ov = self.score(it, now, feel, query, weight)
            if kind == "voluntary" and ov == 0:
                continue
            if s > 0.05:
                scored.append((s, it))
        got = [it for s, it in sorted(scored, key=lambda x: -x[0])[:n]]
        rec = {"t": round(now), "kind": kind, "during": during, "query": query[:500], "feel": [round(x, 3) for x in feel],
               "got": [{"id": it["id"], "score": round(s, 4)} for s, it in sorted(scored, key=lambda x: -x[0])[:n]]}
        return got, rec

    def consolidate(self, now, since, gist_fn, read_fn, when_fn):
        """going to sleep: the day's moments, strength + tagging, the strongest kept as one short memory"""
        day = [it for it in self.items() if it["kind"] != "memory" and since <= it["t"] <= now]
        strong = [it for it in day if it["strength"] >= STRONG]
        cands = []
        for it in day:
            boost = max([TAG_SHARE * s["strength"] for s in strong
                         if s["id"] != it["id"] and abs(s["t"] - it["t"]) <= TAG_HOURS * 3600] or [0.0])
            cands.append({"id": it["id"], "strength": it["strength"], "eff": round(min(1.0, it["strength"] + boost), 3),
                          "it": it})
        keep = sorted([c for c in cands if c["eff"] >= KEEP], key=lambda c: -c["eff"])[:KEEP_MAX]
        mem = None
        if keep:
            moments = "\n".join(f"- [{c['it']['when']}] {c['it']['text']}" for c in sorted(keep, key=lambda c: c["it"]["t"]))
            gist = gist_fn(moments)
            if gist:
                mem = {"id": f"m{round(now)}", "t": round(now), "when": when_fn(now), "gist": gist,
                       "strength": max(c["eff"] for c in keep), "from": [c["id"] for c in keep], "feel": read_fn(gist)}
                self.log("memories.jsonl", mem)
        self.log("saves.jsonl", {"t": round(now), "since": round(since), "memory": mem and mem["id"],
                                 "candidates": [{k: c[k] for k in ("id", "strength", "eff")} | {"kept": c in keep}
                                                for c in cands]})
        return mem


def lines(got):
    return "\n".join(f"- [{it['when']}] {it['text']}" for it in got)


def split_remember(text):
    """(query or None, the text without any REMEMBER: lines)"""
    m = REMEMBER.search(text)
    q = re.split(r"\s*\[|(?<=[.!?])\s", m.group(1))[0].strip(" *\"'.")[:300] if m else None   # up to a [face] or a full stop
    rest = re.sub(r"[ \t]+\n", "\n", REMEMBER.sub("", text)).strip()
    return (q or None), re.sub(r"\n{3,}", "\n\n", rest)
