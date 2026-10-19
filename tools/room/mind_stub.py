#!/usr/bin/env python3
"""A stand-in for his mind loop, so the room can be built and tested before the real one moves in.

Every PERIOD seconds: read anything new in his inbox, write a note in his notebook, and drop a mood note (and now and
then a say note) in his outbox. It only touches the three places the walls allow; the real mind loop (life_sim.py's
pacemaker, on the board's brain) will do the same through the same three places.
"""
import json, os, time

INBOX = "/var/lib/littleguy/inbox"
NOTEBOOK = "/var/lib/littleguy/room/notebook"
OUTBOX = "/var/lib/littleguy/room/outbox"
PERIOD = float(os.environ.get("PERIOD", 60))


def post(note):
    name = f"{time.time():.6f}.json"
    tmp = os.path.join(OUTBOX, "." + name)       # write then rename, so the guard never reads half a note
    with open(tmp, "w") as f:
        json.dump(note, f)
    os.replace(tmp, os.path.join(OUTBOX, name))


seen, tick = set(), 0
while True:
    heard = []
    for name in sorted(os.listdir(INBOX)):
        if name.startswith(".") or name in seen:
            continue
        seen.add(name)
        try:
            heard.append(json.load(open(os.path.join(INBOX, name)))["text"])
        except (OSError, ValueError, KeyError):
            pass
    with open(os.path.join(NOTEBOOK, "notes.txt"), "a") as f:
        f.write(f"{time.strftime('%a %H:%M')} heard {heard!r}. The desk is quiet.\n")
    post({"kind": "mood", "valence": 0.1, "arousal": -0.1, "word": "settled"})
    if heard:
        post({"kind": "say", "text": "I heard you."})
    tick += 1
    time.sleep(PERIOD)
