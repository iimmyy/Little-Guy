#!/usr/bin/env python3
"""The guard: the other side of Little Guy's mail slot (blueprint sandbox-plan-v0.md, Wall 1).

He never gets hands. Everything he "does" is a small note dropped in his outbox. The guard runs as its own user,
reads each note, and acts only on the allowlist below; anything else is dropped and written to the decision log.
  say   {"kind": "say", "text": "..."}                    one short line out loud (on AWS: written to said.log;
                                                            on the board: the speaker), at most one per SAY_REST s
  mood  {"kind": "mood", "valence": x, "arousal": y, "word": "..."}   the status page's mood (two numbers + a word)
It also writes into his inbox (what his senses and his person say): guard.py tell "<text>".
A note is read only if it is a plain file (never a link), has one name (no hard links), and is small. The guard
never runs anything a note says and never opens a path a note names.
usage: guard.py run | guard.py tell "<text>"
"""
import json, os, stat, sys, time, unicodedata

OUTBOX = "/var/lib/littleguy/room/outbox"
INBOX = "/var/lib/littleguy/inbox"
STATE = "/var/lib/littleguy-guard"
STATUS = "/run/littleguy/status.json"
MAX_BYTES = 4096                       # a note bigger than this is dropped unread
MAX_PER_ROUND = 50                     # a flood is read 50 at a time; the rest wait (his disk image caps the total)
MAX_SAY = 200                          # characters in one spoken line
SAY_REST = float(os.environ.get("SAY_REST", 7200))   # seconds between spoken lines (the urge's 2-hour rest)
WORD_MAX = 20


def log(decision, why, note=None):
    with open(os.path.join(STATE, "decisions.jsonl"), "a") as f:
        f.write(json.dumps({"t": round(time.time(), 1), "decision": decision, "why": why,
                            "note": (note[:300] if isinstance(note, str) else note)}) + "\n")


def clean(text, limit):
    """plain printable text only: no control characters or terminal escapes, at most `limit` characters"""
    if not isinstance(text, str) or not text.strip() or len(text) > limit:
        return None
    if any(unicodedata.category(c).startswith("C") for c in text):
        return None
    return text.strip()


def read_note(dirfd, name):
    """open without following links; must be a plain file with a single name and small"""
    try:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=dirfd)
    except OSError as e:
        return None, f"could not open safely ({e.strerror})"
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            return None, "not a plain file"
        if st.st_nlink != 1:
            return None, "file has more than one name (hard link)"
        if st.st_size > MAX_BYTES:
            return None, f"too big ({st.st_size} bytes)"
        return os.read(fd, MAX_BYTES + 1).decode("utf-8", "strict"), None
    except UnicodeDecodeError:
        return None, "not text"
    finally:
        os.close(fd)


def act(raw, last_say):
    try:
        note = json.loads(raw)
    except ValueError:
        return "drop", "not a valid note", last_say
    if not isinstance(note, dict):
        return "drop", "not a valid note", last_say
    kind = note.get("kind")
    if kind == "say":
        text = clean(note.get("text"), MAX_SAY)
        if text is None:
            return "drop", "say: text missing, too long, or has control characters", last_say
        if time.time() - last_say < SAY_REST:
            return "drop", "say: resting (spoke too recently)", last_say
        with open(os.path.join(STATE, "said.log"), "a") as f:   # on the board: the speaker
            f.write(f"{time.strftime('%H:%M:%S')} {text}\n")
        return "said", text, time.time()
    if kind == "mood":
        try:
            v, a = float(note["valence"]), float(note["arousal"])
        except (KeyError, TypeError, ValueError):
            return "drop", "mood: numbers missing", last_say
        word = clean(note.get("word", ""), WORD_MAX) or ""
        if not (-1 <= v <= 1 and -1 <= a <= 1) or (word and not word.replace(" ", "").isalpha()):
            return "drop", "mood: out of range or odd word", last_say
        tmp = STATUS + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"valence": round(v, 3), "arousal": round(a, 3), "word": word, "t": int(time.time())}, f)
        os.replace(tmp, STATUS)
        return "mood", f"{v:+.2f} {a:+.2f} {word}", last_say
    return "drop", f"not on the allowlist: {str(kind)[:40]!r}", last_say


def run():
    last_say = 0.0
    dirfd = os.open(OUTBOX, os.O_RDONLY | os.O_DIRECTORY)
    log("start", f"guard up, say rest {SAY_REST:.0f}s")
    while True:
        names = sorted(e.name for e in os.scandir(dirfd))[:MAX_PER_ROUND]
        for name in names:
            raw, why = read_note(dirfd, name)
            try:
                os.unlink(name, dir_fd=dirfd)                 # every note is used once, whatever it was
            except OSError:
                pass
            if raw is None:
                log("drop", why, name)
                continue
            decision, why, last_say = act(raw, last_say)
            log(decision, why, raw if decision == "drop" else None)
        time.sleep(1)


def tell(text):
    """put something in his inbox (what his senses or his person say)"""
    name = f"{time.time():.6f}.json"
    tmp = os.path.join(INBOX, "." + name)
    with open(tmp, "w") as f:
        json.dump({"t": time.time(), "text": text}, f)
    os.replace(tmp, os.path.join(INBOX, name))


if __name__ == "__main__":
    if sys.argv[1:2] == ["run"]:
        run()
    elif sys.argv[1:2] == ["tell"] and len(sys.argv) == 3:
        tell(sys.argv[2])
    else:
        sys.exit(__doc__)
