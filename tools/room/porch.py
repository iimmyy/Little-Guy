#!/usr/bin/env python3
"""The porch: the Mac's side of Little Guy's room (Immy: he should both speak out loud and show in the menu bar).

Every few seconds it looks in through the window (read only, as the machine's admin; nothing goes back in):
  what the guard let out   said.log (his calls out loud), chat.jsonl (his replies), status.json (his mood)
  his notebook             notes.txt (what he's thinking)
Then it
  speaks each new call and each new reply out loud with the Mac's voice (`say`; on the Jetson, his own speaker;
  Immy 2026-10-19: "I want to hear him"); his faces and *asides* are for screens, his inner thoughts stay inside
  serves the menu bar app's /api/status on 127.0.0.1 only: his notes, calls, replies and mood as the "brain" feed,
  the Mac's memory and his three services as the vitals
usage: porch.py [port, default 8080]       settings (env): MACHINE (default lgroom), VOICE (a `say -v` voice)
His voice (Immy 2026-10-19: ElevenLabs): with a key in ~/.littleguy/elevenlabs.key (Immy's, never in git) each line is
made by ElevenLabs and played here; the porch makes the call, from the Mac, so nothing in his room goes online.
Without a key, or when it fails (offline, out of credits), the Mac's own voice.
  ELEVEN_VOICE  a voice id (default George, warm British)      ELEVEN_MODEL  default eleven_multilingual_v2
"""
import json, os, re, subprocess, sys, tempfile, threading, time, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ORB = os.environ.get("ORB", "/opt/homebrew/bin/orb")
MACHINE = os.environ.get("MACHINE", "lgroom")
VOICE = os.environ.get("VOICE", "")
ELEVEN_KEY = os.path.expanduser(os.environ.get("ELEVEN_KEY", "~/.littleguy/elevenlabs.key"))
ELEVEN_VOICE = os.environ.get("ELEVEN_VOICE", "JBFqnCBsd6RMkjVDRZzb")
ELEVEN_MODEL = os.environ.get("ELEVEN_MODEL", "eleven_multilingual_v2")
LOOK = r'''
import json, os, subprocess
def tail(p, n):
    try:
        return open(p, encoding="utf-8", errors="replace").read().splitlines()[-n:]
    except OSError:
        return []
G, NB = "/var/lib/littleguy-guard", "/var/lib/littleguy/room/notebook"
try:
    mood = json.load(open("/run/littleguy/status.json"))
except (OSError, ValueError):
    mood = None
units = ["littleguy-guard", "littleguy-mind", "littleguy-relay"]
active = subprocess.run(["systemctl", "is-active"] + units, capture_output=True, text=True).stdout.split()
print(json.dumps({"said": tail(G + "/said.log", 200), "chat": tail(NB + "/chat.jsonl", 30), "notes": tail(NB + "/notes.jsonl", 30),
                  "inner": tail(NB + "/thoughts.jsonl", 40), "scores": tail(NB + "/scores.jsonl", 10),
                  "mood": mood, "units": dict(zip(units, active))}))
'''
view = {"status": None}
started = time.time()


def look():
    r = subprocess.run([ORB, "-m", MACHINE, "-u", "root", "python3", "-c", LOOK], capture_output=True, text=True,
                       timeout=20)
    return json.loads(r.stdout)


def mac_memory():
    total = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout)
    vm = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = int(vm.split("page size of ")[1].split()[0])
    n = {l.split(":")[0]: int(l.split(":")[1].strip(" .")) for l in vm.splitlines()[1:] if ":" in l}
    used = (n.get("Pages active", 0) + n.get("Pages wired down", 0) + n.get("Pages occupied by compressor", 0)) * page
    return used >> 20, total >> 20


def feed(w):
    """one timeline of what he did, oldest first: notes, inner thoughts, what was said, tasks; then his mood now"""
    rows = []
    def each(lines):
        for l in lines:
            try:
                yield json.loads(l)
            except ValueError:
                continue
    stamp = lambda t: time.strftime("%a %-I:%M %p", time.localtime(t))
    for n in each(w["notes"]):
        rows.append((n["t"], f"[{stamp(n['t'])}] note: {n['thought']}"))
    for x in each(w.get("inner", [])):
        about = "wandering" if x.get("subject") == "wander" else "about a task"
        rows.append((x["t"], f"[{stamp(x['t'])}] inner ({about}): {x['text']}"))
    for c in each(w["chat"]):
        rows.append((c["t"], f"[{stamp(c['t'])}] heard: {c['heard']}"))
        rows.append((c["t"] + 0.1, f"[{stamp(c['t'])}] replied{' (thought first)' if c.get('thinking') else ''}: "
                                   f"{c['reply']}"))
    for x in each(w.get("scores", [])):
        t = x.get("done_t") or x["due"]
        rows.append((t, f"[{stamp(t)}] task {x['status']}: {x['what']} (reward {x['reward']:+.2f}, "
                        f"{x['thoughts']} thoughts about it)"))
    lines = [r for _, r in sorted(rows)][-60:]
    lines += [f"said out loud ({l[:8]}): {l[9:]}" for l in w["said"][-5:]]
    m = w["mood"]
    if m:
        lines.append(f"mood now: pleasant {m['valence']:+.2f}, keyed up {m['arousal']:+.2f}"
                     f" (as of {time.strftime('%-I:%M %p', time.localtime(m['t']))})")
    return lines


def speak(text):
    """out loud, without his faces ([ happy ]) or *asides* (screens only)"""
    text = " ".join(re.sub(r"\*[^*\n]*\*|\[[^\]\n]*\]|\*", " ", text).split())
    if not text:
        return
    try:
        if os.path.exists(ELEVEN_KEY):
            eleven(text)
            return
    except (OSError, ValueError, subprocess.SubprocessError) as e:     # urllib errors are OSErrors
        print(f"porch: ElevenLabs failed, the Mac's voice instead: {e}", flush=True)
    subprocess.run(["say"] + (["-v", VOICE] if VOICE else []) + [text], timeout=60)


def eleven(text):
    req = urllib.request.Request(
        f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE}?output_format=mp3_44100_128",
        data=json.dumps({"text": text, "model_id": ELEVEN_MODEL}).encode(), method="POST",
        headers={"xi-api-key": open(ELEVEN_KEY).read().strip(), "Content-Type": "application/json"})
    audio = urllib.request.urlopen(req, timeout=30).read()
    with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
        f.write(audio); f.flush()
        subprocess.run(["afplay", f.name], timeout=120, check=True)


def loop():
    spoken, replied = None, None
    while True:
        try:
            w = look()
            chat = []
            for l in w["chat"]:
                try:
                    chat.append(json.loads(l))
                except ValueError:
                    pass
            if spoken is None:
                spoken = len(w["said"])                      # only what he says from now on
                replied = max([c["t"] for c in chat], default=0)
            for line in w["said"][spoken:]:
                speak(line[9:])
            spoken = len(w["said"])
            for c in chat:
                if c["t"] > replied and c.get("reply"):
                    speak(c["reply"])
            replied = max([c["t"] for c in chat] + [replied])
            used, total = mac_memory()
            alive = w["units"].get("littleguy-mind") == "active"
            view["status"] = {
                "time": time.strftime("%H:%M:%S"), "cpu_temp": 0, "cores": [], "procs": [],
                "journal": [f"{u}: {s}" for u, s in w["units"].items()], "feed": feed(w),
                "feed_age_s": 0 if alive else 10**6, "model": alive, "ram_used": used, "ram_total": total,
                "uptime_min": int((time.time() - started) / 60), "throttled": False, "brain_service": alive,
                "quiet": None}
        except (subprocess.SubprocessError, OSError, ValueError, KeyError, IndexError) as e:
            print(f"porch: {e}", flush=True)
        time.sleep(3)


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path != "/api/status" or view["status"] is None:
            self.send_response(404); self.end_headers(); return
        who = self.headers.get("User-Agent", "?").split()[0]
        if who not in view.setdefault("seen", set()):
            view["seen"].add(who)
            print(f"first visit from {who}", flush=True)
        b = json.dumps(view["status"]).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)


if __name__ == "__main__":
    threading.Thread(target=loop, daemon=True).start()
    print("PORCH_READY", flush=True)
    ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1]) if sys.argv[1:] else 8080), H).serve_forever()
