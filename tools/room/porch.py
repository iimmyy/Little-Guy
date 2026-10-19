#!/usr/bin/env python3
"""The porch: the Mac's side of Little Guy's room (Immy: he should both speak out loud and show in the menu bar).

Every few seconds it looks in through the window (read only, as the machine's admin; nothing goes back in):
  what the guard let out   said.log (his calls out loud), chat.jsonl (his replies), status.json (his mood)
  his notebook             notes.txt (what he's thinking)
Then it
  speaks each new call out loud with the Mac's voice (`say`; on the Jetson, his own speaker)
  serves the menu bar app's /api/status on 127.0.0.1 only: his notes, calls, replies and mood as the "brain" feed,
  the Mac's memory and his three services as the vitals
usage: porch.py [port, default 8080]       settings (env): MACHINE (default lgroom), VOICE (a `say -v` voice)
"""
import json, os, re, subprocess, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ORB = os.environ.get("ORB", "/opt/homebrew/bin/orb")
MACHINE = os.environ.get("MACHINE", "lgroom")
VOICE = os.environ.get("VOICE", "")
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


def loop():
    spoken = None
    while True:
        try:
            w = look()
            if spoken is None:
                spoken = len(w["said"])                      # only what he says from now on
            for line in w["said"][spoken:]:
                spoken_text = re.sub(r"\[[^\]]*\]|\*", "", line[9:]).strip()   # his faces are for screens, not voices
                if spoken_text:
                    subprocess.run(["say"] + (["-v", VOICE] if VOICE else []) + [spoken_text], timeout=60)
            spoken = len(w["said"])
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
