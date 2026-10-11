"""Little Guy's live page: board vitals plus his feed, view-only.

Runs on the board as littleguy-web.service. Open http://<board>:8080
"""
import collections, json, os, subprocess, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOME = os.path.expanduser("~")
FEED = os.environ.get("LITTLEGUY_FEED", f"{HOME}/spec/live.log")
PAGE = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html"), "rb").read()
# CPU core sensors (cpu0..cpu11-thermal); the hottest one is what the heat guard reacts to
CPU_ZONES = (1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14)
PORT = int(os.environ.get("LITTLEGUY_WEB_PORT", "8080"))


def read_int(path):
    try:
        return int(open(path).read())
    except (OSError, ValueError):
        return None


def hottest():
    temps = []
    for zone in os.listdir("/sys/class/thermal"):
        t = read_int(f"/sys/class/thermal/{zone}/temp")
        if zone.startswith("thermal_zone") and t and 0 < t < 150000:
            temps.append(t)
    return round(max(temps) / 1000) if temps else None


def ram():
    info = {}
    for line in open("/proc/meminfo"):
        key, value = line.split(":")
        info[key] = int(value.split()[0]) // 1024
    return info["MemTotal"] - info["MemAvailable"], info["MemTotal"]


def feed(lines=40):
    try:
        with open(FEED, errors="replace") as f:
            return [line.rstrip("\n") for line in f.readlines()[-lines:]]
    except OSError:
        return []


TICK = os.sysconf("SC_CLK_TCK")
PAGE_KB = os.sysconf("SC_PAGE_SIZE") // 1024
activity = {"cores": [], "procs": [], "model": False}
last_viewed = 0.0  # when someone last asked for the status; the sampler rests when nobody is watching
journal = collections.deque(maxlen=60)


def cpu_times():
    """Per-core (busy, total) jiffies from /proc/stat."""
    out = []
    for line in open("/proc/stat"):
        if line.startswith("cpu") and line[3].isdigit():
            v = [int(x) for x in line.split()[1:]]
            idle = v[3] + v[4]
            out.append((sum(v) - idle, sum(v)))
    return out


def proc_ticks():
    """{pid: (name, cpu ticks, rss MB)} for every process."""
    out = {}
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            raw = open(f"/proc/{pid}/stat").read()
            name = raw[raw.index("(") + 1:raw.rindex(")")]
            if name.startswith("python"):  # "python3" says nothing; use the script's name instead
                args = open(f"/proc/{pid}/cmdline").read().split("\0")
                script = next((a for a in args[1:] if a and not a.startswith("-")), "")
                name = os.path.basename(script).removesuffix(".py").replace("_", "-") or name
            fields = raw[raw.rindex(")") + 2:].split()
            rss_mb = int(fields[21]) * PAGE_KB // 1024
            out[pid] = (name, int(fields[11]) + int(fields[12]), rss_mb)
        except (OSError, ValueError, IndexError):
            pass
    return out


def sample_activity():
    """Once a second while someone is watching: how busy each core is and which programs are using the CPU.
    Walking every process costs real CPU time, so with nobody watching it rests and starts fresh when they return."""
    prev_cores, prev_procs, prev_t = cpu_times(), proc_ticks(), time.time()
    while True:
        time.sleep(1)
        if time.time() - last_viewed > 5:
            prev_procs = None
            continue
        if prev_procs is None:
            prev_cores, prev_procs, prev_t = cpu_times(), proc_ticks(), time.time()
            continue
        cores, procs, t = cpu_times(), proc_ticks(), time.time()
        activity["cores"] = [round(100 * (b - pb) / max(tt - pt, 1)) for (b, tt), (pb, pt) in zip(cores, prev_cores)]
        rows = []
        for pid, (name, ticks, rss) in procs.items():
            if pid in prev_procs:
                pct = 100 * (ticks - prev_procs[pid][1]) / TICK / (t - prev_t)
                rows.append({"pid": int(pid), "name": name, "cpu": round(pct), "mb": rss})
        rows.sort(key=lambda r: (-r["cpu"], -r["mb"]))
        activity["procs"] = rows[:8]
        activity["model"] = any(name.startswith("llama-") for name, _, _ in procs.values())
        prev_cores, prev_procs, prev_t = cores, procs, t


def follow_journal():
    """Streams the board's whole system log (every service) into a short buffer."""
    while True:
        p = subprocess.Popen(["journalctl", "-f", "-n", "40", "-o", "short", "--no-pager"],
                             stdout=subprocess.PIPE, text=True, errors="replace")
        for line in p.stdout:
            journal.append(line.rstrip("\n"))
        time.sleep(3)  # journalctl exited; start again


def gpu_mhz():
    return (read_int("/sys/class/devfreq/3d00000.gpu/cur_freq") or 0) // 1_000_000


def throttled():
    """True when the heat guard has lowered any CPU cluster's speed limit below its hardware maximum."""
    for p in (0, 4, 7):
        cur = read_int(f"/sys/devices/system/cpu/cpufreq/policy{p}/scaling_max_freq")
        hw = read_int(f"/sys/devices/system/cpu/cpufreq/policy{p}/cpuinfo_max_freq")
        if cur and hw and cur < hw:
            return True
    return False


def status():
    used, total = ram()
    return {
        "cores": activity["cores"], "procs": activity["procs"], "journal": list(journal), "gpu_mhz": gpu_mhz(),
        "time": time.strftime("%H:%M:%S"),
        "cpu_temp": round(max(read_int(f"/sys/class/thermal/thermal_zone{z}/temp") or 0 for z in CPU_ZONES) / 1000),
        "hottest": hottest(),
        "cpu_mhz": [(read_int(f"/sys/devices/system/cpu/cpufreq/policy{p}/scaling_cur_freq") or 0) // 1000 for p in (0, 4, 7)],
        "ram_used": used, "ram_total": total,
        "uptime_min": int(float(open("/proc/uptime").read().split()[0]) // 60),
        "model": activity["model"],
        "feed": feed(),
        "throttled": throttled(),
        "brain_service": os.path.exists("/etc/systemd/system/littleguy-brain.service"),
        "quiet": os.path.exists("/var/lib/littleguy/quiet"),  # thermal-guard's night mode: lights off, clocks capped
        "feed_age_s": int(time.time() - os.path.getmtime(FEED)) if os.path.exists(FEED) else None,
    }


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/status":
            global last_viewed
            last_viewed = time.time()
            body, kind = json.dumps(status()).encode(), "application/json"
        elif self.path in ("/", "/index.html"):
            body, kind = PAGE, "text/html; charset=utf-8"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass  # one request a second per viewer; keep the journal quiet


threading.Thread(target=sample_activity, daemon=True).start()
threading.Thread(target=follow_journal, daemon=True).start()
ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
