#!/usr/bin/env python3
"""The brain slot: the one door between Little Guy's mind (no network) and his brain (the brainstem on the host).
Listens on a socket in /var/lib/littleguy/brain that only the mail group can open; each connection carries one JSON
line, {"op": "think" | "read" | "choose", ...}, which goes to the brainstem with the shared key, and one JSON line comes back.
Anything else is refused here, before it leaves the machine. Every request is logged (op, size, outcome).
Settings (env): BRAINSTEM (default http://host.orb.internal:8771), KEY_FILE (default /etc/littleguy/brain.key)
"""
import json, os, socketserver, time, urllib.error, urllib.request

SOCK = "/var/lib/littleguy/brain/brain.sock"
LOG = "/var/lib/littleguy-relay/relay.jsonl"
URL = os.environ.get("BRAINSTEM", "http://host.orb.internal:8771")
KEY = open(os.environ.get("KEY_FILE", "/etc/littleguy/brain.key")).read().strip()
MAX_LINE = 64 * 1024


def log(**kw):
    with open(LOG, "a") as f:
        f.write(json.dumps({"t": time.strftime("%Y-%m-%d %H:%M:%S"), **kw}) + "\n")


class Slot(socketserver.StreamRequestHandler):
    timeout = 300

    def handle(self):
        line = self.rfile.readline(MAX_LINE + 1)
        try:
            if len(line) > MAX_LINE:
                raise ValueError("request too big")
            r = json.loads(line)
            op = r.pop("op", None) if isinstance(r, dict) else None
            if op not in ("think", "read", "choose"):
                raise ValueError(f"not a brain request: {op!r}")
            req = urllib.request.Request(f"{URL}/{op}", json.dumps(r).encode(),
                                         {"Content-Type": "application/json", "X-LG-Key": KEY})
            try:
                out = json.load(urllib.request.urlopen(req, timeout=280))
            except urllib.error.HTTPError as e:
                out = json.load(e)
            log(op=op, bytes=len(line), ok="error" not in out)
        except Exception as e:                                    # noqa: BLE001 (anything odd is refused)
            out = {"error": str(e)[:200]}
            log(op="refused", bytes=len(line), why=out["error"])
        self.wfile.write((json.dumps(out) + "\n").encode())


class Server(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True


if __name__ == "__main__":
    if os.path.exists(SOCK):
        os.unlink(SOCK)
    old = os.umask(0o117)                                         # socket 0660: the relay and the mail group
    srv = Server(SOCK, Slot)
    os.umask(old)
    log(op="start", url=URL)
    srv.serve_forever()
