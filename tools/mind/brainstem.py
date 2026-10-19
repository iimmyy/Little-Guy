#!/usr/bin/env python3
"""Little Guy's brainstem on the Mac: owns his brain (tools/mind/brain.py on the Mac's GPU) and answers two kinds of
request, nothing else. His mind never reaches it directly: it lives in his room (an OrbStack Linux machine with no
network), and the room's relay (tools/room/relay.py) forwards only these two, with a shared key.

  POST /think  {"messages": [...], "v": .., "a": .., "max_tokens": .., "temperature": .., "top_p": .., "seed": ..}
               -> {"text": ...}     a reply with his feelings as a drip
  POST /read   {"text": ...}  -> {"feel": [valence, arousal]}   how a text reads inside him

usage: brainstem.py <brain.gguf> <dials.npz> <key file> [port, default 8771]
Listens on 127.0.0.1 only. Writes his calibration (the natural spread of each dial) next to the key file.
"""
import hmac, json, os, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brain import Brain

MAX_BODY = 64 * 1024
ROLES = {"system", "user", "assistant"}
lock = threading.Lock()                      # one brain, one thought at a time


def check_think(r):
    m = r.get("messages")
    if not isinstance(m, list) or not 1 <= len(m) <= 40:
        raise ValueError("messages: 1 to 40")
    for x in m:
        if not isinstance(x, dict) or x.get("role") not in ROLES or not isinstance(x.get("content"), str):
            raise ValueError("each message needs a role and text")
    if sum(len(x["content"]) for x in m) > 32000:
        raise ValueError("messages too long")
    num = lambda k, d, lo, hi: min(hi, max(lo, float(r.get(k, d))))
    return dict(msgs=m, v=num("v", 0, -1.5, 1.5), a=num("a", 0, -1.5, 1.5), max_tokens=int(num("max_tokens", 200, 1, 300)),
                temperature=num("temperature", 0.8, 0, 2), top_p=num("top_p", 0.95, 0.05, 1), seed=int(num("seed", 0, 0, 2**31)))


def main(gguf, dials, key_file, port="8771"):
    key = open(key_file).read().strip().encode()
    brain = Brain(gguf, dials)
    spread = brain.calibrate()
    json.dump({k: [round(float(x), 5) for x in v] for k, v in spread.items()},
              open(os.path.join(os.path.dirname(os.path.abspath(key_file)), "calibration.json"), "w"))

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            try:
                if not hmac.compare_digest(self.headers.get("X-LG-Key", "").encode(), key):
                    return self.send(403, {"error": "wrong key"})
                n = int(self.headers.get("Content-Length", 0))
                if not 0 < n <= MAX_BODY:
                    return self.send(413, {"error": "body size"})
                r = json.loads(self.rfile.read(n))
                with lock:
                    if self.path == "/think":
                        return self.send(200, {"text": brain.think(**check_think(r))})
                    if self.path == "/read":
                        t = r.get("text")
                        if not isinstance(t, str) or not 0 < len(t) <= 8000:
                            raise ValueError("text: 1 to 8000 characters")
                        return self.send(200, {"feel": brain.read_feel(t)})
                return self.send(404, {"error": "only /think and /read"})
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
                return self.send(400, {"error": str(e)})

        def send(self, code, obj):
            b = json.dumps(obj).encode()
            self.send_response(code); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    print(f"BRAINSTEM_READY 127.0.0.1:{port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", int(port)), H).serve_forever()


if __name__ == "__main__":
    main(*sys.argv[1:])
