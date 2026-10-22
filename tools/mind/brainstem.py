#!/usr/bin/env python3
"""Little Guy's brainstem on the Mac: owns his brain (tools/mind/brain.py on the Mac's GPU) and answers two kinds of
request, nothing else. His mind never reaches it directly: it lives in his room (an OrbStack Linux machine with no
network), and the room's relay (tools/room/relay.py) forwards only these two, with a shared key.

  POST /think  {"messages": [...], "v": .., "a": .., "max_tokens": .., "temperature": .., "top_p": .., "seed": ..}
               -> {"text": ...}     a reply with his feelings as a drip
  POST /read   {"text": ...}  -> {"feel": [valence, arousal]}   how a text reads inside him
  POST /print  {"text": ...}  -> {"print": [n_embd floats]}   how a message sounds to him (who's writing): his ear
               learns each person's print from these (tools/room/mind_live.py, voices)
  POST /choose {"messages": [...], "choices": ["no", "wander", ...], "v": .., "a": ..}
               -> {"face": "[ .. ]", "logprobs": {choice: lp}}   the odds of each choice as his whole answer, after
               the face he opens with (2 to 8 plain words)
  /think with "thinking": true uses the brain's own thinking switch -> {"text", "thought", "n_tok", "n_thought"};
               "think_tokens" is the thought's own budget (up to 400), so the answer always keeps its max_tokens
  /think with "hushable": true is a daydream: POST /hush stops it at the next word -> "hushed": true. /hush does nothing
               to anything else (a reply, a choice, a reading), so a late hush can't cut what he says to you.
  a user message may carry "audio": base64 of a wav, mp3 or flac (up to 2 sounds, 8 MB each): he hears it just before
               that message's words (needs his ears: env LG_EARS). The sound is saved here, outside his room, so a
               request can only hand him a sound, never point his ears at a file.

usage: brainstem.py <brain.gguf> <dials.npz> <key file> [port, default 8771]
Listens on 127.0.0.1 only. Writes his calibration (the natural spread of each dial) next to the key file.
"""
import base64, hmac, json, os, sys, threading
import numpy as np
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brain import Brain

MAX_BODY = 64 * 1024
MAX_BODY_SOUND = 24 * 1024 * 1024            # a request that carries sounds (base64 is 4/3 of the bytes)
MAX_SOUND = 8 * 1024 * 1024
MARKER = "<__media__>"                       # where a sound goes in the prompt (llama.cpp's mtmd marker)
ROLES = {"system", "user", "assistant"}
lock = threading.Lock()                      # one brain, one thought at a time
now = {"hushable": False}                    # is the thought being written right now a daydream?


def check_msgs(r):
    m = r.get("messages")
    if not isinstance(m, list) or not 1 <= len(m) <= 40:
        raise ValueError("messages: 1 to 40")
    for x in m:
        if not isinstance(x, dict) or x.get("role") not in ROLES or not isinstance(x.get("content"), str):
            raise ValueError("each message needs a role and text")
        if MARKER in x["content"]:
            raise ValueError("text can't contain the sound marker")
    if sum(len(x["content"]) for x in m) > 32000:
        raise ValueError("messages too long")
    return m


def sound_kind(b):
    if b[:4] == b"RIFF" and b[8:12] == b"WAVE":
        return "wav"
    if b[:4] == b"fLaC":
        return "flac"
    if b[:3] == b"ID3" or (len(b) > 1 and b[0] == 0xFF and b[1] & 0xE0 == 0xE0):
        return "mp3"
    return None


def take_sounds(m, folder, count):
    """the sounds riding on user messages: each saved in the brainstem's own folder, its marker put before that
    message's words; returns (messages for the brain, sound files in order)"""
    out, files = [], []
    for x in m:
        a = x.get("audio")
        if a is None:
            out.append({"role": x["role"], "content": x["content"]})
            continue
        if x["role"] != "user" or not isinstance(a, str):
            raise ValueError("audio: base64 text on a user message")
        b = base64.b64decode(a, validate=True)
        kind = sound_kind(b)
        if not kind or not 0 < len(b) <= MAX_SOUND:
            raise ValueError("audio: a wav, mp3 or flac up to 8 MB")
        count[0] += 1
        path = os.path.join(folder, f"sound{count[0]}.{kind}")
        with open(path, "wb") as f:
            f.write(b)
        files.append(path)
        out.append({"role": "user", "content": MARKER + "\n" + x["content"]})
    if len(files) > 2:
        raise ValueError("audio: at most 2 sounds per request")
    return out, files


num = lambda r, k, d, lo, hi: min(hi, max(lo, float(r.get(k, d))))
MOST_WORDS = int(os.environ.get("LG_MOST_WORDS", "300"))  # the most one answer may be; practice for judging raises it


def check_think(r, folder=None, count=None):
    m = check_msgs(r)
    sounds = []
    if any("audio" in x for x in m):
        if folder is None:
            raise ValueError("he has no ears loaded")
        m, sounds = take_sounds(m, folder, count)
    thinking = r.get("thinking", False) is True
    return dict(msgs=m, audio=sounds, v=num(r, "v", 0, -1.5, 1.5), a=num(r, "a", 0, -1.5, 1.5),
                max_tokens=int(num(r, "max_tokens", 200, 1, MOST_WORDS)),
                think_tokens=int(num(r, "think_tokens", 0, 0, 400)) if thinking else 0,
                temperature=num(r, "temperature", 0.8, 0, 2), top_p=num(r, "top_p", 0.95, 0.05, 1),
                seed=int(num(r, "seed", 0, 0, 2**31)), thinking=thinking)


def check_choose(r):
    c = r.get("choices")
    if not isinstance(c, list) or not 2 <= len(c) <= 8 or not all(isinstance(x, str) and 0 < len(x) <= 40
                                                                      and x.replace(" ", "").isalnum() for x in c):
        raise ValueError("choices: 2 to 8 short plain words")       # no markup: they go straight to the tokenizer
    return dict(msgs=check_msgs(r), choices=c, v=num(r, "v", 0, -1.5, 1.5), a=num(r, "a", 0, -1.5, 1.5),
                seed=int(num(r, "seed", 0, 0, 2**31)))


def main(gguf, dials, key_file, port="8771"):
    key = open(key_file).read().strip().encode()
    brain = Brain(gguf, dials, ears=os.environ.get("LG_EARS") or None)
    sound_dir = brain.tmp if brain.ears else None
    sound_n = [0]
    if os.environ.get("LG_CALIBRATION"):                  # reuse a measured one (a test brainstem with an adapter
        brain.spread = {k: np.array(v, dtype=np.float32)   # feels on the same scale as the live one)
                        for k, v in json.load(open(os.environ["LG_CALIBRATION"])).items()}
    else:
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
                if not 0 < n <= (MAX_BODY_SOUND if brain.ears and self.path == "/think" else MAX_BODY):
                    return self.send(413, {"error": "body size"})
                r = json.loads(self.rfile.read(n))
                if self.path == "/hush":                    # never waits for the lock: it's for the thought holding it
                    if now["hushable"]:
                        brain.hush()
                    return self.send(200, {"hushed": now["hushable"]})
                with lock:
                    if self.path == "/think":
                        req = check_think(r, sound_dir, sound_n)
                        now["hushable"] = r.get("hushable", False) is True
                        try:
                            return self.send(200, brain.think_full(**req))
                        finally:
                            now["hushable"] = False
                            for f in req["audio"]:          # heard, not kept: what he remembers is his own words
                                os.remove(f)
                    if self.path == "/choose":
                        return self.send(200, brain.choose(**check_choose(r)))
                    if self.path == "/read":
                        t = r.get("text")
                        if not isinstance(t, str) or not 0 < len(t) <= 8000:
                            raise ValueError("text: 1 to 8000 characters")
                        return self.send(200, {"feel": brain.read_feel(t)})
                    if self.path == "/print":
                        t = r.get("text")
                        if not isinstance(t, str) or not 0 < len(t) <= 8000:
                            raise ValueError("text: 1 to 8000 characters")
                        return self.send(200, {"print": [round(x, 4) for x in brain.style_print(t)]})
                return self.send(404, {"error": "only /think, /read, /choose, /print and /hush"})
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
