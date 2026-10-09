"""TTFT step 1: where does the wait go on the board, with Immy's real persona?
Per turn: words read, reading time (server clock), time to the face flag, time to the first spoken word.

Usage on the board: python3 ttft_split.py
"""
import json, os, re, subprocess, time, urllib.request

HOME = os.path.expanduser("~")
PORT = 8096
FEED = f"{HOME}/spec/live.log"
env = dict(os.environ, LD_LIBRARY_PATH=f"{HOME}/llama/lib", ADSP_LIBRARY_PATH=f"{HOME}/llama/lib", GGML_HEXAGON_NDEV="0")
PERSONA = "\n".join(l for l in open(f"{HOME}/spec/little_guy_persona.txt").read().splitlines() if l.strip() and not l.startswith("#"))
TURNS = ["Hey Little Guy, I'm home.", "Work was long. The pharmacy was packed all day.",
         "I'm thinking of making pasta for dinner. Any ideas?", "I have garlic, spinach, and some parmesan.",
         "Remind me, I have a dentist appointment Thursday at 3.", "Also my friend Sam is coming over Saturday.",
         "What day is my dentist appointment again?", "Okay, I'm going to start cooking now."]


def hottest():
    return max(int(open(f"/sys/class/thermal/thermal_zone{z}/temp").read()) for z in (1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14)) // 1000


def feed(line):
    with open(FEED, "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')}  {line}\n")


def ask(messages):
    body = {"messages": messages, "max_tokens": 60, "stream": True, "temperature": 1.0, "top_k": 64, "top_p": 0.95,
            "seed": 1}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    t_any = t_word = None
    text, timings = "", {}
    with urllib.request.urlopen(req, timeout=900) as resp:
        for line in resp:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            timings = chunk.get("timings", timings)
            if not chunk.get("choices"):
                continue
            text += chunk["choices"][0].get("delta", {}).get("content") or ""
            if t_any is None and text.strip():
                t_any = time.time() - t0
            spoken = re.sub(r"\[[^\]]*\]?", "", text)  # face flags are for the screen, not the voice
            if t_word is None and re.search(r"\w", spoken):
                t_word = time.time() - t0
    return text.strip(), t_any, t_word, timings


while hottest() > 45:
    feed(f"(waiting to cool: hottest core {hottest()} C)")
    time.sleep(10)
server = subprocess.Popen([f"{HOME}/llama/bin/llama-server", "-m", f"{HOME}/models/gemma-4-E4B_q4_0-it.gguf",
                           "-md", f"{HOME}/models/gemma-4-E4B-assistant-masked-q8_0.gguf", "--spec-type", "draft-mtp",
                           "--spec-draft-n-max", "3", "-ngl", "0", "-ngld", "0", "--device", "none", "-t", "4", "-td", "4",
                           "-c", "4096", "--reasoning", "off", "--port", str(PORT), "--no-webui"],
                          env=env, stdout=open(f"{HOME}/spec/ttft_split_server.log", "w"), stderr=subprocess.STDOUT)
try:
    for _ in range(240):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
            break
        except Exception:
            time.sleep(1)
    feed("=== TTFT split test: real persona, thinking off ===")
    print(f"{'turn':>4} | {'read':>5} {'reading':>8} | {'face flag':>9} | {'1st word':>8} | temp | reply")
    messages = [{"role": "system", "content": PERSONA}]
    for turn, user in enumerate(TURNS, 1):
        messages.append({"role": "user", "content": user})
        feed(f"heard:  {user}")
        reply, t_any, t_word, tm = ask(messages)
        messages.append({"role": "assistant", "content": reply})
        read, read_ms = tm.get("prompt_n", 0), tm.get("prompt_ms", 0)
        fmt = lambda t: f"{t:7.2f}s" if t else "    none"
        print(f"{turn:4d} | {read:5d} {read_ms / 1000:7.2f}s | {fmt(t_any):>9} | {fmt(t_word):>8} | {hottest():3d}C | {reply[:50]!r}",
              flush=True)
        feed(f"said:   {reply}")
        feed(f"        (read {read} words in {read_ms / 1000:.2f} s, face at {fmt(t_any).strip()}, first word at {fmt(t_word).strip()})")
        if t_word:
            feed(f"stats: first word {t_word:.1f} s · {tm.get('predicted_per_second', 0):.1f} words/s")
finally:
    server.terminate()
    server.wait()
