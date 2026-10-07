"""Step 1 of conversation memory: how long until the first answer word as a chat grows,
with the prompt cache on (only new words are read) vs off (whole chat re-read every turn).

Usage on the board: python3 memory_test.py
"""
import json, os, subprocess, time, urllib.request

HOME = os.path.expanduser("~")
PORT = 8099
env = dict(os.environ, LD_LIBRARY_PATH=f"{HOME}/llama/lib", ADSP_LIBRARY_PATH=f"{HOME}/llama/lib", GGML_HEXAGON_NDEV="0")
SYSTEM = "You are Little Guy, a small friendly robot who lives with Immy. Keep replies short and warm, like talking out loud."
USER_TURNS = [
    "Hey Little Guy, I'm home.",
    "Work was long. The pharmacy was packed all day.",
    "Yeah, flu season. Everyone wanted their shots at once.",
    "I'm thinking of making pasta for dinner. Any ideas?",
    "I have garlic, spinach, and some parmesan.",
    "That sounds good. Remind me, I have a dentist appointment Thursday at 3.",
    "Also my friend Sam is coming over Saturday.",
    "Sam likes board games. What should we play?",
    "We have Catan and Codenames.",
    "Okay Codenames it is. What day is my dentist appointment again?",
    "Nice, you remembered. I'm going to work on your servos this weekend too.",
    "One of them is making a grinding noise.",
    "I think it needs grease. Have you ever been greased before?",
    "Haha. Okay, I'm going to start cooking now.",
    "The water's boiling. How long do I cook the pasta?",
    "Got it. While I wait, tell me something interesting.",
    "That's cool. Do you like being a robot?",
    "What would you want to learn next?",
    "I'll teach you. Who's coming over Saturday?",
    "Perfect. Okay, dinner's ready. Talk later!",
]
MEASURE_OFF = {int(t) for t in os.environ.get("MEASURE_OFF", "1,5,10,15,20").split(",") if t}


def cpu_temp():
    return int(open("/sys/class/thermal/thermal_zone1/temp").read()) // 1000


def ask(messages, cache, max_tokens):
    body = {"messages": messages, "max_tokens": max_tokens, "stream": True, "cache_prompt": cache,
            "temperature": 1.0, "top_k": 64, "top_p": 0.95, "seed": 1, "timings_per_token": True}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    first, text, timings = None, "", {}
    with urllib.request.urlopen(req, timeout=900) as resp:
        for line in resp:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            timings = chunk.get("timings", timings)
            if not chunk.get("choices"):
                continue
            piece = chunk["choices"][0].get("delta", {}).get("content") or ""
            if piece.strip() and first is None:
                first = time.time() - t0
            text += piece
    # prompt_n = words actually read this turn; cache_n = words reused from memory
    return first, text, timings.get("prompt_n", 0), timings.get("cache_n", 0)


server = subprocess.Popen([f"{HOME}/llama/bin/llama-server", "-m", f"{HOME}/models/gemma-4-E4B_q4_0-it.gguf",
                           "-md", f"{HOME}/models/gemma-4-E4B-assistant-masked-q8_0.gguf", "--spec-type", "draft-mtp",
                           "--spec-draft-n-max", "3", "-ngl", "0", "-ngld", "0", "--device", "none", "-t", "4", "-td", "4",
                           "-c", "4096", "--reasoning", "off", "--port", str(PORT), "--no-webui"],
                          env=env, stdout=open(f"{HOME}/spec/memory_server.log", "w"), stderr=subprocess.STDOUT)
try:
    for _ in range(240):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
            break
        except Exception:
            time.sleep(1)

    print("MEMORY ON (real chat)", flush=True)
    messages = [{"role": "system", "content": SYSTEM}]
    histories = {}
    for turn, user in enumerate(USER_TURNS, 1):
        messages.append({"role": "user", "content": user})
        histories[turn] = list(messages)
        first, reply, read, reused = ask(messages, True, 60)
        messages.append({"role": "assistant", "content": reply.strip()})
        print(f"turn {turn:2d}: first word {first:5.2f} s | read {read:4d} new, reused {reused:4d} | {cpu_temp()} C"
              f" | {reply.strip()[:60]!r}", flush=True)

    print("MEMORY OFF (same chats, re-read from scratch)", flush=True)
    for turn in sorted(MEASURE_OFF):
        while cpu_temp() > 50:
            time.sleep(5)
        first, _, read, reused = ask(histories[turn], False, 4)
        print(f"turn {turn:2d}: first word {first:5.2f} s | read {read:4d} new, reused {reused:4d}", flush=True)
finally:
    server.terminate()
    server.wait()
