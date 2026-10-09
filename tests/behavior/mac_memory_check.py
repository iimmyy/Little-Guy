"""Mac check for the SWA checkpoint fix: with memory on, each turn should read only its new words,
and the answers must match a from-scratch read of the same chat (temperature 0)."""
import json, subprocess, time, urllib.request

from common import BRAINS, DRAFTER, LOGS, SERVER
PORT = 8095
TURNS = ["Hey Little Guy, I'm home.", "Work was long. The pharmacy was packed.", "I'm making pasta. Ideas?",
         "I have garlic and spinach.", "My dentist appointment is Thursday at 3.", "What day is my dentist appointment?"]
server = subprocess.Popen([SERVER, "-m", BRAINS / "qat-Q4_0-f16proj.gguf",
                           "-md", DRAFTER / "gemma-4-E4B-assistant-masked-q8_0.gguf", "--spec-type", "draft-mtp",
                           "--spec-draft-n-max", "3", "-ngl", "99", "-ngld", "99", "-c", "4096", "--reasoning", "off",
                           "--port", str(PORT), "--no-webui"], stdout=open(LOGS / "mac_memory_server.log", "w"),
                          stderr=subprocess.STDOUT)


def ask(messages, cache):
    body = {"messages": messages, "max_tokens": 40, "temperature": 0, "cache_prompt": cache}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=300))
    return r["choices"][0]["message"]["content"].strip(), r["timings"]["prompt_n"]


try:
    for _ in range(120):
        try:
            if json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)).get("status") == "ok":
                break
        except Exception:
            time.sleep(1)
    messages = [{"role": "system", "content": "You are Little Guy, a small friendly robot. Keep replies short."}]
    for turn, user in enumerate(TURNS, 1):
        messages.append({"role": "user", "content": user})
        cached, read_cached = ask(messages, True)
        fresh, read_fresh = ask(messages, False)
        ask(messages, True)  # put the cached chat back in memory after the from-scratch read
        same = "same" if cached == fresh else "DIFFERENT"
        print(f"turn {turn}: memory read {read_cached:3d} new vs scratch {read_fresh:3d} | answers {same} | {cached[:50]!r}", flush=True)
        messages.append({"role": "assistant", "content": cached})
finally:
    server.terminate()
    server.wait()
