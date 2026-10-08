"""Can thinking be switched per message without a full re-read? (Mac test)

Setups: 1 off, 2 official on, 3 trick A (top off, open the notes ourselves),
4 trick B (top on, open+close empty notes ourselves). Same 6-message chat, temperature 0.
Prompts are rendered by the server's own template (/apply-template), then sent raw to /completion
so we can add text where his reply starts.
"""
import json, re, subprocess, time, urllib.request

RUN = "/Users/immyyyyy/Documents/LittleGuy/.run"
PORT = 8094
OPEN, CLOSE = "<|channel>thought\n", "<channel|>"
SYSTEM = "You are Little Guy, a small friendly robot who lives with Immy. Keep replies short and warm."
CHAT = [  # (message, right answer or None)
    ("Hey Little Guy, I'm home.", None),
    ("A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?", ["0.05", "5 cents", "five cents", "5¢"]),
    ("Tell me a short robot joke.", None),
    ("How many times does the letter r appear in the word strawberry?", ["3", "three"]),
    ("How was your day?", None),
    ("I have 4 apples, buy 2 dozen more, then give away half. How many apples do I have left?", ["14", "fourteen"]),
]
SETUPS = {  # name: (top switch on?, text added where his reply starts)
    "1 off":      (False, ""),
    "2 on":       (True, ""),
    "3 trick A":  (False, OPEN),
    "4 trick B":  (True, OPEN + CLOSE),
}

server = subprocess.Popen([f"{RUN}/lcpp-src/build-mac/bin/llama-server", "-m", f"{RUN}/models/gemma-4-E4B_q4_0-it.gguf",
                           "-ngl", "99", "-c", "8192", "-np", "1", "--special", "--port", str(PORT), "--no-webui"],
                          stdout=open(f"{RUN}/spec-test/think_tricks_server.log", "w"), stderr=subprocess.STDOUT)


def post(path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=600)


def render(messages, think_on):
    r = json.load(post("/apply-template", {"messages": messages, "add_generation_prompt": True,
                                           "chat_template_kwargs": {"enable_thinking": think_on}}))
    return r["prompt"]


def generate(prompt):
    """Streams one reply. Returns full text, seconds to first answer word, words read, words reused."""
    t0, text, first, timings = time.time(), "", None, {}
    with post("/completion", {"prompt": prompt, "n_predict": 700, "temperature": 0, "cache_prompt": True,
                              "stream": True}) as resp:
        for line in resp:
            line = line.decode().strip()
            if not line.startswith("data: "):
                continue
            chunk = json.loads(line[6:])
            text += chunk.get("content", "")
            timings = chunk.get("timings", timings)
            answer = text.split(CLOSE, 1)[1] if CLOSE in text else ("" if OPEN in text else text)
            if first is None and answer.strip():
                first = time.time() - t0
    return text, first, timings.get("prompt_n", 0), timings.get("cache_n", 0)


try:
    for _ in range(120):
        try:
            if json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)).get("status") == "ok":
                break
        except Exception:
            time.sleep(1)
    for name, (think_on, prefix) in SETUPS.items():
        print(f"\n=== setup {name} ===", flush=True)
        messages = [{"role": "system", "content": SYSTEM}]
        right = 0
        for msg, answers in CHAT:
            messages.append({"role": "user", "content": msg})
            prompt = render(messages, think_on) + prefix
            full, first, read, reused = generate(prompt)
            full = prefix + full
            notes = full.split(OPEN, 1)[1].split(CLOSE, 1)[0] if OPEN in full else ""
            answer = (full.split(CLOSE, 1)[1] if CLOSE in full else ("" if OPEN in full else full))
            answer = re.sub(r"<turn\|>.*", "", answer, flags=re.S).strip()
            mark = ""
            if answers:
                ok = any(a.lower() in answer.lower() for a in answers)
                right += ok
                mark = "RIGHT" if ok else "WRONG"
            shown = f"{first:5.2f} s" if first else "  none "
            print(f"notes {len(notes.split()):4d} words | first answer word {shown} | read {read:4d}, reused {reused:4d} "
                  f"| {mark:5s} | {answer[:70]!r}", flush=True)
            messages.append({"role": "assistant", "content": answer})
        print(f"hard questions right: {right}/3", flush=True)
finally:
    server.terminate()
    server.wait()
