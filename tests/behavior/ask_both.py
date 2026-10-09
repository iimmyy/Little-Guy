"""Ask Little Guy one question with thinking off and with trick A (we open the notes). Mac only."""
import json, subprocess, sys, time, urllib.request

RUN = "/Users/immyyyyy/Documents/LittleGuy/.run"
PORT = 8093
OPEN, CLOSE = "<|channel>thought\n", "<channel|>"
import os
PERSONA = os.environ.get("PERSONA")  # a file; lines starting with # are skipped
SYSTEM = "\n".join(l for l in open(PERSONA).read().splitlines() if l.strip() and not l.startswith("#")) if PERSONA else os.environ.get("SYSTEM", "You are Little Guy, a small friendly robot who lives with Immy. Keep replies short and warm.")
question = sys.argv[1]
server = subprocess.Popen([f"{RUN}/lcpp-src/build-mac/bin/llama-server", "-m", os.environ.get("MODEL", f"{RUN}/models/gemma-4-E4B_q4_0-it.gguf"),
                           "-ngl", "99", "-c", "8192", "-np", "1", "--special", "--port", str(PORT), "--no-webui"],
                          stdout=open(f"{RUN}/spec-test/ask_both_server.log", "w"), stderr=subprocess.STDOUT)


def feed(*lines):
    """Append plain heard/said lines to Little Guy's feed on the board (shown in the menu bar app)."""
    stamp = time.strftime("%H:%M:%S")
    text = "".join(f"{stamp}  {l}\n" for l in lines)
    subprocess.run(["ssh", "-o", "ConnectTimeout=3", "q6a", "cat >> ~/spec/live.log"], input=text.encode(),
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def post(path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))


try:
    for _ in range(120):
        try:
            if post("/apply-template", {"messages": [{"role": "user", "content": "hi"}]}):
                break
        except Exception:
            time.sleep(1)
    messages = ([{"role": "system", "content": SYSTEM}] if SYSTEM else []) + [{"role": "user", "content": question}]
    prompt = post("/apply-template", {"messages": messages, "add_generation_prompt": True,
                                      "chat_template_kwargs": {"enable_thinking": False}})["prompt"]
    feed("", f"(Mac test{', persona ' + os.path.basename(PERSONA) if PERSONA else ''})", f"heard:  {question}")
    for name, prefix in (("THINKING OFF", ""), ("TRICK A (thinks first)", OPEN)):
        r = post("/completion", {"prompt": prompt + prefix, "n_predict": 1200, "temperature": 0, "cache_prompt": False})
        full = prefix + r["content"]
        notes = full.split(OPEN, 1)[1].split(CLOSE, 1)[0] if OPEN in full else ""
        answer = full.split(CLOSE, 1)[1] if CLOSE in full else full
        answer = answer.split("<turn|>")[0].strip()
        print(f"\n===== {name} | notes: {len(notes.split())} words =====")
        if notes:
            print("--- his notes ---\n" + notes.strip() + "\n--- his answer ---")
        print(answer)
        feed(f"said:   {answer.replace(chr(10), ' ')}" + (f"   (thought first: {len(notes.split())} words)" if notes else "   (no thinking)"))
finally:
    server.terminate()
    server.wait()
