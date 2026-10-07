"""Time until the first visible answer word, with Gemma 4 thinking on (default) vs --reasoning off.

Usage on the board: python3 think_test.py
"""
import json, os, subprocess, time, urllib.request

HOME = os.path.expanduser("~")
PORT = 8098
env = dict(os.environ, LD_LIBRARY_PATH=f"{HOME}/llama/lib", ADSP_LIBRARY_PATH=f"{HOME}/llama/lib", GGML_HEXAGON_NDEV="0")
PROMPTS = ["Little Guy, I just got home from a long shift at the pharmacy. How was your day?",
           "Can you remind me what we talked about this morning? I mentioned a dentist appointment at 3.",
           "Tell me a short joke about robots."]


def cpu_temp():
    return int(open("/sys/class/thermal/thermal_zone1/temp").read()) // 1000


def first_answer_word(prompt):
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": 400, "stream": True,
            "temperature": 1.0, "top_k": 64, "top_p": 0.95, "seed": 1}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    thought_chars = 0
    with urllib.request.urlopen(req, timeout=900) as resp:
        for line in resp:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            delta = json.loads(line[6:])["choices"][0].get("delta", {})
            thought_chars += len(delta.get("reasoning_content") or "")
            if (delta.get("content") or "").strip():
                return time.time() - t0, thought_chars
    return None, thought_chars


for mode in ("default", "off"):
    while cpu_temp() > 45:
        time.sleep(5)
    args = [f"{HOME}/llama/bin/llama-server", "-m", f"{HOME}/models/gemma-4-E4B_q4_0-it.gguf",
            "-md", f"{HOME}/models/gemma-4-E4B-assistant-masked-q8_0.gguf", "--spec-type", "draft-mtp",
            "--spec-draft-n-max", "3", "-ngl", "0", "-ngld", "0", "--device", "none", "-t", "4", "-td", "4",
            "-c", "4096", "--port", str(PORT), "--no-webui"]
    if mode == "off":
        args += ["--reasoning", "off"]
    server = subprocess.Popen(args, env=env, stdout=open(f"{HOME}/spec/think_{mode}.log", "w"), stderr=subprocess.STDOUT)
    try:
        for _ in range(240):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
                break
            except Exception:
                time.sleep(1)
        for p in PROMPTS:
            secs, thought = first_answer_word(p)
            shown = f"{secs:5.1f} s" if secs is not None else "no answer within 400 tokens"
            print(f"thinking {mode:7s}: first answer word after {shown} (hidden thinking: {thought} chars)", flush=True)
    finally:
        server.terminate()
        server.wait()
