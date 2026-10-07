"""Generation speed vs temperature, with the clustered drafter (3 guesses), on the board CPU.

Usage on the board: python3 temp_sweep.py
"""
import json, os, subprocess, time, urllib.request

HOME = os.path.expanduser("~")
PORT = 8097
env = dict(os.environ, LD_LIBRARY_PATH=f"{HOME}/llama/lib", ADSP_LIBRARY_PATH=f"{HOME}/llama/lib", GGML_HEXAGON_NDEV="0")
server = subprocess.Popen([f"{HOME}/llama/bin/llama-server", "-m", f"{HOME}/models/gemma-4-E4B_q4_0-it.gguf",
                           "-md", f"{HOME}/models/gemma-4-E4B-assistant-masked-q8_0.gguf", "--spec-type", "draft-mtp",
                           "--spec-draft-n-max", "3", "-ngl", "0", "-ngld", "0", "--device", "none", "-t", "4", "-td", "4",
                           "-c", "4096", "--port", str(PORT), "--no-webui"],
                          env=env, stdout=open(f"{HOME}/spec/temp_server.log", "w"), stderr=subprocess.STDOUT)


def post(body):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=900))


def cpu_temp():
    return int(open("/sys/class/thermal/thermal_zone1/temp").read()) // 1000


try:
    for _ in range(240):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
            break
        except Exception:
            time.sleep(1)
    prompts = json.load(open(f"{HOME}/spec/spec-test/prompts.json"))[:4]
    for temp in (0.0, 0.5, 0.7, 1.0):
        while cpu_temp() > 45:
            time.sleep(5)
        drafted = accepted = generated = 0
        seconds = 0.0
        for p in prompts:
            t = post({"messages": [{"role": "user", "content": p}], "max_tokens": 96, "seed": 1,
                      "temperature": temp, "top_k": 64, "top_p": 0.95})["timings"]
            drafted += t.get("draft_n", 0)
            accepted += t.get("draft_n_accepted", 0)
            generated += t["predicted_n"]
            seconds += t["predicted_ms"] / 1000
        print(f"temp {temp:.1f}: {generated / seconds:5.2f} tok/s | guesses accepted {accepted}/{drafted} "
              f"= {accepted / max(drafted, 1):5.1%}", flush=True)
finally:
    server.terminate()
    server.wait()
