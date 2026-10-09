"""How much do the trays (weight repacking) help on this board? Same model, with vs without repack.
Reading = a ~256-token prompt processed in one batch; writing = 64 generated tokens. CPU only, 4 threads.
Each run starts with the hottest core at or below 45 C. Usage on the board: python3 tray_effect.py model.gguf [...]
"""
import json, os, subprocess, sys, time, urllib.request

HOME = os.path.expanduser("~")
PORT = 8084
env = dict(os.environ, LD_LIBRARY_PATH=f"{HOME}/llama/lib", ADSP_LIBRARY_PATH=f"{HOME}/llama/lib", GGML_HEXAGON_NDEV="0")
PROMPT = ("Little Guy is a small robot who lives with Immy. " * 26).strip()   # ~260 tokens


def hottest():
    return max(int(open(f"/sys/class/thermal/thermal_zone{z}/temp").read()) for z in (1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14)) // 1000


def feed(line):
    with open(f"{HOME}/spec/live.log", "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')}  {line}\n")
    print(line, flush=True)


feed("=== tray effect: same model with vs without repack ===")
for model in sys.argv[1:]:
    for trays in (True, False):
        while hottest() > 45:
            time.sleep(10)
        args = [f"{HOME}/llama/bin/llama-server", "-m", f"{HOME}/models/{model}", "-t", "4", "-c", "1024", "--port", str(PORT),
                "--no-webui", "--device", "none", "-ngl", "0"] + ([] if trays else ["--no-repack"])
        srv = subprocess.Popen(args, env=env, stdout=open(f"{HOME}/spec/tray_effect_server.log", "w"), stderr=subprocess.STDOUT)
        try:
            for _ in range(240):
                try:
                    urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2); break
                except Exception:
                    time.sleep(1)
            req = urllib.request.Request(f"http://127.0.0.1:{PORT}/completion", json.dumps(
                {"prompt": PROMPT, "n_predict": 64, "temperature": 0, "cache_prompt": False}).encode(), {"Content-Type": "application/json"})
            t = json.load(urllib.request.urlopen(req, timeout=900))["timings"]
            feed(f"{model:24s} trays {'ON ' if trays else 'OFF'}  reading {t['prompt_per_second']:6.1f} tok/s ({t['prompt_n']} tokens)"
                 f"   writing {t['predicted_per_second']:5.2f} tok/s   peak {hottest()} C")
        finally:
            srv.terminate(); srv.wait()
