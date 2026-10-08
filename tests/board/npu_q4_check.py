"""Is llama.cpp's 4-bit NPU matmul correct on v68? Same prompt, temperature 0:
CPU only (answer key) vs CPU + NPU doing only the 4-bit MUL_MATs of a few layers.

Usage on the board: python3 npu_q4_check.py [n_layers_on_npu]
"""
import json, os, subprocess, sys, time, urllib.request

HOME = os.path.expanduser("~")
PORT = 8091
N_NPU = sys.argv[1] if len(sys.argv) > 1 else "4"
PROMPT = "<|turn>user\nIn two sentences, why is the sky blue?<turn|>\n<|turn>model\n"
base = dict(os.environ, LD_LIBRARY_PATH=f"{HOME}/llama/lib", ADSP_LIBRARY_PATH=f"{HOME}/llama/lib")
SETUPS = {
    "CPU only (answer key)": (dict(base, GGML_HEXAGON_NDEV="0"), ["--device", "none", "-ngl", "0"]),
    f"CPU + NPU ({N_NPU} layers, 4-bit matmuls only)": (
        dict(base, GGML_HEXAGON_OPFILTER="(?!(MUL_MAT|NONE|VIEW|RESHAPE|PERMUTE|TRANSPOSE)$).*"),  # refuse all math except MUL_MAT
        ["--device", "HTP0", "-ngl", N_NPU, "-nkvo"]),       # KV cache stays on the CPU
}


def feed(line):
    with open(f"{HOME}/spec/live.log", "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')}  {line}\n")


results = {}
for name, (env, args) in SETUPS.items():
    log = f"{HOME}/spec/npu_q4_{len(results)}.log"
    server = subprocess.Popen([f"{HOME}/llama/bin/llama-server", "-m", f"{HOME}/models/gemma-4-E4B_q4_0-it.gguf",
                               "-t", "4", "-c", "1024", "--port", str(PORT), "--no-webui", *args],
                              env=env, stdout=open(log, "w"), stderr=subprocess.STDOUT)
    try:
        for _ in range(300):
            if server.poll() is not None:
                break
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
                break
            except Exception:
                time.sleep(1)
        if server.poll() is not None:
            text, speed = f"SERVER DIED (exit {server.returncode}), see {log}", 0
        else:
            req = urllib.request.Request(f"http://127.0.0.1:{PORT}/completion", json.dumps(
                {"prompt": PROMPT, "n_predict": 32, "temperature": 0, "cache_prompt": False}).encode(),
                {"Content-Type": "application/json"})
            try:
                r = json.load(urllib.request.urlopen(req, timeout=600))
                text, speed = r["content"].strip(), r["timings"]["predicted_per_second"]
            except Exception as e:
                text, speed = f"REQUEST FAILED ({e}), see {log}", 0
    finally:
        server.terminate()
        server.wait()
    results[name] = text
    print(f"=== {name}  ({speed:.1f} tok/s)\n{text}\n", flush=True)
    feed(f"{name}: {text[:120]}")

a, b = results.values()
same = sum(1 for x, y in zip(a.split(), b.split()) if x == y)
print(f"matching words, in order: {same} of {len(a.split())}")
feed(f"NPU 4-bit check: {same}/{len(a.split())} words match the CPU")
