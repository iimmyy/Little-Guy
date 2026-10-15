"""His behavior test on the board's NPU vs its CPU: the same brain, persona, questions, seeds and dice as
tests/behavior/persona_ab.py, scored with the same checks. The board runs llama-server (fork engine in ~/preview2);
this script sends the questions over the network.

usage: npu_behavior.py [N] [brain on the board] [devices]   e.g. npu_behavior.py 40 mix-21-preview NPU
devices: comma-separated subset of CPU,NPU (default both)
"""
import json, os, subprocess, sys, time, urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "behavior"))
from common import BOARD_SSH, load_persona
from scoring import QUESTIONS, checks

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
BRAIN = sys.argv[2] if len(sys.argv) > 2 else "mix-21-preview"
HOST = BOARD_SSH.split("@")[-1] or "q6a"
PORT = 8095
SEED0 = int(os.environ.get("SEED0", "1000"))   # a different SEED0 replicates a result on fresh dice rolls
SAMPLING = {"max_tokens": 90, "temperature": 0.8, "top_k": 64, "top_p": 0.95}
ENV = "cd ~/preview2 && LD_LIBRARY_PATH=lib ADSP_LIBRARY_PATH=$HOME/preview2/lib GGML_HEXAGON_USE_HMX=0 GGML_HEXAGON_OPBATCH=4096"
DEVICES = {"CPU": "GGML_HEXAGON_NDEV=0 bin/llama-server --device none -ngl 0",
           "NPU": "bin/llama-server --device HTP0 -ngl 99"}


def board_ip():
    return subprocess.run(["ssh", "-G", HOST], capture_output=True, text=True).stdout.split("hostname ")[1].split()[0]


def run(device):
    cmd = (f"{ENV} {DEVICES[device]} -m ~/models/{BRAIN}.gguf -c 4096 -t 4 -C 0xf0 --cpu-strict 1 -fa on "
           f"--reasoning off --host 0.0.0.0 --port {PORT} --no-webui > /tmp/npu_behavior_srv.log 2>&1")
    srv = subprocess.Popen(["ssh", HOST, cmd])
    url = f"http://{board_ip()}:{PORT}"
    try:
        for _ in range(600):
            try:
                if b"ok" in urllib.request.urlopen(url + "/health", timeout=2).read():
                    break
            except Exception:
                time.sleep(1)
        out, speed = {}, []
        for key, q in QUESTIONS.items():
            out[key] = []
            for i in range(N):
                body = {"messages": [{"role": "system", "content": load_persona()}, {"role": "user", "content": q}],
                        "seed": SEED0 + i, **SAMPLING}
                req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                             {"Content-Type": "application/json"})
                r = json.load(urllib.request.urlopen(req, timeout=900))
                out[key].append(r["choices"][0]["message"]["content"])
                speed.append(r["timings"]["predicted_per_second"])
            print(f"  {device} {key}: done", flush=True)
        return out, sum(speed) / len(speed)
    finally:
        subprocess.run(["ssh", HOST, "pkill -x llama-server"])
        srv.wait()


ONLY = sys.argv[3].split(",") if len(sys.argv) > 3 else list(DEVICES)
results = {d: run(d) for d in DEVICES if d in ONLY}
print(f"\n{BRAIN}, {N} replies per question, seeds {SEED0}..{SEED0 + N - 1}\n")
rows = {}
for d, (out, _) in results.items():
    for key, replies in out.items():
        for r in replies:
            for name, hit in checks(key, r).items():
                rows.setdefault(f"{key}: {name}", {}).setdefault(d, 0)
                rows[f"{key}: {name}"][d] += bool(hit)
print(f"{'check':34s} " + "  ".join(f"{d:>6s}" for d in results))
for name, per in rows.items():
    print(f"{name:34s} " + "  ".join(f"{per.get(d, 0):>3d}/{N}" for d in results))
if len(results) == 2:
    same = sum(a == b for k in QUESTIONS for a, b in zip(results["CPU"][0][k], results["NPU"][0][k]))
    print(f"\nword-for-word identical replies: {same}/{N * len(QUESTIONS)}")
print("talking speed (server): " + "  ".join(f"{d} {s:.2f} words/s" for d, (_, s) in results.items()))
json.dump({d: o for d, (o, _) in results.items()}, open(Path("/tmp") / f"npu_behavior_{BRAIN}.json", "w"), indent=1)
