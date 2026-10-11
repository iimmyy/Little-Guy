"""Head to head on the board: the 4-bit and 2-bit brains in the exact same setup, speed and behavior together.

Same llama-server build and flags (CPU, 4 threads, 4096 context, thinking off), same persona, same four test
questions, same sampling (temperature 0.8, top-k 64, top-p 0.95) and same seeds. Every block starts with the
hottest core at or below 45 C, and the brains alternate so heat can't favor one.
  no assistant:   N_SPEED replies per question, for talking speed
  assistant (3):  N_BEHAVIOR replies per question, for talking speed and the behavior checks
                  (the assistant only changes speed: every word is still chosen by the main brain)
Reading speed: the first 3 replies of each block re-read the whole persona + question (prompt cache off).
The Mac only sends questions through an ssh tunnel; all timings are measured on the board.

One block per run, so the fan routine can happen between blocks (fan on 1 minute before, 5 minutes of
cooldown after, which the script waits out and logs):
  python head_to_head.py block 4-bit none        # N_SPEED replies per question
  python head_to_head.py block 2-bit assistant   # N_BEHAVIOR replies per question
  python head_to_head.py report                  # the tables, once all four blocks are in
"""
import json, os, re, statistics, subprocess, sys, time, urllib.request
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "behavior"))
from common import BOARD_SSH, RESULTS, load_persona, persona_path
from scoring import QUESTIONS, checks

N_BEHAVIOR, N_SPEED = int(os.environ.get("H2H_N", "20")), 5
COOLDOWN = FAN_LEAD = 0   # runs start once the CPU is back at COOL_TO, instead of fixed waits
PORT = 8096
BRAINS = {"4-bit": "qat-Q4_0-f16proj", "2-bit": "qat-Q2_K-vip4-f16proj", "C": "clean-Q2_K-C", "C2": "clean-Q2_K-C2",
          "D": "clean-Q2_K-D"}
# OVEN=new runs the upgraded oven staged in ~/oven-ultra (fused few-word 2-bit path, vector q8_K packing);
# OVEN=gpu runs the same server on the board-built libraries with the Vulkan engine, brain and drafter on the GPU
OVEN = os.environ.get("OVEN", "old")
OVEN_DIR = "~/oven-ultra" if OVEN in ("new", "gpu") else "~/llama"
OVEN_LIB = {"new": "lib:$HOME/llama/lib", "gpu": "$HOME/llama-gpu/src/build/bin:lib"}.get(OVEN, "lib")
DEVICE = "--device Vulkan0 -ngl 99" if OVEN == "gpu" else "--device none -ngl 0"
DEVICE_DRAFT = "--device-draft Vulkan0 -ngld 99" if OVEN == "gpu" else "-ngld 0"
DRAFT = "gemma-4-E4B-assistant-masked-q8_0"
PERSONA = load_persona()
SAMPLING = {"max_tokens": 90, "temperature": 0.8, "top_k": 64, "top_p": 0.95}
ZONES = "1 2 3 4 5 6 7 8 9 10 11 12 13 14"      # every CPU sensor, for the peak during a run
APP_ZONES = "1 2 3 4 7 8 9 10 11 12 13 14"        # the menu bar app's CPU temperature, for cooling to a target
COOL_TO = int(os.environ.get("COOL_TO", "24"))


def board(cmd, **kw):
    return subprocess.run(["ssh", BOARD_SSH, cmd], capture_output=True, text=True, **kw).stdout.strip()


def hottest(zones=APP_ZONES):
    return int(board(f"m=0; for z in {zones}; do v=$(cat /sys/class/thermal/thermal_zone$z/temp); "
                     f"[ $v -gt $m ] && m=$v; done; echo $((m/1000))"))


def cool():   # wait until the app's CPU temperature is at or below COOL_TO
    t = hottest()
    while t > COOL_TO:
        print(f"   cooling: {t} C (target {COOL_TO} C)", flush=True); time.sleep(10); t = hottest()
    return t


def post(body):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))


def run_block(brain, assistant, n):
    flags = (f"-m ~/models/{BRAINS[brain]}.gguf -c 4096 --port {PORT} --host 127.0.0.1 --no-webui --reasoning off "
             f"{DEVICE} -t 4")
    if assistant:
        flags += f" -md ~/models/{DRAFT}.gguf {DEVICE_DRAFT} -td 4 --spec-type draft-mtp --spec-draft-n-max 3"
    board("pkill -x llama-server; sleep 1")
    server = subprocess.Popen(["ssh", BOARD_SSH, f"cd {OVEN_DIR} && LD_LIBRARY_PATH={OVEN_LIB} GGML_HEXAGON_NDEV=0 "
                               f"exec bin/llama-server {flags} > /tmp/head_to_head_server.log 2>&1"],
                              stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    tunnel = subprocess.Popen(["ssh", "-N", "-L", f"{PORT}:127.0.0.1:{PORT}", BOARD_SSH])
    try:
        for _ in range(180):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2); break
            except Exception:
                time.sleep(1)
        else:
            sys.exit("server never became ready:\n" + board("tail -20 /tmp/head_to_head_server.log"))
        start_temp = cool()
        since = board("date -u '+%Y-%m-%d %H:%M:%S'")
        sampler = subprocess.Popen(["ssh", BOARD_SSH, f"while true; do m=0; for z in {ZONES}; do "
                                    f"v=$(cat /sys/class/thermal/thermal_zone$z/temp); [ $v -gt $m ] && m=$v; done; "
                                    f"echo $((m/1000)); sleep 1; done"], stdout=subprocess.PIPE, text=True)
        reading, gen_tok, gen_ms, drafted, accepted, replies, answer_s = [], 0, 0.0, 0, 0, defaultdict(list), []
        for key, q in QUESTIONS.items():
            for seed in range(n):
                fresh = len(reading) < 3
                r = post({"messages": [{"role": "system", "content": PERSONA}, {"role": "user", "content": q}],
                          "seed": 1000 + seed, "cache_prompt": not fresh, **SAMPLING})
                t = r["timings"]
                if fresh:
                    reading.append(t["prompt_per_second"])
                gen_tok += t["predicted_n"]; gen_ms += t["predicted_ms"]
                answer_s.append((t["prompt_ms"] + t["predicted_ms"]) / 1000)   # what you wait for: reading + talking
                drafted += t.get("draft_n", 0); accepted += t.get("draft_n_accepted", 0)
                replies[key].append(r["choices"][0]["message"]["content"].strip())
        sampler.terminate()
        temps = [int(x) for x in sampler.communicate()[0].split()]
        slowdowns = int(board(f"journalctl -t thermal-guard --since '{since}' --no-pager | grep -c slowing") or 0)
        return {"talking": 1000 * gen_tok / gen_ms, "reading": statistics.median(reading),
                "accepted": accepted / drafted if drafted else None, "start": start_temp, "peak": max(temps),
                "mean_temp": statistics.mean(temps), "answer_s": statistics.mean(answer_s), "oven": OVEN_DIR, "seconds": len(temps), "slowdowns": slowdowns,
                "clocks": board("for p in 0 4 7; do cat /sys/devices/system/cpu/cpufreq/policy$p/scaling_max_freq; done | tr '\\n' ' '"),
                "replies": dict(replies)}
    finally:
        tunnel.terminate()
        board("pkill -x llama-server")
        server.terminate()


def block(brain, mode):
    assistant = mode == "assistant"
    n = N_BEHAVIOR if assistant else N_SPEED
    label = f"{brain}, {'assistant' if assistant else 'no assistant'}"
    print(f"== {label}: {n} replies per question, persona {persona_path().name}, temperature 0.8", flush=True)
    for left in range(FAN_LEAD, 0, -10):            # the fan runs this long before the test starts
        print(f"   fan lead-in: {left} s to go, {hottest()} C", flush=True)
        time.sleep(10)
    b = run_block(brain, assistant, n)
    acc = f", assistant guesses kept {b['accepted']:.0%}" if b["accepted"] is not None else ""
    print(f"   time per answer {b['answer_s']:.2f} s | talking {b['talking']:.2f} words/s, reading {b['reading']:.1f} words/s{acc}", flush=True)
    print(f"   heat over {b['seconds']} s: start {b['start']} C, average {b['mean_temp']:.1f} C, peak {b['peak']} C, "
          f"heat-guard slowdowns {b['slowdowns']}, clock caps {b['clocks']}", flush=True)
    path = RESULTS / os.environ.get("H2H_OUT", "head_to_head.json")
    out = json.load(open(path)) if path.exists() else {}
    out[label] = b
    json.dump(out, open(path, "w"), indent=1, ensure_ascii=False)
    for m in range(COOLDOWN // 60):
        time.sleep(60)
        print(f"   cooldown {m + 1}/{COOLDOWN // 60} min: {hottest()} C", flush=True)


def report():
    out = json.load(open(RESULTS / "head_to_head.json"))
    base = out["4-bit, no assistant"]["talking"]
    print(f"{'':26s} {'talking':>9s} {'vs 4-bit alone':>15s} {'reading':>9s} {'start/peak':>11s}")
    for label, b in out.items():
        print(f"{label:26s} {b['talking']:9.2f} {b['talking'] / base:14.2f}x {b['reading']:9.1f} {b['start']:>6d}/{b['peak']} C")
    behav = {k: v for k, v in out.items() if "no assistant" not in k}
    n = len(next(iter(behav.values()))["replies"]["day"])
    print(f"\nbehavior, {n} replies per question (with the assistant):")
    print(f"{'check':34s} " + "  ".join(f"{k.split(',')[0]:>8s}" for k in behav))
    for key in QUESTIONS:
        for c in checks(key, ""):
            print(f"{key + ': ' + c:34s} " + "  ".join(
                f"{sum(checks(key, t)[c] for t in b['replies'][key]):>5d}/{n}" for b in behav.values()))
    words = {k: statistics.mean(len(re.sub(r"\[[^\]]*\]", "", t).split()) for rs in b["replies"].values() for t in rs)
             for k, b in behav.items()}
    print(f"{'average words per reply':34s} " + "  ".join(f"{w:>8.1f}" for w in words.values()))


if __name__ == "__main__":
    if sys.argv[1] == "block":
        if len(sys.argv) > 4:                       # setup check: block <brain> <mode> <n>, no cooldown, not saved
            N_BEHAVIOR = N_SPEED = int(sys.argv[4]); COOLDOWN = FAN_LEAD = 0; RESULTS = Path("/tmp")
        block(sys.argv[2], sys.argv[3])
    else:
        report()
