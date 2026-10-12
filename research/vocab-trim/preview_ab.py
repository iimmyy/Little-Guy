"""Word-for-word check of the dictionary preview inside the brain: same brain file, same prompts, same dice,
once with the preview off (LG_DICT_PREVIEW=0) and once on (default shortlist, or LG_DICT_PREVIEW=<n>).
CPU only, 4 threads, like the board. Greedy replies first, then his normal dice (temp 0.8, top-k 64, top-p 0.95).

usage: preview_ab.py BRAIN_GGUF [SHORTLIST]
"""
import json, os, subprocess, sys, time, urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests/behavior"))
from common import SERVER, LOGS, load_persona
from scoring import QUESTIONS

BRAIN = sys.argv[1]
SHORT = sys.argv[2] if len(sys.argv) > 2 else "1024"
PORT = 8091
EXTRA = ["Tell me a short story about a robot who learns to bake bread.",
         "Explain how a bike gear works, simply.",
         "Was ist der Unterschied zwischen Wetter und Klima?",
         "Give me three ideas for a rainy Sunday."]
PROMPTS = list(QUESTIONS.values()) + EXTRA
SETTINGS = {"greedy": {"temperature": 0}, "his dice": {"temperature": 0.8, "top_k": 64, "top_p": 0.95, "seed": 7}}


def post(path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=1800))


def run(shortlist):
    env = dict(os.environ, LG_DICT_PREVIEW=shortlist)
    srv = subprocess.Popen([str(SERVER), "-m", BRAIN, "-ngl", "0", "-t", "4", "-c", "4096", "--port", str(PORT), "--no-webui",
                            "--reasoning", "off"], env=env, stdout=open(LOGS / f"preview_ab_{shortlist}.log", "w"), stderr=subprocess.STDOUT)
    try:
        for _ in range(300):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2); break
            except Exception:
                time.sleep(1)
        out, speed = {}, []
        for name, s in SETTINGS.items():
            for q in PROMPTS:
                prompt = post("/apply-template", {"messages": [{"role": "system", "content": load_persona()}, {"role": "user", "content": q}],
                                                  "add_generation_prompt": True, "chat_template_kwargs": {"enable_thinking": False}})["prompt"]
                r = post("/completion", {"prompt": prompt, "n_predict": 120, "cache_prompt": False, **s})
                out[(name, q)] = r["content"]
                speed.append(r["timings"]["predicted_per_second"])
        return out, sum(speed) / len(speed)
    finally:
        srv.terminate(); srv.wait()


off, s_off = run("0")
print(f"preview OFF done ({s_off:.1f} words/s on the Mac CPU)", flush=True)
on, s_on = run(SHORT)
print(f"preview ON  done ({s_on:.1f} words/s on the Mac CPU), shortlist {SHORT}\n", flush=True)

same = 0
for key in off:
    a, b = off[key], on[key]
    if a == b:
        same += 1
        print(f"SAME  [{key[0]}] {key[1][:50]}")
    else:
        i = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        print(f"DIFF  [{key[0]}] {key[1][:50]}\n      off: ...{a[max(0, i - 40):i + 40]!r}\n      on:  ...{b[max(0, i - 40):i + 40]!r}")
print(f"\n{same} of {len(off)} replies identical, word for word")
print(f"Mac CPU speed: {s_off:.2f} -> {s_on:.2f} words/s ({s_on / s_off - 1:+.1%}); the board's number comes later")
