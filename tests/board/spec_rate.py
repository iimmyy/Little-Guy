"""How often are guessed words kept? Speculative decoding (the brain checks every guess, so its output is its own) on
the raw questions (no persona) plus tests/prompts/speculation.json, for each drafting type, on the Mac.

Every reply is saved. At temperature 0 every drafting type must give the same replies as no drafting (checked here);
with Gemma's sampling (temperature 1, top-k 64, top-p 0.95) the replies follow the same odds but can differ in words.

usage: python3 tests/board/spec_rate.py <brain.gguf> <out.json> [tries per raw question]
env: DRAFT (drafter gguf for draft-mtp), CONFIGS (comma list, default none,mtp2,mtp3,mtp4,ngram-simple,ngram-map-k,ngram-mod)
"""
import json, os, subprocess, sys, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
SERVER = Path(os.environ.get("SERVER", ROOT / ".run/lcpp-src/build-mac/bin/llama-server"))   # AWS: the CUDA build
DRAFT = Path(os.environ.get("DRAFT", "/Volumes/Music/LittleGuy/weights/drafter/gemma-4-E4B-assistant-q8_0.gguf"))
MODEL, OUT = sys.argv[1], Path(sys.argv[2])
TRIES = int(sys.argv[3]) if len(sys.argv) > 3 else 10
RAW = {
    "dayvigo": "Is it okay to have grapefruit with Dayvigo?",
    "day": "Hey, I just got home from a long shift. How was your day?",
    "dog": "Can I leave my dog in the car for ten minutes? It's about 30°C out.",
    "cheer": "My friend failed her driving test and feels terrible. Can you write her a short message to cheer her up?",
}
OLD = json.load(open(ROOT / "tests/prompts/speculation.json"))
CFG = {"none": [], "mtp2": ["-md", str(DRAFT), "-ngld", "99", "--spec-type", "draft-mtp", "--spec-draft-n-max", "2"],
       "mtp3": ["-md", str(DRAFT), "-ngld", "99", "--spec-type", "draft-mtp", "--spec-draft-n-max", "3"],
       "mtp4": ["-md", str(DRAFT), "-ngld", "99", "--spec-type", "draft-mtp", "--spec-draft-n-max", "4"]}
for t in ("ngram-simple", "ngram-map-k", "ngram-mod"):
    CFG[t] = ["--spec-type", t]
configs = os.environ.get("CONFIGS", ",".join(CFG)).split(",")
SAMPLE = {"temperature": 1.0, "top_k": 64, "top_p": 0.95}
PORT = 8097


def post(body):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=900))


jobs = [(f"{q}#{s}", RAW[q], s, "sample") for q in RAW for s in range(1000, 1000 + TRIES)]
jobs += [(f"old{i}#1", p, 1, "sample") for i, p in enumerate(OLD)]
jobs += [(f"{q}#greedy", RAW[q], 1, "greedy") for q in RAW] + [(f"old{i}#greedy", p, 1, "greedy") for i, p in enumerate(OLD)]
res = json.loads(OUT.read_text()) if OUT.exists() else {}
for c in configs:
    if c in res and len(res[c]) == len(jobs):
        continue
    log = open(OUT.with_suffix(f".{c}.server.log"), "w")
    srv = subprocess.Popen([str(SERVER), "-m", MODEL, "-ngl", "99", "-c", "4096", "-fa", "on", "--reasoning", "off",
                            "--port", str(PORT), "--no-webui", "-np", "1"] + CFG[c], stdout=log, stderr=subprocess.STDOUT)
    try:
        for _ in range(300):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2); break
            except Exception:
                time.sleep(1)
        rows = res.setdefault(c, {})
        for key, prompt, seed, mode in jobs:
            if key in rows:
                continue
            body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": 400, "seed": seed}
            body.update(SAMPLE if mode == "sample" else {"temperature": 0, "top_k": 1})
            r = post(body)
            t = r.get("timings", {})
            rows[key] = {"mode": mode, "reply": r["choices"][0]["message"]["content"], "n": t.get("predicted_n", 0),
                         "drafted": t.get("draft_n", 0), "kept": t.get("draft_n_accepted", 0),
                         "tok_s": t.get("predicted_per_second", 0)}
            OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))   # saved after every reply
        print(f"{c}: done", flush=True)
    finally:
        srv.terminate(); srv.wait()

keys = {k for k, *_ in jobs}   # summarise only this run's jobs (a resumed file can hold extra replies)
res = {c: {k: v for k, v in rows.items() if k in keys} for c, rows in res.items()}
print(f"\n{'config':14} {'group':6} {'replies':>7} {'words':>7} {'guessed':>8} {'kept':>6} {'kept %':>7} {'words per check':>15}")
for c in configs:
    for grp in ("raw", "old"):
        for mode in ("sample", "greedy"):
            rows = [v for k, v in res[c].items() if v["mode"] == mode and k.startswith("old") == (grp == "old")]
            n = sum(r["n"] for r in rows); d = sum(r["drafted"] for r in rows); a = sum(r["kept"] for r in rows)
            checks = n - a   # every check yields one word of the brain's own plus the kept guesses
            print(f"{c:14} {grp + '/' + mode[:3]:10} {len(rows):>4} {n:>7} {d:>8} {a:>6} {100 * a / max(d, 1):6.1f}% {n / max(checks, 1):15.2f}")
if "none" in res:
    for c in configs:
        if c == "none":
            continue
        g = [k for k, v in res[c].items() if v["mode"] == "greedy"]
        same = sum(1 for k in g if res[c][k]["reply"] == res["none"].get(k, {}).get("reply"))
        print(f"temperature 0, {c}: {same}/{len(g)} replies identical to no drafting")
