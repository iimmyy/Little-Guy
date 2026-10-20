"""Persona A/B on both brains, run on the Mac's CPU like the board (4 threads, no GPU, low priority).

Each brain answers the four test questions N times with each persona, same seeds and talking temperature, and
every reply is scored with the shared checks plus two outro meters:
  outro:     text after a blank line, a second beat added once the answer is done
  asks back: the reply contains a question

Usage: python persona_ab.py <persona B> [N]      e.g. python persona_ab.py history/outro.private.txt 20
(persona A is your current persona; give "-" as persona B to run only your current persona)
Brains: AB_BRAINS="name=path,name=path" picks others (paths relative to .run/weights), default 4-bit and 2-bit.
Seeds: AB_SEED0 (default 1000) sets the first seed, so a rerun with fresh dice can check whether a difference holds.
"""
import json, os, re, statistics, subprocess, sys, time, urllib.request
from collections import defaultdict

from common import BRAINS, LOGS, RESULTS, RUN, SERVER, load_persona, persona_path
from scoring import QUESTIONS, checks

B_NAME = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 20
PORT = 8086
BRAIN_FILES = ({k: RUN / "weights" / v for k, v in (x.split("=") for x in os.environ["AB_BRAINS"].split(","))}
               if os.environ.get("AB_BRAINS") else
               {"4-bit": BRAINS / "qat-Q4_0-f16proj.gguf", "2-bit": BRAINS / "qat-Q2_K-vip4-f16proj.gguf"})
PERSONAS = {"current": load_persona()} | ({} if B_NAME == "-" else {"new": load_persona(B_NAME)})
SEED0 = int(os.environ.get("AB_SEED0", "1000"))
SAMPLING = {"max_tokens": 90, "temperature": 0.8, "top_k": 64, "top_p": 0.95}


def post(body):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))


def spoken(t):
    return re.sub(r"\[[^\]]*\]", "", t)


results = {}
t0 = time.time()
for brain, path in BRAIN_FILES.items():
    # AB_GPU=1: on the Mac's GPU instead (the board is gone; behavior is what's compared, not speed)
    where = ["-ngl", "99"] if os.environ.get("AB_GPU") else ["--device", "none", "-ngl", "0", "-t", "4"]
    # AB_LORA=adapter.gguf AB_LORA_FOR=name,...: those brains load what training added (tools/train/choices.py)
    if brain in os.environ.get("AB_LORA_FOR", "").split(","):
        where += ["--lora", os.environ["AB_LORA"]]
    srv = subprocess.Popen(["nice", "-n", "15", SERVER, "-m", path, "-c", "4096", "--port", str(PORT), "--no-webui",
                            "--reasoning", "off"] + where,
                           stdout=open(LOGS / "persona_ab_server.log", "w"), stderr=subprocess.STDOUT)
    try:
        for _ in range(180):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2); break
            except Exception:
                time.sleep(1)
        for pname, persona in PERSONAS.items():
            replies, tokens = defaultdict(list), []
            for key, q in QUESTIONS.items():
                for seed in range(N):
                    r = post({"messages": [{"role": "system", "content": persona}, {"role": "user", "content": q}],
                              "seed": SEED0 + seed, **SAMPLING})
                    replies[key].append(r["choices"][0]["message"]["content"].strip())
                    tokens.append(r["usage"]["completion_tokens"])
            results[(brain, pname)] = (dict(replies), tokens)
            print(f"{brain}, {pname} persona: done [{time.time() - t0:.0f} s]", flush=True)
    finally:
        srv.terminate(); srv.wait()

json.dump({f"{b}, {p}": r for (b, p), (r, _) in results.items()},
          open(RESULTS / f"persona_ab_{'brains' if B_NAME == '-' else persona_path(B_NAME).name.split('.')[0]}.json", "w"), indent=1, ensure_ascii=False)
cols = list(results)
print(f"\npersona B: {B_NAME} | {N} replies per question | {'Mac GPU' if os.environ.get('AB_GPU') else 'Mac CPU, 4 threads'}, temperature 0.8\n")
print(f"{'check':30s} " + "  ".join(f"{b + ' ' + p:>15s}" for b, p in cols))
for key in QUESTIONS:
    for c in checks(key, ""):
        print(f"{key + ': ' + c:30s} " + "  ".join(
            f"{sum(checks(key, t)[c] for t in results[k][0][key]):>12d}/{N}" for k in cols))
all_replies = {k: [t for rs in results[k][0].values() for t in rs] for k in cols}
rows = {"outro (after a blank line)": lambda t: bool(re.search(r"\n\s*\n\s*\S", t)),
        "asks back": lambda t: "?" in spoken(t)}
for name, f in rows.items():
    print(f"{name:30s} " + "  ".join(f"{sum(map(f, all_replies[k])):>12d}/{4 * N}" for k in cols))
print(f"{'average words per reply':30s} " + "  ".join(
    f"{statistics.mean(len(spoken(t).split()) for t in all_replies[k]):>15.1f}" for k in cols))
print(f"{'average tokens per reply':30s} " + "  ".join(f"{statistics.mean(results[k][1]):>15.1f}" for k in cols))
