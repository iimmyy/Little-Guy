"""Measures how often the Gemma 4 drafter's guesses are accepted, on Little Guy style prompts.

Usage: python3 run_spec.py <draft_n_max> [port]
Starts llama-server with E4B Q4_0 + the E4B assistant (draft-mtp), sends every prompt
at temperature 0 and at Gemma's default sampling, and prints accept rates.
"""
import json, subprocess, sys, time, urllib.request
from pathlib import Path

import os
RUN = Path(os.environ.get("RUN", "/Users/immyyyyy/Documents/LittleGuy/.run"))
SERVER = Path(os.environ.get("SERVER", RUN / "lcpp-src/build-mac/bin/llama-server"))
MAIN = Path(os.environ.get("MAIN", RUN / "models/gemma-4-E4B_q4_0-it.gguf"))
NGL = os.environ.get("NGL", "99")
EXTRA = os.environ.get("EXTRA", "").split()
N_PROMPTS = int(os.environ.get("N_PROMPTS", "8"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "160"))
DRAFT = Path(os.environ.get("DRAFT", RUN / "e4b-drafter/gemma-4-E4B-assistant-q8_0.gguf"))
n_max = int(sys.argv[1]) if len(sys.argv) > 1 else 3
port = int(sys.argv[2]) if len(sys.argv) > 2 else 8090
use_draft = n_max > 0

args = [str(SERVER), "-m", str(MAIN), "-ngl", NGL, "-c", "4096", "--port", str(port), "--no-webui"]
if use_draft:
    args += ["-md", str(DRAFT), "-ngld", NGL, "--spec-type", "draft-mtp", "--spec-draft-n-max", str(n_max)]
args += EXTRA
log = open(RUN / f"spec-test/server_n{n_max}.log", "w")
server = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)


def post(path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))


try:
    for _ in range(240):
        try:
            if json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)).get("status") == "ok":
                break
        except Exception:
            time.sleep(1)
    else:
        sys.exit("server never became ready, see log")

    prompts = json.load(open(RUN / "spec-test/prompts.json"))[:N_PROMPTS]
    for label, sampling in (("temp 0", {"temperature": 0}),
                            ("temp 1.0", {"temperature": 1.0, "top_k": 64, "top_p": 0.95})):
        drafted = accepted = generated = 0
        speeds = []
        for p in prompts:
            r = post("/v1/chat/completions", {"messages": [{"role": "user", "content": p}],
                                              "max_tokens": MAX_TOKENS, "seed": 1, **sampling})
            t = r.get("timings", {})
            drafted += t.get("draft_n", 0)
            accepted += t.get("draft_n_accepted", 0)
            generated += t.get("predicted_n", 0)
            speeds.append(t.get("predicted_per_second", 0))
        passes = generated - accepted  # every pass yields one verified token plus its accepted drafts
        rate = accepted / drafted if drafted else 0
        print(f"n_max={n_max:2d} {label:8s}: accepted {accepted}/{drafted} drafted = {rate:5.1%} | "
              f"{generated / max(passes, 1):.2f} tokens per main pass | Mac speed {sum(speeds) / len(speeds):5.1f} tok/s",
              flush=True)
finally:
    server.terminate()
    server.wait()
