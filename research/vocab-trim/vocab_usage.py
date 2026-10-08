"""Dictionary trim, step 1: how small can his "pocket dictionary" be?

Builds a candidate set from English (macOS word list in 4 forms), every single-character / byte /
control entry (so he can still spell anything), and local English text. Then checks coverage against
tokens he actually picks in conversation (generated separately, with his persona), which are NOT used
to build the set.
"""
import json, os, random, subprocess, sys, time, urllib.request

RUN = "/Users/immyyyyy/Documents/LittleGuy/.run"
sys.path.insert(0, f"{RUN}/lcpp-src/gguf-py")
from gguf import GGUFReader

MODEL = f"{RUN}/models/gemma-4-E4B_q4_0-it.gguf"
PORT = 8089
OUT = f"{RUN}/trim"


def post(path, body, timeout=300):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def tokenize(text):
    return post("/tokenize", {"content": text, "add_special": False})["tokens"]


# --- the whole dictionary, straight from the model file
r = GGUFReader(MODEL)
f_tok, f_type = r.fields["tokenizer.ggml.tokens"], r.fields["tokenizer.ggml.token_type"]
pieces = [bytes(f_tok.parts[i]).decode("utf-8", "replace") for i in f_tok.data]
types = [int(f_type.parts[i][0]) for i in f_type.data]  # 1 normal, 2 unknown, 3 control, 4 user-defined, 5 unused, 6 byte
VOCAB = len(pieces)
print(f"full dictionary: {VOCAB} entries")

server = subprocess.Popen([f"{RUN}/lcpp-src/build-mac/bin/llama-server", "-m", MODEL, "-ngl", "99", "-c", "4096",
                           "--port", str(PORT), "--no-webui", "--reasoning", "off"],
                          stdout=open(f"{OUT}/server.log", "w"), stderr=subprocess.STDOUT)
try:
    for _ in range(120):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2); break
        except Exception:
            time.sleep(1)

    keep = set()
    # (b) spell-anything safety net: control/byte/user-defined entries and every single character
    safety = {i for i, (p, t) in enumerate(zip(pieces, types)) if t in (3, 4, 6) or len(p.replace("▁", " ")) <= 1}
    keep |= safety
    # (a) every English word in four forms
    base = [w.strip() for w in open("/usr/share/dict/words") if w.strip()]
    # the macOS list has base forms only ("human", not "humans"): add the common endings
    words = base + [w + e for w in base if w.islower() and len(w) > 2 for e in ("s", "es", "ed", "ing", "er", "ly")]
    english = set()
    for form in (lambda w: w.lower(), lambda w: w.capitalize()):
        for lead in ("", " "):
            for i in range(0, len(words), 4000):
                chunk = words[i:i + 4000]
                english |= set(tokenize(lead + (" " if lead else "\n").join(form(w) for w in chunk)))
    keep |= english
    # (c) local English text: punctuation, numbers, names, contractions
    text_toks = set()
    for path in (f"{RUN}/twobit/practice.txt", f"{RUN}/twobit/test.txt"):
        text_toks |= set(tokenize(open(path, errors="ignore").read()))
    keep |= text_toks
    print(f"safety net {len(safety)} | English words {len(english)} | local text {len(text_toks)} | union {len(keep)}")

    # --- held-out check: what does he actually say?
    persona = "\n".join(l for l in open(f"{RUN}/spec-test/little_guy_persona.txt").read().splitlines()
                        if l.strip() and not l.startswith("#"))
    prompts = [l.strip() for l in open(f"{OUT}/prompts.txt") if l.strip()]
    said = []
    for p in prompts:
        msgs = [{"role": "system", "content": persona}, {"role": "user", "content": p}]
        prompt = post("/apply-template", {"messages": msgs, "add_generation_prompt": True,
                                          "chat_template_kwargs": {"enable_thinking": False}})["prompt"]
        res = post("/completion", {"prompt": prompt, "n_predict": 80, "temperature": 1.0, "top_k": 64,
                                   "top_p": 0.95, "seed": random.randint(0, 99999), "return_tokens": True})
        said.extend(res["tokens"])
    missing = [t for t in said if t not in keep]
    print(f"conversation: {len(said)} tokens from {len(prompts)} prompts | covered {1 - len(missing) / len(said):.3%}")
    print("not covered:", sorted({pieces[t] for t in missing})[:40])
    json.dump({"keep": sorted(keep), "said": said}, open(f"{OUT}/vocab_usage.json", "w"))
finally:
    server.terminate()
    server.wait()
