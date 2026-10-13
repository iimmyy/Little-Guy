"""ffn_down track by track: which layers can take 2-bit without changing Little Guy?

Brain C keeps ffn_down at Q4_0 on all 42 layers; a brain with it at Q2_K everywhere (same careful recipe, same master)
was never usable. This mixes the two per layer and measures, against brain C:
  kld   mean KL divergence of the next-word odds over a test text (his persona chat + the 2026-09-28 write-up)
  warn  the Dayvigo warn-first probe: brain C's 40 saved replies, importance-weighted by how likely the mixed brain
        finds each opening (send_skip.py's probe; 1.0 = same lean as brain C, lower = drifting away from the warning)
Hidden states come from llama-server --embeddings (CPU, 4 threads); odds = softcap(h . dictionary), the dictionary is
brain C's own token_embd, so nothing is approximated.

usage: down_mix.py single           one layer at Q2_K at a time (42 runs) -> ranking
       down_mix.py set 3,7,9-12     one mixed brain, any set of layers at Q2_K
"""
import json, math, os, re, subprocess, sys, time, urllib.request
import numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / ".run"
sys.path.insert(0, str(RUN / "lcpp-src/gguf-py"))
sys.path.insert(0, str(ROOT / "tests/behavior"))
import gguf
from common import SERVER, LOGS, load_persona
from scoring import QUESTIONS

BASE = RUN / "weights/experiments/clean-Q2_K-C.gguf"
DOWN2 = RUN / "weights/experiments/down-mix/all-down-q2k.gguf"
OUT = RUN / "weights/experiments/down-mix"
RES = RUN / "results/down-mix"
DICT = RUN / "results/vocab-trim/ufs-writeup/E.npy"
PROBE = RUN / "results/send_skip_probe_dayvigo.json"
PORT, SOFTCAP = 8096, 30.0
RES.mkdir(parents=True, exist_ok=True)


def mix(layers, path):
    """brain C with ffn_down of `layers` taken from the all-Q2_K brain"""
    rb, rd = gguf.GGUFReader(BASE), gguf.GGUFReader(DOWN2)
    swap = {f"blk.{i}.ffn_down.weight" for i in layers}
    down = {t.name: t for t in rd.tensors if t.name in swap}
    w = gguf.GGUFWriter(str(path), rb.fields[gguf.Keys.General.ARCHITECTURE].contents(), endianess=rb.endianess)
    for f in rb.fields.values():
        if f.name == gguf.Keys.General.ARCHITECTURE or f.name.startswith("GGUF."):
            continue
        vt = f.types[0]
        w.add_key_value(f.name, f.contents(), vt, sub_type=f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None)
    tensors = [down.get(t.name, t) for t in rb.tensors]
    for t in tensors:
        w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
    for t in tensors:
        w.write_tensor_data(t.data, tensor_endianess=rb.endianess)
    w.close()


def post(path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=3600))


def template(messages):
    return post("/apply-template", {"messages": messages, "add_generation_prompt": True,
                                    "chat_template_kwargs": {"enable_thinking": False}})["prompt"]


def hidden(text):
    # the reply is ~80 MB of JSON and llama-server now and then cuts it off; ask again rather than crash
    for attempt in range(4):
        try:
            r = post("/embedding", {"content": text})
            break
        except Exception as e:
            if attempt == 3:
                raise
            print(f"   (embedding reply cut off: {type(e).__name__}, retrying)", flush=True)
    return np.array(r[0]["embedding"] if isinstance(r, list) else r["embedding"], dtype=np.float32)


def log_softmax(h, E):
    lg = SOFTCAP * np.tanh((h @ E.T) / SOFTCAP)
    lg -= lg.max(1, keepdims=True)
    return lg - np.log(np.exp(lg).sum(1, keepdims=True))


def measure(brain, E, cache, probe=True):
    """hidden states of the test text and (unless probe=False) of the probe openings under `brain`"""
    try:   # a server left over from a crashed run would answer for the wrong brain
        urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
        raise SystemExit(f"port {PORT} is already taken by another server; stop it first")
    except OSError:
        pass
    srv = subprocess.Popen([str(SERVER), "-m", str(brain), "-ngl", "0", "-t", "8", "-c", "4096", "--port", str(PORT), "--no-webui",
                            "--embeddings", "--pooling", "none", "-ub", "4096", "-b", "4096", "--timeout", "3600", "-cram", "0", "-np", "1"],
                           stdout=open(LOGS / "down_mix.log", "w"), stderr=subprocess.STDOUT)
    try:
        for _ in range(300):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2); break
            except Exception:
                time.sleep(1)
        if "text" not in cache:
            chat = template([{"role": "system", "content": load_persona()}, {"role": "user", "content": QUESTIONS["day"]}])
            cache["text"] = chat + open(ROOT / "docs/2026-09-28.md").read()[:3500]
            cache["probe_prompt"] = template([{"role": "system", "content": load_persona()}, {"role": "user", "content": QUESTIONS["dayvigo"]}])
            cache["n_probe"] = len(post("/tokenize", {"content": cache["probe_prompt"], "add_special": True})["tokens"])
            sets = json.load(open(PROBE))
            cache["openings"] = [(opening(s["reply"]), s["warns"]) for s in sets]
            cache["text_toks"] = np.array(post("/tokenize", {"content": cache["text"], "add_special": True})["tokens"])
            cache["open_toks"] = [np.array(post("/tokenize", {"content": cache["probe_prompt"] + o, "add_special": True})["tokens"])
                                  for o, _ in cache["openings"]]
        lp_text = log_softmax(hidden(cache["text"])[:-1], E)
        lp_open = []
        for _ in [0] if probe else []:
          for (o, _), toks in zip(cache["openings"], cache["open_toks"]):
            h = hidden(cache["probe_prompt"] + o)
            n = cache["n_probe"]
            lp = log_softmax(h[n - 1:-1], E)
            lp_open.append(float(lp[np.arange(len(lp)), toks[n:n + len(lp)]].sum()))
        return lp_text, np.array(lp_open)
    finally:
        srv.terminate(); srv.wait()


def opening(reply):   # up to the end of the first spoken sentence, as in send_skip.py
    clean = re.sub(r"\[[^\]]*\]", lambda t: " " * len(t.group()), reply)
    m = re.search(r"[.!?](\s|$)", clean.lstrip() and clean)
    return reply[:m.end()] if m else reply


def compare(base, other, cache):
    p = np.exp(base[0])
    kld = float((p * (base[0] - other[0])).sum(1).mean())
    if len(other[1]) == 0:
        return kld, float("nan")
    w = np.exp(other[1] - base[1])
    warns = np.array([wf for _, wf in cache["openings"]])
    warn = float((w * warns).sum() / w.sum()) / float(warns.mean())
    return kld, warn


def parse(spec):
    s = set()
    for part in spec.split(","):
        a, _, b = part.partition("-")
        s |= set(range(int(a), int(b or a) + 1))
    return sorted(s)


if __name__ == "__main__":
    E = np.load(DICT)
    cache = {}
    mode = sys.argv[1]
    base = measure(BASE, E, cache, probe=(mode != "single"))
    if mode == "single":
        # ranking pass: drift only (the warn probe re-reads the persona 40 times; it runs on the final mixes)
        done = RES / "single.json"
        rows = json.load(open(done)) if done.exists() else []
        for i in range(len(rows), 42):
            path = OUT / "mix-single.gguf"
            mix([i], path)
            for attempt in range(3):
                try:
                    kld, warn = compare(base, measure(path, E, cache, probe=False), cache)
                    break
                except Exception as e:
                    print(f"layer {i}: retry after {e!r}", flush=True)
            rows.append({"layer": i, "kld": kld, "warn": warn})
            print(f"layer {i:2d} at 2-bit: KL {kld:.5f}   warn lean {warn:.3f}", flush=True)
            json.dump(rows, open(RES / "single.json", "w"), indent=1)
        path.unlink()
    else:
        layers = parse(sys.argv[2])
        path = OUT / f"mix-{len(layers)}.gguf"
        mix(layers, path)
        kld, warn = compare(base, measure(path, E, cache), cache)
        print(f"{len(layers)} layers at 2-bit {sys.argv[2]}: KL {kld:.5f}   warn lean {warn:.3f}   -> {path.name}", flush=True)
